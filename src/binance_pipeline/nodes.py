import pandas as pd
import numpy as np
import ta
import logging
import requests
import zipfile
import os
from pathlib import Path
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple
import numba
import time
from tqdm import tqdm
import polars as pl

log = logging.getLogger(__name__)

# --- Download Node (Unchanged) ---
def download_and_unzip(url: str, output_dir: str):
    """Downloads and extracts data from URL with a live progress bar."""
    p_output_dir = Path(output_dir)
    p_output_dir.mkdir(parents=True, exist_ok=True)
    file_name = url.split('/')[-1]
    zip_path = p_output_dir / file_name

    log.info(f"Downloading from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(
            desc=f"Downloading {file_name}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

    log.info(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(p_output_dir)
    os.remove(zip_path)
    log.info(f"Download and unzip complete for {file_name}.")

# --- Merge Node (Unchanged, already high-performance) ---
def merge_book_trade_asof(book_raw: pd.DataFrame, trade_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Merges book and trade data using a high-performance as-of join with Polars.
    This is significantly faster and more memory-efficient than the Pandas equivalent.
    """
    log.info("Starting fast as-of merge with Polars...")
    log.info(f"  - Input 'book_raw' shape: {book_raw.shape}")
    log.info(f"  - Input 'trade_raw' shape: {trade_raw.shape}")

    log.info("  - Preparing trade data (converting to Polars, sorting, and casting)...")
    trades_pl = (
        pl.from_pandas(trade_raw)
        .select(["time", "price", "qty", "is_buyer_maker"])
        .rename({"time": "timestamp"})
        .cast({"timestamp": pl.Int64, "price": pl.Float64, "qty": pl.Float64})
        .drop_nulls(subset="timestamp")
        .sort("timestamp")
    )

    log.info("  - Preparing book data (converting to Polars, sorting, and casting)...")
    book_pl = (
        pl.from_pandas(book_raw)
        .select(["event_time", "best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty"])
        .rename({"event_time": "timestamp"})
        .cast({"timestamp": pl.Int64, "best_bid_price": pl.Float64, "best_ask_price": pl.Float64, "best_bid_qty": pl.Float64, "best_ask_qty": pl.Float64})
        .drop_nulls(subset="timestamp")
        .unique(subset="timestamp", keep="last", maintain_order=True)
        .sort("timestamp")
    )

    log.info("  - Performing Polars join_asof (this is the fastest part of this node)...")
    start_time = time.time()
    merged_pl = trades_pl.join_asof(book_pl, on="timestamp", strategy="backward")
    log.info(f"  - Polars as-of join completed in {time.time() - start_time:.2f} seconds.")
    
    final_pl = (
        merged_pl
        .drop_nulls(subset=["best_bid_price", "best_ask_price"])
        .with_columns([
            (((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2).alias("mid_price")),
            ((pl.col("best_ask_price") - pl.col("best_bid_price")).alias("spread"))
        ])
        .with_columns([
            ((pl.col("spread") / pl.col("mid_price")) * 10000).alias("spread_bps")
        ])
    )
    
    merged_df = final_pl.to_pandas()

    log.info(f"Polars as-of merge complete. Resulting shape: {merged_df.shape}")
    return merged_df

# --- Tick-Level Features Node (Unchanged) ---
def calculate_tick_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates advanced tick-level features."""
    log.info(f"Calculating advanced tick-level features for dataframe of shape {df.shape}...")
    df['microprice'] = (
        (df['best_bid_price'] * df['best_ask_qty']) +
        (df['best_ask_price'] * df['best_bid_qty'])
    ) / (df['best_bid_qty'] + df['best_ask_qty'])
    df['microprice'].ffill(inplace=True)
    df['taker_flow'] = np.where(df['is_buyer_maker'], -df['qty'], df['qty'])
    bid_price_diff = df['best_bid_price'].diff()
    ask_price_diff = df['best_ask_price'].diff()
    bid_qty_diff = df['best_bid_qty'].diff()
    ask_qty_diff = df['best_ask_qty'].diff()
    bid_pressure = np.where(bid_price_diff >= 0, bid_qty_diff, 0)
    ask_pressure = np.where(ask_price_diff <= 0, ask_qty_diff, 0)
    df['ofi'] = bid_pressure - ask_pressure
    df['ofi'].fillna(0, inplace=True)
    df['book_imbalance'] = (df['best_bid_qty'] - df['best_ask_qty']) / (
        df['best_bid_qty'] + df['best_ask_qty'] + 1e-10
    )
    log.info("Tick-level feature calculation complete.")
    return df

# --- EWMA TBT Feature Calculation Node (UPGRADED to POLARS and CORRECTED) ---
def calculate_ewma_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Exponentially Weighted Moving Average (EWMA) features
    and Rolling Sums using the high-performance Polars library for a significant speedup.
    Polars uses all available CPU cores for these calculations.
    """
    log.info(f"Calculating TBT EWMA & Wall Clock features with Polars for shape {df.shape}...")
    start_time = time.time()

    df_pl = pl.from_pandas(df).lazy()

    # --- FIX: Convert time spans to an approximate number of rows (integers) ---
    # This is based on the target 25ms grid frequency (40 rows per second).
    ewma_half_life_rows = {
        '5s': 200,      # 5s * 40 rows/sec
        '15s': 600,     # 15s * 40 rows/sec
        '1m': 2400,     # 60s * 40 rows/sec
        '3m': 7200,     # 180s * 40 rows/sec
        '15m': 36000,   # 900s * 40 rows/sec
    }
    FASTEST_SPAN_KEY = '5s'
    SLOWEST_SPAN_KEY = '15m'
    wall_clock_windows = {'60s': '60s'}

    df_pl = df_pl.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime")
    ).sort("datetime")

    features_to_smooth = [
        'mid_price', 'spread', 'spread_bps', 'microprice',
        'taker_flow', 'ofi', 'book_imbalance', 'price', 'qty'
    ]

    ewma_exprs = []
    for feature in features_to_smooth:
        for name, half_life_int in ewma_half_life_rows.items():
            # Pass the integer half-life to ewm_mean
            ewma_exprs.append(
                pl.col(feature).ewm_mean(half_life=half_life_int).alias(f'{feature}_ewma_{name}')
            )

    df_pl = df_pl.with_columns(ewma_exprs)

    momentum_accel_exprs = []
    for feature in features_to_smooth:
        fast_col = f'{feature}_ewma_{FASTEST_SPAN_KEY}'
        slow_col = f'{feature}_ewma_{SLOWEST_SPAN_KEY}'
        
        if feature in ['mid_price', 'microprice', 'taker_flow', 'ofi', 'spread_bps']:
            momentum_accel_exprs.append((pl.col(fast_col) - pl.col(slow_col)).alias(f'{feature}_momentum'))

        momentum_accel_exprs.append(pl.col(fast_col).diff(1).alias(f'{feature}_velocity'))
        momentum_accel_exprs.append(pl.col(fast_col).diff(1).diff(1).alias(f'{feature}_accel'))

    df_pl = df_pl.with_columns(momentum_accel_exprs)

    wall_clock_exprs = []
    for feature in ['taker_flow', 'ofi']:
        for name, window_str in wall_clock_windows.items():
            # The rolling sum IS time-based and uses the duration string correctly.
            wall_clock_exprs.append(
                pl.col(feature).rolling(index_column="datetime", period=window_str).sum().alias(f'{feature}_rollsum_{name}')
            )

    df_pl = df_pl.with_columns(wall_clock_exprs)

    df_out = df_pl.collect().to_pandas()

    log.info(f"Polars TBT feature calculation complete in {time.time() - start_time:.2f} seconds. Total features: {len(df_out.columns)}")
    return df_out

# --- Sampling Node (FIXED to prevent OOM error, unchanged) ---
def sample_features_to_grid(df: pd.DataFrame, rule: str = '25ms') -> pd.DataFrame:
    """
    Samples the EWMA tick-by-tick features onto a fixed time grid (e.g., 25ms)
    using a memory-efficient aggregation method instead of a full resample-ffill
    to prevent Out-Of-Memory errors.
    """
    log.info(f"Sampling TBT features onto a fixed {rule} grid using memory-efficient aggregation...")
    
    if df.empty:
        log.warning("Input to 'sample_features_to_grid' is empty. Returning empty DataFrame.")
        return pd.DataFrame()
    
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')

    feature_cols = [col for col in df.columns if col != 'price']
    aggregations = {col: 'last' for col in feature_cols}
    aggregations['price'] = 'ohlc'

    sampled_df = df.resample(rule).agg(aggregations)

    if isinstance(sampled_df.columns, pd.MultiIndex):
        sampled_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in sampled_df.columns.values]
        rename_map = {'price_open': 'open', 'price_high': 'high', 'price_low': 'low', 'price_close': 'close'}
        sampled_df.rename(columns=rename_map, inplace=True)

    sampled_df.ffill(inplace=True)
    sampled_df.dropna(inplace=True)
    
    log.info(f"Memory-efficient sampling complete. Output shape: {sampled_df.shape}")
    return sampled_df.reset_index()


# =============================================================================
# Helper Functions and Bar Features Node (Unchanged)
# =============================================================================
@numba.jit(nopython=True, fastmath=True)
def _rolling_slope_numba(y: np.ndarray, window: int) -> np.ndarray:
    n = len(y)
    out = np.full(n, np.nan)
    x = np.arange(window)
    sum_x = np.sum(x)
    sum_x2 = np.sum(x * x)
    denominator = window * sum_x2 - sum_x * sum_x
    if denominator == 0: return out
    for i in range(window - 1, n):
        y_win = y[i - window + 1 : i + 1]
        sum_y = np.sum(y_win)
        sum_xy = np.sum(x * y_win)
        slope = (window * sum_xy - sum_x * sum_y) / denominator
        out[i] = slope
    return out

@numba.jit(nopython=True, fastmath=True)
def _rolling_rank_pct_numba(y: np.ndarray, window: int) -> np.ndarray:
    n = len(y)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        win = y[i - window + 1 : i + 1]
        last_val = win[-1]
        count_le = 0
        for val in win:
            if val <= last_val:
                count_le += 1
        out[i] = count_le / window
    return out

def apply_rolling_numba(series: pd.Series, func, window: int) -> pd.Series:
    values = series.to_numpy()
    result = func(values, window)
    return pd.Series(result, index=series.index, name=series.name)

def generate_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates secondary bar-based features (e.g., RSI, Hurst) on the 25ms grid.
    """
    if df.empty:
        log.warning("Input to 'generate_bar_features' is empty. Skipping calculations.")
        return df.copy()
    
    log.info(f"Generating secondary/legacy bar features (RSI, Hurst) for dataframe of shape {df.shape}...")
    df = df.copy()
    
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['rsi_28'] = ta.momentum.RSIIndicator(close=df['close'], window=28).rsi()

    def hurst(ts):
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    window_hurst = 100
    if len(df) > window_hurst:
        log.info(f"  -> Calculating Hurst Exponent (Window: {window_hurst} periods)")
        df['hurst_100'] = df['close'].rolling(window_hurst).apply(hurst, raw=True)
    else:
        log.warning(f"  -> Skipping Hurst calculation, DataFrame size ({len(df)}) is too small.")
        
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    log.info(f"Secondary feature generation complete. Final shape: {df.shape}")
    return df
