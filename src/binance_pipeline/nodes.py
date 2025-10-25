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
import polars as pl  # <-- NEW: Import Polars

log = logging.getLogger(__name__)

# --- Download Node ---
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

# --- Merge Node (High-Performance with Polars) ---
def merge_book_trade_asof(book_raw: pd.DataFrame, trade_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Merges book and trade data using a high-performance as-of join with Polars.
    This is significantly faster and more memory-efficient than the Pandas equivalent.
    """
    log.info("Starting fast as-of merge with Polars...")
    log.info(f"  - Input 'book_raw' shape: {book_raw.shape}")
    log.info(f"  - Input 'trade_raw' shape: {trade_raw.shape}")

    # Prepare trades using the expressive and fast Polars API
    log.info("  - Preparing trade data (converting to Polars, sorting, and casting)...")
    trades_pl = (
        pl.from_pandas(trade_raw)
        .select(["time", "price", "qty", "is_buyer_maker"])
        .rename({"time": "timestamp"})
        .cast({"timestamp": pl.Int64, "price": pl.Float64, "qty": pl.Float64})
        .drop_nulls(subset="timestamp")
        .sort("timestamp")
    )

    # Prepare book data
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

    # Perform the high-speed as-of join using Polars
    log.info("  - Performing Polars join_asof (this is the fastest part of this node)...")
    start_time = time.time()
    merged_pl = trades_pl.join_asof(book_pl, on="timestamp", strategy="backward")
    log.info(f"  - Polars as-of join completed in {time.time() - start_time:.2f} seconds.")
    
    # Calculate basic features and clean up within Polars
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
    
    # Convert back to Pandas for compatibility with the rest of the pipeline
    merged_df = final_pl.to_pandas()

    log.info(f"Polars as-of merge complete. Resulting shape: {merged_df.shape}")
    return merged_df

# --- Tick-Level Features Node ---
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

# --- EWMA TBT Feature Calculation Node ---
def calculate_ewma_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Exponentially Weighted Moving Average (EWMA) features 
    (Activity Clock) and Rolling Sums (Wall Clock) directly on the tick data 
    using a multi-span half-life approach (V4: 5s, 15s, 1m, 3m, 15m).
    """
    log.info(f"Calculating TBT EWMA and Wall Clock features for dataframe of shape {df.shape}...")
    df_out = df.copy()

    # Define EWMA time-based half-life spans (in milliseconds) - Activity Clock
    ewma_time_spans = {
        '5s': 5000,
        '15s': 15000,
        '1m': 60000,
        '3m': 180000,
        '15m': 900000, 
    }
    # Define the fastest and slowest span keys for core momentum calculation
    FASTEST_SPAN_KEY = '5s'
    SLOWEST_SPAN_KEY = '15m'
    
    # Define Wall Clock rolling windows (in Pandas time strings)
    wall_clock_windows = {
        '60s': '60s' # For CVD calculation
    }

    # Time column must be datetime for EWM implementation
    df_out['datetime'] = pd.to_datetime(df_out['timestamp'], unit='ms')
    df_out = df_out.set_index('datetime')
    
    # Features to apply EWMA to
    features_to_smooth = [
        'mid_price', 'spread', 'spread_bps', 'microprice', 
        'taker_flow', 'ofi', 'book_imbalance', 'price', 'qty'
    ]
    
    for feature in features_to_smooth:
        
        # 1. EWMA Features (Activity Clock)
        for name, span_ms in ewma_time_spans.items():
            # Use halflife (exponential decay) based on the time span
            ewm_series = df_out[feature].ewm(halflife=f'{span_ms}ms', times=df_out.index).mean()
            df_out[f'{feature}_ewma_{name}'] = ewm_series

        # 2. Momentum (Difference between fastest and slowest EWMA)
        fast_col = f'{feature}_ewma_{FASTEST_SPAN_KEY}'
        slow_col = f'{feature}_ewma_{SLOWEST_SPAN_KEY}'
        
        if fast_col in df_out.columns and slow_col in df_out.columns:
             # Calculate momentum only if the feature is one where momentum makes sense (e.g., price, flow, spread)
             if feature in ['mid_price', 'microprice', 'taker_flow', 'ofi', 'spread_bps']:
                 df_out[f'{feature}_momentum'] = df_out[fast_col] - df_out[slow_col]

        # 3. Wall Clock Features (Rolling Sum/CVD)
        if feature in ['taker_flow', 'ofi']:
            for name, window_str in wall_clock_windows.items():
                df_out[f'{feature}_rollsum_{name}'] = df_out[feature].rolling(window=window_str).sum()
                
        # 4. Basic Velocity/Acceleration (of the fastest EWMA)
        df_out[f'{feature}_velocity'] = df_out[fast_col].diff(1)
        df_out[f'{feature}_accel'] = df_out[f'{feature}_velocity'].diff(1)
            
    df_out.reset_index(inplace=True)
    log.info(f"TBT EWMA & Wall Clock feature calculation complete. Total features: {len(df_out.columns)}")
    return df_out

def sample_features_to_grid(df: pd.DataFrame, rule: str = '25ms') -> pd.DataFrame:
    """
    Samples the EWMA tick-by-tick features onto a fixed time grid (e.g., 25ms) 
    using the 'last' observation for each interval and then forward-filling.
    """
    log.info(f"Sampling TBT features onto a fixed {rule} grid...")
    
    df_temp = df.copy()
    if 'datetime' not in df_temp.columns:
        df_temp['datetime'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
    df_temp = df_temp.set_index('datetime')

    # 1. Resample by taking the LAST observation in the interval
    sampled_df = df_temp.resample(rule).last()
    
    # 2. Forward fill missing grid points (essential for continuous features)
    sampled_df.ffill(inplace=True)
    
    # 3. Drop initial NaNs generated by the EWMA warm-up period
    sampled_df.dropna(inplace=True)

    # Add OHLC price metrics to the final sampled grid for labeling
    ohlc_df = df_temp['price'].resample(rule).agg('ohlc')
    ohlc_df.columns = ['open', 'high', 'low', 'close']
    
    final_df = sampled_df.join(ohlc_df, how='inner')
    
    log.info(f"Sampling complete. Output shape: {final_df.shape}")
    return final_df.reset_index()


# =============================================================================
# High-Performance Helper Functions for Feature Engineering
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

# --- Secondary Feature Generation Node (RSI, Hurst, Temporal) ---
def generate_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates secondary bar-based features (e.g., RSI, Hurst) on the 25ms grid.
    """
    if df.empty:
        log.warning("Input to 'generate_bar_features' is empty. Skipping calculations.")
        return df.copy()
    
    log.info(f"Generating secondary/legacy bar features (RSI, Hurst) for dataframe of shape {df.shape}...")
    df = df.copy()
    
    # Basic price derivatives
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Technical Indicators
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['rsi_28'] = ta.momentum.RSIIndicator(close=df['close'], window=28).rsi()

    # Hurst Exponent Calculation (Regime Detection)
    def hurst(ts):
        # Calculates Hurst Exponent using a simplified Rescaled Range approximation
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    window_hurst = 100 # 100 * 25ms = 2.5 seconds
    if len(df) > window_hurst:
        log.info(f"  -> Calculating Hurst Exponent (Window: {window_hurst} periods)")
        df['hurst_100'] = df['close'].rolling(window_hurst).apply(hurst, raw=True)
    else:
        log.warning(f"  -> Skipping Hurst calculation, DataFrame size ({len(df)}) is too small.")
        
    # Temporal Features
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)


    # Final cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    log.info(f"Secondary feature generation complete. Final shape: {df.shape}")
    return df
