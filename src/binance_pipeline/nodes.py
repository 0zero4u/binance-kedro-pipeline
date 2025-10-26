import pandas as pd
import numpy as np
import ta
import logging
import requests
import zipfile
import os
from pathlib import Path
from typing import Dict
import numba
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
            desc=f"Downloading {file_name}", total=total_size, unit='iB',
            unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
    log.info(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(p_output_dir)
    os.remove(zip_path)
    log.info(f"Download and unzip complete for {file_name}.")

# =============================================================================
# NEW "GRID-FIRST" PIPELINE NODES (POLARS-POWERED)
# =============================================================================

def create_and_merge_grids_with_polars(trade_raw: pd.DataFrame, book_raw: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    SUPERIOR METHODOLOGY: Creates, merges, and handles ghost grids for both
    trade and book data in a single, hyper-optimized Polars function.
    """
    log.info(f"Starting hyper-optimized grid creation with Polars (freq: {rule})...")
    
    trades_pl = (
        pl.from_pandas(trade_raw)
        .lazy()
        .with_columns(
            pl.from_epoch(pl.col("time"), time_unit="ms").alias("datetime"),
            pl.when(pl.col("is_buyer_maker")).then(-pl.col("qty")).otherwise(pl.col("qty")).alias("taker_flow")
        )
        .group_by_dynamic("datetime", every=rule)
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("qty").sum().alias("volume"),
            pl.col("taker_flow").sum().alias("taker_flow")
        ])
    )

    book_pl = (
        pl.from_pandas(book_raw)
        .lazy()
        .with_columns(pl.from_epoch(pl.col("event_time"), time_unit="ms").alias("datetime"))
        .group_by_dynamic("datetime", every=rule)
        .agg([
            pl.col("best_bid_price").last(),
            pl.col("best_ask_price").last(),
            pl.col("best_bid_qty").last(),
            pl.col("best_ask_qty").last()
        ])
    )
    
    trades_pl_eager = trades_pl.collect()
    book_pl_eager = book_pl.collect()

    if trades_pl_eager.is_empty() or book_pl_eager.is_empty():
        log.warning("One of the input dataframes is empty, returning an empty grid.")
        return pd.DataFrame()

    min_time = min(trades_pl_eager["datetime"].min(), book_pl_eager["datetime"].min())
    max_time = max(trades_pl_eager["datetime"].max(), book_pl_eager["datetime"].max())
    
    full_grid = pl.DataFrame({
        "datetime": pl.datetime_range(min_time, max_time, rule, time_unit="ms", eager=True)
    })

    merged = full_grid.join(trades_pl_eager, on="datetime", how="left")
    merged = merged.join(book_pl_eager, on="datetime", how="left")
    
    final_grid = merged.with_columns([
        pl.col(["volume", "taker_flow"]).fill_null(0),
        pl.col(["open", "high", "low", "close", "best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty"]).forward_fill()
    ]).drop_nulls()

    log.info(f"Polars grid creation complete. Shape: {final_grid.shape}")
    return final_grid.to_pandas()


def calculate_primary_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates primary features on the clean, merged grid."""
    log.info(f"Calculating primary grid features for shape {df.shape}...")
    df_out = df.copy()
    df_out['mid_price'] = (df_out['best_bid_price'] + df_out['best_ask_price']) / 2
    df_out['spread'] = df_out['best_ask_price'] - df_out['best_bid_price']
    df_out['spread_bps'] = (df_out['spread'] / df_out['mid_price']) * 10000
    df_out['microprice'] = ((df_out['best_bid_price'] * df_out['best_ask_qty']) + (df_out['best_ask_price'] * df_out['best_bid_qty'])) / (df_out['best_bid_qty'] + df_out['best_ask_qty'])
    df_out['book_imbalance'] = (df_out['best_bid_qty'] - df_out['best_ask_qty']) / (df_out['best_bid_qty'] + df_out['best_ask_qty'] + 1e-10)
    bid_price_diff = df_out['best_bid_price'].diff()
    ask_price_diff = df_out['best_ask_price'].diff()
    bid_qty_diff = df_out['best_bid_qty'].diff()
    ask_qty_diff = df_out['best_ask_qty'].diff()
    bid_pressure = np.where(bid_price_diff >= 0, bid_qty_diff, 0)
    ask_pressure = np.where(ask_price_diff <= 0, ask_qty_diff, 0)
    df_out['ofi'] = bid_pressure - ask_pressure
    df_out['ofi'].fillna(0, inplace=True)
    log.info(f"Primary grid features calculation complete. Shape: {df_out.shape}")
    return df_out

def calculate_ewma_features_on_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates EWMA and rolling features on the clean time grid.
    DEFINITIVE FIX: Uses explicit integer row counts for all windowing functions
    to prevent internal pandas TypeErrors.
    """
    log.info(f"Calculating EWMA features on grid for shape {df.shape}...")
    df_out = df.copy() # No need to set datetime index
    
    # --- FIX IS HERE: Convert all time spans to integer row counts ---
    # Grid frequency = 15ms
    ewma_spans_rows = {
        '5s': 333,    # 5s / 0.015s
        '15s': 1000,   # 15s / 0.015s
        '1m': 4000,    # 60s / 0.015s
        '3m': 12000,   # 180s / 0.015s
        '15m': 60000,  # 900s / 0.015s
    }
    wall_clock_windows_rows = {'60s': 4000}

    features_to_smooth = [
        'mid_price', 'spread', 'spread_bps', 'microprice', 'taker_flow', 
        'ofi', 'book_imbalance', 'close', 'volume'
    ]
    
    for feature in features_to_smooth:
        for name, span_rows in ewma_spans_rows.items():
            df_out[f'{feature}_ewma_{name}'] = df_out[feature].ewm(span=span_rows).mean()
            
    for feature in ['taker_flow', 'ofi']:
        for name, window_rows in wall_clock_windows_rows.items():
            df_out[f'{feature}_rollsum_{name}'] = df_out[feature].rolling(window=window_rows).sum()
            
    log.info(f"EWMA grid feature calculation complete. Total features: {len(df_out.columns)}")
    return df_out


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
    """Calculates secondary bar-based features (e.g., RSI, Hurst) on the grid."""
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
