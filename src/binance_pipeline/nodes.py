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
# UPGRADED "GRID-FIRST" PIPELINE NODES (POLARS-POWERED)
# =============================================================================
def create_and_merge_grids_with_polars(trade_raw: pd.DataFrame, book_raw: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    UPGRADED METHODOLOGY: Creates and merges grids efficiently and ensures the
    output is chronologically sorted.
    """
    log.info(f"Starting memory-efficient grid creation with Polars (freq: {rule})...")
    
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

    merged = trades_pl_eager.join(book_pl_eager, on="datetime", how="outer_coalesce")
    
    # --- CRITICAL FIX: Sort the data chronologically after the join ---
    # This ensures the integrity of all subsequent time-series feature calculations.
    final_grid = (
        merged.sort("datetime")
        .with_columns([
            pl.col(["volume", "taker_flow"]).fill_null(0),
            pl.col(["open", "high", "low", "close", "best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty"]).forward_fill()
        ])
        .drop_nulls()
    )

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
    UPGRADED: Calculates EWMA and rolling features on the time grid by dynamically
    detecting the data's median frequency. This is robust to data gaps and
    variations in grid resolution. Also includes limited forward-fill to prevent data leakage.
    """
    log.info(f"Calculating ADAPTIVE EWMA features on grid for shape {df.shape}...")
    df_out = df.copy()

    time_diffs_ms = df['datetime'].diff().dt.total_seconds().mul(1000)
    median_freq_ms = time_diffs_ms.median()
    log.info(f"Detected median grid frequency: {median_freq_ms:.2f}ms")
    
    if pd.isna(median_freq_ms) or median_freq_ms <= 0:
        log.warning("Could not detect valid frequency. Falling back to 15ms assumption.")
        median_freq_ms = 15.0
        
    rows_per_second = 1000 / median_freq_ms

    ewma_spans_rows = {
        '5s': int(5 * rows_per_second), '15s': int(15 * rows_per_second), 
        '1m': int(60 * rows_per_second), '3m': int(180 * rows_per_second), 
        '15m': int(900 * rows_per_second),
    }
    wall_clock_windows_rows = {'60s': int(60 * rows_per_second)}
    log.info(f"Calculated adaptive row spans for EWMA: {ewma_spans_rows}")

    features_to_smooth = [
        'mid_price', 'spread', 'spread_bps', 'microprice', 'taker_flow', 
        'ofi', 'book_imbalance', 'close', 'volume'
    ]
    
    for feature in features_to_smooth:
        for name, span_rows in ewma_spans_rows.items():
            if span_rows > 1:
                df_out[f'{feature}_ewma_{name}'] = df_out[feature].ewm(span=span_rows).mean()
            
    for feature in ['taker_flow', 'ofi']:
        for name, window_rows in wall_clock_windows_rows.items():
            if window_rows > 1:
                df_out[f'{feature}_rollsum_{name}'] = df_out[feature].rolling(window=window_rows).sum()

    max_fill_limit = 10
    log.info(f"Applying limited forward-fill (limit={max_fill_limit}) to prevent data leakage...")
    for col in df_out.select_dtypes(include=np.number).columns:
        df_out[col] = df_out[col].ffill(limit=max_fill_limit)
            
    log.info(f"EWMA grid feature calculation complete. Total features: {len(df_out.columns)}")
    return df_out

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
    Calculates secondary bar-based features.
    UPGRADED: Now includes robust pre-cleaning and logical correction for TA-Lib features.
    """
    if df.empty:
        log.warning("Input to 'generate_bar_features' is empty. Skipping calculations.")
        return df.copy()
    
    log.info(f"Generating secondary/legacy bar features (RSI, ADX) for shape {df.shape}...")
    df = df.copy()

    # --- FIX 1: Pre-emptive cleaning and type enforcement ---
    # Ensure input columns are numeric and clean before passing to TA-Lib to prevent silent failures.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # --- Standard TA-Lib features (now safer to call) ---
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['rsi_28'] = ta.momentum.RSIIndicator(close=df['close'], window=28).rsi()

    # --- OPTIMIZATION: Replace Hurst with ADX ---
    log.info("  -> Calculating ADX (fast proxy for Hurst) with window 100...")
    adx_indicator = ta.trend.ADXIndicator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=100
    )
    df['adx_100'] = adx_indicator.adx()
    
    # --- FIX 2: Make ADX more robust by treating 0 as NaN (undefined trend) ---
    # This prevents the model from learning from misleading "zero trend" signals.
    df['adx_100'].replace(0, np.nan, inplace=True)
        
    # --- Time-based features ---
    df['datetime'] = pd.to_datetime(df['datetime']) # Ensure datetime type
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Final cleaning (now largely redundant but safe)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    log.info(f"Secondary feature generation complete. Final shape: {df.shape}")
    return df
