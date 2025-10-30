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
# Data Engineering Pipeline Nodes (Polars-Powered)
# =============================================================================
def create_and_merge_grids_with_polars(trade_raw: pd.DataFrame, book_raw: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Creates time-based grids from raw trade and book data using Polars for
    efficiency, then merges them into a single chronologically sorted DataFrame.
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
    
    # Sort chronologically to ensure time-series integrity for subsequent calculations.
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

def calculate_intelligent_multi_scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates multi-scale features using logarithmically spaced timeframes to
    reduce redundancy and adds derivatives for richer information.
    """
    log.info(f"Calculating INTELLIGENT multi-scale features for shape {df.shape}...")
    df_out = df.copy()

    time_diffs_ms = df['datetime'].diff().dt.total_seconds().mul(1000)
    median_freq_ms = time_diffs_ms.median()
    
    if pd.isna(median_freq_ms) or median_freq_ms <= 0:
        log.warning("Could not detect valid frequency. Falling back to 15ms assumption.")
        median_freq_ms = 15.0
        
    rows_per_second = 1000 / median_freq_ms

    # Use logarithmically spaced timeframes to capture different dynamics with less overlap
    timeframe_configs = {
        'short': int(15 * rows_per_second),   # Approx 15s
        'medium': int(60 * rows_per_second),  # Approx 1m
        'long': int(300 * rows_per_second),   # Approx 5m
    }
    log.info(f"Using intelligent timeframes (rows): {timeframe_configs}")

    core_features = ['mid_price', 'spread_bps', 'microprice', 'taker_flow', 'ofi', 'book_imbalance', 'volume']
    
    for feature in core_features:
        for name, span_rows in timeframe_configs.items():
            if span_rows > 1:
                ewma_col = f'{feature}_ewma_{name}'
                df_out[ewma_col] = df_out[feature].ewm(span=span_rows).mean()
                
                # Add velocity (1st derivative) for dynamic features
                if name in ['short', 'medium']:
                    df_out[f'{ewma_col}_velo'] = df_out[ewma_col].diff()
    
    # Rolling sum for flow features on medium timeframe
    for feature in ['taker_flow', 'ofi']:
        window = timeframe_configs['medium']
        if window > 1:
            df_out[f'{feature}_rollsum_medium'] = df_out[feature].rolling(window=window).sum()
            
    log.info(f"Intelligent multi-scale feature calculation complete. Shape: {df_out.shape}")
    return df_out


def generate_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates secondary bar-based features like RSI and ADX, including
    pre-cleaning of input data for robustness.
    """
    if df.empty:
        log.warning("Input to 'generate_bar_features' is empty. Skipping calculations.")
        return df.copy()
    
    log.info(f"Generating secondary/legacy bar features (RSI, ADX) for shape {df.shape}...")
    df = df.copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['adx_100'] = ta.trend.ADXIndicator(
        high=df['high'], low=df['low'], close=df['close'], window=100
    ).adx()
    df['adx_100'].replace(0, np.nan, inplace=True)
        
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    log.info(f"Secondary feature generation complete. Final shape: {df.shape}")
    return df
