import pandas as pd
import numpy as np
import ta
import logging
import requests
import zipfile
import os
from pathlib import Path

log = logging.getLogger(__name__)

# --- Download Node ---
def download_and_unzip(url: str, output_dir: str):
    p_output_dir = Path(output_dir)
    p_output_dir.mkdir(parents=True, exist_ok=True)
    file_name = url.split('/')[-1]
    zip_path = p_output_dir / file_name
    log.info(f"Downloading from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    log.info(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(p_output_dir)
    os.remove(zip_path)
    log.info(f"Download and unzip complete for {file_name}.")

# --- Merge Node (High-Performance) ---
def merge_book_trade_asof(book_raw: pd.DataFrame, trade_raw: pd.DataFrame) -> pd.DataFrame:
    log.info("Starting fast as-of merge...")
    trades = trade_raw[['time', 'price', 'qty']].copy()
    trades.rename(columns={'time': 'timestamp'}, inplace=True)
    trades['timestamp'] = pd.to_numeric(trades['timestamp'], errors='coerce')
    trades.dropna(subset=['timestamp'], inplace=True)
    trades.sort_values('timestamp', inplace=True)
    book = book_raw[['event_time', 'best_bid_price', 'best_ask_price']].copy()
    book.rename(columns={'event_time': 'timestamp'}, inplace=True)
    book['timestamp'] = pd.to_numeric(book['timestamp'], errors='coerce')
    book.dropna(subset=['timestamp'], inplace=True)
    book = book.drop_duplicates(subset='timestamp', keep='last')
    book.sort_values('timestamp', inplace=True)
    merged_df = pd.merge_asof(left=trades, right=book, on='timestamp', direction='backward')
    merged_df.dropna(inplace=True)
    merged_df["mid_price"] = (merged_df["best_bid_price"] + merged_df["best_ask_price"]) / 2
    merged_df["spread"] = merged_df["best_ask_price"] - merged_df["best_bid_price"]
    log.info(f"As-of merge complete. Resulting shape: {merged_df.shape}")
    return merged_df[['timestamp', 'price', 'qty', 'mid_price', 'spread']]

# --- Resample Node ---
def resample_to_time_bars(df: pd.DataFrame, rule: str = "100ms") -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')
    aggregations = {'price': 'ohlc', 'qty': 'sum', 'mid_price': 'last', 'spread': 'mean'}
    resampled_df = df.resample(rule).agg(aggregations)
    resampled_df.columns = ['_'.join(col).strip() for col in resampled_df.columns.values]
    resampled_df.rename(columns={'price_open': 'open', 'price_high': 'high', 'price_low': 'low', 'price_close': 'close', 'qty_sum': 'volume', 'mid_price_last': 'mid_price', 'spread_mean': 'spread'}, inplace=True)
    resampled_df[['open', 'high', 'low', 'close', 'mid_price', 'spread']] = resampled_df[['open', 'high', 'low', 'close', 'mid_price', 'spread']].ffill()
    resampled_df['volume'].fillna(0, inplace=True)
    resampled_df.dropna(inplace=True)
    return resampled_df.reset_index()

# --- Feature Node (Bar-Based) ---
def generate_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    df['vwap_50'] = (df['volume'] * df['close']).rolling(50).sum() / (df['volume'].rolling(50).sum() + 1e-10)
    df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
    df['volatility_20'] = df['log_return_5'].rolling(window=20).std()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df
