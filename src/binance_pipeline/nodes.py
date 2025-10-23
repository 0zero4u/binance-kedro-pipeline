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
    # MODIFIED: Keep 'is_buyer_maker' for taker flow calculation
    trades = trade_raw[['time', 'price', 'qty', 'is_buyer_maker']].copy()
    trades.rename(columns={'time': 'timestamp'}, inplace=True)
    trades['timestamp'] = pd.to_numeric(trades['timestamp'], errors='coerce')
    trades.dropna(subset=['timestamp'], inplace=True)
    trades.sort_values('timestamp', inplace=True)
    # MODIFIED: Keep best bid/ask quantities for microprice calculation
    book = book_raw[['event_time', 'best_bid_price', 'best_ask_price', 'best_bid_qty', 'best_ask_qty']].copy()
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
    # MODIFIED: Return all necessary columns for the next step
    return merged_df

# --- NEW NODE ---
def calculate_tick_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates features at the tick level before resampling.
    - Microprice: A volume-weighted price that is more robust than mid-price.
    - Taker Flow: The signed volume of aggressive trades (market orders).
    - Order Flow Imbalance (OFI): The net pressure on bid vs. ask side.
    """
    log.info("Calculating tick-level features: Microprice, Taker Flow, and OFI...")

    # Calculate Microprice
    df['microprice'] = (
        (df['best_bid_price'] * df['best_ask_qty']) +
        (df['best_ask_price'] * df['best_bid_qty'])
    ) / (df['best_bid_qty'] + df['best_ask_qty'])
    # Forward fill to handle ticks where book data might be missing
    df['microprice'].ffill(inplace=True)

    # Calculate Taker Flow
    # is_buyer_maker=False means the buyer was the taker (a market buy) -> positive flow
    # is_buyer_maker=True means the seller was the taker (a market sell) -> negative flow
    df['taker_flow'] = df.apply(
        lambda row: row['qty'] if not row['is_buyer_maker'] else -row['qty'],
        axis=1
    )

    # --- NEW: Calculate Order Flow Imbalance (OFI) ---
    # OFI measures the net pressure on the bid vs. ask side by looking at quantity changes
    # at the best bid/ask prices.
    bid_price_diff = df['best_bid_price'].diff()
    ask_price_diff = df['best_ask_price'].diff()
    bid_qty_diff = df['best_bid_qty'].diff()
    ask_qty_diff = df['best_ask_qty'].diff()

    # Increase in bid quantity when price holds or increases is buying pressure
    bid_pressure = np.where(bid_price_diff >= 0, bid_qty_diff, 0)
    # Increase in ask quantity when price holds or decreases is selling pressure
    ask_pressure = np.where(ask_price_diff <= 0, ask_qty_diff, 0)
    
    df['ofi'] = bid_pressure - ask_pressure
    df['ofi'].fillna(0, inplace=True)
    
    log.info("Tick-level feature calculation complete.")
    return df

# --- Resample Node ---
def resample_to_time_bars(df: pd.DataFrame, rule: str = "100ms") -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')
    # MODIFIED: Add aggregations for the new tick-level features
    aggregations = {
        'price': 'ohlc', 
        'qty': 'sum', 
        'mid_price': 'last', 
        'spread': 'mean',
        'microprice': 'ohlc', # Aggregate microprice like a normal price
        'taker_flow': 'sum',  # Sum the taker flow over the bar
        'ofi': 'sum'          # NEW: Sum the OFI over the bar
    }
    resampled_df = df.resample(rule).agg(aggregations)
    resampled_df.columns = ['_'.join(col).strip() for col in resampled_df.columns.values]
    # MODIFIED: Rename new columns
    resampled_df.rename(columns={
        'price_open': 'open', 'price_high': 'high', 'price_low': 'low', 'price_close': 'close', 
        'qty_sum': 'volume', 'mid_price_last': 'mid_price', 'spread_mean': 'spread',
        'microprice_open': 'micro_open', 'microprice_high': 'micro_high',
        'microprice_low': 'micro_low', 'microprice_close': 'micro_close',
        'taker_flow_sum': 'taker_flow',
        'ofi_sum': 'ofi'
    }, inplace=True)
    # MODIFIED: Forward fill all price-like columns
    price_cols = ['open', 'high', 'low', 'close', 'mid_price', 'spread', 
                  'micro_open', 'micro_high', 'micro_low', 'micro_close']
    resampled_df[price_cols] = resampled_df[price_cols].ffill()
    resampled_df['volume'].fillna(0, inplace=True)
    resampled_df['taker_flow'].fillna(0, inplace=True)
    resampled_df['ofi'].fillna(0, inplace=True)
    resampled_df.dropna(inplace=True)
    return resampled_df.reset_index()

# --- Feature Node (Bar-Based) ---
def generate_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Original Features ---
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    df['vwap_50'] = (df['volume'] * df['close']).rolling(50).sum() / (df['volume'].rolling(50).sum() + 1e-10)
    df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
    df['volatility_20'] = df['log_return_5'].rolling(window=20).std()
    
    # --- NEW: Garman-Klass Volatility & Volatility Regime ---
    log.info("Generating Garman-Klass Volatility and Regime...")
    df['gk_vol'] = 0.5 * np.log(df['high'] / df['low'])**2 - \
                   (2 * np.log(2) - 1) * np.log(df['close'] / df['open'])**2
    df['gk_vol_100'] = df['gk_vol'].rolling(window=100).mean()
    df['vol_regime'] = df['gk_vol'] / df['gk_vol_100']

    # --- NEW: Microprice Momentum ---
    log.info("Generating Microprice Momentum...")
    df['micro_mom_5'] = df['micro_close'] - df['micro_close'].shift(5)
    df['micro_mom_20'] = df['micro_close'] - df['micro_close'].shift(20)

    # --- NEW: Taker Flow CVD & Acceleration ---
    log.info("Generating Taker Flow CVD and Acceleration...")
    df['cvd_taker_50'] = df['taker_flow'].rolling(window=50).sum()
    df['cvd_taker_velocity'] = df['cvd_taker_50'] - df['cvd_taker_50'].shift(1)
    df['cvd_taker_accel'] = df['cvd_taker_velocity'] - df['cvd_taker_velocity'].shift(1)

    # --- NEW: Order Flow Imbalance (OFI) Pressure ---
    log.info("Generating Order Flow Imbalance features...")
    df['ofi_50'] = df['ofi'].rolling(window=50).sum()

    # --- NEW: VPIN Proxy (Flow Toxicity) ---
    log.info("Generating VPIN Proxy...")
    # This proxy measures the intensity of aggressive orders (taker flow) relative to total volume.
    # High values suggest more "toxic" or informed flow.
    df['vpin_proxy_50'] = df['taker_flow'].abs().rolling(50).sum() / (df['volume'].rolling(50).sum() + 1e-10)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# --- Multi-Timeframe Merge Node ---
def merge_multi_timeframe_features(base_features: pd.DataFrame, **other_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merges features from multiple timeframes onto a base (highest frequency) dataframe.
    """
    log.info(f"Starting multi-timeframe feature merge onto base shape {base_features.shape}.")
    merged_df = base_features.sort_values('datetime').copy()

    for tf_name, df_tf in other_features.items():
        suffix = f"_{tf_name.split('_')[1]}"
        df_to_merge = df_tf.sort_values('datetime').copy()
        
        # MODIFIED: Be more specific about which columns are features to avoid OHLC duplication
        feature_cols = [c for c in df_to_merge.columns if c not in [
            'datetime', 'open', 'high', 'low', 'close', 'volume', 'mid_price', 'spread',
            'micro_open', 'micro_high', 'micro_low', 'micro_close', 'taker_flow', 'ofi'
        ]]
        cols_to_rename = {col: col + suffix for col in feature_cols}
        df_to_merge = df_to_merge[['datetime'] + feature_cols].rename(columns=cols_to_rename)
        
        merged_df = pd.merge_asof(merged_df, df_to_merge, on='datetime', direction='backward')
    
    merged_df.dropna(inplace=True)
    log.info(f"Multi-timeframe merge complete. Final shape: {merged_df.shape}")
    return merged_df
