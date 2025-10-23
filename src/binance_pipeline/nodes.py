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

log = logging.getLogger(__name__)

# --- Download Node ---
def download_and_unzip(url: str, output_dir: str):
    """Downloads and extracts data from URL."""
    p_output_dir = Path(output_dir)
    p_output_dir.mkdir(parents=True, exist_ok=True)
    file_name = url.split('/')[-1]
    zip_path = p_output_dir / file_name

    log.info(f"Downloading from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)

    log.info(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: 
        zip_ref.extractall(p_output_dir)
    os.remove(zip_path)
    log.info(f"Download and unzip complete for {file_name}.")

# --- Merge Node (High-Performance) ---
def merge_book_trade_asof(book_raw: pd.DataFrame, trade_raw: pd.DataFrame) -> pd.DataFrame:
    """Merges book and trade data using as-of join."""
    log.info("Starting fast as-of merge...")

    # Prepare trades
    trades = trade_raw[['time', 'price', 'qty', 'is_buyer_maker']].copy()
    trades.rename(columns={'time': 'timestamp'}, inplace=True)
    trades['timestamp'] = pd.to_numeric(trades['timestamp'], errors='coerce')
    trades.dropna(subset=['timestamp'], inplace=True)
    trades.sort_values('timestamp', inplace=True)

    # Prepare book
    book = book_raw[['event_time', 'best_bid_price', 'best_ask_price', 
                     'best_bid_qty', 'best_ask_qty']].copy()
    book.rename(columns={'event_time': 'timestamp'}, inplace=True)
    book['timestamp'] = pd.to_numeric(book['timestamp'], errors='coerce')
    book.dropna(subset=['timestamp'], inplace=True)
    book = book.drop_duplicates(subset='timestamp', keep='last')
    book.sort_values('timestamp', inplace=True)

    # Merge
    merged_df = pd.merge_asof(left=trades, right=book, on='timestamp', direction='backward')
    merged_df.dropna(inplace=True)

    # Basic features
    merged_df["mid_price"] = (merged_df["best_bid_price"] + merged_df["best_ask_price"]) / 2
    merged_df["spread"] = merged_df["best_ask_price"] - merged_df["best_bid_price"]
    merged_df["spread_bps"] = (merged_df["spread"] / merged_df["mid_price"]) * 10000

    log.info(f"As-of merge complete. Resulting shape: {merged_df.shape}")
    return merged_df

# --- Tick-Level Features Node ---
def calculate_tick_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates advanced tick-level features including:
    - Microprice (volume-weighted)
    - Taker flow (signed volume)
    - Order Flow Imbalance (OFI)
    """
    log.info("Calculating advanced tick-level features...")

    # Microprice (volume-weighted price more robust than mid)
    df['microprice'] = (
        (df['best_bid_price'] * df['best_ask_qty']) +
        (df['best_ask_price'] * df['best_bid_qty'])
    ) / (df['best_bid_qty'] + df['best_ask_qty'])
    df['microprice'].ffill(inplace=True)

    # Taker Flow (signed volume) - VECTORIZED
    df['taker_flow'] = np.where(df['is_buyer_maker'], -df['qty'], df['qty'])

    # Order Flow Imbalance (OFI)
    bid_price_diff = df['best_bid_price'].diff()
    ask_price_diff = df['best_ask_price'].diff()
    bid_qty_diff = df['best_bid_qty'].diff()
    ask_qty_diff = df['best_ask_qty'].diff()

    bid_pressure = np.where(bid_price_diff >= 0, bid_qty_diff, 0)
    ask_pressure = np.where(ask_price_diff <= 0, ask_qty_diff, 0)
    df['ofi'] = bid_pressure - ask_pressure
    df['ofi'].fillna(0, inplace=True)

    # Order book imbalance
    df['book_imbalance'] = (df['best_bid_qty'] - df['best_ask_qty']) / (
        df['best_bid_qty'] + df['best_ask_qty'] + 1e-10
    )

    log.info("Tick-level feature calculation complete.")
    return df

# --- Resample Node ---
def resample_to_time_bars(df: pd.DataFrame, rule: str = "100ms") -> pd.DataFrame:
    """Resamples tick data to time bars with comprehensive aggregations."""
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')

    aggregations = {
        'price': 'ohlc',
        'qty': ['sum', 'mean', 'std', 'count'],
        'mid_price': 'last',
        'spread': ['mean', 'std', 'min', 'max'],
        'spread_bps': ['mean', 'std'],
        'microprice': 'ohlc',
        'taker_flow': ['sum', 'mean', 'std'],
        'ofi': ['sum', 'mean'],
        'book_imbalance': ['mean', 'std', 'min', 'max'],
    }

    resampled_df = df.resample(rule).agg(aggregations)
    resampled_df.columns = ['_'.join(col).strip() for col in resampled_df.columns.values]

    # Rename key columns
    rename_map = {
        'price_open': 'open', 'price_high': 'high', 'price_low': 'low', 'price_close': 'close',
        'qty_sum': 'volume', 'qty_mean': 'avg_trade_size', 'qty_std': 'trade_size_std',
        'qty_count': 'num_trades',
        'mid_price_last': 'mid_price', 'spread_mean': 'spread', 'spread_std': 'spread_std',
        'spread_bps_mean': 'spread_bps',
        'microprice_open': 'micro_open', 'microprice_high': 'micro_high',
        'microprice_low': 'micro_low', 'microprice_close': 'micro_close',
        'taker_flow_sum': 'taker_flow',
        'ofi_sum': 'ofi',
        'book_imbalance_mean': 'book_imbalance',
    }
    resampled_df.rename(columns=rename_map, inplace=True)

    price_cols = [col for col in resampled_df.columns if 'price' in col.lower() or 
                  col in ['open', 'high', 'low', 'close', 'mid_price', 'spread',
                          'micro_open', 'micro_high', 'micro_low', 'micro_close']]
    resampled_df[price_cols] = resampled_df[price_cols].ffill()

    fill_zero_cols = ['volume', 'taker_flow', 'ofi', 'num_trades']
    for col in fill_zero_cols:
        if col in resampled_df.columns:
            resampled_df[col].fillna(0, inplace=True)

    resampled_df.dropna(inplace=True)
    return resampled_df.reset_index()

# --- Advanced Feature Engineering Node ---
def generate_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates comprehensive bar-based features across 14 categories.
    """
    log.info(f"Generating comprehensive bar features for dataframe of shape {df.shape}...")

    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # =======================
    # 1. BASIC TECHNICAL INDICATORS
    # =======================
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['rsi_28'] = ta.momentum.RSIIndicator(close=df['close'], window=28).rsi()
    df['atr_14'] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    
    for window in [50, 100, 200]:
        df[f'vwap_{window}'] = (df['volume'] * df['close']).rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-10)
        df[f'price_to_vwap_{window}'] = df['close'] / df[f'vwap_{window}']

    # =======================
    # 2. VOLATILITY FEATURES
    # =======================
    df['gk_vol'] = 0.5 * np.log(df['high'] / df['low'])**2 - (2 * np.log(2) - 1) * np.log(df['close'] / df['open'])**2
    for window in [20, 50, 100]:
        df[f'gk_vol_{window}'] = df['gk_vol'].rolling(window=window).mean()
        df[f'vol_regime_{window}'] = df['gk_vol'] / (df[f'gk_vol_{window}'] + 1e-10)
    for window in [10, 20, 50]:
        df[f'realized_vol_{window}'] = df['log_returns'].rolling(window).std() * np.sqrt(window)

    # =======================
    # 3. MOMENTUM FEATURES
    # =======================
    for period in [5, 10, 20, 50, 100]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        df[f'momentum_pct_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
    df['momentum_acceleration_10'] = df['momentum_10'] - df['momentum_10'].shift(1)
    df['momentum_acceleration_20'] = df['momentum_20'] - df['momentum_20'].shift(1)

    # =======================
    # 4. ORDER FLOW FEATURES
    # =======================
    for window in [20, 50, 100, 200]:
        df[f'cvd_taker_{window}'] = df['taker_flow'].rolling(window=window).sum()
        df[f'cvd_velocity_{window}'] = df[f'cvd_taker_{window}'].diff(1)
        df[f'cvd_accel_{window}'] = df[f'cvd_velocity_{window}'].diff(1)
    for window in [20, 50, 100]:
        df[f'ofi_{window}'] = df['ofi'].rolling(window=window).sum()
        df[f'ofi_std_{window}'] = df['ofi'].rolling(window=window).std()

    # =======================
    # 5. VOLUME FEATURES
    # =======================
    for window in [20, 50, 100]:
        df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
        df[f'volume_ratio_{window}'] = df['volume'] / (df[f'volume_ma_{window}'] + 1e-10)
    for window in [50, 100]:
        df[f'vpin_proxy_{window}'] = df['taker_flow'].abs().rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-10)

    # =======================
    # 6. SPREAD & LIQUIDITY FEATURES
    # =======================
    for window in [20, 50]:
        df[f'amihud_{window}'] = abs(df['returns']).rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-10)
        rolling_cov = df['returns'].rolling(window).cov(df['taker_flow'])
        rolling_var = df['taker_flow'].rolling(window).var()
        df[f'kyles_lambda_{window}'] = rolling_cov / (rolling_var + 1e-10)
    if 'book_imbalance' in df.columns:
        for window in [20, 50]:
            df[f'book_imb_ma_{window}'] = df['book_imbalance'].rolling(window).mean()
            df[f'book_imb_std_{window}'] = df['book_imbalance'].rolling(window).std()

    # =======================
    # 7. TEMPORAL FEATURES
    # =======================
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

    # =======================
    # 8. AUTOCORRELATION & LAG FEATURES
    # =======================
    for lag in [1, 2, 3, 5, 10]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
    for window in [50, 100]:
        for lag in [1, 5]:
            df[f'returns_autocorr_lag{lag}_w{window}'] = df['returns'].rolling(window).apply(lambda x: x.autocorr(lag=lag), raw=False)

    # =======================
    # 9. STATISTICAL FEATURES
    # =======================
    for window in [20, 50, 100]:
        df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
        df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()

    # =======================
    # 10. FOURIER FEATURES
    # =======================
    window_fft = 256
    if len(df) >= window_fft:
        fft_vals = fft(df['close'].values)
        fft_power = np.abs(fft_vals)**2
        df['dominant_cycle_power'] = pd.Series(fft_power).rolling(window_fft).mean().values
        df['dominant_cycle_power'].ffill(inplace=True)

    # =======================
    # 11. REGIME DETECTION FEATURES
    # =======================
    for window in [20, 50, 100]:
        df[f'trend_strength_{window}'] = df['close'].rolling(window).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    def hurst(ts):
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    if len(df) > 100:
        df['hurst_100'] = df['close'].rolling(100).apply(hurst, raw=True)
    
    # =======================
    # 12. INTERACTION FEATURES
    # =======================
    df['mom_vol_ratio_20'] = df['momentum_20'] / (df['realized_vol_20'] + 1e-10)
    df['cvd_momentum_div_50'] = (np.sign(df['momentum_50']) != np.sign(df['cvd_taker_50'])).astype(int)
    
    # =======================
    # 13. PERCENTILE RANK FEATURES
    # =======================
    features_to_rank = ['cvd_taker_50', 'vol_regime_20', 'ofi_50', 'vpin_proxy_50', 'momentum_20', 'rsi_14']
    for feature in features_to_rank:
        if feature in df.columns:
            for window in [100, 600]: # short, medium
                df[f'{feature}_pct_rank_{window}'] = df[feature].rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    # =======================
    # FINALIZATION
    # =======================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    log.info(f"Feature generation complete. Final shape: {df.shape}, Total features: {len(df.columns)}")
    return df

# --- Multi-Timeframe Merge Node (Improved) ---
def merge_multi_timeframe_features(base_features: pd.DataFrame, **other_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merges features from multiple timeframes onto a base dataframe.
    """
    log.info(f"Starting multi-timeframe feature merge. Base shape: {base_features.shape}")
    merged_df = base_features.sort_values('datetime').copy()
    
    exclude_cols = [
        'datetime', 'open', 'high', 'low', 'close', 'volume', 
        'mid_price', 'spread', 'micro_open', 'micro_high', 'micro_low', 
        'micro_close', 'taker_flow', 'ofi', 'num_trades',
        'hour', 'minute', 'day_of_week'
    ]
    
    for tf_name, df_tf in other_features.items():
        suffix = f"_{tf_name.split('_')[-1]}"
        df_to_merge = df_tf.sort_values('datetime').copy()
        
        feature_cols = [c for c in df_to_merge.columns if c not in exclude_cols]
        cols_to_rename = {col: col + suffix for col in feature_cols}
        df_to_merge = df_to_merge[['datetime'] + feature_cols].rename(columns=cols_to_rename)
        
        merged_df = pd.merge_asof(merged_df, df_to_merge, on='datetime', direction='backward')
        log.info(f"Merged {tf_name}: added {len(feature_cols)} features.")
    
    merged_df.dropna(inplace=True)
    log.info(f"Multi-timeframe merge complete. Final shape: {merged_df.shape}")
    
    return merged_df
