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

# --- Merge Node (High-Performance) ---
def merge_book_trade_asof(book_raw: pd.DataFrame, trade_raw: pd.DataFrame) -> pd.DataFrame:
    """Merges book and trade data using as-of join."""
    log.info("Starting fast as-of merge...")
    log.info(f"  - Input 'book_raw' shape: {book_raw.shape}")
    log.info(f"  - Input 'trade_raw' shape: {trade_raw.shape}")

    # Prepare trades
    log.info("  - Preparing trade data (sorting, cleaning, and type conversion)...")
    trades = trade_raw[['time', 'price', 'qty', 'is_buyer_maker']].copy()
    trades.rename(columns={'time': 'timestamp'}, inplace=True)
    trades['timestamp'] = pd.to_numeric(trades['timestamp'], errors='coerce')
    
    # Explicitly convert price and quantity columns to numeric types.
    trades['price'] = pd.to_numeric(trades['price'], errors='coerce')
    trades['qty'] = pd.to_numeric(trades['qty'], errors='coerce')

    trades.dropna(subset=['timestamp'], inplace=True)
    trades.sort_values('timestamp', inplace=True)

    # Prepare book
    log.info("  - Preparing book data (sorting, cleaning, and type conversion)...")
    book = book_raw[['event_time', 'best_bid_price', 'best_ask_price',
                     'best_bid_qty', 'best_ask_qty']].copy()
    book.rename(columns={'event_time': 'timestamp'}, inplace=True)
    book['timestamp'] = pd.to_numeric(book['timestamp'], errors='coerce')

    # Explicitly convert all book-related columns to numeric types.
    book['best_bid_price'] = pd.to_numeric(book['best_bid_price'], errors='coerce')
    book['best_ask_price'] = pd.to_numeric(book['best_ask_price'], errors='coerce')
    book['best_bid_qty'] = pd.to_numeric(book['best_bid_qty'], errors='coerce')
    book['best_ask_qty'] = pd.to_numeric(book['best_ask_qty'], errors='coerce')

    book.dropna(subset=['timestamp'], inplace=True)
    book = book.drop_duplicates(subset='timestamp', keep='last')
    book.sort_values('timestamp', inplace=True)

    # Merge
    log.info("  - Performing pd.merge_asof (this is the slowest part of this node)...")
    start_time = time.time()
    merged_df = pd.merge_asof(left=trades, right=book, on='timestamp', direction='backward')
    log.info(f"  - Merge completed in {time.time() - start_time:.2f} seconds.")
    
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
    Calculates advanced tick-level features.
    """
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

# --- Resample Node ---
def resample_to_time_bars(df: pd.DataFrame, rule: str = "100ms") -> pd.DataFrame:
    """Resamples tick data to time bars with comprehensive aggregations."""
    log.info(f"Resampling data to '{rule}' bars...")
    log.info(f"  - Input shape: {df.shape}")
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

    log.info("  - Starting .resample().agg() (this can be slow)...")
    start_time = time.time()
    resampled_df = df.resample(rule).agg(aggregations)
    log.info(f"  - Aggregation completed in {time.time() - start_time:.2f} seconds.")
    
    # Robustly flatten the column MultiIndex, handling both tuples and strings.
    resampled_df.columns = [
        '_'.join(col).strip() if isinstance(col, tuple) else col
        for col in resampled_df.columns.values
    ]

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
    log.info(f"Resampling complete. Output shape: {resampled_df.shape}")
    return resampled_df.reset_index()


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

# --- Advanced Feature Engineering Node ---
def generate_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates comprehensive bar-based features with detailed progress logging.
    """
    log.info(f"Generating comprehensive bar features for dataframe of shape {df.shape}...")
    df = df.copy()
    
    total_steps = 13
    
    def log_progress(step, message):
        log.info(f"  [{(step/total_steps)*100:3.0f}%] ({step}/{total_steps}) {message}")

    log_progress(1, "Calculating basic technical indicators...")
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['rsi_28'] = ta.momentum.RSIIndicator(close=df['close'], window=28).rsi()
    df['atr_14'] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    for window in [50, 100, 200]:
        df[f'vwap_{window}'] = (df['volume'] * df['close']).rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-10)
        df[f'price_to_vwap_{window}'] = df['close'] / df[f'vwap_{window}']

    log_progress(2, "Calculating volatility features...")
    df['gk_vol'] = 0.5 * np.log(df['high'] / df['low'])**2 - (2 * np.log(2) - 1) * np.log(df['close'] / df['open'])**2
    for window in [20, 50, 100]:
        df[f'gk_vol_{window}'] = df['gk_vol'].rolling(window=window).mean()
        df[f'vol_regime_{window}'] = df['gk_vol'] / (df[f'gk_vol_{window}'] + 1e-10)
    for window in [10, 20, 50]:
        df[f'realized_vol_{window}'] = df['log_returns'].rolling(window).std() * np.sqrt(window)

    log_progress(3, "Calculating momentum features...")
    for period in [5, 10, 20, 50, 100]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        df[f'momentum_pct_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
    df['momentum_acceleration_10'] = df['momentum_10'] - df['momentum_10'].shift(1)
    df['momentum_acceleration_20'] = df['momentum_20'] - df['momentum_20'].shift(1)

    log_progress(4, "Calculating order flow features...")
    for window in [20, 50, 100, 200]:
        df[f'cvd_taker_{window}'] = df['taker_flow'].rolling(window=window).sum()
        df[f'cvd_velocity_{window}'] = df[f'cvd_taker_{window}'].diff(1)
        df[f'cvd_accel_{window}'] = df[f'cvd_velocity_{window}'].diff(1)
    for window in [20, 50, 100]:
        df[f'ofi_{window}'] = df['ofi'].rolling(window=window).sum()
        df[f'ofi_std_{window}'] = df['ofi'].rolling(window=window).std()

    log_progress(5, "Calculating volume features...")
    for window in [20, 50, 100]:
        df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
        df[f'volume_ratio_{window}'] = df['volume'] / (df[f'volume_ma_{window}'] + 1e-10)
    for window in [50, 100]:
        df[f'vpin_proxy_{window}'] = df['taker_flow'].abs().rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-10)

    log_progress(6, "Calculating spread & liquidity features...")
    for window in [20, 50]:
        df[f'amihud_{window}'] = abs(df['returns']).rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-10)
        rolling_cov = df['returns'].rolling(window).cov(df['taker_flow'])
        rolling_var = df['taker_flow'].rolling(window).var()
        df[f'kyles_lambda_{window}'] = rolling_cov / (rolling_var + 1e-10)
    if 'book_imbalance' in df.columns:
        for window in [20, 50]:
            df[f'book_imb_ma_{window}'] = df['book_imbalance'].rolling(window).mean()
            df[f'book_imb_std_{window}'] = df['book_imbalance'].rolling(window).std()

    log_progress(7, "Calculating temporal features...")
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

    log_progress(8, "Calculating autocorrelation & lag features...")
    for lag in [1, 2, 3, 5, 10]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
    for window in [50, 100]:
        for lag in [1, 5]:
            df[f'returns_autocorr_lag{lag}_w{window}'] = df['returns'].rolling(window).apply(lambda x: x.autocorr(lag=lag), raw=False)

    log_progress(9, "Calculating statistical features...")
    for window in [20, 50, 100]:
        df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
        df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()

    log_progress(10, "Calculating fourier features...")
    window_fft = 256
    if len(df) >= window_fft:
        rolling_fft_power = df['close'].rolling(window_fft).apply(lambda x: np.max(np.abs(fft(x.values)[1:len(x)//2])**2) if len(x) > 2 else 0.0, raw=False)
        df['dominant_cycle_power'] = rolling_fft_power
        df['dominant_cycle_power'].ffill(inplace=True)

    log_progress(11, "Calculating regime detection features (Hurst is slow)...")
    for window in [20, 50, 100]:
        df[f'trend_strength_{window}'] = apply_rolling_numba(df['close'], _rolling_slope_numba, window)
    
    def hurst(ts):
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    if len(df) > 100:
        df['hurst_100'] = df['close'].rolling(100).apply(hurst, raw=True)

    log_progress(12, "Calculating interaction features...")
    df['mom_vol_ratio_20'] = df['momentum_20'] / (df['realized_vol_20'] + 1e-10)
    df['cvd_momentum_div_50'] = (np.sign(df['momentum_50']) != np.sign(df['cvd_taker_50'])).astype(int)

    log_progress(13, "Calculating percentile rank features...")
    features_to_rank = ['cvd_taker_50', 'vol_regime_20', 'ofi_50', 'vpin_proxy_50', 'momentum_20', 'rsi_14']
    for feature in features_to_rank:
        if feature in df.columns:
            for window in [100, 600]:
                df[f'{feature}_pct_rank_{window}'] = apply_rolling_numba(df[feature], _rolling_rank_pct_numba, window)

    log.info("  -> Finalizing dataframe (handling inf, NaN)...")
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
