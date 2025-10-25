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
import tempfile

log = logging.getLogger(__name__)

# --- Download Node (Unchanged) ---
def download_and_unzip(url: str, output_dir: str):
    p_output_dir = Path(output_dir); p_output_dir.mkdir(parents=True, exist_ok=True)
    file_name = url.split('/')[-1]; zip_path = p_output_dir / file_name
    log.info(f"Downloading from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status(); total_size = int(r.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(desc=f"Downloading {file_name}", total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk); bar.update(size)
    log.info(f"Unzipping {zip_path}...");
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(p_output_dir)
    os.remove(zip_path); log.info(f"Download and unzip complete for {file_name}.")

# --- Merge Node (Unchanged) ---
def merge_book_trade_asof(book_raw: pd.DataFrame, trade_raw: pd.DataFrame) -> pd.DataFrame:
    log.info("Starting fast as-of merge with Polars...")
    trades_pl = pl.from_pandas(trade_raw).select(["time", "price", "qty", "is_buyer_maker"]).rename({"time": "timestamp"}).cast({"timestamp": pl.Int64, "price": pl.Float64, "qty": pl.Float64}).drop_nulls(subset="timestamp").sort("timestamp")
    book_pl = pl.from_pandas(book_raw).select(["event_time", "best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty"]).rename({"event_time": "timestamp"}).cast({"timestamp": pl.Int64, "best_bid_price": pl.Float64, "best_ask_price": pl.Float64, "best_bid_qty": pl.Float64, "best_ask_qty": pl.Float64}).drop_nulls(subset="timestamp").unique(subset="timestamp", keep="last", maintain_order=True).sort("timestamp")
    merged_pl = trades_pl.join_asof(book_pl, on="timestamp", strategy="backward")
    final_pl = merged_pl.drop_nulls(subset=["best_bid_price", "best_ask_price"]).with_columns([(((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2).alias("mid_price")), ((pl.col("best_ask_price") - pl.col("best_bid_price")).alias("spread"))]).with_columns([((pl.col("spread") / pl.col("mid_price")) * 10000).alias("spread_bps")])
    return final_pl.to_pandas()

# --- Tick-Level Features Node (Unchanged) ---
def calculate_tick_level_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Calculating advanced tick-level features for dataframe of shape {df.shape}...")
    df['microprice'] = ((df['best_bid_price'] * df['best_ask_qty']) + (df['best_ask_price'] * df['best_bid_qty'])) / (df['best_bid_qty'] + df['best_ask_qty'])
    df['microprice'].ffill(inplace=True); df['taker_flow'] = np.where(df['is_buyer_maker'], -df['qty'], df['qty'])
    bid_price_diff, ask_price_diff = df['best_bid_price'].diff(), df['best_ask_price'].diff()
    bid_qty_diff, ask_qty_diff = df['best_bid_qty'].diff(), df['best_ask_qty'].diff()
    bid_pressure = np.where(bid_price_diff >= 0, bid_qty_diff, 0); ask_pressure = np.where(ask_price_diff <= 0, ask_qty_diff, 0)
    df['ofi'] = bid_pressure - ask_pressure; df['ofi'].fillna(0, inplace=True)
    df['book_imbalance'] = (df['best_bid_qty'] - df['best_ask_qty']) / (df['best_bid_qty'] + df['best_ask_qty'] + 1e-10)
    log.info("Tick-level feature calculation complete."); return df

# --- EWMA TBT Feature Node (UPGRADED with THREE-STAGE EXECUTION) ---
def calculate_ewma_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates features using a THREE-STAGE streaming approach to be both memory-safe and high-performance,
    isolating the expensive time-based rolling window calculation.
    """
    log.info(f"Calculating TBT features with THREE-STAGE STREAMING IO for shape {df.shape}...")
    start_time = time.time()

    ewma_half_life_rows = {'5s': 200, '15s': 600, '1m': 2400, '3m': 7200, '15m': 36000}
    FASTEST_SPAN_KEY, SLOWEST_SPAN_KEY = '5s', '15m'
    wall_clock_windows = {'60s': '60s'}
    features_to_smooth = ['mid_price', 'spread', 'spread_bps', 'microprice', 'taker_flow', 'ofi', 'book_imbalance', 'price', 'qty']

    with tempfile.TemporaryDirectory() as tmpdir:
        stage1_path = os.path.join(tmpdir, "stage1.parquet")
        stage2_path = os.path.join(tmpdir, "stage2.parquet")
        final_path = os.path.join(tmpdir, "final.parquet")

        # --- STAGE 1 (Fast): Calculate expensive EWMA features and sink to disk ---
        log.info("  - Starting Stage 1: Calculating core EWMAs...")
        ewma_exprs = [pl.col(feat).ewm_mean(half_life=hl).alias(f'{feat}_ewma_{name}') for feat in features_to_smooth for name, hl in ewma_half_life_rows.items()]
        (pl.from_pandas(df).lazy()
         .with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime")).sort("datetime")
         .with_columns(ewma_exprs).sink_parquet(stage1_path))
        log.info(f"  - Stage 1 complete in {time.time() - start_time:.2f}s.")

        # --- STAGE 2 (Instant): Calculate fast, simple derivatives and sink to disk ---
        stage2_start_time = time.time()
        log.info("  - Starting Stage 2: Calculating simple derivatives (momentum, diff)...")
        momentum_accel_exprs = []
        for feature in features_to_smooth:
            fast_col, slow_col = f'{feature}_ewma_{FASTEST_SPAN_KEY}', f'{feature}_ewma_{SLOWEST_SPAN_KEY}'
            if feature in ['mid_price', 'microprice', 'taker_flow', 'ofi', 'spread_bps']:
                momentum_accel_exprs.append((pl.col(fast_col) - pl.col(slow_col)).alias(f'{feature}_momentum'))
            momentum_accel_exprs.append(pl.col(fast_col).diff(1).alias(f'{feature}_velocity'))
            momentum_accel_exprs.append(pl.col(fast_col).diff(1).diff(1).alias(f'{feature}_accel'))
        
        (pl.scan_parquet(stage1_path).with_columns(momentum_accel_exprs).sink_parquet(stage2_path))
        log.info(f"  - Stage 2 complete in {time.time() - stage2_start_time:.2f}s.")

        # --- STAGE 3 (The Bottleneck): Isolate and calculate the slow time-based rolling sum ---
        stage3_start_time = time.time()
        log.info("  - Starting Stage 3: Calculating time-based rolling sums...")
        wall_clock_exprs = [pl.col(feat).rolling(index_column="datetime", period=win_str).sum().alias(f'{feat}_rollsum_{name}') for feat in ['taker_flow', 'ofi'] for name, win_str in wall_clock_windows.items()]
        
        (pl.scan_parquet(stage2_path).with_columns(wall_clock_exprs).sink_parquet(final_path))
        log.info(f"  - Stage 3 complete in {time.time() - stage3_start_time:.2f}s.")
        
        df_out = pd.read_parquet(final_path)

    log.info(f"Staged feature calculation complete in {time.time() - start_time:.2f}s. Final shape: {df_out.shape}")
    return df_out

# --- Sampling Node (Unchanged) ---
def sample_features_to_grid(df: pd.DataFrame, rule: str = '25ms') -> pd.DataFrame:
    log.info(f"Sampling TBT features onto a fixed {rule} grid using memory-efficient aggregation...")
    if df.empty: log.warning("Input to 'sample_features_to_grid' is empty."); return pd.DataFrame()
    if 'datetime' not in df.columns: df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')
    feature_cols = [col for col in df.columns if col != 'price']
    aggregations = {col: 'last' for col in feature_cols}; aggregations['price'] = 'ohlc'
    sampled_df = df.resample(rule).agg(aggregations)
    if isinstance(sampled_df.columns, pd.MultiIndex):
        sampled_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in sampled_df.columns.values]
        sampled_df.rename(columns={'price_open': 'open', 'price_high': 'high', 'price_low': 'low', 'price_close': 'close'}, inplace=True)
    sampled_df.ffill(inplace=True); sampled_df.dropna(inplace=True)
    log.info(f"Memory-efficient sampling complete. Output shape: {sampled_df.shape}"); return sampled_df.reset_index()

# --- Helper Functions and Bar Features Node (Unchanged) ---
@numba.jit(nopython=True, fastmath=True)
def _rolling_slope_numba(y: np.ndarray, window: int) -> np.ndarray:
    n=len(y);out=np.full(n,np.nan);x=np.arange(window);sum_x=np.sum(x);sum_x2=np.sum(x*x);denominator=window*sum_x2-sum_x*sum_x
    if denominator==0:return out
    for i in range(window-1,n):
        y_win=y[i-window+1:i+1];sum_y=np.sum(y_win);sum_xy=np.sum(x*y_win)
        slope=(window*sum_xy-sum_x*sum_y)/denominator;out[i]=slope
    return out
@numba.jit(nopython=True, fastmath=True)
def _rolling_rank_pct_numba(y: np.ndarray, window: int) -> np.ndarray:
    n=len(y);out=np.full(n,np.nan)
    for i in range(window-1,n):
        win=y[i-window+1:i+1];last_val=win[-1];count_le=0
        for val in win:
            if val<=last_val:count_le+=1
        out[i]=count_le/window
    return out
def apply_rolling_numba(series: pd.Series, func, window: int) -> pd.Series:
    values=series.to_numpy();result=func(values, window);return pd.Series(result, index=series.index, name=series.name)
def generate_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: log.warning("Input to 'generate_bar_features' is empty."); return df.copy()
    log.info(f"Generating secondary/legacy bar features for dataframe of shape {df.shape}..."); df=df.copy()
    df['returns']=df['close'].pct_change(); df['log_returns']=np.log(df['close']/df['close'].shift(1))
    df['rsi_14']=ta.momentum.RSIIndicator(close=df['close'], window=14).rsi(); df['rsi_28']=ta.momentum.RSIIndicator(close=df['close'], window=28).rsi()
    def hurst(ts):
        lags=range(2,100);tau=[np.sqrt(np.std(np.subtract(ts[lag:],ts[:-lag]))) for lag in lags]
        poly=np.polyfit(np.log(lags),np.log(tau),1);return poly[0]*2.0
    window_hurst=100
    if len(df)>window_hurst: df['hurst_100']=df['close'].rolling(window_hurst).apply(hurst, raw=True)
    else: log.warning(f"Skipping Hurst calculation, DataFrame size ({len(df)}) is too small.")
    df['hour']=df['datetime'].dt.hour; df['minute']=df['datetime'].dt.minute; df['day_of_week']=df['datetime'].dt.dayofweek
    df['hour_sin']=np.sin(2*np.pi*df['hour']/24); df['hour_cos']=np.cos(2*np.pi*df['hour']/24)
    df.replace([np.inf,-np.inf],np.nan,inplace=True); df.dropna(inplace=True)
    log.info(f"Secondary feature generation complete. Final shape: {df.shape}"); return df
