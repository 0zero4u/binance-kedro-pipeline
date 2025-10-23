
import pandas as pd
import numpy as np
from river import preprocessing, forest, metrics
import numba
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

def generate_river_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features for the ARF model, now including percentile ranks
    over multiple lookback windows for dynamic, adaptive normalization.
    """
    log.info("Generating V2 features with multi-window percentile ranks...")
    
    
    # Define windows in terms of number of 100ms bars
    PERCENTILE_WINDOWS = {
        '10s': 100,      # Short-term, tactical window
        '1min': 600,     # Medium-term, trend context
        '3min': 1800     # Long-term, strategic context
    }
    
    features_to_rank = ['cvd_taker_50', 'vol_regime']
    
    for feature in features_to_rank:
        for name, window_size in PERCENTILE_WINDOWS.items():
            log.info(f"Calculating {name} percentile rank for {feature}...")
            # The lambda function calculates the percentile rank of the most recent value in the window.
            df[f'{feature}_pct_rank_{name}'] = df[feature].rolling(
                window=window_size, min_periods=int(window_size / 2)
            ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # --- Part 3: Finalize DataFrame ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    final_df = df.reset_index(drop=True)
    log.info(f"Feature generation complete. Final shape with all features: {final_df.shape}")
    return final_df


@numba.jit(nopython=True)
def _compute_labels_with_endtime_jit(prices, timestamps, upper_barriers, lower_barriers, time_barrier_periods):
    """Numba-accelerated function to compute triple-barrier labels and end times."""
    n_events = len(prices)
    labels = np.full(n_events, np.nan)
    t_end = np.full(n_events, np.nan, dtype=np.int64) # Store as int64 for numba compatibility

    for i in range(n_events - time_barrier_periods):
        upper = upper_barriers[i]
        lower = lower_barriers[i]
        time_barrier_end_time = timestamps[i + time_barrier_periods]

        first_upper_hit_idx = -1
        first_lower_hit_idx = -1

        # Check future prices within the time barrier
        for j in range(i + 1, i + 1 + time_barrier_periods):
            if prices[j] >= upper and first_upper_hit_idx == -1:
                first_upper_hit_idx = j
            if prices[j] <= lower and first_lower_hit_idx == -1:
                first_lower_hit_idx = j
            if first_upper_hit_idx != -1 and first_lower_hit_idx != -1:
                break # Both found

        if first_upper_hit_idx != -1 and (first_lower_hit_idx == -1 or first_upper_hit_idx < first_lower_hit_idx):
            labels[i] = 1 # Profit take
            t_end[i] = timestamps[first_upper_hit_idx]
        elif first_lower_hit_idx != -1:
            labels[i] = -1 # Stop loss
            t_end[i] = timestamps[first_lower_hit_idx]
        else:
            labels[i] = 0 # Time barrier
            t_end[i] = time_barrier_end_time
    
    return labels, t_end

def generate_triple_barrier_labels_with_endtime(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Applies the triple-barrier labeling method and adds the timestamp 't_end'
    when the event concluded. This is required for unbiased sample filtering.    
    Supports two methods controlled by params['method']:
    1. 'volatility': Dynamic barriers based on price volatility.
    2. 'fixed_pct': Fixed percentage barriers that account for fees.
    """
    log.info(f"Generating triple-barrier labels with end times using '{params.get('method', 'volatility')}' method...")
    
    df = df.set_index('datetime')
    price = df['close']
    
    # --- Barrier Calculation ---
    if params.get('method') == 'fixed_pct':
        upper_barrier = price * (1 + params['profit_take_gross_pct'])
        lower_barrier = price * (1 - params['stop_loss_gross_pct'])
        log.info(f"ARF using fixed barriers: TP at {params['profit_take_gross_pct']*100:.4f}%, SL at {params['stop_loss_gross_pct']*100:.4f}%.")
    else: # Default to original volatility-based method
        log.info("ARF using volatility-based barriers.")
        daily_vol = price.diff().rolling(window=params['vol_lookback']).std()
        upper_barrier = price + daily_vol * params['vol_multiplier'] * params['profit_take_mult']
        lower_barrier = price - daily_vol * params['vol_multiplier'] * params['stop_loss_mult']

    time_barrier_periods = params['time_barrier_periods']
    
    # --- Accelerated Event Detection Loop ---
    log.info("Starting accelerated labeling process with end times...")
    labels_arr, t_end_arr = _compute_labels_with_endtime_jit(
        price.to_numpy(),
        df.index.to_numpy().astype(np.int64), # Pass timestamps as integers
        upper_barrier.to_numpy(),
        lower_barrier.to_numpy(),
        time_barrier_periods
    )
    
    df['label'] = labels_arr
    # Convert integer timestamps back to datetime objects
    df['t_end'] = pd.to_datetime(t_end_arr, unit='ns') 

    df.dropna(subset=['label', 't_end'], inplace=True)
    df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2}).astype(int)
    log.info(f"Labeling complete. Label distribution:\n{df['label'].value_counts(normalize=True)}")
    return df.reset_index()

def filter_unbiased_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters labeled data to select only non-overlapping samples based on t_end.
    """
    log.info(f"Filtering for unbiased samples. Starting with {len(df)} samples.")
    
    unique_indices = []
    last_event_end_time = pd.Timestamp.min

    df = df.sort_values('datetime').reset_index(drop=True)

    for idx, row in df.iterrows():
        event_start_time = row['datetime']
        event_end_time = row['t_end']
        
        if event_start_time >= last_event_end_time:
            unique_indices.append(idx)
            last_event_end_time = event_end_time
            
    unbiased_df = df.loc[unique_indices]
    log.info(f"Filtering complete. {len(unbiased_df)} unique samples remaining.")
    
    return unbiased_df

def fit_hybrid_scaler(df: pd.DataFrame) -> StandardScaler:
    """
    Fits a scikit-learn StandardScaler on the data and returns it.
    """
    log.info("Fitting hybrid scaler...")
    
    feature_cols = [col for col in df.columns if col not in [
        'datetime', 'label', 't_end', 'open', 'high', 'low', 'close'
    ]]
    
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    
    log.info("Scaler fitting complete.")
    return scaler

def apply_hybrid_scaling(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Applies a pre-fit StandardScaler and then Soft Clipping to the dataset.
    """
    log.info("Applying hybrid scaling (StandardScaler + Soft Clipping)...")
    
    non_feature_cols = df[['datetime', 'label', 't_end']]
    feature_cols = [col for col in df.columns if col in scaler.feature_names_in_]
    X = df[feature_cols]
    
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    SOFT_CLIP_RANGE = 3.0
    X_clipped_df = SOFT_CLIP_RANGE * np.tanh(X_scaled_df / SOFT_CLIP_RANGE)
    
    final_df = pd.concat([non_feature_cols, X_clipped_df], axis=1)
    
    log.info("Hybrid scaling complete.")
    return final_df

def train_arf_model_on_scaled_data(df: pd.DataFrame):
    """
    Trains a River ARFClassifier on pre-scaled and clipped data.
    """
    log.info("Starting offline training of River ARF model on scaled data...")
    
    features = [col for col in df.columns if col not in ['datetime', 'label', 't_end']]
    target = 'label'
    
    X = df[features]
    y = df[target]
    
    model = forest.ARFClassifier(n_models=10, seed=42)
    metric = metrics.Accuracy() + metrics.MacroF1() + metrics.CohenKappa()

    for i in tqdm(range(len(X)), desc="Training ARF Model"):
        x_i = X.iloc[i].to_dict()
        y_i = y.iloc[i]
        
        y_pred = model.predict_one(x_i)
        if y_pred is not None:
            metric.update(y_true=y_i, y_pred=y_pred)
        model.learn_one(x=x_i, y=y_i)
        
    log.info("ARF Model training complete.")
    log.info(f"Final evaluation metrics (Progressive Validation): {metric}")
    
    return model
