
import pandas as pd
import numpy as np
from river import preprocessing, forest, metrics
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

def generate_river_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features for the ARF model, now including percentile ranks
    for dynamic, adaptive normalization.
    """
    log.info("Generating V2 features with percentile ranks...")
    
    # --- Part 1: Calculate Base Features (as before) ---
    df['order_flow'] = np.sign(df['close'] - df['open']) * df['volume']
    df['order_flow'].fillna(0, inplace=True)
    df['cvd_10'] = df['order_flow'].rolling(window=10).sum()
    df['cvd_50'] = df['order_flow'].rolling(window=50).sum()
    df['price_mom_5'] = np.log(df['close'] / df['close'].shift(5))
    df['price_mom_20'] = np.log(df['close'] / df['close'].shift(20))
    df['volatility_20'] = df['price_mom_5'].rolling(window=20).std()
    
    # --- Part 2: NEW - Calculate Percentile Rank Features ---
    # This transforms features into a measure of their relative strength
    # compared to the recent past, making them regime-adaptive.
    PERCENTILE_WINDOW = 400 
    
    # The lambda function calculates the percentile rank of the most recent value in the window.
    # `raw=False` ensures pandas passes a Series object to the lambda function.
    df['cvd_10_pct_rank'] = df['cvd_10'].rolling(
        window=PERCENTILE_WINDOW, min_periods=int(PERCENTILE_WINDOW/2)
    ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    df['volatility_20_pct_rank'] = df['volatility_20'].rolling(
        window=PERCENTILE_WINDOW, min_periods=int(PERCENTILE_WINDOW/2)
    ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # --- Part 3: Finalize DataFrame ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    final_df = df.reset_index(drop=True)
    log.info(f"Feature generation complete. Final shape with all features: {final_df.shape}")
    return final_df


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

    labels = pd.Series(np.nan, index=df.index)
    t_end = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    time_barrier_periods = params['time_barrier_periods']

    for i in tqdm(range(len(df) - time_barrier_periods), desc="Labeling Events"):
        start_time = df.index[i]
        time_barrier_end_time = df.index[i + time_barrier_periods]
        
        future_slice = df.iloc[i+1 : i+1 + time_barrier_periods]
        
        hit_upper_slice = future_slice[future_slice['close'] >= upper_barrier.loc[start_time]]
        hit_lower_slice = future_slice[future_slice['close'] <= lower_barrier.loc[start_time]]
        
        first_upper_hit_time = hit_upper_slice.index.min()
        first_lower_hit_time = hit_lower_slice.index.min()

        if pd.notna(first_upper_hit_time) and pd.notna(first_lower_hit_time):
            if first_upper_hit_time < first_lower_hit_time:
                labels.loc[start_time] = 1
                t_end.loc[start_time] = first_upper_hit_time
            else:
                labels.loc[start_time] = -1
                t_end.loc[start_time] = first_lower_hit_time
        elif pd.notna(first_upper_hit_time):
            labels.loc[start_time] = 1
            t_end.loc[start_time] = first_upper_hit_time
        elif pd.notna(first_lower_hit_time):
            labels.loc[start_time] = -1
            t_end.loc[start_time] = first_lower_hit_time
        else:
            labels.loc[start_time] = 0
            t_end.loc[start_time] = time_barrier_end_time
            
    df['label'] = labels
    df['t_end'] = t_end
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
