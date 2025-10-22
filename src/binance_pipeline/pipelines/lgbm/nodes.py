import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

log = logging.getLogger(__name__)

def generate_triple_barrier_labels(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Applies the triple-barrier labeling method on resampled bar data.
    Supports two methods controlled by params['method']:
    1. 'volatility': Dynamic barriers based on price volatility.
    2. 'fixed_pct': Fixed percentage barriers that account for fees.
    """
    log.info(f"Generating triple-barrier labels using '{params.get('method', 'volatility')}' method...")
    price = df['close']

    # --- Barrier Calculation ---
    if params.get('method') == 'fixed_pct':
        # --- LOGIC UPDATED FOR GROSS PROFIT TARGET ---
        # The profit-take barrier is now the gross percentage directly.
        upper_barrier = price * (1 + params['profit_take_gross_pct'])
        
        # The stop-loss barrier remains the same (gross percentage).
        lower_barrier = price * (1 - params['stop_loss_gross_pct'])
        
        # For logging, calculate the expected net profit
        net_profit_pct = params['profit_take_gross_pct'] - (2 * params['fee_pct'])
        
        log.info(
            f"Fixed barriers: TP triggers at {params['profit_take_gross_pct']*100:.4f}% gross up-move "
            f"(expected net: {net_profit_pct*100:.4f}%), "
            f"SL triggers at {params['stop_loss_gross_pct']*100:.4f}% gross down-move."
        )

    else: # Default to original volatility-based method
        log.info("Using volatility-based barriers.")
        daily_vol = price.diff().rolling(window=params['vol_lookback']).std()
        upper_barrier = price + daily_vol * params['vol_multiplier'] * params['profit_take_mult']
        lower_barrier = price - daily_vol * params['vol_multiplier'] * params['stop_loss_mult']

    labels = pd.Series(np.nan, index=df.index)
    time_barrier_periods = params['time_barrier_periods']

    # --- Event Detection Loop ---
    # NOTE: This loop can be slow on large datasets. Consider vectorizing for production use.
    for i in range(len(df) - time_barrier_periods):
        future_prices = price.iloc[i+1 : i+1 + time_barrier_periods]
        hit_upper_times = future_prices[future_prices >= upper_barrier.iloc[i]].index
        hit_lower_times = future_prices[future_prices <= lower_barrier.iloc[i]].index
        
        if not hit_upper_times.empty and not hit_lower_times.empty:
            labels.iloc[i] = 1 if hit_upper_times[0] < hit_lower_times[0] else -1
        elif not hit_upper_times.empty:
            labels.iloc[i] = 1
        elif not hit_lower_times.empty:
            labels.iloc[i] = -1
        else:
            labels.iloc[i] = 0
            
    df['label'] = labels
    df.dropna(inplace=True)
    df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2}).astype(int)
    log.info(f"Labeling complete. Label distribution:\n{df['label'].value_counts(normalize=True)}")
    return df


def train_lgbm_model(labeled_data: pd.DataFrame, params: dict):
    """Trains a LightGBM classifier on the bar-based features."""
    log.info("Starting LightGBM model training on bar data...")
    
    features = [col for col in labeled_data.columns if col not in ['datetime', 'label']]
    target = 'label'
    
    X = labeled_data[features]
    y = labeled_data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='multi_logloss',
              callbacks=[lgb.early_stopping(15, verbose=False)])
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Lower', 'Time', 'Upper'])
    
    log.info(f"LGBM Model training complete. Test Accuracy: {acc:.4f}")
    log.info(f"Classification Report:\n{report}")
    return model
