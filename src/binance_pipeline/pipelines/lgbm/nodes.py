import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

log = logging.getLogger(__name__)

def generate_triple_barrier_labels(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Applies the triple-barrier labeling method on resampled bar data."""
    log.info("Generating triple-barrier labels on bar data...")
    price = df['close']
    
    daily_vol = price.diff().rolling(window=params['vol_lookback']).std()
    upper_barrier = price + daily_vol * params['vol_multiplier'] * params['profit_take_mult']
    lower_barrier = price - daily_vol * params['vol_multiplier'] * params['stop_loss_mult']
    
    labels = pd.Series(np.nan, index=df.index)
    
    for i in range(len(df) - params['time_barrier_periods']):
        future_prices = price.iloc[i+1 : i+1+params['time_barrier_periods']]
        hit_upper = future_prices[future_prices >= upper_barrier.iloc[i]]
        hit_lower = future_prices[future_prices <= lower_barrier.iloc[i]]
        
        if not hit_upper.empty and not hit_lower.empty:
            labels.iloc[i] = 1 if hit_upper.index[0] < hit_lower.index[0] else -1
        elif not hit_upper.empty:
            labels.iloc[i] = 1
        elif not hit_lower.empty:
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
