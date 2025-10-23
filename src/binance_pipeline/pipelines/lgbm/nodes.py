import pandas as pd
import numpy as np
import logging
import numba
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score
import lightgbm as lgb
from typing import Tuple, Dict

log = logging.getLogger(__name__)

# =======================
# OPTIMIZED TRIPLE BARRIER LABELING
# =======================

@numba.jit(nopython=True)
def _compute_triple_barrier_labels_vectorized(
    prices: np.ndarray,
    upper_barriers: np.ndarray,
    lower_barriers: np.ndarray,
    time_barrier_periods: int
) -> np.ndarray:
    """Numba-accelerated triple barrier labeling."""
    n = len(prices)
    labels = np.full(n, np.nan)
    
    for i in range(n - time_barrier_periods):
        upper = upper_barriers[i]
        lower = lower_barriers[i]
        
        first_upper_idx, first_lower_idx = -1, -1
        
        for j in range(i + 1, min(i + 1 + time_barrier_periods, n)):
            if prices[j] >= upper and first_upper_idx == -1:
                first_upper_idx = j
            if prices[j] <= lower and first_lower_idx == -1:
                first_lower_idx = j
            if first_upper_idx != -1 and first_lower_idx != -1:
                break
        
        if first_upper_idx != -1 and (first_lower_idx == -1 or first_upper_idx < first_lower_idx):
            labels[i] = 1
        elif first_lower_idx != -1:
            labels[i] = -1
        else:
            labels[i] = 0
    
    return labels

def generate_triple_barrier_labels(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Improved triple-barrier labeling with Numba."""
    log.info(f"Generating triple-barrier labels using '{params.get('method', 'volatility')}' method...")
    
    price = df['close'].values
    
    if params.get('method') == 'fixed_pct':
        profit_pct, loss_pct = params['profit_take_gross_pct'], params['stop_loss_gross_pct']
        upper_barrier = price * (1 + profit_pct)
        lower_barrier = price * (1 - loss_pct)
        log.info(f"Fixed barriers: TP={profit_pct*100:.4f}%, SL={loss_pct*100:.4f}%")
    else:
        log.info("Using volatility-based dynamic barriers.")
        price_series = pd.Series(price)
        daily_vol = price_series.diff().rolling(window=params['vol_lookback']).std().values
        vol_mult = params['vol_multiplier']
        upper_barrier = price + daily_vol * vol_mult * params['profit_take_mult']
        lower_barrier = price - daily_vol * vol_mult * params['stop_loss_mult']
    
    labels = _compute_triple_barrier_labels_vectorized(price, upper_barrier, lower_barrier, params['time_barrier_periods'])
    
    df['label_raw'] = labels
    df.dropna(subset=['label_raw'], inplace=True)
    df['label'] = df['label_raw'].map({-1: 0, 0: 1, 1: 2}).astype(int)
    
    log.info(f"Label distribution:\n{df['label'].value_counts(normalize=True).sort_index()}")
    return df

# =======================
# IMPROVED TRAINING WITH TIME-SERIES CV
# =======================

def train_lgbm_model(labeled_data: pd.DataFrame, params: dict) -> Tuple[lgb.LGBMClassifier, Dict]:
    """Improved LightGBM training with Time-series CV and class balancing."""
    log.info("Starting LightGBM model training with time-series CV...")
    
    exclude_cols = ['datetime', 'label', 'label_raw', 'open', 'high', 'low', 'close']
    features = [col for col in labeled_data.columns if col not in exclude_cols]
    log.info(f"Training with {len(features)} features.")
    
    X, y = labeled_data[features], labeled_data['label']
    
    class_counts = y.value_counts()
    class_weights = {cls: len(y) / (len(class_counts) * count) for cls, count in class_counts.items()}
    sample_weights = y.map(class_weights).values
    
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = {'accuracy': [], 'f1_macro': [], 'cohen_kappa': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        log.info(f"\n--- Fold {fold + 1}/3 ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weights[train_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
        
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        kappa = cohen_kappa_score(y_val, y_pred)
        
        cv_scores['accuracy'].append(acc)
        cv_scores['f1_macro'].append(f1_macro)
        cv_scores['cohen_kappa'].append(kappa)
        log.info(f"Fold {fold + 1} - Accuracy: {acc:.4f}, F1-Macro: {f1_macro:.4f}, Kappa: {kappa:.4f}")

    log.info("\n=== Cross-Validation Results ===")
    for metric, scores in cv_scores.items():
        log.info(f"{metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    log.info("\nRetraining model on all data...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y, sample_weight=sample_weights)
    
    feature_importance = pd.DataFrame({'feature': features, 'importance': final_model.feature_importances_}).sort_values('importance', ascending=False)
    log.info("\n=== Top 20 Important Features ===")
    log.info(feature_importance.head(20))
    
    eval_results = {'cv_scores': cv_scores, 'feature_importance': feature_importance.to_dict()}
    
    return final_model, eval_results
