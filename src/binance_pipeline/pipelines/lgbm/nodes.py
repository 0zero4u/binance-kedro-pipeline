import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score
import lightgbm as lgb
from typing import Tuple, Dict

log = logging.getLogger(__name__)

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
  
