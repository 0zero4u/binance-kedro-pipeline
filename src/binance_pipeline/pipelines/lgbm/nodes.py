import pandas as pd
import numpy as np
import logging
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report
import lightgbm as lgb
from typing import Tuple, Dict, Any
import optuna
from binance_pipeline.features import AdvancedFeatureEngine

log = logging.getLogger(__name__)

class PurgedKFold:
    # ... (rest of the existing code in this file is unchanged) ...
    """Purged K-Fold for time-series to prevent lookahead bias."""
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame):
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]
            train_end = max(0, test_start - embargo_size)
            train_indices = indices[:train_end]
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

def _objective_lgbm(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, base_params: Dict) -> float:
    """Optuna objective function for LightGBM hyperparameter optimization."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        **base_params
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='macro')

def train_lgbm_model(labeled_data: pd.DataFrame, lgbm_params: Dict, training_params: Dict) -> Tuple[lgb.LGBMClassifier, Dict]:
    """
    UPGRADED LightGBM training with Optuna hyperparameter search, PurgedKFold CV,
    and integrated feature selection.
    """
    log.info("--- Starting Upgraded LightGBM Training ---")
    
    exclude_cols = ['datetime', 'label', 'label_raw', 'open', 'high', 'low', 'close', 'net_return', 'gross_return', 'barrier_hit_time']
    features = [col for col in labeled_data.columns if col not in exclude_cols]
    X, y = labeled_data[features], labeled_data['label']
    
    # --- 1. Hyperparameter Optimization with Optuna ---
    best_params = {}
    if training_params.get('use_optuna', False):
        log.info(f"Running Optuna hyperparameter search for {training_params['n_optuna_trials']} trials...")
        pkf = PurgedKFold(n_splits=training_params['n_cv_splits'], embargo_pct=training_params['embargo_pct'])
        train_idx, val_idx = next(pkf.split(X))
        X_train, X_val, y_train, y_val = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: _objective_lgbm(trial, X_train, y_train, X_val, y_val, lgbm_params), n_trials=training_params['n_optuna_trials'])
        best_params = study.best_params
        log.info(f"Best Optuna F1-Macro: {study.best_value:.4f}. Best params: {best_params}")

    final_params = {**lgbm_params, **best_params}

    # --- 2. Feature Selection (if enabled) ---
    selected_features = features
    adv_feature_engine = AdvancedFeatureEngine()
    if training_params.get('feature_selection', {}).get('enabled', False):
        log.info("Performing feature selection...")
        temp_model = lgb.LGBMClassifier(**final_params)
        temp_model.fit(X, y)
        importance_dict = dict(zip(temp_model.feature_name_, temp_model.feature_importances_))
        top_k = training_params['feature_selection']['top_k_features']
        selected_features = adv_feature_engine.select_features(X, importance_dict, top_k)
        X = X[selected_features] # Update X with selected features

    # --- 3. Final Model Training and CV on Selected Features ---
    log.info(f"Training final model on {len(selected_features)} features with PurgedKFold CV...")
    pkf = PurgedKFold(n_splits=training_params['n_cv_splits'], embargo_pct=training_params['embargo_pct'])
    cv_scores = {'f1_macro': [], 'cohen_kappa': []}

    for fold, (train_idx, val_idx) in enumerate(pkf.split(X)):
        X_train, X_val, y_train, y_val = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMClassifier(**final_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        y_pred = model.predict(X_val)
        f1, kappa = f1_score(y_val, y_pred, average='macro'), cohen_kappa_score(y_val, y_pred)
        cv_scores['f1_macro'].append(f1)
        cv_scores['cohen_kappa'].append(kappa)
        log.info(f"Fold {fold+1} - F1-Macro: {f1:.4f}, Kappa: {kappa:.4f}")

    log.info("\n--- Cross-Validation Results ---")
    log.info(f"F1-Macro: {np.mean(cv_scores['f1_macro']):.4f} (+/- {np.std(cv_scores['f1_macro']):.4f})")
    
    log.info("Retraining final model on all data...")
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X, y)
    
    feature_importance_df = pd.DataFrame({'feature': selected_features, 'importance': final_model.feature_importances_}).sort_values('importance', ascending=False)
    log.info("\n--- Top 20 Important Features ---")
    log.info(feature_importance_df.head(20).to_string(index=False))
    
    eval_results = {'cv_scores': cv_scores, 'feature_importance': feature_importance_df.to_dict(), 'best_params': best_params}
    
    return final_model, eval_results

# --- NEW FUNCTION: Evaluate on the holdout test set ---
def evaluate_model(model: lgb.LGBMClassifier, test_labeled_data: pd.DataFrame) -> Dict:
    """
    Evaluates the final trained model on the unseen holdout test set to get a
    true measure of its generalization performance.
    """
    log.info(f"Evaluating final model on holdout test set of shape {test_labeled_data.shape}...")
    
    # Ensure the test set has the features the model was trained on
    X_test = test_labeled_data[model.feature_name_]
    y_test = test_labeled_data['label']
    
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    log.info("\n" + "="*50)
    log.info("--- HOLDOUT TEST SET PERFORMANCE ---")
    log.info(f"F1-Macro: {f1:.4f}")
    log.info(f"Cohen's Kappa: {kappa:.4f}")
    log.info("Classification Report:")
    log.info(classification_report(y_test, y_pred))
    log.info("="*50 + "\n")
    
    return {"test_f1_macro": f1, "test_cohen_kappa": kappa, "test_classification_report": report}
