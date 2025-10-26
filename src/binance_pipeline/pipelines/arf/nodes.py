import pandas as pd
import numpy as np
from river import forest, metrics
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report
from tqdm import tqdm
import logging
from typing import Dict, Any
import optuna

log = logging.getLogger(__name__)

from binance_pipeline.nodes import apply_rolling_numba, _rolling_rank_pct_numba
from binance_pipeline.pipelines.data_science.nodes import generate_triple_barrier_labels

def generate_arf_features(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Generates percentile rank features for the ARF model using parameters."""
    log.info("Generating ARF features with multi-window percentile ranks...")
    
    percentile_windows = params.get('percentile_windows', {})
    features_to_rank = params.get('features_to_rank', [])

    if not percentile_windows or not features_to_rank:
        log.warning("'percentile_windows' or 'features_to_rank' not found in arf.feature_params. Skipping.")
        return df.copy()

    log.info(f"Applying percentile rank to {len(features_to_rank)} features.")
    
    df_out = df.copy()
    for feature in features_to_rank:
        if feature in df_out.columns:
            for name, window_size in percentile_windows.items():
                df_out[f'{feature}_pct_rank_{name}'] = apply_rolling_numba(
                    df_out[feature], _rolling_rank_pct_numba, window_size
                )
        else:
            log.warning(f"Feature '{feature}' not found in dataframe for ARF ranking.")

    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_out.dropna(inplace=True)
    
    log.info(f"ARF feature generation complete. Final shape: {df_out.shape}")
    return df_out.reset_index(drop=True)

def generate_arf_hpo_configs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Uses Optuna to generate a dictionary of hyperparameter configurations to test."""
    log.info(f"Generating {params['n_trials']} ARF hyperparameter configurations with Optuna...")

    def objective(trial: optuna.Trial):
        config = {
            "lambda_value": trial.suggest_int("lambda_value", **params['search_space']['lambda_value']),
            "grace_period": trial.suggest_int("grace_period", **params['search_space']['grace_period']),
            "n_models": trial.suggest_int("n_models", **params['search_space']['n_models']),
            "seed": trial.number + 42
        }
        return 1.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=params['n_trials'])

    hpo_configs = {f"trial_{i}": trial.params for i, trial in enumerate(study.trials)}
    log.info(f"Generated {len(hpo_configs)} configurations.")
    return hpo_configs


def fit_robust_scaler(df: pd.DataFrame) -> RobustScaler:
    """Fits a RobustScaler, which is better for data with outliers."""
    log.info("Fitting RobustScaler...")
    feature_cols = [col for col in df.columns if col not in ['datetime', 'label', 'label_raw', 'open', 'high', 'low', 'close', 'net_return', 'gross_return', 'barrier_hit_time']]
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    scaler.fit(df[feature_cols])
    return scaler

def apply_robust_scaling(df: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """Applies a pre-fit RobustScaler and soft clipping."""
    log.info("Applying RobustScaler and soft clipping...")
    non_feature_cols = df[['datetime', 'label']]
    feature_cols = [col for col in df.columns if col in scaler.feature_names_in_]
    X = df[feature_cols]
    
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    X_clipped_df = 3.0 * np.tanh(X_scaled_df / 3.0)
    
    return pd.concat([non_feature_cols, X_clipped_df], axis=1)

def train_arf_ensemble(df: pd.DataFrame, hpo_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Trains an ensemble of ARF models using Optuna-generated configurations."""
    log.info(f"Starting ARF ensemble training with {len(hpo_configs)} auto-generated models...")
    
    features = [col for col in df.columns if col not in ['datetime', 'label']]
    X, y = df[features], df['label']
    
    models = {
        name: forest.ARFClassifier(**model_params)
        for name, model_params in hpo_configs.items()
    }
    metrics_dict = {name: metrics.Accuracy() + metrics.MacroF1() for name in models.keys()}
    
    for i in tqdm(range(len(X)), desc="Training ARF HPO Ensemble"):
        x_i, y_i = X.iloc[i].to_dict(), y.iloc[i]
        for name, model in models.items():
            y_pred = model.predict_one(x_i)
            if y_pred is not None:
                metrics_dict[name].update(y_true=y_i, y_pred=y_pred)
            model.learn_one(x=x_i, y=y_i)
            
    log.info("\n=== ARF HPO Training Complete ===")
    for name, metric in metrics_dict.items():
        log.info(f"{name} ({hpo_configs[name]}): {metric}")
        
    best_model_name = max(metrics_dict, key=lambda name: metrics_dict[name].get('MacroF1').get())
    log.info(f"\nBest model selected: {best_model_name} with F1={metrics_dict[best_model_name].get('MacroF1').get():.4f}")
    
    return {'best_model': models[best_model_name], 'metrics': str(metrics_dict)}

def select_best_arf_model(results: Dict[str, Any]):
    """Extracts the best model from the training results dictionary."""
    return results['best_model']

# --- NEW FUNCTION: Evaluate the final ARF model on the holdout test set ---
def evaluate_arf_model(
    model: forest.ARFClassifier,
    scaler: RobustScaler,
    test_labeled_data: pd.DataFrame
) -> Dict:
    """
    Evaluates the final trained ARF model on the unseen holdout test set.
    """
    log.info(f"Evaluating final ARF model on holdout test set of shape {test_labeled_data.shape}...")

    # Prepare test data: scale features and soft clip
    non_feature_cols = test_labeled_data[['datetime', 'label']]
    feature_cols = [col for col in test_labeled_data.columns if col in scaler.feature_names_in_]
    X_test = test_labeled_data[feature_cols]
    y_test = test_labeled_data['label']
    
    X_scaled = scaler.transform(X_test)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_test.index, columns=X_test.columns)
    X_clipped_df = 3.0 * np.tanh(X_scaled_df / 3.0)
    
    # Get predictions one by one, as ARF is a streaming model
    y_pred = []
    for i in tqdm(range(len(X_clipped_df)), desc="Evaluating ARF on Holdout Set"):
        x_i = X_clipped_df.iloc[i].to_dict()
        pred = model.predict_one(x_i)
        y_pred.append(pred if pred is not None else 1) # Default to 'neutral' if no prediction

    # Calculate and log performance metrics
    f1 = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    log.info("\n" + "="*50)
    log.info("--- ARF HOLDOUT TEST SET PERFORMANCE ---")
    log.info(f"F1-Macro: {f1:.4f}")
    log.info(f"Cohen's Kappa: {kappa:.4f}")
    log.info("Classification Report:")
    log.info(classification_report(y_test, y_pred))
    log.info("="*50 + "\n")

    return {"test_f1_macro": f1, "test_cohen_kappa": kappa, "test_classification_report": report}
