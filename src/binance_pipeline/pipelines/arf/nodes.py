import pandas as pd
import numpy as np
from river import forest, metrics
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

# Re-use the high-performance labeling from the LGBM pipeline
from binance_pipeline.pipelines.lgbm.nodes import generate_triple_barrier_labels

def generate_arf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates percentile rank features for the ARF model."""
    log.info("Generating ARF features with multi-window percentile ranks...")
    
    PERCENTILE_WINDOWS = {'5s': 50, '10s': 100, '30s': 300, '1min': 600, '3min': 1800, '5min': 3000}
    
    features_to_rank = ['cvd_taker_50', 'vol_regime_20', 'ofi_50', 'vpin_proxy_50', 'momentum_20', 'rsi_14']
    
    for feature in features_to_rank:
        if feature in df.columns:
            for name, window_size in PERCENTILE_WINDOWS.items():
                df[f'{feature}_pct_rank_{name}'] = df[feature].rolling(
                    window=window_size, min_periods=int(window_size / 4)
                ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    log.info(f"ARF feature generation complete. Final shape: {df.shape}")
    return df.reset_index(drop=True)

def fit_robust_scaler(df: pd.DataFrame) -> RobustScaler:
    """Fits a RobustScaler, which is better for data with outliers."""
    log.info("Fitting RobustScaler...")
    feature_cols = [col for col in df.columns if col not in ['datetime', 'label', 'label_raw', 'open', 'high', 'low', 'close']]
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
    
    # Soft clipping to bound features
    X_clipped_df = 3.0 * np.tanh(X_scaled_df / 3.0)
    
    return pd.concat([non_feature_cols, X_clipped_df], axis=1)

def train_arf_ensemble(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Trains an ensemble of ARF models and selects the best one."""
    log.info("Starting ARF ensemble training with progressive validation...")
    
    features = [col for col in df.columns if col not in ['datetime', 'label']]
    X, y = df[features], df['label']
    
    # Dynamically create models from parameters
    models = {
        name: forest.ARFClassifier(**model_params)
        for name, model_params in params["models"].items()
    }
    metrics_dict = {name: metrics.Accuracy() + metrics.MacroF1() for name in models.keys()}
    
    for i in tqdm(range(len(X)), desc="Training ARF Ensemble"):
        x_i, y_i = X.iloc[i].to_dict(), y.iloc[i]
        for name, model in models.items():
            y_pred = model.predict_one(x_i)
            if y_pred is not None:
                metrics_dict[name].update(y_true=y_i, y_pred=y_pred)
            model.learn_one(x=x_i, y=y_i)
            
    log.info("\n=== ARF Training Complete ===")
    for name, metric in metrics_dict.items():
        log.info(f"{name}: {metric}")
        
    best_model_name = max(metrics_dict, key=lambda name: metrics_dict[name].get('MacroF1').get())
    log.info(f"\nBest model selected: {best_model_name}")
    
    return {'best_model': models[best_model_name], 'metrics': str(metrics_dict)}

def select_best_arf_model(results: Dict[str, Any]):
    """Extracts the best model from the training results dictionary."""
    return results['best_model']
