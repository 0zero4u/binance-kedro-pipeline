import pandas as pd
import numpy as np
import logging
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict

log = logging.getLogger(__name__)


def select_and_validate_features(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Applies an advanced feature selection process to eliminate redundant and
    unstable features before model training.
    """
    log.info(f"Starting advanced feature selection on {df.shape[1]} features...")
    df_out = df.copy().select_dtypes(include=np.number)
    initial_cols = set(df_out.columns)

    # 1. Remove zero-variance features
    variances = df_out.var()
    zero_var_cols = variances[variances < 1e-10].index.tolist()
    df_out.drop(columns=zero_var_cols, inplace=True)
    if zero_var_cols:
        log.info(f"Removed {len(zero_var_cols)} zero-variance features.")

    # 2. Remove highly correlated features
    corr_matrix = df_out.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_threshold = params.get("correlation_threshold", 0.90)
    to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    df_out.drop(columns=to_drop_corr, inplace=True)
    if to_drop_corr:
        log.info(f"Removed {len(to_drop_corr)} highly correlated features (threshold={corr_threshold}).")

    # 3. Iteratively remove features with high VIF (Variance Inflation Factor)
    vif_threshold = params.get("vif_threshold", 10.0)
    features = df_out.dropna()._get_numeric_data()
    
    if 'const' not in features.columns:
        features['const'] = 1 

    vif_cols = [col for col in features.columns if col != 'const']
    vif_dropped_count = 0
    
    while True:
        vif = pd.Series(
            [variance_inflation_factor(features[vif_cols].values, i) for i in range(len(vif_cols))],
            index=vif_cols
        )
        if vif.max() < vif_threshold or len(vif_cols) <= 2:
            break
        
        drop_col = vif.idxmax()
        vif_cols.remove(drop_col)
        vif_dropped_count += 1
    
    df_out = df[list(set(df.columns) - (initial_cols - set(vif_cols)))]
    if vif_dropped_count > 0:
        log.info(f"Removed {vif_dropped_count} features due to high VIF (threshold={vif_threshold}).")
    
    final_cols = set(df_out.columns)
    log.info(f"Feature selection complete. {len(initial_cols)} -> {len(final_cols)} features.")
    return df_out


def validate_features_data_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies logical guardrails to the final features data. This includes dropping
    the initial warm-up period and asserting a positive correlation between order
    flow and returns to catch potential data processing bugs.
    """
    log.info(f"Applying LOGICAL guardrail to features_data (shape: {df.shape})...")
    
    key_feature = 'taker_flow_rollsum_medium'
    if key_feature not in df.columns:
        log.warning(f"Key feature '{key_feature}' not in DataFrame for validation. Skipping correlation check.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        core_features = [c for c in ['close', 'mid_price', 'taker_flow', 'returns'] if c in df.columns]
        df.dropna(subset=core_features, inplace=True)
        return df

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    initial_rows = len(df)
    df.dropna(subset=[key_feature], inplace=True)
    log.info(f"Dropped {initial_rows - len(df)} warm-up rows. Final shape: {df.shape}")

    if df.empty:
        log.warning("DataFrame is empty AFTER dropping NaNs. Logical checks will be skipped.")
        return df

    try:
        correlation = df[key_feature].corr(df['returns'])
        log.info(f"Correlation(returns, {key_feature}) = {correlation:.4f}")
        assert correlation > 0, f"Logical check failed: Correlation between returns and order flow should be positive but was {correlation:.4f}."
        log.info("✅ Logical guardrail PASSED for features_data.")
        return df
    except Exception as e:
        log.error("🔥 Logical guardrail FAILED for features_data!")
        raise e
