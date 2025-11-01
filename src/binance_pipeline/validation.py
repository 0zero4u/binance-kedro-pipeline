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
    df_out = df.copy()
    numeric_df = df_out.select_dtypes(include=np.number)
    initial_cols_count = len(numeric_df.columns)

    # 1. Remove zero-variance features
    variances = numeric_df.var()
    zero_var_cols = variances[variances < 1e-10].index.tolist()
    if zero_var_cols:
        numeric_df.drop(columns=zero_var_cols, inplace=True)
        log.info(f"Removed {len(zero_var_cols)} zero-variance features.")

    # 2. Remove highly correlated features
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_threshold = params.get("correlation_threshold", 0.90)
    to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    if to_drop_corr:
        numeric_df.drop(columns=to_drop_corr, inplace=True)
        log.info(f"Removed {len(to_drop_corr)} highly correlated features (threshold={corr_threshold}).")

    # 3. Iteratively remove features with high VIF (Variance Inflation Factor)
    vif_threshold = params.get("vif_threshold", 10.0)
    
    # --- START: HIGH-SPEED VIF IMPLEMENTATION ---
    features_for_vif = numeric_df.dropna()
    
    # If the dataframe is large, use a random sample to calculate VIF.
    # This is the key to making the process fast.
    sample_size = 50000 
    if len(features_for_vif) > sample_size:
        log.info(f"Sub-sampling data to {sample_size} rows for faster VIF calculation.")
        sampled_features = features_for_vif.sample(n=sample_size, random_state=42)
    else:
        sampled_features = features_for_vif

    vif_cols = sampled_features.columns.tolist()
    vif_dropped_count = 0
    
    while True:
        # Prevent infinite loops and handle cases with few features
        if len(vif_cols) <= 2:
            break

        # Calculate VIF on the (potentially sampled) data
        vif_values = [variance_inflation_factor(sampled_features[vif_cols].values, i) for i in range(len(vif_cols))]
        vif = pd.Series(vif_values, index=vif_cols)
        
        max_vif = vif.max()
        if max_vif < vif_threshold:
            log.info(f"VIF calculation complete. Max VIF is {max_vif:.2f} (below threshold of {vif_threshold}).")
            break
        
        # Drop the feature with the highest VIF
        drop_col = vif.idxmax()
        vif_cols.remove(drop_col)
        vif_dropped_count += 1
    
    if vif_dropped_count > 0:
        log.info(f"Removed {vif_dropped_count} features due to high VIF (threshold={vif_threshold}).")
    
    # The final set of columns are those remaining in `vif_cols`
    final_numeric_cols = vif_cols
    # --- END: HIGH-SPEED VIF IMPLEMENTATION ---
    
    # Reconstruct the final dataframe with original non-numeric columns + selected numeric ones
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    final_df = df[non_numeric_cols + final_numeric_cols]

    log.info(f"Feature selection complete. {initial_cols_count} -> {len(final_numeric_cols)} numeric features.")
    return final_df


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
