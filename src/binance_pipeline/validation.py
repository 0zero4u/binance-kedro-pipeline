import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

def validate_features_data_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies logical guardrails to the final features data. This includes dropping
    the initial warm-up period and asserting a positive correlation between order
    flow and returns to catch potential data processing bugs.
    """
    log.info(f"Applying LOGICAL guardrail to features_data (shape: {df.shape})...")
    
    key_feature = 'taker_flow_rollsum_60s'
    if key_feature not in df.columns:
        log.error(f"FATAL: Key feature '{key_feature}' not in DataFrame. Cannot proceed.")
        raise ValueError(f"Missing key feature for cleaning: {key_feature}")

    # Step 1: Replace any infinite values that may have been created
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Step 3: Use a single reliable long-window feature to drop the initial warm-up period.
    # This removes rows where long-term features could not be calculated.
    initial_rows = len(df)
    df.dropna(subset=[key_feature], inplace=True)
    log.info(f"Dropped {initial_rows - len(df)} warm-up rows. Final shape: {df.shape}")

    if df.empty:
        log.warning("DataFrame is empty AFTER dropping NaNs. Logical checks will be skipped.")
        return df

    try:
        log.debug("--- Logical Guardrail Diagnostics ---")
        log.debug(f"'returns' stats:\n{df['returns'].describe().to_string()}")
        log.debug(f"'{key_feature}' stats:\n{df[key_feature].describe().to_string()}")

        # Logical Check: Correlation between returns and order flow.
        correlation = df[key_feature].corr(df['returns'])
        log.info(f"Correlation(returns, {key_feature}) = {correlation:.4f}")
        
        # A negative correlation would indicate a bug; near-zero is acceptable for noisy data.
        assert correlation > 0, f"Logical check failed: Correlation between returns and CVD should be positive but was {correlation:.4f}. This indicates a potential bug in feature logic."

        log.info("✅ Logical guardrail PASSED for features_data.")
        return df
    except Exception as e:
        log.error("🔥 Logical guardrail FAILED for features_data!")
        raise e
