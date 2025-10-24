import pandas as pd
import logging
from typing import Dict

log = logging.getLogger(__name__)

def create_unified_traceability_report(
    merged_tick_data: pd.DataFrame,
    ewma_features_tbt: pd.DataFrame,
    features_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates a unified CSV report for the first 5 minutes of data,
    tracing a tick through its feature transformations (TBT EWMA and 25ms Grid).
    """
    log.info("Generating unified traceability report for the first 5 minutes of data...")
    
    # Ensure datetime columns exist and are sorted
    for df, name in [
        (merged_tick_data, 'merged_tick_data'), 
        (ewma_features_tbt, 'ewma_features_tbt'), 
        (features_data, 'features_data')
    ]:
        if 'datetime' not in df.columns:
            if 'timestamp' in df.columns:
                 df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                raise ValueError(f"Missing 'datetime' or 'timestamp' in {name}")
        df.sort_values('datetime', inplace=True)

    # Filter to first 5 minutes
    start_time = merged_tick_data['datetime'].min()
    end_time = start_time + pd.Timedelta(minutes=5)
    
    tbt = merged_tick_data[merged_tick_data['datetime'] <= end_time].copy()
    ewma = ewma_features_tbt[ewma_features_tbt['datetime'] <= end_time].copy()
    feats = features_data[features_data['datetime'] <= end_time].copy()
    
    log.info(f"Data shapes for report (first 5 mins): Ticks={tbt.shape}, EWMA TBT={ewma.shape}, Features={feats.shape}")
    
    # Add prefixes to avoid column name collisions
    tbt.columns = ["tbt_" + col for col in tbt.columns]
    ewma.columns = ["ewma_" + col for col in ewma.columns]
    feats.columns = ["feat_" + col for col in feats.columns]
    
    # Rename datetime columns for merging
    tbt.rename(columns={'tbt_datetime': 'datetime'}, inplace=True)
    ewma.rename(columns={'ewma_datetime': 'datetime'}, inplace=True)
    feats.rename(columns={'feat_datetime': 'datetime'}, inplace=True)

    # Merge tick data with its corresponding EWMA TBT feature set
    merged_df = pd.merge_asof(
        left=tbt,
        right=ewma,
        on='datetime',
        direction='backward'
    )
    
    # Merge the result with the final features data (25ms grid)
    final_df = pd.merge_asof(
        left=merged_df,
        right=feats,
        on='datetime',
        direction='backward'
    )
    
    # --- FIX FOR EMPTY REPORT: Use a more targeted dropna ---
    # The old dropna was too aggressive and deleted rows during feature warm-up.
    # This new version only drops rows if a key, fast-moving feature is null.
    final_df.dropna(how='any', subset=['ewma_mid_price_ewma_5s', 'feat_mid_price'], inplace=True)
    
    log.info(f"Unified traceability report created with shape: {final_df.shape}")
    
    return final_df
