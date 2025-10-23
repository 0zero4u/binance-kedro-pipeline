import pandas as pd
import logging
from typing import Dict

log = logging.getLogger(__name__)

def create_unified_traceability_report(
    merged_tick_data: pd.DataFrame,
    resampled_data_100ms: pd.DataFrame,
    features_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates a unified CSV report for the first 5 minutes of data,
    tracing a tick through its bar and feature transformations.
    """
    log.info("Generating unified traceability report for the first 5 minutes of data...")
    
    # Ensure datetime columns exist and are sorted
    for df, name in [
        (merged_tick_data, 'merged_tick_data'), 
        (resampled_data_100ms, 'resampled_data_100ms'), 
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
    bars = resampled_data_100ms[resampled_data_100ms['datetime'] <= end_time].copy()
    feats = features_data[features_data['datetime'] <= end_time].copy()
    
    log.info(f"Data shapes for report (first 5 mins): Ticks={tbt.shape}, Bars={bars.shape}, Features={feats.shape}")
    
    # Add prefixes to avoid column name collisions
    tbt.columns = ["tbt_" + col for col in tbt.columns]
    bars.columns = ["bar_" + col for col in bars.columns]
    feats.columns = ["feat_" + col for col in feats.columns]
    
    # Rename datetime columns for merging
    tbt.rename(columns={'tbt_datetime': 'datetime'}, inplace=True)
    bars.rename(columns={'bar_datetime': 'datetime'}, inplace=True)
    feats.rename(columns={'feat_datetime': 'datetime'}, inplace=True)

    # Merge tick data with its corresponding bar data
    # Each tick will be associated with the bar that starts at or before it
    merged_df = pd.merge_asof(
        left=tbt,
        right=bars,
        on='datetime',
        direction='backward'
    )
    
    # Merge the result with the final features data
    # Each tick will be associated with the final feature set for its bar
    final_df = pd.merge_asof(
        left=merged_df,
        right=feats,
        on='datetime',
        direction='backward'
    )
    
    final_df.dropna(how='any', subset=[c for c in final_df.columns if c.startswith('bar_') or c.startswith('feat_')], inplace=True)
    
    log.info(f"Unified traceability report created with shape: {final_df.shape}")
    
    return final_df
