import pandas as pd
import logging
from typing import List

log = logging.getLogger(__name__)

def _get_ordered_columns(df: pd.DataFrame) -> List[str]:
    """Dynamically creates a readable column order for the report."""
    
    core_identifiers = ['datetime']
    ohlc = sorted([c for c in df.columns if c in ['open', 'high', 'low', 'close', 'volume']])
    
    # Use prefixes to identify columns from each stage
    primary_cols = sorted([c for c in df.columns if c.startswith('primary_')])
    multiscale_cols = sorted([c for c in df.columns if c.startswith('multiscale_')])
    final_cols = sorted([c for c in df.columns if c.startswith('final_')])
    
    ordered_cols = core_identifiers + ohlc + primary_cols + multiscale_cols + final_cols
    remaining_cols = sorted([c for c in df.columns if c not in ordered_cols])
    
    return ordered_cols + remaining_cols

def create_unified_traceability_report(
    merged_grid: pd.DataFrame,
    primary_features_grid: pd.DataFrame,
    intelligent_multi_scale_features: pd.DataFrame, # <-- FIX: Accept new dataset
    features_data_unvalidated: pd.DataFrame
) -> pd.DataFrame:
    """
    CORRECTED: Generates an intelligent, human-readable traceability report for the first 5 minutes
    of data by correctly identifying and merging only new columns from each stage.
    """
    log.info("Generating traceability report (first 5 mins) with new pipeline structure...")
    
    start_time = merged_grid['datetime'].min()
    end_time = start_time + pd.Timedelta(minutes=5)
    
    report_df = merged_grid[merged_grid['datetime'] <= end_time].copy()
    
    # Identify new columns from each stage
    primary_cols = set(primary_features_grid.columns) - set(merged_grid.columns)
    multiscale_cols = set(intelligent_multi_scale_features.columns) - set(primary_features_grid.columns)
    final_cols = set(features_data_unvalidated.columns) - set(intelligent_multi_scale_features.columns)

    # Prefix and merge new columns for clarity
    df_primary_new = primary_features_grid[list(primary_cols) + ['datetime']].rename(columns={c: f"primary_{c}" for c in primary_cols})
    df_multiscale_new = intelligent_multi_scale_features[list(multiscale_cols) + ['datetime']].rename(columns={c: f"multiscale_{c}" for c in multiscale_cols})
    df_final_new = features_data_unvalidated[list(final_cols) + ['datetime']].rename(columns={c: f"final_{c}" for c in final_cols})

    report_df = pd.merge(report_df, df_primary_new[df_primary_new['datetime'] <= end_time], on='datetime', how='left')
    report_df = pd.merge(report_df, df_multiscale_new[df_multiscale_new['datetime'] <= end_time], on='datetime', how='left')
    report_df = pd.merge(report_df, df_final_new[df_final_new['datetime'] <= end_time], on='datetime', how='left')
    
    final_ordered_cols = _get_ordered_columns(report_df)
    report_df = report_df[final_ordered_cols]
    
    log.info(f"Unified traceability report created with clean shape: {report_df.shape}")
    return report_df
