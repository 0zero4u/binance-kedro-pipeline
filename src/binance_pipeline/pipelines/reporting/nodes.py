import pandas as pd
import logging
from typing import List

log = logging.getLogger(__name__)

def _get_ordered_columns(df: pd.DataFrame) -> List[str]:
    """Dynamically creates a readable column order for the report."""
    
    core_identifiers = ['datetime']
    ohlc = sorted([c for c in df.columns if c in ['open', 'high', 'low', 'close', 'volume']])
    
    primary_cols = sorted([c for c in df.columns if c.startswith('primary_')])
    ewma_cols = sorted([c for c in df.columns if c.startswith('ewma_')])
    final_cols = sorted([c for c in df.columns if c.startswith('final_')])
    
    ordered_cols = core_identifiers + ohlc + primary_cols + ewma_cols + final_cols
    remaining_cols = sorted([c for c in df.columns if c not in ordered_cols])
    
    return ordered_cols + remaining_cols

def create_unified_traceability_report(
    merged_grid: pd.DataFrame,
    primary_features_grid: pd.DataFrame,
    ewma_grid: pd.DataFrame,
    features_data_unvalidated: pd.DataFrame # <-- FIX: Accept the unvalidated data
) -> pd.DataFrame:
    """
    CORRECTED: Generates an intelligent, human-readable traceability report for the first 5 minutes
    of data by correctly identifying and merging only new columns from each stage.
    """
    log.info("Generating CORRECTED traceability report (first 5 mins) using unvalidated features for full history...")
    
    start_time = merged_grid['datetime'].min()
    end_time = start_time + pd.Timedelta(minutes=5)
    
    report_df = merged_grid[merged_grid['datetime'] <= end_time].copy()
    
    primary_cols = set(primary_features_grid.columns) - set(merged_grid.columns)
    ewma_cols = set(ewma_grid.columns) - set(primary_features_grid.columns)
    # Use the unvalidated dataframe here
    final_cols = set(features_data_unvalidated.columns) - set(ewma_grid.columns)

    df_primary_new = primary_features_grid[list(primary_cols) + ['datetime']].rename(columns={c: f"primary_{c}" for c in primary_cols})
    df_ewma_new = ewma_grid[list(ewma_cols) + ['datetime']].rename(columns={c: f"ewma_{c}" for c in ewma_cols})
    # And use the unvalidated dataframe here as well
    df_final_new = features_data_unvalidated[list(final_cols) + ['datetime']].rename(columns={c: f"final_{c}" for c in final_cols})

    report_df = pd.merge(report_df, df_primary_new[df_primary_new['datetime'] <= end_time], on='datetime', how='left')
    report_df = pd.merge(report_df, df_ewma_new[df_ewma_new['datetime'] <= end_time], on='datetime', how='left')
    report_df = pd.merge(report_df, df_final_new[df_final_new['datetime'] <= end_time], on='datetime', how='left')
    
    final_ordered_cols = _get_ordered_columns(report_df)
    report_df = report_df[final_ordered_cols]
    
    log.info(f"Unified traceability report created with clean shape: {report_df.shape}")
    return report_df
