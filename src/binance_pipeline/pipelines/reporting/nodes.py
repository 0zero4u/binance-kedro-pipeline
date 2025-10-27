import pandas as pd
import logging
from typing import List

log = logging.getLogger(__name__)

def _get_ordered_columns(df: pd.DataFrame) -> List[str]:
    """Dynamically creates a readable column order for the report."""
    
    # --- FIX: Simplified ordering logic for clarity ---
    core_identifiers = ['datetime']
    ohlc = sorted([c for c in df.columns if c in ['open', 'high', 'low', 'close', 'volume']])
    
    # Use prefixes to group columns logically
    primary_cols = sorted([c for c in df.columns if c.startswith('primary_')])
    ewma_cols = sorted([c for c in df.columns if c.startswith('ewma_')])
    final_cols = sorted([c for c in df.columns if c.startswith('final_')])
    
    ordered_cols = core_identifiers + ohlc + primary_cols + ewma_cols + final_cols
    remaining_cols = sorted([c for c in df.columns if c not in ordered_cols])
    
    return ordered_cols + remaining_cols

def create_unified_traceability_report(
    merged_grid: pd.DataFrame,
    primary_features_grid: pd.DataFrame, # <-- ADDED this input
    ewma_grid: pd.DataFrame,
    features_data: pd.DataFrame
) -> pd.DataFrame:
    """
    CORRECTED: Generates an intelligent, human-readable traceability report for the first 5 minutes
    of data by correctly identifying and merging only new columns from each stage.
    """
    log.info("Generating CORRECTED traceability report (first 5 mins)...")
    
    # --- Part 1: Prepare and filter data ---
    start_time = merged_grid['datetime'].min()
    end_time = start_time + pd.Timedelta(minutes=5)
    
    # Base dataframe
    report_df = merged_grid[merged_grid['datetime'] <= end_time].copy()
    
    # --- Part 2: Identify NEW columns from each stage ---
    primary_cols = set(primary_features_grid.columns) - set(merged_grid.columns)
    ewma_cols = set(ewma_grid.columns) - set(primary_features_grid.columns)
    final_cols = set(features_data.columns) - set(ewma_grid.columns)

    # --- Part 3: Create prefixed dataframes with ONLY new columns ---
    df_primary_new = primary_features_grid[list(primary_cols) + ['datetime']].rename(columns={c: f"primary_{c}" for c in primary_cols})
    df_ewma_new = ewma_grid[list(ewma_cols) + ['datetime']].rename(columns={c: f"ewma_{c}" for c in ewma_cols})
    df_final_new = features_data[list(final_cols) + ['datetime']].rename(columns={c: f"final_{c}" for c in final_cols})

    # --- Part 4: Sequentially merge the NEW, prefixed columns ---
    report_df = pd.merge(report_df, df_primary_new[df_primary_new['datetime'] <= end_time], on='datetime', how='left')
    report_df = pd.merge(report_df, df_ewma_new[df_ewma_new['datetime'] <= end_time], on='datetime', how='left')
    report_df = pd.merge(report_df, df_final_new[df_final_new['datetime'] <= end_time], on='datetime', how='left')
    
    # --- Part 5: Final Formatting ---
    final_ordered_cols = _get_ordered_columns(report_df)
    report_df = report_df[final_ordered_cols]
    
    log.info(f"Unified traceability report created with clean shape: {report_df.shape}")
    return report_df
