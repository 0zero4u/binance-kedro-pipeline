import pandas as pd
import logging
from typing import List

log = logging.getLogger(__name__)

def _get_ordered_columns(df: pd.DataFrame) -> List[str]:
    """Dynamically creates a readable column order for the report."""
    
    core_identifiers = ['datetime']
    ohlc = sorted([c for c in df.columns if c in ['open', 'high', 'low', 'close', 'volume']])
    book_state = sorted([c for c in df.columns if c in [
        'best_bid_price', 'best_ask_price', 'best_bid_qty', 'best_ask_qty'
    ]])
    primary_features = sorted([c for c in df.columns if c.startswith('primary_')])
    ewma_features = sorted([c for c in df.columns if c.startswith('ewma_')])
    final_features = sorted([c for c in df.columns if c.startswith('final_')])
    
    ordered_cols = core_identifiers + ohlc + book_state + primary_features + ewma_features + final_features
    remaining_cols = sorted([c for c in df.columns if c not in ordered_cols])
    
    return ordered_cols + remaining_cols

def create_unified_traceability_report(
    merged_grid: pd.DataFrame,
    ewma_grid: pd.DataFrame,
    features_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates an intelligent, human-readable traceability report for the first 5 minutes of data,
    adapted for the new "Grid-First" architecture.
    """
    log.info("Generating intelligent traceability report (first 5 mins) for new grid architecture...")
    
    # --- Part 1: Prepare and filter data ---
    start_time = merged_grid['datetime'].min()
    end_time = start_time + pd.Timedelta(minutes=5)
    
    grid = merged_grid[merged_grid['datetime'] <= end_time].copy()
    ewma = ewma_grid[ewma_grid['datetime'] <= end_time].copy()
    feats = features_data[features_data['datetime'] <= end_time].copy()
    
    log.info(f"Data shapes for report: Grid={grid.shape}, EWMA={ewma.shape}, Features={feats.shape}")
    
    # --- Part 2: Intelligent Prefixing ---
    grid_cols = set(grid.columns)
    ewma_cols = set(ewma.columns)
    feat_cols = set(feats.columns)
    
    new_ewma_cols = ewma_cols - grid_cols
    new_feat_cols = feat_cols - ewma_cols
    
    ewma_rename_dict = {col: f"ewma_{col}" for col in new_ewma_cols}
    feat_rename_dict = {col: f"final_{col}" for col in new_feat_cols}
    
    ewma.rename(columns=ewma_rename_dict, inplace=True)
    feats.rename(columns=feat_rename_dict, inplace=True)

    # --- Part 3: Merge and Format ---
    merged_df = pd.merge(grid, ewma, on='datetime', how='left')
    final_df = pd.merge(merged_df, feats, on='datetime', how='left')
    
    final_ordered_cols = _get_ordered_columns(final_df)
    final_df = final_df[final_ordered_cols]
    
    log.info(f"Unified traceability report created with shape: {final_df.shape}")
    return final_df
