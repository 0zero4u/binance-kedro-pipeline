import pandas as pd
import logging
from typing import List

log = logging.getLogger(__name__)

def _get_ordered_columns(df: pd.DataFrame) -> List[str]:
    """Dynamically creates a readable column order for the report."""
    
    # 1. Define the desired order of column groups
    core_identifiers = ['datetime', 'timestamp']
    raw_trade_book = sorted([c for c in df.columns if c in [
        'price', 'qty', 'is_buyer_maker', 'best_bid_price', 'best_ask_price', 
        'best_bid_qty', 'best_ask_qty'
    ]])
    derived_tick = sorted([c for c in df.columns if c in [
        'mid_price', 'spread', 'spread_bps', 'microprice', 'ofi', 'book_imbalance', 'taker_flow'
    ]])
    ewma_features = sorted([c for c in df.columns if c.startswith('ewma_')])
    feat_features = sorted([c for c in df.columns if c.startswith('feat_')])
    
    # 2. Combine all known columns
    ordered_cols = core_identifiers + raw_trade_book + derived_tick + ewma_features + feat_features
    
    # 3. Add any remaining columns (e.g., intermediate columns) to the end
    remaining_cols = sorted([c for c in df.columns if c not in ordered_cols])
    
    return ordered_cols + remaining_cols

def create_unified_traceability_report(
    merged_tick_data: pd.DataFrame,
    ewma_features_tbt: pd.DataFrame,
    features_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates an intelligent, human-readable traceability report for the first 5 minutes of data.
    This version eliminates redundancy by only prefixing newly generated columns at each stage.
    """
    log.info("Generating intelligent traceability report (first 5 mins)...")
    
    # --- Part 1: Prepare and filter data ---
    
    # Ensure datetime columns exist and are sorted
    for df, name in [
        (merged_tick_data, 'merged_tick_data'), 
        (ewma_features_tbt, 'ewma_features_tbt'), 
        (features_data, 'features_data')
    ]:
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df.get('timestamp', pd.NaT), unit='ms')
        df.sort_values('datetime', inplace=True)

    # Filter to first 5 minutes
    start_time = merged_tick_data['datetime'].min()
    end_time = start_time + pd.Timedelta(minutes=5)
    
    tbt = merged_tick_data[merged_tick_data['datetime'] <= end_time].copy()
    ewma = ewma_features_tbt[ewma_features_tbt['datetime'] <= end_time].copy()
    feats = features_data[features_data['datetime'] <= end_time].copy()
    
    log.info(f"Data shapes for report (first 5 mins): Ticks={tbt.shape}, EWMA TBT={ewma.shape}, Features={feats.shape}")
    
    # --- Part 2: Intelligent Prefixing to Avoid Redundancy ---

    # Identify which columns are NEW at each stage
    tbt_cols = set(tbt.columns)
    ewma_cols = set(ewma.columns)
    feat_cols = set(feats.columns)
    
    # EWMA features are those in `ewma` but not in the original `tbt`
    new_ewma_cols = ewma_cols - tbt_cols
    # Final features are those in `feats` but not in the `ewma` intermediate set
    new_feat_cols = feat_cols - ewma_cols
    
    # Create rename dictionaries to prefix ONLY the new columns
    ewma_rename_dict = {col: f"ewma_{col}" for col in new_ewma_cols}
    feat_rename_dict = {col: f"feat_{col}" for col in new_feat_cols}
    
    ewma.rename(columns=ewma_rename_dict, inplace=True)
    feats.rename(columns=feat_rename_dict, inplace=True)
    
    log.info(f"Identified and prefixed {len(new_ewma_cols)} new EWMA features.")
    log.info(f"Identified and prefixed {len(new_feat_cols)} new final grid features.")

    # --- Part 3: Merge and Format ---

    # Merge tick data with its corresponding EWMA TBT feature set
    # Common columns like 'price', 'is_buyer_maker' will merge cleanly without duplication
    merged_df = pd.merge_asof(
        left=tbt,
        right=ewma,
        on='datetime',
        direction='backward',
        suffixes=('', '_ewma_dup') # Handle any unexpected duplicates
    )
    
    # Merge the result with the final features data (25ms grid)
    final_df = pd.merge_asof(
        left=merged_df,
        right=feats,
        on='datetime',
        direction='backward',
        suffixes=('', '_feat_dup') # Handle any unexpected duplicates
    )
    
    # Clean up and reorder for readability
    final_df.dropna(how='any', subset=[c for c in final_df.columns if c.startswith('ewma_') or c.startswith('feat_')], inplace=True)
    
    # Reorder columns into a logical, human-readable format
    final_ordered_cols = _get_ordered_columns(final_df)
    final_df = final_df[final_ordered_cols]
    
    log.info(f"Unified traceability report created with shape: {final_df.shape}")
    log.info(f"Final report columns (first 5): {list(final_df.columns[:5])}")
    
    return final_df
