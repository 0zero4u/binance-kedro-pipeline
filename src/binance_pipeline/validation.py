import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
import logging

log = logging.getLogger(__name__)

class EnrichedTickSchema(pa.SchemaModel):
    """Schema for the structurally validated enriched tick data."""

    timestamp: Series[int] = pa.Field(nullable=False)

    price: Series[float] = pa.Field(nullable=False)
    best_bid_price: Series[float] = pa.Field(nullable=False)
    best_ask_price: Series[float] = pa.Field(nullable=False)
    microprice: Series[float] = pa.Field(nullable=False)
    ofi: Series[float] = pa.Field(nullable=False)
    book_imbalance: Series[float] = pa.Field(nullable=False)
    spread: Series[float] = pa.Field(nullable=False)

    # Column-level checks as SchemaModel validators:
    
    @pa.check("timestamp")
    def timestamp_monotonic(cls, series: Series[int]) -> Series[bool]:
        # This allows for duplicate values, as long as they are in order.
        return series.is_monotonic_increasing
    
    @pa.check("price", "best_bid_price", "best_ask_price")
    def must_be_positive(cls, series: Series[float]) -> Series[bool]:
        return series > 0

    @pa.check("book_imbalance")
    def balance_in_range(cls, series: Series[float]) -> Series[bool]:
        return (series >= -1.0) & (series <= 1.0)

    @pa.check("spread")
    def spread_non_negative(cls, series: Series[float]) -> Series[bool]:
        return series >= 0

    @pa.dataframe_check
    def check_ask_greater_than_bid(cls, df: DataFrame) -> Series[bool]:
        return df["best_ask_price"] > df["best_bid_price"]

    class Config:
        coerce = True


def validate_enriched_tick_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the structural guardrail to the enriched tick data.
    Halts the pipeline if validation fails.
    """
    log.info(f"Applying STRUCTURAL guardrail to enriched_tick_data (shape: {df.shape})...")
    if df.empty:
        log.warning("Input to 'validate_enriched_tick_data' is an empty DataFrame. Validation will pass, but no data will be processed downstream.")
        return df

    try:
        EnrichedTickSchema.validate(df, lazy=True)
        log.info("✅ Structural guardrail PASSED for enriched_tick_data.")
        return df
    except pa.errors.SchemaErrors as err:
        log.error("🔥 Structural guardrail FAILED for enriched_tick_data!")
        log.error("Validation failure details below:")
        log.error(err.failure_cases)
        raise err

def validate_features_data_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a robust, two-stage cleaning process and logical guardrail to the final features data.
    """
    log.info(f"Applying LOGICAL guardrail to features_data (shape: {df.shape})...")
    
    # --- FIX: Implement a robust two-stage cleaning process ---
    
    # Stage 1: Drop initial warm-up period based on a RELIABLE long-window feature.
    log.info("  - Stage 1: Dropping initial warm-up period...")
    initial_rows = len(df)
    reliable_warmup_feature = 'taker_flow_rollsum_60s'
    if reliable_warmup_feature in df.columns:
        df.dropna(subset=[reliable_warmup_feature], inplace=True)
        log.info(f"  - Dropped {initial_rows - len(df)} initial rows. New shape: {df.shape}")
    else:
        log.warning(f"  - Reliable feature '{reliable_warmup_feature}' not found. Skipping initial drop.")
        
    # Stage 2: Forward-fill to handle intermittent NaNs from sensitive features.
    log.info("  - Stage 2: Forward-filling intermittent NaNs from sensitive features...")
    # Replace any infinite values that may have been created
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    nans_before = df.isna().sum().sum()
    df.ffill(inplace=True)
    nans_after = df.isna().sum().sum()
    log.info(f"  - Filled {nans_before - nans_after} NaN values.")

    # A final dropna for any NaNs that might remain at the very start after ffill
    df.dropna(inplace=True)
    log.info(f"  - Final clean shape: {df.shape}")

    if df.empty:
        log.warning("Input to 'validate_features_data_logic' is an empty DataFrame AFTER cleaning. Skipping logical checks.")
        return df

    try:
        # Logical Check: Correlation between returns and order flow.
        cvd_feature = 'taker_flow_rollsum_60s'
        correlation = df['returns'].corr(df[cvd_feature])
        log.info(f"Correlation(returns, {cvd_feature}) = {correlation:.4f}")
        
        assert correlation > 0.001, f"Logical check failed: Correlation between returns and CVD is not positive ({correlation:.4f}). This indicates a potential bug in feature logic."

        log.info("✅ Logical guardrail PASSED for features_data.")
        return df
    except Exception as e:
        log.error("🔥 Logical guardrail FAILED for features_data!")
        raise e
