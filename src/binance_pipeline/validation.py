
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
    Applies the logical guardrail to the final features data.
    This node is now responsible for the final cleaning of the features data.
    """
    log.info(f"Applying LOGICAL guardrail to features_data (shape: {df.shape})...")
    
    # --- NEW: Final cleaning step to handle all feature warm-up periods ---
    log.info("  - Performing final dropna to remove rows with NaNs from all feature warm-ups...")
    initial_rows = len(df)
    
    # Define features that have the longest warm-up periods from each node
    longest_warmup_features = ['hurst_100', 'kyle_lambda_50', 'vpin_50', 'rsi_28', 'taker_flow_rollsum_60s']
    
    # Find which of these are actually in the dataframe to avoid errors
    valid_cols_to_check = [col for col in longest_warmup_features if col in df.columns]
    
    if valid_cols_to_check:
        log.info(f"  - Cleaning NaNs based on key features: {valid_cols_to_check}")
        df.dropna(subset=valid_cols_to_check, inplace=True)
        log.info(f"  - Dropped {initial_rows - len(df)} rows. New shape: {df.shape}")
    else:
        log.warning("  - No long-warmup features found to drop NaNs on. Skipping.")

    if df.empty:
        log.warning("Input to 'validate_features_data_logic' is an empty DataFrame AFTER cleaning. Skipping logical checks.")
        return df

    try:
        # Use a stable, Wall Clock CVD feature for the correlation check
        cvd_feature = 'taker_flow_rollsum_60s'
        correlation = df['returns'].corr(df[cvv_feature])
        log.info(f"Correlation(returns, {cvd_feature}) = {correlation:.4f}")
        
        assert correlation > 0.001, f"Logical check failed: Correlation between returns and CVD is not positive ({correlation:.4f}). This indicates a potential bug in feature logic."

        log.info("✅ Logical guardrail PASSED for features_data.")
        return df
    except Exception as e:
        log.error("🔥 Logical guardrail FAILED for features_data!")
        raise e
