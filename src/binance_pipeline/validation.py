import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
import logging

log = logging.getLogger(__name__)

# ==================================
# 1. Structural Guardrail Schema (Final )
# ==================================

class EnrichedTickSchema(pa.SchemaModel):
    """Schema for the structurally validated enriched tick data."""

    # --- START OF FIX ---
    # Corrected the keyword argument from `checks` (plural) to `check` (singular),
    # which is the correct API for pandera.Field.
    timestamp: Series[int] = pa.Field(
        nullable=False, 
        unique=True,
        check=pa.Check(lambda s: s.is_monotonic_increasing, name="monotonic_increasing")
    )
    # --- END OF FIX ---

    price: Series[float] = pa.Field(nullable=False, ge=0)
    best_bid_price: Series[float] = pa.Field(nullable=False, ge=0)
    best_ask_price: Series[float] = pa.Field(nullable=False, ge=0)

    microprice: Series[float] = pa.Field(
        nullable=False, 
        description="Microprice should not have any missing values after ffill."
    )

    ofi: Series[float] = pa.Field(
        nullable=False, 
        description="Order Flow Imbalance should not have missing values after fillna(0)."
    )

    book_imbalance: Series[float] = pa.Field(
        nullable=False, 
        in_range={"min_value": -1.0, "max_value": 1.0}
    )

    spread: Series[float] = pa.Field(nullable=False, ge=0)

    @pa.dataframe_check
    def check_ask_greater_than_bid(cls, df: DataFrame) -> Series[bool]:
        """Ensures that the best ask price is always greater than the best bid price."""
        return df["best_ask_price"] > df["best_bid_price"]

    class Config:
        coerce = True
        strict = "filter"

# ==================================
# 2. Validator Nodes 
# ==================================

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
        validated_df = EnrichedTickSchema.validate(df, lazy=True)
        log.info("✅ Structural guardrail PASSED for enriched_tick_data.")
        return validated_df
    except pa.errors.SchemaErrors as err:
        log.error("🔥 Structural guardrail FAILED for enriched_tick_data!")
        log.error("Validation failure details below:")
        log.error(err.failure_cases.to_string())
        raise err

def validate_features_data_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the logical guardrail to the final features data.
    Checks for plausible market dynamics.
    """
    log.info(f"Applying LOGICAL guardrail to features_data (shape: {df.shape})...")
    if df.empty:
        log.warning("Input to 'validate_features_data_logic' is an empty DataFrame. Skipping logical checks.")
        return df

    try:
        required_cols = ['returns', 'cvd_taker_50']
        if not all(col in df.columns for col in required_cols):
             log.warning(f"Skipping logical check: Missing one of {required_cols} in the dataframe.")
             return df

        correlation = df['returns'].corr(df['cvd_taker_50'])
        log.info(f"Correlation(returns, cvd_taker_50) = {correlation:.4f}")
        
        assert correlation > 0.001, f"Logical check failed: Correlation between returns and CVD is not positive ({correlation:.4f}). This indicates a potential bug in feature logic."

        log.info("✅ Logical guardrail PASSED for features_data.")
        return df
    except Exception as e:
        log.error("🔥 Logical guardrail FAILED for features_data!")
        raise e
