import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
import logging
from typing import Annotated # <-- IMPORTANT IMPORT

log = logging.getLogger(__name__)

# ==================================
# 1. Structural Guardrail Schema (Correct Modern Syntax)
# ==================================

class EnrichedTickSchema(pa.SchemaModel):
    """Schema for the structurally validated enriched tick data."""

    # This is the correct syntax: use typing.Annotated to link checks to the type.
    # pa.Field is now only used for metadata like nullable and unique.

    timestamp: Annotated[
        Series[int],
        pa.Field(nullable=False, unique=True),
        pa.Check(lambda s: s.is_monotonic_increasing, name="monotonic_increasing")
    ]

    price: Annotated[Series[float], pa.Field(nullable=False), pa.Check.gt(0)]
    best_bid_price: Annotated[Series[float], pa.Field(nullable=False), pa.Check.gt(0)]
    best_ask_price: Annotated[Series[float], pa.Field(nullable=False), pa.Check.gt(0)]

    microprice: Annotated[
        Series[float], pa.Field(nullable=False, description="Microprice should not have missing values after ffill.")
    ]

    ofi: Annotated[
        Series[float], pa.Field(nullable=False, description="Order Flow Imbalance should not have missing values after fillna(0).")
    ]

    book_imbalance: Annotated[
        Series[float], pa.Field(nullable=False), pa.Check.in_range(-1.0, 1.0)
    ]

    spread: Annotated[
        Series[float], pa.Field(nullable=False), pa.Check.ge(0)
    ]

    @pa.dataframe_check
    def check_ask_greater_than_bid(cls, df: DataFrame) -> Series[bool]:
        """Ensures that the best ask price is always greater than the best bid price."""
        return df["best_ask_price"] > df["best_bid_price"]

    class Config:
        # This tells pandera to process the Annotated types correctly
        coerce = True

# ==================================
# 2. Validator Nodes (Unchanged)
# ==================================

def validate_enriched_tick_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the structural guardrail to the enriched tick data.
    Halts the pipeline if validation fails.
    """
    log.info(f"Applying STRUCTURAL guardrail to enriched_tick_data (shape: {df.shape})...")
    try:
        EnrichedTickSchema.validate(df, lazy=True)
        log.info("✅ Structural guardrail PASSED for enriched_tick_data.")
        return df
    except pa.errors.SchemaErrors as err:
        log.error("🔥 Structural guardrail FAILED for enriched_tick_data!")
        log.error("Validation failure details below:")
        log.error(err.failure_cases)
        raise err # Halt the pipeline

def validate_features_data_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the logical guardrail to the final features data.
    Checks for plausible market dynamics.
    """
    log.info(f"Applying LOGICAL guardrail to features_data (shape: {df.shape})...")
    try:
        # Check 1: Correlation between order flow and returns should be positive
        correlation = df['returns'].corr(df['cvd_taker_50'])
        log.info(f"Correlation(returns, cvd_taker_50) = {correlation:.4f}")

        # A very weak positive correlation is expected. If it's negative, something is likely wrong.
        assert correlation > 0.001, f"Logical check failed: Correlation between returns and CVD is not positive ({correlation:.4f}). This indicates a potential bug in feature logic."

        log.info("✅ Logical guardrail PASSED for features_data.")
        return df
    except Exception as e:
        log.error("🔥 Logical guardrail FAILED for features_data!")
        raise e # Halt the pipeline
