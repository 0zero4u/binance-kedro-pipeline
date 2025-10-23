import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
import logging
from typing import Annotated

log = logging.getLogger(__name__)

# ==================================
# 1. Structural Guardrail Schema
# ==================================

class EnrichedTickSchema(pa.SchemaModel):
    """Schema for the structurally validated enriched tick data."""

    timestamp: Series[int] = pa.Field(
        nullable=False,
        unique=True,
        checks=pa.Check(lambda s: s.is_monotonic_increasing, name="monotonic_increasing")
    )

    price: Series[float] = pa.Field(nullable=False, gt=0)
    best_bid_price: Series[float] = pa.Field(nullable=False, gt=0)
    best_ask_price: Series[float] = pa.Field(nullable=False, gt=0)

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
        """Ensures best ask > best bid at all times."""
        return df["best_ask_price"] > df["best_bid_price"]

    class Config:
        coerce = True


# ==================================
# 2. Validator Nodes
# ==================================

def validate_enriched_tick_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the structural guardrail to enriched tick data.
    Halts the pipeline if validation fails.
    """
    log.info(f"Applying STRUCTURAL guardrail to enriched_tick_data (shape: {df.shape})...")

    if df.empty:
        log.warning("Input DataFrame is empty — skipping structural validation.")
        return df

    try:
        EnrichedTickSchema.validate(df, lazy=True)
        log.info("✅ Structural guardrail PASSED for enriched_tick_data.")
        return df

    except pa.errors.SchemaErrors as err:
        log.error("🔥 Structural guardrail FAILED for enriched_tick_data!")
        log.error(err.failure_cases)
        raise


def validate_features_data_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the logical guardrail to the final feature set.
    Checks plausible market dynamics (returns vs CVD correlation).
    """
    log.info(f"Applying LOGICAL guardrail to features_data (shape: {df.shape})...")

    if df.empty:
        log.warning("Input DataFrame is empty — skipping logical validation.")
        return df

    correlation = df['returns'].corr(df['cvd_taker_50'])
    log.info(f"Correlation(returns, cvd_taker_50) = {correlation:.4f}")

    if correlation <= 0.001:
        raise AssertionError(
            f"Logical check failed: Correlation between returns and CVD "
            f"must be positive ({correlation:.4f}). Potential feature logic failure."
        )

    log.info("✅ Logical guardrail PASSED for features_data.")
    return df


# ==================================
# 3. Return Pipeline
# ==================================

def create_pipeline(
    initial_merge_node,
    tick_feature_node,
    structural_guardrail_node,
    resample_and_feature_nodes,
    merge_node,
    logical_guardrail_node,
):
    """Returns the final Kedro pipeline with guardrails included."""
    from kedro.pipeline import Pipeline

    return Pipeline([
        initial_merge_node,
        tick_feature_node,
        structural_guardrail_node,
        *resample_and_feature_nodes,
        merge_node,
        logical_guardrail_node,
    ])
