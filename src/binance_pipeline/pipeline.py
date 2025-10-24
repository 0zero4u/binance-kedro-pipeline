from kedro.pipeline import Pipeline, node
from functools import partial
from .nodes import (
    merge_book_trade_asof,
    calculate_tick_level_features,
    calculate_ewma_features, 
    sample_features_to_grid, 
    generate_bar_features, 
)
from .validation import validate_enriched_tick_data, validate_features_data_logic


def create_pipeline(**kwargs) -> Pipeline:
    
    # 1. Initial merge (Trade + Book)
    initial_merge_node = node(
        func=merge_book_trade_asof,
        inputs=["book_raw", "trade_raw"],
        outputs="merged_tick_data",
        name="merge_ticks_asof_node",
    )

    # 2. Enrich tick data (Microprice, OFI, Book Imbalance)
    tick_feature_node = node(
        func=calculate_tick_level_features,
        inputs="merged_tick_data",
        outputs="enriched_tick_data_unvalidated",
        name="calculate_tick_features_node",
    )

    # 3. Validate tick structure
    structural_guardrail_node = node(
        func=validate_enriched_tick_data,
        inputs="enriched_tick_data_unvalidated",
        outputs="enriched_tick_data",
        name="structural_guardrail_node",
    )

    # 4. Calculate EWMA/CVD features at the TBT level (Activity + Wall Clock)
    ewma_node = node(
        func=calculate_ewma_features,
        inputs="enriched_tick_data",
        outputs="ewma_features_tbt",
        name="calculate_ewma_features_node"
    )

    # 5. Sample TBT features onto a high-frequency grid (25ms)
    sampling_node = node(
        func=partial(sample_features_to_grid, rule='25ms'),
        inputs="ewma_features_tbt",
        outputs="features_data_unvalidated_raw",
        name="sample_features_to_25ms_grid_node",
    )
    
    # 6. Calculate secondary features (RSI, Hurst Exponent) on the 25ms grid
    secondary_feature_node = node(
        func=generate_bar_features,
        inputs="features_data_unvalidated_raw",
        outputs="features_data_unvalidated",
        name="calculate_secondary_features_on_grid_node"
    )
    
    # 7. Logical validation
    logical_guardrail_node = node(
        func=validate_features_data_logic,
        inputs="features_data_unvalidated",
        outputs="features_data",
        name="logical_guardrail_node",
    )

    return Pipeline([
        initial_merge_node,
        tick_feature_node,
        structural_guardrail_node,
        ewma_node,
        sampling_node,
        secondary_feature_node,
        logical_guardrail_node,
    ])
