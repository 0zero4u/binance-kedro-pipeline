from kedro.pipeline import Pipeline, node
from functools import partial
from .nodes import (
    create_and_merge_grids_with_polars,
    calculate_primary_grid_features,
    calculate_ewma_features_on_grid,
    generate_bar_features,
)
from .validation import validate_features_data_logic
from .features import AdvancedFeatureEngine


def create_pipeline(**kwargs) -> Pipeline:
    
    adv_feature_engine = AdvancedFeatureEngine()

    # --- HYPER-OPTIMIZED "GRID-FIRST" PIPELINE ---

    # 1. Create, merge, and fill grids in one go with Polars
    create_and_merge_grids_node = node(
        func=partial(create_and_merge_grids_with_polars, rule='15ms'),
        inputs=["trade_raw", "book_raw"],
        outputs="merged_grid_15ms",
        name="create_and_merge_grids_with_polars_node",
    )
    
    # 2. Calculate primary features (spread, microprice, OFI, etc.) on the grid
    primary_features_node = node(
        func=calculate_primary_grid_features,
        inputs="merged_grid_15ms",
        outputs="primary_features_grid",
        name="calculate_primary_features_node",
    )

    # 3. Calculate EWMA/rolling features on the grid
    ewma_node = node(
        func=calculate_ewma_features_on_grid,
        inputs="primary_features_grid",
        outputs="ewma_features_grid",
        name="calculate_ewma_features_node"
    )

    # 4. Calculate bar-based features (RSI, Hurst) on the grid
    bar_features_node = node(
        func=generate_bar_features,
        inputs="ewma_features_grid",
        outputs="grid_with_bar_features",
        name="calculate_bar_features_node"
    )

    # 5. Add advanced microstructure features
    microstructure_node = node(
        func=adv_feature_engine.calculate_microstructure_features,
        inputs="grid_with_bar_features",
        outputs="grid_with_microstructure_features",
        name="add_microstructure_features_node"
    )
    
    # 6. Add advanced order flow features
    order_flow_node = node(
        func=adv_feature_engine.calculate_order_flow_derivatives,
        inputs="grid_with_microstructure_features",
        outputs="features_data_unvalidated",
        name="add_order_flow_derivatives_node"
    )

    # 7. Final logical validation
    logical_guardrail_node = node(
        func=validate_features_data_logic,
        inputs="features_data_unvalidated",
        outputs="features_data",
        name="logical_guardrail_node",
    )

    return Pipeline([
        create_and_merge_grids_node,
        primary_features_node,
        ewma_node,
        bar_features_node,
        microstructure_node,
        order_flow_node,
        logical_guardrail_node,
    ])
