from kedro.pipeline import Pipeline, node
from functools import partial
from .nodes import (
    merge_book_trade_asof,
    calculate_tick_level_features,
    resample_to_time_bars,
    generate_bar_features,
    merge_multi_timeframe_features,
)
from .validation import validate_enriched_tick_data, validate_features_data_logic


def create_pipeline(**kwargs) -> Pipeline:
    timeframes = ["100ms", "3s", "15s", "1min", "3min"]
    base_tf = timeframes[0]
    other_tfs = timeframes[1:]

    # Initial merge
    initial_merge_node = node(
        func=merge_book_trade_asof,
        inputs=["book_raw", "trade_raw"],
        outputs="merged_tick_data",
        name="merge_ticks_asof_node",
    )

    # Enrich tick data
    tick_feature_node = node(
        func=calculate_tick_level_features,
        inputs="merged_tick_data",
        outputs="enriched_tick_data_unvalidated",
        name="calculate_tick_features_node",
    )

    # Validate tick structure
    structural_guardrail_node = node(
        func=validate_enriched_tick_data,
        inputs="enriched_tick_data_unvalidated",
        outputs="enriched_tick_data",
        name="structural_guardrail_node",
    )

    # Resample + Feature generation per timeframe
    resample_and_feature_nodes = []
    for tf in timeframes:
        resample_and_feature_nodes.extend([
            node(
                func=partial(resample_to_time_bars, rule=tf),
                inputs="enriched_tick_data",
                outputs=f"resampled_data_{tf}",
                name=f"resample_to_{tf}_bars_node",
            ),
            node(
                func=generate_bar_features,
                inputs=f"resampled_data_{tf}",
                outputs=f"features_data_{tf}",
                name=f"generate_bar_features_{tf}_node",
            ),
        ])

    # Merge multi-timeframe feature sets
    merge_inputs = {
        f"features_{tf.replace('min', 'm')}": f"features_data_{tf}"
        for tf in other_tfs
    }

    merge_node = node(
        func=merge_multi_timeframe_features,
        inputs={"base_features": f"features_data_{base_tf}", **merge_inputs},
        outputs="features_data_unvalidated",
        name="merge_multi_timeframe_features_node",
    )

    # Logical validation
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
        *resample_and_feature_nodes,
        merge_node,
        logical_guardrail_node,
    ])
