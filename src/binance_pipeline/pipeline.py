from kedro.pipeline import Pipeline, node
from .nodes import (
    download_and_unzip, # Assuming download might be part of it, keeping for context
    merge_book_trade_asof,
    resample_to_time_bars,
    generate_bar_features,
    merge_multi_timeframe_features,
)

def create_pipeline(**kwargs) -> Pipeline:
    # This is the data engineering pipeline
    timeframes = ["100ms", "3s", "15s", "1min", "3min"]
    base_tf = timeframes[0]
    other_tfs = timeframes[1:]

    # 1. Start with the initial merge of raw data
    initial_merge_node = node(
        func=merge_book_trade_asof,
        inputs=["book_raw", "trade_raw"],
        outputs="merged_tick_data",
        name="merge_ticks_asof_node",
    )

    # 2. Fan out: Create resample and feature nodes for each timeframe
    resample_and_feature_nodes = []
    for tf in timeframes:
        resample_and_feature_nodes.append(
            node(
                func=resample_to_time_bars,
                inputs="merged_tick_data",
                outputs=f"resampled_data_{tf}",
                name=f"resample_to_{tf}_bars_node",
                kwargs={"rule": tf},
            )
        )
        resample_and_feature_nodes.append(
            node(
                func=generate_bar_features,
                inputs=f"resampled_data_{tf}",
                outputs=f"features_data_{tf}",
                name=f"generate_bar_features_{tf}_node",
            )
        )

    # 3. Fan in: Merge all feature sets into one DataFrame
    merge_inputs = {f"features_{tf.replace('min', 'm')}": f"features_data_{tf}" for tf in other_tfs}
    merge_node = node(
        func=merge_multi_timeframe_features,
        inputs={"base_features": f"features_data_{base_tf}", **merge_inputs},
        outputs="features_data",
        name="merge_multi_timeframe_features_node",
    )

    return Pipeline([initial_merge_node] + resample_and_feature_nodes + [merge_node])
    
