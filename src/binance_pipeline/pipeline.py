from kedro.pipeline import Pipeline, node
from .nodes import (
    merge_book_trade_asof,
    resample_to_time_bars,
    generate_bar_features,
)

def create_pipeline(**kwargs) -> Pipeline:
    # This is the data engineering pipeline
    return Pipeline([
        node(
            func=merge_book_trade_asof,
            inputs=["book_raw", "trade_raw"],
            outputs="merged_tick_data",
            name="merge_ticks_asof_node"
        ),
        node(
            func=resample_to_time_bars,
            inputs="merged_tick_data",
            outputs="resampled_data",
            name="resample_to_100ms_bars_node"
        ),
        node(
            func=generate_bar_features,
            inputs="resampled_data",
            outputs="features_data",
            name="generate_bar_features_node"
        ),
    ])
  
