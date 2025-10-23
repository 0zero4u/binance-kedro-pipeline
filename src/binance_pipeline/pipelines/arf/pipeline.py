from kedro.pipeline import Pipeline, node
from .nodes import (
    generate_river_features,
    generate_triple_barrier_labels_with_endtime,
    filter_unbiased_samples,
    fit_hybrid_scaler,
    apply_hybrid_scaling,
    train_arf_model_on_scaled_data
)

def create_pipeline(**kwargs) -> Pipeline:
    """Creates the ARF model training pipeline."""
    return Pipeline(
        [
            node(
                func=generate_river_features,
                inputs="features_data",
                outputs="features_data_arf",
                name="generate_arf_features_node",
            ),
            node(
                func=generate_triple_barrier_labels_with_endtime,
                inputs=["features_data_arf", "params:lgbm.labeling_params"],
                outputs="labeled_data_arf",
                name="generate_arf_labels_node",
            ),
            node(
                func=filter_unbiased_samples,
                inputs="labeled_data_arf",
                outputs="unbiased_labeled_data_arf",
                name="filter_unbiased_samples_node",
            ),
            node(
                func=fit_hybrid_scaler,
                inputs="unbiased_labeled_data_arf",
                outputs="hybrid_scaler",
                name="fit_hybrid_scaler_node",
            ),
            node(
                func=apply_hybrid_scaling,
                inputs=["unbiased_labeled_data_arf", "hybrid_scaler"],
                outputs="scaled_unbiased_data_arf",
                name="apply_hybrid_scaling_node",
            ),
            node(
                func=train_arf_model_on_scaled_data,
                inputs="scaled_unbiased_data_arf",
                outputs="arf_model",
                name="train_arf_model_node",
            ),
        ]
    )
