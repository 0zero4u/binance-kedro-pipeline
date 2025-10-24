from kedro.pipeline import Pipeline, node
from .nodes import (
    generate_arf_features,
    generate_arf_hpo_configs,
    fit_robust_scaler,
    apply_robust_scaling,
    train_arf_ensemble,
    select_best_arf_model
)
from binance_pipeline.pipelines.data_science.nodes import generate_triple_barrier_labels


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=generate_arf_features,
            inputs=["features_data", "params:arf.feature_params"],
            outputs="features_data_arf",
            name="generate_arf_features_node",
        ),
        node(
            func=generate_triple_barrier_labels,
            inputs=["features_data_arf", "params:labeling_params"],
            outputs="labeled_data_arf",
            name="generate_arf_labels_node",
        ),
        node(
            func=fit_robust_scaler,
            inputs="labeled_data_arf",
            outputs="robust_scaler",
            name="fit_robust_scaler_node",
        ),
        node(
            func=apply_robust_scaling,
            inputs=["labeled_data_arf", "robust_scaler"],
            outputs="scaled_data_arf",
            name="apply_robust_scaling_node",
        ),
        # --- NEW NODE to generate HPO configs ---
        node(
            func=generate_arf_hpo_configs,
            inputs="params:arf.hpo_params",
            outputs="arf_hpo_configs",
            name="generate_arf_hpo_configs_node"
        ),
        # --- UPDATED NODE to use HPO configs ---
        node(
            func=train_arf_ensemble,
            inputs=["scaled_data_arf", "arf_hpo_configs"],
            outputs="arf_training_results",
            name="train_arf_ensemble_with_hpo_node",
        ),
        node(
            func=select_best_arf_model,
            inputs="arf_training_results",
            outputs="arf_model",
            name="select_best_arf_model_node",
        ),
    ])
