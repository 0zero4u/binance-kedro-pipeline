
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    # generate_arf_features, <-- REMOVED
    generate_arf_hpo_configs,
    fit_robust_scaler,
    apply_robust_scaling,
    train_arf_ensemble,
    select_best_arf_model,
    evaluate_arf_model
)
from binance_pipeline.pipelines.data_science.nodes import (
    generate_triple_barrier_labels,
    split_data
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # 1. Split data into a training set and a final holdout test set
        node(
            func=split_data,
            inputs="features_data",
            outputs=["train_features", "test_features"],
            name="split_arf_train_test_data_node",
        ),
        
        # --- TRAINING PATH ---
        # 2. REMOVED: The ARF-specific feature generation node is gone.
        
        # 3. Label the TRAINING data directly
        node(
            func=generate_triple_barrier_labels,
            inputs=["train_features", "params:labeling_params"], # <-- FIX: Input changed
            outputs="labeled_data_arf",
            name="generate_arf_labels_for_training_node",
        ),
        # 4. Fit the scaler ONLY on the TRAINING data
        node(
            func=fit_robust_scaler,
            inputs="labeled_data_arf",
            outputs="robust_scaler",
            name="fit_robust_scaler_node",
        ),
        # 5. Apply scaling to the TRAINING data
        node(
            func=apply_robust_scaling,
            inputs=["labeled_data_arf", "robust_scaler"],
            outputs="scaled_data_arf",
            name="apply_scaling_to_training_data_node",
        ),
        # 6. Generate HPO configs for the ensemble
        node(
            func=generate_arf_hpo_configs,
            inputs="params:arf.hpo_params",
            outputs="arf_hpo_configs",
            name="generate_arf_hpo_configs_node"
        ),
        # 7. Train the ARF model ONLY on the scaled TRAINING data
        node(
            func=train_arf_ensemble,
            inputs=["scaled_data_arf", "arf_hpo_configs"],
            outputs="arf_training_results",
            name="train_arf_ensemble_node",
        ),
        # 8. Select the best performing model from the ensemble
        node(
            func=select_best_arf_model,
            inputs="arf_training_results",
            outputs="arf_model",
            name="select_best_arf_model_node",
        ),
        
        # --- EVALUATION PATH ---
        # 9. REMOVED: The ARF-specific feature generation node for test data is gone.
        
        # 10. Label the ARF TEST data directly
        node(
            func=generate_triple_barrier_labels,
            inputs=["test_features", "params:labeling_params"], # <-- FIX: Input changed
            outputs="test_labeled_data_arf",
            name="generate_arf_labels_for_testing_node",
        ),
        # 11. Apply the FITTED scaler to the TEST data
        node(
            func=apply_robust_scaling,
            inputs=["test_labeled_data_arf", "robust_scaler"],
            outputs="test_scaled_data_arf",
            name="apply_scaling_to_testing_data_node",
        ),
        # 12. Evaluate the final model on the unseen, scaled TEST data
        node(
            func=evaluate_arf_model,
            inputs=["arf_model", "test_scaled_data_arf"],
            outputs="arf_holdout_test_results",
            name="evaluate_arf_on_holdout_node",
        ),
    ])
