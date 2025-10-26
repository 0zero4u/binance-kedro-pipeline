from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_lgbm_model, evaluate_model  # Import evaluate_model
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
            name="split_train_test_data_node",
        ),
        # 2. Label the training data for model training and CV
        node(
            func=generate_triple_barrier_labels,
            inputs=["train_features", "params:labeling_params"],
            outputs="labeled_data",
            name="generate_lgbm_labels_for_training_node",
        ),
        # 3. Train the model ONLY on the labeled training data
        node(
            func=train_lgbm_model,
            inputs=["labeled_data", "params:lgbm.lgbm_params", "params:lgbm.training_params"],
            outputs=["lgbm_model", "lgbm_eval_results"],
            name="train_lgbm_with_optuna_node",
        ),
        # 4. Label the holdout test data (using the same parameters)
        node(
            func=generate_triple_barrier_labels,
            inputs=["test_features", "params:labeling_params"],
            outputs="test_labeled_data",
            name="generate_lgbm_labels_for_testing_node",
        ),
        # 5. Evaluate the final model on the unseen, labeled test data
        node(
            func=evaluate_model,
            inputs=["lgbm_model", "test_labeled_data"],
            outputs="lgbm_holdout_test_results",
            name="evaluate_lgbm_on_holdout_node",
        ),
    ])
