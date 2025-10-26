from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_lgbm_model
from binance_pipeline.pipelines.data_science.nodes import (
    generate_triple_barrier_labels,
    split_data # <-- Import the new function
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Add the new data splitting node
        node(
            func=split_data,
            inputs="features_data",
            outputs=["train_features", "test_features"], # You will need to add these to catalog.yml
            name="split_train_test_data_node",
        ),
        node(
            func=generate_triple_barrier_labels,
            inputs=["train_features", "params:labeling_params"], # Use train_features
            outputs="labeled_data",
            name="generate_lgbm_labels_node",
        ),
        node(
            func=train_lgbm_model,
            inputs=["labeled_data", "params:lgbm.lgbm_params", "params:lgbm.training_params"],
            outputs=["lgbm_model", "lgbm_eval_results"],
            name="train_lgbm_with_optuna_node",
        ),
        # You can add a new node here to evaluate the model on 'test_features'
    ])
