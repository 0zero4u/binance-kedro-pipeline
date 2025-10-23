from kedro.pipeline import Pipeline, node
from .nodes import generate_triple_barrier_labels, train_lgbm_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=generate_triple_barrier_labels,
            inputs=["features_data", "params:labeling_params"],
            outputs="labeled_data",
            name="generate_lgbm_labels_node",
        ),
        node(
            func=train_lgbm_model,
            inputs=["labeled_data", "params:lgbm_params"],
            outputs=["lgbm_model", "lgbm_eval_results"],
            name="train_lgbm_node",
        ),
    ])
    
