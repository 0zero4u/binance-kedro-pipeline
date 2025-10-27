from kedro.pipeline import Pipeline, node
from .nodes import create_unified_traceability_report

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=create_unified_traceability_report,
            inputs=[
                "merged_grid_15ms",
                "primary_features_grid",
                "ewma_features_grid",
                "features_data_unvalidated", # <-- FIX: Use the unvalidated data
            ],
            outputs="unified_5min_traceability_report",
            name="create_traceability_report_node",
        )
    ])
