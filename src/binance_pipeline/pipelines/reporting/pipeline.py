from kedro.pipeline import Pipeline, node
from .nodes import create_unified_traceability_report

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=create_unified_traceability_report,
            inputs=[
                "merged_grid_15ms",
                "primary_features_grid",
                "intelligent_multi_scale_features", # <-- FIX: Use the new dataset
                "features_data_unvalidated", 
            ],
            outputs="unified_5min_traceability_report",
            name="create_traceability_report_node",
        )
    ])
