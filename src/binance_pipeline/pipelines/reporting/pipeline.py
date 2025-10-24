from kedro.pipeline import Pipeline, node
from .nodes import create_unified_traceability_report

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=create_unified_traceability_report,
            inputs=[
                "merged_tick_data",
                "ewma_features_tbt",
                "features_data",
            ],
            outputs="unified_5min_traceability_report",
            name="create_traceability_report_node",
        )
    ])
  
