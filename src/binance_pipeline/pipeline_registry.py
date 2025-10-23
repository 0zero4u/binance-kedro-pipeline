"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from binance_pipeline.pipeline import create_pipeline as create_data_engineering_pipeline
from binance_pipeline.pipelines.lgbm.pipeline import create_pipeline as create_lgbm_pipeline
from binance_pipeline.pipelines.arf.pipeline import create_pipeline as create_arf_pipeline
from binance_pipeline.pipelines.reporting.pipeline import create_pipeline as create_reporting_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    # Create instances of the individual pipelines
    de_pipeline = create_data_engineering_pipeline()
    lgbm_pipeline = create_lgbm_pipeline()
    arf_pipeline = create_arf_pipeline()
    reporting_pipeline = create_reporting_pipeline()

    # Define the dictionary of all pipelines to be returned
    return {
        # Default pipeline when running "kedro run"
        "__default__": de_pipeline,

        # Individual pipelines
        "de": de_pipeline,
        "lgbm": lgbm_pipeline,
        "arf": arf_pipeline,
        "reporting": reporting_pipeline,

        # End-to-end pipelines
        "e2e_lgbm": de_pipeline + lgbm_pipeline,
        "e2e_arf": de_pipeline + arf_pipeline,
        "e2e_reporting": de_pipeline + reporting_pipeline,
}
