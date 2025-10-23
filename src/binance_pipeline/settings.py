from kedro.pipeline import pipeline
from binance_pipeline.pipeline import create_pipeline as create_data_engineering_pipeline
from binance_pipeline.pipelines.lgbm.pipeline import create_pipeline as create_lgbm_pipeline
from binance_pipeline.pipelines.arf.pipeline import create_pipeline as create_arf_pipeline
from binance_pipeline.pipelines.reporting.pipeline import create_pipeline as create_reporting_pipeline

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how data is loaded and saved.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog


# This dictionary registers all pipelines in your project.
PIPELINES = {
    # The pipeline that runs when you just type "kedro run"
    "__default__": create_data_engineering_pipeline(),

    # The data engineering pipeline, now with validation built-in
    "de": create_data_engineering_pipeline(),

    # The original modeling pipeline
    "lgbm": create_lgbm_pipeline(),

    # The new ARF modeling pipeline
    "arf": create_arf_pipeline(),
    
    # The new reporting pipeline
    "reporting": create_reporting_pipeline(),
    
    # New end-to-end pipelines for convenience
    "e2e_lgbm": pipeline(create_data_engineering_pipeline()) + pipeline(create_lgbm_pipeline()),
    "e2e_arf": pipeline(create_data_engineering_pipeline()) + pipeline(create_arf_pipeline()),
    
    # The new end-to-end pipeline with the traceability report
    "e2e_reporting": pipeline(create_data_engineering_pipeline()) + pipeline(create_reporting_pipeline()),
}
