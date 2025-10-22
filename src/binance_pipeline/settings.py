from binance_pipeline.pipeline import create_pipeline as create_data_engineering_pipeline
from binance_pipeline.pipelines.lgbm.pipeline import create_pipeline as create_lgbm_pipeline

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

    # The data engineering pipeline, which you can also run by name
    "de": create_data_engineering_pipeline(),

    # The modeling pipeline, which you can run by name
    "lgbm": create_lgbm_pipeline(),
}
