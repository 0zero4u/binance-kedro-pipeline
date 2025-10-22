from binance_pipeline.pipelines import lgbm
from kedro.config import ConfigLoader
from kedro.io import DataCatalog

#_config_loader = ConfigLoader(conf_source="conf")
#_catalog_patterns = _config_loader.get("catalog*", "catalog*/**")
#_catalog = DataCatalog.from_config(_catalog_patterns)

#_params_patterns = _config_loader.get("parameters*", "parameters*/**")
#_parameters = _config_loader.get(_params_patterns)

# HOOKS = ()
# SESSION_STORE_CLASS = ShelveStore
# SESSION_STORE_ARGS = {"path": "./sessions"}

# Class that manages how configuration is loaded.
# from kedro.config import TemplatedConfigLoader
# CONFIG_LOADER_CLASS = TemplatedConfigLoader
# CONFIG_LOADER_ARGS = {
#       "globals_pattern": "*globals.yml",
# }

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how data is loaded and saved.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog

# Lets Kedro know about our new pipeline
PIPELINES = {
    "__default__": lgbm.create_pipeline(),
    "lgbm": lgbm.create_pipeline(),
}
