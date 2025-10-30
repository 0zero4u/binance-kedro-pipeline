"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

# --- CORRECT IMPORT AND CLASS NAME FOR KEDRO 0.18.x ---
from kedro.config import OmegaConfigLoader

# Instantiated project hooks.
# from binance_pipeline.hooks import ProjectHooks
# HOOKS = (ProjectHooks(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import ShelveStore
# SESSION_STORE_CLASS = ShelveStore
# SESSION_STORE_ARGS = {"path": "./sessions"}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how data is loaded and saved.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog

# --- CONFIGURATION LOADER TO FIND ALL PARAMETER FILES ---
# This tells Kedro to look for and load all .yml files inside the 'conf/base' directory,
# correctly finding the consolidated parameters.yml.
CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "config_patterns": {
        "parameters": ["parameters.yml"], # Correctly points to the single file
        "catalog": ["catalog.yml"],
    }
}
