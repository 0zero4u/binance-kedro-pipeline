from typing import Dict
from kedro.pipeline import Pipeline

    def register_pipelines() -> Dict[str, Pipeline]:
        """Register the project's pipelines.
        Returns:
            A mapping from pipeline names to ``Pipeline`` objects.
        """
        return {}
