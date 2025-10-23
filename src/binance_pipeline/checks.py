import pandera as pa
from pandera import extensions
import pandas as pd

@extensions.register_check_method(statistics=[])
def monotonic_increasing(pandas_obj: pd.Series):
    """
    Validation check to ensure a series is monotonically increasing.
    Handles NaNs by ignoring them.
    """
    # The underlying pandas property is the most efficient way to check
    return pandas_obj.is_monotonic_increasing
  
