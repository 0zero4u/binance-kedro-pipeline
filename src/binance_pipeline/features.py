import pandas as pd
import numpy as np
import numba
import logging
from typing import Dict, List

log = logging.getLogger(__name__)

class AdvancedFeatureEngine:
    """Advanced feature engineering with microstructure and regime detection."""

    def __init__(self):
        pass

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def _kyle_lambda_from_flow_numba(prices: np.ndarray, taker_flows: np.ndarray, window: int) -> np.ndarray:
        """
        Calculates Kyle's lambda by regressing price returns on signed taker flow.
        """
        n = len(prices)
        out = np.full(n, np.nan)
        for i in range(window, n):
            p_win = prices[i - window + 1 : i + 1]
            flow_win = taker_flows[i - window + 1 : i + 1]
            returns = np.diff(p_win)
            
            signed_flow_win = flow_win[1:]
            
            if len(returns) > 5 and np.std(signed_flow_win) > 1e-9:
                mean_flow = np.mean(signed_flow_win)
                mean_ret = np.mean(returns)
                num = np.sum((signed_flow_win - mean_flow) * (returns - mean_ret))
                den = np.sum((signed_flow_win - mean_flow) ** 2)
                if den > 1e-10:
                    out[i] = num / den
        return out

    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates advanced market microstructure features using high-fidelity taker flow."""
        log.info(f"Calculating market microstructure features for shape {df.shape}...")
        df_out = df.copy()
        
        df_out['kyle_lambda_50'] = self._kyle_lambda_from_flow_numba(
            df_out['close'].values, df_out['taker_flow'].values, 50
        )
        
        df_out['amihud_illiq'] = abs(df_out['returns']) / (df_out['volume'] * df_out['close'] + 1e-10)
        df_out['amihud_illiq_ewma_50'] = df_out['amihud_illiq'].ewm(span=50).mean()
        
        log.info(f"Microstructure features complete. Final shape: {df_out.shape}")
        return df_out

    def calculate_order_flow_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CORRECTED: Advanced order flow imbalance features.
        Calculates velocity and acceleration on the 'short' timeframe EWMA of OFI,
        which is guaranteed to exist in the input dataframe from a previous node.
        """
        log.info(f"Calculating order flow derivative features for shape {df.shape}...")
        df_out = df.copy()
        
        # This is the correct column name created by `calculate_intelligent_multi_scale_features`
        ofi_ewma_col = 'ofi_ewma_short' 
        
        if ofi_ewma_col in df_out.columns:
            # First derivative (velocity)
            df_out['ofi_velocity'] = df_out[ofi_ewma_col].diff(1)
            
            # Second derivative (acceleration)
            df_out['ofi_acceleration'] = df_out['ofi_velocity'].diff(1)
            log.info(f"Order flow derivatives calculated successfully.")
        else:
            log.warning(f"Column '{ofi_ewma_col}' not found. Skipping order flow derivative calculation.")
        
        log.info(f"Order flow derivatives complete. Final shape: {df_out.shape}")
        return df_out

    def select_features(self, df: pd.DataFrame, importance_dict: Dict[str, float], top_k: int = 100) -> List[str]:
        """Selects top K features based on importance scores."""
        log.info(f"Selecting top {top_k} features from {len(importance_dict)} available.")
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        selected = []
        for feat, importance in sorted_features:
            if feat in df.columns and len(selected) < top_k:
                selected.append(feat)
                
        log.info(f"Selected features count: {len(selected)}. Top 10: {[f[0] for f in sorted_features[:10]]}")
        return selected
