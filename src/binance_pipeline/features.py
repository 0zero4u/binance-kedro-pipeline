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
        Calculates Kyle's lambda using true taker flow.
        This version regresses price returns directly on signed taker flow, which is
        more accurate than inferring trade direction from price ticks.
        """
        n = len(prices)
        out = np.full(n, np.nan)
        for i in range(window, n):
            p_win = prices[i - window + 1 : i + 1]
            flow_win = taker_flows[i - window + 1 : i + 1]
            returns = np.diff(p_win)
            
            # Use the taker flow from the corresponding periods of the returns
            signed_flow_win = flow_win[1:]
            
            if len(returns) > 5 and np.std(signed_flow_win) > 1e-9:
                # Regression: returns ~ signed_taker_flow
                mean_flow = np.mean(signed_flow_win)
                mean_ret = np.mean(returns)
                num = np.sum((signed_flow_win - mean_flow) * (returns - mean_ret))
                den = np.sum((signed_flow_win - mean_flow) ** 2)
                if den > 1e-10:
                    out[i] = num / den
        return out

    def _calculate_vpin_from_flow(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """
        Volume-Synchronized Probability of Informed Trading, calculated from true taker flow.
        This is more robust than inferring buy/sell volume from price changes.
        """
        # Volume imbalance is simply the absolute value of the net taker flow within the bar.
        volume_imbalance = abs(df['taker_flow'])
        
        # Total volume is the sum of all trade quantities in the bar.
        total_volume = df['volume']
        
        vpin = volume_imbalance.rolling(window).sum() / (total_volume.rolling(window).sum() + 1e-10)
        return vpin.fillna(method='ffill')

    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates advanced market microstructure features using high-fidelity taker flow."""
        log.info(f"Calculating market microstructure features for shape {df.shape}...")
        df_out = df.copy()
        
        # 1. Kyle's Lambda (price impact)
        # --- FIX: Use the new function that leverages the true 'taker_flow' ---
        df_out['kyle_lambda_50'] = self._kyle_lambda_from_flow_numba(
            df_out['close'].values, df_out['taker_flow'].values, 50
        )
        
        # 2. VPIN
        # --- FIX: Use the new function that leverages the true 'taker_flow' ---
        df_out['vpin_50'] = self._calculate_vpin_from_flow(df_out, window=50)

        # 3. Amihud illiquidity measure (this feature is independent of taker flow and remains the same)
        df_out['amihud_illiq'] = abs(df_out['returns']) / (df_out['volume'] * df_out['close'] + 1e-10)
        df_out['amihud_illiq_ewma_50'] = df_out['amihud_illiq'].ewm(span=50).mean()
        
        log.info(f"Microstructure features complete. Final shape: {df_out.shape}")
        return df_out

    def calculate_order_flow_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced order flow imbalance features."""
        log.info(f"Calculating order flow derivative features for shape {df.shape}...")
        df_out = df.copy()
        
        # OFI acceleration (2nd derivative)
        df_out['ofi_velocity'] = df_out['ofi_ewma_5s'].diff(1)
        df_out['ofi_acceleration'] = df_out['ofi_velocity'].diff(1)
        
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
            
