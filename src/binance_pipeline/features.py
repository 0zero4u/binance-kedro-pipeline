
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
    def _kyle_lambda_numba(prices: np.ndarray, volumes: np.ndarray, window: int) -> np.ndarray:
        """Calculates Kyle's lambda (price impact measure) using Numba."""
        n = len(prices)
        out = np.full(n, np.nan)
        for i in range(window, n):
            p_win = prices[i - window + 1 : i + 1]
            v_win = volumes[i - window + 1 : i + 1]
            returns = np.diff(p_win)
            if len(returns) > 5 and np.std(v_win[1:]) > 1e-9:
                # Regression: returns ~ signed_volume
                signed_vol = v_win[1:] * np.sign(returns)
                mean_sv = np.mean(signed_vol)
                mean_ret = np.mean(returns)
                num = np.sum((signed_vol - mean_sv) * (returns - mean_ret))
                den = np.sum((signed_vol - mean_sv) ** 2)
                if den > 1e-10:
                    out[i] = num / den
        return out

    def _calculate_vpin(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """Volume-Synchronized Probability of Informed Trading."""
        df['price_change'] = df['price'].diff()
        df['buy_volume'] = np.where(df['price_change'] > 0, df['qty'], 0)
        df['sell_volume'] = np.where(df['price_change'] < 0, df['qty'], 0)
        df['volume_imbalance'] = abs(df['buy_volume'] - df['sell_volume'])
        df['total_volume'] = df['buy_volume'] + df['sell_volume']
        vpin = df['volume_imbalance'].rolling(window).sum() / (df['total_volume'].rolling(window).sum() + 1e-10)
        return vpin.fillna(method='ffill')

    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates advanced market microstructure features."""
        log.info("Calculating market microstructure features...")
        df_out = df.copy()
        
        # 1. Kyle's Lambda (price impact)
        df_out['kyle_lambda_50'] = self._kyle_lambda_numba(df_out['price'].values, df_out['qty'].values, 50)
        
        # 2. VPIN
        df_out['vpin_50'] = self._calculate_vpin(df_out, window=50)

        # 3. Amihud illiquidity measure
        df_out['amihud_illiq'] = abs(df_out['returns']) / (df_out['qty'] * df_out['price'] + 1e-10)
        df_out['amihud_illiq_ewma_50'] = df_out['amihud_illiq'].ewm(span=50).mean()
        
        return df_out

    def calculate_order_flow_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced order flow imbalance features."""
        log.info("Calculating order flow derivative features...")
        df_out = df.copy()
        
        # OFI acceleration (2nd derivative)
        df_out['ofi_velocity'] = df_out['ofi_ewma_5s'].diff(1)
        df_out['ofi_acceleration'] = df_out['ofi_velocity'].diff(1)
        
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
        
