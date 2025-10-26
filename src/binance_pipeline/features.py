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
        # --- FIX: Changed 'price' to 'close' ---
        df['price_change'] = df['close'].diff()
        df['buy_volume'] = np.where(df['price_change'] > 0, df['qty'], 0)
        df['sell_volume'] = np.where(df['price_change'] < 0, df['qty'], 0)
        df['volume_imbalance'] = abs(df['buy_volume'] - df['sell_volume'])
        df['total_volume'] = df['buy_volume'] + df['sell_volume']
        vpin = df['volume_imbalance'].rolling(window).sum() / (df['total_volume'].rolling(window).sum() + 1e-10)
        return vpin.fillna(method='ffill')

    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates advanced market microstructure features."""
        log.info(f"Calculating market microstructure features for shape {df.shape}...")
        df_out = df.copy()
        
        # 1. Kyle's Lambda (price impact)
        # --- FIX: Changed 'price' to 'close' ---
        # Note: The 'qty' column from the original trade data is now aggregated as 'volume' in the grid.
        # For microstructure features like VPIN and Kyle's lambda, using the total bar volume ('volume')
        # is a standard and correct adaptation from tick-level to bar-level data.
        df_out['kyle_lambda_50'] = self._kyle_lambda_numba(df_out['close'].values, df_out['volume'].values, 50)
        
        # 2. VPIN
        # We need a 'qty' column for _calculate_vpin. Since we are on a grid, 'volume' is the equivalent.
        # We'll temporarily rename it for the function call.
        temp_df_for_vpin = df_out.rename(columns={'volume': 'qty'})
        df_out['vpin_50'] = self._calculate_vpin(temp_df_for_vpin, window=50)

        # 3. Amihud illiquidity measure
        # --- FIX: Use 'volume' instead of 'qty' and 'close' instead of 'price' ---
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
        
