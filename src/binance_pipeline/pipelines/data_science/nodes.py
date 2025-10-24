
import pandas as pd
import numpy as np
import logging
import numba

log = logging.getLogger(__name__)

# =======================
# OPTIMIZED TRIPLE BARRIER LABELING
# =======================

@numba.jit(nopython=True)
def _compute_triple_barrier_labels_vectorized(
    prices: np.ndarray,
    upper_barriers: np.ndarray,
    lower_barriers: np.ndarray,
    time_barrier_periods: int
) -> np.ndarray:
    """Numba-accelerated triple barrier labeling."""
    n = len(prices)
    labels = np.full(n, np.nan)
    
    for i in range(n - time_barrier_periods):
        upper = upper_barriers[i]
        lower = lower_barriers[i]
        
        first_upper_idx, first_lower_idx = -1, -1
        
        for j in range(i + 1, min(i + 1 + time_barrier_periods, n)):
            if prices[j] >= upper and first_upper_idx == -1:
                first_upper_idx = j
            if prices[j] <= lower and first_lower_idx == -1:
                first_lower_idx = j
            if first_upper_idx != -1 and first_lower_idx != -1:
                break
        
        if first_upper_idx != -1 and (first_lower_idx == -1 or first_upper_idx < first_lower_idx):
            labels[i] = 1
        elif first_lower_idx != -1:
            labels[i] = -1
        else:
            labels[i] = 0
    
    return labels

def generate_triple_barrier_labels(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Improved triple-barrier labeling with Numba."""
    log.info(f"Generating triple-barrier labels using '{params.get('method', 'volatility')}' method...")
    
    price = df['close'].values
    
    if params.get('method') == 'fixed_pct':
        profit_pct, loss_pct = params['profit_take_gross_pct'], params['stop_loss_gross_pct']
        upper_barrier = price * (1 + profit_pct)
        lower_barrier = price * (1 - loss_pct)
        log.info(f"Fixed barriers: TP={profit_pct*100:.4f}%, SL={loss_pct*100:.4f}%")
    else:
        log.info("Using volatility-based dynamic barriers.")
        price_series = pd.Series(price)
        daily_vol = price_series.diff().rolling(window=params['vol_lookback']).std().values
        vol_mult = params['vol_multiplier']
        upper_barrier = price + daily_vol * vol_mult * params['profit_take_mult']
        lower_barrier = price - daily_vol * vol_mult * params['stop_loss_mult']
    
    labels = _compute_triple_barrier_labels_vectorized(price, upper_barrier, lower_barrier, params['time_barrier_periods'])
    
    df['label_raw'] = labels
    df.dropna(subset=['label_raw'], inplace=True)
    df['label'] = df['label_raw'].map({-1: 0, 0: 1, 1: 2}).astype(int)
    
    log.info(f"Label distribution:\n{df['label'].value_counts(normalize=True).sort_index()}")
    return df
