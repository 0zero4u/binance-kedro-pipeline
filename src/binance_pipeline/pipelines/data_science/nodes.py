import pandas as pd
import numpy as np
import logging
import numba

log = logging.getLogger(__name__)

# =========================================================
# UPGRADED FEE-AWARE TRIPLE BARRIER LABELING
# =========================================================

@numba.jit(nopython=True, fastmath=True)
def _compute_fee_aware_barriers_vectorized(
    prices: np.ndarray,
    upper_barriers_gross: np.ndarray,
    lower_barriers_gross: np.ndarray,
    time_barrier_periods: int,
    fee_pct: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized triple barrier with fee-aware net P&L calculation.
    Returns: (labels, barrier_hit_times, gross_returns, net_returns)
    """
    n = len(prices)
    labels = np.full(n, np.nan)
    barrier_times = np.full(n, np.nan)
    gross_returns = np.full(n, np.nan)
    net_returns = np.full(n, np.nan)

    for i in range(n - time_barrier_periods):
        entry_price = prices[i]
        upper_barrier = upper_barriers_gross[i]
        lower_barrier = lower_barriers_gross[i]
        
        first_upper_idx, first_lower_idx = -1, -1

        for j in range(i + 1, min(i + 1 + time_barrier_periods, n)):
            exit_price = prices[j]
            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - (2 * fee_pct)

            if exit_price >= upper_barrier and first_upper_idx == -1:
                if net_return > 0:  # Only label as profit if NET return is positive
                    first_upper_idx = j
            if exit_price <= lower_barrier and first_lower_idx == -1:
                first_lower_idx = j
            if first_upper_idx != -1 and first_lower_idx != -1:
                break

        if first_upper_idx != -1 and (first_lower_idx == -1 or first_upper_idx < first_lower_idx):
            labels[i], barrier_times[i], exit_idx = 1, first_upper_idx - i, first_upper_idx
        elif first_lower_idx != -1:
            labels[i], barrier_times[i], exit_idx = -1, first_lower_idx - i, first_lower_idx
        else:
            labels[i], barrier_times[i], exit_idx = 0, time_barrier_periods, min(i + time_barrier_periods, n - 1)
        
        exit_price = prices[exit_idx]
        gross_returns[i] = (exit_price - entry_price) / entry_price
        net_returns[i] = gross_returns[i] - (2 * fee_pct)

    return labels, barrier_times, gross_returns, net_returns


def generate_triple_barrier_labels(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    UPGRADED triple-barrier labeling with fee awareness and diagnostics.
    It calculates barriers required to achieve a target NET profit/loss.
    """
    log.info(f"Generating FEE-AWARE triple-barrier labels using '{params.get('method', 'fixed_pct')}' method...")

    price = df['close'].values
    fee_pct = params.get('fee_pct', 0.00025)

    if params.get('method') == 'fixed_pct':
        profit_net_target = params['profit_take_net_pct']
        loss_net_target = params['stop_loss_net_pct']
        
        # Calculate the required GROSS movement to achieve the desired NET return
        profit_gross_required = profit_net_target + (2 * fee_pct)
        loss_gross_required = loss_net_target + (2 * fee_pct)
        
        upper_barrier = price * (1 + profit_gross_required)
        lower_barrier = price * (1 - loss_gross_required)
        log.info(f"Fee-adjusted barriers: TP_gross={profit_gross_required*100:.4f}%, SL_gross={loss_gross_required*100:.4f}% for net targets.")
    else: # Volatility-based
        price_series = pd.Series(price)
        daily_vol = price_series.diff().rolling(window=params['vol_lookback']).std().values
        upper_barrier = price + daily_vol * params['vol_multiplier'] * params['profit_take_mult']
        lower_barrier = price - daily_vol * params['vol_multiplier'] * params['stop_loss_mult']

    labels, barrier_times, gross_returns, net_returns = _compute_fee_aware_barriers_vectorized(
        price, upper_barrier, lower_barrier, params['time_barrier_periods'], fee_pct
    )

    df_out = df.copy()
    df_out['label_raw'] = labels
    df_out['barrier_hit_time'] = barrier_times
    df_out['gross_return'] = gross_returns
    df_out['net_return'] = net_returns
    df_out.dropna(subset=['label_raw'], inplace=True)
    df_out['label'] = df_out['label_raw'].map({-1: 0, 0: 1, 1: 2}).astype(int)

    log.info("\n--- LABELING DIAGNOSTICS ---")
    log.info(f"Label distribution:\n{df_out['label'].value_counts(normalize=True).sort_index()}")
    profitable_trades = df_out[df_out['label'] == 2]
    if not profitable_trades.empty:
        avg_net_profit = profitable_trades['net_return'].mean()
        log.info(f"Average NET profit on winning trades: {avg_net_profit*100:.4f}%")
        if avg_net_profit <= 0:
            log.warning("WARNING: Average NET profit is <= 0! Barriers may be too tight.")
    
    return df_out
    
