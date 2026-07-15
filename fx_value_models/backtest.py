"""Fade-the-residual backtest logic."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(series: pd.Series) -> float:
    equity = series.fillna(0.0).cumsum()
    peak = equity.cummax()
    return float((equity - peak).min())


def fade_backtest(
    model_frame: pd.DataFrame,
    entry_z: float,
    exit_z: float,
    max_hold_days: int,
    cost_bps_per_side: float,
    stop_z: float | None = None,
) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Backtest fading residual z-scores with next-close execution."""
    df = model_frame[["log_spot", "residual_z"]].dropna().copy()
    if len(df) < max_hold_days + 5:
        return _empty_metrics(entry_z), pd.DataFrame(), pd.Series(dtype=float)

    log_spot = df["log_spot"].to_numpy(dtype=float)
    z = df["residual_z"].to_numpy(dtype=float)
    dates = df.index
    n = len(df)
    cost = float(cost_bps_per_side) / 10000.0

    position_on_return = np.zeros(n)
    costs = np.zeros(n)
    trades = []
    i = 0

    while i < n - 2:
        signal_z = z[i]
        if not np.isfinite(signal_z) or abs(signal_z) < entry_z:
            i += 1
            continue

        entry_signal_i = i
        entry_exec_i = i + 1
        entry_sign = np.sign(signal_z)
        position = -entry_sign
        entry_price = log_spot[entry_exec_i]
        costs[entry_exec_i] -= cost

        exit_signal_i = None
        exit_reason = "end"
        last_signal_i = min(n - 2, entry_exec_i + max_hold_days)
        for j in range(entry_exec_i, last_signal_i + 1):
            held = j - entry_exec_i
            if held < 1 or not np.isfinite(z[j]):
                continue
            if stop_z is not None and np.sign(z[j]) == entry_sign and abs(z[j]) >= stop_z:
                exit_signal_i = j
                exit_reason = "stop"
                break
            if abs(z[j]) <= exit_z:
                exit_signal_i = j
                exit_reason = "mean_revert"
                break
            if held >= max_hold_days:
                exit_signal_i = j
                exit_reason = "timeout"
                break

        if exit_signal_i is None:
            exit_signal_i = last_signal_i
            exit_reason = "timeout"
        exit_exec_i = min(exit_signal_i + 1, n - 1)
        exit_price = log_spot[exit_exec_i]
        costs[exit_exec_i] -= cost

        position_on_return[entry_exec_i + 1 : exit_exec_i + 1] = position
        gross_pnl = float(position * (exit_price - entry_price))
        net_pnl = gross_pnl - 2.0 * cost
        trades.append(
            {
                "signal_date": dates[entry_signal_i],
                "entry_date": dates[entry_exec_i],
                "exit_signal_date": dates[exit_signal_i],
                "exit_date": dates[exit_exec_i],
                "entry_z": float(signal_z),
                "exit_z": float(z[exit_signal_i]),
                "position": int(position),
                "holding_days": int(exit_exec_i - entry_exec_i),
                "exit_reason": exit_reason,
                "gross_pnl_bp": gross_pnl * 10000.0,
                "net_pnl_bp": net_pnl * 10000.0,
            }
        )
        i = exit_exec_i + 1

    log_ret = np.zeros(n)
    log_ret[1:] = np.diff(log_spot)
    daily_pnl = pd.Series(
        position_on_return * log_ret + costs,
        index=dates,
        name="daily_net_pnl_log",
    )
    trades_df = pd.DataFrame(trades)
    metrics = summarize_backtest(
        daily_pnl=daily_pnl,
        trades=trades_df,
        position_on_return=position_on_return,
        entry_z=entry_z,
        exit_z=exit_z,
        max_hold_days=max_hold_days,
        cost_bps_per_side=cost_bps_per_side,
        stop_z=stop_z,
    )
    return metrics, trades_df, daily_pnl


def _empty_metrics(entry_z: float) -> dict:
    return {
        "entry_z": entry_z,
        "trades": 0,
        "sharpe": np.nan,
        "hit_rate": np.nan,
        "convergence_rate": np.nan,
        "avg_holding_days": np.nan,
        "net_pnl_bp_total": 0.0,
        "max_drawdown_bp": np.nan,
    }


def summarize_backtest(
    daily_pnl: pd.Series,
    trades: pd.DataFrame,
    position_on_return: np.ndarray,
    entry_z: float,
    exit_z: float,
    max_hold_days: int,
    cost_bps_per_side: float,
    stop_z: float | None,
) -> dict:
    ann = 252.0
    vol = float(daily_pnl.std(ddof=1))
    sharpe = float(daily_pnl.mean() / vol * np.sqrt(ann)) if vol > 0 else np.nan
    trade_count = int(len(trades))

    if trade_count:
        hit_rate = float((trades["net_pnl_bp"] > 0).mean())
        convergence_rate = float((trades["exit_reason"] == "mean_revert").mean())
        timeout_rate = float((trades["exit_reason"] == "timeout").mean())
        stop_rate = float((trades["exit_reason"] == "stop").mean())
        avg_holding = float(trades["holding_days"].mean())
        median_holding = float(trades["holding_days"].median())
        avg_trade_pnl = float(trades["net_pnl_bp"].mean())
        median_trade_pnl = float(trades["net_pnl_bp"].median())
    else:
        hit_rate = convergence_rate = timeout_rate = stop_rate = np.nan
        avg_holding = median_holding = avg_trade_pnl = median_trade_pnl = np.nan

    return {
        "entry_z": float(entry_z),
        "exit_z": float(exit_z),
        "stop_z": float(stop_z) if stop_z is not None else np.nan,
        "max_hold_days": int(max_hold_days),
        "cost_bps_per_side": float(cost_bps_per_side),
        "trades": trade_count,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "convergence_rate": convergence_rate,
        "timeout_rate": timeout_rate,
        "stop_rate": stop_rate,
        "avg_holding_days": avg_holding,
        "median_holding_days": median_holding,
        "avg_trade_net_pnl_bp": avg_trade_pnl,
        "median_trade_net_pnl_bp": median_trade_pnl,
        "net_pnl_bp_total": float(daily_pnl.sum() * 10000.0),
        "ann_pnl_bp": float(daily_pnl.mean() * ann * 10000.0),
        "ann_vol_bp": float(vol * np.sqrt(ann) * 10000.0),
        "max_drawdown_bp": _max_drawdown(daily_pnl) * 10000.0,
        "exposure": float(np.mean(position_on_return != 0)),
    }
