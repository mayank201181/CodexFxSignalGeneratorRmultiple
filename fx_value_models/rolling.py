"""Rolling fair-value estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_ridge(
    train: pd.DataFrame,
    target: str,
    features: list[str],
    alpha: float,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    x = train[features]
    y = train[target].to_numpy(dtype=float)
    means = x.mean()
    stds = x.std(ddof=0).replace(0, 1.0)
    x_std = ((x - means) / stds).to_numpy(dtype=float)
    design = np.column_stack([np.ones(x_std.shape[0]), x_std])

    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    xtx = design.T @ design
    xty = design.T @ y
    try:
        beta = np.linalg.solve(xtx + penalty, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtx + penalty, xty, rcond=None)[0]
    return beta, means, stds


def rolling_fair_value(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    window: int,
    ridge_alpha: float,
    min_train_frac: float = 0.8,
    z_window: int = 252,
) -> pd.DataFrame:
    """Estimate rolling fair value using only information available at t-1."""
    cols = [target, *features]
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for model: {missing}")

    work = frame[cols].copy()
    min_obs = max(len(features) + 20, int(window * min_train_frac))
    yhat = pd.Series(np.nan, index=work.index, name="log_fair_value")
    fit_obs = pd.Series(np.nan, index=work.index, name="fit_obs")

    for i in range(window, len(work)):
        current = work.iloc[i]
        if current[features].isna().any():
            continue
        train = work.iloc[i - window : i].dropna()
        if len(train) < min_obs:
            continue
        beta, means, stds = _fit_ridge(train, target, features, ridge_alpha)
        x_now = ((current[features] - means) / stds).to_numpy(dtype=float)
        yhat.iloc[i] = float(beta[0] + x_now @ beta[1:])
        fit_obs.iloc[i] = float(len(train))

    result = pd.DataFrame(index=work.index)
    result["log_spot"] = work[target]
    result["log_fair_value"] = yhat
    result["fair_value"] = np.exp(result["log_fair_value"])
    result["residual_log"] = result["log_spot"] - result["log_fair_value"]
    result["residual_pct"] = np.expm1(result["residual_log"]) * 100.0
    result["fit_obs"] = fit_obs

    residual = result["residual_log"]
    min_z_obs = max(60, int(z_window * 0.6))
    resid_mean = residual.rolling(z_window, min_periods=min_z_obs).mean().shift(1)
    resid_std = residual.rolling(z_window, min_periods=min_z_obs).std().shift(1)
    result["residual_mean"] = resid_mean
    result["residual_std"] = resid_std
    result["trade_residual_log"] = residual - resid_mean
    result["trade_gap_pct"] = np.expm1(result["trade_residual_log"]) * 100.0
    result["trade_log_fair_value"] = result["log_fair_value"] + resid_mean
    result["trade_fair_value"] = np.exp(result["trade_log_fair_value"])
    result["residual_z"] = result["trade_residual_log"] / resid_std
    return result


def latest_contributions(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    asof,
    window: int,
    ridge_alpha: float,
    min_train_frac: float = 0.8,
) -> pd.DataFrame:
    """Return the latest standardized driver contributions for one model."""
    if asof not in frame.index:
        raise ValueError("asof date is not present in frame")
    i = frame.index.get_loc(asof)
    if not isinstance(i, int) or i < window:
        return pd.DataFrame()

    work = frame[[target, *features]].copy()
    min_obs = max(len(features) + 20, int(window * min_train_frac))
    train = work.iloc[i - window : i].dropna()
    current = work.iloc[i]
    if len(train) < min_obs or current[features].isna().any():
        return pd.DataFrame()

    beta, means, stds = _fit_ridge(train, target, features, ridge_alpha)
    standardized = (current[features] - means) / stds
    rows = []
    for feature, z_value, coef in zip(features, standardized, beta[1:]):
        rows.append(
            {
                "feature": feature,
                "standardized_value": float(z_value),
                "beta_on_standardized_feature": float(coef),
                "log_fv_contribution": float(z_value * coef),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "log_fv_contribution", key=lambda s: s.abs(), ascending=False
    )
