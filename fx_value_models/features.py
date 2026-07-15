"""Feature construction for practical FX fair-value models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _series(data: pd.DataFrame, ticker: str) -> pd.Series:
    if ticker not in data.columns:
        return pd.Series(np.nan, index=data.index, name=ticker)
    return pd.to_numeric(data[ticker], errors="coerce")


def build_feature_frame(
    raw: pd.DataFrame,
    pair_ticker: str,
    feature_defs: list[dict],
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Build target, spot, and model features from raw Bloomberg history."""
    data = raw.sort_index().copy()
    data = data.apply(pd.to_numeric, errors="coerce").ffill(limit=ffill_limit)

    spot = _series(data, pair_ticker)
    frame = pd.DataFrame(index=data.index)
    frame["spot"] = spot
    frame["log_spot"] = np.log(spot.where(spot > 0))

    for spec in feature_defs:
        kind = spec["kind"]
        name = spec["name"]
        if kind == "diff":
            values = _series(data, spec["left"]) - _series(data, spec["right"])
        elif kind == "log_ratio":
            left = _series(data, spec["left"])
            right = _series(data, spec["right"])
            values = np.log(left.where(left > 0) / right.where(right > 0))
        elif kind == "log":
            base = _series(data, spec["ticker"])
            values = np.log(base.where(base > 0))
        elif kind == "level":
            values = _series(data, spec["ticker"])
        else:
            raise ValueError(f"Unknown feature kind: {kind}")
        frame[name] = values

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature availability."""
    rows = []
    for column in frame.columns:
        series = frame[column].dropna()
        rows.append(
            {
                "feature": column,
                "observations": int(series.shape[0]),
                "first": series.index.min().date().isoformat() if len(series) else "",
                "last": series.index.max().date().isoformat() if len(series) else "",
                "latest": float(series.iloc[-1]) if len(series) else None,
            }
        )
    return pd.DataFrame(rows)
