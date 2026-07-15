"""Bloomberg data access helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _to_pandas(frame) -> pd.DataFrame:
    """Normalize xbbg/narwhals/pyarrow/pandas outputs to pandas."""
    if hasattr(frame, "to_native"):
        frame = frame.to_native()
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    return frame


def fetch_history(
    tickers: list[str],
    field: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch daily Bloomberg history and return a date-indexed wide frame."""
    from xbbg import blp

    raw = _to_pandas(blp.bdh(tickers, field, start, end))
    required = {"ticker", "date", "value"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Bloomberg response missing columns: {sorted(missing)}")

    raw["date"] = pd.to_datetime(raw["date"])
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    wide = raw.pivot_table(
        index="date", columns="ticker", values="value", aggfunc="last"
    ).sort_index()
    wide.index.name = "date"
    return wide


def load_or_fetch_history(
    cache_path: Path,
    tickers: list[str],
    field: str,
    start: str,
    end: str,
    refresh: bool,
) -> pd.DataFrame:
    """Load cached Bloomberg history unless a refresh is requested."""
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_history(tickers, field, start, end)
    data.to_csv(cache_path, index_label="date")
    return data


def coverage_table(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize non-null history by ticker."""
    rows = []
    for column in data.columns:
        series = data[column].dropna()
        rows.append(
            {
                "ticker": column,
                "observations": int(series.shape[0]),
                "first": series.index.min().date().isoformat() if len(series) else "",
                "last": series.index.max().date().isoformat() if len(series) else "",
                "latest": float(series.iloc[-1]) if len(series) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
