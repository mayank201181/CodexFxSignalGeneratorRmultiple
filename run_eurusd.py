"""Run the EURUSD practical fair-value and residual-fade research harness."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fx_value_models.backtest import fade_backtest
from fx_value_models.bbg import coverage_table, load_or_fetch_history
from fx_value_models.eurusd_config import (
    COST_BPS_PER_SIDE,
    ENTRY_ZS,
    EXIT_ZS,
    FEATURE_DEFS,
    FIELD,
    MAX_HOLD_DAYS,
    MODEL_SPECS,
    PAIR_TICKER,
    ROLLING_WINDOWS,
    STOP_Z,
    TICKERS,
    Z_WINDOW,
)
from fx_value_models.features import build_feature_frame, feature_coverage
from fx_value_models.report import save_best_chart, write_index_page
from fx_value_models.rolling import latest_contributions, rolling_fair_value


def parse_args() -> argparse.Namespace:
    default_end = date.today()
    default_start = default_end - timedelta(days=3653)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=default_start.strftime("%Y%m%d"))
    parser.add_argument("--end", default=default_end.strftime("%Y%m%d"))
    parser.add_argument("--refresh", action="store_true", help="Refresh Bloomberg cache")
    parser.add_argument(
        "--cache",
        default="data/eurusd_bbg_history.csv",
        help="CSV cache for Bloomberg history",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for summaries, trades, and chart output",
    )
    return parser.parse_args()


def _score(row: pd.Series) -> float:
    if row["trades"] < 20 or not np.isfinite(row["sharpe"]):
        return -np.inf
    convergence = row.get("convergence_rate", np.nan)
    hit_rate = row.get("hit_rate", np.nan)
    drawdown_penalty = abs(row.get("max_drawdown_bp", 0.0)) / 1000.0
    convergence_value = float(convergence if np.isfinite(convergence) else 0.0)
    hit_value = float(hit_rate if np.isfinite(hit_rate) else 0.0)
    convergence_shortfall = max(0.0, 0.50 - convergence_value)
    return (
        float(row["sharpe"])
        + 1.50 * convergence_value
        + 0.50 * hit_value
        - 2.00 * convergence_shortfall
        - 0.15 * drawdown_penalty
    )


def _direction_from_z(z: float, threshold: float = 1.5) -> str:
    if not np.isfinite(z) or abs(z) < threshold:
        return "neutral"
    return "short EURUSD" if z > 0 else "long EURUSD"


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = load_or_fetch_history(
        cache_path=Path(args.cache),
        tickers=TICKERS,
        field=FIELD,
        start=args.start,
        end=args.end,
        refresh=args.refresh,
    )
    coverage_table(raw).to_csv(results_dir / "bbg_coverage.csv", index=False)

    features = build_feature_frame(raw, PAIR_TICKER, FEATURE_DEFS)
    feature_coverage(features).to_csv(results_dir / "feature_coverage.csv", index=False)

    summary_rows = []
    latest_rows = []
    model_frames: dict[tuple[str, int], pd.DataFrame] = {}
    trade_frames: dict[tuple[str, int, float], pd.DataFrame] = {}
    pnl_frames: dict[tuple[str, int, float], pd.Series] = {}

    for spec in MODEL_SPECS:
        for window in ROLLING_WINDOWS:
            model = rolling_fair_value(
                frame=features,
                target="log_spot",
                features=spec["features"],
                window=window,
                ridge_alpha=float(spec.get("ridge_alpha", 0.0)),
                z_window=Z_WINDOW,
            )
            model_key = (spec["name"], window)
            model_frames[model_key] = model

            latest = model.dropna(subset=["residual_z", "fair_value"]).tail(1)
            if not latest.empty:
                latest_date = latest.index[-1]
                latest_rows.append(
                    {
                        "model": spec["name"],
                        "window_days": window,
                        "date": latest_date.date().isoformat(),
                        "spot": float(features.loc[latest_date, "spot"]),
                        "fair_value": float(latest.loc[latest_date, "fair_value"]),
                        "trade_fair_value": float(
                            latest.loc[latest_date, "trade_fair_value"]
                        ),
                        "residual_pct": float(latest.loc[latest_date, "residual_pct"]),
                        "trade_gap_pct": float(latest.loc[latest_date, "trade_gap_pct"]),
                        "residual_z": float(latest.loc[latest_date, "residual_z"]),
                        "direction_at_1_5z": _direction_from_z(
                            float(latest.loc[latest_date, "residual_z"]), 1.5
                        ),
                        "features": ", ".join(spec["features"]),
                    }
                )

            for entry_z in ENTRY_ZS:
                for exit_z in EXIT_ZS:
                    if exit_z >= entry_z:
                        continue
                    metrics, trades, daily_pnl = fade_backtest(
                        model_frame=model,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        max_hold_days=MAX_HOLD_DAYS,
                        cost_bps_per_side=COST_BPS_PER_SIDE,
                        stop_z=STOP_Z,
                    )
                    row = {
                        "model": spec["name"],
                        "window_days": window,
                        "features": ", ".join(spec["features"]),
                        **metrics,
                    }
                    summary_rows.append(row)
                    trade_frames[(spec["name"], window, entry_z, exit_z)] = trades
                    pnl_frames[(spec["name"], window, entry_z, exit_z)] = daily_pnl

    summary = pd.DataFrame(summary_rows)
    summary["score"] = summary.apply(_score, axis=1)
    summary["passes_barometer"] = (
        (summary["trades"] >= 20)
        & (summary["sharpe"] > 0)
        & (summary["hit_rate"] >= 0.50)
        & (summary["convergence_rate"] >= 0.50)
        & (summary["avg_holding_days"] <= MAX_HOLD_DAYS)
    )
    summary = summary.sort_values(
        ["passes_barometer", "score", "sharpe", "convergence_rate"],
        ascending=False,
    ).reset_index(drop=True)
    summary.to_csv(results_dir / "backtest_summary.csv", index=False)

    latest_signals = pd.DataFrame(latest_rows).sort_values(
        "residual_z", key=lambda s: s.abs(), ascending=False
    )
    latest_signals.to_csv(results_dir / "latest_signals.csv", index=False)

    viable = summary[np.isfinite(summary["score"])].copy()
    if not viable.empty:
        best = viable.iloc[0]
    else:
        best = summary.iloc[0]

    best_key = (
        best["model"],
        int(best["window_days"]),
        float(best["entry_z"]),
        float(best["exit_z"]),
    )
    best_trades = trade_frames.get(best_key, pd.DataFrame())
    best_pnl = pnl_frames.get(best_key, pd.Series(dtype=float))
    best_trades.to_csv(results_dir / "best_trades.csv", index=False)
    best_pnl.to_csv(results_dir / "best_daily_pnl.csv", index_label="date")

    best_spec = next(spec for spec in MODEL_SPECS if spec["name"] == best["model"])
    best_model_frame = model_frames[(best["model"], int(best["window_days"]))]
    save_best_chart(
        best_model_frame,
        results_dir / "best_model_chart.html",
        title=f"EURUSD {best['model']} {int(best['window_days'])}d",
    )

    best_latest = latest_signals[
        (latest_signals["model"] == best["model"])
        & (latest_signals["window_days"] == int(best["window_days"]))
    ].iloc[0]
    write_index_page(
        output_path=results_dir / "index.html",
        chart_file="best_model_chart.html",
        best_row=best,
        latest_row=best_latest,
    )

    latest_date = best_model_frame.dropna(subset=["residual_z"]).index[-1]
    contrib = latest_contributions(
        frame=features,
        target="log_spot",
        features=best_spec["features"],
        asof=latest_date,
        window=int(best["window_days"]),
        ridge_alpha=float(best_spec.get("ridge_alpha", 0.0)),
    )
    contrib.to_csv(results_dir / "best_latest_contributions.csv", index=False)

    print("\nTop model variants")
    columns = [
        "model",
        "window_days",
        "entry_z",
        "exit_z",
        "passes_barometer",
        "trades",
        "sharpe",
        "hit_rate",
        "convergence_rate",
        "avg_holding_days",
        "net_pnl_bp_total",
        "max_drawdown_bp",
        "score",
    ]
    print(summary[columns].head(12).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\nLatest signals")
    latest_cols = [
        "model",
        "window_days",
        "date",
        "spot",
        "trade_fair_value",
        "trade_gap_pct",
        "residual_z",
        "direction_at_1_5z",
    ]
    print(latest_signals[latest_cols].head(12).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nWrote outputs to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
