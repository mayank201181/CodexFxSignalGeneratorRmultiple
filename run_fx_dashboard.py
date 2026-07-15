"""Run the multi-pair FX practical fair-value dashboard."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from fx_value_models.backtest import fade_backtest
from fx_value_models.bbg import coverage_table, load_or_fetch_history
from fx_value_models.features import build_feature_frame, feature_coverage
from fx_value_models.pair_config import (
    COST_BPS_PER_SIDE,
    ENTRY_ZS,
    EXIT_ZS,
    FIELD,
    MAX_HOLD_DAYS,
    PAIR_CONFIGS,
    ROLLING_WINDOWS,
    STOP_Z,
    Z_WINDOW,
    all_tickers,
)
from fx_value_models.report import save_best_chart, write_dashboard_page
from fx_value_models.rolling import latest_contributions, rolling_fair_value


FEATURE_LABELS = {
    "rate_2y_diff": "2y yield differential",
    "rate_10y_diff": "10y yield differential",
    "cesi_diff": "economic surprise differential",
    "equity_ratio": "relative equity performance",
    "vix_log": "VIX / global risk",
    "move_log": "MOVE / rates volatility",
    "fx_1m_vol": "1m FX implied volatility",
    "fx_1m_rr": "1m 25-delta risk reversal",
    "brent_log": "Brent oil",
    "ttf_gas_log": "European gas",
    "gold_log": "gold",
}


def parse_args() -> argparse.Namespace:
    default_end = date.today()
    default_start = default_end - timedelta(days=3653)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=default_start.strftime("%Y%m%d"))
    parser.add_argument("--end", default=default_end.strftime("%Y%m%d"))
    parser.add_argument("--refresh", action="store_true", help="Refresh Bloomberg cache")
    parser.add_argument("--cache", default="data/fx_dashboard_bbg_history.csv")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def score(row: pd.Series) -> float:
    if row["trades"] < 20 or not np.isfinite(row["sharpe"]):
        return -np.inf
    convergence = row.get("convergence_rate", np.nan)
    hit_rate = row.get("hit_rate", np.nan)
    convergence_value = float(convergence if np.isfinite(convergence) else 0.0)
    hit_value = float(hit_rate if np.isfinite(hit_rate) else 0.0)
    convergence_shortfall = max(0.0, 0.50 - convergence_value)
    drawdown_penalty = abs(row.get("max_drawdown_bp", 0.0)) / 1000.0
    return (
        float(row["sharpe"])
        + 1.50 * convergence_value
        + 0.50 * hit_value
        - 2.00 * convergence_shortfall
        - 0.15 * drawdown_penalty
    )


def direction_from_z(pair: str, z: float, threshold: float = 1.5) -> str:
    if not np.isfinite(z) or abs(z) < threshold:
        return "neutral"
    return f"short {pair}" if z > 0 else f"long {pair}"


def lean_from_z(pair: str, z: float) -> str:
    if not np.isfinite(z) or abs(z) < 0.05:
        return "roughly fair"
    return f"leans short {pair}" if z > 0 else f"leans long {pair}"


def display_lean_from_z(z: float) -> str:
    if not np.isfinite(z) or abs(z) < 0.05:
        return "roughly fair"
    return "leans short" if z > 0 else "leans long"


def build_model_explanation(pair_name: str, best: pd.Series) -> str:
    features = [f.strip() for f in str(best["features"]).split(",") if f.strip()]
    driver_text = ", ".join(FEATURE_LABELS.get(feature, feature) for feature in features)
    return (
        f"The selected model for {pair_name} is {best['model']} using a "
        f"{int(best['window_days'])}-business-day rolling regression of log spot "
        f"against: {driver_text}. Raw fair value is the fitted value from those "
        f"drivers. Trade fair value adjusts raw fair value by the model's recent "
        f"residual bias, and the trading signal is the z-score of spot versus "
        f"trade fair value. The backtest enters at {float(best['entry_z']):.2f}z, "
        f"exits at {float(best['exit_z']):.2f}z, stops at 3.00z, and times out "
        f"after 20 business days."
    )


def build_trade_takeaway(pair_name: str, z_value: float, best: pd.Series) -> str:
    entry_z = float(best["entry_z"])
    if not np.isfinite(z_value):
        return "No current signal because the latest z-score is unavailable."
    lean = lean_from_z(pair_name, z_value)
    if abs(z_value) >= entry_z:
        action = f"short {pair_name}" if z_value > 0 else f"long {pair_name}"
        return (
            f"Active model signal: {action}. Current z-score is {z_value:.2f}z, "
            f"beyond the model's {entry_z:.2f}z entry threshold."
        )
    return (
        f"No active trade signal. Current z-score is {z_value:.2f}z versus the "
        f"model's {entry_z:.2f}z entry threshold, so it only {lean}."
    )


def build_driver_table(contrib: pd.DataFrame) -> str:
    if contrib.empty:
        return ""
    rows = []
    for _, row in contrib.iterrows():
        feature = str(row["feature"])
        rows.append(
            "<tr>"
            f"<td>{escape(FEATURE_LABELS.get(feature, feature))}</td>"
            f"<td>{float(row['standardized_value']):+.2f}</td>"
            f"<td>{float(row['beta_on_standardized_feature']):+.4f}</td>"
            f"<td>{float(row['log_fv_contribution']):+.4f}</td>"
            "</tr>"
        )
    return (
        '<div class="driver-table">'
        "<table>"
        "<caption>Latest standardized driver contributions to raw fair value</caption>"
        "<thead><tr><th>Driver</th><th>Current z</th><th>Beta</th>"
        "<th>Log FV contribution</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def run_pair(pair, raw: pd.DataFrame, results_dir: Path) -> dict:
    pair_dir = results_dir / pair.name.lower()
    pair_dir.mkdir(parents=True, exist_ok=True)

    features = build_feature_frame(raw[pair.tickers], pair.spot_ticker, pair.feature_defs)
    feature_coverage(features).to_csv(pair_dir / "feature_coverage.csv", index=False)

    summary_rows = []
    latest_rows = []
    model_frames: dict[tuple[str, int], pd.DataFrame] = {}
    trade_frames: dict[tuple[str, int, float, float], pd.DataFrame] = {}
    pnl_frames: dict[tuple[str, int, float, float], pd.Series] = {}

    for spec in pair.model_specs:
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
                z_value = float(latest.loc[latest_date, "residual_z"])
                latest_rows.append(
                    {
                        "pair": pair.name,
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
                        "residual_z": z_value,
                        "direction_at_1_5z": direction_from_z(pair.name, z_value, 1.5),
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
                    summary_rows.append(
                        {
                            "pair": pair.name,
                            "model": spec["name"],
                            "window_days": window,
                            "features": ", ".join(spec["features"]),
                            **metrics,
                        }
                    )
                    trade_frames[(spec["name"], window, entry_z, exit_z)] = trades
                    pnl_frames[(spec["name"], window, entry_z, exit_z)] = daily_pnl

    summary = pd.DataFrame(summary_rows)
    summary["score"] = summary.apply(score, axis=1)
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
    summary.to_csv(pair_dir / "backtest_summary.csv", index=False)

    latest_signals = pd.DataFrame(latest_rows).sort_values(
        "residual_z", key=lambda s: s.abs(), ascending=False
    )
    latest_signals.to_csv(pair_dir / "latest_signals.csv", index=False)

    viable = summary[np.isfinite(summary["score"])].copy()
    best = viable.iloc[0] if not viable.empty else summary.iloc[0]
    best_key = (
        best["model"],
        int(best["window_days"]),
        float(best["entry_z"]),
        float(best["exit_z"]),
    )
    trade_frames.get(best_key, pd.DataFrame()).to_csv(pair_dir / "best_trades.csv", index=False)
    pnl_frames.get(best_key, pd.Series(dtype=float)).to_csv(
        pair_dir / "best_daily_pnl.csv", index_label="date"
    )

    best_spec = next(spec for spec in pair.model_specs if spec["name"] == best["model"])
    best_model_frame = model_frames[(best["model"], int(best["window_days"]))]
    chart_name = f"{pair.name.lower()}_best_model_chart.html"
    save_best_chart(
        best_model_frame,
        results_dir / chart_name,
        title=f"{pair.name} {best['model']} {int(best['window_days'])}d",
        spot_label=pair.name,
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
    contrib.to_csv(pair_dir / "best_latest_contributions.csv", index=False)

    best_latest = latest_signals[
        (latest_signals["model"] == best["model"])
        & (latest_signals["window_days"] == int(best["window_days"]))
    ].iloc[0]
    z_value = float(best_latest["residual_z"])
    return {
        **best.to_dict(),
        **{
            "date": best_latest["date"],
            "spot": best_latest["spot"],
            "fair_value": best_latest["fair_value"],
            "trade_fair_value": best_latest["trade_fair_value"],
            "residual_pct": best_latest["residual_pct"],
            "trade_gap_pct": best_latest["trade_gap_pct"],
            "residual_z": z_value,
            "direction_at_1_5z": direction_from_z(
                pair.name, z_value, float(best["entry_z"])
            ),
            "display_signal": display_lean_from_z(z_value),
            "chart_file": chart_name,
            "model_explanation": build_model_explanation(pair.name, best),
            "trade_takeaway": build_trade_takeaway(pair.name, z_value, best),
            "driver_table_html": build_driver_table(contrib),
        },
    }


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = load_or_fetch_history(
        cache_path=Path(args.cache),
        tickers=all_tickers(),
        field=FIELD,
        start=args.start,
        end=args.end,
        refresh=args.refresh,
    )
    coverage_table(raw).to_csv(results_dir / "fx_dashboard_bbg_coverage.csv", index=False)

    dashboard_rows = []
    for pair in PAIR_CONFIGS:
        print(f"\nRunning {pair.name}...")
        dashboard_rows.append(run_pair(pair, raw, results_dir))

    dashboard = pd.DataFrame(dashboard_rows)
    dashboard = dashboard.sort_values("residual_z", key=lambda s: s.abs(), ascending=False)
    dashboard.to_csv(results_dir / "fx_dashboard_summary.csv", index=False)
    write_dashboard_page(
        output_path=results_dir / "index.html",
        rows=dashboard,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    print("\nDashboard summary")
    cols = [
        "pair",
        "direction_at_1_5z",
        "display_signal",
        "residual_z",
        "spot",
        "trade_fair_value",
        "trade_gap_pct",
        "model",
        "window_days",
        "sharpe",
        "hit_rate",
        "convergence_rate",
    ]
    print(dashboard[cols].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nWrote dashboard to: {(results_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
