"""Reporting helpers for FX valuation dashboard."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def save_best_chart(model_frame: pd.DataFrame, output_path: Path, title: str, spot_label: str = "Spot") -> None:
    """Save an interactive spot/fair-value/z-score chart if plotly is available."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart = model_frame.dropna(subset=["log_spot", "log_fair_value"]).copy()
    if chart.empty:
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=chart.index, y=np.exp(chart["log_spot"]), name="Spot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart.index, y=chart["fair_value"], name="Raw FV"), row=1, col=1)
    if "trade_fair_value" in chart:
        fig.add_trace(go.Scatter(x=chart.index, y=chart["trade_fair_value"], name="Trade FV", line={"dash": "dash"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart.index, y=chart["residual_z"], name="Residual z"), row=2, col=1)
    for level in (-2, -1.5, 0, 1.5, 2):
        fig.add_hline(y=level, line_dash="dot", line_width=1, row=2, col=1)
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        height=760,
        margin={"l": 70, "r": 35, "t": 105, "b": 55},
        showlegend=False,
    )
    fig.write_html(output_path)


def write_index_page(output_path: Path, chart_file: str, best_row: pd.Series, latest_row: pd.Series) -> None:
    """Write a simple single-pair landing page."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>EURUSD Practical Fair Value</title></head>
<body><h1>EURUSD Practical Fair Value</h1><iframe src=\"{chart_file}\" style=\"width:100%;height:800px;border:0\"></iframe></body></html>""",
        encoding="utf-8",
    )


def write_dashboard_page(output_path: Path, rows: pd.DataFrame, generated_at: str) -> None:
    """Write the multi-pair tabbed dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows.copy()

    def fmt(value: float, digits: int = 4) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):,.{digits}f}"

    def text(value) -> str:
        if pd.isna(value):
            return ""
        return escape(str(value))

    def signal_class(signal: str) -> str:
        lower = str(signal).lower()
        if "long" in lower:
            return "long"
        if "short" in lower:
            return "short"
        return "neutral"

    buttons, panels, opportunity_rows = [], [], []
    legend = """<div class=\"chart-legend\" aria-label=\"Chart legend\"><span><i class=\"swatch spot\"></i>Spot</span><span><i class=\"swatch raw\"></i>Raw FV</span><span><i class=\"swatch trade\"></i>Trade FV</span><span><i class=\"swatch zed\"></i>Residual z</span></div>"""
    for idx, row in rows.iterrows():
        active = " active" if idx == 0 else ""
        pair = row["pair"]
        signal = str(row.get("display_signal", row["direction_at_1_5z"]))
        sig_class = signal_class(signal)
        buttons.append(f'<button class="tab{active}" data-target="{pair}"><span>{pair}</span><strong class="{sig_class}">{signal}</strong></button>')
        opportunity_rows.append(
            f"<tr><td>{pair}</td><td class=\"{sig_class}\">{signal}</td><td>{fmt(row['residual_z'], 2)}z</td><td>{fmt(row['spot'])}</td><td>{fmt(row['trade_fair_value'])}</td><td>{fmt(row['trade_gap_pct'], 2)}%</td><td>{row['model']} {int(row['window_days'])}d</td></tr>"
        )
        panels.append(
            f"""<section class="panel{active}" id="{pair}">
<div class="metrics"><div><span>Spot</span><strong>{fmt(row['spot'])}</strong></div><div><span>Trade FV</span><strong>{fmt(row['trade_fair_value'])}</strong></div><div><span>Gap</span><strong>{fmt(row['trade_gap_pct'], 2)}%</strong></div><div><span>Z-score</span><strong>{fmt(row['residual_z'], 2)}z</strong></div><div><span>Best model</span><strong>{row['model']} {int(row['window_days'])}d</strong></div><div><span>Sharpe</span><strong>{fmt(row['sharpe'], 2)}</strong></div><div><span>Hit rate</span><strong>{fmt(row['hit_rate'] * 100, 1)}%</strong></div><div><span>Convergence</span><strong>{fmt(row['convergence_rate'] * 100, 1)}%</strong></div></div>
{legend}<iframe src="{row['chart_file']}" title="{pair} chart"></iframe>
<section class="explain"><h2>Model Used</h2><p>{text(row.get('model_explanation', ''))}</p><div class="takeaway {sig_class}">{text(row.get('trade_takeaway', ''))}</div>{row.get('driver_table_html', '')}</section></section>"""
        )

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FX Practical Fair Value Dashboard</title>
<style>
:root {{ font-family: Arial, Helvetica, sans-serif; color: #15171a; background: #f5f6f8; }} body {{ margin:0; background:#f5f6f8; }} header {{ background:#fff; border-bottom:1px solid #d9dde3; padding:18px 22px 14px; }} .header-row {{ display:flex; justify-content:space-between; gap:14px; }} h1 {{ margin:0 0 4px; font-size:22px; }} .sub,#refreshStatus {{ color:#5d6673; font-size:13px; }} .refresh-actions {{ display:flex; flex-direction:column; align-items:flex-end; gap:4px; min-width:190px; }} #refreshButton {{ border:1px solid #1f5f99; border-radius:6px; background:#fff; color:#174b79; cursor:pointer; font-weight:700; padding:8px 11px; }} #refreshButton:disabled {{ cursor:wait; opacity:.65; }}
.tabs {{ display:flex; gap:8px; overflow-x:auto; padding:12px 14px; border-bottom:1px solid #d9dde3; background:#fff; }} .tab {{ flex:0 0 auto; min-width:118px; border:1px solid #cfd5dd; border-radius:6px; background:#f9fafb; padding:9px 10px; text-align:left; cursor:pointer; }} .tab.active {{ border-color:#1f5f99; background:#eef6fd; }} .tab span,.tab strong {{ display:block; }} .tab strong {{ margin-top:3px; font-size:12px; }} .long {{ color:#087443; }} .short {{ color:#a33b22; }} .neutral {{ color:#5d6673; }} main {{ padding:14px; }}
.opps,.explain,.chart-legend,.metrics div {{ border:1px solid #d9dde3; border-radius:6px; background:#fff; }} .opps {{ margin-bottom:14px; overflow:auto; }} table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ border-bottom:1px solid #e4e7eb; padding:8px 10px; text-align:left; white-space:nowrap; }} th {{ color:#5d6673; background:#fbfcfd; }} .panel {{ display:none; }} .panel.active {{ display:block; }} .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); gap:10px; margin-bottom:12px; }} .metrics div {{ padding:10px 12px; }} .metrics span {{ display:block; color:#5d6673; font-size:12px; margin-bottom:4px; }} .metrics strong {{ font-size:16px; }}
.chart-legend {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; padding:8px 10px; margin-bottom:8px; font-size:13px; }} .chart-legend span {{ display:inline-flex; align-items:center; gap:6px; white-space:nowrap; }} .swatch {{ display:inline-block; width:22px; height:0; border-top:2px solid #636efa; }} .swatch.raw {{ border-color:#ef553b; }} .swatch.trade {{ border-color:#00cc96; border-top-style:dashed; }} .swatch.zed {{ border-color:#ab63fa; }} iframe {{ width:100%; min-height:760px; border:1px solid #d9dde3; border-radius:6px; background:#fff; }} .explain {{ margin-top:12px; padding:14px 16px; }} .explain h2 {{ font-size:17px; margin:0 0 8px; }} .explain p {{ margin:0 0 10px; color:#303741; line-height:1.45; }} .takeaway {{ border:1px solid #d9dde3; border-radius:6px; background:#fbfcfd; padding:9px 10px; margin-bottom:12px; font-weight:700; }} .driver-table {{ max-width:760px; overflow:auto; }} .driver-table table {{ font-size:12px; }} .driver-table caption {{ text-align:left; color:#5d6673; font-weight:700; padding:0 0 6px; }}
</style></head><body><header><div class="header-row"><div><h1>FX Practical Fair Value Dashboard</h1><div class="sub">Last refresh: {generated_at}. Signal uses spot versus trade fair value and each pair's selected entry threshold.</div></div><div class="refresh-actions"><button id="refreshButton" type="button">Refresh Bloomberg Data</button><span id="refreshStatus">Ready</span></div></div></header><nav class="tabs">{''.join(buttons)}</nav><main><section class="opps"><table><thead><tr><th>Pair</th><th>Signal</th><th>Z</th><th>Spot</th><th>Trade FV</th><th>Gap</th><th>Model</th></tr></thead><tbody>{''.join(opportunity_rows)}</tbody></table></section>{''.join(panels)}</main>
<script>const tabs=document.querySelectorAll('.tab');const panels=document.querySelectorAll('.panel');tabs.forEach((tab)=>{{tab.addEventListener('click',()=>{{tabs.forEach((t)=>t.classList.remove('active'));panels.forEach((p)=>p.classList.remove('active'));tab.classList.add('active');document.getElementById(tab.dataset.target).classList.add('active');}});}});const refreshButton=document.getElementById('refreshButton');const refreshStatus=document.getElementById('refreshStatus');refreshButton.addEventListener('click',async()=>{{refreshButton.disabled=true;refreshStatus.textContent='Refreshing Bloomberg data...';try{{const response=await fetch('/refresh',{{method:'POST'}});const data=await response.json();if(!response.ok||!data.ok){{throw new Error(data.error||'Refresh failed');}}refreshStatus.textContent=`Last refresh completed ${{data.completed_at}}`;window.location.reload();}}catch(error){{refreshStatus.textContent=error.message||'Refresh failed';refreshButton.disabled=false;}}}});</script></body></html>"""
    output_path.write_text(html, encoding="utf-8")
