"""Outcome tracking dashboard — performance KPIs, outcomes table, score correlation.

Renders the feedback loop: did our recommendations actually work?
All data comes from the OutcomeTracker backend (core.db.outcome_tracker).
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _fmt_pct(val: Any, show_sign: bool = True) -> str:
    try:
        v = float(val)
        sign = "+" if v > 0 and show_sign else ""
        return f"{sign}{v:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_price(val: Any) -> str:
    try:
        return f"${float(val):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _return_color(val: Any) -> str:
    try:
        v = float(val)
        if v > 0.5:
            return "#22c55e"  # green
        if v < -0.5:
            return "#ef4444"  # red
        return "var(--ss-text-muted)"
    except (TypeError, ValueError):
        return "var(--ss-text-muted)"


def _status_badge(status: str) -> str:
    colors = {
        "complete": ("#22c55e", "Complete"),
        "partial": ("#f59e0b", "Tracking"),
        "pending": ("#94a3b8", "Pending"),
    }
    color, label = colors.get(str(status).lower(), ("#94a3b8", status))
    return (
        f'<span style="display:inline-block; padding:2px 8px; border-radius:10px; '
        f'font-size:0.7rem; font-weight:600; color:white; background:{color};">'
        f'{label}</span>'
    )


# ──────────────────────────────────────────────────────────────
# KPI Summary Cards
# ──────────────────────────────────────────────────────────────

def render_performance_kpis(perf: Dict[str, Any]) -> str:
    """Render performance summary as HTML KPI cards.

    Args:
        perf: Dict from OutcomeTracker.get_performance_summary()

    Returns:
        HTML string with 2 rows of 3 metric cards.
    """
    n = perf.get("n_completed", 0)

    if n == 0:
        return ""

    win_rate = perf.get("win_rate")
    avg_ret = perf.get("avg_return_20d")
    excess = perf.get("avg_excess_return")
    hit_target = perf.get("hit_target_rate")
    hit_stop = perf.get("hit_stop_rate")
    avg_dd = perf.get("avg_max_drawdown")

    def _card(label: str, value: str, color: str = "var(--ss-text-primary)") -> str:
        return f"""
        <div style="
            background:var(--ss-bg-card, #fff); border:1px solid var(--ss-border, #e2e8f0);
            border-radius:var(--ss-radius-md, 8px); padding:16px; text-align:center;
        ">
            <div style="font-size:0.72rem; font-weight:600; color:var(--ss-text-muted, #64748b);
                         text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
                {label}
            </div>
            <div style="font-size:1.5rem; font-weight:700; color:{color};">
                {value}
            </div>
        </div>"""

    wr_color = "#22c55e" if win_rate and win_rate > 0.5 else "#ef4444" if win_rate else "var(--ss-text-primary)"
    ret_color = "#22c55e" if avg_ret and avg_ret > 0 else "#ef4444" if avg_ret else "var(--ss-text-primary)"
    exc_color = "#22c55e" if excess and excess > 0 else "#ef4444" if excess else "var(--ss-text-primary)"

    row1 = f"""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:12px; direction:ltr;">
        {_card("Win Rate", _fmt_pct(win_rate * 100 if win_rate else None, show_sign=False), wr_color)}
        {_card("Avg Return (20d)", _fmt_pct(avg_ret), ret_color)}
        {_card("vs SPY", _fmt_pct(excess), exc_color)}
    </div>"""

    row2 = f"""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; direction:ltr;">
        {_card("Hit Target", _fmt_pct(hit_target * 100 if hit_target else None, show_sign=False))}
        {_card("Hit Stop", _fmt_pct(hit_stop * 100 if hit_stop else None, show_sign=False))}
        {_card("Tracked", str(n))}
    </div>"""

    return row1 + row2


# ──────────────────────────────────────────────────────────────
# Recent Outcomes Table
# ──────────────────────────────────────────────────────────────

def render_outcomes_table(outcomes_df: pd.DataFrame) -> str:
    """Render a table of recent outcomes as HTML.

    Args:
        outcomes_df: DataFrame with columns from outcomes + recommendations tables.
            Expected columns: ticker, entry_date, entry_price, return_20d,
            status, spy_return_20d, hit_target, hit_stop

    Returns:
        HTML string with a styled table.
    """
    if outcomes_df is None or outcomes_df.empty:
        return ""

    rows_html = []
    for _, row in outcomes_df.iterrows():
        ticker = str(row.get("ticker", ""))
        entry_date = row.get("entry_date", "")
        entry_price = row.get("entry_price", None)
        ret_20d = row.get("return_20d", None)
        status = str(row.get("status", "pending"))
        spy_ret = row.get("spy_return_20d", None)
        hit_target = row.get("hit_target", False)
        hit_stop = row.get("hit_stop", False)

        # Format entry date
        if isinstance(entry_date, (date, pd.Timestamp)):
            entry_str = entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, 'strftime') else str(entry_date)[:10]
        else:
            entry_str = str(entry_date)[:10]

        # Return formatting
        ret_str = _fmt_pct(ret_20d) if ret_20d is not None else "—"
        ret_color = _return_color(ret_20d)

        # Target/stop icons
        icons = ""
        if hit_target:
            icons += " 🎯"
        if hit_stop:
            icons += " 🛑"

        # SPY comparison
        spy_str = _fmt_pct(spy_ret) if spy_ret is not None else "—"

        rows_html.append(f"""
        <tr style="border-bottom:1px solid var(--ss-border, #e2e8f0);">
            <td style="padding:8px 10px; font-weight:600;">{ticker}</td>
            <td style="padding:8px 10px; font-size:0.82rem;">{entry_str}</td>
            <td style="padding:8px 10px;">{_fmt_price(entry_price)}</td>
            <td style="padding:8px 10px; font-weight:600; color:{ret_color};">{ret_str}{icons}</td>
            <td style="padding:8px 10px;">{_status_badge(status)}</td>
            <td style="padding:8px 10px; font-size:0.82rem;">{spy_str}</td>
        </tr>""")

    return f"""
    <div style="overflow-x:auto; direction:ltr;">
    <table style="
        width:100%; border-collapse:collapse; font-size:0.85rem;
        background:var(--ss-bg-card, #fff); border:1px solid var(--ss-border, #e2e8f0);
        border-radius:var(--ss-radius-md, 8px); overflow:hidden;
    ">
        <thead>
            <tr style="background:var(--ss-bg-badge, #f1f5f9); border-bottom:2px solid var(--ss-border, #e2e8f0);">
                <th style="padding:10px; text-align:left; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Ticker</th>
                <th style="padding:10px; text-align:left; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Entry Date</th>
                <th style="padding:10px; text-align:left; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Entry Price</th>
                <th style="padding:10px; text-align:left; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Return 20d</th>
                <th style="padding:10px; text-align:left; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Status</th>
                <th style="padding:10px; text-align:left; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">vs SPY</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    </div>"""


# ──────────────────────────────────────────────────────────────
# Score Effectiveness (bar chart as HTML)
# ──────────────────────────────────────────────────────────────

def render_score_correlation(corr_df: pd.DataFrame) -> str:
    """Render score-vs-outcome correlation as HTML bar charts.

    Shows average return by score decile for each component.
    Answers: "Which scoring component best predicts actual returns?"

    Args:
        corr_df: DataFrame from OutcomeTracker.get_score_vs_outcome()

    Returns:
        HTML string with bar charts per component.
    """
    if corr_df is None or corr_df.empty:
        return ""

    component_labels = {
        "final_score": "Final Score",
        "tech_score": "Technical Score",
        "fundamental_score": "Fundamental Score",
        "ml_prob": "ML Probability",
    }

    sections = []
    for comp in ["final_score", "tech_score", "fundamental_score", "ml_prob"]:
        comp_data = corr_df[corr_df["component"] == comp].sort_values("decile")
        if comp_data.empty:
            continue

        label = component_labels.get(comp, comp)
        max_abs = max(abs(comp_data["avg_return_20d"].max()), abs(comp_data["avg_return_20d"].min()), 1)

        bars_html = []
        for _, row in comp_data.iterrows():
            ret = float(row["avg_return_20d"])
            wr = float(row.get("win_rate", 0)) * 100
            n = int(row.get("n", 0))
            dec = int(row["decile"]) + 1  # 1-indexed for display

            # Bar width proportional to return magnitude
            width = min(abs(ret) / max_abs * 100, 100)
            color = "#22c55e" if ret > 0 else "#ef4444"
            direction = "right" if ret >= 0 else "left"

            bars_html.append(f"""
            <div style="display:grid; grid-template-columns:40px 1fr 80px; gap:8px; align-items:center; margin:3px 0;">
                <div style="font-size:0.75rem; color:var(--ss-text-muted); text-align:right;">D{dec}</div>
                <div style="position:relative; height:20px; background:var(--ss-bg-badge, #f1f5f9); border-radius:4px; overflow:hidden;">
                    <div style="
                        position:absolute; {'right:50%' if ret < 0 else 'left:50%'}; top:0; bottom:0;
                        width:{width/2}%; background:{color}; opacity:0.7; border-radius:4px;
                    "></div>
                    <div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:var(--ss-border);"></div>
                </div>
                <div style="font-size:0.72rem; font-weight:600; color:{color};">
                    {'+' if ret > 0 else ''}{ret:.1f}% <span style="color:var(--ss-text-muted); font-weight:400;">({wr:.0f}%W)</span>
                </div>
            </div>""")

        sections.append(f"""
        <div style="margin-bottom:16px;">
            <div style="font-size:0.78rem; font-weight:700; color:var(--ss-text-primary);
                         margin-bottom:6px; text-transform:uppercase; letter-spacing:0.04em;">
                {label}
            </div>
            {''.join(bars_html)}
        </div>""")

    if not sections:
        return ""

    return f"""
    <div style="direction:ltr; background:var(--ss-bg-card, #fff); border:1px solid var(--ss-border, #e2e8f0);
                border-radius:var(--ss-radius-md, 8px); padding:16px;">
        <div style="font-size:0.7rem; color:var(--ss-text-muted); margin-bottom:12px;">
            Avg 20-day return by score decile (D1=lowest, D10=highest). If the system works,
            higher deciles should show higher returns.
        </div>
        {''.join(sections)}
    </div>"""


# ──────────────────────────────────────────────────────────────
# Pending State
# ──────────────────────────────────────────────────────────────

def render_pending_state(n_pending: int, n_partial: int = 0) -> str:
    """Render a message for when no outcomes are completed yet."""
    tracking = n_pending + n_partial
    if tracking == 0:
        return """
        <div style="
            text-align:center; padding:24px 16px;
            color:var(--ss-text-muted); font-size:0.85rem;
            background:var(--ss-bg-card); border:1px solid var(--ss-border);
            border-radius:var(--ss-radius-md); direction:ltr;
        ">
            📊 No recommendations tracked yet. Run a scan to start tracking outcomes.
        </div>"""

    return f"""
    <div style="
        text-align:center; padding:24px 16px;
        color:var(--ss-text-muted); font-size:0.85rem;
        background:var(--ss-bg-card); border:1px solid var(--ss-border);
        border-radius:var(--ss-radius-md); direction:ltr;
    ">
        📊 Performance tracking is active. Results will appear after recommendations
        complete their 20-day evaluation period.<br>
        <span style="font-size:0.95rem; font-weight:600; color:var(--ss-text-primary); margin-top:8px; display:inline-block;">
            Currently tracking {tracking} recommendation{'s' if tracking != 1 else ''}
        </span>
        {f'<br><span style="font-size:0.78rem;">({n_partial} partially evaluated, {n_pending} pending)</span>' if n_partial > 0 else ''}
    </div>"""
