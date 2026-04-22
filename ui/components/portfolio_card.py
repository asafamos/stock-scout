"""Portfolio position card renderer — compact HTML cards for open/closed positions."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

import pandas as pd


def _fmt_pct(val: Any) -> str:
    """Format a percentage value."""
    try:
        v = float(val)
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_price(val: Any) -> str:
    """Format a price value."""
    try:
        return f"${float(val):.2f}"
    except (TypeError, ValueError):
        return "—"


def _return_class(val: Any) -> str:
    """Return CSS class based on return value."""
    try:
        v = float(val)
        if v > 0.5:
            return "positive"
        if v < -0.5:
            return "negative"
        return "neutral"
    except (TypeError, ValueError):
        return "neutral"


def _days_held(entry_date: Any) -> int:
    """Calculate days held from entry date."""
    try:
        if isinstance(entry_date, date):
            ed = entry_date
        else:
            ed = pd.Timestamp(entry_date).date()
        return (date.today() - ed).days
    except Exception:
        return 0


def render_open_position_card(pos: Dict[str, Any]) -> str:
    """Render a compact card for an open portfolio position.

    Args:
        pos: Dict/Series with position fields from portfolio_positions table.

    Returns:
        HTML string for the position card.
    """
    ticker = str(pos.get("ticker", "???"))
    entry_price = pos.get("entry_price", 0)
    current_price = pos.get("current_price", entry_price)
    target_price = pos.get("target_price")
    stop_price = pos.get("stop_price")
    ret_pct = pos.get("current_return_pct", 0)
    entry_dt = pos.get("entry_date")
    target_dt = pos.get("target_date")
    days = _days_held(entry_dt)
    holding_days = pos.get("holding_days", 20)
    risk_class = str(pos.get("risk_class", ""))
    sector = str(pos.get("sector", ""))

    ret_cls = _return_class(ret_pct)

    # Progress towards target or holding period
    progress_pct = min(100, int((days / max(holding_days, 1)) * 100))

    return f"""
<div class="ss-portfolio-card">
  <div class="pf-header">
    <div>
      <span class="pf-ticker">{ticker}</span>
      <span style="font-size:0.68rem; color:var(--ss-text-muted); margin-left:8px;">{sector}</span>
    </div>
    <span class="pf-return {ret_cls}">{_fmt_pct(ret_pct)}</span>
  </div>
  <div class="pf-metrics">
    <div>
      <div class="pf-metric-label">Entry</div>
      <div class="pf-metric-value">{_fmt_price(entry_price)}</div>
      <div class="pf-metric-sub">{str(entry_dt)[:10] if entry_dt else '—'}</div>
    </div>
    <div>
      <div class="pf-metric-label">Current</div>
      <div class="pf-metric-value">{_fmt_price(current_price)}</div>
    </div>
    <div>
      <div class="pf-metric-label">Target</div>
      <div class="pf-metric-value">{_fmt_price(target_price)}</div>
      <div class="pf-metric-sub">{str(target_dt)[:10] if target_dt else '—'}</div>
    </div>
    <div>
      <div class="pf-metric-label">Stop</div>
      <div class="pf-metric-value">{_fmt_price(stop_price)}</div>
    </div>
    <div>
      <div class="pf-metric-label">Days</div>
      <div class="pf-metric-value">{days}/{holding_days}</div>
    </div>
  </div>
  <div style="margin-top:8px;">
    <div style="height:4px; background:var(--ss-bar-bg); border-radius:2px; overflow:hidden;">
      <div style="width:{progress_pct}%; height:100%; background:linear-gradient(90deg, #3b82f6, #60a5fa); border-radius:2px;"></div>
    </div>
  </div>
</div>
"""


def render_closed_position_card(pos: Dict[str, Any]) -> str:
    """Render a compact card for a closed portfolio position.

    Args:
        pos: Dict/Series with position fields.

    Returns:
        HTML string.
    """
    ticker = str(pos.get("ticker", "???"))
    entry_price = pos.get("entry_price", 0)
    exit_price = pos.get("exit_price", 0)
    ret_pct = pos.get("realized_return_pct", 0)
    exit_reason = str(pos.get("exit_reason", "manual"))
    correct = pos.get("prediction_correct", False)
    entry_dt = pos.get("entry_date")
    exit_dt = pos.get("exit_date")

    ret_cls = _return_class(ret_pct)

    # Exit reason icons
    reason_icons = {
        "target": "🎯",
        "stop": "🛑",
        "expiry": "⏰",
        "manual": "✋",
    }
    reason_icon = reason_icons.get(exit_reason, "📌")
    correct_icon = "✅" if correct else "❌"

    return f"""
<div class="ss-portfolio-card" style="opacity:0.85;">
  <div class="pf-header">
    <div>
      <span class="pf-ticker">{ticker}</span>
      <span style="font-size:0.68rem; margin-left:6px;">{reason_icon} {exit_reason}</span>
    </div>
    <div style="display:flex; align-items:center; gap:8px;">
      <span style="font-size:0.72rem;">{correct_icon}</span>
      <span class="pf-return {ret_cls}">{_fmt_pct(ret_pct)}</span>
    </div>
  </div>
  <div class="pf-metrics">
    <div>
      <div class="pf-metric-label">Entry</div>
      <div class="pf-metric-value">{_fmt_price(entry_price)}</div>
    </div>
    <div>
      <div class="pf-metric-label">Exit</div>
      <div class="pf-metric-value">{_fmt_price(exit_price)}</div>
    </div>
    <div>
      <div class="pf-metric-label">Entry Date</div>
      <div class="pf-metric-value">{str(entry_dt)[:10] if entry_dt else '—'}</div>
    </div>
    <div>
      <div class="pf-metric-label">Exit Date</div>
      <div class="pf-metric-value">{str(exit_dt)[:10] if exit_dt else '—'}</div>
    </div>
  </div>
</div>
"""


def render_portfolio_sidebar_summary(stats: Dict[str, Any]) -> str:
    """Render compact portfolio summary for sidebar.

    Args:
        stats: Dict from PortfolioManager.get_portfolio_stats()

    Returns:
        HTML string for sidebar summary card.
    """
    open_count = stats.get("open_count", 0)
    closed_count = stats.get("closed_count", 0)
    total_return = stats.get("total_return_pct", 0)
    win_rate = stats.get("win_rate", 0)
    avg_return = stats.get("avg_return", 0)
    # New professional metrics
    weighted_avg = stats.get("weighted_avg_return", avg_return)
    profit_factor = stats.get("profit_factor", 0)
    sharpe_lite = stats.get("sharpe_lite", 0)
    expectancy = stats.get("expectancy", 0)
    avg_win_pct = stats.get("avg_win_pct", 0)
    avg_loss_pct = stats.get("avg_loss_pct", 0)

    ret_cls = _return_class(total_return)

    # NOTE: No leading whitespace before <div> tags — Streamlit's markdown
    # parser treats 4+ leading spaces as a code block.
    rows = (
        f'<div class="pf-stat-row">'
        f'<span class="pf-stat-label">Open Positions</span>'
        f'<span class="pf-stat-value">{open_count}</span>'
        f'</div>'
        f'<div class="pf-stat-row">'
        f'<span class="pf-stat-label">Closed</span>'
        f'<span class="pf-stat-value">{closed_count}</span>'
        f'</div>'
    )

    if closed_count > 0:
        # Color profit factor: green ≥1.5, yellow 1.0-1.5, red <1.0
        pf_color = (
            "var(--ss-green)" if profit_factor >= 1.5
            else "var(--ss-warn, #e0a800)" if profit_factor >= 1.0
            else "var(--ss-red, #d33)"
        )
        # Sharpe color: green ≥1.0, yellow 0.5-1.0, red <0.5
        sh_color = (
            "var(--ss-green)" if sharpe_lite >= 1.0
            else "var(--ss-warn, #e0a800)" if sharpe_lite >= 0.5
            else "var(--ss-red, #d33)"
        )
        rows += (
            f'<div class="pf-stat-row">'
            f'<span class="pf-stat-label">Target Hit Rate</span>'
            f'<span class="pf-stat-value" style="color:var(--ss-green);">{win_rate:.0%}</span>'
            f'</div>'
            # Weighted avg replaces simple avg — more accurate reflection of $ impact
            f'<div class="pf-stat-row" title="Size-weighted avg return (reflects dollar impact)">'
            f'<span class="pf-stat-label">Avg Return (wtd)</span>'
            f'<span class="pf-stat-value">{_fmt_pct(weighted_avg)}</span>'
            f'</div>'
            f'<div class="pf-stat-row" title="Gross wins / gross losses. &gt;1.5 is strong edge.">'
            f'<span class="pf-stat-label">Profit Factor</span>'
            f'<span class="pf-stat-value" style="color:{pf_color};">'
            f'{profit_factor:.2f}×</span>'
            f'</div>'
            f'<div class="pf-stat-row" title="Risk-adjusted return proxy. ≥1.0 is good.">'
            f'<span class="pf-stat-label">Sharpe (lite)</span>'
            f'<span class="pf-stat-value" style="color:{sh_color};">'
            f'{sharpe_lite:.2f}</span>'
            f'</div>'
            f'<div class="pf-stat-row" title="Expected $ P&amp;L per closed trade">'
            f'<span class="pf-stat-label">Expectancy</span>'
            f'<span class="pf-stat-value">${expectancy:+.0f}/trade</span>'
            f'</div>'
            f'<div class="pf-stat-row" title="Typical win size vs typical loss size" '
            f'style="font-size:0.78rem; color:var(--ss-text-muted);">'
            f'<span class="pf-stat-label">Avg W / L</span>'
            f'<span class="pf-stat-value">{_fmt_pct(avg_win_pct)} / {_fmt_pct(avg_loss_pct)}</span>'
            f'</div>'
        )

    if open_count > 0 or closed_count > 0:
        rows += (
            f'<div class="pf-stat-row" style="border-top:1px solid var(--ss-border); padding-top:6px; margin-top:4px;">'
            f'<span class="pf-stat-label">Total P&amp;L</span>'
            f'<span class="pf-stat-value pf-return {ret_cls}">{_fmt_pct(total_return)}</span>'
            f'</div>'
        )

    return (
        f'<div class="ss-portfolio-summary">'
        f'<div style="font-size:0.78rem; font-weight:700; color:var(--ss-text-muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">'
        f'💼 Virtual Portfolio'
        f'</div>'
        f'{rows}'
        f'</div>'
    )


def _render_stats_rows(stats: Dict[str, Any]) -> str:
    """Render hit rate / avg return / P&L rows from a stats dict."""
    closed = stats.get("closed_count", 0)
    if closed == 0:
        return (
            '<div class="pf-stat-row">'
            '<span class="pf-stat-label" style="color:var(--ss-text-muted); font-style:italic;">No closes yet</span>'
            '</div>'
        )
    win_rate = stats.get("win_rate", 0)
    avg_return = stats.get("avg_return", 0)
    total_return = stats.get("total_return_pct", 0)
    ret_cls = _return_class(total_return)
    return (
        f'<div class="pf-stat-row">'
        f'<span class="pf-stat-label">Target Hit Rate</span>'
        f'<span class="pf-stat-value" style="color:var(--ss-green);">{win_rate:.0%}</span>'
        f'</div>'
        f'<div class="pf-stat-row">'
        f'<span class="pf-stat-label">Avg Return</span>'
        f'<span class="pf-stat-value">{_fmt_pct(avg_return)}</span>'
        f'</div>'
        f'<div class="pf-stat-row">'
        f'<span class="pf-stat-label">Total P&amp;L</span>'
        f'<span class="pf-stat-value pf-return {ret_cls}">{_fmt_pct(total_return)}</span>'
        f'</div>'
    )


_EXIT_ICONS = {
    "target": "🎯",
    "stop": "🛑",
    "expiry": "⏰",
    "time_stop": "⌛",
}


def render_portfolio_sidebar_full(
    all_stats: Dict[str, Any],
    recent_stats: Dict[str, Any],
    exit_counts: Dict[str, int],
) -> str:
    """Render portfolio sidebar with all-time / recent split and exit breakdown.

    Args:
        all_stats: Dict from get_portfolio_stats() (no date filter).
        recent_stats: Dict from get_portfolio_stats(since_date=30d ago).
        exit_counts: Dict from get_exit_reason_counts().
    """
    open_count = all_stats.get("open_count", 0)
    closed_count = all_stats.get("closed_count", 0)

    # Header + open/closed counts
    html = (
        '<div class="ss-portfolio-summary">'
        '<div style="font-size:0.78rem; font-weight:700; color:var(--ss-text-muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">'
        '💼 Virtual Portfolio'
        '</div>'
        '<div class="pf-stat-row">'
        f'<span class="pf-stat-label">Open Positions</span>'
        f'<span class="pf-stat-value">{open_count}</span>'
        '</div>'
        '<div class="pf-stat-row">'
        f'<span class="pf-stat-label">Closed</span>'
        f'<span class="pf-stat-value">{closed_count}</span>'
        '</div>'
    )

    # All Time section
    html += (
        '<div class="pf-section-label">All Time</div>'
        + _render_stats_rows(all_stats)
    )

    # Recent 30d section
    html += (
        '<div class="pf-section-label">Recent 30d</div>'
        + _render_stats_rows(recent_stats)
    )

    # Exit reason breakdown
    if exit_counts:
        badges = ""
        for reason in ("target", "stop", "expiry", "time_stop"):
            count = exit_counts.get(reason, 0)
            if count > 0:
                icon = _EXIT_ICONS.get(reason, "")
                badges += f'<span>{icon} {count}</span>'
        if badges:
            html += f'<div class="pf-exit-row">{badges}</div>'

    html += '</div>'
    return html
