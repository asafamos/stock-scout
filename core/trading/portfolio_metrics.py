"""Portfolio-level performance metrics — CAGR, Sharpe, Max Drawdown.

Complements ledger.stats() (per-trade record) with account-level metrics
that answer "how has the strategy performed as a portfolio, not per trade."

CAGR       = (final_equity/starting_equity)^(365/days_elapsed) - 1
Sharpe     = mean(daily_return) / std(daily_return) * sqrt(252)   [rf=0]
Max DD     = maximum peak-to-trough drop of the equity curve, in %

Data source: `core.trading.ledger.load()` — the append-only executions
journal is the source of truth for realized P&L per day. We reconstruct
the daily equity curve as:

  equity(day) = starting_capital + Σ realized_pnl(<=day)

Note: this treats unrealized swings as invisible day-to-day (positions in
flight don't yet contribute to equity). Once a position closes, all of
its realized P&L lands on the close date. This gives cleaner day-to-day
returns than shipping mark-to-market unrealized swings from IB (which we
don't have a historical series for anyway).

For the CURRENT day's unrealized, we top-up the last equity point with
`open_unrealized` so the reported max-DD and current-DD reflect today's
actual account value, not stale-as-of-last-close.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from core.trading import ledger
from core.trading.config import CONFIG

logger = logging.getLogger(__name__)


def _parse_time(s):
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def daily_realized_series(cfg=None) -> dict:
    """Return {date: realized_pnl_that_day} from ledger executions.

    Only SELL executions carry realized_pnl. Sums per calendar date (UTC).
    """
    rows = ledger.load(cfg or CONFIG)
    daily: dict = {}
    for r in rows:
        pnl = r.get("realized_pnl")
        if pnl is None or pnl == 0:
            continue
        dt = _parse_time(r.get("time"))
        if not dt:
            continue
        d = dt.date()
        daily[d] = daily.get(d, 0.0) + float(pnl)
    return daily


def equity_curve(current_net_liq: float, current_unrealized: float,
                 cfg=None) -> list:
    """Reconstruct a daily equity series from ledger + IB truth.

    Returns [(date, equity_usd), ...] sorted ascending.

    Anchors:
      * Starting equity = starting_capital (from config / ledger baseline)
      * Between-close days: equity carries forward (realized P&L is what
        moves it, and no realized events happened those days).
      * TODAY's equity is overridden with `current_net_liq` from IB,
        which includes today's unrealized swings — so the current-DD is
        anchored to broker truth.
    """
    cfg = cfg or CONFIG
    start_equity = float(getattr(cfg, "starting_capital", 0.0) or 0.0)
    # Add pre-ledger anchor if migration set one
    pre_ledger = 0.0
    try:
        pre_ledger = float(getattr(cfg, "pre_ledger_realized", 0.0) or 0.0)
    except Exception:
        pre_ledger = 0.0

    daily = daily_realized_series(cfg)
    if not daily and start_equity == 0:
        return []

    # Baseline equity as of ledger cutover
    first_day = min(daily.keys()) if daily else datetime.now(timezone.utc).date()
    today = datetime.now(timezone.utc).date()

    equity_today_running = start_equity + pre_ledger
    curve = [(first_day - timedelta(days=1), equity_today_running)]

    # Iterate every calendar day so drawdown captures flat stretches
    d = first_day
    while d <= today:
        if d in daily:
            equity_today_running += daily[d]
        curve.append((d, equity_today_running))
        d += timedelta(days=1)

    # Override the FINAL point with IB truth (includes today's unrealized).
    # NetLiquidation = starting + realized (all-time) + unrealized (open).
    # If NetLiq is available and > 0, trust it for today's point.
    if current_net_liq and current_net_liq > 0:
        curve[-1] = (curve[-1][0], round(float(current_net_liq), 2))
    return curve


def compute_metrics(curve: list, risk_free_annual: float = 0.0) -> dict:
    """Standard portfolio metrics from an equity curve.

    curve: [(date, equity), ...]
    risk_free_annual: annual risk-free rate (default 0)

    Returns dict with keys:
      * days_elapsed, days_traded_data
      * start_equity, end_equity, total_return_pct
      * cagr_pct  — annualized (compound), None if <7 days elapsed
      * sharpe_annual — annualized (mean/std * sqrt(252)); None if too few
      * max_drawdown_pct  — worst peak-to-trough drop
      * current_drawdown_pct  — DD from all-time high right now
      * peak_equity, trough_equity
    """
    if len(curve) < 2:
        return dict(days_elapsed=0, days_traded_data=len(curve),
                    start_equity=0, end_equity=0, total_return_pct=0,
                    cagr_pct=None, sharpe_annual=None,
                    max_drawdown_pct=0, current_drawdown_pct=0,
                    peak_equity=0, trough_equity=0)

    start_eq = curve[0][1]
    end_eq = curve[-1][1]
    days_elapsed = (curve[-1][0] - curve[0][0]).days

    total_ret_pct = ((end_eq / start_eq - 1) * 100) if start_eq > 0 else 0.0

    # CAGR
    cagr = None
    if days_elapsed >= 7 and start_eq > 0 and end_eq > 0:
        years = days_elapsed / 365.25
        try:
            cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100
        except Exception:
            cagr = None

    # Daily returns for Sharpe
    daily_returns = []
    prev = start_eq
    for _, eq in curve[1:]:
        if prev > 0:
            daily_returns.append(eq / prev - 1)
        prev = eq

    sharpe = None
    if len(daily_returns) >= 10:
        mean_r = sum(daily_returns) / len(daily_returns)
        # sample stdev
        var = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_r = math.sqrt(var) if var > 0 else 0
        if std_r > 0:
            rf_daily = risk_free_annual / 252
            sharpe = (mean_r - rf_daily) / std_r * math.sqrt(252)

    # Max drawdown — walk equity, track running peak, worst drop
    peak_eq = curve[0][1]
    max_dd_pct = 0.0
    trough_eq = curve[0][1]
    for _, eq in curve:
        if eq > peak_eq:
            peak_eq = eq
        if peak_eq > 0:
            dd = (eq - peak_eq) / peak_eq * 100
            if dd < max_dd_pct:
                max_dd_pct = dd
                trough_eq = eq

    current_dd = 0.0
    if peak_eq > 0:
        current_dd = (end_eq - peak_eq) / peak_eq * 100

    return dict(
        days_elapsed=days_elapsed,
        days_traded_data=len(curve),
        start_equity=round(start_eq, 2),
        end_equity=round(end_eq, 2),
        total_return_pct=round(total_ret_pct, 2),
        cagr_pct=(round(cagr, 2) if cagr is not None else None),
        sharpe_annual=(round(sharpe, 2) if sharpe is not None else None),
        max_drawdown_pct=round(max_dd_pct, 2),
        current_drawdown_pct=round(current_dd, 2),
        peak_equity=round(peak_eq, 2),
        trough_equity=round(trough_eq, 2),
    )


def format_perf_summary(current_net_liq: float, current_unrealized: float,
                        cfg=None) -> str:
    """Formatted Telegram string. Called by /perf command."""
    cfg = cfg or CONFIG
    curve = equity_curve(current_net_liq, current_unrealized, cfg)
    if len(curve) < 2:
        start = float(getattr(cfg, "starting_capital", 0.0) or 0.0)
        return (
            "📊 <b>Portfolio Performance</b>\n\n"
            f"⏳ Not enough history yet\n"
            f"Starting: ${start:,.2f}\n"
            f"Current: ${current_net_liq:,.2f}\n\n"
            "Need at least a few closed trades for CAGR/Sharpe."
        )

    m = compute_metrics(curve)

    def _pct(x, prefix_plus=True):
        if x is None:
            return "—"
        sign = "+" if (prefix_plus and x >= 0) else ""
        return f"{sign}{x:.2f}%"

    def _sharpe_emoji(s):
        if s is None:
            return "⏳"
        if s >= 2.0:
            return "🟢🟢"
        if s >= 1.0:
            return "🟢"
        if s >= 0.5:
            return "🟡"
        if s >= 0.0:
            return "🟠"
        return "🔴"

    def _dd_emoji(dd):
        # dd is negative or zero
        if dd >= -5:
            return "🟢"
        if dd >= -10:
            return "🟡"
        if dd >= -20:
            return "🟠"
        return "🔴"

    def _ret_emoji(r):
        return "🟢" if r >= 0 else "🔴"

    # Ledger-level per-trade record for context
    try:
        st = ledger.stats(cfg)
    except Exception:
        st = {}
    n_closed = st.get("n_closed", 0)
    wr = st.get("win_rate", 0.0)
    pf = st.get("profit_factor")
    pf_str = f"{pf:.2f}" if pf is not None else ("∞" if st.get("realized_total", 0) > 0 else "—")

    # Post-freeze record — every closed trip since the 2026-07-09 gate freeze.
    # This is the ONLY sample that reflects the current (validated) config.
    # See CLAUDE.md `Gates frozen Jul 9` in memory.
    freeze_date = date(2026, 7, 9)
    post_freeze_trips = []
    try:
        for t in ledger.closed_round_trips(cfg):
            if t.get("realized_pnl") is None:
                continue
            sell_time = _parse_time(t.get("sell_time") or t.get("close_time") or t.get("time"))
            if sell_time and sell_time.date() >= freeze_date:
                post_freeze_trips.append(t)
    except Exception:
        pass
    pf_wins = [t for t in post_freeze_trips if t["realized_pnl"] > 0]
    pf_losses = [t for t in post_freeze_trips if t["realized_pnl"] < 0]
    pf_realized = sum(t["realized_pnl"] for t in post_freeze_trips)
    pf_n = len(post_freeze_trips)

    # Build line-by-line so conditional formatting is unambiguous
    lines = ["📊 <b>Portfolio Performance</b>", ""]

    lines.append(
        f"{_ret_emoji(m['total_return_pct'])} <b>Total Return:</b> "
        f"{_pct(m['total_return_pct'])} "
        f"<i>(${m['start_equity']:,.0f} → ${m['end_equity']:,.0f})</i>"
    )

    # Small-sample thresholds are strict on purpose: annualizing weeks of data
    # produces nonsense numbers (e.g. -11% over 47d → -61% CAGR).
    if m["cagr_pct"] is not None:
        if m["days_elapsed"] < 90:
            caveat = " <i>⚠ SMALL SAMPLE (need 90+d)</i>"
        elif m["days_elapsed"] < 180:
            caveat = " <i>⚠ small-sample</i>"
        else:
            caveat = ""
        lines.append(f"📈 <b>CAGR:</b> {_pct(m['cagr_pct'])}{caveat}")
    else:
        lines.append("📈 <b>CAGR:</b> ⏳ need 7+ days")

    if m["sharpe_annual"] is not None:
        if m["days_elapsed"] < 60:
            sh_caveat = " <i>⚠ SMALL SAMPLE (need 60+d)</i>"
        elif m["days_elapsed"] < 120:
            sh_caveat = " <i>⚠ small-sample</i>"
        else:
            sh_caveat = ""
        lines.append(
            f"{_sharpe_emoji(m['sharpe_annual'])} <b>Sharpe (annual):</b> "
            f"{m['sharpe_annual']:.2f}{sh_caveat}"
        )
    else:
        lines.append(
            f"{_sharpe_emoji(None)} <b>Sharpe (annual):</b> "
            "⏳ need 10+ trading days"
        )

    lines.append(
        f"{_dd_emoji(m['max_drawdown_pct'])} <b>Max Drawdown:</b> "
        f"{m['max_drawdown_pct']:.2f}% "
        f"<i>(peak ${m['peak_equity']:,.0f} → trough ${m['trough_equity']:,.0f})</i>"
    )
    lines.append(
        f"📉 <b>Current DD:</b> {m['current_drawdown_pct']:.2f}% "
        "<i>from all-time high</i>"
    )
    lines.append("")
    lines.append(
        f"<b>Record (all-time):</b> {st.get('wins', 0)}W / {st.get('losses', 0)}L "
        f"({wr:.0f}% WR, {n_closed} closed)"
    )
    lines.append(f"<b>Profit Factor:</b> {pf_str}")

    # Post-freeze breakdown — separates buggy-era losses from new-config trades
    if pf_n > 0:
        pf_wr = len(pf_wins) / pf_n * 100
        lines.append(
            f"<b>Post-freeze (since 2026-07-09):</b> "
            f"{len(pf_wins)}W / {len(pf_losses)}L ({pf_wr:.0f}% WR, "
            f"${pf_realized:+.2f}, n={pf_n})"
        )
    else:
        lines.append(
            "<b>Post-freeze (since 2026-07-09):</b> "
            "⏳ no closes yet — new config unproven"
        )

    lines.append(f"<b>Days elapsed:</b> {m['days_elapsed']}")
    lines.append("")
    lines.append(
        "<i>CAGR = annualized compound growth. "
        "Sharpe &gt; 1 = risk-adjusted good. Max DD = worst peak-to-trough loss.</i>"
    )
    return "\n".join(lines)
