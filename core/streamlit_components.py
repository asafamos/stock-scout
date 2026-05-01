"""Streamlit components for the automated-mode dashboard.

Reads from the state-feed branch (raw GitHub URL) so the dashboard
reflects live VPS state without lagging behind git checkouts.

Components:
- pipeline_status_widget — top of dashboard
- todays_actions_section — what the system did today (skips + buys)
- earnings_calendar_section — earnings dates for current positions
- emergency_controls_section — sell-all, pause, manual scan buttons
- manual_scan_guard — block manual scan button if pipeline running
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Public raw URL for the state-feed branch (no auth, public repo)
STATE_FEED_URL = (
    "https://raw.githubusercontent.com/asafamos/stock-scout/"
    "state-feed/data/state/system_state.json"
)

# Cache state in session for 30s to avoid hammering raw.githubusercontent
_STATE_CACHE: Dict[str, Any] = {"data": None, "fetched_at": 0}


def evaluate_scan_row_for_buy(row, state: Optional[Dict] = None) -> Dict:
    """Evaluate whether a single scan row would TRIGGER A BUY if the
    auto-trade pipeline ran on this scan right now.

    Returns:
      {
        "would_buy": True/False,
        "verdict": "BUY ELIGIBLE" | "SKIP",
        "reason": "<human-readable why>",
        "color": "#hex",
        "icon": "emoji",
        "gates_passed": [list of names],
        "gates_failed": [list of names],
      }

    Implementation note: this function is now a THIN WRAPPER around
    `core.trading.policy.evaluate_static_gates`, which is the single
    source of truth for buy-eligibility gates shared with
    risk_manager / order_manager. Hardcoded constants previously here
    (0.33 / 2.0 / 'Consumer Defensive' / a parallel REGIME_FLOORS
    table) drifted whenever the production CONFIG changed — exactly
    the parity bug we no longer want to ship.

    Cannot check (those run only in production): live cash, market
    hours, day-trade history, earnings calendar, sector momentum
    (yfinance), portfolio correlation, slippage hard-reject.
    """
    state = state or fetch_state() or {}

    try:
        from core.trading.policy import evaluate_static_gates
    except Exception as _imp_err:
        # Fallback — if policy.py is unavailable for any reason, fail
        # OPEN with a clear note rather than silently skipping the row.
        return {
            "would_buy": False,
            "verdict": "UNKNOWN",
            "reason": f"policy module unavailable: {_imp_err}",
            "color": "#94a3b8",
            "icon": "❓",
            "gates_passed": [],
            "gates_failed": [f"policy import failed: {_imp_err}"],
        }

    result = evaluate_static_gates(row, state=state)
    if result.would_buy:
        return {
            "would_buy": True,
            "verdict": "BUY ELIGIBLE",
            "reason": f"Passes all {len(result.gates_passed)} gates",
            "color": "#10b981",
            "icon": "🚀",
            "gates_passed": result.gates_passed,
            "gates_failed": [],
        }
    return {
        "would_buy": False,
        "verdict": "SKIP",
        "reason": result.primary_reason,
        "color": "#94a3b8",
        "icon": "⏭",
        "gates_passed": result.gates_passed,
        "gates_failed": result.gates_failed,
    }


def buy_pre_check_widget(st, scan_df=None):
    """Top-of-page widget showing how many scan candidates would
    pass all gates if the auto-trade pipeline ran on the current scan now.

    Renders inline with pipeline_status_widget for at-a-glance visibility.
    Falls back gracefully if scan_df is missing or empty.

    For each eligible candidate, also renders the **execution preview**:
    the actual entry / stop / target / trail % the order_manager would
    submit to IB (via core.trading.policy.compute_execution_preview).
    Eliminates the "scan numbers ≠ trade numbers" trust gap.
    """
    state = fetch_state() or {}
    if scan_df is None or len(scan_df) == 0:
        return  # silent — nothing to evaluate

    evals = []
    for _idx, row in scan_df.iterrows():
        evals.append(evaluate_scan_row_for_buy(row, state))
    eligible = [e for e in evals if e["would_buy"]]
    n_eligible = len(eligible)
    n_total = len(evals)

    # Render a hand-rolled LTR-locked box. st.success/st.warning inherit
    # the global RTL CSS and end up flipping the leading numbers/symbols
    # to the wrong end of the line — confusing for English-content alerts.
    if n_eligible > 0:
        eligible_tickers = [
            str(scan_df.iloc[i]["Ticker"])
            for i, e in enumerate(evals) if e["would_buy"]
        ]
        ticker_list = ", ".join(eligible_tickers[:8])
        if len(eligible_tickers) > 8:
            ticker_list += f" +{len(eligible_tickers) - 8} more"
        st.markdown(
            f'<div dir="ltr" style="direction:ltr;text-align:left;'
            f'background:rgba(16,185,129,0.10);border-left:4px solid #10b981;'
            f'border-radius:8px;padding:12px 16px;margin:8px 0;'
            f'font-size:0.92rem;color:var(--ss-text-primary,#0c4a36);">'
            f'🚀 <b>{n_eligible} of {n_total}</b> would BUY if pipeline ran now · '
            f'<code style="direction:ltr;">{ticker_list}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        # Top 3 reasons
        reason_counts: Dict[str, int] = {}
        for e in evals:
            reason_counts[e["reason"]] = reason_counts.get(e["reason"], 0) + 1
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:3]
        reason_summary = " · ".join(
            f'<code style="direction:ltr;">{r}</code> ({c})'
            for r, c in sorted_reasons
        )
        st.markdown(
            f'<div dir="ltr" style="direction:ltr;text-align:left;'
            f'background:rgba(234,179,8,0.10);border-left:4px solid #eab308;'
            f'border-radius:8px;padding:12px 16px;margin:8px 0;'
            f'font-size:0.92rem;color:var(--ss-text-primary,#78350f);">'
            f'⏭ <b>0 of {n_total}</b> would BUY · Top blocks: {reason_summary}'
            f'</div>',
            unsafe_allow_html=True,
        )


def earnings_warning_widget(st):
    """Banner above Live Positions when any current position has earnings
    within 7 trading days — flags binary risk and reminds the user that
    the system will auto-tighten TRAIL to 2% at <2d."""
    state = fetch_state() or {}
    positions = state.get("positions", []) or []
    soon = []
    for p in positions:
        d = p.get("days_to_earnings")
        if d is not None and 0 <= d <= 7:
            soon.append((p.get("ticker"), d, p.get("earnings_date", "")))
    if not soon:
        return
    soon.sort(key=lambda x: x[1])
    parts = [
        f"<b>{t}</b> in {d}d ({date_str})"
        + (" 🔴 auto-tighten zone" if d <= 2 else "")
        for t, d, date_str in soon
    ]
    # Hand-rolled LTR box — st.warning was rendering RTL-flipped on the
    # Hebrew-default page (".Earnings approaching..." with leading period).
    st.markdown(
        f'<div dir="ltr" style="direction:ltr;text-align:left;'
        f'background:rgba(234,179,8,0.10);border-left:4px solid #eab308;'
        f'border-radius:8px;padding:12px 16px;margin:8px 0;'
        f'font-size:0.92rem;color:var(--ss-text-primary,#78350f);">'
        f'📅 <b>Earnings approaching:</b> ' + " · ".join(parts) +
        ' — system will auto-tighten TRAIL to 2% at &lt;2d.'
        f'</div>',
        unsafe_allow_html=True,
    )


def execution_preview_section(st, scan_df=None, max_rows: int = 5):
    """Render the EXECUTION PREVIEW table — what the order_manager would
    actually submit for the top eligible candidates.

    Closes the trust gap between "scan numbers" and "trade numbers":
    - Live-refreshed entry (when state-feed has cached prices).
    - Proportionally-rescaled stop & target.
    - Trail % after regime multiplier + safety clamp.
    - Estimated qty + spend given current cash.

    Identical math to `order_manager._execute_single` (via the shared
    `core.trading.policy.compute_execution_preview`). Eliminates the
    "5 trades 20% WR vs throttle 28%" class of inconsistency: this
    table is pulled from the same code path the trader runs.
    """
    if scan_df is None or len(scan_df) == 0:
        return
    try:
        from core.trading.policy import (
            compute_execution_preview,
            evaluate_static_gates,
        )
    except Exception as _e:
        st.caption(f"⚠️ execution preview unavailable: {_e}")
        return

    state = fetch_state() or {}
    # Filter to rows that pass all static gates first — no point previewing
    # rows that won't be considered by the trader.
    eligible_rows = []
    for _idx, row in scan_df.iterrows():
        gr = evaluate_static_gates(row, state=state)
        if gr.would_buy:
            eligible_rows.append(row)
    if not eligible_rows:
        return  # silent — pre_check_widget already shows "0 of N would BUY"

    # Cash budget for sizing — read from state-feed account snapshot
    account = state.get("account", {}) or {}
    cash = float(account.get("cash") or 0)
    cfg_reserve = 20.0  # cfg.cash_reserve default
    available_cash = max(0.0, cash - cfg_reserve)

    # Throttle multiplier — match production sizing exactly. When the
    # rolling-window WR is in WARN range (under 30%), production halves
    # position sizes; the preview must apply the same factor or qty
    # estimates will be off by 2× from what actually executes.
    throttle = state.get("throttle", {}) or {}
    throttle_mult = float(throttle.get("size_multiplier", 1.0) or 1.0)

    previews = []
    for row in eligible_rows[:max_rows]:
        try:
            p = compute_execution_preview(
                row,
                available_cash=available_cash if available_cash > 0 else None,
                throttle_mult=throttle_mult,
            )
            previews.append(p)
        except Exception:
            continue

    if not previews:
        return

    # Render as LTR-locked HTML table (same pattern as Live Positions)
    headers = ["Ticker", "Entry", "Stop", "Target", "Trail %",
               "Qty", "Spend", "Notes"]
    rows_html = []
    for p in previews:
        notes_short = "; ".join(p.notes)[:48] if p.notes else "—"
        cells = [
            f"<b>{p.ticker}</b>",
            f"${p.entry:.2f}",
            f"${p.stop:.2f}",
            f"${p.target:.2f}",
            f"{p.trail_pct:.1f}%",
            str(p.qty_estimate),
            f"${p.spend_estimate:,.0f}",
            f'<span style="font-size:0.74rem;color:rgba(0,0,0,0.55)">{notes_short}</span>',
        ]
        cells_html = "".join(
            f'<td style="padding:7px 11px;direction:ltr;text-align:left;'
            f'border-bottom:1px solid rgba(0,0,0,0.06);font-feature-settings:\'tnum\'">'
            f'{c}</td>' for c in cells
        )
        rows_html.append(f"<tr>{cells_html}</tr>")
    headers_html = "".join(
        f'<th style="padding:9px 11px;direction:ltr;text-align:left;'
        f'border-bottom:2px solid rgba(0,0,0,0.12);font-size:0.72rem;'
        f'font-weight:600;text-transform:uppercase;letter-spacing:0.04em;'
        f'color:rgba(0,0,0,0.6)">{h}</th>'
        for h in headers
    )
    st.markdown(
        '<div dir="ltr" style="direction:ltr;width:100%;overflow-x:auto;'
        'background:var(--ss-bg-card,white);border-radius:8px;'
        'border:1px solid rgba(0,0,0,0.08);margin:8px 0;">'
        f'<table dir="ltr" style="direction:ltr;width:100%;border-collapse:collapse;">'
        f'<thead><tr>{headers_html}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
        f'<div style="padding:8px 12px;font-size:0.72rem;color:rgba(0,0,0,0.55);direction:ltr;text-align:left;">'
        f'These are the numbers the trader will actually submit to IB '
        f'(after live-price refresh + proportional rescale + regime-aware trail). '
        f'Static gates passed; runtime gates (slippage, earnings, sector momentum) '
        f'are evaluated at execution time.'
        f'</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_eligibility_badge(eval_result: Dict) -> str:
    """Return inline HTML badge HTML for use in scan result cards."""
    return (
        f'<span dir="ltr" style="direction:ltr;display:inline-flex;'
        f'align-items:center;gap:6px;padding:3px 10px;border-radius:999px;'
        f'background:{eval_result["color"]};color:white;font-size:0.72rem;'
        f'font-weight:700;text-transform:uppercase;letter-spacing:0.04em;'
        f'margin-right:8px;font-family:-apple-system,sans-serif;">'
        f'{eval_result["icon"]} {eval_result["verdict"]}'
        f'</span>'
    )


def fetch_state(max_age_sec: int = 30) -> Optional[Dict]:
    """Fetch system_state.json from state-feed. Cached 30s.

    Uses GitHub Contents API instead of raw.githubusercontent.com because
    raw URLs are fronted by Fastly CDN with ~5 minute cache that ignores
    `?t=...` cache-busting on small static files. The Contents API
    returns base64-encoded content but has no aggressive CDN cache,
    so we get sub-30s freshness reliably.

    Auth: if GITHUB_TOKEN is in env or st.secrets, uses it for higher
    rate limit (5000/h vs 60/h unauth). Streamlit polls every 30s = 2/min
    so unauth is fine for now.
    """
    import base64
    import os

    now = time.time()
    if _STATE_CACHE["data"] and (now - _STATE_CACHE["fetched_at"]) < max_age_sec:
        return _STATE_CACHE["data"]

    # Try Streamlit secrets for the token first (cloud), then env (local)
    token = None
    try:
        import streamlit as _st
        token = _st.secrets.get("GITHUB_TOKEN") if hasattr(_st, "secrets") else None
    except Exception:
        token = None
    if not token:
        token = os.environ.get("GITHUB_TOKEN")

    api_url = (
        "https://api.github.com/repos/asafamos/stock-scout/"
        "contents/data/state/system_state.json?ref=state-feed"
    )
    headers = {
        "User-Agent": "StockScoutDashboard/1.0",
        "Accept": "application/vnd.github+json",
        "Cache-Control": "no-cache",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            api_resp = json.loads(resp.read())
        # Content is base64-encoded
        content_b64 = api_resp.get("content", "")
        if content_b64:
            content = base64.b64decode(content_b64).decode("utf-8")
            data = json.loads(content)
            _STATE_CACHE["data"] = data
            _STATE_CACHE["fetched_at"] = now
            return data
    except Exception:
        pass  # fall through to raw URL fallback

    # FALLBACK: raw URL with cache-busting (less reliable due to CDN
    # but works without auth and might cache us a result)
    try:
        url = f"{STATE_FEED_URL}?t={int(now)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "StockScoutDashboard/1.0",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        _STATE_CACHE["data"] = data
        _STATE_CACHE["fetched_at"] = now
        return data
    except Exception:
        return _STATE_CACHE.get("data")  # last-known on total failure


def _state_age_seconds(state: Dict) -> Optional[float]:
    try:
        ts = state.get("last_updated", "")
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return None


def pipeline_status_widget(st):
    """Top-of-dashboard widget showing live pipeline + system state."""
    state = fetch_state()
    if not state:
        st.info(
            "🔄 System state unavailable (state-feed not yet pushed). "
            "VPS broadcaster may be initializing — refresh in 30s."
        )
        return

    age_sec = _state_age_seconds(state)
    if age_sec is not None and age_sec > 120:
        st.warning(
            f"⚠️ State data is {int(age_sec)}s old (broadcaster may be down). "
            "Showing last-known values."
        )

    pipeline = state.get("pipeline", {}) or {}
    health = state.get("system_health", {}) or {}
    account = state.get("account", {}) or {}
    throttle = state.get("throttle", {}) or {}
    positions = state.get("positions", []) or []

    # ── Top row: pipeline state + next fire ──
    pipe_state = pipeline.get("state", "unknown")
    state_emoji = {
        "idle": "🟢",
        "dispatched": "📡",
        "polling": "⏳",
        "trading": "💹",
        "done": "✅",
        "failed": "❌",
        "failed_timeout": "⏰",
        "unknown": "❓",
    }.get(pipe_state, "❓")
    next_fire = health.get("pipeline_next_fire", "—")
    pipe_paused = state.get("paused", False)

    # When pipeline is running, compute elapsed and rough ETA
    elapsed_str = None
    eta_str = None
    if pipe_state in ("dispatched", "polling", "trading"):
        last_fire = pipeline.get("last_fire", "")
        if last_fire and len(last_fire) >= 8:
            try:
                from datetime import datetime as _dt, timezone as _tz
                # last_fire is "HH:MM:SS" — assume today UTC
                today = _dt.now(_tz.utc).date()
                hh, mm, ss = last_fire.split(":")
                start_dt = _dt(today.year, today.month, today.day,
                               int(hh), int(mm), int(ss), tzinfo=_tz.utc)
                elapsed_sec = (_dt.now(_tz.utc) - start_dt).total_seconds()
                if elapsed_sec < 0:
                    elapsed_sec += 86400  # crossed midnight
                elapsed_min = int(elapsed_sec // 60)
                elapsed_str = f"{elapsed_min}m"
                # Typical scan completes in ~50 min, trade is fast
                # ETA = max(0, 55 - elapsed)
                eta_min = max(0, 55 - elapsed_min)
                eta_str = f"~{eta_min}m left"
            except Exception:
                pass

    cols = st.columns([2, 2, 2, 2, 2])
    with cols[0]:
        if pipe_paused:
            label = "⏸ PAUSED"
            delta = None
        elif elapsed_str:
            label = f"{state_emoji} {pipe_state.upper()}"
            delta = f"{elapsed_str} · {eta_str}" if eta_str else elapsed_str
        else:
            label = f"{state_emoji} {pipe_state.upper()}"
            delta = None
        st.metric("Pipeline", label, delta)
    with cols[1]:
        # next_fire might be ISO string ("2026-05-01T13:30:00+00:00"),
        # an int (microseconds-since-epoch), None, or "—". Handle all.
        nf_display = "—"
        try:
            if isinstance(next_fire, str) and len(next_fire) >= 16:
                nf_display = next_fire[11:16] + " UTC"
            elif isinstance(next_fire, (int, float)) and next_fire > 0:
                from datetime import datetime as _dt, timezone as _tz
                _v = next_fire / 1_000_000 if next_fire > 1e12 else next_fire
                nf_display = _dt.fromtimestamp(_v, tz=_tz.utc).strftime("%H:%M UTC")
        except Exception:
            pass
        st.metric("Next fire", nf_display)
    with cols[2]:
        net = account.get("net_liquidation", 0)
        tier_emoji = {"sub_2k": "🟠", "cash": "🟡", "margin_pdt": "🟢"}.get(
            account.get("tier", ""), "⚪"
        )
        st.metric(f"Net Liq {tier_emoji}", f"${net:,.0f}")
    with cols[3]:
        st.metric("Cash", f"${account.get('cash', 0):,.0f}")
    with cols[4]:
        if throttle.get("active"):
            level = throttle.get("level", "warn")
            mult = throttle.get("size_multiplier", 1.0)
            wr = throttle.get("win_rate", 0)
            color = "🟡" if level == "warn" else "🔴"
            st.metric(f"Throttle {color}", f"{int(mult*100)}% size",
                      f"WR {int(wr*100)}%")
        else:
            st.metric("Throttle", "🟢 inactive")

    # ── Service health line ──
    services = ["stockscout-monitor", "stockscout-pipeline.timer",
                "stockscout-statusbot", "ibgateway"]
    badges = []
    for s in services:
        ok = health.get(s, False)
        emoji = "🟢" if ok else "🔴"
        badges.append(f"{emoji} {s.replace('stockscout-', '')}")
    st.caption(" • ".join(badges))


def positions_with_earnings_section(st):
    state = fetch_state()
    if not state:
        return
    positions = state.get("positions", []) or []
    if not positions:
        st.info("📭 No open positions.")
        return

    st.markdown("### 📊 Live Positions")

    # Build an HTML table manually — st.dataframe was being mangled by
    # the global RTL CSS even with overrides, truncating cells to a
    # single char. Hand-rolled HTML with explicit dir="ltr" and inline
    # styles gives us guaranteed reliable rendering.
    headers = ["Ticker", "Qty", "Entry", "Mkt", "P&L", "Peak", "Trail",
               "Target", "Earnings"]
    rows_html = []
    for p in positions:
        days_ed = p.get("days_to_earnings")
        ed_str = p.get("earnings_date", "")
        if days_ed is None:
            ed_disp = "—"
        elif days_ed < 0:
            ed_disp = f"{ed_str} (past)"
        elif days_ed == 0:
            ed_disp = f"⚠️ TODAY"
        elif days_ed <= 3:
            ed_disp = f"🔴 in {days_ed}d"
        elif days_ed <= 7:
            ed_disp = f"🟡 in {days_ed}d"
        else:
            ed_disp = f"🟢 in {days_ed}d"

        pnl_pct = p.get("pnl_pct", 0) or 0
        pnl_color = "#10b981" if pnl_pct >= 0 else "#ef4444"
        pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
        pnl_disp = f"{pnl_emoji} {pnl_pct:+.1f}%" if pnl_pct else "—"

        cells = [
            f"<b>{p.get('ticker','?')}</b>",
            str(p.get("qty", "")),
            f"${p.get('entry', 0):.2f}" if p.get("entry") else "—",
            f"${p.get('mkt', 0):.2f}" if p.get("mkt") else "—",
            f'<span style="color:{pnl_color};font-weight:600">{pnl_disp}</span>',
            # Use +.1f so negative peaks render "-1.1%" not "+-1.1%"
            f"{p.get('peak_pct', 0):+.1f}%" if p.get("peak_pct") else "—",
            f"{p.get('trail_pct', 0):.1f}%",
            f"${p.get('target', 0):.2f}" if p.get("target") else "—",
            ed_disp,
        ]
        cells_html = "".join(
            f'<td style="padding:8px 12px;direction:ltr;text-align:left;'
            f'border-bottom:1px solid rgba(0,0,0,0.06);font-feature-settings:\'tnum\'">'
            f'{c}</td>' for c in cells
        )
        rows_html.append(f"<tr>{cells_html}</tr>")

    headers_html = "".join(
        f'<th style="padding:10px 12px;direction:ltr;text-align:left;'
        f'border-bottom:2px solid rgba(0,0,0,0.12);font-size:0.78rem;'
        f'font-weight:600;text-transform:uppercase;letter-spacing:0.04em;'
        f'color:rgba(0,0,0,0.6)">{h}</th>'
        for h in headers
    )
    table_html = (
        f'<div dir="ltr" style="direction:ltr;width:100%;overflow-x:auto;'
        f'background:var(--ss-bg-card,white);border-radius:8px;'
        f'border:1px solid rgba(0,0,0,0.08);margin:8px 0;">'
        f'<table dir="ltr" style="direction:ltr;width:100%;'
        f'border-collapse:collapse;font-family:-apple-system,sans-serif;'
        f'font-size:0.92rem">'
        f'<thead><tr>{headers_html}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def todays_actions_section(st):
    state = fetch_state()
    if not state:
        return
    actions = state.get("trade_log_today", []) or []
    if not actions:
        st.info("📋 No actions yet today.")
        return

    headers = ["Time", "Action", "Ticker", "Qty", "Price", "P&L", "Reason"]
    rows_html = []
    for a in actions[-20:]:
        ts = str(a.get("timestamp", ""))[11:19]
        action = a.get("action", "")
        emoji = {"OPEN": "🟢", "CLOSE": "🔴", "PARTIAL": "🟡",
                 "RECONCILE_DROP": "🧹"}.get(action, "•")
        pnl = a.get("pnl")
        pnl_color = "inherit"
        pnl_str = "—"
        if isinstance(pnl, (int, float)):
            pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
            pnl_str = f"${pnl:+.2f}"
        cells = [
            ts,
            f"{emoji} {action}",
            f"<b>{a.get('ticker','')}</b>",
            str(a.get("quantity", "")),
            f"${a.get('price', 0):.2f}" if a.get("price") else "—",
            f'<span style="color:{pnl_color}">{pnl_str}</span>',
            str(a.get("reason", ""))[:40],
        ]
        cells_html = "".join(
            f'<td style="padding:6px 10px;direction:ltr;text-align:left;'
            f'border-bottom:1px solid rgba(0,0,0,0.06);font-size:0.88rem">'
            f'{c}</td>' for c in cells
        )
        rows_html.append(f"<tr>{cells_html}</tr>")

    headers_html = "".join(
        f'<th style="padding:8px 10px;direction:ltr;text-align:left;'
        f'border-bottom:2px solid rgba(0,0,0,0.12);font-size:0.72rem;'
        f'font-weight:600;text-transform:uppercase;letter-spacing:0.04em;'
        f'color:rgba(0,0,0,0.6)">{h}</th>'
        for h in headers
    )
    table_html = (
        f'<div dir="ltr" style="direction:ltr;width:100%;overflow-x:auto;'
        f'background:var(--ss-bg-card,white);border-radius:8px;'
        f'border:1px solid rgba(0,0,0,0.08);margin:8px 0;">'
        f'<table dir="ltr" style="direction:ltr;width:100%;'
        f'border-collapse:collapse;font-family:-apple-system,sans-serif">'
        f'<thead><tr>{headers_html}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def emergency_controls_section(st):
    """Action buttons that trigger commands on the VPS via repository_dispatch."""
    state = fetch_state()
    if not state:
        return

    pipe_state = state.get("pipeline", {}).get("state", "unknown")
    is_paused = state.get("paused", False)
    is_pipeline_running = pipe_state in ("dispatched", "polling", "trading")

    st.markdown("### 🎛️ System Controls")
    st.caption(
        "Commands are sent to the VPS via GitHub workflow_dispatch. "
        "They take ~30-60s to execute. For instant action use the Telegram bot."
    )

    # Manual scan with conflict guard
    cols = st.columns(3)
    with cols[0]:
        if is_pipeline_running:
            st.warning(
                f"⏳ Pipeline already running ({pipe_state}). "
                f"Manual scan would conflict."
            )
            st.button("🔍 Run Scan Now", disabled=True,
                     help="Pipeline is currently running — wait for it to finish")
        else:
            if st.button("🔍 Run Scan Now"):
                _dispatch_command(st, "scan")

    with cols[1]:
        if is_paused:
            if st.button("▶️ Resume Auto-Trading"):
                _dispatch_command(st, "resume")
        else:
            if st.button("⏸ Pause Auto-Trading"):
                _dispatch_command(st, "pause")

    with cols[2]:
        if st.button("🚨 SELL ALL (emergency)", type="primary"):
            st.session_state["_confirm_sellall"] = True

    if st.session_state.get("_confirm_sellall"):
        st.error(
            "⚠️ **CONFIRM:** This will market-sell ALL open positions immediately. "
            "Real money. Are you sure?"
        )
        cc = st.columns(2)
        with cc[0]:
            if st.button("✅ YES — sell everything"):
                _dispatch_command(st, "sellall")
                st.session_state["_confirm_sellall"] = False
        with cc[1]:
            if st.button("❌ Cancel"):
                st.session_state["_confirm_sellall"] = False
                st.rerun()


def _dispatch_command(st, command: str, **inputs):
    """Send a command to VPS via repository_dispatch.

    Requires GITHUB_TOKEN in st.secrets. Falls back to displaying the
    Telegram equivalent if no token configured.
    """
    token = None
    try:
        token = st.secrets.get("GITHUB_TOKEN") if hasattr(st, "secrets") else None
    except Exception:
        token = None

    if not token:
        st.info(
            f"💡 To execute `{command}` directly, send `/{command}` to "
            f"the Telegram bot (@stockscout_asaf_bot). "
            f"Streamlit dispatch needs GITHUB_TOKEN in secrets.toml."
        )
        return

    # Issue repository_dispatch event with the command
    payload = json.dumps({
        "event_type": "stockscout-command",
        "client_payload": {"command": command, **inputs},
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.github.com/repos/asafamos/stock-scout/dispatches",
        data=payload,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 204:
                st.success(
                    f"✅ Dispatched `{command}` to VPS. "
                    f"Watch Telegram for confirmation (~30-60s)."
                )
            else:
                st.error(f"Dispatch HTTP {resp.status}")
    except Exception as e:
        st.error(f"Dispatch failed: {e}")
