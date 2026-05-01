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

    Mirrors the gates in risk_manager.can_open_position +
    order_manager._filter_candidates that DON'T require IB live data.
    Cannot check: cash availability, day-trade history, IB rejection.
    For those we rely on the actual pipeline run.

    The point is to give the user an at-a-glance view in the dashboard
    of which scan candidates would survive the gauntlet — so the
    research view feels connected to the trading decisions.
    """
    state = state or fetch_state() or {}
    # Pull current account state for gates that depend on it
    positions = state.get("positions", []) or []
    held_tickers = {p.get("ticker", "").upper() for p in positions}
    pipeline = state.get("pipeline", {}) or {}
    regime = (pipeline.get("regime") or
              str(row.get("Market_Regime", "") or "")).upper()
    is_paused = state.get("paused", False)
    throttle = state.get("throttle", {}) or {}
    throttle_halt = throttle.get("level") == "halt"

    # Extract row fields (defensive — pandas Series or dict both work)
    def g(key, default=None):
        if hasattr(row, "get"):
            return row.get(key, default)
        try:
            return row[key]
        except Exception:
            return default

    ticker = str(g("Ticker", g("ticker", ""))).upper()
    score = float(g("FinalScore_20d", g("Score", 0)) or 0)
    rr = float(g("RewardRisk", g("RR", 0)) or 0)
    ml_prob = float(g("ML_20d_Prob", g("ml_prob", 0)) or 0)
    sector = str(g("Sector", g("sector", "")))
    confidence = str(g("SignalQuality", g("Confidence_Level", "")))

    # Regime-based score floor (mirrors risk_manager logic)
    REGIME_FLOORS = {
        "TREND_UP": 60, "BULLISH": 60, "MODERATE_UP": 65,
        "SIDEWAYS": 75, "NEUTRAL": 75,
        "DISTRIBUTION": 80, "CORRECTION": 85,
        "BEARISH": 80, "PANIC": 100,
    }
    score_floor = REGIME_FLOORS.get(regime, 73)

    # Run the gates
    gates_failed = []
    gates_passed = []

    # 1. Auto-trade paused?
    if is_paused:
        gates_failed.append("AUTO-TRADING PAUSED")
    else:
        gates_passed.append("not paused")

    # 2. Throttle halt?
    if throttle_halt:
        gates_failed.append("Performance throttle HALT")
    else:
        gates_passed.append("throttle ok")

    # 3. Already holding?
    if ticker in held_tickers:
        gates_failed.append(f"Already holding {ticker}")
    else:
        gates_passed.append("not held")

    # 4. Regime block?
    if regime in ("PANIC", "CORRECTION"):
        gates_failed.append(f"Market regime blocked: {regime}")
    else:
        gates_passed.append(f"regime ok ({regime})")

    # 5. Score floor (regime-aware)
    if score < score_floor:
        gates_failed.append(
            f"Score {score:.1f} < {score_floor} ({regime} floor)"
        )
    else:
        gates_passed.append(f"score ≥ {score_floor}")

    # 6. RR floor
    if rr < 2.0:
        gates_failed.append(f"R:R {rr:.2f} < 2.0")
    else:
        gates_passed.append("R:R ≥ 2.0")

    # 7. ML floor
    if ml_prob < 0.33:
        gates_failed.append(f"ML {ml_prob:.3f} < 0.33")
    else:
        gates_passed.append("ML ≥ 0.33")

    # 8. Confidence floor (regime-aware: TREND_UP allows Medium)
    conf_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "SPECULATIVE": 1}
    conf_val = conf_map.get(confidence.upper(), 0)
    min_conf = 2 if regime in ("TREND_UP", "MODERATE_UP", "BULLISH") else 3
    if conf_val < min_conf:
        gates_failed.append(
            f"Confidence {confidence} < required ({['n/a','Low','Medium','High'][min_conf]})"
        )
    else:
        gates_passed.append(f"confidence ≥ {confidence}")

    # 9. Blocked sector
    if sector == "Consumer Defensive":
        gates_failed.append(f"Blocked sector: {sector}")
    else:
        gates_passed.append("sector ok")

    # Verdict
    would_buy = len(gates_failed) == 0
    if would_buy:
        return {
            "would_buy": True,
            "verdict": "BUY ELIGIBLE",
            "reason": f"Passes all {len(gates_passed)} gates",
            "color": "#10b981",
            "icon": "🚀",
            "gates_passed": gates_passed,
            "gates_failed": [],
        }
    else:
        primary = gates_failed[0]
        return {
            "would_buy": False,
            "verdict": "SKIP",
            "reason": primary,
            "color": "#94a3b8",
            "icon": "⏭",
            "gates_passed": gates_passed,
            "gates_failed": gates_failed,
        }


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

    Uses a cache-busting query parameter to defeat the Fastly CDN that
    fronts raw.githubusercontent.com (~5 minute CDN cache by default).
    Without `?t=...` the dashboard would show 5-minute-stale state even
    though the VPS broadcaster pushes every 30s.
    """
    now = time.time()
    if _STATE_CACHE["data"] and (now - _STATE_CACHE["fetched_at"]) < max_age_sec:
        return _STATE_CACHE["data"]
    try:
        # Cache-bust by appending current epoch — every fetch is a new URL.
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
        return _STATE_CACHE.get("data")  # Return stale on network error


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

    cols = st.columns([2, 2, 2, 2, 2])
    with cols[0]:
        label = "PAUSED ⏸" if pipe_paused else f"{state_emoji} {pipe_state.upper()}"
        st.metric("Pipeline", label)
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
            st.metric("Throttle", "✅ inactive")

    # ── Service health line ──
    services = ["stockscout-monitor", "stockscout-pipeline.timer",
                "stockscout-statusbot", "ibgateway"]
    badges = []
    for s in services:
        ok = health.get(s, False)
        emoji = "✅" if ok else "🔴"
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
            ed_disp = f"✅ in {days_ed}d"

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
            f"+{p.get('peak_pct', 0):.1f}%" if p.get("peak_pct") else "—",
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
