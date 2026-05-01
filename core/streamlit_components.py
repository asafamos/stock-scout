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
    rows = []
    for p in positions:
        days_ed = p.get("days_to_earnings")
        ed_str = p.get("earnings_date", "")
        if days_ed is None:
            ed_disp = "—"
        elif days_ed < 0:
            ed_disp = f"{ed_str} (past)"
        elif days_ed == 0:
            ed_disp = f"⚠️ TODAY ({ed_str})"
        elif days_ed <= 3:
            ed_disp = f"🔴 in {days_ed}d ({ed_str})"
        elif days_ed <= 7:
            ed_disp = f"🟡 in {days_ed}d ({ed_str})"
        else:
            ed_disp = f"✅ in {days_ed}d ({ed_str})"

        pnl_pct = p.get("pnl_pct", 0) or 0
        pnl_color = "🟢" if pnl_pct >= 0 else "🔴"

        rows.append({
            "Ticker": p.get("ticker"),
            "Qty": p.get("qty"),
            "Entry": f"${p.get('entry', 0):.2f}" if p.get("entry") else "—",
            "Mkt": f"${p.get('mkt', 0):.2f}" if p.get("mkt") else "—",
            "P&L": f"{pnl_color} {pnl_pct:+.1f}%" if pnl_pct else "—",
            "Peak": f"+{p.get('peak_pct', 0):.1f}%" if p.get("peak_pct") else "—",
            "Trail": f"{p.get('trail_pct', 0):.1f}%",
            "Target": f"${p.get('target', 0):.2f}" if p.get("target") else "—",
            "Earnings": ed_disp,
        })

    import pandas as _pd
    st.dataframe(_pd.DataFrame(rows), hide_index=True, use_container_width=True)


def todays_actions_section(st):
    state = fetch_state()
    if not state:
        return
    actions = state.get("trade_log_today", []) or []
    if not actions:
        st.info("📋 No actions yet today.")
        return

    st.markdown("### 📋 Today's Actions")
    import pandas as _pd
    rows = []
    for a in actions[-20:]:
        ts = str(a.get("timestamp", ""))[11:19]
        action = a.get("action", "")
        emoji = {"OPEN": "🟢", "CLOSE": "🔴", "PARTIAL": "🟡",
                 "RECONCILE_DROP": "🧹"}.get(action, "•")
        pnl = a.get("pnl")
        rows.append({
            "Time": ts,
            "Action": f"{emoji} {action}",
            "Ticker": a.get("ticker", ""),
            "Qty": a.get("quantity", ""),
            "Price": f"${a.get('price', 0):.2f}" if a.get("price") else "—",
            "P&L": f"${pnl:+.2f}" if isinstance(pnl, (int, float)) else "—",
            "Reason": str(a.get("reason", ""))[:40],
        })
    st.dataframe(_pd.DataFrame(rows), hide_index=True, use_container_width=True)


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
