"""Live Portfolio — shows real-time IB portfolio from VPS.

Reads the latest snapshot pushed by the VPS monitor to Supabase
(upserted every 5 min during market hours).
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Live Portfolio", page_icon="💼", layout="wide")

st.title("💼 Live Portfolio")
st.caption("Real-time state from VPS auto-trade system")


@st.cache_data(ttl=30)
def load_snapshot():
    """Load snapshot from Supabase (preferred) or local file fallback."""
    # Try Supabase
    try:
        from core.db.supabase_client import get_supabase_client
        sb = get_supabase_client()
        if sb:
            res = sb.table("trading_snapshot").select("*").eq("id", 1).execute()
            if res.data:
                return res.data[0].get("data"), "supabase", res.data[0].get("updated_at")
    except Exception as e:
        st.warning(f"Supabase read failed: {e}")

    # Fallback: local file
    try:
        snapshot_file = Path("data/trades/portfolio_snapshot.json")
        if snapshot_file.exists():
            data = json.loads(snapshot_file.read_text())
            return data, "local file", data.get("updated_at")
    except Exception:
        pass

    return None, "none", None


snapshot, source, updated_at = load_snapshot()

if not snapshot:
    st.error(
        "❌ No portfolio snapshot available.\n\n"
        "This can mean:\n"
        "- VPS monitor hasn't pushed a snapshot yet\n"
        "- Supabase connection is unavailable\n"
        "- Market is closed (monitor doesn't run outside hours)\n\n"
        "**Send 'status' in Telegram for live data.**"
    )
    st.stop()

# ── Header: account summary ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

cash = snapshot.get("cash", 0)
net = snapshot.get("net_liquidation", 0)
pnl = snapshot.get("total_unrealized_pnl", 0)
pos_count = snapshot.get("position_count", 0)

with col1:
    st.metric("Net Liquidation", f"${net:,.2f}")
with col2:
    st.metric("Cash Available", f"${cash:,.2f}")
with col3:
    pnl_delta = f"{pnl:+,.2f}"
    st.metric("Unrealized P&L", f"${pnl:+,.2f}", delta=pnl_delta)
with col4:
    st.metric("Open Positions", pos_count)

# Timestamp
if updated_at:
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        age_sec = (datetime.now(timezone.utc) - dt).total_seconds()
        age_str = f"{int(age_sec)}s ago" if age_sec < 60 else f"{int(age_sec/60)}m ago"
        st.caption(f"📡 Data source: {source} • Last updated: {age_str} ({updated_at})")
    except Exception:
        st.caption(f"📡 Data source: {source}")

st.divider()

# ── Positions ─────────────────────────────────────────────────────────
st.subheader("📊 Positions")

positions = snapshot.get("positions", [])
if not positions:
    st.info("No open positions.")
else:
    for pos in positions:
        ticker = pos["ticker"]
        qty = pos["quantity"]
        avg_cost = pos["avg_cost"]
        mkt = pos["market_price"]
        upnl = pos["unrealized_pnl"]
        pnl_pct = pos.get("pnl_pct", 0)
        target = pos.get("target_price") or 0
        stop = pos.get("stop_loss") or 0
        target_date = pos.get("target_date", "")
        peak = pos.get("peak_price") or avg_cost
        stop_floor = pos.get("stop_floor") or 0

        # Card
        emoji = "🟢" if upnl >= 0 else "🔴"
        with st.container(border=True):
            header_cols = st.columns([3, 2, 2, 2, 2])
            with header_cols[0]:
                st.markdown(f"### {emoji} **{ticker}**")
                st.caption(f"{qty} shares")
            with header_cols[1]:
                st.metric("Entry", f"${avg_cost:.2f}")
            with header_cols[2]:
                st.metric("Market", f"${mkt:.2f}", delta=f"{pnl_pct:+.2f}%")
            with header_cols[3]:
                st.metric("P&L", f"${upnl:+,.2f}")
            with header_cols[4]:
                if target:
                    upside_pct = (target / avg_cost - 1) * 100
                    st.metric("Target", f"${target:.2f}", delta=f"+{upside_pct:.1f}%")

            # Protection details
            detail_cols = st.columns(4)
            with detail_cols[0]:
                if stop:
                    st.caption(f"🛡 Stop: ${stop:.2f}")
            with detail_cols[1]:
                if peak:
                    peak_gain = (peak / avg_cost - 1) * 100
                    st.caption(f"⬆️ Peak: ${peak:.2f} ({peak_gain:+.1f}%)")
            with detail_cols[2]:
                if stop_floor:
                    st.caption(f"🔒 Floor: ${stop_floor:.2f}")
            with detail_cols[3]:
                if target_date:
                    st.caption(f"📅 Exit by: {target_date[:10]}")

st.divider()

# ── Protective Orders ─────────────────────────────────────────────────
st.subheader("🛡 Active Orders")

orders = snapshot.get("orders", [])
if not orders:
    st.warning("⚠️ No active orders!")
else:
    for o in orders:
        ticker = o["ticker"]
        otype = o["type"]
        qty = o["qty"]
        action = o["action"]

        if otype == "TRAIL":
            detail = f"Trail {o.get('trail_pct')}% → stop ${o.get('stop_price', 0):.2f}"
        elif otype == "LMT":
            detail = f"Limit ${o.get('limit_price', 0):.2f}"
        elif otype == "STP":
            detail = f"Stop ${o.get('stop_price', 0):.2f}"
        else:
            detail = otype

        st.markdown(f"- **{ticker}** — {action} {qty} ({otype}) — {detail} — {o.get('tif', 'DAY')}")

st.divider()

# ── Recent Trades ─────────────────────────────────────────────────────
st.subheader("📋 Recent Trade Log")

try:
    log_path = Path("data/trades/trade_log.json")
    if log_path.exists():
        log = json.loads(log_path.read_text())
        recent = sorted(log, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]

        if recent:
            import pandas as pd
            df = pd.DataFrame(recent)
            # Format
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].str[:19].str.replace("T", " ")
            cols_to_show = [c for c in ["timestamp", "action", "ticker", "quantity",
                                         "price", "pnl", "reason"] if c in df.columns]
            st.dataframe(df[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info("No trades yet.")
    else:
        st.info("No trade log file found.")
except Exception as e:
    st.error(f"Failed to load trade log: {e}")

# Refresh button
st.divider()
if st.button("🔄 Refresh", type="primary"):
    st.cache_data.clear()
    st.rerun()
