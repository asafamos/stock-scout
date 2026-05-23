"""Telegram Status Bot — reply to 'status' with portfolio overview.

Runs as a daemon alongside the monitor. Listens for messages
and replies with current positions, P&L, and order status.

Also includes the IB Gateway 2FA watchdog (merged from
scripts/ibkey_telegram_bot.py on 2026-05-16) so only one process
calls Telegram's getUpdates — two competing pollers caused silent
409 Conflict drops where commands stopped getting replies.

Usage:
    python -m scripts.telegram_status_bot

Optional env vars:
    TRADE_IBKEY_WATCHDOG=1    enable IB Gateway 2FA watchdog thread
                              (auto-enabled when the `ibgateway` docker
                              container is present)
    TRADE_BOT_HEARTBEAT_SEC=300  heartbeat log interval (default 5min)
"""

import json
import logging
import os
import re
import subprocess
import threading
import time
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TRADE_TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TRADE_TELEGRAM_CHAT_ID", "")

if not TOKEN:
    try:
        from pathlib import Path
        secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            for line in secrets_path.read_text().splitlines():
                if "TELEGRAM_TOKEN" in line and "=" in line:
                    TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
                if "TELEGRAM_CHAT_ID" in line and "=" in line:
                    CHAT_ID = line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass


def send_message(text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logger.error("Send failed: %s", e)


def get_updates(offset: int = 0) -> list:
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{TOKEN}/getUpdates",
            params={"offset": offset, "timeout": 30},
            timeout=35,
        )
        if resp.ok:
            return resp.json().get("result", [])
        if resp.status_code == 409:
            # Another instance is polling getUpdates with the same token.
            # Silently swallowing this caused the May 16 outage where the
            # bot looked alive but messages went nowhere. Log loudly + back
            # off so duplicate pollers are obvious in journalctl.
            logger.error(
                "Telegram 409 Conflict — another getUpdates poller is "
                "competing for this token. Check: ps aux | grep telegram"
            )
            time.sleep(15)
        else:
            logger.warning(
                "Telegram getUpdates HTTP %s: %s",
                resp.status_code, resp.text[:200],
            )
    except Exception as e:
        logger.warning("getUpdates exception: %s", e)
    return []


def get_portfolio_status() -> str:
    """Get live portfolio status from IBKR."""
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect("127.0.0.1", 7496, clientId=99, timeout=10)

        lines = ["<b>📊 Portfolio Status</b>\n"]

        # Positions
        portfolio = ib.portfolio()
        total_pnl = 0.0
        for p in portfolio:
            if p.position != 0:
                pnl_emoji = "🟢" if p.unrealizedPNL >= 0 else "🔴"
                lines.append(
                    f"{pnl_emoji} <b>{p.contract.symbol}</b>: "
                    f"{int(p.position)} shares @ ${p.averageCost:.2f}\n"
                    f"   Mkt: ${p.marketPrice:.2f} | "
                    f"PnL: ${p.unrealizedPNL:+.2f}"
                )
                total_pnl += p.unrealizedPNL

        if not any(p.position != 0 for p in portfolio):
            lines.append("No open positions")

        # Orders
        ib.reqAllOpenOrders()
        ib.sleep(1)
        orders = [t for t in ib.trades()
                  if t.orderStatus.status in ("Submitted", "PreSubmitted")]

        if orders:
            lines.append("\n<b>🛡 Protective Orders</b>")
            # Build per-ticker coverage map (stop-side + limit-sell)
            coverage: dict = {}  # ticker -> {"stop": bool, "limit": bool}
            for t in orders:
                o = t.order
                sym = t.contract.symbol
                c = coverage.setdefault(sym, {"stop": False, "limit": False})
                if o.orderType in ("TRAIL", "STP", "STP LMT", "STOP_LIMIT"):
                    c["stop"] = True
                elif o.orderType == "LMT":
                    c["limit"] = True

            for t in orders:
                o = t.order
                if o.orderType == "TRAIL":
                    lines.append(
                        f"  {t.contract.symbol}: TRAIL {o.trailingPercent}% GTC ✅"
                    )
                elif o.orderType == "STP":
                    # Hard stop (from ratcheting — locks profit at fixed price)
                    lines.append(
                        f"  {t.contract.symbol}: STOP ${o.auxPrice:.2f} GTC ✅"
                    )
                elif o.orderType == "LMT":
                    lines.append(
                        f"  {t.contract.symbol}: LIMIT ${o.lmtPrice:.2f} GTC ✅"
                    )
                elif o.orderType in ("STP LMT", "STOP_LIMIT"):
                    lines.append(
                        f"  {t.contract.symbol}: STP-LMT ${o.auxPrice:.2f}/${o.lmtPrice:.2f} GTC ✅"
                    )

            # Coverage check: every held ticker should have BOTH stop and limit.
            # 2026-05-21: for cash<$2k accounts, LIMIT orders are systematically
            # rejected by IB (Error 201 — margin requirement of OCA group). The
            # monitor's _target_hit_pass is the software replacement that fires
            # a market sell when current_price >= target_price. So a missing
            # LIMIT in those accounts is EXPECTED, not an alert condition.
            # We still flag missing STOP loudly because that's never tolerable.
            held_tickers = {p.contract.symbol for p in ib.positions() if p.position > 0}
            # Detect cash<$2k tier from account summary (we already read it above)
            try:
                _net_liq = 0.0
                for it in ib.accountSummary():
                    if it.tag == "NetLiquidation":
                        _net_liq = float(it.value or 0); break
                cash_under_2k = (0 < _net_liq < 2000)
            except Exception:
                cash_under_2k = False

            for tk in sorted(held_tickers):
                c = coverage.get(tk, {"stop": False, "limit": False})
                missing = []
                if not c["stop"]:
                    missing.append("STOP")  # always critical
                if not c["limit"] and not cash_under_2k:
                    missing.append("LIMIT")  # only critical above $2k
                if missing:
                    lines.append(f"  ⚠️ {tk}: MISSING {'+'.join(missing)}")
                elif not c["limit"] and cash_under_2k:
                    # Informational note — software replacement is active
                    lines.append(
                        f"  ℹ️ {tk}: target via monitor (LIMIT blocked by IB cash<$2k tier)"
                    )
        else:
            lines.append("\n⚠️ No protective orders!")

        # Account
        lines.append("")
        for item in ib.accountSummary():
            if item.tag == "NetLiquidation":
                lines.append(f"💰 Net: ${float(item.value):,.2f}")
            elif item.tag == "TotalCashValue":
                lines.append(f"💵 Cash: ${float(item.value):,.2f}")

        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        lines.append(f"{pnl_emoji} Total PnL: ${total_pnl:+.2f}")

        ib.disconnect()
        return "\n".join(lines)

    except Exception as e:
        return f"⚠️ Cannot connect to IB Gateway: {e}\n\nTry: http://87.99.142.12:5800/vnc.html"


def get_pnl_summary() -> str:
    """Today's P&L + running totals from trade_log + live unrealized."""
    from datetime import date
    try:
        from pathlib import Path
        log_path = Path(__file__).resolve().parents[1] / "data" / "trades" / "trade_log.json"
        log = json.loads(log_path.read_text()) if log_path.exists() else []
    except Exception:
        log = []

    today = date.today().isoformat()
    realized_today = sum(
        float(t.get("pnl", 0) or 0)
        for t in log
        if t.get("action") == "CLOSE" and str(t.get("timestamp", "")).startswith(today)
    )
    realized_total = sum(
        float(t.get("pnl", 0) or 0)
        for t in log if t.get("action") == "CLOSE"
    )
    wins = sum(1 for t in log if t.get("action") == "CLOSE" and float(t.get("pnl", 0) or 0) > 0)
    losses = sum(1 for t in log if t.get("action") == "CLOSE" and float(t.get("pnl", 0) or 0) < 0)
    total_closes = wins + losses

    # Unrealized from IB
    unrealized = 0.0
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect("127.0.0.1", 7496, clientId=97, timeout=10)
        for p in ib.portfolio():
            if p.position != 0:
                unrealized += float(p.unrealizedPNL or 0)
        ib.disconnect()
    except Exception as e:
        logger.warning("Could not fetch unrealized P&L: %s", e)

    win_rate = (wins / total_closes * 100) if total_closes else 0.0
    today_emoji = "🟢" if realized_today >= 0 else "🔴"
    total_emoji = "🟢" if realized_total >= 0 else "🔴"
    un_emoji = "🟢" if unrealized >= 0 else "🔴"

    # Audit M7: surface MAX DRAWDOWN and PROFIT FACTOR — both already
    # computed by analytics, just weren't exposed via /pnl. Profit factor
    # = gross wins ÷ gross losses (>1.0 = strategy is positive expectancy).
    # Max DD = peak-to-trough decline of the realized equity curve.
    extra_block = ""
    try:
        from core.trading.analytics import (
            compute_drawdown,
            build_equity_curve,
            pair_buy_sell_events,
        )
        pairs = pair_buy_sell_events(log)
        if pairs and len(pairs) >= 2:
            curve = build_equity_curve(pairs, starting_balance=1000.0)
            dd = compute_drawdown(curve)
            max_dd_pct = float(dd.get("max_dd_pct", 0) or 0)

            # Profit factor — gross wins ÷ |gross losses|
            gross_wins = sum(
                float(t.get("pnl", 0) or 0)
                for t in log if t.get("action") == "CLOSE"
                and float(t.get("pnl", 0) or 0) > 0
            )
            gross_losses = abs(sum(
                float(t.get("pnl", 0) or 0)
                for t in log if t.get("action") == "CLOSE"
                and float(t.get("pnl", 0) or 0) < 0
            ))
            pf = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")
            pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
            pf_emoji = "🟢" if pf >= 1.5 else ("🟡" if pf >= 1.0 else "🔴")
            dd_emoji = "🟢" if max_dd_pct < 5 else ("🟡" if max_dd_pct < 10 else "🔴")
            extra_block = (
                f"\n<b>Profit Factor:</b> {pf_emoji} {pf_str} "
                f"<i>(gross wins ÷ gross losses)</i>\n"
                f"<b>Max Drawdown:</b> {dd_emoji} -{max_dd_pct:.1f}%"
            )
    except Exception as _e:
        logger.debug("PF/DD enrichment skipped: %s", _e)

    return (
        f"<b>💰 P&L Summary</b>\n\n"
        f"{today_emoji} <b>Today (realized):</b> ${realized_today:+.2f}\n"
        f"{un_emoji} <b>Open (unrealized):</b> ${unrealized:+.2f}\n"
        f"{total_emoji} <b>Lifetime (closed):</b> ${realized_total:+.2f}\n\n"
        f"<b>Record:</b> {wins}W / {losses}L "
        f"({win_rate:.0f}% win rate, {total_closes} trades)"
        f"{extra_block}"
    )


def get_recent_trades(limit: int = 10) -> str:
    """Show last N closed trades."""
    try:
        from pathlib import Path
        log_path = Path(__file__).resolve().parents[1] / "data" / "trades" / "trade_log.json"
        log = json.loads(log_path.read_text()) if log_path.exists() else []
    except Exception:
        return "⚠️ Could not read trade log"

    closes = [t for t in log if t.get("action") == "CLOSE"]
    recent = closes[-limit:][::-1]  # newest first
    if not recent:
        return "<b>📜 Trade History</b>\n\nNo closed trades yet."

    lines = [f"<b>📜 Last {len(recent)} Closed Trades</b>\n"]
    for t in recent:
        pnl = float(t.get("pnl", 0) or 0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        when = str(t.get("timestamp", ""))[:10]
        reason = str(t.get("reason", ""))[:20]
        lines.append(
            f"{emoji} <b>{t.get('ticker', '?')}</b> "
            f"${pnl:+.2f} <i>({when}, {reason})</i>"
        )
    return "\n".join(lines)


def _panic_scan(execute: bool) -> tuple:
    """Shared logic for preview + execute. Returns (summary_text, count, notional)."""
    from ib_insync import IB, Stock, MarketOrder
    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=96, timeout=15)
    try:
        ib.reqAllOpenOrders()
        ib.sleep(1)
        open_orders = list(ib.openTrades())
        positions = [p for p in ib.positions() if p.position != 0]

        lines = []
        lines.append(f"📋 <b>{len(open_orders)}</b> open orders to cancel")
        lines.append(f"📦 <b>{len(positions)}</b> positions to close:")
        total_notional = 0.0
        for p in positions:
            qty = p.position
            sym = p.contract.symbol
            cost = p.avgCost * qty
            total_notional += cost
            lines.append(f"  • {sym}: {qty:g} shares (cost ${cost:,.2f})")
        lines.append(f"💵 Total notional: <b>${total_notional:,.2f}</b>")

        if execute:
            # Cancel all orders first
            cancelled = 0
            for tr in open_orders:
                try:
                    ib.cancelOrder(tr.order)
                    cancelled += 1
                except Exception as e:
                    logger.error("Cancel failed for %s: %s", tr.contract.symbol, e)
            ib.sleep(2)

            # Market-sell each position
            sold = 0
            for p in positions:
                try:
                    contract = Stock(p.contract.symbol, "SMART", "USD")
                    ib.qualifyContracts(contract)
                    order = MarketOrder("SELL", abs(p.position))
                    order.tif = "DAY"
                    ib.placeOrder(contract, order)
                    sold += 1
                except Exception as e:
                    logger.error("Sell failed for %s: %s", p.contract.symbol, e)
                    lines.append(f"  ❌ {p.contract.symbol}: {e}")
            ib.sleep(3)
            lines.append(f"\n✅ Cancelled {cancelled} orders, submitted {sold} market sells")

        return "\n".join(lines), len(positions), total_notional
    finally:
        ib.disconnect()


def get_diagnostic() -> str:
    """Diagnose why the latest scan would or wouldn't produce buys.

    Loads the most recent scan parquet, runs the top-15 candidates
    through `policy.evaluate_static_gates`, shows verdict per ticker
    plus aggregate skip-reason histogram. When the user asks "why
    didn't it buy?" this is the answer in one Telegram message.

    Cheap and read-only — no IB calls, no order flow.
    """
    try:
        from pathlib import Path
        import pandas as pd
        from core.trading.config import CONFIG
        from core.trading.policy import evaluate_static_gates
        from core.trading.position_tracker import PositionTracker
    except Exception as e:
        return f"⚠️ /diag import failed: {e}"

    # Load latest scan from disk (the trader's actual source of truth)
    candidates = [
        Path("data/scans/latest_scan.parquet"),
        Path("data/scans/latest_scan_live.parquet"),
        Path("data/scans/latest_scan_live.json"),
    ]
    scan_path = None
    for p in candidates:
        if p.exists() and (scan_path is None or p.stat().st_mtime > scan_path.stat().st_mtime):
            scan_path = p
    if scan_path is None:
        return "⚠️ No scan file found on disk."

    try:
        if scan_path.suffix == ".parquet":
            df = pd.read_parquet(scan_path)
        else:
            df = pd.read_json(scan_path)
    except Exception as e:
        return f"⚠️ Failed to load scan: {e}"

    import time as _time
    age_min = (_time.time() - scan_path.stat().st_mtime) / 60

    # Pull current state for the gate evaluator (paused, throttle, held)
    try:
        tracker = PositionTracker()
        held = {p.get("ticker", "").upper() for p in tracker.get_open_positions()}
    except Exception:
        held = set()

    state = {
        "paused": False,
        "throttle": {},
        "positions": [{"ticker": t} for t in held],
    }

    # Evaluate top 15 by score
    score_col = "FinalScore_20d" if "FinalScore_20d" in df.columns else "Score"
    top = df.nlargest(15, score_col)

    eligible = []
    skipped = []
    skip_reasons: dict = {}

    for _, row in top.iterrows():
        result = evaluate_static_gates(row, cfg=CONFIG, state=state)
        ticker = str(row.get("Ticker", "?"))
        score = float(row.get(score_col, 0) or 0)
        rr = float(row.get("RewardRisk", 0) or 0)
        sector = str(row.get("Sector", ""))[:14]
        if result.would_buy:
            eligible.append((ticker, score, rr, sector))
        else:
            primary = result.primary_reason or "unknown"
            skipped.append((ticker, primary[:48]))
            # Categorize for histogram
            for cat in ("Score", "R:R", "ML", "Confidence", "Blocked sector",
                        "Reliability", "regime", "Already holding", "Stop", "Target"):
                if cat.lower() in primary.lower():
                    skip_reasons[cat] = skip_reasons.get(cat, 0) + 1
                    break
            else:
                skip_reasons["Other"] = skip_reasons.get("Other", 0) + 1

    # Whole-scan aggregate (not just top 15)
    all_evals = []
    for _, row in df.iterrows():
        all_evals.append(evaluate_static_gates(row, cfg=CONFIG, state=state))
    total_eligible = sum(1 for r in all_evals if r.would_buy)

    # Build the response
    lines = [
        f"<b>🎯 /diag — top scan candidates</b>",
        f"<i>Source: {scan_path.name}, {len(df)} rows, {age_min:.0f}m old</i>",
        f"<i>Held: {', '.join(sorted(held)) or 'none'}</i>",
        f"<i>Cap: {CONFIG.max_open_positions} open / {CONFIG.max_daily_buys} daily</i>",
        "",
    ]

    if eligible:
        lines.append(f"<b>🚀 Eligible (top 15): {len(eligible)}</b>")
        for tk, sc, rr, sec in eligible[:8]:
            lines.append(f"  ✓ <code>{tk}</code> score={sc:.1f} R:R={rr:.1f} {sec}")
        if len(eligible) > 8:
            lines.append(f"  ... +{len(eligible) - 8} more")
    else:
        lines.append("<b>⏭ Top 15: 0 eligible</b>")

    if skipped:
        lines.append("")
        lines.append(f"<b>⏭ Skipped ({len(skipped)}/15):</b>")
        for tk, reason in skipped[:6]:
            lines.append(f"  • <code>{tk}</code>: {reason}")

    if skip_reasons:
        lines.append("")
        lines.append("<b>Skip reason histogram (top 15):</b>")
        for cat, cnt in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"  {cnt:2d}× {cat}")

    lines.append("")
    lines.append(f"<i>Whole scan ({len(df)} rows): {total_eligible} pass static gates</i>")
    lines.append("<i>Note: only static gates checked. Runtime gates "
                 "(earnings, sector momentum, correlation, IB-side) "
                 "may still reject during execution.</i>")

    return "\n".join(lines)


def panic_preview() -> str:
    try:
        summary, _, _ = _panic_scan(execute=False)
        return (
            f"👀 <b>PANIC PREVIEW (no action taken)</b>\n\n{summary}\n\n"
            f"To execute: reply <code>/panic confirm</code>"
        )
    except Exception as e:
        return f"❌ Panic preview failed: {e}"


def panic_execute() -> str:
    try:
        summary, _, _ = _panic_scan(execute=True)
        return f"🚨 <b>PANIC EXECUTED</b>\n\n{summary}"
    except Exception as e:
        return f"❌ <b>PANIC FAILED — ACTION NEEDED</b>\n\n{e}\n\nCheck IB directly."


# ──────────────────────────────────────────────────────────────────────
# IB Gateway 2FA watchdog (merged from scripts/ibkey_telegram_bot.py
# on 2026-05-16 to eliminate two-poller getUpdates conflict). Runs in
# a background thread inside this process. When the Gateway log shows
# a 2FA challenge, the watchdog selects "IB Key", screenshots the
# challenge, and pushes it to Telegram. The user's reply (a 6+ digit
# numeric message) is intercepted by the main message loop while
# WATCHDOG_STATE["awaiting_2fa_code"] is True, then typed into the
# Gateway window via xdotool.
# ──────────────────────────────────────────────────────────────────────

DOCKER_CONTAINER = os.getenv("TRADE_IBGW_CONTAINER", "ibgateway")
WATCHDOG_CHECK_INTERVAL = 30  # seconds
WATCHDOG_RESPONSE_TIMEOUT = 300  # 5 min to enter the 2FA code

WATCHDOG_STATE: dict = {
    "awaiting_2fa_code": False,
    "deadline_ts": 0.0,
}


def _docker_exec(cmd: str) -> str:
    try:
        result = subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=15,
        )
        return (result.stdout or "") + (result.stderr or "")
    except Exception as e:
        return f"Error: {e}"


def _docker_available() -> bool:
    """True iff the ibgateway container exists. Watchdog skips otherwise."""
    try:
        r = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={DOCKER_CONTAINER}"],
            capture_output=True, text=True, timeout=5,
        )
        return bool((r.stdout or "").strip())
    except Exception:
        return False


def _gw_take_screenshot() -> str:
    _docker_exec("DISPLAY=:0 scrot /tmp/ibkey_screen.png")
    subprocess.run(
        ["docker", "cp",
         f"{DOCKER_CONTAINER}:/tmp/ibkey_screen.png",
         "/tmp/ibkey_screen.png"],
        capture_output=True, timeout=10,
    )
    return "/tmp/ibkey_screen.png"


def _send_photo(photo_path: str, caption: str = "") -> bool:
    if not TOKEN or not CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        with open(photo_path, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": CHAT_ID, "caption": caption},
                files={"photo": f},
                timeout=15,
            )
        return resp.ok
    except Exception as e:
        logger.error("Telegram photo send failed: %s", e)
        return False


def _gw_check_status() -> str:
    logs = _docker_exec("cat /tmp/ibg-jauto.log 2>/dev/null | tail -5")
    if "maintenance cycle" in logs:
        ports = _docker_exec("cat /proc/net/tcp")
        if "0FA1" in ports or "0FA0" in ports:  # 4001 / 4000 in hex
            return "connected"
        return "maintenance_disconnected"
    lower = logs.lower()
    if "two-factor" in lower or "2fa" in lower or "second factor" in lower:
        return "needs_2fa"
    if "login" in lower:
        return "logging_in"
    return "unknown"


def _gw_prompt_2fa():
    """Show the 2FA challenge to the user and arm the code-listener."""
    logger.info("Watchdog: 2FA detected — prompting user")
    _gw_take_screenshot()
    _docker_exec("DISPLAY=:0 xdotool mousemove 655 388 click 1")
    time.sleep(1)
    _docker_exec("DISPLAY=:0 xdotool mousemove 546 429 click 1")
    time.sleep(3)
    screenshot = _gw_take_screenshot()
    send_message(
        "🔐 <b>IB Gateway needs authentication!</b>\n\n"
        "Open IBKR Mobile → IB Key → Generate Response\n"
        "Look at the screen below for the challenge number.\n"
        "Reply with the response code (digits only)."
    )
    _send_photo(screenshot, "IB Gateway 2FA screen")
    WATCHDOG_STATE["awaiting_2fa_code"] = True
    WATCHDOG_STATE["deadline_ts"] = time.time() + WATCHDOG_RESPONSE_TIMEOUT


def _gw_enter_code(code: str):
    logger.info("Watchdog: entering 2FA code (len=%d)", len(code))
    _docker_exec(f"DISPLAY=:0 xdotool type '{code}'")
    time.sleep(1)
    _docker_exec("DISPLAY=:0 xdotool key Return")
    time.sleep(10)
    status = _gw_check_status()
    if status == "connected":
        send_message("✅ IB Gateway connected.")
    else:
        send_message(f"⚠️ Gateway status after code: {status}")


def _watchdog_loop():
    logger.info("IB Gateway 2FA watchdog started")
    while True:
        try:
            # Clear stale awaiting-flag if user never replied
            if (WATCHDOG_STATE["awaiting_2fa_code"]
                    and time.time() > WATCHDOG_STATE["deadline_ts"]):
                logger.warning("Watchdog: 2FA reply timeout — clearing flag")
                WATCHDOG_STATE["awaiting_2fa_code"] = False

            if not WATCHDOG_STATE["awaiting_2fa_code"]:
                status = _gw_check_status()
                if status == "needs_2fa":
                    _gw_prompt_2fa()
        except Exception:
            logger.exception("Watchdog error")
        time.sleep(WATCHDOG_CHECK_INTERVAL)


def _maybe_start_watchdog():
    """Start the IB Gateway watchdog thread if env opt-in or container exists."""
    opt_in = os.getenv("TRADE_IBKEY_WATCHDOG", "").lower() in ("1", "true", "yes")
    if not (opt_in or _docker_available()):
        logger.info(
            "IB Gateway watchdog disabled "
            "(no '%s' container and TRADE_IBKEY_WATCHDOG not set)",
            DOCKER_CONTAINER,
        )
        return
    t = threading.Thread(target=_watchdog_loop, name="ibgw-watchdog", daemon=True)
    t.start()


def main():
    logger.info("Telegram Status Bot started (PID %d)", os.getpid())
    if not TOKEN:
        logger.error("No Telegram token configured")
        return

    _maybe_start_watchdog()
    heartbeat_sec = int(os.getenv("TRADE_BOT_HEARTBEAT_SEC", "300"))
    last_heartbeat = time.time()

    # Record startup time BEFORE announcing — so messages sent IN RESPONSE
    # to the "Status bot started" announce don't get caught in the
    # skip-backlog logic below.
    startup_ts = int(time.time())

    send_message("🤖 Status bot started. Send <b>status</b> for portfolio update.")

    # Skip messages sent BEFORE we were alive (avoids re-executing stale
    # /sell or /panic commands across restarts) — but PROCESS anything
    # newer than startup_ts (a /status sent right after seeing the
    # announce above must be honored, not dropped).
    #
    # 2026-05-23: previous logic was `offset = updates[-1]["update_id"] + 1`
    # which unconditionally skipped EVERY pending update including the
    # /status the user sent in direct response to the announce. With
    # frequent restarts (4 in 10 minutes during deploy iteration today),
    # the user fell into this race every time. Now we filter by Telegram's
    # message `date` (unix epoch) and only skip messages older than startup.
    offset = 0
    updates = get_updates(0)
    if updates:
        # Walk in order. Advance offset past each OLD message; stop at the
        # first NEW message so it stays in the queue for the main loop to
        # pick up on the next getUpdates(offset) call.
        for u in updates:
            msg_ts = int(u.get("message", {}).get("date", 0) or 0)
            if msg_ts < startup_ts - 5:  # 5s grace for clock skew
                offset = u["update_id"] + 1
            else:
                # First new message — leave it (and everything after) in
                # the queue. Setting offset = u["update_id"] means the next
                # call returns starting from this update.
                offset = u["update_id"]
                break
        else:
            # All updates were old — advance offset past the last one
            offset = updates[-1]["update_id"] + 1

    while True:
        try:
            now = time.time()
            if now - last_heartbeat >= heartbeat_sec:
                logger.info(
                    "heartbeat: offset=%d awaiting_2fa=%s",
                    offset, WATCHDOG_STATE["awaiting_2fa_code"],
                )
                last_heartbeat = now

            updates = get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "").strip().lower()
                chat_id = str(msg.get("chat", {}).get("id", ""))

                if chat_id != CHAT_ID:
                    continue

                # 2FA reply interception — when the watchdog is awaiting a
                # response code, a numeric message (≥6 digits) is treated
                # as the IB Key response and typed into the Gateway window.
                if WATCHDOG_STATE["awaiting_2fa_code"]:
                    digits = re.sub(r"\D", "", text)
                    if len(digits) >= 6:
                        logger.info("Watchdog: received 2FA code from user")
                        try:
                            _gw_enter_code(digits)
                        except Exception as _e:
                            send_message(f"⚠️ 2FA entry failed: {_e}")
                        WATCHDOG_STATE["awaiting_2fa_code"] = False
                        continue

                if text in ("status", "סטטוס", "s", "/status"):
                    logger.info("Status requested")
                    status = get_portfolio_status()
                    send_message(status)
                elif text in ("/pnl", "pnl", "רווח"):
                    logger.info("P&L requested")
                    send_message(get_pnl_summary())
                elif text in ("/history", "history", "היסטוריה"):
                    logger.info("History requested")
                    send_message(get_recent_trades(10))
                elif text in ("/panic preview", "panic preview", "פאניקה תצוגה"):
                    logger.info("Panic preview requested")
                    send_message(panic_preview())
                elif text in ("/panic confirm", "panic confirm", "פאניקה אישור"):
                    logger.warning("🚨 PANIC CONFIRMED — closing all positions")
                    send_message(panic_execute())
                elif text in ("/panic", "panic", "פאניקה"):
                    send_message(
                        "🚨 <b>PANIC MODE</b>\n\n"
                        "This will <b>cancel all orders</b> and "
                        "<b>close all positions</b> at market.\n\n"
                        "To proceed: reply <code>/panic confirm</code>\n"
                        "To preview (no action): reply <code>/panic preview</code>"
                    )
                # ── New commands via shared command_bus ──────────────
                elif text in ("/today", "today", "היום"):
                    logger.info("Today log requested")
                    try:
                        from core.control.command_bus import execute as _exec
                        r = _exec("today", source="telegram")
                        actions = r.get("actions", [])
                        if not actions:
                            send_message("📋 No actions today.")
                        else:
                            lines = [f"📋 <b>Today's actions ({len(actions)}):</b>"]
                            for a in actions[-15:]:
                                ts = str(a.get("timestamp", ""))[11:19]
                                act = a.get("action", "")
                                tk = a.get("ticker", "")
                                px = a.get("price", "?")
                                pnl = a.get("pnl", "")
                                emoji = {"OPEN": "🟢", "CLOSE": "🔴",
                                         "PARTIAL": "🟡"}.get(act, "•")
                                pnl_str = f" P&L ${pnl:+.2f}" if isinstance(pnl, (int, float)) else ""
                                lines.append(f"  {emoji} {ts} {act} {tk} @ ${px}{pnl_str}")
                            send_message("\n".join(lines))
                    except Exception as _e:
                        send_message(f"⚠️ today failed: {_e}")
                elif text in ("/pause", "pause", "השהה"):
                    logger.info("Pause requested")
                    try:
                        from core.control.command_bus import execute as _exec
                        r = _exec("pause", source="telegram")
                        send_message(f"⏸ Auto-trading paused ({r.get('paused_until')})")
                    except Exception as _e:
                        send_message(f"⚠️ pause failed: {_e}")
                elif text.startswith("/pause "):
                    try:
                        from core.control.command_bus import execute as _exec
                        days = int(text.split()[1])
                        r = _exec("pause", source="telegram", days=days)
                        send_message(f"⏸ Paused for {days}d (until {r.get('paused_until')})")
                    except Exception as _e:
                        send_message(f"⚠️ {_e}")
                elif text in ("/resume", "resume", "המשך"):
                    logger.info("Resume requested")
                    try:
                        from core.control.command_bus import execute as _exec
                        _exec("resume", source="telegram")
                        send_message("▶️ Auto-trading resumed.")
                    except Exception as _e:
                        send_message(f"⚠️ resume failed: {_e}")
                elif text in ("/scan", "scan", "סריקה"):
                    logger.info("Manual scan triggered")
                    try:
                        from core.control.command_bus import execute as _exec
                        r = _exec("scan", source="telegram")
                        if r.get("already_running"):
                            send_message(
                                f"⏳ Scan already running.\n{r.get('url', '')}\n"
                                f"Started: {r.get('started_at', '?')}"
                            )
                        elif r.get("ok"):
                            send_message("🔍 Scan dispatched — watch GH Actions.")
                        else:
                            send_message(f"⚠️ {r.get('error', 'unknown error')}")
                    except Exception as _e:
                        send_message(f"⚠️ scan failed: {_e}")
                elif text.startswith("/sell "):
                    try:
                        from core.control.command_bus import execute as _exec
                        ticker = text.split()[1].upper()
                        send_message(f"⏳ Selling {ticker}...")
                        r = _exec("sell", source="telegram", ticker=ticker)
                        if r.get("ok"):
                            send_message(
                                f"✅ Sold {r.get('qty')} x {ticker} @ "
                                f"${r.get('fill_price', 0):.2f}"
                            )
                        else:
                            send_message(f"❌ {r.get('error', 'sell failed')}")
                    except Exception as _e:
                        send_message(f"⚠️ {_e}")
                elif text.startswith("/block "):
                    try:
                        from core.control.command_bus import execute as _exec
                        parts = text.split()
                        ticker = parts[1].upper()
                        days = int(parts[2]) if len(parts) > 2 else 30
                        r = _exec("block", source="telegram", ticker=ticker, days=days)
                        send_message(
                            f"🚫 Blocked {ticker} until {r.get('blocked_until', '?')[:10]}"
                        )
                    except Exception as _e:
                        send_message(f"⚠️ {_e}")
                elif text.startswith("/unblock "):
                    try:
                        from core.control.command_bus import execute as _exec
                        ticker = text.split()[1].upper()
                        _exec("unblock", source="telegram", ticker=ticker)
                        send_message(f"✅ {ticker} unblocked")
                    except Exception as _e:
                        send_message(f"⚠️ {_e}")
                elif text.startswith("/resubmit "):
                    try:
                        from core.control.command_bus import execute as _exec
                        ticker = text.split()[1].upper()
                        send_message(f"⏳ Resubmitting protection for {ticker}...")
                        r = _exec("resubmit", source="telegram", ticker=ticker)
                        if r.get("ok"):
                            send_message(f"✅ {ticker} protected.")
                        else:
                            send_message(f"❌ {r.get('error', 'resubmit failed')}")
                    except Exception as _e:
                        send_message(f"⚠️ {_e}")
                elif text in ("/login", "login", "/relogin", "relogin",
                              "התחבר", "כניסה", "/ib", "ib"):
                    logger.info("IB Gateway relogin requested")
                    try:
                        from core.control.command_bus import execute as _exec
                        send_message("🔄 Restarting IB Gateway — approve the push on IBKR Mobile when it arrives...")
                        r = _exec("login", source="telegram")
                        if r.get("ok"):
                            send_message(
                                "✅ IB Gateway restarted.\n"
                                "📱 Open IBKR Mobile and approve the push notification.\n"
                                "Send <b>status</b> in ~60s to verify the session is back."
                            )
                        else:
                            send_message(f"❌ {r.get('error', 'login failed')}")
                    except Exception as _e:
                        send_message(f"⚠️ login failed: {_e}")
                elif text in ("/diag", "diag", "דיאג", "/debug"):
                    logger.info("Diag requested")
                    try:
                        send_message(get_diagnostic())
                    except Exception as _e:
                        send_message(f"⚠️ /diag failed: {_e}")
                elif text in ("help", "עזרה", "/help"):
                    send_message(
                        "<b>📋 Available commands:</b>\n\n"
                        "<b>VIEW</b>\n"
                        "• <b>status</b> — portfolio + orders\n"
                        "• <b>/today</b> — today's actions\n"
                        "• <b>/pnl</b> — P&L summary\n"
                        "• <b>/history</b> — recent closed trades\n"
                        "• <b>/diag</b> — debug top scan candidates\n\n"
                        "<b>CONTROL</b>\n"
                        "• <b>/pause [N]</b> — pause auto-trading [N days]\n"
                        "• <b>/resume</b> — resume auto-trading\n"
                        "• <b>/scan</b> — trigger manual scan\n"
                        "• <b>/login</b> — restart IB Gateway → fresh push to IBKR Mobile\n\n"
                        "<b>POSITIONS</b>\n"
                        "• <b>/sell TICKER</b> — close one position\n"
                        "• <b>/resubmit TICKER</b> — re-place protective orders\n"
                        "• <b>/block TICKER [N]</b> — do-not-buy [N days]\n"
                        "• <b>/unblock TICKER</b> — remove from block list\n\n"
                        "<b>EMERGENCY</b>\n"
                        "• <b>/panic</b> — close everything (with confirm)"
                    )

        except Exception as e:
            logger.error("Bot error: %s", e)
            time.sleep(10)


def _run_forever():
    """Outer crash-restart wrapper.

    main() is not allowed to silently exit. If it returns or throws,
    we log the reason and sleep with exponential backoff before
    restarting. systemd's Restart=always covers process kills; this
    covers in-process exceptions that would otherwise leave the
    bot 'alive but mute' (as happened on 2026-05-16).
    """
    backoff = 5
    while True:
        try:
            main()
            logger.error("main() returned cleanly — restarting in %ds", backoff)
        except KeyboardInterrupt:
            logger.info("Shutdown via SIGINT")
            return
        except SystemExit:
            raise
        except Exception:
            logger.exception("main() crashed — restarting in %ds", backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, 300)


if __name__ == "__main__":
    _run_forever()
