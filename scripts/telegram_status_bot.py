"""Telegram Status Bot — reply to 'status' with portfolio overview.

Runs as a daemon alongside the monitor. Listens for messages
and replies with current positions, P&L, and order status.

Usage:
    python -m scripts.telegram_status_bot
"""

import json
import logging
import os
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
    except Exception:
        pass
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

            # Coverage check: every held ticker should have BOTH stop and limit
            held_tickers = {p.contract.symbol for p in ib.positions() if p.position > 0}
            for tk in sorted(held_tickers):
                c = coverage.get(tk, {"stop": False, "limit": False})
                if not c["stop"] or not c["limit"]:
                    missing = []
                    if not c["stop"]:
                        missing.append("STOP")
                    if not c["limit"]:
                        missing.append("LIMIT")
                    lines.append(f"  ⚠️ {tk}: MISSING {'+'.join(missing)}")
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

    return (
        f"<b>💰 P&L Summary</b>\n\n"
        f"{today_emoji} <b>Today (realized):</b> ${realized_today:+.2f}\n"
        f"{un_emoji} <b>Open (unrealized):</b> ${unrealized:+.2f}\n"
        f"{total_emoji} <b>Lifetime (closed):</b> ${realized_total:+.2f}\n\n"
        f"<b>Record:</b> {wins}W / {losses}L "
        f"({win_rate:.0f}% win rate, {total_closes} trades)"
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


def main():
    logger.info("Telegram Status Bot started")
    if not TOKEN:
        logger.error("No Telegram token configured")
        return

    send_message("🤖 Status bot started. Send <b>status</b> for portfolio update.")

    offset = 0
    # Get current offset to skip old messages
    updates = get_updates(0)
    if updates:
        offset = updates[-1]["update_id"] + 1

    while True:
        try:
            updates = get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "").strip().lower()
                chat_id = str(msg.get("chat", {}).get("id", ""))

                if chat_id != CHAT_ID:
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
                elif text in ("help", "עזרה", "/help"):
                    send_message(
                        "<b>Available commands:</b>\n"
                        "• <b>status</b> — portfolio + orders\n"
                        "• <b>/pnl</b> — today + total P&L\n"
                        "• <b>/history</b> — recent closed trades\n"
                        "• <b>/panic</b> — emergency close-all (with confirm)\n"
                        "• <b>help</b> — this message"
                    )

        except Exception as e:
            logger.error("Bot error: %s", e)
            time.sleep(10)


if __name__ == "__main__":
    main()
