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
            for t in orders:
                o = t.order
                if o.orderType == "TRAIL":
                    lines.append(
                        f"  {t.contract.symbol}: TRAIL {o.trailingPercent}% GTC ✅"
                    )
                elif o.orderType == "LMT":
                    lines.append(
                        f"  {t.contract.symbol}: LIMIT ${o.lmtPrice:.2f} GTC ✅"
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
                elif text in ("help", "עזרה", "/help"):
                    send_message(
                        "<b>Available commands:</b>\n"
                        "• <b>status</b> — portfolio + orders\n"
                        "• <b>help</b> — this message"
                    )

        except Exception as e:
            logger.error("Bot error: %s", e)
            time.sleep(10)


if __name__ == "__main__":
    main()
