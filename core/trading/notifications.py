"""Telegram notification service for trade alerts.

Setup:
    1. Create a bot via @BotFather on Telegram → get BOT_TOKEN
    2. Send /start to your bot, then get your chat ID via
       https://api.telegram.org/bot<TOKEN>/getUpdates
    3. Set env vars: TRADE_TELEGRAM_TOKEN, TRADE_TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TRADE_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TRADE_TELEGRAM_CHAT_ID", "")
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _send(text: str) -> bool:
    """Send a Telegram message. Returns True on success."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured — skipping notification")
        return False
    try:
        resp = requests.post(
            TELEGRAM_API.format(token=TELEGRAM_TOKEN),
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return True
        logger.warning("Telegram send failed: %s", resp.text[:200])
        return False
    except Exception as e:
        logger.error("Telegram error: %s", e)
        return False


# ── Trade Notifications ──────────────────────────────────────

def notify_buy(ticker: str, qty: int, price: float,
               stop: float, target: float, score: float,
               trail_pct: float = 5.0, rr: float = 0.0,
               target_date: str = ""):
    rr_line = f"  R:R: {rr:.1f}\n" if rr > 0 else ""
    date_line = f"  Exit by: {target_date}\n" if target_date else ""
    _send(
        f"<b>BUY {ticker}</b>\n"
        f"  Qty: {qty} shares @ ${price:.2f}\n"
        f"  Stop: {trail_pct:.1f}% trailing (${stop:.2f})\n"
        f"  Target: ${target:.2f} (+{(target/price-1)*100:.1f}%)\n"
        f"{rr_line}"
        f"  Score: {score:.1f}\n"
        f"{date_line}"
        f"  Total: ${qty * price:,.0f}"
    )


def notify_sell(ticker: str, qty: int, price: float,
                reason: str, pnl: float = 0.0):
    emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"  # green/red circle
    _send(
        f"{emoji} <b>SELL {ticker}</b>\n"
        f"  Qty: {qty} shares @ ${price:.2f}\n"
        f"  Reason: {reason}\n"
        f"  P&L: ${pnl:+,.2f}"
    )


def notify_scan_complete(total: int, candidates: int, bought: int):
    _send(
        f"<b>Scan Complete</b>\n"
        f"  Scanned: {total} stocks\n"
        f"  Candidates: {candidates}\n"
        f"  Bought: {bought}"
    )


def notify_error(context: str, error: str):
    _send(
        f"\u26a0\ufe0f <b>Error: {context}</b>\n"
        f"  {error[:500]}"
    )


def notify_daily_summary(positions: List[Dict], cash: float,
                         total_value: float):
    if not positions:
        pos_text = "  No open positions"
    else:
        pos_lines = []
        for p in positions:
            pos_lines.append(
                f"  {p.get('ticker')}: {p.get('quantity')} @ "
                f"${p.get('entry_price', 0):.2f}"
            )
        pos_text = "\n".join(pos_lines)

    _send(
        f"<b>Daily Summary</b>\n"
        f"  Cash: ${cash:,.0f}\n"
        f"  Portfolio: ${total_value:,.0f}\n"
        f"  Positions: {len(positions)}\n\n"
        f"{pos_text}"
    )


def notify_emergency(message: str):
    _send(f"\U0001f6a8 <b>EMERGENCY</b>\n{message}")
