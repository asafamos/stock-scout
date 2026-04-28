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

def _load_telegram_creds() -> tuple:
    """Load Telegram creds from env vars or .streamlit/secrets.toml."""
    token = os.getenv("TRADE_TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TRADE_TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        return token, chat_id
    # Fallback: read from .streamlit/secrets.toml (local Mac runs)
    try:
        from pathlib import Path
        secrets_path = Path(__file__).resolve().parents[2] / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            for line in secrets_path.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    key, val = line.split("=", 1)
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    if key == "TRADE_TELEGRAM_TOKEN" and not token:
                        token = val
                    elif key == "TRADE_TELEGRAM_CHAT_ID" and not chat_id:
                        chat_id = val
    except Exception:
        pass
    return token, chat_id


TELEGRAM_TOKEN, TELEGRAM_CHAT_ID = _load_telegram_creds()
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
               target_date: str = "", prefix: str = "BUY"):
    rr_line = f"  R:R: {rr:.1f}\n" if rr > 0 else ""
    date_line = f"  Exit by: {target_date}\n" if target_date else ""
    _send(
        f"<b>{prefix} {ticker}</b>\n"
        f"  Qty: {qty} shares @ ${price:.2f}\n"
        f"  Stop: {trail_pct:.1f}% trailing (${stop:.2f})\n"
        f"  Target: ${target:.2f} (+{(target/price-1)*100:.1f}%)\n"
        f"{rr_line}"
        f"  Score: {score:.1f}\n"
        f"{date_line}"
        f"  Total: ${qty * price:,.0f}"
    )


def notify_resubmit(ticker: str, qty: int, entry: float,
                    trail_pct: float, target: float,
                    peak_price: float = 0.0,
                    score: float = 0.0,
                    target_date: str = ""):
    """Notification for AUTO-RESUBMIT (re-bracketing existing position).

    Unlike notify_buy (which shows the SCAN-DERIVED stop_loss as a
    static reference), this shows the ACTUAL projected TRAIL stop based
    on the current peak price — which is what the user cares about for
    a position that's already running. If peak_price isn't provided,
    falls back to entry × (1 - trail_pct/100) as a conservative floor.
    """
    if peak_price and peak_price > entry:
        proj_stop = peak_price * (1 - trail_pct / 100)
        lock_pct = (proj_stop - entry) / entry * 100
        stop_line = (
            f"  Trail: {trail_pct:.1f}% (peak ${peak_price:.2f} → "
            f"stop ~${proj_stop:.2f}, +{lock_pct:.1f}% locked)\n"
        )
    else:
        # Position hasn't moved up yet (or peak unknown) — show initial floor
        floor = entry * (1 - trail_pct / 100)
        stop_line = f"  Trail: {trail_pct:.1f}% (initial floor ~${floor:.2f})\n"

    date_line = f"  Exit by: {target_date}\n" if target_date else ""
    score_line = f"  Score: {score:.1f}\n" if score > 0 else ""
    _send(
        f"<b>🔄 AUTO-RESUBMIT {ticker}</b>\n"
        f"  Qty: {qty} shares @ ${entry:.2f}\n"
        f"{stop_line}"
        f"  Target: ${target:.2f} (+{(target/entry-1)*100:.1f}%)\n"
        f"{score_line}"
        f"{date_line}"
        f"  Total cost basis: ${qty * entry:,.0f}"
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
