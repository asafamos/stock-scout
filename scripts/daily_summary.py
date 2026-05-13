"""End-of-day performance summary — fires via cron at market close.

Sends a Telegram message covering:
  - Today's realized P&L (from trade_log)
  - Open positions with current P&L (from IB live)
  - Win/loss count for the day
  - Lifetime stats (win rate, total P&L)
  - Slippage on any fills today (if tracked)

Invoke:
    python -m scripts.daily_summary

Schedule (cron suggested 16:15 ET = 20:15 UTC during DST):
    15 20 * * 1-5  /home/stockscout/stock-scout-2/.venv/bin/python -m scripts.daily_summary
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TRADE_TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TRADE_TELEGRAM_CHAT_ID", "")

if not TOKEN:
    secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            if "TELEGRAM_TOKEN" in line and "=" in line:
                TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
            if "TELEGRAM_CHAT_ID" in line and "=" in line:
                CHAT_ID = line.split("=", 1)[1].strip().strip('"').strip("'")


ROOT = Path(__file__).resolve().parents[1]
TRADE_LOG = ROOT / "data" / "trades" / "trade_log.json"
OPEN_POS = ROOT / "data" / "trades" / "open_positions.json"


def _load_json(path: Path) -> list:
    try:
        return json.loads(path.read_text()) if path.exists() else []
    except Exception:
        return []


def send_telegram(text: str) -> None:
    if not TOKEN or not CHAT_ID:
        logger.error("Telegram not configured")
        print(text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as e:
        logger.error("Telegram send failed: %s", e)


def _get_ib_portfolio() -> list:
    """Return live IB positions with unrealized P&L. Safe-fails to []."""
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect("127.0.0.1", 7496, clientId=95, timeout=15)
        data = [
            {
                "ticker": p.contract.symbol,
                "qty": p.position,
                "avg_cost": p.averageCost if hasattr(p, "averageCost") else p.avgCost,
                "market_price": float(p.marketPrice or 0),
                "market_value": float(p.marketValue or 0),
                "unrealized": float(p.unrealizedPNL or 0),
            }
            for p in ib.portfolio() if p.position != 0
        ]
        ib.disconnect()
        return data
    except Exception as e:
        logger.warning("Could not fetch IB portfolio: %s", e)
        return []


def build_summary(for_date: str = None) -> str:
    for_date = for_date or date.today().isoformat()
    log = _load_json(TRADE_LOG)

    # Today's events
    todays_closes = [
        t for t in log
        if t.get("action") in ("CLOSE", "PARTIAL")
        and str(t.get("timestamp", "")).startswith(for_date)
    ]
    todays_opens = [
        t for t in log
        if t.get("action") == "OPEN"
        and str(t.get("timestamp", "")).startswith(for_date)
    ]
    realized_today = sum(float(t.get("pnl", 0) or 0) for t in todays_closes)
    wins_today = sum(1 for t in todays_closes if float(t.get("pnl", 0) or 0) > 0)
    losses_today = sum(1 for t in todays_closes if float(t.get("pnl", 0) or 0) < 0)

    # Lifetime
    all_closes = [t for t in log if t.get("action") == "CLOSE"]
    realized_total = sum(float(t.get("pnl", 0) or 0) for t in all_closes)
    life_wins = sum(1 for t in all_closes if float(t.get("pnl", 0) or 0) > 0)
    life_losses = sum(1 for t in all_closes if float(t.get("pnl", 0) or 0) < 0)
    life_n = life_wins + life_losses
    win_rate = life_wins / life_n * 100 if life_n else 0.0

    # Expectancy per trade
    avg_win = (sum(float(t.get("pnl", 0)) for t in all_closes if float(t.get("pnl", 0) or 0) > 0) / life_wins) if life_wins else 0.0
    avg_loss = (sum(float(t.get("pnl", 0)) for t in all_closes if float(t.get("pnl", 0) or 0) < 0) / life_losses) if life_losses else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss) if life_n else 0.0

    # Open positions from IB
    portfolio = _get_ib_portfolio()
    unrealized_total = sum(p["unrealized"] for p in portfolio)

    lines = [f"<b>📊 Daily Summary — {for_date}</b>\n"]

    # Today
    today_emoji = "🟢" if realized_today >= 0 else "🔴"
    lines.append(
        f"{today_emoji} <b>Today's Realized P&L: ${realized_today:+.2f}</b>"
    )
    if todays_closes or todays_opens:
        lines.append(f"  Opened: {len(todays_opens)}  Closed: {len(todays_closes)}  ({wins_today}W / {losses_today}L)")
    else:
        lines.append("  <i>No trades today</i>")

    # Today's closed trades detail
    if todays_closes:
        lines.append("\n<b>📜 Closed Today:</b>")
        for t in todays_closes:
            pnl = float(t.get("pnl", 0) or 0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            reason = str(t.get("reason", ""))[:25]
            lines.append(
                f"  {emoji} {t.get('ticker')}: ${pnl:+.2f} <i>({reason})</i>"
            )

    # Open positions
    if portfolio:
        lines.append(f"\n<b>📦 Open Positions ({len(portfolio)}):</b>")
        un_emoji = "🟢" if unrealized_total >= 0 else "🔴"
        for p in portfolio:
            pct = (p["unrealized"] / (p["avg_cost"] * abs(p["qty"])) * 100) if p["avg_cost"] else 0
            pe = "🟢" if p["unrealized"] >= 0 else "🔴"
            lines.append(
                f"  {pe} {p['ticker']}: {int(p['qty'])} @ ${p['avg_cost']:.2f} "
                f"→ ${p['market_price']:.2f} ({pct:+.1f}%, ${p['unrealized']:+.2f})"
            )
        lines.append(f"  {un_emoji} <b>Total Unrealized: ${unrealized_total:+.2f}</b>")

    # Slippage tracking (added 2026-05-13): average slippage on buys
    # over the last 7 days vs lifetime. Negative slippage means we
    # filled BELOW scan-time expectation (better for us on buys).
    # Sustained negative → MKT order is fine. Sustained positive >0.5%
    # → consider switching to LIMIT to control entry price.
    from datetime import timedelta as _td
    cutoff_7d = (datetime.utcnow() - _td(days=7)).isoformat()
    slip_7d = [
        float(t.get("slippage_pct"))
        for t in log
        if t.get("action") == "OPEN"
        and t.get("slippage_pct") is not None
        and str(t.get("timestamp", "")) >= cutoff_7d
    ]
    slip_life = [
        float(t.get("slippage_pct"))
        for t in log
        if t.get("action") == "OPEN" and t.get("slippage_pct") is not None
    ]

    # Strategic metrics (added 2026-05-07): profit factor, hold time,
    # close-reason ratio. These are the three numbers that tell us if
    # the strategy has edge — separate from the headline P&L which can
    # be dominated by a single trade.
    gross_wins = sum(float(t.get("pnl", 0)) for t in all_closes if float(t.get("pnl", 0) or 0) > 0)
    gross_losses = abs(sum(float(t.get("pnl", 0)) for t in all_closes if float(t.get("pnl", 0) or 0) < 0))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

    # Hold time analysis (paired OPEN+CLOSE)
    from collections import defaultdict
    opens_by_ticker = defaultdict(list)
    for t in log:
        if t.get("action") == "OPEN":
            opens_by_ticker[t.get("ticker")].append(t)
    hold_days_list = []
    for c in all_closes:
        opens = opens_by_ticker.get(c.get("ticker"), [])
        if opens:
            o = opens.pop(0)
            try:
                ts_o = datetime.fromisoformat(str(o.get("timestamp", ""))[:19])
                ts_c = datetime.fromisoformat(str(c.get("timestamp", ""))[:19])
                hold_days_list.append((ts_c - ts_o).days)
            except Exception:
                pass
    avg_hold = sum(hold_days_list) / len(hold_days_list) if hold_days_list else 0.0

    # Close reason breakdown — proves whether trail-too-tight is killing us
    close_reasons = defaultdict(int)
    for t in all_closes:
        r = str(t.get("reason", "unknown"))
        # Normalize reasons into buckets
        if "trail" in r.lower():
            close_reasons["trail_fired"] += 1
        elif "target" in r.lower():
            close_reasons["target_hit"] += 1
        elif "ladder" in r.lower():
            close_reasons["ladder_partial"] += 1
        else:
            close_reasons[r[:20] or "unknown"] += 1

    # Lifetime
    lifetime_emoji = "🟢" if realized_total >= 0 else "🔴"
    pf_emoji = "🟢" if profit_factor >= 1.5 else ("🟡" if profit_factor >= 1.0 else "🔴")
    lines.append(
        f"\n<b>📈 Lifetime</b>\n"
        f"  {lifetime_emoji} Realized: ${realized_total:+.2f}\n"
        f"  Record: {life_wins}W / {life_losses}L ({win_rate:.0f}%)\n"
        f"  Expectancy: ${expectancy:+.2f}/trade"
        + (f"\n  Avg win: ${avg_win:+.2f}  |  Avg loss: ${avg_loss:+.2f}" if life_n else "")
    )

    # Strategic Edge Indicators
    if life_n >= 3:
        lines.append(
            f"\n<b>🎯 Edge Indicators</b>\n"
            f"  {pf_emoji} Profit Factor: <b>{profit_factor:.2f}</b> "
            f"(target ≥1.5 for &quot;has edge&quot;)\n"
            f"  Avg hold: <b>{avg_hold:.1f}d</b> (target 5-15d for swing)"
        )
        if close_reasons:
            reason_str = " | ".join(
                f"{k}:{v}" for k, v in sorted(close_reasons.items(), key=lambda x: -x[1])
            )
            lines.append(f"  Close reasons: <code>{reason_str}</code>")

        # Slippage analytics
        if slip_life:
            avg_slip_life = sum(slip_life) / len(slip_life)
            avg_slip_7d = sum(slip_7d) / len(slip_7d) if slip_7d else None
            slip_emoji = "🟢" if avg_slip_life < 0 else ("🟡" if avg_slip_life < 0.3 else "🔴")
            slip_line = (
                f"  {slip_emoji} Slippage lifetime: <b>{avg_slip_life:+.2f}%</b> "
                f"({len(slip_life)} buys)"
            )
            if avg_slip_7d is not None:
                slip_line += f" | 7d: {avg_slip_7d:+.2f}% ({len(slip_7d)})"
            lines.append(slip_line)
            # Recommendation
            if avg_slip_life > 0.5 and len(slip_life) >= 5:
                lines.append(
                    "  <i>Slippage &gt;0.5% sustained — consider LIMIT orders on $50+ stocks.</i>"
                )

    return "\n".join(lines)


if __name__ == "__main__":
    msg = build_summary()
    send_telegram(msg)
    logger.info("Daily summary sent (%d chars)", len(msg))
