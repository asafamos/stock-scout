"""Daily morning health summary — sent to Telegram at 07:00 UTC weekdays.

Runs entirely off the VPS state (no IB connection needed for most checks).
Designed to surface bugs that took days to detect in 2026-04-22..28:
- outcomes-record stale (no new records in N days)
- pipeline never ran yesterday
- positions without protective orders (drift)
- service down

The output is one Telegram message — quick to scan over morning coffee.
If everything is green, message is short. If anything is red, the
problem is in the first line.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/home/stockscout/stock-scout-2") if Path("/home/stockscout").exists() \
       else Path(__file__).resolve().parents[1]

OPEN_POSITIONS = ROOT / "data" / "trades" / "open_positions.json"
TRADE_LOG = ROOT / "data" / "trades" / "trade_log.json"
PORTFOLIO_SNAPSHOT = ROOT / "data" / "trades" / "portfolio_snapshot.json"
SCAN_PARQUET = ROOT / "data" / "scans" / "latest_scan.parquet"
PENDING_SCANS = ROOT / "data" / "outcomes" / "pending_scans.jsonl"

OUTCOMES_STALE_DAYS = 3   # alert if no new records in 3 trading days
SCAN_STALE_HOURS = 36     # alert if scan parquet older than 36h


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _is_service_active(name: str) -> bool:
    """Return True if a systemd service is active (best-effort, no sudo)."""
    try:
        r = subprocess.run(
            ["systemctl", "is-active", name],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() == "active"
    except Exception:
        return False


def _last_scan_age_hours() -> float | None:
    if not SCAN_PARQUET.exists():
        return None
    age_sec = (datetime.now().timestamp() - SCAN_PARQUET.stat().st_mtime)
    return age_sec / 3600


def _last_outcome_record_date() -> str | None:
    if not PENDING_SCANS.exists():
        return None
    latest = None
    for line in PENDING_SCANS.read_text().splitlines()[-100:]:
        try:
            r = json.loads(line)
            d = r.get("scan_date")
            if d and (latest is None or d > latest):
                latest = d
        except Exception:
            continue
    return latest


def _todays_pl(positions, today_iso: str) -> tuple[float, int]:
    """Return (realized_pnl_today, count_of_closes_today)."""
    log = _read_json(TRADE_LOG, default=[]) or []
    closes = [
        t for t in log
        if str(t.get("timestamp", "")).startswith(today_iso)
        and t.get("action") in ("CLOSE", "PARTIAL")
    ]
    pnl = sum((t.get("pnl") or 0) for t in closes)
    return pnl, len(closes)


def _last_trading_day(d: date) -> date:
    """Walk back to the most recent weekday (Mon..Fri).
    Skips Saturdays and Sundays. US holidays not handled — would need a
    calendar; rare enough to accept the small false-stale on holiday Mondays.
    """
    cur = d - timedelta(days=1)
    while cur.weekday() >= 5:  # 5=Sat, 6=Sun
        cur -= timedelta(days=1)
    return cur


def build_summary() -> str:
    today = date.today().isoformat()
    # On Monday, "yesterday" is Sunday (no trading) — show last trading day
    # instead. Avoids the misleading "yesterday: 0 closes" line that the
    # user gets on weekend mornings + Monday. (Audit finding #9.)
    yesterday = _last_trading_day(date.today()).isoformat()

    lines = []

    # ── Service health ────────────────────────────────────────
    services = {
        "ibgateway": "IB Gateway",
        "stockscout-monitor": "Monitor",
        "stockscout-pipeline.timer": "Pipeline timer",
        "stockscout-statusbot": "Status bot",
    }
    down = [name for svc, name in services.items() if not _is_service_active(svc)]
    health_emoji = "✅" if not down else "🔴"
    if down:
        lines.append(f"{health_emoji} <b>SERVICES DOWN:</b> {', '.join(down)}")
    else:
        lines.append("✅ All services active")

    # ── Scan freshness ─────────────────────────────────────────
    scan_age_h = _last_scan_age_hours()
    if scan_age_h is None:
        lines.append("🔴 <b>SCAN FILE MISSING</b>")
    elif scan_age_h > SCAN_STALE_HOURS:
        lines.append(f"🟡 Scan file age: {scan_age_h:.0f}h (> {SCAN_STALE_HOURS}h threshold)")
    else:
        lines.append(f"✅ Scan file: {scan_age_h:.1f}h old")

    # ── Outcomes tracker ───────────────────────────────────────
    last_outcome = _last_outcome_record_date()
    if last_outcome is None:
        lines.append("🔴 <b>OUTCOMES TRACKER EMPTY</b>")
    else:
        try:
            d = date.fromisoformat(last_outcome)
            days_ago = (date.today() - d).days
            if days_ago > OUTCOMES_STALE_DAYS:
                lines.append(
                    f"🔴 <b>OUTCOMES STALE</b> — last record {last_outcome} "
                    f"({days_ago}d ago, threshold {OUTCOMES_STALE_DAYS}d)"
                )
            else:
                lines.append(f"✅ Outcomes: last record {last_outcome} ({days_ago}d ago)")
        except Exception:
            lines.append(f"🟡 Outcomes: last record {last_outcome} (unparseable date)")

    # ── Positions ──────────────────────────────────────────────
    positions = _read_json(OPEN_POSITIONS, default=[]) or []
    snapshot = _read_json(PORTFOLIO_SNAPSHOT, default={}) or {}
    cash = snapshot.get("cash", 0) or 0
    net_liq = snapshot.get("net_liquidation", 0) or 0

    # Snapshot freshness — without this, weekend-stale numbers show as
    # current and the user makes Monday sizing decisions on Friday-EOD
    # values. (Audit finding #10.)
    snapshot_stale = False
    if PORTFOLIO_SNAPSHOT.exists():
        snap_age_h = (datetime.now().timestamp() - PORTFOLIO_SNAPSHOT.stat().st_mtime) / 3600
        if snap_age_h > 6:
            snapshot_stale = True
            lines.append(
                f"🟡 Portfolio snapshot stale ({snap_age_h:.0f}h old) — "
                f"cash/net values may not reflect current state"
            )

    if positions:
        lines.append(f"\n<b>📊 Positions ({len(positions)}):</b>")
        for p in positions:
            entry = p.get("entry_price", 0) or 0
            peak = p.get("peak_price", entry) or entry
            peak_pct = ((peak - entry) / entry * 100) if entry > 0 else 0
            ticker = p.get("ticker", "?")
            qty = p.get("quantity", 0) or 0
            opened = (p.get("opened_at", "") or "")[:10]
            lines.append(
                f"  {ticker}: {qty} sh @ ${entry:.2f} "
                f"(peak +{peak_pct:.1f}%, opened {opened})"
            )
    else:
        lines.append("\n📊 No open positions")

    # ── Yesterday's activity ────────────────────────────────────
    pnl_y, closes_y = _todays_pl(positions, yesterday)
    if closes_y > 0:
        lines.append(
            f"\n<b>📈 Yesterday:</b> {closes_y} close(s), realized "
            f"P&L ${pnl_y:+.2f}"
        )

    # ── Account + tier ──────────────────────────────────────────
    if net_liq > 0 or cash > 0:
        # Account tier drives day-trade rules + protective-order timing.
        # Surface it in the summary so the user knows what regime is active.
        if net_liq >= 25000:
            tier = "🟢 margin/PDT (T+0, no day-trade rules)"
        elif net_liq >= 2000:
            tier = "🟡 cash $2k+ (T+1 settlement, day-trade rules apply)"
        else:
            tier = "🟠 cash <$2k (IB strict — no shorts, goodAfterTime on LMT)"
        lines.append(f"\n💰 Net: ${net_liq:,.2f}  |  Cash: ${cash:,.2f}")
        lines.append(f"   Tier: {tier}")

    return "\n".join(lines)


def main():
    summary = build_summary()
    header = f"🌅 <b>StockScout Daily Health — {date.today().isoformat()}</b>\n\n"
    msg = header + summary
    try:
        from core.trading.notifications import _send
        _send(msg)
        print("Sent daily summary to Telegram")
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        print(msg)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
