"""State broadcaster — VPS → GitHub state-feed branch → Streamlit.

Builds a comprehensive system state JSON every 30s and force-pushes it to
the `state-feed` branch on GitHub. Streamlit reads via the raw URL so the
dashboard reflects live VPS state without spamming `main` history.

Why this matters:
- Streamlit was reading stale `portfolio_snapshot.json` (61h old in extreme)
- Manual scan from Streamlit was blind to whether VPS pipeline is running
- Today's auto-trade decisions weren't visible — user had to guess
- Earnings dates not visible — couldn't know binary risk

This script writes a SINGLE source-of-truth JSON consumed by both
Telegram alerts (already via _send) and Streamlit (new).

Schema documented in docs/system_state_schema.md (informally below):
  pipeline: state machine (idle/dispatched/polling/trading)
  monitor:  health + drift alerts
  ib:       connection status, account tier
  account:  net liquidation, cash, day-trade tier
  positions: current holdings with earnings, peak, trail
  trade_log_today: today's actions (BUY/SELL/SKIP)
  system_health: services + scan freshness
  throttle: rolling win rate + size multiplier
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/home/stockscout/stock-scout-2") if Path("/home/stockscout").exists() \
       else Path(__file__).resolve().parents[1]

STATE_DIR = ROOT / "data" / "state"
STATE_FILE = STATE_DIR / "system_state.json"
OPEN_POSITIONS = ROOT / "data" / "trades" / "open_positions.json"
TRADE_LOG = ROOT / "data" / "trades" / "trade_log.json"
PORTFOLIO_SNAPSHOT = ROOT / "data" / "trades" / "portfolio_snapshot.json"
SCAN_PARQUET = ROOT / "data" / "scans" / "latest_scan.parquet"
PENDING_SCANS = ROOT / "data" / "outcomes" / "pending_scans.jsonl"

STATE_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _is_active(unit: str) -> bool:
    try:
        r = subprocess.run(["systemctl", "is-active", unit],
                           capture_output=True, text=True, timeout=4)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def _list_timer(unit: str) -> Optional[str]:
    """Return next-fire timestamp for a systemd timer (ISO format) or None."""
    try:
        r = subprocess.run(
            ["systemctl", "list-timers", unit, "--no-pager", "--output=json"],
            capture_output=True, text=True, timeout=5,
        )
        rows = json.loads(r.stdout) if r.stdout.strip() else []
        if rows and isinstance(rows, list):
            ts = rows[0].get("next")
            if ts and ts != "n/a" and ts != 0:
                # systemctl returns microseconds-since-epoch as int. Convert
                # to ISO format so downstream consumers can parse it.
                if isinstance(ts, int):
                    try:
                        dt = datetime.fromtimestamp(ts / 1_000_000, tz=timezone.utc)
                        return dt.isoformat()
                    except Exception:
                        return None
                return str(ts)
    except Exception:
        pass
    return None


def _journal(unit: str, since: str = "1 hour ago", lines: int = 200) -> str:
    try:
        r = subprocess.run(
            ["journalctl", "-u", unit, "--since", since, "--no-pager",
             "-o", "cat", "-n", str(lines)],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout
    except Exception:
        return ""


def _build_pipeline_state() -> Dict:
    """Read the most recent pipeline run from journal + parse outcomes."""
    # Try last 24h first — captures both today's runs and yesterday's
    # for context (so we don't show "unknown" between scheduled fires).
    j = _journal("stockscout-pipeline.service", since="24 hours ago", lines=500)
    if not j:
        # No journal at all → service has never run OR journal disabled.
        # That's "idle" not "unknown" — the timer is set up correctly,
        # we just haven't seen a fire yet.
        return {"state": "idle", "last_fire": None, "last_outcome": None}

    # Find the last START line and trace forward
    lines = j.splitlines()
    state = "idle"
    last_fire = None
    last_complete = None
    candidates_total = None
    candidates_passed = None
    skips: List[Dict] = []
    regime = None
    new_scan_hash = None

    # Most-recent-first scan
    for i, ln in enumerate(lines):
        if "Starting pipeline pipeline" in ln:
            # Extract timestamp from `[HH:MM:SS] Starting pipeline pipeline`
            m = re.search(r"\[(\d{2}:\d{2}:\d{2})\]", ln)
            if m:
                last_fire = m.group(1)
        if "✓ Workflow dispatched" in ln:
            state = "dispatched"
        if "Polling for new scan" in ln:
            state = "polling"
        if "✓ New scan detected" in ln:
            state = "trading"
            m = re.search(r"hash ([a-f0-9]+)", ln)
            if m:
                new_scan_hash = m.group(1)
        if "Recording outcomes" in ln:
            pass  # in-flight
        if "regime=" in ln:
            m = re.search(r"regime=(\w+)", ln)
            if m:
                regime = m.group(1)
        if "Triggering auto-trade" in ln:
            state = "trading"
        if "No candidates passed filters" in ln:
            candidates_passed = 0
        if "BUY " in ln and "EXECUTING" in ln:
            m = re.search(r"BUY (\d+) x (\w+)", ln)
            if m:
                # Could parse all buys but for now keep simple
                pass
        if "Trade finished" in ln:
            state = "done"
        if "Pipeline complete" in ln or "stockscout-pipeline.service: Deactivated successfully" in ln:
            state = "idle"
            last_complete = "yes"
        if "TIMEOUT after" in ln:
            state = "failed_timeout"

    # Today's skips — read trade_log for SKIP entries (from order_manager)
    today_iso = date.today().isoformat()
    log = _read_json(TRADE_LOG, default=[]) or []
    todays = [t for t in log if str(t.get("timestamp", "")).startswith(today_iso)]
    todays_buys = [t for t in todays if t.get("action") == "OPEN"]
    todays_closes = [t for t in todays if t.get("action") in ("CLOSE", "PARTIAL")]

    return {
        "state": state,
        "last_fire": last_fire,
        "last_complete": last_complete,
        "regime": regime,
        "candidates_passed": candidates_passed,
        "skips": skips,  # populated below if we can extract from journal
        "today_buys": len(todays_buys),
        "today_closes": len(todays_closes),
    }


def _build_positions_with_earnings() -> List[Dict]:
    """Augment open_positions with current PnL + earnings dates."""
    pos = _read_json(OPEN_POSITIONS, default=[]) or []
    snap = _read_json(PORTFOLIO_SNAPSHOT, default={}) or {}
    snap_positions = {p.get("ticker"): p for p in snap.get("positions", [])}

    # Earnings cache (lazy import to avoid circulars)
    try:
        sys.path.insert(0, str(ROOT))
        from core.trading.risk_manager import RiskManager
        from core.trading.position_tracker import PositionTracker
        from core.trading.ibkr_client import IBKRClient
        rm = RiskManager(IBKRClient(), PositionTracker())
        get_earnings = rm._fetch_earnings_date
    except Exception:
        get_earnings = lambda t: "none"

    out = []
    today = date.today()
    for p in pos:
        ticker = p.get("ticker", "")
        entry = float(p.get("entry_price", 0) or 0)
        qty = int(p.get("quantity", 0) or 0)
        peak = float(p.get("peak_price", entry) or entry)
        trail_pct = float(p.get("trailing_stop_pct", 0) or 0)
        target = float(p.get("target_price", 0) or 0)

        # Live mkt price from snapshot if available
        snap_p = snap_positions.get(ticker, {})
        mkt = float(snap_p.get("market_price", 0) or 0)
        pnl_pct = ((mkt / entry - 1) * 100) if entry > 0 and mkt > 0 else None
        pnl_abs = ((mkt - entry) * qty) if entry > 0 and mkt > 0 else None
        peak_pct = ((peak / entry - 1) * 100) if entry > 0 else None

        # Earnings — cheap if cached
        ed_str = get_earnings(ticker)
        days_to_earnings = None
        if ed_str and ed_str != "none":
            try:
                ed = date.fromisoformat(ed_str)
                days_to_earnings = (ed - today).days
            except Exception:
                pass

        out.append({
            "ticker": ticker,
            "qty": qty,
            "entry": round(entry, 2),
            "mkt": round(mkt, 2) if mkt else None,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "pnl_abs": round(pnl_abs, 2) if pnl_abs is not None else None,
            "peak_pct": round(peak_pct, 2) if peak_pct is not None else None,
            "trail_pct": trail_pct,
            "target": target,
            "opened_at": str(p.get("opened_at", ""))[:10],
            "earnings_date": ed_str if ed_str != "none" else None,
            "days_to_earnings": days_to_earnings,
        })
    return out


def _build_today_log() -> List[Dict]:
    today_iso = date.today().isoformat()
    log = _read_json(TRADE_LOG, default=[]) or []
    return [t for t in log if str(t.get("timestamp", "")).startswith(today_iso)]


def _build_throttle_state() -> Dict:
    """Compute current throttle state without IB connection."""
    log = _read_json(TRADE_LOG, default=[]) or []
    closes = [t for t in log if t.get("action") == "CLOSE" and t.get("pnl") is not None]
    recent = closes[-10:]  # match config default window
    n = len(recent)
    if n < 5:
        return {"active": False, "level": "inactive",
                "win_rate": None, "size_multiplier": 1.0,
                "trades_in_window": n}
    wins = sum(1 for t in recent if (t.get("pnl") or 0) > 0)
    win_rate = wins / n
    if win_rate < 0.20:
        return {"active": True, "level": "halt",
                "win_rate": round(win_rate, 3), "size_multiplier": 0.0,
                "trades_in_window": n}
    if win_rate < 0.30:
        return {"active": True, "level": "warn",
                "win_rate": round(win_rate, 3), "size_multiplier": 0.5,
                "trades_in_window": n}
    return {"active": False, "level": "inactive",
            "win_rate": round(win_rate, 3), "size_multiplier": 1.0,
            "trades_in_window": n}


def _build_health() -> Dict:
    services = ["stockscout-monitor", "stockscout-pipeline.timer",
                "stockscout-statusbot", "stockscout-healthcheck.timer",
                "stockscout-daily-summary.timer", "ibgateway"]
    health = {s: _is_active(s) for s in services}
    health["all_active"] = all(health.values())

    # Scan freshness
    scan_age_hours = None
    if SCAN_PARQUET.exists():
        scan_age_hours = round(
            (datetime.now().timestamp() - SCAN_PARQUET.stat().st_mtime) / 3600, 1
        )
    health["scan_age_hours"] = scan_age_hours

    # Outcomes freshness
    outcomes_age_days = None
    if PENDING_SCANS.exists():
        try:
            for line in reversed(PENDING_SCANS.read_text().splitlines()):
                if line.strip():
                    rec = json.loads(line)
                    d = rec.get("scan_date")
                    if d:
                        outcomes_age_days = (date.today() - date.fromisoformat(d)).days
                        break
        except Exception:
            pass
    health["outcomes_age_days"] = outcomes_age_days

    # Pipeline timer next fire
    health["pipeline_next_fire"] = _list_timer("stockscout-pipeline.timer")

    return health


def _build_account() -> Dict:
    snap = _read_json(PORTFOLIO_SNAPSHOT, default={}) or {}
    net = float(snap.get("net_liquidation", 0) or 0)
    cash = float(snap.get("cash", 0) or 0)
    if net >= 25000:
        tier = "margin_pdt"
    elif net >= 2000:
        tier = "cash"
    else:
        tier = "sub_2k"
    snap_age_min = None
    if PORTFOLIO_SNAPSHOT.exists():
        snap_age_min = round(
            (datetime.now().timestamp() - PORTFOLIO_SNAPSHOT.stat().st_mtime) / 60, 1
        )
    return {
        "net_liquidation": round(net, 2),
        "cash": round(cash, 2),
        "tier": tier,
        "snapshot_age_minutes": snap_age_min,
    }


def build_state() -> Dict:
    return {
        "schema_version": 1,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "pipeline": _build_pipeline_state(),
        "positions": _build_positions_with_earnings(),
        "trade_log_today": _build_today_log(),
        "system_health": _build_health(),
        "account": _build_account(),
        "throttle": _build_throttle_state(),
    }


def push_to_state_feed():
    """Force-push system_state.json to the state-feed branch.

    Single-commit branch — every push overwrites. Streamlit reads via
    raw URL so the contents land near-instantly without spamming main.
    """
    if not os.environ.get("GITHUB_TOKEN"):
        logger.debug("No GITHUB_TOKEN — skipping push to state-feed")
        return False

    # Use a tmpdir to avoid contaminating the main repo
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            # Init bare-ish repo
            subprocess.run(["git", "init", "-q", "-b", "state-feed", "."],
                           cwd=tmp, check=True, timeout=10)
            subprocess.run(["git", "config", "user.name", "stockscout-broadcaster"],
                           cwd=tmp, check=True, timeout=5)
            subprocess.run(["git", "config", "user.email", "broadcaster@stockscout"],
                           cwd=tmp, check=True, timeout=5)
            # Copy the state file
            (tmp / "data" / "state").mkdir(parents=True, exist_ok=True)
            shutil.copy(STATE_FILE, tmp / "data" / "state" / "system_state.json")
            # Add + commit + force-push
            subprocess.run(["git", "add", "data/state/system_state.json"],
                           cwd=tmp, check=True, timeout=5)
            ts = datetime.now(timezone.utc).isoformat()
            subprocess.run(["git", "commit", "-q", "-m", f"state {ts}"],
                           cwd=tmp, check=True, timeout=10)
            origin = f"https://x:{os.environ['GITHUB_TOKEN']}@github.com/asafamos/stock-scout.git"
            subprocess.run(["git", "remote", "add", "origin", origin],
                           cwd=tmp, check=True, timeout=5)
            r = subprocess.run(
                ["git", "push", "-q", "--force", "origin", "state-feed"],
                cwd=tmp, capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                logger.warning("state-feed push failed: %s", r.stderr)
                return False
        except Exception as e:
            logger.warning("state-feed push error: %s", e)
            return False
    return True


def main():
    state = build_state()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_FILE)
    logger.info("Wrote state to %s (%d bytes)", STATE_FILE, STATE_FILE.stat().st_size)
    # Push (best-effort)
    pushed = push_to_state_feed()
    logger.info("state-feed push: %s", "ok" if pushed else "skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
