"""Event-sourced execution ledger — the deep fix for tracker↔IB drift.

WHY THIS EXISTS
---------------
The legacy design kept TWO authoritative stores of position & P&L state —
the local JSON tracker (``open_positions.json`` / ``trade_log.json``) and
IBKR — and continuously tried (and failed for months) to keep them in
sync. Every prior "fix" (grace windows, miss-counters, drift suppression,
reconcile_adopt/drop) was reconciliation logic: treating the symptom of a
dual-source-of-truth architecture. Drift is the inevitable consequence,
not a bug to patch.

THE MODEL
---------
IB is the single source of truth. This ledger is an append-only,
idempotent projection of IB's OWN executions, keyed by ``exec_id`` (IB's
globally-unique, stable execution id). Consequences:

  * An execution can never be counted twice  → kills the double-CLOSE class.
  * An execution is never estimated/inferred  → kills the phantom-CLOSE
    class (KNX 2026-04-28, PAAS 2026-06-03 that vanished with pnl=None).
  * ``realized_pnl`` is IB's CommissionReport value — NET OF COMMISSIONS —
    so the ledger reconciles to NetLiquidation by construction.

WHAT'S TRUTH WHERE
------------------
  * "Do we hold X / qty / avg cost / unrealized"  → IB live (never cached
    as authoritative).
  * "Realized P&L per close (net of fees)"         → this ledger.
  * "Lifetime realized (headline)"                 → NetLiq − starting_capital
    − open_unrealized (pure IB; always true, covers pre-ledger history too).
  * "Our metadata (score, thesis, target_date, intended trail, opened_at)"
    → the local positions file, which is now a metadata ANNOTATION layer
    keyed by ticker, NOT a source of truth for existence or P&L.

``ib.fills()`` is session-scoped (~today). The durable ledger accumulates
across sessions because the monitor ingests every cycle and only ever
APPENDS what IB reports — it never re-derives, so a stale session cannot
corrupt history.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from core.trading.config import CONFIG

logger = logging.getLogger(__name__)


def _ledger_path(cfg=None) -> Path:
    cfg = cfg or CONFIG
    return Path(getattr(cfg, "ledger_path", "data/trades/executions.jsonl"))


def _parse_time(ts) -> Optional[datetime]:
    if not ts:
        return None
    s = str(ts).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            dt = datetime.fromisoformat(s[:19])
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ─────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────

def load(cfg=None) -> List[dict]:
    """Load all ledger executions (oldest first). Safe-fails to []."""
    path = _ledger_path(cfg)
    if not path.exists():
        return []
    rows: List[dict] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("ledger: skipping corrupt line: %.80s", line)
    except Exception as e:
        logger.error("ledger load failed: %s", e)
        return []
    return rows


def seen_exec_ids(cfg=None) -> set:
    return {r.get("exec_id") for r in load(cfg) if r.get("exec_id")}


# ─────────────────────────────────────────────────────────────────────
# Write (idempotent append, keyed by exec_id)
# ─────────────────────────────────────────────────────────────────────

def _append_rows(rows: List[dict], cfg=None) -> int:
    """Append rows under an exclusive lock. Returns count written."""
    if not rows:
        return 0
    path = _ledger_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("")
    fd = open(path, "a")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        for r in rows:
            fd.write(json.dumps(r, default=str) + "\n")
        fd.flush()
        os.fsync(fd.fileno())
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        fd.close()
    return len(rows)


def ingest(client, cfg=None, source: str = "ib_fills") -> List[dict]:
    """Pull IB executions and upsert any NEW ones (idempotent by exec_id).

    Returns the list of newly-recorded executions so callers (the monitor)
    can react to fresh fills — e.g. fire a SELL notification and an
    opportunistic-buy check when a position closes. This replaces the old
    "diff positions to infer a close" logic that was the drift engine.
    """
    cfg = cfg or CONFIG
    try:
        fills = client.get_fills()
    except Exception as e:
        logger.warning("ledger.ingest: client.get_fills() failed: %s", e)
        return []
    if not fills:
        return []
    have = seen_exec_ids(cfg)
    now_iso = datetime.now(timezone.utc).isoformat()
    new_rows: List[dict] = []
    for f in fills:
        ex = f.get("exec_id")
        if not ex or ex in have:
            continue
        row = dict(f)
        row["ingested_at"] = now_iso
        row["source"] = source
        new_rows.append(row)
        have.add(ex)
    written = _append_rows(new_rows, cfg)
    if written:
        logger.info("ledger.ingest: recorded %d new execution(s): %s",
                    written, [r.get("ticker") for r in new_rows])
    return new_rows


def append_legacy(rows: List[dict], cfg=None) -> int:
    """Append legacy (pre-ledger) rows that have NO IB exec_id.

    Used once by the migration to backfill historical closed trades from
    trade_log.json so /history retains continuity. Each row is tagged
    source='legacy' and given a synthetic exec_id (``legacy:<...>``) so it
    dedups cleanly and is filterable. These do NOT count toward the
    account-truth reconciliation (which uses NetLiq − starting_capital).
    """
    have = seen_exec_ids(cfg)
    fresh = [r for r in rows if r.get("exec_id") and r["exec_id"] not in have]
    return _append_rows(fresh, cfg)


# ─────────────────────────────────────────────────────────────────────
# Derived views
# ─────────────────────────────────────────────────────────────────────

def _sells(rows: List[dict]) -> List[dict]:
    return [r for r in rows if (r.get("side") == "SELL"
                                and r.get("realized_pnl") is not None)]


def realized_pnl(since: Optional[date] = None, until: Optional[date] = None,
                 cfg=None) -> float:
    """Sum of broker realized P&L (net of commission) over SELL executions
    in [since, until]. None bounds = unbounded."""
    total = 0.0
    for r in _sells(load(cfg)):
        if since or until:
            dt = _parse_time(r.get("time"))
            d = dt.date() if dt else None
            if d is None:
                continue
            if since and d < since:
                continue
            if until and d > until:
                continue
        total += float(r.get("realized_pnl") or 0.0)
    return round(total, 2)


def realized_today(cfg=None) -> float:
    return realized_pnl(since=datetime.now(timezone.utc).date(), cfg=cfg)


def realized_ledger_total(cfg=None) -> float:
    """Cumulative ledger realized P&L (broker truth, since ledger cutover)."""
    return realized_pnl(cfg=cfg)


def closed_round_trips(cfg=None) -> List[dict]:
    """FIFO-pair BUY and SELL executions per ticker into closed trades.

    Each closed trade carries the broker's realized P&L for its SELL leg
    (net of commission). Entry price is the FIFO-weighted average of the
    matched BUY shares. Used by /history.
    """
    rows = sorted(load(cfg), key=lambda r: str(r.get("time") or ""))
    by_ticker: Dict[str, List[dict]] = {}
    for r in rows:
        tkr = r.get("ticker")
        if tkr:
            by_ticker.setdefault(tkr, []).append(r)

    trips: List[dict] = []
    for tkr, evs in by_ticker.items():
        buys: List[dict] = []  # FIFO queue of {shares_left, price, time}
        for e in evs:
            side = e.get("side")
            shares = float(e.get("shares") or 0)
            price = float(e.get("price") or 0)
            if side == "BUY" and shares > 0:
                buys.append({"left": shares, "price": price, "time": e.get("time")})
            elif side == "SELL" and shares > 0:
                # Match against FIFO buys for entry price + hold days.
                need = shares
                cost_sum = 0.0
                matched = 0.0
                first_buy_time = None
                while need > 0 and buys:
                    b = buys[0]
                    take = min(need, b["left"])
                    cost_sum += take * b["price"]
                    matched += take
                    if first_buy_time is None:
                        first_buy_time = b["time"]
                    b["left"] -= take
                    need -= take
                    if b["left"] <= 1e-9:
                        buys.pop(0)
                entry = (cost_sum / matched) if matched > 0 else 0.0
                edt = _parse_time(first_buy_time)
                xdt = _parse_time(e.get("time"))
                hold_days = (xdt - edt).days if (edt and xdt) else None
                trips.append({
                    "ticker": tkr,
                    "entry_price": round(entry, 4) if entry else None,
                    "exit_price": round(price, 4),
                    "shares": shares,
                    "realized_pnl": (round(float(e["realized_pnl"]), 2)
                                     if e.get("realized_pnl") is not None else None),
                    "exit_time": e.get("time"),
                    "hold_days": hold_days,
                    "source": e.get("source", "ib_fills"),
                })
    trips.sort(key=lambda t: str(t.get("exit_time") or ""))
    return trips


def stats(cfg=None) -> dict:
    """Win/loss record + profit factor from broker-truth closed trades.

    Only counts SELL executions that carry a realized_pnl (IB-confirmed).
    There is no 'unreconciled / dropped' bucket here — nothing is ever
    dropped, because we only record real IB executions.
    """
    trips = [t for t in closed_round_trips(cfg) if t.get("realized_pnl") is not None]
    wins = [t for t in trips if t["realized_pnl"] > 0]
    losses = [t for t in trips if t["realized_pnl"] < 0]
    gross_w = sum(t["realized_pnl"] for t in wins)
    gross_l = abs(sum(t["realized_pnl"] for t in losses))
    pf = (gross_w / gross_l) if gross_l > 0 else (float("inf") if gross_w > 0 else 0.0)
    n = len(trips)
    holds = [t["hold_days"] for t in trips if t.get("hold_days") is not None]
    return {
        "n_closed": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
        "avg_win": round(gross_w / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_l / len(losses), 2) if losses else 0.0,
        "profit_factor": (round(pf, 2) if pf != float("inf") else None),
        "realized_total": round(sum(t["realized_pnl"] for t in trips), 2),
        "avg_hold_days": round(sum(holds) / len(holds), 1) if holds else None,
    }


# ─────────────────────────────────────────────────────────────────────
# Account-truth reconciliation — the single check that catches drift
# ─────────────────────────────────────────────────────────────────────

def lifetime_realized_truth(ib_net_liquidation: float, open_unrealized: float,
                            cfg=None) -> float:
    """The ONE always-true lifetime realized number, derived purely from IB:

        realized_lifetime = NetLiquidation − starting_capital − open_unrealized

    Independent of the ledger and of any per-trade bookkeeping, so it can
    never be inflated by dropped/double-counted/gross-of-fee rows. This is
    the headline P&L the bot should report.
    """
    cfg = cfg or CONFIG
    start = float(getattr(cfg, "starting_capital", 0.0) or 0.0)
    return round(ib_net_liquidation - start - open_unrealized, 2)


def reconcile(ib_net_liquidation: float, open_unrealized: float,
              cfg=None) -> dict:
    """Compare the ledger's cumulative realized against the account-truth
    realized. Agreement → the books are clean. Divergence → flags dropped /
    double-counted / fee-gross rows (exactly the months-old drift).

    Note: ledger total only covers post-cutover executions, so on a freshly
    migrated system ``account_truth`` (which includes ALL history) will
    legitimately exceed ``ledger_total`` by the pre-ledger realized amount.
    The migration records that baseline; ``delta`` is measured against it.
    """
    cfg = cfg or CONFIG
    start = float(getattr(cfg, "starting_capital", 0.0) or 0.0)
    tol = float(getattr(cfg, "reconcile_tolerance_usd", 5.0) or 5.0)
    account_truth = lifetime_realized_truth(ib_net_liquidation, open_unrealized, cfg)
    ledger_total = realized_ledger_total(cfg)
    baseline = _pre_ledger_baseline(cfg)
    delta = round(account_truth - (ledger_total + baseline), 2)
    return {
        "account_truth_realized": account_truth,
        "ledger_realized": ledger_total,
        "pre_ledger_baseline": round(baseline, 2),
        "delta": delta,
        "tolerance": tol,
        "ok": abs(delta) <= max(tol, 0.005 * max(abs(ib_net_liquidation), 1.0)),
        "starting_capital": round(start, 2),
        "net_liquidation": round(ib_net_liquidation, 2),
        "open_unrealized": round(open_unrealized, 2),
    }


_BASELINE_FILE = "data/trades/ledger_baseline.json"


def _pre_ledger_baseline(cfg=None) -> float:
    """Realized P&L that occurred BEFORE ledger cutover (set once by the
    migration so reconciliation has a fair baseline). 0 if not migrated."""
    p = Path(_BASELINE_FILE)
    if not p.exists():
        return 0.0
    try:
        return float(json.loads(p.read_text()).get("pre_ledger_realized", 0.0) or 0.0)
    except Exception:
        return 0.0


def set_pre_ledger_baseline(value: float) -> None:
    p = Path(_BASELINE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "pre_ledger_realized": round(float(value), 2),
        "set_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
