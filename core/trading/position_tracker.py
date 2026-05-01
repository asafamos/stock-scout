"""Persistent position tracking — JSON file-based.

Tracks open positions and trade history independently of IBKR,
so we can reconcile and monitor even when not connected.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

from core.trading.config import CONFIG

logger = logging.getLogger(__name__)


@contextmanager
def _file_lock(path: Path, mode: str = "r+"):
    """Exclusive advisory lock on `path` for concurrent-safe read-modify-write.

    Used to prevent race conditions between run_auto_trade (places buys and
    writes new positions) and monitor_positions (records partials/closes).
    fcntl.flock is advisory — both writers must use this to be protected.
    """
    # Ensure file exists so we can open for locking
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]")
    fd = open(path, mode)
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        fd.close()


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON via tempfile + rename so readers never see a half-written file.

    Combined with _file_lock on the target path, this prevents both torn
    reads and lost writes from concurrent producers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # tempfile in same directory so rename is atomic on the same filesystem
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as tmp:
            json.dump(data, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class PositionTracker:
    """File-backed position and trade log manager."""

    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self._positions_path = Path(self.cfg.open_positions_path)
        self._log_path = Path(self.cfg.trade_log_path)
        self._ensure_files()

    def _ensure_files(self):
        for p in (self._positions_path, self._log_path):
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text("[]")

    # ── Read ──────────────────────────────────────────────────
    # These return [] on ANY read error to avoid crashing callers — but
    # log the actual error at WARNING so silent JSON corruption doesn't
    # hide. "Returning []" has two wildly different meanings (no trades
    # vs. file corrupted), and the previous bare-except hid the second.

    def get_open_positions(self) -> List[dict]:
        try:
            text = self._positions_path.read_text() or "[]"
            return json.loads(text)
        except FileNotFoundError:
            return []  # expected on first run
        except json.JSONDecodeError as e:
            logger.error(
                "CORRUPTED open_positions.json (%s) — returning [] as fallback. "
                "Check disk integrity + restore from state-backup branch if needed.",
                e,
            )
            return []
        except Exception as e:
            logger.error("Failed to read open_positions.json: %s", e)
            return []

    def get_trade_log(self) -> List[dict]:
        try:
            text = self._log_path.read_text() or "[]"
            return json.loads(text)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as e:
            logger.error(
                "CORRUPTED trade_log.json (%s) — returning [] as fallback. "
                "Historical P&L analytics will be wrong until restored.",
                e,
            )
            return []
        except Exception as e:
            logger.error("Failed to read trade_log.json: %s", e)
            return []

    def get_position(self, ticker: str) -> Optional[dict]:
        for p in self.get_open_positions():
            if p.get("ticker") == ticker:
                return p
        return None

    def is_holding(self, ticker: str) -> bool:
        return self.get_position(ticker) is not None

    @property
    def open_count(self) -> int:
        return len(self.get_open_positions())

    @property
    def total_exposure(self) -> float:
        return sum(
            p.get("entry_price", 0) * p.get("quantity", 0)
            for p in self.get_open_positions()
        )

    # ── Write ─────────────────────────────────────────────────

    def add_position(
        self,
        ticker: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        target_date: Optional[str] = None,
        trailing_stop_pct: float = 0.0,
        score: float = 0.0,
        order_ids: Optional[Dict[str, int]] = None,
    ):
        # Read-modify-write under exclusive lock to prevent race with monitor
        with _file_lock(self._positions_path):
            positions = self._read_positions_unlocked()
            # Prevent duplicates
            if any(p["ticker"] == ticker for p in positions):
                logger.warning("Already holding %s — skipping add", ticker)
                return
            positions.append({
                "ticker": ticker,
                "quantity": quantity,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "target_date": target_date,
                "trailing_stop_pct": trailing_stop_pct,
                "score": score,
                "opened_at": datetime.utcnow().isoformat(),
                "order_ids": order_ids or {},
            })
            _atomic_write_json(self._positions_path, positions)

        self._log_trade("OPEN", ticker, quantity, entry_price, {
            "stop_loss": stop_loss,
            "target_price": target_price,
            "target_date": target_date,
            "score": score,
        })
        logger.info("Position added: %s x%d @ $%.2f", ticker, quantity, entry_price)

    def remove_position(self, ticker: str, exit_price: float = 0.0,
                        reason: str = "closed"):
        # ── Duplicate-CLOSE guard ──────────────────────────────────
        # If a CLOSE row for this ticker was already written today,
        # refuse to write a second one. This prevents the exact
        # scenario from 2026-05-01:
        #   1. CF was sold at 14:02 UTC, monitor wrote a clean CLOSE row.
        #   2. Operator ran `git pull` on VPS at 19:25 UTC — this
        #      overwrote `open_positions.json` with a stale committed
        #      snapshot that had CF as still-open.
        #   3. Monitor's next cycles saw CF "missing from IB" → after
        #      the consecutive-miss threshold, called remove_position
        #      again → wrote a SECOND CLOSE row with the same +$22.24
        #      P&L (re-derived from IB's still-cached fill data).
        #   4. /pnl double-counted — reported +$44.48 instead of +$22.24.
        #
        # Cheap defensive write — same-day same-ticker CLOSE never makes
        # sense in a long-only swing system (we don't open and close the
        # same name twice in one day; the day-trade gate refuses re-buys).
        try:
            today_iso = date.today().isoformat()
            log = self.get_trade_log()
            for entry in log:
                if (
                    entry.get("ticker") == ticker
                    and entry.get("action") == "CLOSE"
                    and str(entry.get("timestamp", "")).startswith(today_iso)
                ):
                    logger.warning(
                        "remove_position(%s): refusing duplicate CLOSE — "
                        "already logged at %s with P&L $%.2f. Tracker entry "
                        "will still be removed but no second CLOSE row written.",
                        ticker,
                        entry.get("timestamp", "")[:19],
                        entry.get("pnl") or 0,
                    )
                    # Still need to remove the (probably-stale) tracker entry
                    # so subsequent cycles don't keep firing this guard.
                    with _file_lock(self._positions_path):
                        positions = self._read_positions_unlocked()
                        remaining = [p for p in positions if p["ticker"] != ticker]
                        if len(remaining) != len(positions):
                            _atomic_write_json(self._positions_path, remaining)
                    return
        except Exception as _dup_err:
            # Never let the guard break a real close — if dedup fails,
            # fall through to the normal close path. Worst case: we get
            # the duplicate row, which is exactly the pre-guard behavior.
            logger.debug("duplicate-CLOSE guard skipped: %s", _dup_err)

        # Read-modify-write under exclusive lock
        with _file_lock(self._positions_path):
            positions = self._read_positions_unlocked()
            removed = [p for p in positions if p["ticker"] == ticker]
            remaining = [p for p in positions if p["ticker"] != ticker]
            if not removed:
                logger.warning("No position found for %s", ticker)
                return
            _atomic_write_json(self._positions_path, remaining)

        pos = removed[0]
        pnl = (exit_price - pos["entry_price"]) * pos["quantity"] if exit_price else 0
        self._log_trade("CLOSE", ticker, pos["quantity"], exit_price, {
            "entry_price": pos["entry_price"],
            "pnl": round(pnl, 2),
            "reason": reason,
            "held_since": pos.get("opened_at"),
        })
        logger.info("Position closed: %s @ $%.2f (P&L: $%.2f, reason: %s)",
                     ticker, exit_price, pnl, reason)

    def reconcile_drop(self, ticker: str, reason: str = "reconcile_drop"):
        """Drop a position from the tracker when IB doesn't have it,
        WITHOUT writing a CLOSE entry to trade_log.

        Used by the monitor's reconciliation flow when:
          - IB reports zero position for a ticker the tracker thinks we hold
          - We can't determine an exit price from fills/executions/trades

        Why a separate path: the old behavior wrote a phantom CLOSE row
        with an ESTIMATED exit_price (from stop_loss or peak_price),
        producing fake P&L numbers that polluted statistics, daily summary,
        and ML feedback. (See KNX phantom 2026-04-28; audit finding #1.)

        A RECONCILE_DROP entry IS written (action=RECONCILE_DROP, pnl=null)
        so analytics layers can filter these out by `action != "CLOSE"`.
        """
        with _file_lock(self._positions_path):
            positions = self._read_positions_unlocked()
            removed = [p for p in positions if p["ticker"] == ticker]
            remaining = [p for p in positions if p["ticker"] != ticker]
            if not removed:
                logger.warning("reconcile_drop: no position found for %s", ticker)
                return
            _atomic_write_json(self._positions_path, remaining)

        pos = removed[0]
        # IMPORTANT: pnl=None (not 0) — distinguishes "dropped without
        # knowing P&L" from "closed at break-even". Downstream stats code
        # already filters None.
        self._log_trade("RECONCILE_DROP", ticker, pos["quantity"], 0.0, {
            "entry_price": pos["entry_price"],
            "pnl": None,
            "reason": reason,
            "held_since": pos.get("opened_at"),
        })
        logger.warning(
            "Position %s dropped from tracker (no P&L recorded): %s",
            ticker, reason,
        )

    def reconcile_adopt(
        self,
        ticker: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        trailing_stop_pct: float = 0.0,
        oca_group: str = "",
        order_ids: Optional[Dict[str, int]] = None,
        reason: str = "reconcile_adopt",
    ):
        """Adopt an IB-only position into the tracker.

        Symmetric to reconcile_drop(): the monitor's drift check found a
        position in IB that the tracker doesn't know about. Without
        adoption the position would (a) keep firing DRIFT alerts every
        cooldown window, and (b) be invisible to monitor's exit logic
        (target_date check, peak/ratchet, earnings exit). Adoption brings
        it back under management.

        Writes a RECONCILE_ADOPT entry to trade_log so analytics can
        distinguish adopted positions from organic OPENs (action != "OPEN").
        Caller is responsible for verifying the OCA group is one of ours
        (e.g. SS_* prefix) before calling — this method does not validate.
        """
        with _file_lock(self._positions_path):
            positions = self._read_positions_unlocked()
            if any(p["ticker"] == ticker for p in positions):
                logger.warning("reconcile_adopt: %s already tracked — skipping", ticker)
                return
            positions.append({
                "ticker": ticker,
                "quantity": quantity,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "target_date": None,  # unknown for adopted positions
                "trailing_stop_pct": trailing_stop_pct,
                "score": 0.0,
                "opened_at": datetime.utcnow().isoformat(),
                "order_ids": {**(order_ids or {}), "oca_group": oca_group},
                "adopted": True,
                "adopt_reason": reason,
            })
            _atomic_write_json(self._positions_path, positions)

        # pnl=None — adoption is not a P&L event; keep stats unaffected.
        self._log_trade("RECONCILE_ADOPT", ticker, quantity, entry_price, {
            "stop_loss": stop_loss,
            "target_price": target_price,
            "trailing_stop_pct": trailing_stop_pct,
            "oca_group": oca_group,
            "reason": reason,
            "pnl": None,
        })
        logger.warning(
            "Position %s adopted into tracker (qty=%d, entry=$%.2f, reason=%s)",
            ticker, quantity, entry_price, reason,
        )

    # ── Target Date Exits ─────────────────────────────────────

    def check_target_date_exits(self) -> List[str]:
        """Return tickers whose target_date has passed."""
        today = date.today().isoformat()
        expired = []
        for p in self.get_open_positions():
            td = p.get("target_date")
            if td and td <= today:
                expired.append(p["ticker"])
        return expired

    # ── Daily Buy Counter ────────────────────────────────────

    def daily_buy_count(self) -> int:
        today = date.today().isoformat()
        return sum(
            1 for t in self.get_trade_log()
            if t.get("action") == "OPEN"
            and t.get("timestamp", "").startswith(today)
        )

    # ── Internals ─────────────────────────────────────────────

    def _read_positions_unlocked(self) -> List[dict]:
        """Read positions without taking the lock (caller is holding it)."""
        try:
            return json.loads(self._positions_path.read_text() or "[]")
        except Exception:
            return []

    def _save_positions(self, positions: List[dict]):
        """Locked atomic save — safe when called concurrently with other writers."""
        with _file_lock(self._positions_path):
            _atomic_write_json(self._positions_path, positions)

    def _log_trade(self, action: str, ticker: str, qty: int,
                   price: float, extra: dict):
        """Locked atomic append to trade log."""
        with _file_lock(self._log_path):
            try:
                log = json.loads(self._log_path.read_text() or "[]")
            except Exception:
                log = []
            log.append({
                "action": action,
                "ticker": ticker,
                "quantity": qty,
                "price": price,
                "timestamp": datetime.utcnow().isoformat(),
                **extra,
            })
            _atomic_write_json(self._log_path, log)

    def summary(self) -> str:
        positions = self.get_open_positions()
        if not positions:
            return "No open positions."
        lines = [f"Open Positions ({len(positions)}):"]
        for p in positions:
            lines.append(
                f"  {p['ticker']}: {p['quantity']} shares @ ${p['entry_price']:.2f} "
                f"| Stop: ${p['stop_loss']:.2f} | Target: ${p['target_price']:.2f} "
                f"| Opened: {p.get('opened_at', 'N/A')[:10]}"
            )
        lines.append(f"Total exposure: ${self.total_exposure:,.0f}")
        return "\n".join(lines)
