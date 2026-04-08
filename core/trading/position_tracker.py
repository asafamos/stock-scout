"""Persistent position tracking — JSON file-based.

Tracks open positions and trade history independently of IBKR,
so we can reconcile and monitor even when not connected.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

from core.trading.config import CONFIG

logger = logging.getLogger(__name__)


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

    def get_open_positions(self) -> List[dict]:
        try:
            return json.loads(self._positions_path.read_text())
        except Exception:
            return []

    def get_trade_log(self) -> List[dict]:
        try:
            return json.loads(self._log_path.read_text())
        except Exception:
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
        positions = self.get_open_positions()

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

        self._save_positions(positions)
        self._log_trade("OPEN", ticker, quantity, entry_price, {
            "stop_loss": stop_loss,
            "target_price": target_price,
            "target_date": target_date,
            "score": score,
        })
        logger.info("Position added: %s x%d @ $%.2f", ticker, quantity, entry_price)

    def remove_position(self, ticker: str, exit_price: float = 0.0,
                        reason: str = "closed"):
        positions = self.get_open_positions()
        removed = [p for p in positions if p["ticker"] == ticker]
        remaining = [p for p in positions if p["ticker"] != ticker]

        if not removed:
            logger.warning("No position found for %s", ticker)
            return

        self._save_positions(remaining)
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

    def _save_positions(self, positions: List[dict]):
        self._positions_path.write_text(
            json.dumps(positions, indent=2, default=str)
        )

    def _log_trade(self, action: str, ticker: str, qty: int,
                   price: float, extra: dict):
        log = self.get_trade_log()
        log.append({
            "action": action,
            "ticker": ticker,
            "quantity": qty,
            "price": price,
            "timestamp": datetime.utcnow().isoformat(),
            **extra,
        })
        self._log_path.write_text(
            json.dumps(log, indent=2, default=str)
        )

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
