"""Virtual portfolio manager — track paper positions from recommendations.

Usage::

    pm = get_portfolio_manager()

    # Add from recommendation card:
    pm.add_position(ticker="AAPL", entry_price=185.0, target_price=200.0,
                    stop_price=175.0, holding_days=20)

    # Daily update:
    summary = pm.update_prices()

    # Query:
    open_df = pm.get_open_positions()
    stats   = pm.get_portfolio_stats()
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from core.db.store import ScanStore, _safe_float, _safe_int, _safe_str, get_scan_store

logger = logging.getLogger("stock_scout.db.portfolio")

DEFAULT_USER = "default"
DEFAULT_SHARES = 100

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_PM_SINGLETON: Optional["PortfolioManager"] = None
_PM_LOCK = threading.Lock()


def get_portfolio_manager(user_id: str = DEFAULT_USER) -> "PortfolioManager":
    """Return (and lazily create) the global PortfolioManager singleton."""
    global _PM_SINGLETON
    if _PM_SINGLETON is None:
        with _PM_LOCK:
            if _PM_SINGLETON is None:
                _PM_SINGLETON = PortfolioManager(get_scan_store(), user_id)
    return _PM_SINGLETON


# ---------------------------------------------------------------------------
# PortfolioManager
# ---------------------------------------------------------------------------
class PortfolioManager:
    """DuckDB-backed virtual portfolio for tracking recommendation accuracy."""

    def __init__(self, store: ScanStore, user_id: str = DEFAULT_USER):
        self._store = store
        self._user_id = user_id

    # ------------------------------------------------------------------
    # Add / Remove
    # ------------------------------------------------------------------
    def add_position(
        self,
        ticker: str,
        entry_price: float,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        target_date: Optional[date] = None,
        holding_days: int = 20,
        shares: int = DEFAULT_SHARES,
        scan_id: Optional[str] = None,
        recommendation_id: Optional[str] = None,
        final_score: Optional[float] = None,
        risk_class: Optional[str] = None,
        sector: Optional[str] = None,
    ) -> str:
        """Add a new position to the virtual portfolio.

        Returns the position_id (UUID hex string).
        Raises ValueError if ticker already has an open position.
        """
        if self.is_in_portfolio(ticker):
            raise ValueError(f"{ticker} already in portfolio")

        position_id = uuid.uuid4().hex
        entry_dt = date.today()
        target_dt = target_date

        con = self._store._connect()
        try:
            con.execute(
                """
                INSERT INTO portfolio_positions (
                    position_id, user_id, ticker,
                    entry_price, target_price, stop_price, shares,
                    entry_date, target_date, holding_days,
                    scan_id, recommendation_id,
                    final_score, risk_class, sector,
                    current_price, current_return_pct, max_price, min_price,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?,  ?, ?, ?, ?,  ?, ?, ?,  ?, ?,  ?, ?, ?,
                          ?, 0.0, ?, ?,  'open', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                [
                    position_id, self._user_id, ticker,
                    _safe_float(entry_price),
                    _safe_float(target_price),
                    _safe_float(stop_price),
                    shares,
                    entry_dt,
                    target_dt,
                    holding_days,
                    _safe_str(scan_id),
                    _safe_str(recommendation_id),
                    _safe_float(final_score),
                    _safe_str(risk_class),
                    _safe_str(sector),
                    _safe_float(entry_price),  # current_price = entry at start
                    _safe_float(entry_price),  # max_price
                    _safe_float(entry_price),  # min_price
                ],
            )
            logger.info("Added %s to portfolio (id=%s, entry=%.2f)", ticker, position_id[:8], entry_price)
            return position_id
        finally:
            con.close()

    def remove_position(
        self,
        position_id: str,
        exit_price: Optional[float] = None,
        exit_reason: str = "manual",
    ) -> bool:
        """Close a position manually. Returns True if found and closed."""
        con = self._store._connect()
        try:
            row = con.execute(
                "SELECT entry_price, current_price FROM portfolio_positions "
                "WHERE position_id = ? AND status = 'open'",
                [position_id],
            ).fetchone()
            if row is None:
                return False

            entry_p = float(row[0])
            actual_exit = float(exit_price) if exit_price is not None else float(row[1] or entry_p)
            ret_pct = ((actual_exit / entry_p) - 1.0) * 100.0 if entry_p > 0 else 0.0
            correct = ret_pct > 0

            con.execute(
                """
                UPDATE portfolio_positions
                SET exit_price = ?, exit_date = ?, exit_reason = ?,
                    realized_return_pct = ?, prediction_correct = ?,
                    status = 'closed', updated_at = CURRENT_TIMESTAMP
                WHERE position_id = ?
                """,
                [actual_exit, date.today(), exit_reason, ret_pct, correct, position_id],
            )
            logger.info("Closed position %s (%s, ret=%.1f%%)", position_id[:8], exit_reason, ret_pct)
            return True
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Price updates + auto-close
    # ------------------------------------------------------------------
    def update_prices(self, as_of_date: Optional[date] = None) -> Dict[str, Any]:
        """Fetch current prices and update all open positions.

        Also evaluates exit signals (stop/target/expiry) and auto-closes
        positions that hit exit conditions.

        Returns: {updated, auto_closed, open_count}
        """
        if as_of_date is None:
            as_of_date = date.today()

        open_df = self.get_open_positions()
        if open_df.empty:
            return {"updated": 0, "auto_closed": 0, "open_count": 0}

        tickers = open_df["ticker"].unique().tolist()
        prices = self._fetch_prices(tickers, as_of_date)

        updated = 0
        auto_closed = 0
        con = self._store._connect()
        try:
            for _, pos in open_df.iterrows():
                tkr = pos["ticker"]
                price = prices.get(tkr)
                if price is None:
                    continue

                entry_p = float(pos["entry_price"])
                ret_pct = ((price / entry_p) - 1.0) * 100.0 if entry_p > 0 else 0.0
                old_max = float(pos.get("max_price") or entry_p)
                old_min = float(pos.get("min_price") or entry_p)
                new_max = max(old_max, price)
                new_min = min(old_min, price)

                # Check exit conditions (priority: stop > target > expiry)
                stop_p = _safe_float(pos.get("stop_price"))
                target_p = _safe_float(pos.get("target_price"))
                holding = _safe_int(pos.get("holding_days")) or 20
                entry_date = pos.get("entry_date")
                days_held = 0
                if entry_date is not None:
                    try:
                        # Always normalize to plain datetime.date (DuckDB returns
                        # pandas.Timestamp which inherits from date but breaks
                        # arithmetic with plain date objects)
                        ed = pd.Timestamp(entry_date).date()
                        days_held = (as_of_date - ed).days
                    except Exception:
                        pass

                exit_reason = None
                if stop_p is not None and price <= stop_p:
                    exit_reason = "stop"
                elif target_p is not None and price >= target_p:
                    exit_reason = "target"
                elif days_held >= holding:
                    exit_reason = "expiry"

                pid = pos["position_id"]

                if exit_reason:
                    correct = ret_pct > 0
                    con.execute(
                        """
                        UPDATE portfolio_positions
                        SET current_price = ?, current_return_pct = ?,
                            max_price = ?, min_price = ?,
                            exit_price = ?, exit_date = ?, exit_reason = ?,
                            realized_return_pct = ?, prediction_correct = ?,
                            status = 'closed', updated_at = CURRENT_TIMESTAMP
                        WHERE position_id = ?
                        """,
                        [price, ret_pct, new_max, new_min,
                         price, as_of_date, exit_reason,
                         ret_pct, correct, pid],
                    )
                    auto_closed += 1
                    logger.info("Auto-closed %s (%s, ret=%.1f%%)", tkr, exit_reason, ret_pct)
                else:
                    con.execute(
                        """
                        UPDATE portfolio_positions
                        SET current_price = ?, current_return_pct = ?,
                            max_price = ?, min_price = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE position_id = ?
                        """,
                        [price, ret_pct, new_max, new_min, pid],
                    )
                updated += 1

            return {
                "updated": updated,
                "auto_closed": auto_closed,
                "open_count": len(open_df) - auto_closed,
            }
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_open_positions(self) -> pd.DataFrame:
        """Return all open positions as DataFrame, sorted by entry_date desc."""
        con = self._store._connect()
        try:
            return con.execute(
                "SELECT * FROM portfolio_positions "
                "WHERE user_id = ? AND status = 'open' "
                "ORDER BY entry_date DESC",
                [self._user_id],
            ).fetchdf()
        finally:
            con.close()

    def get_closed_positions(self, days: int = 90) -> pd.DataFrame:
        """Return closed positions from last N days."""
        cutoff = date.today() - timedelta(days=days)
        con = self._store._connect()
        try:
            return con.execute(
                "SELECT * FROM portfolio_positions "
                "WHERE user_id = ? AND status = 'closed' AND exit_date >= ? "
                "ORDER BY exit_date DESC",
                [self._user_id, cutoff],
            ).fetchdf()
        finally:
            con.close()

    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Return portfolio-level aggregate statistics."""
        con = self._store._connect()
        try:
            # Open positions
            open_row = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(entry_price * shares), 0), "
                "COALESCE(SUM(current_price * shares), 0) "
                "FROM portfolio_positions WHERE user_id = ? AND status = 'open'",
                [self._user_id],
            ).fetchone()
            open_count = int(open_row[0])
            total_invested = float(open_row[1])
            current_value = float(open_row[2])

            # Closed positions
            closed_row = con.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(CASE WHEN realized_return_pct > 0 THEN 1 ELSE 0 END), 0), "
                "COALESCE(AVG(realized_return_pct), 0), "
                "COALESCE(SUM(CASE WHEN prediction_correct THEN 1 ELSE 0 END), 0) "
                "FROM portfolio_positions WHERE user_id = ? AND status = 'closed'",
                [self._user_id],
            ).fetchone()
            closed_count = int(closed_row[0])
            win_count = int(closed_row[1])
            avg_return = float(closed_row[2])
            correct_count = int(closed_row[3])

            total_return_pct = ((current_value / total_invested) - 1.0) * 100.0 if total_invested > 0 else 0.0
            win_rate = win_count / closed_count if closed_count > 0 else 0.0
            prediction_accuracy = correct_count / closed_count if closed_count > 0 else 0.0

            return {
                "open_count": open_count,
                "closed_count": closed_count,
                "total_invested": total_invested,
                "current_value": current_value,
                "total_return_pct": total_return_pct,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "prediction_accuracy": prediction_accuracy,
            }
        finally:
            con.close()

    def is_in_portfolio(self, ticker: str) -> bool:
        """Check if ticker has an open position."""
        con = self._store._connect()
        try:
            row = con.execute(
                "SELECT 1 FROM portfolio_positions "
                "WHERE user_id = ? AND ticker = ? AND status = 'open' LIMIT 1",
                [self._user_id, ticker],
            ).fetchone()
            return row is not None
        finally:
            con.close()

    def get_portfolio_tickers(self) -> Set[str]:
        """Return set of all tickers with open positions (batch check for UI)."""
        con = self._store._connect()
        try:
            rows = con.execute(
                "SELECT DISTINCT ticker FROM portfolio_positions "
                "WHERE user_id = ? AND status = 'open'",
                [self._user_id],
            ).fetchall()
            return {r[0] for r in rows}
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Price fetching (reuses OutcomeTracker pattern)
    # ------------------------------------------------------------------
    def _fetch_prices(self, symbols: List[str], as_of: date) -> Dict[str, Optional[float]]:
        """Fetch latest close prices for symbols via yfinance."""
        result: Dict[str, Optional[float]] = {}
        if not symbols:
            return result

        try:
            import yfinance as yf

            start = as_of - timedelta(days=7)
            data = yf.download(
                symbols,
                start=start.isoformat(),
                end=(as_of + timedelta(days=1)).isoformat(),
                progress=False,
                threads=True,
            )
            if data.empty:
                return result

            # Handle single vs multi-ticker response
            if len(symbols) == 1:
                ticker = symbols[0]
                close = data.get("Close")
                if close is not None and not close.empty:
                    result[ticker] = float(close.iloc[-1])
            else:
                close = data.get("Close")
                if close is not None:
                    for sym in symbols:
                        try:
                            col = close[sym] if sym in close.columns else None
                            if col is not None and not col.empty:
                                val = col.dropna().iloc[-1]
                                result[sym] = float(val)
                        except Exception:
                            pass
        except Exception as e:
            logger.warning("Price fetch failed: %s", e)

        return result
