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
# Per-user instance cache (thread-safe)
# ---------------------------------------------------------------------------
_PM_INSTANCES: Dict[str, "PortfolioManager"] = {}
_PM_LOCK = threading.Lock()


def get_portfolio_manager(user_id: str = DEFAULT_USER) -> "PortfolioManager":
    """Return (and lazily create) a PortfolioManager for *user_id*.

    If Supabase credentials are configured the returned instance uses
    Supabase (PostgreSQL) for persistence — surviving Streamlit Cloud
    redeployments.  Otherwise falls back to local DuckDB.

    Instances are cached in a dict keyed by user_id so each user gets their
    own manager instance with the correct user_id for DB queries.
    """
    if user_id not in _PM_INSTANCES:
        with _PM_LOCK:
            if user_id not in _PM_INSTANCES:
                # Try Supabase first (persistent across deploys)
                try:
                    from core.db.supabase_client import get_supabase_client

                    sb = get_supabase_client()
                    if sb is not None:
                        _PM_INSTANCES[user_id] = SupabasePortfolioManager(sb, user_id)
                        logger.info("Using Supabase portfolio backend for user=%s", user_id)
                    else:
                        _PM_INSTANCES[user_id] = PortfolioManager(get_scan_store(), user_id)
                        logger.info("Using DuckDB portfolio backend for user=%s", user_id)
                except Exception as exc:
                    logger.warning("Supabase init failed (%s), falling back to DuckDB", exc)
                    _PM_INSTANCES[user_id] = PortfolioManager(get_scan_store(), user_id)
    return _PM_INSTANCES[user_id]


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
                else:
                    # Time-stop: exit stagnant positions after halfway_days
                    # if less than min_progress_pct of entry→target has been covered.
                    try:
                        from core.scoring_config import TIME_STOP_CONFIG
                        _halfway = int(TIME_STOP_CONFIG.get("halfway_days", 10))
                        _min_prog = float(TIME_STOP_CONFIG.get("min_progress_pct", 0.30))
                        _buffer = int(TIME_STOP_CONFIG.get("buffer_days", 2))
                        if (
                            days_held >= _halfway
                            and days_held < holding - _buffer
                            and target_p is not None
                            and entry_p > 0
                        ):
                            _upside = target_p - entry_p
                            if _upside > 0:
                                _progress = (price - entry_p) / _upside
                                if _progress < _min_prog:
                                    exit_reason = "time_stop"
                    except Exception:
                        pass

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
                    # Break-even / trailing stop: ratchet stop up as position advances
                    new_stop_p = _safe_float(pos.get("stop_price"))
                    try:
                        from core.scoring_config import TRAILING_STOP_CONFIG
                        if target_p is not None and entry_p > 0:
                            _upside = target_p - entry_p
                            _progress = (price - entry_p) / _upside if _upside > 0 else 0.0
                            _be_trig = TRAILING_STOP_CONFIG.get("breakeven_trigger_pct", 0.50)
                            _trail_trig = TRAILING_STOP_CONFIG.get("trail_trigger_pct", 0.75)
                            _trail_mult = TRAILING_STOP_CONFIG.get("trail_atr_mult", 1.5)
                            _atr_pct = float(pos.get("atr_pct") or 0.03)
                            _atr_abs = entry_p * _atr_pct
                            if _progress >= _trail_trig and _atr_abs > 0:
                                # Trailing stop: price - 1.5×ATR (never below break-even)
                                _trail_stop = price - _trail_mult * _atr_abs
                                _trail_stop = max(_trail_stop, entry_p)
                                if new_stop_p is None or _trail_stop > new_stop_p:
                                    new_stop_p = _trail_stop
                            elif _progress >= _be_trig:
                                # Break-even: move stop to entry price
                                if new_stop_p is None or entry_p > new_stop_p:
                                    new_stop_p = entry_p
                    except Exception:
                        pass

                    con.execute(
                        """
                        UPDATE portfolio_positions
                        SET current_price = ?, current_return_pct = ?,
                            max_price = ?, min_price = ?, stop_price = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE position_id = ?
                        """,
                        [price, ret_pct, new_max, new_min, new_stop_p, pid],
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

    def get_portfolio_stats(self, since_date: Optional[date] = None) -> Dict[str, Any]:
        """Return portfolio-level aggregate statistics.

        Win Rate = fraction of auto-closed positions that hit the target price.
        Portfolio P&L = combined unrealized (open) + realized (closed) return
                        as a % of total cost basis across all positions.
        Manual closes are excluded — they reflect user overrides, not system perf.

        Args:
            since_date: If provided, only include positions entered/exited on or after this date.
        """
        con = self._store._connect()
        try:
            # Open positions
            open_sql = (
                "SELECT COUNT(*), COALESCE(SUM(entry_price * shares), 0), "
                "COALESCE(SUM(current_price * shares), 0) "
                "FROM portfolio_positions WHERE user_id = ? AND status = 'open'"
            )
            open_params: list = [self._user_id]
            if since_date is not None:
                open_sql += " AND entry_date >= ?"
                open_params.append(since_date)
            open_row = con.execute(open_sql, open_params).fetchone()
            open_count = int(open_row[0])
            open_invested = float(open_row[1])
            current_value = float(open_row[2])

            # Closed positions — exclude manual closes, win = hit target
            closed_sql = (
                "SELECT COUNT(*), "
                "COALESCE(SUM(CASE WHEN exit_reason = 'target' THEN 1 ELSE 0 END), 0), "
                "COALESCE(AVG(realized_return_pct), 0), "
                "COALESCE(SUM(CASE WHEN prediction_correct THEN 1 ELSE 0 END), 0), "
                "COALESCE(SUM(entry_price * shares), 0), "
                "COALESCE(SUM((realized_return_pct / 100.0) * entry_price * shares), 0) "
                "FROM portfolio_positions WHERE user_id = ? AND status = 'closed' "
                "AND COALESCE(exit_reason, '') != 'manual'"
            )
            closed_params: list = [self._user_id]
            if since_date is not None:
                closed_sql += " AND exit_date >= ?"
                closed_params.append(since_date)
            closed_row = con.execute(closed_sql, closed_params).fetchone()
            closed_count = int(closed_row[0])
            win_count = int(closed_row[1])
            avg_return = float(closed_row[2])
            correct_count = int(closed_row[3])
            closed_invested = float(closed_row[4])
            realized_pnl = float(closed_row[5])

            # Combined P&L: unrealized (open) + realized (closed) over total cost basis
            unrealized_pnl = current_value - open_invested
            total_cost_basis = open_invested + closed_invested
            combined_pnl_pct = (
                (unrealized_pnl + realized_pnl) / total_cost_basis * 100.0
                if total_cost_basis > 0 else 0.0
            )

            win_rate = win_count / closed_count if closed_count > 0 else 0.0
            prediction_accuracy = correct_count / closed_count if closed_count > 0 else 0.0

            return {
                "open_count": open_count,
                "closed_count": closed_count,
                "total_invested": open_invested,
                "current_value": current_value,
                "total_return_pct": combined_pnl_pct,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "prediction_accuracy": prediction_accuracy,
            }
        finally:
            con.close()

    def get_exit_reason_counts(self, since_date: Optional[date] = None) -> Dict[str, int]:
        """Return count of closed positions grouped by exit_reason."""
        con = self._store._connect()
        try:
            sql = (
                "SELECT exit_reason, COUNT(*) FROM portfolio_positions "
                "WHERE user_id = ? AND status = 'closed' "
                "AND COALESCE(exit_reason, '') != 'manual'"
            )
            params: list = [self._user_id]
            if since_date is not None:
                sql += " AND exit_date >= ?"
                params.append(since_date)
            sql += " GROUP BY exit_reason"
            rows = con.execute(sql, params).fetchall()
            return {r[0]: int(r[1]) for r in rows if r[0]}
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


# ---------------------------------------------------------------------------
# SupabasePortfolioManager — Supabase (PostgreSQL) backend
# ---------------------------------------------------------------------------
class SupabasePortfolioManager:
    """Supabase-backed virtual portfolio — persists across Streamlit deploys.

    API-compatible with :class:`PortfolioManager` so callers don't need to
    change.  Uses the ``supabase-py`` REST client internally.
    """

    def __init__(self, client, user_id: str = DEFAULT_USER):
        self._sb = client
        self._user_id = user_id

    # Helper -----------------------------------------------------------------
    @property
    def _table(self):
        return self._sb.table("portfolio_positions")

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
        """Add a new position. Returns position_id."""
        if self.is_in_portfolio(ticker):
            raise ValueError(f"{ticker} already in portfolio")

        position_id = uuid.uuid4().hex
        entry_dt = date.today()

        row = {
            "position_id": position_id,
            "user_id": self._user_id,
            "ticker": ticker,
            "entry_price": _safe_float(entry_price),
            "target_price": _safe_float(target_price),
            "stop_price": _safe_float(stop_price),
            "shares": shares,
            "entry_date": entry_dt.isoformat(),
            "target_date": target_date.isoformat() if target_date else None,
            "holding_days": holding_days,
            "scan_id": _safe_str(scan_id),
            "recommendation_id": _safe_str(recommendation_id),
            "final_score": _safe_float(final_score),
            "risk_class": _safe_str(risk_class),
            "sector": _safe_str(sector),
            "current_price": _safe_float(entry_price),
            "current_return_pct": 0.0,
            "max_price": _safe_float(entry_price),
            "min_price": _safe_float(entry_price),
            "status": "open",
        }
        self._table.insert(row).execute()
        logger.info("Added %s to Supabase portfolio (id=%s, entry=%.2f)",
                     ticker, position_id[:8], entry_price)
        return position_id

    def remove_position(
        self,
        position_id: str,
        exit_price: Optional[float] = None,
        exit_reason: str = "manual",
    ) -> bool:
        """Close a position manually. Returns True if found and closed."""
        resp = (
            self._table
            .select("entry_price, current_price")
            .eq("position_id", position_id)
            .eq("status", "open")
            .limit(1)
            .execute()
        )
        if not resp.data:
            return False

        row = resp.data[0]
        entry_p = float(row["entry_price"])
        actual_exit = float(exit_price) if exit_price is not None else float(row.get("current_price") or entry_p)
        ret_pct = ((actual_exit / entry_p) - 1.0) * 100.0 if entry_p > 0 else 0.0
        correct = ret_pct > 0

        (
            self._table
            .update({
                "exit_price": actual_exit,
                "exit_date": date.today().isoformat(),
                "exit_reason": exit_reason,
                "realized_return_pct": ret_pct,
                "prediction_correct": correct,
                "status": "closed",
            })
            .eq("position_id", position_id)
            .execute()
        )
        logger.info("Closed position %s (%s, ret=%.1f%%)", position_id[:8], exit_reason, ret_pct)
        return True

    # ------------------------------------------------------------------
    # Price updates + auto-close
    # ------------------------------------------------------------------
    def update_prices(self, as_of_date: Optional[date] = None) -> Dict[str, Any]:
        """Fetch current prices and update all open positions."""
        if as_of_date is None:
            as_of_date = date.today()

        open_df = self.get_open_positions()
        if open_df.empty:
            return {"updated": 0, "auto_closed": 0, "open_count": 0}

        tickers = open_df["ticker"].unique().tolist()
        prices = self._fetch_prices(tickers, as_of_date)

        updated = 0
        auto_closed = 0

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

            # Check exit conditions
            stop_p = _safe_float(pos.get("stop_price"))
            target_p = _safe_float(pos.get("target_price"))
            holding = _safe_int(pos.get("holding_days")) or 20
            entry_date = pos.get("entry_date")
            days_held = 0
            if entry_date is not None:
                try:
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
            else:
                # Time-stop: exit stagnant positions after halfway_days
                try:
                    from core.scoring_config import TIME_STOP_CONFIG
                    _halfway = int(TIME_STOP_CONFIG.get("halfway_days", 10))
                    _min_prog = float(TIME_STOP_CONFIG.get("min_progress_pct", 0.30))
                    _buffer = int(TIME_STOP_CONFIG.get("buffer_days", 2))
                    if (
                        days_held >= _halfway
                        and days_held < holding - _buffer
                        and target_p is not None
                        and entry_p > 0
                    ):
                        _upside = target_p - entry_p
                        if _upside > 0:
                            _progress = (price - entry_p) / _upside
                            if _progress < _min_prog:
                                exit_reason = "time_stop"
                except Exception:
                    pass

            pid = pos["position_id"]

            if exit_reason:
                correct = ret_pct > 0
                (
                    self._table
                    .update({
                        "current_price": price,
                        "current_return_pct": ret_pct,
                        "max_price": new_max,
                        "min_price": new_min,
                        "exit_price": price,
                        "exit_date": as_of_date.isoformat(),
                        "exit_reason": exit_reason,
                        "realized_return_pct": ret_pct,
                        "prediction_correct": correct,
                        "status": "closed",
                    })
                    .eq("position_id", pid)
                    .execute()
                )
                auto_closed += 1
                logger.info("Auto-closed %s (%s, ret=%.1f%%)", tkr, exit_reason, ret_pct)
            else:
                # Break-even / trailing stop: ratchet stop up as position advances
                update_payload: Dict[str, Any] = {
                    "current_price": price,
                    "current_return_pct": ret_pct,
                    "max_price": new_max,
                    "min_price": new_min,
                }
                try:
                    from core.scoring_config import TRAILING_STOP_CONFIG
                    _cur_stop = _safe_float(pos.get("stop_price"))
                    if target_p is not None and entry_p > 0:
                        _upside = target_p - entry_p
                        _progress = (price - entry_p) / _upside if _upside > 0 else 0.0
                        _be_trig = TRAILING_STOP_CONFIG.get("breakeven_trigger_pct", 0.50)
                        _trail_trig = TRAILING_STOP_CONFIG.get("trail_trigger_pct", 0.75)
                        _trail_mult = TRAILING_STOP_CONFIG.get("trail_atr_mult", 1.5)
                        _atr_pct = float(pos.get("atr_pct") or 0.03)
                        _atr_abs = entry_p * _atr_pct
                        new_stop_p = _cur_stop
                        if _progress >= _trail_trig and _atr_abs > 0:
                            _trail_stop = max(price - _trail_mult * _atr_abs, entry_p)
                            if new_stop_p is None or _trail_stop > new_stop_p:
                                new_stop_p = _trail_stop
                        elif _progress >= _be_trig:
                            if new_stop_p is None or entry_p > new_stop_p:
                                new_stop_p = entry_p
                        if new_stop_p != _cur_stop:
                            update_payload["stop_price"] = new_stop_p
                except Exception:
                    pass
                (
                    self._table
                    .update(update_payload)
                    .eq("position_id", pid)
                    .execute()
                )
                updated += 1

        return {
            "updated": updated,
            "auto_closed": auto_closed,
            "open_count": len(open_df) - auto_closed,
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_open_positions(self) -> pd.DataFrame:
        """Return all open positions as DataFrame."""
        resp = (
            self._table
            .select("*")
            .eq("user_id", self._user_id)
            .eq("status", "open")
            .order("entry_date", desc=True)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        return pd.DataFrame(resp.data)

    def get_closed_positions(self, days: int = 90) -> pd.DataFrame:
        """Return closed positions from last N days."""
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        resp = (
            self._table
            .select("*")
            .eq("user_id", self._user_id)
            .eq("status", "closed")
            .gte("exit_date", cutoff)
            .order("exit_date", desc=True)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        return pd.DataFrame(resp.data)

    def get_portfolio_stats(self, since_date: Optional[date] = None) -> Dict[str, Any]:
        """Return portfolio-level aggregate statistics.

        Win Rate = fraction of auto-closed positions that hit the target price.
        Portfolio P&L = combined unrealized (open) + realized (closed) return
                        as a % of total cost basis across all positions.
        Manual closes are excluded — they reflect user overrides, not system perf.

        Args:
            since_date: If provided, only include positions entered/exited on or after this date.
        """
        # Open positions
        open_query = (
            self._table
            .select("entry_price, current_price, shares")
            .eq("user_id", self._user_id)
            .eq("status", "open")
        )
        if since_date is not None:
            open_query = open_query.gte("entry_date", since_date.isoformat())
        open_resp = open_query.execute()
        open_rows = open_resp.data or []
        open_count = len(open_rows)
        open_invested = sum(float(r["entry_price"] or 0) * int(r["shares"] or 100) for r in open_rows)
        current_value = sum(float(r["current_price"] or r["entry_price"] or 0) * int(r["shares"] or 100) for r in open_rows)

        # Closed positions — exclude manual closes, win = hit target
        closed_query = (
            self._table
            .select("realized_return_pct, prediction_correct, exit_reason, entry_price, shares")
            .eq("user_id", self._user_id)
            .eq("status", "closed")
            .neq("exit_reason", "manual")
        )
        if since_date is not None:
            closed_query = closed_query.gte("exit_date", since_date.isoformat())
        closed_resp = closed_query.execute()
        closed_rows = closed_resp.data or []
        closed_count = len(closed_rows)
        win_count = sum(1 for r in closed_rows if r.get("exit_reason") == "target")
        correct_count = sum(1 for r in closed_rows if r.get("prediction_correct"))
        avg_return = (
            sum(float(r.get("realized_return_pct") or 0) for r in closed_rows) / closed_count
            if closed_count > 0
            else 0.0
        )
        closed_invested = sum(
            float(r.get("entry_price") or 0) * int(r.get("shares") or 100) for r in closed_rows
        )
        realized_pnl = sum(
            (float(r.get("realized_return_pct") or 0) / 100.0)
            * float(r.get("entry_price") or 0)
            * int(r.get("shares") or 100)
            for r in closed_rows
        )

        # Combined P&L: unrealized (open) + realized (closed) over total cost basis
        unrealized_pnl = current_value - open_invested
        total_cost_basis = open_invested + closed_invested
        combined_pnl_pct = (
            (unrealized_pnl + realized_pnl) / total_cost_basis * 100.0
            if total_cost_basis > 0 else 0.0
        )

        win_rate = win_count / closed_count if closed_count > 0 else 0.0
        prediction_accuracy = correct_count / closed_count if closed_count > 0 else 0.0

        return {
            "open_count": open_count,
            "closed_count": closed_count,
            "total_invested": open_invested,
            "current_value": current_value,
            "total_return_pct": combined_pnl_pct,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "prediction_accuracy": prediction_accuracy,
        }

    def get_exit_reason_counts(self, since_date: Optional[date] = None) -> Dict[str, int]:
        """Return count of closed positions grouped by exit_reason."""
        query = (
            self._table
            .select("exit_reason")
            .eq("user_id", self._user_id)
            .eq("status", "closed")
            .neq("exit_reason", "manual")
        )
        if since_date is not None:
            query = query.gte("exit_date", since_date.isoformat())
        rows = query.execute().data or []
        counts: Dict[str, int] = {}
        for r in rows:
            reason = r.get("exit_reason")
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    def is_in_portfolio(self, ticker: str) -> bool:
        """Check if ticker has an open position."""
        resp = (
            self._table
            .select("position_id")
            .eq("user_id", self._user_id)
            .eq("ticker", ticker)
            .eq("status", "open")
            .limit(1)
            .execute()
        )
        return bool(resp.data)

    def get_portfolio_tickers(self) -> Set[str]:
        """Return set of all tickers with open positions."""
        resp = (
            self._table
            .select("ticker")
            .eq("user_id", self._user_id)
            .eq("status", "open")
            .execute()
        )
        return {r["ticker"] for r in (resp.data or [])}

    # ------------------------------------------------------------------
    # Price fetching (shared with DuckDB backend)
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
