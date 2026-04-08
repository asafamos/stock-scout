"""Outcome tracker — closes the feedback loop.

After each scan produces recommendations, this module tracks what actually
happened: did the stock go up?  Did it hit the target?  The stop?  How did
it compare to SPY?

The tracker runs daily (CI or manual) and incrementally fills forward returns
as trading days pass.  After 40 trading days a recommendation is marked
``complete``.

Usage::

    store   = get_scan_store()
    tracker = OutcomeTracker(store)

    # After a new scan:
    tracker.register_recommendations(scan_id, results_df)

    # Daily update (called by CI):
    summary = tracker.update_outcomes()
    print(summary)

    # Analytics:
    perf = tracker.get_performance_summary(days=90)
    corr = tracker.get_score_vs_outcome()
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.db.store import ScanStore

logger = logging.getLogger("stock_scout.db.outcomes")

# Horizons (trading days) we track
_HORIZONS = [5, 10, 20, 40]
_COMPLETE_HORIZON = 40  # mark complete after this many trading days


class OutcomeTracker:
    """Track forward returns for past recommendations."""

    def __init__(self, store: ScanStore):
        self._store = store

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_recommendations(
        self,
        scan_id: str,
        recommendations: pd.DataFrame,
    ) -> int:
        """Register new recommendations for outcome tracking.

        Creates rows in the ``outcomes`` table with status='pending'.
        Skips rows that already exist (idempotent).

        Returns:
            Number of newly registered rows.
        """
        if recommendations is None or recommendations.empty:
            return 0

        con = self._store._connect()
        registered = 0
        try:
            for _, row in recommendations.iterrows():
                ticker = str(
                    row.get("Ticker", row.get("ticker", row.name))
                )
                if not ticker:
                    continue

                rec_id = f"{scan_id}::{ticker}"
                entry_price = _to_float(
                    row.get("Entry", row.get("Close", row.get("entry_price")))
                )
                if entry_price is None or entry_price <= 0:
                    continue

                # Determine entry date (scan date)
                entry_dt = row.get("scan_timestamp", row.get("Date"))
                if entry_dt is None:
                    entry_dt = date.today()
                elif isinstance(entry_dt, datetime):
                    entry_dt = entry_dt.date()
                elif isinstance(entry_dt, str):
                    try:
                        entry_dt = datetime.fromisoformat(
                            entry_dt.replace("Z", "+00:00")
                        ).date()
                    except Exception:
                        entry_dt = date.today()

                # Skip if already registered
                existing = con.execute(
                    "SELECT 1 FROM outcomes WHERE recommendation_id = ?",
                    [rec_id],
                ).fetchone()
                if existing:
                    continue

                con.execute(
                    """INSERT INTO outcomes
                       (recommendation_id, ticker, entry_date, entry_price,
                        status, last_updated)
                       VALUES (?, ?, ?, ?, 'pending', ?)""",
                    [
                        rec_id,
                        ticker,
                        entry_dt,
                        entry_price,
                        datetime.now(timezone.utc),
                    ],
                )
                registered += 1

            logger.info(
                "Registered %d new outcome rows for scan %s",
                registered, scan_id,
            )
            return registered
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Daily update
    # ------------------------------------------------------------------
    def update_outcomes(self, as_of_date: Optional[date] = None) -> Dict[str, Any]:
        """Fetch prices and update forward returns for pending recommendations.

        This is the main daily job.  For each pending/partial outcome:
          1. Compute trading days elapsed since entry
          2. Fetch current price via yfinance (lightweight)
          3. Update return columns as horizons are reached
          4. Track max/min prices and drawdown
          5. Check target/stop hits
          6. Update SPY returns for comparison
          7. Mark 'complete' after 40 trading days

        Returns:
            Summary dict with counts and diagnostics.
        """
        if as_of_date is None:
            as_of_date = date.today()

        con = self._store._connect()
        try:
            # Load pending/partial outcomes
            pending = con.execute(
                """SELECT recommendation_id, ticker, entry_date, entry_price,
                          return_5d, return_10d, return_20d, return_40d,
                          max_price_20d, min_price_20d
                   FROM outcomes
                   WHERE status IN ('pending', 'partial')"""
            ).fetchdf()

            if pending.empty:
                logger.info("No pending outcomes to update")
                return {"updated": 0, "completed": 0, "pending": 0}

            # Collect unique tickers to fetch prices
            tickers = pending["ticker"].unique().tolist()

            # Also fetch SPY for benchmark comparison
            all_symbols = tickers + ["SPY"]
            prices = self._fetch_prices(all_symbols, as_of_date)

            spy_price_today = prices.get("SPY")
            updated = 0
            completed = 0

            for _, row in pending.iterrows():
                rec_id = row["recommendation_id"]
                ticker = row["ticker"]
                entry_date = row["entry_date"]
                entry_price = float(row["entry_price"])

                if isinstance(entry_date, str):
                    entry_date = datetime.fromisoformat(entry_date).date()
                elif isinstance(entry_date, pd.Timestamp):
                    entry_date = entry_date.date()

                current_price = prices.get(ticker)
                if current_price is None:
                    continue

                # Approximate trading days (calendar days * 5/7)
                cal_days = (as_of_date - entry_date).days
                trading_days = int(cal_days * 5 / 7)

                # Current return
                current_return = (current_price / entry_price - 1) * 100

                # Determine which horizon columns to fill
                updates: Dict[str, Any] = {}
                if trading_days >= 5 and pd.isna(row["return_5d"]):
                    updates["return_5d"] = current_return
                if trading_days >= 10 and pd.isna(row["return_10d"]):
                    updates["return_10d"] = current_return
                if trading_days >= 20 and pd.isna(row["return_20d"]):
                    updates["return_20d"] = current_return
                if trading_days >= 40 and pd.isna(row["return_40d"]):
                    updates["return_40d"] = current_return

                # Track extremes
                prev_max = _to_float(row["max_price_20d"]) or entry_price
                prev_min = _to_float(row["min_price_20d"]) or entry_price
                new_max = max(prev_max, current_price)
                new_min = min(prev_min, current_price)
                updates["max_price_20d"] = new_max
                updates["min_price_20d"] = new_min

                if trading_days <= 20:
                    dd = (new_min / entry_price - 1) * 100
                    up = (new_max / entry_price - 1) * 100
                    updates["max_drawdown_20d"] = dd
                    updates["max_upside_20d"] = up

                # Check target/stop (from recommendations table)
                target_stop = con.execute(
                    """SELECT target_price, stop_price
                       FROM recommendations WHERE id = ?""",
                    [rec_id],
                ).fetchone()
                if target_stop:
                    target_price = target_stop[0]
                    stop_price = target_stop[1]
                    if target_price and new_max >= target_price:
                        updates["hit_target"] = True
                    if stop_price and new_min <= stop_price:
                        updates["hit_stop"] = True

                # SPY benchmark
                if spy_price_today and trading_days >= 20:
                    spy_entry = self._get_spy_price_on_date(entry_date)
                    if spy_entry and spy_entry > 0:
                        spy_ret = (spy_price_today / spy_entry - 1) * 100
                        updates["spy_return_20d"] = spy_ret
                        if "return_20d" in updates:
                            updates["excess_return_20d"] = (
                                updates["return_20d"] - spy_ret
                            )

                # Status
                new_status = "partial"
                if trading_days >= _COMPLETE_HORIZON:
                    new_status = "complete"
                    completed += 1
                updates["status"] = new_status
                updates["last_updated"] = datetime.now(timezone.utc)

                # Build UPDATE statement
                if updates:
                    set_clauses = ", ".join(
                        f"{k} = ?" for k in updates.keys()
                    )
                    vals = list(updates.values()) + [rec_id]
                    con.execute(
                        f"UPDATE outcomes SET {set_clauses} WHERE recommendation_id = ?",
                        vals,
                    )
                    updated += 1

            summary = {
                "updated": updated,
                "completed": completed,
                "pending": len(pending) - completed,
                "as_of_date": str(as_of_date),
            }
            logger.info("Outcome update: %s", summary)
            return summary
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def get_performance_summary(self, days: int = 90) -> Dict[str, Any]:
        """Aggregate performance for completed recommendations."""
        con = self._store._connect()
        try:
            days = int(days)  # sanitise
            df = con.execute(
                f"""SELECT r.final_score, r.tech_score, r.fundamental_score,
                          r.ml_prob, r.market_regime, r.sector,
                          o.return_5d, o.return_10d, o.return_20d,
                          o.max_drawdown_20d, o.max_upside_20d,
                          o.hit_target, o.hit_stop,
                          o.spy_return_20d, o.excess_return_20d
                   FROM outcomes o
                   JOIN recommendations r ON r.id = o.recommendation_id
                   WHERE o.status = 'complete'
                     AND o.entry_date >= CURRENT_DATE - INTERVAL '{days}' DAY"""
            ).fetchdf()

            if df.empty:
                return {"n_completed": 0, "message": "No completed outcomes yet"}

            ret_20d = df["return_20d"].dropna()
            excess = df["excess_return_20d"].dropna()

            return {
                "n_completed": len(df),
                "avg_return_20d": float(ret_20d.mean()) if len(ret_20d) else None,
                "median_return_20d": float(ret_20d.median()) if len(ret_20d) else None,
                "win_rate": float((ret_20d > 0).mean()) if len(ret_20d) else None,
                "avg_excess_return": float(excess.mean()) if len(excess) else None,
                "hit_target_rate": float(df["hit_target"].mean()) if "hit_target" in df else None,
                "hit_stop_rate": float(df["hit_stop"].mean()) if "hit_stop" in df else None,
                "avg_max_drawdown": float(df["max_drawdown_20d"].dropna().mean()),
                "avg_max_upside": float(df["max_upside_20d"].dropna().mean()),
                "period_days": days,
            }
        finally:
            con.close()

    def get_score_vs_outcome(self, min_scans: int = 5) -> pd.DataFrame:
        """Score-outcome correlation for weight calibration.

        Groups recommendations into score deciles and computes
        average actual return for each decile.
        """
        con = self._store._connect()
        try:
            df = con.execute(
                """SELECT r.final_score, r.tech_score, r.fundamental_score,
                          r.ml_prob, o.return_20d, o.excess_return_20d
                   FROM outcomes o
                   JOIN recommendations r ON r.id = o.recommendation_id
                   WHERE o.status = 'complete'
                     AND o.return_20d IS NOT NULL"""
            ).fetchdf()

            if len(df) < min_scans:
                return pd.DataFrame()

            # Create deciles for each score component
            result_rows = []
            for col in ["final_score", "tech_score", "fundamental_score", "ml_prob"]:
                if col not in df.columns or df[col].isna().all():
                    continue
                df["decile"] = pd.qcut(
                    df[col], q=10, labels=False, duplicates="drop"
                )
                for dec, grp in df.groupby("decile"):
                    result_rows.append({
                        "component": col,
                        "decile": int(dec),
                        "n": len(grp),
                        "avg_return_20d": float(grp["return_20d"].mean()),
                        "avg_excess": float(grp["excess_return_20d"].mean())
                        if "excess_return_20d" in grp and not grp["excess_return_20d"].isna().all()
                        else None,
                        "win_rate": float((grp["return_20d"] > 0).mean()),
                    })

            return pd.DataFrame(result_rows)
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fetch_prices(
        self, symbols: List[str], as_of: date
    ) -> Dict[str, Optional[float]]:
        """Fetch latest close prices for symbols via yfinance."""
        prices: Dict[str, Optional[float]] = {}
        try:
            import yfinance as yf

            # Fetch 5 days of data to handle weekends/holidays
            start = as_of - timedelta(days=7)
            data = yf.download(
                symbols, start=start.isoformat(),
                end=(as_of + timedelta(days=1)).isoformat(),
                progress=False, threads=True,
            )
            if data.empty:
                return prices

            close = data.get("Close", data)
            if isinstance(close, pd.Series):
                # Single symbol
                last = close.dropna().iloc[-1] if len(close.dropna()) > 0 else None
                if last is not None:
                    prices[symbols[0]] = float(last)
            else:
                for sym in symbols:
                    if sym in close.columns:
                        vals = close[sym].dropna()
                        if len(vals) > 0:
                            prices[sym] = float(vals.iloc[-1])
        except Exception as e:
            logger.warning("Price fetch failed: %s", e)
        return prices

    def _get_spy_price_on_date(self, dt: date) -> Optional[float]:
        """Get SPY close price on a specific date."""
        try:
            import yfinance as yf

            start = dt - timedelta(days=5)
            end = dt + timedelta(days=3)
            data = yf.download(
                "SPY", start=start.isoformat(), end=end.isoformat(),
                progress=False,
            )
            if data.empty:
                return None
            close = data["Close"] if "Close" in data.columns else data
            # Find closest date <= dt
            mask = close.index.date <= dt
            if not mask.any():
                return None
            return float(close[mask].iloc[-1])
        except Exception:
            return None


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None
