"""Exit signal evaluator for live recommendations.

Checks active recommendations against current market prices to detect
stop-loss, target-hit, and holding-period-expiry conditions.

Exit logic mirrors core/backtest/portfolio_sim.py (lines 224-233) to
ensure consistency between backtest and live signals.

Usage::

    from core.exit_signals import evaluate_exit_signals

    signals = evaluate_exit_signals(
        recommendations=latest_scan_df,
        current_prices={"AAPL": 195.50, "TSLA": 310.20},
    )
    for sig in signals:
        print(f"{sig.ticker}: {sig.action} — {sig.reason}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """A signal indicating a recommendation should be exited."""

    ticker: str
    action: str  # "SELL_STOP", "SELL_TARGET", "SELL_EXPIRY"
    reason: str  # Human-readable reason
    entry_price: float
    current_price: float
    return_pct: float  # (current - entry) / entry
    days_held: int  # Calendar days since entry
    target_price: Optional[float] = None
    stop_price: Optional[float] = None


def evaluate_exit_signals(
    recommendations: pd.DataFrame,
    current_prices: Dict[str, float],
    as_of_date: Optional[datetime] = None,
) -> List[ExitSignal]:
    """Check active recommendations for exit conditions.

    Mirrors the exit logic in core/backtest/portfolio_sim.py (lines 224-233):

    1. **Stop-loss**: ``current_price <= Stop_Loss``
    2. **Target hit**: ``current_price >= Target_Price``
    3. **Holding expiry**: ``as_of_date >= Target_Date``
       or ``days_held >= Holding_Days``

    Priority: stop > target > expiry (same as backtest).

    Args:
        recommendations: DataFrame with columns Ticker, Entry_Price,
            Target_Price, Stop_Loss, Holding_Days (opt), Target_Date (opt),
            As_Of_Date (entry date, opt).
        current_prices: Mapping of ticker -> current market price.
        as_of_date: Current date for expiry check (defaults to now).

    Returns:
        List of :class:`ExitSignal` for recommendations that should exit.
    """
    if as_of_date is None:
        as_of_date = datetime.now()
    elif isinstance(as_of_date, date) and not isinstance(as_of_date, datetime):
        as_of_date = datetime.combine(as_of_date, datetime.min.time())

    signals: List[ExitSignal] = []

    required = {"Ticker", "Entry_Price"}
    if not required.issubset(set(recommendations.columns)):
        logger.warning(
            "Missing required columns: %s", required - set(recommendations.columns)
        )
        return signals

    for _, rec in recommendations.iterrows():
        ticker = rec.get("Ticker")
        if ticker is None:
            continue

        price = current_prices.get(ticker)
        if price is None or not np.isfinite(price):
            continue

        entry = _safe_float(rec.get("Entry_Price"))
        if entry is None or entry <= 0:
            continue

        target = _safe_float(rec.get("Target_Price"))
        stop = _safe_float(rec.get("Stop_Loss"))
        return_pct = (price - entry) / entry

        # Calendar days held
        entry_date = rec.get("As_Of_Date")
        days_held = 0
        if entry_date is not None:
            try:
                ed = pd.Timestamp(entry_date)
                if pd.notna(ed):
                    days_held = (as_of_date - ed.to_pydatetime()).days
            except Exception:
                pass

        # Check exits (same priority as portfolio_sim.py: stop > target > expiry)
        if stop is not None and price <= stop:
            signals.append(
                ExitSignal(
                    ticker=ticker,
                    action="SELL_STOP",
                    reason=f"Stop loss hit ({price:.2f} <= {stop:.2f})",
                    entry_price=entry,
                    current_price=price,
                    return_pct=return_pct,
                    days_held=days_held,
                    target_price=target,
                    stop_price=stop,
                )
            )
        elif target is not None and price >= target:
            signals.append(
                ExitSignal(
                    ticker=ticker,
                    action="SELL_TARGET",
                    reason=f"Target hit ({price:.2f} >= {target:.2f})",
                    entry_price=entry,
                    current_price=price,
                    return_pct=return_pct,
                    days_held=days_held,
                    target_price=target,
                    stop_price=stop,
                )
            )
        else:
            # Expiry: prefer Target_Date, fall back to Holding_Days
            expired = False
            target_date = rec.get("Target_Date")
            holding_days = rec.get("Holding_Days")

            if target_date is not None:
                try:
                    td = pd.Timestamp(target_date)
                    if pd.notna(td) and as_of_date >= td.to_pydatetime():
                        expired = True
                except Exception:
                    pass

            if not expired and holding_days is not None and entry_date is not None:
                try:
                    if days_held >= int(holding_days):
                        expired = True
                except (TypeError, ValueError):
                    pass

            if expired:
                signals.append(
                    ExitSignal(
                        ticker=ticker,
                        action="SELL_EXPIRY",
                        reason=f"Holding period expired ({days_held} days)",
                        entry_price=entry,
                        current_price=price,
                        return_pct=return_pct,
                        days_held=days_held,
                        target_price=target,
                        stop_price=stop,
                    )
                )

    return signals


def _safe_float(val) -> Optional[float]:
    """Convert to float if finite, else None."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
