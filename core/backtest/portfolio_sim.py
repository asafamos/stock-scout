"""Portfolio simulator — manages positions, stops, targets, and equity curve.

Simulates realistic portfolio management: position sizing based on
volatility (ATR), stop-loss enforcement, target exits, and time-based
expiration after the holding period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("stock_scout.backtest.portfolio")


@dataclass
class Trade:
    """One completed round-trip trade."""

    ticker: str
    entry_date: date
    entry_price: float
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    return_pct: float = 0.0
    holding_days: int = 0
    exit_reason: str = ""  # "expiry", "stop", "target"

    # Scores at entry (for attribution)
    final_score: float = 0.0
    tech_score: float = 0.0
    fundamental_score: float = 0.0
    ml_prob: float = 0.5
    market_regime: str = ""
    sector: str = ""

    # Position sizing
    shares: int = 0
    position_value: float = 0.0

    # Risk management
    stop_price: Optional[float] = None
    target_price: Optional[float] = None


@dataclass
class OpenPosition:
    """An open position being tracked."""

    trade: Trade
    max_price: float = 0.0
    min_price: float = float("inf")

    def __post_init__(self):
        self.max_price = self.trade.entry_price
        self.min_price = self.trade.entry_price


class PortfolioSimulator:
    """Simulates portfolio management with position sizing and rebalancing.

    Features:
      - ATR-based position sizing (equal risk per position)
      - Stop-loss and target-price exits
      - Time-based expiration after holding_days
      - Daily equity tracking
      - Slippage and commission modelling
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        max_positions: int = 15,
        max_position_pct: float = 10.0,
        holding_days: int = 20,
        stop_loss_atr_mult: float = 2.0,
        target_atr_mult: float = 4.0,
        slippage_pct: float = 0.05,
        commission_per_trade: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct / 100.0
        self.holding_days = holding_days
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.target_atr_mult = target_atr_mult
        self.slippage_pct = slippage_pct / 100.0
        self.commission = commission_per_trade

        self.cash = initial_capital
        self.positions: List[OpenPosition] = []
        self.closed_trades: List[Trade] = []
        self.equity_history: List[Dict[str, Any]] = []
        self.peak_equity = initial_capital

    # ------------------------------------------------------------------
    # Open positions
    # ------------------------------------------------------------------

    def open_positions(
        self,
        dt: date,
        selections: pd.DataFrame,
        prices: Dict[str, float],
    ) -> List[Trade]:
        """Open new positions from scored selections.

        Args:
            dt: Entry date.
            selections: DataFrame with columns [Ticker, FinalScore_20d,
                TechScore_20d, Fundamental_Score, ML_20d_Prob, ATR_Pct,
                Market_Regime, sector, Close, Target_20d, Stop].
            prices: {ticker: next-day open price} for realistic entry.

        Returns:
            List of newly opened Trade objects.
        """
        available_slots = self.max_positions - len(self.positions)
        if available_slots <= 0:
            return []

        # Already-held tickers
        held_tickers = {p.trade.ticker for p in self.positions}

        new_trades = []
        for _, row in selections.head(available_slots).iterrows():
            ticker = str(row.get("Ticker", row.get("ticker", "")))
            if not ticker or ticker in held_tickers:
                continue

            price = prices.get(ticker) or row.get("Close", row.get("entry_price"))
            if price is None or price <= 0:
                continue

            # Apply slippage (buy slightly higher)
            entry_price = price * (1 + self.slippage_pct)

            # Position sizing: equal weight, capped
            max_val = self.cash * self.max_position_pct
            position_val = min(
                max_val,
                self.cash / max(1, available_slots - len(new_trades)),
            )
            if position_val < 100:  # minimum $100 position
                continue

            shares = int(position_val / entry_price)
            if shares <= 0:
                continue

            actual_cost = shares * entry_price + self.commission

            # Stop / target
            atr_pct = float(row.get("ATR_Pct", 0.03))
            stop_price = row.get("Stop")
            if stop_price is None or (isinstance(stop_price, float) and np.isnan(stop_price)):
                stop_price = entry_price * (1 - self.stop_loss_atr_mult * atr_pct)
            else:
                stop_price = float(stop_price)

            target_price = row.get("Target_20d", row.get("target_price"))
            if target_price is None or (isinstance(target_price, float) and np.isnan(target_price)):
                target_price = entry_price * (1 + self.target_atr_mult * atr_pct)
            else:
                target_price = float(target_price)

            trade = Trade(
                ticker=ticker,
                entry_date=dt,
                entry_price=entry_price,
                shares=shares,
                position_value=actual_cost,
                stop_price=stop_price,
                target_price=target_price,
                final_score=float(row.get("FinalScore_20d", 0)),
                tech_score=float(row.get("TechScore_20d", 0)),
                fundamental_score=float(row.get("Fundamental_Score", 0)),
                ml_prob=float(row.get("ML_20d_Prob", 0.5)),
                market_regime=str(row.get("Market_Regime", "")),
                sector=str(row.get("sector", row.get("Sector", ""))),
            )

            self.cash -= actual_cost
            pos = OpenPosition(trade=trade)
            self.positions.append(pos)
            held_tickers.add(ticker)
            new_trades.append(trade)

        return new_trades

    # ------------------------------------------------------------------
    # Daily update
    # ------------------------------------------------------------------

    def update(
        self,
        dt: date,
        prices: Dict[str, float],
    ) -> float:
        """Update all positions with current prices.

        Checks stop-loss, target, and time expiration.

        Returns:
            Current total equity.
        """
        to_close: List[Tuple[int, str]] = []

        for i, pos in enumerate(self.positions):
            ticker = pos.trade.ticker
            price = prices.get(ticker)
            if price is None:
                continue

            # Track extremes
            pos.max_price = max(pos.max_price, price)
            pos.min_price = min(pos.min_price, price)

            # Check exits
            cal_days = (dt - pos.trade.entry_date).days
            trading_days = int(cal_days * 5 / 7)

            if pos.trade.stop_price and price <= pos.trade.stop_price:
                to_close.append((i, "stop"))
            elif pos.trade.target_price and price >= pos.trade.target_price:
                to_close.append((i, "target"))
            elif trading_days >= self.holding_days:
                to_close.append((i, "expiry"))

        # Close in reverse order to preserve indices
        for idx, reason in sorted(to_close, reverse=True):
            self._close_position(idx, dt, prices, reason)

        # Record equity
        equity = self._compute_equity(prices)
        self.peak_equity = max(self.peak_equity, equity)
        dd = (equity / self.peak_equity - 1) * 100 if self.peak_equity > 0 else 0.0

        self.equity_history.append({
            "date": dt,
            "equity": equity,
            "cash": self.cash,
            "n_positions": len(self.positions),
            "drawdown": dd,
        })

        return equity

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    def _close_position(
        self,
        idx: int,
        dt: date,
        prices: Dict[str, float],
        reason: str,
    ) -> None:
        """Close a position at current price."""
        pos = self.positions.pop(idx)
        trade = pos.trade
        price = prices.get(trade.ticker, trade.entry_price)

        # Apply slippage (sell slightly lower)
        exit_price = price * (1 - self.slippage_pct)
        proceeds = trade.shares * exit_price - self.commission

        trade.exit_date = dt
        trade.exit_price = exit_price
        trade.return_pct = (exit_price / trade.entry_price - 1) * 100
        trade.holding_days = (dt - trade.entry_date).days
        trade.exit_reason = reason

        self.cash += proceeds
        self.closed_trades.append(trade)

    # ------------------------------------------------------------------
    # Equity
    # ------------------------------------------------------------------

    def _compute_equity(self, prices: Dict[str, float]) -> float:
        """Current total equity = cash + market value of all positions."""
        position_value = 0.0
        for pos in self.positions:
            price = prices.get(pos.trade.ticker, pos.trade.entry_price)
            position_value += pos.trade.shares * price
        return self.cash + position_value

    # ------------------------------------------------------------------
    # Force close all
    # ------------------------------------------------------------------

    def close_all(self, dt: date, prices: Dict[str, float]) -> None:
        """Close all remaining open positions (end of backtest)."""
        while self.positions:
            self._close_position(0, dt, prices, "backtest_end")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_equity_curve(self) -> pd.DataFrame:
        """Daily equity values."""
        if not self.equity_history:
            return pd.DataFrame(columns=["date", "equity", "cash", "n_positions", "drawdown"])
        df = pd.DataFrame(self.equity_history)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def get_trade_log(self) -> pd.DataFrame:
        """All completed trades with P&L."""
        if not self.closed_trades:
            return pd.DataFrame()
        rows = []
        for t in self.closed_trades:
            rows.append({
                "ticker": t.ticker,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "return_pct": t.return_pct,
                "holding_days": t.holding_days,
                "exit_reason": t.exit_reason,
                "shares": t.shares,
                "position_value": t.position_value,
                "pnl": t.shares * ((t.exit_price or t.entry_price) - t.entry_price),
                "final_score": t.final_score,
                "tech_score": t.tech_score,
                "fundamental_score": t.fundamental_score,
                "ml_prob": t.ml_prob,
                "market_regime": t.market_regime,
                "sector": t.sector,
            })
        return pd.DataFrame(rows)
