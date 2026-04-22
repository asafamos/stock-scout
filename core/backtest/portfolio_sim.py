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

    # Per-trade max holding period (from dynamic Holding_Days)
    max_holding_days: int = 20

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
    atr_pct: float = 0.03  # ATR as fraction of entry price (for trailing stop)


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
        slippage_pct: float = 0.10,  # raised from 0.05 — realistic for mid/small caps
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

            # `prices` can be a float (close) or a tuple/dict of OHLC.
            _raw = prices.get(ticker)
            if isinstance(_raw, dict):
                price_open = float(_raw.get("open") or _raw.get("close") or 0)
                price_close = float(_raw.get("close") or price_open or 0)
            elif isinstance(_raw, (tuple, list)) and len(_raw) >= 4:
                # assume (open, high, low, close)
                price_open = float(_raw[0] or 0)
                price_close = float(_raw[3] or price_open)
            else:
                price_open = float(_raw or row.get("Close", row.get("entry_price")) or 0)
                price_close = price_open
            price = price_open
            if price is None or price <= 0:
                continue

            # Gap-entry guard (mirrors live order_manager ±2% check).
            # When scan-close and next-day open diverge by >2%, skip the
            # entry — the signal has been invalidated by overnight news.
            scan_close = float(row.get("Close", 0) or 0)
            if scan_close > 0:
                gap_pct = (price - scan_close) / scan_close * 100
                if gap_pct > 2.0 or gap_pct < -2.0:
                    continue  # silent skip — matches live behavior

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

            # Stop / target — read dynamic R:R columns from synced pipeline
            atr_pct = float(row.get("ATR_Pct", 0.03))
            stop_price = row.get("Stop_Loss", row.get("Stop"))
            if stop_price is None or (isinstance(stop_price, float) and np.isnan(stop_price)):
                stop_price = entry_price * (1 - self.stop_loss_atr_mult * atr_pct)
            else:
                stop_price = float(stop_price)

            target_price = row.get(
                "Target_Price", row.get("Target_20d", row.get("target_price"))
            )
            if target_price is None or (isinstance(target_price, float) and np.isnan(target_price)):
                target_price = entry_price * (1 + self.target_atr_mult * atr_pct)
            else:
                target_price = float(target_price)

            # Per-trade holding period from dynamic ATR-based calculation
            _hd = row.get("Holding_Days", self.holding_days)
            max_hd = int(_hd) if isinstance(_hd, (int, float)) and not np.isnan(_hd) else self.holding_days

            trade = Trade(
                ticker=ticker,
                entry_date=dt,
                entry_price=entry_price,
                shares=shares,
                position_value=actual_cost,
                stop_price=stop_price,
                target_price=target_price,
                max_holding_days=max_hd,
                atr_pct=atr_pct,
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
            _raw = prices.get(ticker)
            if _raw is None:
                continue
            # Accept scalar close or {open, high, low, close} / (o,h,l,c).
            if isinstance(_raw, dict):
                price = float(_raw.get("close") or 0)
                day_high = float(_raw.get("high") or price)
                day_low = float(_raw.get("low") or price)
            elif isinstance(_raw, (tuple, list)) and len(_raw) >= 4:
                day_high = float(_raw[1] or 0)
                day_low = float(_raw[2] or 0)
                price = float(_raw[3] or 0)
            else:
                price = float(_raw or 0)
                day_high = price
                day_low = price
            if price <= 0:
                continue

            # Track extremes (use intraday H/L when available — closer to reality)
            pos.max_price = max(pos.max_price, day_high)
            pos.min_price = min(pos.min_price, day_low)

            # Break-even / trailing stop: ratchet stop up as position advances
            try:
                from core.scoring_config import TRAILING_STOP_CONFIG
                if (
                    pos.trade.target_price is not None
                    and pos.trade.entry_price > 0
                    and pos.trade.stop_price is not None
                ):
                    _upside = pos.trade.target_price - pos.trade.entry_price
                    _progress = (price - pos.trade.entry_price) / _upside if _upside > 0 else 0.0
                    _be_trig = TRAILING_STOP_CONFIG.get("breakeven_trigger_pct", 0.50)
                    _trail_trig = TRAILING_STOP_CONFIG.get("trail_trigger_pct", 0.75)
                    _trail_mult = TRAILING_STOP_CONFIG.get("trail_atr_mult", 1.5)
                    _atr_abs = pos.trade.entry_price * float(getattr(pos.trade, "atr_pct", 0.03) or 0.03)
                    if _progress >= _trail_trig and _atr_abs > 0:
                        _new_stop = max(price - _trail_mult * _atr_abs, pos.trade.entry_price)
                        if _new_stop > pos.trade.stop_price:
                            pos.trade.stop_price = _new_stop
                    elif _progress >= _be_trig:
                        if pos.trade.entry_price > pos.trade.stop_price:
                            pos.trade.stop_price = pos.trade.entry_price
            except Exception:
                pass

            # Check exits
            cal_days = (dt - pos.trade.entry_date).days
            trading_days = int(cal_days * 5 / 7)

            # Intraday-aware stop/target detection: use day's H/L, not only
            # close. This matches live order fills — a stop hit midday fills
            # regardless of where the stock closes. Previously: stop only
            # triggered if the CLOSE was below stop, understating drawdowns.
            if pos.trade.stop_price and day_low <= pos.trade.stop_price:
                to_close.append((i, "stop"))
            elif pos.trade.target_price and day_high >= pos.trade.target_price:
                to_close.append((i, "target"))
            elif trading_days >= pos.trade.max_holding_days:
                to_close.append((i, "expiry"))
            else:
                # Time-stop: exit stagnant positions after halfway_days trading days
                try:
                    from core.scoring_config import TIME_STOP_CONFIG
                    _halfway = int(TIME_STOP_CONFIG.get("halfway_days", 10))
                    _min_prog = float(TIME_STOP_CONFIG.get("min_progress_pct", 0.30))
                    _buffer = int(TIME_STOP_CONFIG.get("buffer_days", 2))
                    if (
                        trading_days >= _halfway
                        and trading_days < pos.trade.max_holding_days - _buffer
                        and pos.trade.target_price is not None
                        and pos.trade.entry_price > 0
                    ):
                        _upside = pos.trade.target_price - pos.trade.entry_price
                        if _upside > 0:
                            _progress = (price - pos.trade.entry_price) / _upside
                            if _progress < _min_prog:
                                to_close.append((i, "time_stop"))
                except Exception:
                    pass

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
