"""Tests for core.backtest.portfolio_sim — portfolio simulation logic."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.backtest.portfolio_sim import PortfolioSimulator


@pytest.fixture
def sim():
    return PortfolioSimulator(
        initial_capital=100_000,
        max_positions=5,
        holding_days=20,
        slippage_pct=0.0,
        commission_per_trade=0.0,
    )


@pytest.fixture
def selections():
    """Sample scored universe."""
    return pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "GOOG"],
        "FinalScore_20d": [85.0, 78.0, 72.0],
        "TechScore_20d": [80.0, 75.0, 70.0],
        "Fundamental_Score": [70.0, 80.0, 65.0],
        "ML_20d_Prob": [0.65, 0.58, 0.52],
        "ATR_Pct": [0.03, 0.025, 0.035],
        "Market_Regime": ["TREND_UP", "TREND_UP", "TREND_UP"],
        "sector": ["Technology", "Technology", "Technology"],
        "Close": [185.0, 420.0, 175.0],
        "Target_20d": [200.0, 450.0, 195.0],
        "Stop": [175.0, 400.0, 165.0],
    })


class TestPortfolioSimulator:
    def test_open_positions(self, sim, selections):
        prices = {"AAPL": 185.0, "MSFT": 420.0, "GOOG": 175.0}
        trades = sim.open_positions(date(2024, 6, 3), selections, prices)
        assert len(trades) == 3
        assert sim.cash < 100_000
        assert len(sim.positions) == 3

    def test_max_positions(self, sim, selections):
        """Shouldn't exceed max_positions."""
        # Open 5 positions in two batches
        prices1 = {"AAPL": 185.0, "MSFT": 420.0, "GOOG": 175.0}
        sim.open_positions(date(2024, 6, 3), selections, prices1)

        extra = pd.DataFrame({
            "Ticker": ["NVDA", "META", "TSLA"],
            "FinalScore_20d": [90.0, 85.0, 75.0],
            "Close": [800.0, 500.0, 250.0],
            "ATR_Pct": [0.04, 0.03, 0.05],
        })
        prices2 = {"NVDA": 800.0, "META": 500.0, "TSLA": 250.0}
        trades2 = sim.open_positions(date(2024, 6, 3), extra, prices2)
        assert len(sim.positions) <= 5

    def test_no_duplicate_tickers(self, sim, selections):
        """Same ticker shouldn't be opened twice."""
        prices = {"AAPL": 185.0, "MSFT": 420.0, "GOOG": 175.0}
        sim.open_positions(date(2024, 6, 3), selections, prices)
        trades2 = sim.open_positions(date(2024, 6, 4), selections, prices)
        assert len(trades2) == 0  # all already held

    def test_update_daily(self, sim, selections):
        prices = {"AAPL": 185.0, "MSFT": 420.0, "GOOG": 175.0}
        sim.open_positions(date(2024, 6, 3), selections, prices)

        # Prices go up
        new_prices = {"AAPL": 190.0, "MSFT": 430.0, "GOOG": 180.0}
        equity = sim.update(date(2024, 6, 4), new_prices)
        assert equity > 100_000  # should be profitable

    def test_stop_loss_exit(self, sim, selections):
        prices = {"AAPL": 185.0, "MSFT": 420.0, "GOOG": 175.0}
        sim.open_positions(date(2024, 6, 3), selections, prices)

        # AAPL crashes below stop (175.0)
        crash_prices = {"AAPL": 170.0, "MSFT": 425.0, "GOOG": 176.0}
        sim.update(date(2024, 6, 4), crash_prices)

        # AAPL should be closed
        held_tickers = {p.trade.ticker for p in sim.positions}
        assert "AAPL" not in held_tickers
        assert len(sim.closed_trades) >= 1
        aapl_trade = [t for t in sim.closed_trades if t.ticker == "AAPL"][0]
        assert aapl_trade.exit_reason == "stop"
        assert aapl_trade.return_pct < 0

    def test_target_exit(self, sim, selections):
        prices = {"AAPL": 185.0, "MSFT": 420.0, "GOOG": 175.0}
        sim.open_positions(date(2024, 6, 3), selections, prices)

        # AAPL hits target (200.0)
        target_prices = {"AAPL": 205.0, "MSFT": 425.0, "GOOG": 178.0}
        sim.update(date(2024, 6, 4), target_prices)

        held_tickers = {p.trade.ticker for p in sim.positions}
        assert "AAPL" not in held_tickers
        aapl_trade = [t for t in sim.closed_trades if t.ticker == "AAPL"][0]
        assert aapl_trade.exit_reason == "target"
        assert aapl_trade.return_pct > 0

    def test_expiry_exit(self, sim, selections):
        """Positions close after holding_days."""
        prices = {"AAPL": 185.0, "MSFT": 420.0, "GOOG": 175.0}
        sim.open_positions(date(2024, 6, 3), selections, prices)

        # Jump 30 calendar days (~21 trading days > 20 holding_days)
        future_prices = {"AAPL": 190.0, "MSFT": 425.0, "GOOG": 178.0}
        sim.update(date(2024, 7, 5), future_prices)

        # All should be closed via expiry
        assert len(sim.positions) == 0
        assert len(sim.closed_trades) == 3
        assert all(t.exit_reason == "expiry" for t in sim.closed_trades)

    def test_close_all(self, sim, selections):
        prices = {"AAPL": 185.0, "MSFT": 420.0, "GOOG": 175.0}
        sim.open_positions(date(2024, 6, 3), selections, prices)
        sim.close_all(date(2024, 6, 30), prices)
        assert len(sim.positions) == 0
        assert len(sim.closed_trades) == 3

    def test_equity_curve(self, sim, selections):
        prices = {"AAPL": 185.0, "MSFT": 420.0, "GOOG": 175.0}
        sim.open_positions(date(2024, 6, 3), selections, prices)
        sim.update(date(2024, 6, 3), prices)
        sim.update(date(2024, 6, 4), {"AAPL": 190.0, "MSFT": 430.0, "GOOG": 180.0})

        eq = sim.get_equity_curve()
        assert len(eq) == 2
        assert "equity" in eq.columns

    def test_trade_log(self, sim, selections):
        prices = {"AAPL": 185.0, "MSFT": 420.0, "GOOG": 175.0}
        sim.open_positions(date(2024, 6, 3), selections, prices)
        sim.close_all(date(2024, 6, 30), prices)

        tl = sim.get_trade_log()
        assert len(tl) == 3
        assert "return_pct" in tl.columns
        assert "pnl" in tl.columns

    def test_slippage(self):
        """Slippage should reduce returns."""
        sim_no_slip = PortfolioSimulator(
            initial_capital=100_000, slippage_pct=0.0, commission_per_trade=0.0,
        )
        sim_slip = PortfolioSimulator(
            initial_capital=100_000, slippage_pct=0.5, commission_per_trade=0.0,
        )
        sel = pd.DataFrame({
            "Ticker": ["AAPL"],
            "FinalScore_20d": [80.0],
            "Close": [100.0],
            "ATR_Pct": [0.03],
        })
        prices = {"AAPL": 100.0}
        sim_no_slip.open_positions(date(2024, 6, 3), sel, prices)
        sim_slip.open_positions(date(2024, 6, 3), sel, prices)

        exit_prices = {"AAPL": 110.0}
        sim_no_slip.close_all(date(2024, 6, 30), exit_prices)
        sim_slip.close_all(date(2024, 6, 30), exit_prices)

        # With slippage, return should be lower
        ret_no = sim_no_slip.closed_trades[0].return_pct
        ret_slip = sim_slip.closed_trades[0].return_pct
        assert ret_slip < ret_no
