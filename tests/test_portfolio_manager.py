"""Unit tests for the PortfolioManager virtual portfolio feature."""
from __future__ import annotations

import uuid
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from core.db.schema import ALL_TABLES
from core.db.portfolio_manager import PortfolioManager


# ---------------------------------------------------------------------------
# Fixture: in-memory DuckDB store with portfolio tables
# ---------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path):
    """Create a minimal ScanStore-like object backed by an in-memory DuckDB."""
    import duckdb

    db_path = str(tmp_path / "test_portfolio.duckdb")

    class _MockStore:
        def __init__(self):
            self._db_path = db_path
            # Initialize tables
            con = self._connect()
            for ddl in ALL_TABLES:
                con.execute(ddl)
            con.close()

        def _connect(self):
            return duckdb.connect(self._db_path)

    return _MockStore()


@pytest.fixture
def pm(store):
    """Create a PortfolioManager instance backed by the test store."""
    return PortfolioManager(store, user_id="test_user")


# ---------------------------------------------------------------------------
# Tests: add_position
# ---------------------------------------------------------------------------
class TestAddPosition:
    def test_add_basic_position(self, pm):
        pid = pm.add_position(
            ticker="AAPL",
            entry_price=185.0,
            target_price=200.0,
            stop_price=175.0,
            holding_days=20,
        )
        assert isinstance(pid, str)
        assert len(pid) == 32  # UUID hex

    def test_added_position_is_open(self, pm):
        pm.add_position(ticker="MSFT", entry_price=350.0)
        assert pm.is_in_portfolio("MSFT")

    def test_duplicate_ticker_raises(self, pm):
        pm.add_position(ticker="GOOG", entry_price=140.0)
        with pytest.raises(ValueError, match="already in portfolio"):
            pm.add_position(ticker="GOOG", entry_price=145.0)

    def test_add_position_with_all_fields(self, pm):
        pid = pm.add_position(
            ticker="TSLA",
            entry_price=250.0,
            target_price=280.0,
            stop_price=235.0,
            holding_days=15,
            shares=50,
            scan_id="scan_123",
            recommendation_id="rec_456",
            final_score=72.5,
            risk_class="core",
            sector="Technology",
        )
        df = pm.get_open_positions()
        row = df[df["ticker"] == "TSLA"].iloc[0]
        assert float(row["entry_price"]) == 250.0
        assert float(row["target_price"]) == 280.0
        assert float(row["stop_price"]) == 235.0
        assert int(row["shares"]) == 50
        assert row["scan_id"] == "scan_123"
        assert float(row["final_score"]) == 72.5
        assert row["risk_class"] == "core"


# ---------------------------------------------------------------------------
# Tests: remove_position (manual close)
# ---------------------------------------------------------------------------
class TestRemovePosition:
    def test_remove_closes_position(self, pm):
        pid = pm.add_position(ticker="AAPL", entry_price=185.0)
        result = pm.remove_position(pid, exit_price=190.0, exit_reason="manual")
        assert result is True
        assert not pm.is_in_portfolio("AAPL")

    def test_remove_nonexistent_returns_false(self, pm):
        result = pm.remove_position("nonexistent_id")
        assert result is False

    def test_remove_calculates_return(self, pm):
        pid = pm.add_position(ticker="NVDA", entry_price=100.0)
        pm.remove_position(pid, exit_price=110.0)

        closed = pm.get_closed_positions(days=30)
        row = closed[closed["ticker"] == "NVDA"].iloc[0]
        assert float(row["realized_return_pct"]) == pytest.approx(10.0, abs=0.1)
        assert row["prediction_correct"] == True

    def test_remove_negative_return(self, pm):
        pid = pm.add_position(ticker="META", entry_price=100.0)
        pm.remove_position(pid, exit_price=90.0)

        closed = pm.get_closed_positions(days=30)
        row = closed[closed["ticker"] == "META"].iloc[0]
        assert float(row["realized_return_pct"]) == pytest.approx(-10.0, abs=0.1)
        assert row["prediction_correct"] == False


# ---------------------------------------------------------------------------
# Tests: get_open_positions / get_closed_positions
# ---------------------------------------------------------------------------
class TestQueries:
    def test_get_open_returns_dataframe(self, pm):
        df = pm.get_open_positions()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_open_positions_exclude_closed(self, pm):
        pid = pm.add_position(ticker="AMD", entry_price=100.0)
        pm.add_position(ticker="INTC", entry_price=50.0)
        pm.remove_position(pid, exit_price=110.0)

        open_df = pm.get_open_positions()
        assert len(open_df) == 1
        assert open_df.iloc[0]["ticker"] == "INTC"

    def test_closed_positions_include_closed(self, pm):
        pid = pm.add_position(ticker="AMZN", entry_price=180.0)
        pm.remove_position(pid, exit_price=185.0)

        closed_df = pm.get_closed_positions(days=30)
        assert len(closed_df) == 1
        assert closed_df.iloc[0]["ticker"] == "AMZN"


# ---------------------------------------------------------------------------
# Tests: is_in_portfolio / get_portfolio_tickers
# ---------------------------------------------------------------------------
class TestPortfolioChecks:
    def test_is_in_portfolio_false_when_empty(self, pm):
        assert not pm.is_in_portfolio("AAPL")

    def test_get_portfolio_tickers_returns_set(self, pm):
        pm.add_position(ticker="AAPL", entry_price=185.0)
        pm.add_position(ticker="GOOG", entry_price=140.0)

        tickers = pm.get_portfolio_tickers()
        assert isinstance(tickers, set)
        assert tickers == {"AAPL", "GOOG"}

    def test_closed_ticker_not_in_portfolio(self, pm):
        pid = pm.add_position(ticker="NFLX", entry_price=500.0)
        pm.remove_position(pid, exit_price=520.0)

        assert not pm.is_in_portfolio("NFLX")
        assert "NFLX" not in pm.get_portfolio_tickers()


# ---------------------------------------------------------------------------
# Tests: get_portfolio_stats
# ---------------------------------------------------------------------------
class TestPortfolioStats:
    def test_empty_stats(self, pm):
        stats = pm.get_portfolio_stats()
        assert stats["open_count"] == 0
        assert stats["closed_count"] == 0
        assert stats["total_return_pct"] == 0.0

    def test_stats_with_open_positions(self, pm):
        pm.add_position(ticker="AAPL", entry_price=100.0, shares=100)
        pm.add_position(ticker="GOOG", entry_price=200.0, shares=50)

        stats = pm.get_portfolio_stats()
        assert stats["open_count"] == 2
        # Total invested = 100*100 + 200*50 = 20,000
        assert stats["total_invested"] == 20000.0

    def test_stats_with_closed_positions(self, pm):
        pid1 = pm.add_position(ticker="WIN1", entry_price=100.0)
        pid2 = pm.add_position(ticker="LOSE1", entry_price=100.0)

        pm.remove_position(pid1, exit_price=110.0, exit_reason="target")  # +10%
        pm.remove_position(pid2, exit_price=95.0, exit_reason="stop")     # -5%

        stats = pm.get_portfolio_stats()
        assert stats["closed_count"] == 2
        assert stats["win_rate"] == 0.5
        # Average return: (10 + -5) / 2 = 2.5
        assert stats["avg_return"] == pytest.approx(2.5, abs=0.1)

    def test_manual_closes_excluded_from_stats(self, pm):
        pid1 = pm.add_position(ticker="SYS1", entry_price=100.0)
        pid2 = pm.add_position(ticker="MAN1", entry_price=100.0)

        pm.remove_position(pid1, exit_price=110.0, exit_reason="target")  # system close
        pm.remove_position(pid2, exit_price=90.0, exit_reason="manual")   # manual close

        stats = pm.get_portfolio_stats()
        assert stats["closed_count"] == 1  # only system close counted
        assert stats["win_rate"] == 1.0    # the one system close was a win


# ---------------------------------------------------------------------------
# Tests: update_prices
# ---------------------------------------------------------------------------
class TestUpdatePrices:
    def test_update_no_positions_returns_zeros(self, pm):
        result = pm.update_prices()
        assert result == {"updated": 0, "auto_closed": 0, "open_count": 0}

    @patch.object(PortfolioManager, "_fetch_prices")
    def test_update_with_price_change(self, mock_fetch, pm):
        pm.add_position(ticker="AAPL", entry_price=100.0)
        mock_fetch.return_value = {"AAPL": 105.0}

        result = pm.update_prices()
        assert result["updated"] == 1

        # Check current price updated
        df = pm.get_open_positions()
        assert float(df.iloc[0]["current_price"]) == 105.0
        assert float(df.iloc[0]["current_return_pct"]) == pytest.approx(5.0, abs=0.1)

    @patch.object(PortfolioManager, "_fetch_prices")
    def test_auto_close_on_stop(self, mock_fetch, pm):
        pm.add_position(ticker="STOP", entry_price=100.0, stop_price=90.0)
        mock_fetch.return_value = {"STOP": 85.0}

        result = pm.update_prices()
        assert result["auto_closed"] == 1
        assert not pm.is_in_portfolio("STOP")

        closed = pm.get_closed_positions(days=30)
        assert closed.iloc[0]["exit_reason"] == "stop"

    @patch.object(PortfolioManager, "_fetch_prices")
    def test_auto_close_on_target(self, mock_fetch, pm):
        pm.add_position(ticker="TARGET", entry_price=100.0, target_price=120.0)
        mock_fetch.return_value = {"TARGET": 125.0}

        result = pm.update_prices()
        assert result["auto_closed"] == 1

        closed = pm.get_closed_positions(days=30)
        assert closed.iloc[0]["exit_reason"] == "target"

    @patch.object(PortfolioManager, "_fetch_prices")
    def test_auto_close_on_expiry(self, mock_fetch, pm):
        # Create position that's been held past holding_days
        pid = pm.add_position(ticker="EXPIRE", entry_price=100.0, holding_days=5)

        # Manually set entry_date to 10 days ago so expiry triggers
        con = pm._store._connect()
        try:
            old_date = date.today() - timedelta(days=10)
            con.execute(
                "UPDATE portfolio_positions SET entry_date = ? WHERE position_id = ?",
                [old_date, pid],
            )
        finally:
            con.close()

        mock_fetch.return_value = {"EXPIRE": 102.0}
        result = pm.update_prices()
        assert result["auto_closed"] == 1

        closed = pm.get_closed_positions(days=30)
        assert closed.iloc[0]["exit_reason"] == "expiry"

    @patch.object(PortfolioManager, "_fetch_prices")
    def test_max_min_tracking(self, mock_fetch, pm):
        pm.add_position(ticker="TRACK", entry_price=100.0)

        # First update: price goes up
        mock_fetch.return_value = {"TRACK": 110.0}
        pm.update_prices()

        # Second update: price goes down
        mock_fetch.return_value = {"TRACK": 95.0}
        pm.update_prices()

        df = pm.get_open_positions()
        row = df[df["ticker"] == "TRACK"].iloc[0]
        assert float(row["max_price"]) == 110.0
        assert float(row["min_price"]) == 95.0


# ---------------------------------------------------------------------------
# Tests: portfolio_card rendering
# ---------------------------------------------------------------------------
class TestPortfolioCards:
    def test_render_open_card(self):
        from ui.components.portfolio_card import render_open_position_card
        html = render_open_position_card({
            "ticker": "AAPL",
            "entry_price": 185.0,
            "current_price": 190.0,
            "target_price": 200.0,
            "stop_price": 175.0,
            "current_return_pct": 2.7,
            "entry_date": date.today(),
            "holding_days": 20,
            "risk_class": "core",
            "sector": "Technology",
        })
        assert "AAPL" in html
        assert "$185.00" in html
        assert "$190.00" in html
        assert "+2.7%" in html

    def test_render_closed_card(self):
        from ui.components.portfolio_card import render_closed_position_card
        html = render_closed_position_card({
            "ticker": "MSFT",
            "entry_price": 350.0,
            "exit_price": 360.0,
            "realized_return_pct": 2.86,
            "exit_reason": "target",
            "prediction_correct": True,
            "entry_date": date.today() - timedelta(days=15),
            "exit_date": date.today(),
        })
        assert "MSFT" in html
        assert "target" in html
        assert "+2.9%" in html
        assert "✅" in html

    def test_render_sidebar_summary(self):
        from ui.components.portfolio_card import render_portfolio_sidebar_summary
        html = render_portfolio_sidebar_summary({
            "open_count": 3,
            "closed_count": 5,
            "total_return_pct": 4.2,
            "win_rate": 0.6,
            "avg_return": 2.1,
        })
        assert "3" in html
        assert "5" in html
        assert "60%" in html
        assert "+4.2%" in html
