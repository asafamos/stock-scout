"""Unit tests for the SupabasePortfolioManager.

Uses a mock Supabase client so no real network calls are made.
Validates API compatibility with the DuckDB PortfolioManager.
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from core.db.portfolio_manager import SupabasePortfolioManager


# ---------------------------------------------------------------------------
# Helpers: mock Supabase client and table chain
# ---------------------------------------------------------------------------
class MockSupabaseResponse:
    """Simulates a Supabase REST response."""
    def __init__(self, data=None):
        self.data = data or []


class MockTableChain:
    """Builder-pattern mock that mimics Supabase table().select().eq()...execute()."""

    def __init__(self, rows=None):
        self._rows = rows if rows is not None else []
        self._inserted = []
        self._updates = []

    def select(self, *args, **kwargs):
        return self

    def insert(self, data, **kwargs):
        self._inserted.append(data)
        return self

    def update(self, data, **kwargs):
        self._updates.append(data)
        return self

    def eq(self, col, val):
        # Filter rows in-memory for test accuracy
        self._rows = [
            r for r in self._rows
            if r.get(col) == val
        ]
        return self

    def neq(self, col, val):
        # Added when get_portfolio_stats started filtering out manual closes
        # via `.neq("exit_reason", "manual")` in the Supabase query chain.
        self._rows = [
            r for r in self._rows
            if r.get(col) != val
        ]
        return self

    def gte(self, col, val):
        self._rows = [
            r for r in self._rows
            if (r.get(col) or "") >= val
        ]
        return self

    def order(self, col, desc=False):
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def execute(self):
        return MockSupabaseResponse(self._rows)


def _make_mock_client(rows=None):
    """Create a mock Supabase client with a .table() that returns MockTableChain."""
    client = MagicMock()
    _rows = rows if rows is not None else []

    def _table(name):
        # Return fresh chain each call, sharing the row store
        return MockTableChain(list(_rows))

    client.table = _table
    return client


def _make_position_row(
    ticker="AAPL",
    entry_price=100.0,
    status="open",
    position_id=None,
    user_id="test_user",
    **overrides,
):
    """Helper to build a position dict as Supabase would return."""
    row = {
        "position_id": position_id or uuid.uuid4().hex,
        "user_id": user_id,
        "ticker": ticker,
        "entry_price": entry_price,
        "target_price": overrides.get("target_price"),
        "stop_price": overrides.get("stop_price"),
        "shares": overrides.get("shares", 100),
        "entry_date": overrides.get("entry_date", date.today().isoformat()),
        "target_date": None,
        "holding_days": overrides.get("holding_days", 20),
        "scan_id": None,
        "recommendation_id": None,
        "final_score": overrides.get("final_score"),
        "risk_class": overrides.get("risk_class"),
        "sector": overrides.get("sector"),
        "current_price": overrides.get("current_price", entry_price),
        "current_return_pct": overrides.get("current_return_pct", 0.0),
        "max_price": overrides.get("max_price", entry_price),
        "min_price": overrides.get("min_price", entry_price),
        "exit_price": overrides.get("exit_price"),
        "exit_date": overrides.get("exit_date"),
        "exit_reason": overrides.get("exit_reason"),
        "realized_return_pct": overrides.get("realized_return_pct"),
        "prediction_correct": overrides.get("prediction_correct"),
        "status": status,
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    return row


# ---------------------------------------------------------------------------
# Tests: SupabasePortfolioManager
# ---------------------------------------------------------------------------
class TestSupabaseInit:
    def test_creates_with_user_id(self):
        client = MagicMock()
        pm = SupabasePortfolioManager(client, user_id="u123")
        assert pm._user_id == "u123"

    def test_default_user_id(self):
        client = MagicMock()
        pm = SupabasePortfolioManager(client)
        assert pm._user_id == "default"


class TestSupabaseAddPosition:
    def test_add_position_calls_insert(self):
        """Verify add_position calls table.insert with correct data."""
        client = MagicMock()
        chain = MockTableChain([])  # No existing positions
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        pid = pm.add_position(
            ticker="AAPL",
            entry_price=185.0,
            target_price=200.0,
            stop_price=175.0,
        )

        assert isinstance(pid, str)
        assert len(pid) == 32
        # Verify insert was called
        assert len(chain._inserted) == 1
        inserted = chain._inserted[0]
        assert inserted["ticker"] == "AAPL"
        assert inserted["entry_price"] == 185.0
        assert inserted["target_price"] == 200.0
        assert inserted["stop_price"] == 175.0
        assert inserted["status"] == "open"

    def test_add_duplicate_raises(self):
        """Raises ValueError if ticker already open."""
        existing = _make_position_row("AAPL", 185.0, status="open", user_id="test")
        client = MagicMock()
        # is_in_portfolio will find an existing row
        chain_with_data = MockTableChain([existing])
        client.table = MagicMock(return_value=chain_with_data)

        pm = SupabasePortfolioManager(client, user_id="test")
        with pytest.raises(ValueError, match="already in portfolio"):
            pm.add_position(ticker="AAPL", entry_price=190.0)


class TestSupabaseRemovePosition:
    def test_remove_existing_returns_true(self):
        pid = uuid.uuid4().hex
        existing = _make_position_row("NVDA", 100.0, position_id=pid, user_id="test")
        client = MagicMock()
        chain = MockTableChain([existing])
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        result = pm.remove_position(pid, exit_price=110.0)
        assert result is True
        # Verify update was called with closed status
        assert len(chain._updates) == 1
        update_data = chain._updates[0]
        assert update_data["status"] == "closed"
        assert update_data["exit_price"] == 110.0

    def test_remove_nonexistent_returns_false(self):
        client = MagicMock()
        chain = MockTableChain([])
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        result = pm.remove_position("fake_id")
        assert result is False


class TestSupabaseQueries:
    def test_get_open_positions_returns_df(self):
        rows = [
            _make_position_row("AAPL", 185.0, user_id="test"),
            _make_position_row("GOOG", 140.0, user_id="test"),
        ]
        client = MagicMock()
        chain = MockTableChain(rows)
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        df = pm.get_open_positions()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_get_open_positions_empty(self):
        client = MagicMock()
        chain = MockTableChain([])
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        df = pm.get_open_positions()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_closed_positions(self):
        closed_row = _make_position_row(
            "META", 300.0, status="closed", user_id="test",
            exit_price=310.0, exit_date=date.today().isoformat(),
            realized_return_pct=3.33, exit_reason="target",
        )
        client = MagicMock()
        chain = MockTableChain([closed_row])
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        df = pm.get_closed_positions(days=30)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "META"

    def test_is_in_portfolio_true(self):
        existing = _make_position_row("TSLA", 250.0, user_id="test")
        client = MagicMock()
        chain = MockTableChain([existing])
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        assert pm.is_in_portfolio("TSLA") is True

    def test_is_in_portfolio_false(self):
        client = MagicMock()
        chain = MockTableChain([])
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        assert pm.is_in_portfolio("TSLA") is False

    def test_get_portfolio_tickers(self):
        rows = [
            _make_position_row("AAPL", 185.0, user_id="test"),
            _make_position_row("MSFT", 350.0, user_id="test"),
        ]
        client = MagicMock()
        chain = MockTableChain(rows)
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        tickers = pm.get_portfolio_tickers()
        assert isinstance(tickers, set)
        assert tickers == {"AAPL", "MSFT"}


class TestSupabasePortfolioStats:
    def test_empty_stats(self):
        client = MagicMock()
        chain = MockTableChain([])
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        stats = pm.get_portfolio_stats()
        assert stats["open_count"] == 0
        assert stats["closed_count"] == 0
        assert stats["total_return_pct"] == 0.0
        assert stats["win_rate"] == 0.0

    def test_stats_with_open_positions(self):
        rows = [
            _make_position_row("AAPL", 100.0, shares=100, user_id="test"),
            _make_position_row("GOOG", 200.0, shares=50, user_id="test"),
        ]
        client = MagicMock()

        # First call for open, second call for closed
        call_count = [0]
        def _table(name):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockTableChain(list(rows))  # open
            return MockTableChain([])  # closed
        client.table = _table

        pm = SupabasePortfolioManager(client, user_id="test")
        stats = pm.get_portfolio_stats()
        assert stats["open_count"] == 2
        assert stats["total_invested"] == 20000.0  # 100*100 + 200*50

    def test_stats_with_closed_positions(self):
        closed = [
            _make_position_row("W1", 100.0, status="closed", user_id="test",
                               realized_return_pct=10.0, prediction_correct=True),
            _make_position_row("L1", 100.0, status="closed", user_id="test",
                               realized_return_pct=-5.0, prediction_correct=False),
        ]
        client = MagicMock()
        call_count = [0]
        def _table(name):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockTableChain([])  # open
            return MockTableChain(list(closed))  # closed
        client.table = _table

        pm = SupabasePortfolioManager(client, user_id="test")
        stats = pm.get_portfolio_stats()
        assert stats["closed_count"] == 2
        assert stats["win_rate"] == 0.5
        assert stats["avg_return"] == pytest.approx(2.5, abs=0.1)
        assert stats["prediction_accuracy"] == 0.5


class TestSupabaseUpdatePrices:
    def test_update_no_positions(self):
        client = MagicMock()
        chain = MockTableChain([])
        client.table = MagicMock(return_value=chain)

        pm = SupabasePortfolioManager(client, user_id="test")
        result = pm.update_prices()
        assert result == {"updated": 0, "auto_closed": 0, "open_count": 0}

    @patch.object(SupabasePortfolioManager, "_fetch_prices")
    def test_update_with_price_change(self, mock_fetch):
        rows = [_make_position_row("AAPL", 100.0, user_id="test")]
        client = MagicMock()

        call_count = [0]
        def _table(name):
            call_count[0] += 1
            return MockTableChain(list(rows))
        client.table = _table

        mock_fetch.return_value = {"AAPL": 105.0}
        pm = SupabasePortfolioManager(client, user_id="test")
        result = pm.update_prices()
        assert result["updated"] == 1
        assert result["auto_closed"] == 0

    @patch.object(SupabasePortfolioManager, "_fetch_prices")
    def test_auto_close_on_stop(self, mock_fetch):
        rows = [_make_position_row("BAD", 100.0, stop_price=90.0, user_id="test")]
        client = MagicMock()
        client.table = MagicMock(return_value=MockTableChain(list(rows)))

        mock_fetch.return_value = {"BAD": 85.0}
        pm = SupabasePortfolioManager(client, user_id="test")
        result = pm.update_prices()
        assert result["auto_closed"] == 1

    @patch.object(SupabasePortfolioManager, "_fetch_prices")
    def test_auto_close_on_target(self, mock_fetch):
        rows = [_make_position_row("GOOD", 100.0, target_price=120.0, user_id="test")]
        client = MagicMock()
        client.table = MagicMock(return_value=MockTableChain(list(rows)))

        mock_fetch.return_value = {"GOOD": 125.0}
        pm = SupabasePortfolioManager(client, user_id="test")
        result = pm.update_prices()
        assert result["auto_closed"] == 1


# ---------------------------------------------------------------------------
# Tests: factory function with Supabase
# ---------------------------------------------------------------------------
class TestFactoryFunction:
    def test_factory_prefers_supabase_when_configured(self):
        """When Supabase is configured, factory returns SupabasePortfolioManager."""
        from core.db.portfolio_manager import _PM_INSTANCES

        # Clean slate
        _PM_INSTANCES.pop("sb_test_user", None)

        mock_client = MagicMock()
        mock_sb_mod = MagicMock()
        mock_sb_mod.get_supabase_client = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"core.db.supabase_client": mock_sb_mod}):
            from core.db.portfolio_manager import get_portfolio_manager

            # Force re-creation by using unique user_id
            _PM_INSTANCES.pop("sb_test_user", None)
            pm = get_portfolio_manager("sb_test_user")
            assert isinstance(pm, SupabasePortfolioManager)

            # Clean up
            _PM_INSTANCES.pop("sb_test_user", None)

    def test_factory_falls_back_to_duckdb(self):
        """When Supabase is NOT configured, factory returns DuckDB PortfolioManager."""
        from core.db.portfolio_manager import _PM_INSTANCES, PortfolioManager

        _PM_INSTANCES.pop("duck_test_user", None)

        with patch("core.db.supabase_client.get_supabase_client", return_value=None):
            from core.db.portfolio_manager import get_portfolio_manager
            _PM_INSTANCES.pop("duck_test_user", None)
            pm = get_portfolio_manager("duck_test_user")
            assert isinstance(pm, PortfolioManager)

            _PM_INSTANCES.pop("duck_test_user", None)


# ---------------------------------------------------------------------------
# Tests: supabase_client module
# ---------------------------------------------------------------------------
class TestSupabaseClient:
    def test_returns_none_when_not_configured(self):
        from core.db.supabase_client import reset_supabase_client, get_supabase_client
        reset_supabase_client()

        with patch.dict("os.environ", {}, clear=True):
            with patch("core.db.supabase_client._read_secret", return_value=None):
                result = get_supabase_client()
                assert result is None

        reset_supabase_client()

    def test_caches_client(self):
        import sys
        import core.db.supabase_client as sb_mod
        sb_mod.reset_supabase_client()

        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_client)

        # Create a fake supabase module so the import inside get_supabase_client works
        mock_supabase_mod = MagicMock()
        mock_supabase_mod.create_client = mock_create

        with patch.object(sb_mod, "_read_secret", side_effect=["https://x.supabase.co", "test-key", "https://x.supabase.co", "test-key"]):
            with patch.dict(sys.modules, {"supabase": mock_supabase_mod}):
                c1 = sb_mod.get_supabase_client()
                c2 = sb_mod.get_supabase_client()
                assert c1 is c2
                mock_create.assert_called_once()

        sb_mod.reset_supabase_client()

    def test_reset_clears_cache(self):
        import core.db.supabase_client as sb_mod
        sb_mod._CLIENT_CACHE = "fake"
        sb_mod.reset_supabase_client()
        assert sb_mod._CLIENT_CACHE is None
