"""Unit tests for the SupabaseScanManager.

Uses a mock Supabase client so no real network calls are made.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.db.scan_manager import SupabaseScanManager, get_scan_manager


# ---------------------------------------------------------------------------
# Mock helpers (same pattern as test_supabase_portfolio.py)
# ---------------------------------------------------------------------------
class MockSupabaseResponse:
    def __init__(self, data=None):
        self.data = data or []


class MockTableChain:
    def __init__(self, rows=None):
        self._rows = rows if rows is not None else []
        self._upserted = []

    def select(self, *args, **kwargs):
        return self

    def upsert(self, data, **kwargs):
        self._upserted.append(data)
        return self

    def insert(self, data, **kwargs):
        return self

    def eq(self, col, val):
        self._rows = [r for r in self._rows if r.get(col) == val]
        return self

    def gte(self, col, val):
        self._rows = [r for r in self._rows if (r.get(col) or "") >= val]
        return self

    def order(self, col, desc=False):
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def execute(self):
        return MockSupabaseResponse(self._rows)


def _make_mock_client(rows=None):
    client = MagicMock()
    _rows = rows if rows is not None else []

    def _table(name):
        return MockTableChain(list(_rows))

    client.table = _table
    return client


# ---------------------------------------------------------------------------
# Tests: save_scan
# ---------------------------------------------------------------------------
class TestSaveScan:
    def test_save_empty_returns_zero(self):
        sm = SupabaseScanManager(_make_mock_client(), "default")
        assert sm.save_scan("scan_1", pd.DataFrame(), {}, {}) == 0

    def test_save_none_returns_zero(self):
        sm = SupabaseScanManager(_make_mock_client(), "default")
        assert sm.save_scan("scan_1", None, {}, {}) == 0

    def test_save_returns_count(self):
        client = _make_mock_client()
        sm = SupabaseScanManager(client, "default")

        df = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT"],
            "FinalScore_20d": [75.0, 68.0],
            "Entry_Price": [180.0, 350.0],
            "Target_Price": [200.0, 380.0],
            "Stop_Loss": [170.0, 330.0],
            "RSI": [55.0, 48.0],
            "Holding_Days": [20, 12],
        })
        result = sm.save_scan("scan_test", df, {"key": "val"}, {"universe_size": 1000})
        assert result == 2

    def test_save_nan_values_become_none(self):
        """NaN values should be converted to None for Supabase JSON serialization."""
        upserted_rows = []

        class CapturingChain(MockTableChain):
            def upsert(self, data, **kwargs):
                upserted_rows.append(data)
                return self

        client = MagicMock()
        client.table = lambda name: CapturingChain()

        sm = SupabaseScanManager(client, "default")
        df = pd.DataFrame({
            "Ticker": ["TEST"],
            "FinalScore_20d": [np.nan],
            "RSI": [float("nan")],
        })
        sm.save_scan("scan_nan", df, {}, {})

        # Find recommendation row (not the scan metadata row)
        rec_rows = [r for r in upserted_rows if "ticker" in r]
        assert len(rec_rows) == 1
        rec = rec_rows[0]
        assert rec["final_score"] is None
        assert rec["rsi"] is None

    def test_save_column_mapping(self):
        """Verify DataFrame columns are mapped to DB columns via _REC_COL_MAP."""
        upserted_rows = []

        class CapturingChain(MockTableChain):
            def upsert(self, data, **kwargs):
                upserted_rows.append(data)
                return self

        client = MagicMock()
        client.table = lambda name: CapturingChain()

        sm = SupabaseScanManager(client, "default")
        df = pd.DataFrame({
            "Ticker": ["AAPL"],
            "FinalScore_20d": [80.5],
            "TechScore_20d": [70.0],
            "ML_20d_Prob": [0.72],
            "RiskClass": ["CORE"],
            "Sector": ["Technology"],
            "Entry_Price": [185.0],
            "Target_Price": [200.0],
            "Stop_Loss": [175.0],
            "Holding_Days": [18],
        })
        sm.save_scan("scan_map", df, {}, {})

        rec_rows = [r for r in upserted_rows if "ticker" in r]
        assert len(rec_rows) == 1
        rec = rec_rows[0]
        assert rec["ticker"] == "AAPL"
        assert rec["final_score"] == 80.5
        assert rec["tech_score"] == 70.0
        assert rec["ml_prob"] == 0.72
        assert rec["risk_class"] == "CORE"
        assert rec["sector"] == "Technology"
        assert rec["entry_price"] == 185.0
        assert rec["target_price"] == 200.0
        assert rec["stop_price"] == 175.0
        assert rec["holding_days"] == 18


# ---------------------------------------------------------------------------
# Tests: read operations
# ---------------------------------------------------------------------------
class TestReadOperations:
    def test_get_scan_history_returns_dataframe(self):
        rows = [
            {"scan_id": "scan_1", "timestamp": "2026-03-06T12:00:00Z",
             "user_id": "default", "total_recommended": 20},
            {"scan_id": "scan_2", "timestamp": "2026-03-05T12:00:00Z",
             "user_id": "default", "total_recommended": 15},
        ]
        sm = SupabaseScanManager(_make_mock_client(rows), "default")
        df = sm.get_scan_history(days=7)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_get_scan_history_empty(self):
        sm = SupabaseScanManager(_make_mock_client([]), "default")
        df = sm.get_scan_history(days=7)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_latest_scan_returns_recommendations(self):
        # Need two tables: scan_history returns scan_id, scan_recommendations returns recs
        scan_rows = [{"scan_id": "scan_latest", "user_id": "default"}]
        rec_rows = [
            {"scan_id": "scan_latest", "ticker": "AAPL", "final_score": 80.0},
            {"scan_id": "scan_latest", "ticker": "MSFT", "final_score": 75.0},
        ]

        # Custom client that returns different rows per table
        client = MagicMock()
        table_data = {
            "scan_history": scan_rows,
            "scan_recommendations": rec_rows,
        }
        client.table = lambda name: MockTableChain(list(table_data.get(name, [])))

        sm = SupabaseScanManager(client, "default")
        df = sm.get_latest_scan()
        assert df is not None
        assert len(df) == 2

    def test_get_latest_scan_no_scans(self):
        sm = SupabaseScanManager(_make_mock_client([]), "default")
        result = sm.get_latest_scan()
        assert result is None

    def test_get_recommendations_for_scan(self):
        rows = [
            {"scan_id": "scan_x", "ticker": "TSLA", "final_score": 65.0},
        ]
        sm = SupabaseScanManager(_make_mock_client(rows), "default")
        df = sm.get_recommendations_for_scan("scan_x")
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "TSLA"


# ---------------------------------------------------------------------------
# Tests: factory function
# ---------------------------------------------------------------------------
class TestFactory:
    def test_returns_none_without_supabase(self):
        import core.db.scan_manager as sm_mod
        sm_mod._SM_INSTANCES.clear()

        with patch("core.db.supabase_client.get_supabase_client", return_value=None):
            result = get_scan_manager("test_user_none")
            assert result is None

    def test_returns_manager_with_supabase(self):
        import core.db.scan_manager as sm_mod
        sm_mod._SM_INSTANCES.clear()

        mock_client = _make_mock_client()
        with patch("core.db.supabase_client.get_supabase_client", return_value=mock_client):
            result = get_scan_manager("test_user_ok")
            assert isinstance(result, SupabaseScanManager)
