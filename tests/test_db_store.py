"""Tests for core.db.store — DuckDB scan store."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary DuckDB file path."""
    return str(tmp_path / "test.duckdb")


@pytest.fixture
def store(tmp_db):
    from core.db.store import ScanStore
    s = ScanStore(db_path=tmp_db)
    s.initialize()
    return s


@pytest.fixture
def sample_results():
    """Minimal DataFrame resembling pipeline output."""
    return pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "GOOG"],
        "FinalScore_20d": [82.5, 76.3, 71.0],
        "TechScore_20d": [85.0, 70.0, 68.0],
        "Fundamental_Score": [78.0, 80.0, 65.0],
        "ML_20d_Prob": [0.65, 0.58, 0.52],
        "Close": [185.0, 420.0, 175.0],
        "RSI": [55.0, 62.0, 48.0],
        "ATR_Pct": [0.025, 0.03, 0.022],
        "RR": [2.5, 1.8, 3.1],
        "Risk_Label": ["MODERATE", "MODERATE-HIGH", "MODERATE"],
        "Market_Regime": ["TREND_UP", "TREND_UP", "TREND_UP"],
    })


class TestScanStore:
    def test_initialize_creates_tables(self, store, tmp_db):
        """Tables should exist after initialization."""
        import duckdb
        con = duckdb.connect(tmp_db)
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchdf()
        con.close()
        table_names = set(tables["table_name"].tolist())
        assert "scans" in table_names
        assert "recommendations" in table_names
        assert "outcomes" in table_names
        assert "schema_version" in table_names

    def test_double_initialize_is_safe(self, store):
        """Calling initialize() twice should not error."""
        store.initialize()
        store.initialize()

    def test_save_scan_and_read(self, store, sample_results):
        """Save a scan and read it back."""
        scan_id = store.generate_scan_id()
        n = store.save_scan(
            scan_id, sample_results,
            config={"budget": 5000},
            metadata={"universe_name": "sp500", "universe_size": 500},
        )
        assert n == 3

        latest = store.get_latest_scan()
        assert latest is not None
        assert len(latest) == 3
        assert set(latest["ticker"].tolist()) == {"AAPL", "MSFT", "GOOG"}

    def test_save_empty_df(self, store):
        """Saving empty DataFrame should return 0."""
        n = store.save_scan("empty", pd.DataFrame(), {}, {})
        assert n == 0

    def test_scan_history(self, store, sample_results):
        """Scan history should show recent scans."""
        scan_id = store.generate_scan_id()
        store.save_scan(scan_id, sample_results, {}, {})

        history = store.get_scan_history(days=1)
        assert len(history) >= 1

    def test_query(self, store, sample_results):
        """Custom SQL queries should work."""
        scan_id = store.generate_scan_id()
        store.save_scan(scan_id, sample_results, {}, {})

        result = store.query(
            "SELECT ticker, final_score FROM recommendations WHERE final_score > 75"
        )
        assert len(result) == 2  # AAPL=82.5, MSFT=76.3

    def test_generate_scan_id(self, store):
        """Scan IDs should be unique."""
        id1 = store.generate_scan_id()
        id2 = store.generate_scan_id()
        assert id1 != id2
        assert id1.startswith("scan_")

    def test_get_stats(self, store, sample_results):
        """Stats should reflect stored data."""
        scan_id = store.generate_scan_id()
        store.save_scan(scan_id, sample_results, {}, {})

        stats = store.get_stats()
        assert stats["total_scans"] == 1
        assert stats["total_recommendations"] == 3

    def test_nan_handling(self, store):
        """NaN values should be stored as NULL, not crash."""
        df = pd.DataFrame({
            "Ticker": ["TEST"],
            "FinalScore_20d": [np.nan],
            "ML_20d_Prob": [None],
            "Close": [100.0],
        })
        scan_id = store.generate_scan_id()
        n = store.save_scan(scan_id, df, {}, {})
        assert n == 1

        latest = store.get_latest_scan()
        assert latest is not None
        # NaN stored as NULL, read back as NaN
        assert pd.isna(latest.iloc[0]["final_score"])

    def test_multiple_scans(self, store, sample_results):
        """Multiple scans should be stored independently."""
        id1 = store.generate_scan_id()
        store.save_scan(id1, sample_results.head(2), {}, {})

        id2 = store.generate_scan_id()
        store.save_scan(id2, sample_results.tail(1), {}, {})

        stats = store.get_stats()
        assert stats["total_scans"] == 2
        assert stats["total_recommendations"] == 3

        # Latest should be the second scan
        latest = store.get_latest_scan()
        assert len(latest) == 1
        assert latest.iloc[0]["ticker"] == "GOOG"

    def test_idempotent_save(self, store, sample_results):
        """Saving the same scan_id twice should overwrite (INSERT OR REPLACE)."""
        scan_id = "test_idempotent"
        store.save_scan(scan_id, sample_results, {}, {})
        store.save_scan(scan_id, sample_results, {}, {})

        stats = store.get_stats()
        assert stats["total_scans"] == 1
        assert stats["total_recommendations"] == 3
