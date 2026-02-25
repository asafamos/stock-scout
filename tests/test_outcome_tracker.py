"""Tests for core.db.outcome_tracker — outcome tracking logic."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def store(tmp_path):
    from core.db.store import ScanStore
    s = ScanStore(db_path=str(tmp_path / "test.duckdb"))
    s.initialize()
    return s


@pytest.fixture
def tracker(store):
    from core.db.outcome_tracker import OutcomeTracker
    return OutcomeTracker(store)


@pytest.fixture
def scan_with_recs(store):
    """Save a scan and return (scan_id, results_df)."""
    scan_id = store.generate_scan_id()
    df = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "GOOG"],
        "FinalScore_20d": [82.5, 76.3, 71.0],
        "Close": [185.0, 420.0, 175.0],
        "Target_20d": [200.0, 450.0, 195.0],
        "Stop": [175.0, 400.0, 165.0],
        "RR": [2.5, 1.8, 3.1],
    })
    store.save_scan(scan_id, df, {}, {})
    return scan_id, df


class TestOutcomeTracker:
    def test_register_recommendations(self, tracker, scan_with_recs):
        scan_id, df = scan_with_recs
        n = tracker.register_recommendations(scan_id, df)
        assert n == 3

    def test_register_idempotent(self, tracker, scan_with_recs):
        """Registering same scan twice should not double-count."""
        scan_id, df = scan_with_recs
        n1 = tracker.register_recommendations(scan_id, df)
        n2 = tracker.register_recommendations(scan_id, df)
        assert n1 == 3
        assert n2 == 0  # already registered

    def test_register_skips_zero_price(self, tracker, store):
        """Rows with no entry price should be skipped."""
        scan_id = store.generate_scan_id()
        df = pd.DataFrame({
            "Ticker": ["BAD"],
            "Close": [0.0],
        })
        store.save_scan(scan_id, df, {}, {})
        n = tracker.register_recommendations(scan_id, df)
        assert n == 0

    def test_register_empty_df(self, tracker):
        """Empty DataFrame should return 0."""
        n = tracker.register_recommendations("nope", pd.DataFrame())
        assert n == 0

    def test_performance_summary_empty(self, tracker):
        """No completed outcomes → informative message."""
        perf = tracker.get_performance_summary()
        assert perf["n_completed"] == 0

    def test_score_vs_outcome_empty(self, tracker):
        """No completed outcomes → empty DataFrame."""
        result = tracker.get_score_vs_outcome()
        assert result.empty

    def test_pending_status_after_register(self, tracker, scan_with_recs, store):
        """Newly registered outcomes should be 'pending'."""
        scan_id, df = scan_with_recs
        tracker.register_recommendations(scan_id, df)

        import duckdb
        con = duckdb.connect(store._db_path)
        statuses = con.execute(
            "SELECT status FROM outcomes"
        ).fetchdf()
        con.close()
        assert all(statuses["status"] == "pending")

    def test_update_outcomes_no_pending(self, tracker):
        """With no pending outcomes, update should be a no-op."""
        summary = tracker.update_outcomes()
        assert summary["updated"] == 0
        assert summary["completed"] == 0
