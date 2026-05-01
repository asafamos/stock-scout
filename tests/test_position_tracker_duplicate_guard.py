"""Regression test for the duplicate-CLOSE guard in
`PositionTracker.remove_position`.

Why this test exists: on 2026-05-01 a `git pull` on the VPS
overwrote the live `open_positions.json` with a stale committed
snapshot, putting CF back in the tracker after it had already been
closed earlier the same day. The monitor then re-detected CF as
"missing from IB" and called `remove_position` AGAIN — writing a
SECOND CLOSE row with the same +$22.24 P&L. /pnl reported double.

The guard refuses to write a same-ticker same-day duplicate CLOSE.
This test pins that behavior down so a future refactor can't quietly
re-introduce the duplicate path.
"""
from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from core.trading.position_tracker import PositionTracker


@pytest.fixture
def tracker(tmp_path: Path):
    """Tracker with isolated JSON files per test (via TradingConfig
    pointing at tmp_path so tests can never touch real production state)."""
    from core.trading.config import TradingConfig
    cfg = TradingConfig()
    cfg.open_positions_path = str(tmp_path / "open_positions.json")
    cfg.trade_log_path = str(tmp_path / "trade_log.json")
    return PositionTracker(config=cfg)


def _seed_position(tracker: PositionTracker, ticker: str = "CF",
                   qty: int = 2, entry: float = 111.33):
    """Helper: place an open position via add_position."""
    tracker.add_position(
        ticker=ticker,
        quantity=qty,
        entry_price=entry,
        stop_loss=entry * 0.95,
        target_price=entry * 1.30,
        trailing_stop_pct=5.0,
        score=80.0,
        order_ids={"buy": 1, "trailing_stop": 2, "limit_sell": 3,
                   "oca_group": "SS_TEST"},
    )


def _count_closes(tracker: PositionTracker, ticker: str) -> int:
    """Count CLOSE rows for ticker today in trade_log."""
    today = date.today().isoformat()
    return sum(
        1 for e in tracker.get_trade_log()
        if e.get("ticker") == ticker
        and e.get("action") == "CLOSE"
        and str(e.get("timestamp", "")).startswith(today)
    )


class TestDuplicateCloseGuard:
    def test_first_close_writes_normally(self, tracker):
        _seed_position(tracker, "CF", 2, 111.33)
        tracker.remove_position("CF", exit_price=122.45,
                                reason="stop_or_target_filled")
        assert _count_closes(tracker, "CF") == 1
        # Position removed from open_positions
        assert all(p["ticker"] != "CF" for p in tracker.get_open_positions())

    def test_second_close_same_ticker_same_day_is_dropped(self, tracker):
        """The exact scenario from 2026-05-01: a stale tracker entry
        re-appears (e.g. via git pull overwriting the JSON file) and
        the monitor calls remove_position again. The guard must
        prevent the second CLOSE row from being written."""
        _seed_position(tracker, "CF", 2, 111.33)
        tracker.remove_position("CF", exit_price=122.45,
                                reason="stop_or_target_filled")
        assert _count_closes(tracker, "CF") == 1

        # Simulate a "ghost" CF position appearing (git pull, manual
        # edit, restored backup, etc).
        _seed_position(tracker, "CF", 2, 111.33)
        # Monitor's close path triggers again.
        tracker.remove_position("CF", exit_price=122.45,
                                reason="stop_or_target_filled")

        # Second CLOSE was REFUSED — only one row in the log
        assert _count_closes(tracker, "CF") == 1, (
            "duplicate-CLOSE guard failed — second CLOSE row was written"
        )
        # Position was still removed from open_positions (so the next
        # monitor cycle doesn't keep firing this guard repeatedly)
        assert all(p["ticker"] != "CF" for p in tracker.get_open_positions())

    def test_close_different_ticker_same_day_writes(self, tracker):
        """Same-day CLOSE on a DIFFERENT ticker is fine — guard is
        per-ticker, not global."""
        _seed_position(tracker, "CF", 2, 111.33)
        _seed_position(tracker, "SVM", 15, 12.29)
        tracker.remove_position("CF", exit_price=122.45, reason="trail")
        tracker.remove_position("SVM", exit_price=12.07, reason="trail")
        assert _count_closes(tracker, "CF") == 1
        assert _count_closes(tracker, "SVM") == 1

    def test_reconcile_drop_unaffected_by_guard(self, tracker):
        """reconcile_drop writes a different action ('RECONCILE_DROP'),
        so the CLOSE guard doesn't apply to it."""
        _seed_position(tracker, "TDW", 2, 85.83)
        tracker.reconcile_drop("TDW", reason="test")
        # RECONCILE_DROP is not CLOSE, so the count is 0
        assert _count_closes(tracker, "TDW") == 0
        # And there should be one RECONCILE_DROP row
        rd_count = sum(
            1 for e in tracker.get_trade_log()
            if e.get("ticker") == "TDW"
            and e.get("action") == "RECONCILE_DROP"
        )
        assert rd_count == 1
