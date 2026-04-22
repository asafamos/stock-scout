"""Smoke tests for phase 2/3 trading gates — run locally to catch bugs.

Not a full unit-test suite; just enough to verify each new gate:
  - doesn't crash on missing data
  - returns the right (allowed, reason) shape
  - fail-opens on data errors instead of blocking silently

Run with: pytest tests/test_phase2_gates.py -v
"""

from __future__ import annotations

import pytest

from core.trading.config import TradingConfig
from core.trading.ibkr_client import IBKRClient
from core.trading.position_tracker import PositionTracker
from core.trading.risk_manager import RiskManager


@pytest.fixture
def rm(tmp_path, monkeypatch):
    """RiskManager with dry-run client + temp tracker paths."""
    monkeypatch.setenv("DRY_RUN", "1")
    cfg = TradingConfig()
    cfg.dry_run = True
    cfg.open_positions_path = str(tmp_path / "open.json")
    cfg.trade_log_path = str(tmp_path / "log.json")
    client = IBKRClient(cfg)
    tracker = PositionTracker(cfg)
    return RiskManager(client, tracker, cfg)


# ── Dynamic sizing ────────────────────────────────────────

def test_conviction_tiers_scale_correctly(rm):
    """Higher score+R:R should produce larger size, strictly."""
    # Use identical ATR (remove vol adjustment noise)
    high = rm.calculate_qty(100, cash_available=1000, atr_pct=2.0, score=85, rr=3.0)
    med = rm.calculate_qty(100, cash_available=1000, atr_pct=2.0, score=76, rr=2.1)
    low = rm.calculate_qty(100, cash_available=1000, atr_pct=2.0, score=74, rr=1.9)
    assert high >= med >= low, f"Sizing not monotonic: high={high}, med={med}, low={low}"
    assert high > low, "High-conviction should get strictly more than low"


def test_sizing_respects_cash_ceiling(rm):
    """Never buy more than cash allows."""
    qty = rm.calculate_qty(100, cash_available=150, score=85, rr=3.0, atr_pct=2)
    assert qty * 100 <= 150, f"Exceeded cash: qty={qty}"


def test_sizing_handles_zero_price(rm):
    assert rm.calculate_qty(0, cash_available=1000, score=80, rr=2) == 0


# ── Trade level validation ────────────────────────────────

def test_blocks_trade_with_invalid_target(rm):
    """Target ≤ entry × 1.02 should be rejected."""
    allowed, reason = rm.can_open_position(
        "TEST", price=100, score=80, rr=2.5,
        stop_loss=95, target_price=101,  # only +1% upside
    )
    assert not allowed
    assert "target" in reason.lower()


def test_blocks_trade_with_invalid_stop(rm):
    """Stop ≥ entry × 0.995 should be rejected."""
    allowed, reason = rm.can_open_position(
        "TEST", price=100, score=80, rr=2.5,
        stop_loss=100,  # stop at entry — can never trigger
        target_price=120,
    )
    assert not allowed
    assert "stop" in reason.lower()


# ── Sector momentum ───────────────────────────────────────

def test_sector_momentum_unknown_sector_fails_open(rm):
    """Unknown sector name should be allowed (no data → no gate)."""
    allowed, reason, _ = rm.check_sector_momentum("Nonexistent Sector")
    assert allowed, f"Unknown sector should fail-open: {reason}"


# ── Portfolio correlation ─────────────────────────────────

def test_correlation_empty_portfolio_allows(rm):
    """No existing positions → correlation check always passes."""
    allowed, reason, corr = rm.check_portfolio_correlation("AAPL")
    assert allowed
    assert corr == 0.0


def test_correlation_check_doesnt_crash_on_bad_ticker(rm):
    """A ticker that yfinance can't resolve should fail-open, not error."""
    rm.tracker.add_position(
        ticker="AAPL", quantity=1, entry_price=150,
        stop_loss=145, target_price=170,
    )
    allowed, reason, _ = rm.check_portfolio_correlation("ZZZZZZ_NONEXISTENT")
    # Should not raise, and should allow
    assert allowed, f"Unresolvable ticker should fail-open: {reason}"
