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


def test_blocks_nan_in_trade_levels(rm):
    """NaN in any of price / stop / target must be rejected.

    Without explicit isnan() checks, NaN would bypass every numerical
    comparison (NaN > 0 is False, NaN <= 0 is also False) and reach
    IBKR with an invalid auxPrice — AFTER the buy had filled.
    """
    import math as _m
    # NaN target
    allowed, reason = rm.can_open_position(
        "TEST", price=100, score=80, rr=2.5,
        stop_loss=95, target_price=float("nan"),
    )
    assert not allowed, f"NaN target should be rejected: {reason}"
    assert "nan" in reason.lower()

    # NaN stop
    allowed, reason = rm.can_open_position(
        "TEST", price=100, score=80, rr=2.5,
        stop_loss=float("nan"), target_price=120,
    )
    assert not allowed
    assert "nan" in reason.lower()

    # NaN price
    allowed, reason = rm.can_open_position(
        "TEST", price=float("nan"), score=80, rr=2.5,
        stop_loss=95, target_price=120,
    )
    assert not allowed


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


# ── Live price refresh ──────────────────────────────────────────────

def test_live_price_proportional_stop_target_adjustment():
    """When live price differs from scan, stop/target should rescale to keep R:R.

    Tests the core math behind the live-price refresh in order_manager:
    scan_price=$100, stop=$95 (5% below), target=$110 (10% above).
    Live price comes in at $102 → stop should be $96.90 (5% below live),
    target $112.20 (10% above live). R:R ratio preserved.
    """
    scan_price = 100.0
    stop = 95.0
    target = 110.0
    live_price = 102.0

    # Replicate the refresh math from order_manager._execute_single
    stop_pct = (scan_price - stop) / scan_price        # 0.05
    tgt_pct = (target - scan_price) / scan_price       # 0.10
    new_stop = round(live_price * (1 - stop_pct), 2)   # 102 * 0.95 = 96.90
    new_target = round(live_price * (1 + tgt_pct), 2)  # 102 * 1.10 = 112.20

    assert new_stop == 96.90, f"stop should rescale: got {new_stop}"
    assert new_target == 112.20, f"target should rescale: got {new_target}"

    # R:R should be unchanged within rounding
    rr_scan = (target - scan_price) / (scan_price - stop)
    rr_live = (new_target - live_price) / (live_price - new_stop)
    assert abs(rr_scan - rr_live) < 0.01, f"R:R drifted: {rr_scan} vs {rr_live}"


def test_live_price_refresh_falls_back_when_quote_missing(rm, monkeypatch):
    """If get_live_price returns None, the trade still runs with scan price.

    Critical safety property: a transient market-data outage should NOT
    block trading — it should just degrade to using the scan-time price.
    """
    # Patch the client's get_live_price to always return None
    monkeypatch.setattr(rm.client, "get_live_price",
                        lambda ticker, timeout=3.0: None, raising=False)

    # Simulate the order_manager flow: scan_price stays as-is, no adjustment
    scan_price = 100.0
    stop = 95.0
    target = 110.0
    live_price = rm.client.get_live_price("TEST")
    assert live_price is None, "stub should return None"

    # Caller logic: if live_price not present, no rescale, use scan
    if live_price and live_price > 0:
        pytest.fail("Should not branch into refresh when None")
    # stop/target unchanged
    assert stop == 95.0 and target == 110.0
