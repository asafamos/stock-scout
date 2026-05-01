"""Regression tests for `core.trading.policy` — the single source of truth
for buy-eligibility gates and execution preview.

Why this suite exists: bugs in `policy.py` ripple through TWO code paths
simultaneously (the dashboard preview AND the production trader), so a
silent regression here means the user sees "🚀 BUY ELIGIBLE" while the
real trader also lets through a stock that should have been blocked.
The cost of letting these bugs ship is real money.

Coverage focus:
  - Each individual gate in `evaluate_static_gates` (pass + fail paths)
  - `compute_execution_preview` math for: scan-only, live-refresh,
    slippage-reject, throttle multiplier, qty fallback for high-priced
    stocks, regime-mult for trail %.
  - The two helper functions `regime_score_floor` + `confidence_floor`
    that risk_manager and order_manager also depend on.
  - `normalize_ticker_for_ib` for the share-class tickers (BRK.B etc.).

Tests do NOT touch IB, network, or yfinance — `policy.py` is pure logic.
"""
from __future__ import annotations

import pytest

from core.trading.config import TradingConfig
from core.trading.policy import (
    GateResult,
    compute_execution_preview,
    confidence_floor,
    evaluate_static_gates,
    normalize_ticker_for_ib,
    regime_score_floor,
)


# ────────────────────────────────────────────────────────────────────
# Test fixtures
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg():
    """A fresh config matching production defaults — explicit values
    so a future env-var change can't quietly invalidate these tests."""
    c = TradingConfig()
    c.min_score_to_trade = 73.0
    c.max_score_to_trade = 95.0
    c.min_rr_to_trade = 2.0
    c.min_ml_prob = 0.33
    c.min_confidence = "High"
    c.min_reliability = 50.0
    c.blocked_sectors = "Consumer Defensive"
    c.blocked_regimes = "PANIC,CORRECTION"
    return c


@pytest.fixture
def good_row():
    """A scan row that should pass every static gate."""
    return {
        "Ticker": "AAPL",
        "FinalScore_20d": 80.0,
        "RewardRisk": 2.5,
        "ML_20d_Prob": 0.45,
        "Sector": "Technology",
        "SignalQuality": "High",
        "Market_Regime": "MODERATE_UP",
        "Reliability_Score": 85.0,
        "Entry_Price": 150.0,
        "Stop_Loss": 142.0,
        "Target_Price": 170.0,
        "ATR_Pct": 2.5,
    }


# ────────────────────────────────────────────────────────────────────
# evaluate_static_gates — the eligibility evaluator
# ────────────────────────────────────────────────────────────────────


class TestEvaluateStaticGates:
    """Each gate gets one happy-path test and one fail-path test."""

    def test_clean_row_passes_all_gates(self, good_row, cfg):
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert result.would_buy is True, result.gates_failed
        assert result.verdict == "BUY ELIGIBLE"
        assert result.primary_reason == ""

    # ── 1. Pause flag ───────────────────────────────────────────────
    def test_paused_state_blocks(self, good_row, cfg):
        result = evaluate_static_gates(good_row, cfg=cfg, state={"paused": True})
        assert not result.would_buy
        assert "PAUSED" in result.primary_reason

    # ── 2. Throttle halt ────────────────────────────────────────────
    def test_throttle_halt_blocks(self, good_row, cfg):
        state = {"throttle": {"level": "halt", "win_rate": 0.15}}
        result = evaluate_static_gates(good_row, cfg=cfg, state=state)
        assert not result.would_buy
        assert "throttle" in result.primary_reason.lower()
        assert "halt" in result.primary_reason.lower()

    def test_throttle_warn_allows(self, good_row, cfg):
        # WARN reduces size but doesn't block
        state = {"throttle": {"level": "warn", "win_rate": 0.28}}
        result = evaluate_static_gates(good_row, cfg=cfg, state=state)
        assert result.would_buy

    # ── 3. Already-holding ──────────────────────────────────────────
    def test_already_holding_blocks(self, good_row, cfg):
        state = {"positions": [{"ticker": "AAPL"}]}
        result = evaluate_static_gates(good_row, cfg=cfg, state=state)
        assert not result.would_buy
        assert "Already holding AAPL" in result.primary_reason

    # ── 4. Regime block ─────────────────────────────────────────────
    def test_panic_regime_blocks(self, good_row, cfg):
        good_row["Market_Regime"] = "PANIC"
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        # Should hit regime block first
        assert "PANIC" in result.gates_failed[0]

    def test_correction_regime_blocks(self, good_row, cfg):
        good_row["Market_Regime"] = "CORRECTION"
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert any("CORRECTION" in f for f in result.gates_failed)

    # ── 5. Score floor (regime-aware) ───────────────────────────────
    def test_low_score_blocks(self, good_row, cfg):
        good_row["FinalScore_20d"] = 50
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "Score" in result.primary_reason

    def test_score_above_cap_blocks(self, good_row, cfg):
        # Q5 underperforms — cap at max_score_to_trade
        good_row["FinalScore_20d"] = 99
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "cap" in result.primary_reason.lower()

    def test_sideways_demands_higher_score(self, good_row, cfg):
        """SIDEWAYS regime requires score ≥ 75 (vs 65 in MODERATE_UP)."""
        good_row["Market_Regime"] = "SIDEWAYS"
        good_row["SignalQuality"] = "High"  # Confidence floor is High in SIDEWAYS
        good_row["FinalScore_20d"] = 70  # Would pass MODERATE_UP, fails SIDEWAYS
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "SIDEWAYS" in result.primary_reason

    # ── 6. RR floor ─────────────────────────────────────────────────
    def test_low_rr_blocks(self, good_row, cfg):
        good_row["RewardRisk"] = 1.5
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "R:R" in result.primary_reason

    # ── 7. ML floor ─────────────────────────────────────────────────
    def test_low_ml_blocks(self, good_row, cfg):
        good_row["ML_20d_Prob"] = 0.20
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "ML" in result.primary_reason

    # ── 8. Confidence (regime-aware) ────────────────────────────────
    def test_medium_confidence_blocked_in_sideways(self, good_row, cfg):
        good_row["Market_Regime"] = "SIDEWAYS"
        good_row["FinalScore_20d"] = 80
        good_row["SignalQuality"] = "Medium"  # SIDEWAYS demands High
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "Confidence" in result.primary_reason

    def test_medium_confidence_allowed_in_trend_up(self, good_row, cfg):
        good_row["Market_Regime"] = "MODERATE_UP"
        good_row["SignalQuality"] = "Medium"  # MODERATE_UP allows Medium
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert result.would_buy, result.gates_failed

    # ── 9. Sector blocklist ─────────────────────────────────────────
    def test_blocked_sector_rejected(self, good_row, cfg):
        good_row["Sector"] = "Consumer Defensive"
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "Consumer Defensive" in result.primary_reason

    # ── 10. Reliability ─────────────────────────────────────────────
    def test_low_reliability_blocks(self, good_row, cfg):
        good_row["Reliability_Score"] = 30
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "Reliability" in result.primary_reason

    # ── 11. Trade-level sanity ──────────────────────────────────────
    def test_target_below_minimum_blocks(self, good_row, cfg):
        good_row["Target_Price"] = 151.0  # entry 150 × 1.02 = 153 minimum
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "Target" in result.primary_reason

    def test_stop_above_minimum_blocks(self, good_row, cfg):
        good_row["Stop_Loss"] = 149.5  # entry 150 × 0.995 = 149.25 cap
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "Stop" in result.primary_reason


# ────────────────────────────────────────────────────────────────────
# regime_score_floor — used by risk_manager + order_manager + dashboard
# ────────────────────────────────────────────────────────────────────


class TestRegimeScoreFloor:
    """Parity check — risk_manager and dashboard MUST agree on floors."""

    def test_known_regimes_use_table(self, cfg):
        # MODERATE_UP scan inclusion = 60, +5 buffer = 65
        assert regime_score_floor("MODERATE_UP", cfg) == pytest.approx(65.0)
        # SIDEWAYS scan inclusion = 70, +5 buffer = 75
        assert regime_score_floor("SIDEWAYS", cfg) == pytest.approx(75.0)
        # TREND_UP scan inclusion = 55, +5 buffer = 60
        assert regime_score_floor("TREND_UP", cfg) == pytest.approx(60.0)

    def test_unknown_regime_falls_back_to_static(self, cfg):
        # Falls back to base - 5 + 5 = base = 73
        assert regime_score_floor("UNKNOWN_REGIME", cfg) == pytest.approx(73.0)

    def test_empty_regime_returns_base(self, cfg):
        assert regime_score_floor("", cfg) == pytest.approx(73.0)

    def test_panic_demands_unreachable_score(self, cfg):
        # PANIC = 100 in REGIME_MIN_SCORE, +5 = 105 (effectively blocks)
        assert regime_score_floor("PANIC", cfg) == pytest.approx(105.0)


class TestConfidenceFloor:
    def test_high_required_in_sideways(self, cfg):
        assert confidence_floor("SIDEWAYS", cfg) == 3  # High

    def test_medium_allowed_in_moderate_up(self, cfg):
        assert confidence_floor("MODERATE_UP", cfg) == 2  # Medium

    def test_medium_allowed_in_trend_up(self, cfg):
        assert confidence_floor("TREND_UP", cfg) == 2

    def test_default_is_high(self, cfg):
        assert confidence_floor("NEUTRAL", cfg) == 3
        assert confidence_floor("", cfg) == 3


# ────────────────────────────────────────────────────────────────────
# compute_execution_preview — what the trader actually submits
# ────────────────────────────────────────────────────────────────────


class TestExecutionPreview:
    def test_basic_scan_numbers(self, good_row, cfg):
        p = compute_execution_preview(good_row, cfg=cfg, available_cash=1000)
        assert p.ticker == "AAPL"
        assert p.entry == 150.0
        assert p.stop == 142.0
        assert p.target == 170.0
        # MODERATE_UP regime → 1.20× trail multiplier
        assert p.regime_mult == pytest.approx(1.20)
        # base = avg(scan stop %, 1.5×ATR) = avg(5.3, 3.75) = 4.5
        # final = 4.5 × 1.20 = 5.4 → 5.4 (clamped)
        assert 4.0 <= p.trail_pct <= 6.5

    def test_trail_clamped_to_floor(self, good_row, cfg):
        good_row["Market_Regime"] = "PANIC"  # 0.70× mult
        good_row["ATR_Pct"] = 1.0  # tiny vol
        good_row["Stop_Loss"] = 149.0  # 0.67% scan stop → very tight
        p = compute_execution_preview(good_row, cfg=cfg, available_cash=1000)
        assert p.trail_pct >= 2.0, "trail should never go below 2%"

    def test_trail_clamped_to_cap(self, good_row, cfg):
        good_row["Market_Regime"] = "TREND_UP"  # 1.20×
        good_row["ATR_Pct"] = 10.0  # massive vol → 15% base
        p = compute_execution_preview(good_row, cfg=cfg, available_cash=1000)
        assert p.trail_pct <= 9.0, "trail should never exceed 9%"

    def test_live_price_rescales_stop_target(self, good_row, cfg):
        # Live price up 2% → stop and target both rescale up 2%
        p = compute_execution_preview(
            good_row, cfg=cfg, available_cash=1000, live_price=153.0,
        )
        assert p.entry == 153.0
        # Stop was 142 (5.33% below 150) → 153 × (1 - 0.05333) ≈ 144.84
        assert p.stop == pytest.approx(144.84, abs=0.05)
        # Target was 170 (13.33% above 150) → 153 × 1.13333 ≈ 173.40
        assert p.target == pytest.approx(173.40, abs=0.05)
        assert p.slippage_warning is None

    def test_slippage_above_5pct_rejects(self, good_row, cfg):
        # Live price 6% above scan → over the 5% reject threshold
        p = compute_execution_preview(
            good_row, cfg=cfg, available_cash=1000, live_price=159.0,
        )
        assert p.slippage_warning is not None
        assert "REJECT" in p.slippage_warning

    def test_throttle_warn_halves_qty(self, good_row, cfg):
        normal = compute_execution_preview(
            good_row, cfg=cfg, available_cash=1000, throttle_mult=1.0,
        )
        warn = compute_execution_preview(
            good_row, cfg=cfg, available_cash=1000, throttle_mult=0.5,
        )
        # WARN should produce strictly less spend (qty may round to same int
        # for tiny positions, but spend always shrinks)
        assert warn.spend_estimate < normal.spend_estimate

    def test_throttle_halt_zero_qty(self, good_row, cfg):
        halt = compute_execution_preview(
            good_row, cfg=cfg, available_cash=1000, throttle_mult=0.0,
        )
        assert halt.qty_estimate == 0
        assert halt.spend_estimate == 0.0

    def test_qty_fallback_for_expensive_stock(self, good_row, cfg):
        """A $400 stock with $1000 cash should buy 1 share even though
        the conviction-weighted target spend is < $400."""
        good_row["Entry_Price"] = 400.0
        good_row["Stop_Loss"] = 380.0
        good_row["Target_Price"] = 440.0
        p = compute_execution_preview(good_row, cfg=cfg, available_cash=1000)
        assert p.qty_estimate >= 1, "should fall back to 1 share if affordable"

    def test_qty_zero_when_unaffordable(self, good_row, cfg):
        """A $400 stock with $200 cash → 0 shares (can't afford)."""
        good_row["Entry_Price"] = 400.0
        good_row["Stop_Loss"] = 380.0
        good_row["Target_Price"] = 440.0
        p = compute_execution_preview(good_row, cfg=cfg, available_cash=200)
        assert p.qty_estimate == 0


# ────────────────────────────────────────────────────────────────────
# normalize_ticker_for_ib — share-class tickers
# ────────────────────────────────────────────────────────────────────


class TestNormalizeTicker:
    def test_brk_b_becomes_space(self):
        assert normalize_ticker_for_ib("BRK.B") == "BRK B"

    def test_bf_b_becomes_space(self):
        assert normalize_ticker_for_ib("BF.B") == "BF B"

    def test_normal_ticker_unchanged(self):
        assert normalize_ticker_for_ib("AAPL") == "AAPL"
        assert normalize_ticker_for_ib("GOOG") == "GOOG"

    def test_lowercase_uppercased(self):
        assert normalize_ticker_for_ib("aapl") == "AAPL"

    def test_idempotent(self):
        # Calling twice gives same result
        once = normalize_ticker_for_ib("BRK.B")
        twice = normalize_ticker_for_ib(once)
        assert once == twice

    def test_empty_string_passes_through(self):
        assert normalize_ticker_for_ib("") == ""


# ────────────────────────────────────────────────────────────────────
# Parity contract — dashboard preview vs production
# ────────────────────────────────────────────────────────────────────


class TestParityContract:
    """If these tests fail, the dashboard preview disagrees with what
    the production trader will do — the EXACT class of bug that
    motivated extracting policy.py in the first place."""

    @pytest.mark.parametrize("regime,expected_floor", [
        ("TREND_UP", 60.0),       # 55 + 5
        ("MODERATE_UP", 65.0),    # 60 + 5
        ("SIDEWAYS", 75.0),       # 70 + 5
        ("DISTRIBUTION", 80.0),   # 75 + 5
        ("CORRECTION", 85.0),     # 80 + 5
        ("PANIC", 105.0),         # 100 + 5
    ])
    def test_score_floor_matches_scoring_config(self, regime, expected_floor, cfg):
        """If REGIME_MIN_SCORE in scoring_config.py is changed, this
        test breaks LOUDLY rather than silently — forcing the developer
        to verify the new floor is intentional and update both paths."""
        assert regime_score_floor(regime, cfg) == pytest.approx(expected_floor)

    def test_evaluator_uses_cfg_for_ml_threshold(self, good_row, cfg):
        """Changing cfg.min_ml_prob should change what passes — proves
        the evaluator reads from CONFIG, not hardcoded constants."""
        cfg.min_ml_prob = 0.50
        good_row["ML_20d_Prob"] = 0.45  # below new threshold
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "ML" in result.primary_reason

    def test_evaluator_uses_cfg_for_blocked_sectors(self, good_row, cfg):
        """Same — sector blocklist reads from cfg, not hardcoded."""
        cfg.blocked_sectors = "Healthcare"  # add Healthcare to blocklist
        good_row["Sector"] = "Healthcare"
        result = evaluate_static_gates(good_row, cfg=cfg)
        assert not result.would_buy
        assert "Healthcare" in result.primary_reason
