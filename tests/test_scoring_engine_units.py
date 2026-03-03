"""Unit tests for core/scoring_engine.py pure functions.

Covers: ml_boost_component, normalize_score, safe_divide, evaluate_rr_unified,
        calculate_reliability_score, calculate_conviction_score,
        calculate_risk_meter, generate_warnings.
"""
import math
import numpy as np
import pandas as pd
import pytest

from core.scoring_engine import (
    ml_boost_component,
    normalize_score,
    safe_divide,
    evaluate_rr_unified,
    calculate_reliability_score,
    calculate_conviction_score,
    calculate_risk_meter,
    generate_warnings,
)


# ── ml_boost_component ──────────────────────────────────────────────


class TestMlBoostComponent:
    def test_neutral_returns_zero(self):
        assert ml_boost_component(0.5) == 0.0

    def test_max_returns_plus_six(self):
        assert ml_boost_component(1.0) == pytest.approx(6.0)

    def test_min_returns_minus_six(self):
        assert ml_boost_component(0.0) == pytest.approx(-6.0)

    def test_above_neutral(self):
        assert 0 < ml_boost_component(0.75) <= 6

    def test_below_neutral(self):
        assert -6 <= ml_boost_component(0.25) < 0

    def test_none_returns_zero(self):
        assert ml_boost_component(None) == 0.0

    def test_nan_returns_zero(self):
        assert ml_boost_component(float("nan")) == 0.0

    def test_inf_returns_zero(self):
        assert ml_boost_component(float("inf")) == 0.0

    def test_clipped_above_one(self):
        # prob > 1 should be clipped to 1.0 → +6
        assert ml_boost_component(1.5) == pytest.approx(6.0)

    def test_clipped_below_zero(self):
        assert ml_boost_component(-0.5) == pytest.approx(-6.0)


# ── normalize_score ─────────────────────────────────────────────────


class TestNormalizeScore:
    def test_identity_on_0_100(self):
        assert normalize_score(50.0) == pytest.approx(50.0)

    def test_zero_to_one_range(self):
        assert normalize_score(0.5, 0, 1.0) == pytest.approx(50.0)
        assert normalize_score(0.0, 0, 1.0) == pytest.approx(0.0)
        assert normalize_score(1.0, 0, 1.0) == pytest.approx(100.0)

    def test_clipping_above(self):
        assert normalize_score(150.0) == 100.0

    def test_clipping_below(self):
        assert normalize_score(-50.0) == 0.0

    def test_nan_returns_default(self):
        assert normalize_score(float("nan")) == 50.0

    def test_inf_returns_default(self):
        assert normalize_score(float("inf")) == 50.0

    def test_equal_min_max_returns_default(self):
        assert normalize_score(5.0, 5.0, 5.0) == 50.0

    def test_custom_default(self):
        assert normalize_score(float("nan"), default=75.0) == 75.0


# ── safe_divide ─────────────────────────────────────────────────────


class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10, 2) == pytest.approx(5.0)

    def test_zero_denominator(self):
        assert safe_divide(10, 0) == 0.0

    def test_near_zero_denominator(self):
        assert safe_divide(10, 1e-12) == 0.0

    def test_nan_numerator(self):
        assert safe_divide(float("nan"), 5) == 0.0

    def test_nan_denominator(self):
        assert safe_divide(5, float("nan")) == 0.0

    def test_inf_numerator(self):
        assert safe_divide(float("inf"), 5) == 0.0

    def test_custom_default(self):
        assert safe_divide(10, 0, default=-1.0) == -1.0

    def test_negative_division(self):
        assert safe_divide(-10, 2) == pytest.approx(-5.0)


# ── evaluate_rr_unified ────────────────────────────────────────────


class TestEvaluateRrUnified:
    def test_none_returns_zero(self):
        score, ratio, band = evaluate_rr_unified(None)
        assert score == 0.0
        assert ratio == 0.0
        assert band == "N/A"

    def test_nan_returns_zero(self):
        score, ratio, band = evaluate_rr_unified(float("nan"))
        assert band == "N/A"

    def test_negative_returns_zero(self):
        score, ratio, band = evaluate_rr_unified(-1.0)
        assert band == "N/A"

    def test_very_poor_band(self):
        score, ratio, band = evaluate_rr_unified(0.5)
        assert band == "Very Poor"
        assert 0 <= score <= 20

    def test_poor_band(self):
        score, ratio, band = evaluate_rr_unified(1.2)
        assert band == "Poor"
        assert 20 <= score <= 40

    def test_fair_band(self):
        score, ratio, band = evaluate_rr_unified(1.7)
        assert band == "Fair"
        assert 40 <= score <= 70

    def test_good_band(self):
        score, ratio, band = evaluate_rr_unified(2.5)
        assert band == "Good"
        assert 70 <= score <= 90

    def test_excellent_band(self):
        score, ratio, band = evaluate_rr_unified(4.0)
        assert band == "Excellent"
        assert 90 <= score <= 100

    def test_score_monotonic(self):
        """Higher RR → higher (or equal) score."""
        ratios = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        scores = [evaluate_rr_unified(r)[0] for r in ratios]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], f"Non-monotonic at {ratios[i]}"

    def test_score_clamped_0_100(self):
        score, _, _ = evaluate_rr_unified(100.0)
        assert 0 <= score <= 100

    def test_boundary_1_0(self):
        score, ratio, band = evaluate_rr_unified(1.0)
        assert band == "Poor"
        assert score >= 20

    def test_boundary_3_0(self):
        score, ratio, band = evaluate_rr_unified(3.0)
        assert band == "Excellent"
        assert score >= 90


# ── calculate_reliability_score ─────────────────────────────────────


class TestCalculateReliabilityScore:
    def test_all_defaults(self):
        score = calculate_reliability_score()
        # Only source count component → weight re-normalized to 1.0
        assert 0 <= score <= 100

    def test_perfect_inputs(self):
        score = calculate_reliability_score(
            price_sources=3,
            fund_sources=3,
            price_std=0.1,
            price_mean=100.0,
            fundamental_confidence=100.0,
            data_completeness=100.0,
        )
        assert score >= 90

    def test_zero_completeness(self):
        score = calculate_reliability_score(
            data_completeness=0.0, price_sources=0, fund_sources=0
        )
        assert score < 30

    def test_high_variance_lowers_score(self):
        low_var = calculate_reliability_score(
            price_std=0.5, price_mean=100.0, data_completeness=80.0
        )
        high_var = calculate_reliability_score(
            price_std=5.0, price_mean=100.0, data_completeness=80.0
        )
        assert low_var > high_var


# ── calculate_conviction_score ──────────────────────────────────────


class TestCalculateConvictionScore:
    def test_all_high(self):
        score, bd = calculate_conviction_score(
            fundamental_score=90,
            fundamental_confidence=90,
            momentum_score=90,
            momentum_confidence=90,
            rr_score=90,
            rr_confidence=90,
            reliability_score=90,
        )
        assert score >= 80

    def test_all_low(self):
        score, bd = calculate_conviction_score(
            fundamental_score=10,
            fundamental_confidence=10,
            momentum_score=10,
            momentum_confidence=10,
            rr_score=10,
            rr_confidence=10,
            reliability_score=10,
        )
        assert score <= 30

    def test_breakdown_keys(self):
        _, bd = calculate_conviction_score(50, 50, 50, 50, 50, 50, 50)
        expected = {
            "fundamental_component",
            "momentum_component",
            "rr_component",
            "reliability_component",
            "ml_adjustment",
            "base_conviction",
            "final_conviction",
        }
        assert expected.issubset(bd.keys())

    def test_ml_raises_score(self):
        lo, _ = calculate_conviction_score(60, 80, 60, 80, 60, 80, 80, ml_probability=None)
        hi, _ = calculate_conviction_score(60, 80, 60, 80, 60, 80, 80, ml_probability=0.9)
        assert hi > lo

    def test_ml_lowers_score(self):
        base, _ = calculate_conviction_score(60, 80, 60, 80, 60, 80, 80, ml_probability=None)
        bad, _ = calculate_conviction_score(60, 80, 60, 80, 60, 80, 80, ml_probability=0.1)
        assert bad < base

    def test_score_clamped_0_100(self):
        score, _ = calculate_conviction_score(100, 100, 100, 100, 100, 100, 100, ml_probability=1.0)
        assert 0 <= score <= 100
        score2, _ = calculate_conviction_score(0, 0, 0, 0, 0, 0, 0, ml_probability=0.0)
        assert 0 <= score2 <= 100


# ── calculate_risk_meter ────────────────────────────────────────────


class TestCalculateRiskMeter:
    def test_all_none_returns_moderate(self):
        score, label = calculate_risk_meter(None, None, None, None)
        assert score == 50.0
        assert label == "MODERATE"

    def test_very_low_risk(self):
        score, label = calculate_risk_meter(4.0, 0.5, 0.01, 0.1)
        assert score < 30
        assert label in ("LOW", "VERY LOW")

    def test_very_high_risk(self):
        score, label = calculate_risk_meter(0.5, 2.0, 0.10, 5.0)
        assert score > 70
        assert label in ("HIGH", "VERY HIGH")

    def test_rr_only(self):
        score, label = calculate_risk_meter(0.5, None, None, None)
        assert score >= 70  # Low RR → high risk

    def test_label_coherence(self):
        """Higher risk score → more severe label (or same)."""
        _, label_low = calculate_risk_meter(4.0, 0.5, 0.01, 0.1)
        _, label_high = calculate_risk_meter(0.5, 2.0, 0.10, 5.0)
        severity = ["VERY LOW", "LOW", "MODERATE", "MODERATE-HIGH", "HIGH", "VERY HIGH"]
        assert severity.index(label_high) >= severity.index(label_low)


# ── generate_warnings ──────────────────────────────────────────────


class TestGenerateWarnings:
    def test_no_warnings_clean(self):
        warns = generate_warnings(
            rr_ratio=3.0,
            fundamental_confidence=80.0,
            beta=1.0,
            atr_pct=0.02,
            reliability_score=80.0,
        )
        assert warns == []

    def test_low_rr_warning(self):
        warns = generate_warnings(1.0, 80, 1.0, 0.02, 80)
        assert any("Risk/Reward" in w for w in warns)

    def test_low_fund_confidence(self):
        warns = generate_warnings(3.0, 20.0, 1.0, 0.02, 80)
        assert any("fundamental" in w.lower() for w in warns)

    def test_high_beta(self):
        warns = generate_warnings(3.0, 80, 2.0, 0.02, 80)
        assert any("beta" in w.lower() for w in warns)

    def test_high_volatility(self):
        warns = generate_warnings(3.0, 80, 1.0, 0.08, 80)
        assert any("volatility" in w.lower() or "ATR" in w for w in warns)

    def test_low_reliability(self):
        warns = generate_warnings(3.0, 80, 1.0, 0.02, 30)
        assert any("reliability" in w.lower() for w in warns)

    def test_all_warnings(self):
        warns = generate_warnings(0.5, 10, 3.0, 0.10, 20)
        assert len(warns) == 5

    def test_none_rr_no_crash(self):
        warns = generate_warnings(None, 80, None, None, 80)
        assert isinstance(warns, list)


# ── compute_final_score_20d: entry timing ─────────────────────────

from core.scoring_engine import compute_final_score_20d


def _make_row(**overrides):
    """Build a pd.Series with reasonable defaults for scoring tests."""
    defaults = {
        "Fundamental_S": 70.0,
        "TechScore_20d_raw": 70.0,
        "Reliability_Score": 80.0,
        "RR": 2.5,
        "ML_20d_Prob": 0.55,
        "Volatility_Contraction_Score": 0.0,
        "Tightness_Ratio": 1.0,
        "Coil_Bonus": 0,
        "Pattern_Score": 0.0,
        "Big_Winner_Signal": 0.0,
        "Dist_52w_High": -0.10,   # 10% below 52w high (neutral)
        "Return_20d": 0.05,       # 5% return (neutral)
    }
    defaults.update(overrides)
    return pd.Series(defaults)


class TestEntryTimingAdjustment:
    """Tests for the entry timing penalty/bonus in compute_final_score_20d."""

    def test_near_ath_no_vcp_penalty(self):
        """Stock within 3% of 52w high WITHOUT VCP → penalty applied."""
        neutral = compute_final_score_20d(_make_row(Dist_52w_High=-0.10))
        near_ath = compute_final_score_20d(_make_row(Dist_52w_High=-0.02, Volatility_Contraction_Score=0.1))
        assert near_ath < neutral, "Near-ATH without VCP should score lower"

    def test_near_ath_with_vcp_no_penalty(self):
        """Stock within 3% of 52w high WITH VCP → no penalty."""
        with_vcp = compute_final_score_20d(_make_row(Dist_52w_High=-0.02, Volatility_Contraction_Score=0.5))
        neutral = compute_final_score_20d(_make_row(Dist_52w_High=-0.10, Volatility_Contraction_Score=0.5))
        # With VCP, near-ATH should NOT be penalized (may differ by other factors, but not by 8pts)
        assert abs(with_vcp - neutral) < 6, "Near-ATH with VCP should not get large penalty"

    def test_pullback_bonus(self):
        """Stock 5-15% below 52w high → pullback bonus applied."""
        pullback = compute_final_score_20d(_make_row(Dist_52w_High=-0.08))
        far_away = compute_final_score_20d(_make_row(Dist_52w_High=-0.25))
        assert pullback > far_away, "Pullback zone (5-15% from high) should score higher"

    def test_rapid_runup_penalty(self):
        """Stock with >20% return in 20d → penalty for late entry."""
        normal = compute_final_score_20d(_make_row(Return_20d=0.05))
        runup = compute_final_score_20d(_make_row(Return_20d=0.25))
        assert runup < normal, "Rapid run-up (>20%) should score lower"

    def test_missing_dist_high_no_crash(self):
        """Missing Dist_52w_High should not crash."""
        score = compute_final_score_20d(_make_row(Dist_52w_High=np.nan))
        assert 0 <= score <= 100

    def test_near_ath_penalty_magnitude(self):
        """Penalty should be approximately 8 points for near-ATH."""
        # Use -0.20 (outside pullback zone) as neutral baseline
        neutral = compute_final_score_20d(_make_row(Dist_52w_High=-0.20))
        near_ath = compute_final_score_20d(_make_row(Dist_52w_High=-0.02, Volatility_Contraction_Score=0.1))
        diff = neutral - near_ath
        assert 5 <= diff <= 12, f"Expected ~8pt penalty, got {diff:.1f}"

    def test_score_always_valid_range(self):
        """Score stays in [0, 100] even with stacked penalties."""
        score = compute_final_score_20d(_make_row(
            Dist_52w_High=-0.01,
            Return_20d=0.30,
            Volatility_Contraction_Score=0.0,
            TechScore_20d_raw=10.0,
            Fundamental_S=10.0,
            RR=0.3,
        ))
        assert 0 <= score <= 100


# ── Mediocre RR gate in scoring/final.py ───────────────────────────

from core.scoring.final import compute_final_score_with_patterns


class TestMediocreRRGate:
    """Tests for the new mediocre RR gate (0.95x for RR 1.0-1.5)."""

    def test_mediocre_rr_penalized(self):
        """RR 1.0-1.5 should produce lower score than RR 2.0+."""
        mediocre, _ = compute_final_score_with_patterns(
            tech_score=70, fundamental_score=70, ml_prob=0.55, rr_ratio=1.2,
        )
        good, _ = compute_final_score_with_patterns(
            tech_score=70, fundamental_score=70, ml_prob=0.55, rr_ratio=2.0,
        )
        assert mediocre < good, "Mediocre RR (1.2) should score lower than good RR (2.0)"

    def test_rr_gate_ordering(self):
        """Scores must be monotonically increasing with RR."""
        rr_values = [0.3, 0.8, 1.2, 2.0, 3.0, 5.0]
        scores = []
        for rr in rr_values:
            s, _ = compute_final_score_with_patterns(
                tech_score=70, fundamental_score=70, ml_prob=0.55, rr_ratio=rr,
            )
            scores.append(s)
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], (
                f"Non-monotonic at RR={rr_values[i]}: {scores[i]:.1f} < {scores[i-1]:.1f}"
            )
