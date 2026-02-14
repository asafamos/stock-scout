"""Unit tests for core/scoring_engine.py pure functions.

Covers: ml_boost_component, normalize_score, safe_divide, evaluate_rr_unified,
        calculate_rr_score, calculate_reliability_score, calculate_conviction_score,
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
    calculate_rr_score,
    calculate_reliability_score,
    calculate_conviction_score,
    calculate_risk_meter,
    generate_warnings,
)


# ── ml_boost_component ──────────────────────────────────────────────


class TestMlBoostComponent:
    def test_neutral_returns_zero(self):
        assert ml_boost_component(0.5) == 0.0

    def test_max_returns_plus_ten(self):
        assert ml_boost_component(1.0) == pytest.approx(10.0)

    def test_min_returns_minus_ten(self):
        assert ml_boost_component(0.0) == pytest.approx(-10.0)

    def test_above_neutral(self):
        assert 0 < ml_boost_component(0.75) <= 10

    def test_below_neutral(self):
        assert -10 <= ml_boost_component(0.25) < 0

    def test_none_returns_zero(self):
        assert ml_boost_component(None) == 0.0

    def test_nan_returns_zero(self):
        assert ml_boost_component(float("nan")) == 0.0

    def test_inf_returns_zero(self):
        assert ml_boost_component(float("inf")) == 0.0

    def test_clipped_above_one(self):
        # prob > 1 should be clipped to 1.0 → +10
        assert ml_boost_component(1.5) == pytest.approx(10.0)

    def test_clipped_below_zero(self):
        assert ml_boost_component(-0.5) == pytest.approx(-10.0)


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


# ── calculate_rr_score ──────────────────────────────────────────────


class TestCalculateRrScore:
    def test_direct_ratio(self):
        score, conf = calculate_rr_score(rr_ratio=2.5)
        assert 70 <= score <= 90
        assert conf == 75.0

    def test_derived_from_levels(self):
        # support=90, price=100, resistance=120 → risk=10, reward=20, RR=2
        score, conf = calculate_rr_score(
            atr=5.0, support=90.0, resistance=120.0, current_price=100.0
        )
        assert score > 0
        assert conf == 100.0

    def test_fallback_neutral(self):
        score, conf = calculate_rr_score()
        assert score == 50.0
        assert conf == 0.0

    def test_nan_ratio_falls_through(self):
        score, conf = calculate_rr_score(rr_ratio=float("nan"))
        assert score == 50.0

    def test_zero_ratio_falls_through(self):
        score, conf = calculate_rr_score(rr_ratio=0.0)
        assert score == 50.0


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
