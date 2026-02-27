"""
Integration tests for result quality — ensure scoring produces meaningful,
differentiated outputs instead of degenerate all-same results.

These tests run the scoring engine on synthetic rows and verify that:
- Risk labels are differentiated (not all BUY)
- ConvictionScore ≠ TechScore (multi-factor blend)
- ReliabilityScore varies based on actual data quality
- Risk_Label comes from risk_meter, not from Action enum
"""
import numpy as np
import pandas as pd
import pytest

from core.scoring_engine import (
    calculate_conviction_score,
    calculate_reliability_score,
    calculate_risk_meter,
)
from core.risk_engine import RiskEngine
from core.interfaces import Action, TickerFeatures, ModelOutput, DataQuality
from datetime import datetime


# ---------------------------------------------------------------------------
# Risk Labels differentiation
# ---------------------------------------------------------------------------

class TestRiskLabelDifferentiation:
    """Risk_Label must vary based on actual risk factors, not always be 'BUY'."""

    def test_risk_meter_labels_vary(self):
        """Different risk inputs should produce different labels."""
        # Low risk case
        _, label_low = calculate_risk_meter(
            rr_ratio=3.0, beta=0.8, atr_pct=0.015, leverage=0.3
        )
        # High risk case
        _, label_high = calculate_risk_meter(
            rr_ratio=0.7, beta=1.9, atr_pct=0.09, leverage=3.5
        )
        assert label_low != label_high, (
            f"Same label for very different risk profiles: {label_low}"
        )

    def test_risk_meter_not_always_moderate(self):
        """With no data, default is MODERATE - but with data it should differ."""
        # Extreme risk inputs
        score, label = calculate_risk_meter(
            rr_ratio=0.5, beta=2.0, atr_pct=0.10, leverage=4.0
        )
        assert label != "MODERATE", f"Expected non-MODERATE for extreme risk, got {label}"
        assert score > 70, f"Expected high risk score, got {score}"

    def test_risk_meter_low_risk_stocks(self):
        """Defensive stocks should get LOW or VERY LOW labels."""
        score, label = calculate_risk_meter(
            rr_ratio=4.0, beta=0.5, atr_pct=0.01, leverage=0.1
        )
        assert label in ("LOW", "VERY LOW"), f"Expected LOW/VERY LOW, got {label}"
        assert score < 25, f"Expected low risk score, got {score}"


class TestRiskEngineActions:
    """RiskEngine should produce different actions, not always BUY."""

    def _make_features(self, risk_meta: dict) -> TickerFeatures:
        return TickerFeatures(
            ticker="TEST",
            as_of_date=datetime.utcnow(),
            data_timestamp=datetime.utcnow(),
            source_map={},
            quality=DataQuality.MEDIUM,
            point_in_time_ok=True,
            model_features={},
            risk_metadata=risk_meta,
        )

    def _make_model_output(self, prob: float) -> ModelOutput:
        return ModelOutput(
            prediction_prob=prob,
            expected_return=0.05,
            confidence_score=prob,
            calibration_factor=1.0,
            model_version="test",
            generation_time=0.01,
        )

    def test_high_conviction_gets_buy(self):
        engine = RiskEngine()
        features = self._make_features({"volume_avg": 1_000_000, "atr_pct_raw": 0.02})
        model_out = self._make_model_output(0.80)
        decision = engine.evaluate(features, model_out)
        assert decision.action == Action.BUY
        assert decision.conviction >= 65.0

    def test_moderate_conviction_gets_hold(self):
        engine = RiskEngine()
        features = self._make_features({
            "volume_avg": 1_000_000,
            "atr_pct_raw": 0.06,  # high vol -> -20
            "beta": 1.6,  # elevated beta -> -5
        })
        model_out = self._make_model_output(0.65)  # base = 65
        decision = engine.evaluate(features, model_out)
        # 65 - 20 - 5 = 40 -> below BUY (65), but above or at HOLD threshold
        assert decision.action in (Action.HOLD, Action.REDUCE), (
            f"Expected HOLD or REDUCE for penalized conviction, got {decision.action.value}"
        )

    def test_low_conviction_gets_reduce(self):
        engine = RiskEngine()
        features = self._make_features({
            "volume_avg": 1_000_000,
            "atr_pct_raw": 0.09,  # extreme vol -> -35
            "beta": 2.0,  # high beta -> -10
            "RR": 0.7,  # very poor RR -> -20
        })
        model_out = self._make_model_output(0.55)  # base = 55
        decision = engine.evaluate(features, model_out)
        # 55 - 35 - 10 - 20 < 0 -> REDUCE
        assert decision.action == Action.REDUCE
        assert decision.conviction < 45

    def test_low_volume_rejected(self):
        engine = RiskEngine()
        features = self._make_features({"volume_avg": 50_000})
        model_out = self._make_model_output(0.90)
        decision = engine.evaluate(features, model_out)
        assert decision.action == Action.REJECT

    def test_earnings_risk_rejected(self):
        engine = RiskEngine()
        features = self._make_features({"volume_avg": 1_000_000, "days_to_earnings": 1})
        model_out = self._make_model_output(0.90)
        decision = engine.evaluate(features, model_out)
        assert decision.action == Action.REJECT


# ---------------------------------------------------------------------------
# ConvictionScore multi-factor blend
# ---------------------------------------------------------------------------

class TestConvictionScoreIndependence:
    """ConvictionScore should differ from TechScore — it blends multiple factors."""

    def test_conviction_differs_from_tech_score(self):
        """ConvictionScore blends fund + momentum + RR + reliability; not just momentum."""
        conviction, breakdown = calculate_conviction_score(
            fundamental_score=80.0,       # high
            fundamental_confidence=90.0,
            momentum_score=50.0,          # mediocre
            momentum_confidence=80.0,
            rr_score=70.0,
            rr_confidence=80.0,
            reliability_score=85.0,
            ml_probability=0.6,
        )
        # TechScore would be ~50, but conviction should be higher
        # because of good RR and high reliability (fund weight reduced in swing mode)
        assert conviction > 55, f"Expected conviction > 55 (not just tech=50), got {conviction}"

    def test_conviction_varies_with_fundamentals(self):
        """Two stocks with same tech but different fundamentals → different conviction."""
        conv_good_fund, _ = calculate_conviction_score(
            fundamental_score=90.0, fundamental_confidence=90.0,
            momentum_score=60.0, momentum_confidence=80.0,
            rr_score=60.0, rr_confidence=80.0,
            reliability_score=70.0, ml_probability=0.5,
        )
        conv_bad_fund, _ = calculate_conviction_score(
            fundamental_score=20.0, fundamental_confidence=90.0,
            momentum_score=60.0, momentum_confidence=80.0,
            rr_score=60.0, rr_confidence=80.0,
            reliability_score=70.0, ml_probability=0.5,
        )
        assert conv_good_fund > conv_bad_fund + 5, (
            f"Good fundamentals ({conv_good_fund:.1f}) should clearly beat "
            f"bad fundamentals ({conv_bad_fund:.1f})"
        )

    def test_conviction_varies_with_reliability(self):
        """Two stocks with same tech/fund but different reliability → different conviction."""
        conv_high_rel, _ = calculate_conviction_score(
            fundamental_score=60.0, fundamental_confidence=90.0,
            momentum_score=60.0, momentum_confidence=80.0,
            rr_score=60.0, rr_confidence=80.0,
            reliability_score=95.0, ml_probability=0.5,
        )
        conv_low_rel, _ = calculate_conviction_score(
            fundamental_score=60.0, fundamental_confidence=90.0,
            momentum_score=60.0, momentum_confidence=80.0,
            rr_score=60.0, rr_confidence=80.0,
            reliability_score=20.0, ml_probability=0.5,
        )
        assert conv_high_rel > conv_low_rel, (
            f"High reliability ({conv_high_rel:.1f}) should beat low ({conv_low_rel:.1f})"
        )


# ---------------------------------------------------------------------------
# ReliabilityScore data quality
# ---------------------------------------------------------------------------

class TestReliabilityScoreVariation:
    """ReliabilityScore should vary meaningfully based on data quality."""

    def test_full_data_high_reliability(self):
        """Stock with many sources and low variance → high reliability."""
        score = calculate_reliability_score(
            price_sources=3,
            fund_sources=3,
            price_std=0.5,
            price_mean=100.0,
            fundamental_confidence=90.0,
            data_completeness=95.0,
        )
        assert score > 75, f"Expected high reliability for full data, got {score}"

    def test_sparse_data_low_reliability(self):
        """Stock with no sources and incomplete data → low reliability."""
        score = calculate_reliability_score(
            price_sources=0,
            fund_sources=0,
            price_std=None,
            price_mean=None,
            fundamental_confidence=20.0,
            data_completeness=30.0,
        )
        assert score < 40, f"Expected low reliability for sparse data, got {score}"

    def test_reliability_varies_with_data(self):
        """Different data quality inputs must produce different scores."""
        score_good = calculate_reliability_score(
            price_sources=3, fund_sources=3, price_std=0.3, price_mean=100.0,
            fundamental_confidence=90.0, data_completeness=95.0,
        )
        score_bad = calculate_reliability_score(
            price_sources=0, fund_sources=0, price_std=5.0, price_mean=100.0,
            fundamental_confidence=10.0, data_completeness=20.0,
        )
        assert abs(score_good - score_bad) > 20, (
            f"Expected large gap between good ({score_good:.1f}) and bad ({score_bad:.1f}) data quality"
        )

    def test_reliability_not_fixed_80(self):
        """Reliability should NOT always return 80."""
        scores = set()
        test_cases = [
            {"price_sources": 0, "fund_sources": 0, "data_completeness": 10.0, "fundamental_confidence": 10.0},
            {"price_sources": 1, "fund_sources": 1, "data_completeness": 50.0, "fundamental_confidence": 50.0},
            {"price_sources": 3, "fund_sources": 3, "data_completeness": 90.0, "fundamental_confidence": 90.0},
        ]
        for case in test_cases:
            s = calculate_reliability_score(**case)
            scores.add(round(s))
        assert len(scores) >= 2, (
            f"Expected different reliability scores, got all: {scores}"
        )


# ---------------------------------------------------------------------------
# Scoring config sanity
# ---------------------------------------------------------------------------

class TestScoringConfigSanity:
    """Verify scoring weights sum correctly and ML is not over-weighted."""

    def test_final_score_weights_sum_to_one(self):
        from core.scoring_config import FINAL_SCORE_WEIGHTS
        total = sum(FINAL_SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"FINAL_SCORE_WEIGHTS sum={total}, expected 1.0"

    def test_ml_weight_reasonable(self):
        from core.scoring_config import FINAL_SCORE_WEIGHTS
        ml_weight = FINAL_SCORE_WEIGHTS.get("ml", 0.0)
        assert ml_weight <= 0.20, (
            f"ML weight {ml_weight} too high for AUC=0.553 model. Should be ≤0.20"
        )

    def test_ml_gates_bonus_not_too_easy(self):
        """ML bonus threshold should be above typical model output range (0.46-0.74)."""
        from core.scoring_config import ML_GATES
        bonus_gt = ML_GATES.get("bonus_gt", 0.62)
        assert bonus_gt >= 0.70, (
            f"bonus_gt={bonus_gt} is too low — most stocks would get a bonus"
        )
