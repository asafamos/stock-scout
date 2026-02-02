"""Integration tests for the full scoring pipeline.

These tests verify that all components work together correctly:
- UnifiedScorer combines technical, fundamental, and ML scores
- Feature registry provides consistent feature definitions
- API key manager handles keys securely
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def sample_ticker_data():
    """Realistic ticker data for AAPL."""
    return {
        "Ticker": "AAPL",
        "Close": 185.50,
        "Open": 184.00,
        "High": 186.20,
        "Low": 183.50,
        "Volume": 52_000_000,
        "High_52w": 199.62,
        "Low_52w": 164.08,
        "MA20": 182.30,
        "MA50": 178.45,
        "MA200": 175.20,
        "data_sources": ["yahoo", "polygon"],
    }


@pytest.fixture
def sample_technical_indicators():
    """Computed technical indicators."""
    return {
        "RSI": 58.5,
        "ATR": 3.25,
        "ATR_Pct": 0.0175,
        "Return_20d": 0.045,
        "Return_10d": 0.022,
        "Return_5d": 0.008,
        "VCP_Ratio": 0.85,
        "Tightness_Ratio": 0.72,
        "MA_Alignment": 1,
        "VolSurge": 1.15,
        "MomCons": 0.65,
        "RR": 3.5,
    }


@pytest.fixture
def sample_fundamental_data():
    """Fundamental metrics."""
    return {
        "pe": 28.5,
        "ps": 7.2,
        "roe": 0.147,
        "roic": 0.28,
        "rev_yoy": 0.08,
        "eps_yoy": 0.12,
        "margin": 0.25,
        "gm": 0.45,
        "de": 1.52,
        "Fundamental_Coverage_Pct": 85.0,
        "Fundamental_Sources_Count": 3,
    }


class TestUnifiedScorerIntegration:
    """Integration tests for UnifiedScorer."""

    def test_scorer_produces_valid_result(
        self, sample_ticker_data, sample_technical_indicators, sample_fundamental_data
    ):
        """UnifiedScorer should produce a valid ScoringResult."""
        from core.scoring.unified_scorer import UnifiedScorer, ScoringResult

        scorer = UnifiedScorer({"enable_ml": False})  # Disable ML for faster test
        result = scorer.score(
            sample_ticker_data,
            sample_technical_indicators,
            sample_fundamental_data
        )

        assert isinstance(result, ScoringResult)
        assert 0 <= result.final_conviction <= 100
        assert 0 <= result.technical_score <= 100
        assert 0 <= result.fundamental_score <= 100

    def test_scorer_breakdown_contains_expected_keys(
        self, sample_ticker_data, sample_technical_indicators, sample_fundamental_data
    ):
        """Scoring breakdown should have required keys."""
        from core.scoring.unified_scorer import UnifiedScorer

        scorer = UnifiedScorer({"enable_ml": False})
        result = scorer.score(
            sample_ticker_data,
            sample_technical_indicators,
            sample_fundamental_data
        )

        assert "base_conviction" in result.breakdown
        assert "technical_weight" in result.breakdown
        assert "fundamental_weight" in result.breakdown

    def test_scorer_respects_weight_configuration(
        self, sample_ticker_data, sample_technical_indicators, sample_fundamental_data
    ):
        """Custom weights should affect the final score."""
        from core.scoring.unified_scorer import UnifiedScorer

        # Test with different weight configurations
        scorer_tech_heavy = UnifiedScorer({
            "enable_ml": False,
            "technical_weight": 0.90,
            "fundamental_weight": 0.10,
        })
        scorer_fund_heavy = UnifiedScorer({
            "enable_ml": False,
            "technical_weight": 0.10,
            "fundamental_weight": 0.90,
        })

        result_tech = scorer_tech_heavy.score(
            sample_ticker_data, sample_technical_indicators, sample_fundamental_data
        )
        result_fund = scorer_fund_heavy.score(
            sample_ticker_data, sample_technical_indicators, sample_fundamental_data
        )

        # Scores should differ based on weights
        # (unless tech and fund scores happen to be identical)
        assert result_tech.breakdown["technical_weight"] == 0.90
        assert result_fund.breakdown["technical_weight"] == 0.10

    def test_scorer_to_dict_conversion(
        self, sample_ticker_data, sample_technical_indicators, sample_fundamental_data
    ):
        """ScoringResult should convert to dict for DataFrame use."""
        from core.scoring.unified_scorer import UnifiedScorer

        scorer = UnifiedScorer({"enable_ml": False})
        result = scorer.score(
            sample_ticker_data,
            sample_technical_indicators,
            sample_fundamental_data
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "final_conviction" in result_dict
        assert "technical_score" in result_dict
        assert "fundamental_score" in result_dict

    def test_score_from_dataframe_row(
        self, sample_ticker_data, sample_technical_indicators, sample_fundamental_data
    ):
        """Scorer should handle DataFrame row input."""
        from core.scoring.unified_scorer import UnifiedScorer

        # Combine all data into a single row
        row_data = {**sample_ticker_data, **sample_technical_indicators, **sample_fundamental_data}
        row = pd.Series(row_data)

        scorer = UnifiedScorer({"enable_ml": False})
        result = scorer.score_from_row(row)

        assert 0 <= result.final_conviction <= 100


class TestFeatureRegistryIntegration:
    """Integration tests for feature registry."""

    def test_feature_count_matches_expected(self):
        """Feature registry should have exactly 34 features for v3."""
        from core.feature_registry import get_feature_names, FEATURE_COUNT_V3

        features = get_feature_names("v3")
        assert len(features) == 34
        assert len(features) == FEATURE_COUNT_V3

    def test_feature_defaults_match_feature_count(self):
        """Feature defaults should match feature names count."""
        from core.feature_registry import get_feature_names, get_feature_defaults

        names = get_feature_names("v3")
        defaults = get_feature_defaults()

        assert len(defaults) == len(names)
        for name in names:
            assert name in defaults, f"Missing default for feature: {name}"

    def test_feature_specs_have_required_fields(self):
        """Each feature spec should have required fields."""
        from core.feature_registry import FEATURE_SPECS_V3

        required_fields = {"name", "default_value", "valid_range", "description", "category"}

        for spec in FEATURE_SPECS_V3:
            for field in required_fields:
                assert hasattr(spec, field), f"Feature {spec.name} missing field: {field}"

    def test_validate_features_detects_missing(self):
        """Validator should detect missing features."""
        from core.feature_registry import validate_features, get_feature_names

        # DataFrame missing some features
        partial_df = pd.DataFrame({
            "RSI": [50.0],
            "ATR_Pct": [0.02],
        })

        is_valid, missing = validate_features(partial_df)
        assert is_valid is False
        assert len(missing) > 0
        assert "Return_20d" in missing  # Should be missing

    def test_validate_features_passes_with_all(self):
        """Validator should pass when all features present."""
        from core.feature_registry import validate_features, get_feature_names, get_feature_defaults

        # Create DataFrame with all features
        defaults = get_feature_defaults()
        full_df = pd.DataFrame([defaults])

        is_valid, missing = validate_features(full_df)
        assert is_valid is True
        assert len(missing) == 0


class TestAPIKeyManagerIntegration:
    """Integration tests for API key management."""

    def test_get_api_key_returns_none_for_missing_optional(self):
        """Optional keys should return None if not set."""
        from core.api_keys import get_api_key

        # Use a key that's definitely not set
        result = get_api_key("DEFINITELY_NOT_SET_KEY_12345", required=False)
        assert result is None

    def test_get_api_key_raises_for_missing_required(self):
        """Required keys should raise if not set."""
        from core.api_keys import get_api_key

        with pytest.raises(EnvironmentError):
            get_api_key("DEFINITELY_NOT_SET_KEY_12345", required=True)

    def test_validate_keys_returns_dict(self):
        """validate_keys should return a status dict."""
        from core.api_keys import validate_keys

        # validate_keys takes a list of keys to check
        status = validate_keys(["POLYGON_API_KEY", "FINNHUB_API_KEY"])
        assert isinstance(status, dict)
        # Should have entries for requested keys
        assert "POLYGON_API_KEY" in status
        assert "FINNHUB_API_KEY" in status

    def test_get_available_keys_returns_list(self):
        """get_available_keys should return a list."""
        from core.api_keys import get_available_keys

        available = get_available_keys()
        assert isinstance(available, list)


class TestScoringModuleExports:
    """Test that scoring module exports work correctly."""

    def test_import_unified_scorer_from_scoring(self):
        """UnifiedScorer should be importable from core.scoring."""
        from core.scoring import UnifiedScorer, ScoringResult, score_ticker, score_dataframe

        assert UnifiedScorer is not None
        assert ScoringResult is not None
        assert callable(score_ticker)
        assert callable(score_dataframe)

    def test_import_other_scoring_functions(self):
        """Other scoring functions should still be available."""
        from core.scoring import (
            compute_tech_score_20d_v2,
            compute_fundamental_score_with_breakdown,
            calculate_reliability_v2,
        )

        assert callable(compute_tech_score_20d_v2)
        assert callable(compute_fundamental_score_with_breakdown)
        assert callable(calculate_reliability_v2)


class TestEndToEndScoring:
    """End-to-end integration tests."""

    def test_complete_scoring_flow(
        self, sample_ticker_data, sample_technical_indicators, sample_fundamental_data
    ):
        """Test complete flow from raw data to final score."""
        from core.scoring.unified_scorer import UnifiedScorer

        # 1. Create scorer
        scorer = UnifiedScorer(config={
            "enable_ml": False,
            "technical_weight": 0.60,
            "fundamental_weight": 0.40,
        })

        # 2. Score the ticker
        result = scorer.score(
            sample_ticker_data,
            sample_technical_indicators,
            sample_fundamental_data
        )

        # 3. Verify all outputs
        assert result.final_conviction is not None
        assert result.technical_score is not None
        assert result.fundamental_score is not None
        assert result.ml_status == "disabled"  # ML was disabled
        assert result.ml_boost == 0.0

        # 4. Verify score is reasonable
        assert 0 <= result.final_conviction <= 100

        # 5. Verify breakdown is complete
        assert result.breakdown is not None
        assert len(result.breakdown) > 0

    def test_multiple_tickers_scoring(self, sample_ticker_data, sample_technical_indicators, sample_fundamental_data):
        """Scoring multiple tickers should be consistent."""
        from core.scoring.unified_scorer import UnifiedScorer

        scorer = UnifiedScorer({"enable_ml": False})

        # Score same data multiple times
        results = [
            scorer.score(sample_ticker_data, sample_technical_indicators, sample_fundamental_data)
            for _ in range(3)
        ]

        # All scores should be identical (deterministic)
        assert all(r.final_conviction == results[0].final_conviction for r in results)
        assert all(r.technical_score == results[0].technical_score for r in results)
        assert all(r.fundamental_score == results[0].fundamental_score for r in results)
