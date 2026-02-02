"""
Tests for UnifiedScorer - Consolidated Scoring Logic.

These tests ensure that the UnifiedScorer provides consistent
scoring regardless of how it's called.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestScoringResult:
    """Tests for ScoringResult dataclass."""
    
    def test_scoring_result_creation(self):
        """Test creating a ScoringResult."""
        from core.scoring.unified_scorer import ScoringResult
        
        result = ScoringResult(
            final_conviction=75.5,
            technical_score=80.0,
            fundamental_score=70.0,
            ml_probability=0.72,
            ml_boost=5.0,
            ml_status="enabled",
            reliability_pct=85.0,
            data_quality="high",
            breakdown={"test": "value"}
        )
        
        assert result.final_conviction == 75.5
        assert result.technical_score == 80.0
        assert result.fundamental_score == 70.0
        assert result.ml_probability == 0.72
        assert result.ml_boost == 5.0
        assert result.ml_status == "enabled"
        assert result.reliability_pct == 85.0
        assert result.data_quality == "high"
        assert result.breakdown == {"test": "value"}

    def test_scoring_result_to_dict(self):
        """Test ScoringResult conversion to dictionary."""
        from core.scoring.unified_scorer import ScoringResult
        
        result = ScoringResult(
            final_conviction=75.5,
            technical_score=80.0,
            fundamental_score=70.0,
            ml_probability=0.72,
            ml_boost=5.0,
            ml_status="enabled",
            breakdown={"extra_key": "extra_value"}
        )
        
        d = result.to_dict()
        
        assert d["final_conviction"] == 75.5
        assert d["technical_score"] == 80.0
        assert d["fundamental_score"] == 70.0
        assert d["ml_probability"] == 0.72
        assert d["ml_boost"] == 5.0
        assert d["extra_key"] == "extra_value"

    def test_scoring_result_repr(self):
        """Test ScoringResult string representation."""
        from core.scoring.unified_scorer import ScoringResult
        
        result = ScoringResult(
            final_conviction=75.5,
            technical_score=80.0,
            fundamental_score=70.0,
            ml_probability=0.72,
            ml_boost=5.0,
            ml_status="enabled"
        )
        
        repr_str = repr(result)
        assert "final=75.5" in repr_str
        assert "tech=80.0" in repr_str
        assert "fund=70.0" in repr_str
        assert "ml_boost=+5.0" in repr_str


class TestUnifiedScorerInit:
    """Tests for UnifiedScorer initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        scorer = UnifiedScorer()
        
        assert scorer.ml_enabled is True
        assert scorer.ml_max_boost == 10.0
        assert scorer.technical_weight == 0.60
        assert scorer.fundamental_weight == 0.40
        assert scorer.use_v2_scoring is True

    def test_custom_config(self):
        """Test initialization with custom config."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        config = {
            "enable_ml": False,
            "ml_max_boost_pct": 5.0,
            "technical_weight": 0.50,
            "fundamental_weight": 0.50,
            "use_v2_scoring": False,
        }
        scorer = UnifiedScorer(config=config)
        
        assert scorer.ml_enabled is False
        assert scorer.ml_max_boost == 5.0
        assert scorer.technical_weight == 0.50
        assert scorer.fundamental_weight == 0.50
        assert scorer.use_v2_scoring is False

    def test_weight_warning(self, caplog):
        """Test warning when weights don't sum to 1.0."""
        from core.scoring.unified_scorer import UnifiedScorer
        import logging
        
        config = {
            "technical_weight": 0.70,
            "fundamental_weight": 0.50,  # Sum = 1.2
        }
        
        with caplog.at_level(logging.WARNING):
            scorer = UnifiedScorer(config=config)
        
        assert "weights sum to 1.2" in caplog.text


class TestUnifiedScorerScoring:
    """Tests for UnifiedScorer scoring methods."""
    
    @pytest.fixture
    def mock_scoring_deps(self):
        """Mock external scoring dependencies."""
        with patch("core.unified_logic.compute_tech_score_20d_v2") as mock_tech, \
             patch("core.scoring.fundamental.compute_fundamental_score_with_breakdown") as mock_fund, \
             patch("core.ml_integration.integrate_ml_with_conviction") as mock_ml, \
             patch("core.v2_risk_engine.calculate_reliability_v2") as mock_rel:
            
            mock_tech.return_value = 70.0
            
            fund_result = MagicMock()
            fund_result.total = 60.0
            fund_result.breakdown = None
            mock_fund.return_value = fund_result
            
            mock_ml.return_value = (68.0, {"ml_probability": 0.65, "ml_boost": 2.0, "ml_status": "enabled"})
            
            mock_rel.return_value = 80.0
            
            yield {
                "tech": mock_tech,
                "fund": mock_fund,
                "ml": mock_ml,
                "rel": mock_rel
            }

    def test_score_basic(self, mock_scoring_deps):
        """Test basic scoring flow."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        scorer = UnifiedScorer(config={"enable_ml": False})
        
        result = scorer.score(
            ticker_data={"Close": 100.0, "Ticker": "TEST"},
            technical_indicators={"RSI": 55.0},
            fundamental_data={"roe": 0.20}
        )
        
        # With 60% tech (70) + 40% fund (60) = 42 + 24 = 66
        assert result.final_conviction == pytest.approx(66.0, rel=0.01)
        assert result.technical_score == 70.0
        assert result.fundamental_score == 60.0
        assert result.ml_boost == 0.0
        assert result.ml_status == "disabled"

    def test_score_with_ml(self, mock_scoring_deps):
        """Test scoring with ML boost enabled."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        scorer = UnifiedScorer(config={"enable_ml": True})
        
        result = scorer.score(
            ticker_data={"Close": 100.0, "Ticker": "TEST"},
            technical_indicators={"RSI": 55.0},
            fundamental_data={"roe": 0.20}
        )
        
        # Base: 66.0 + ML boost 2.0 = 68.0
        assert result.final_conviction == pytest.approx(68.0, rel=0.01)
        assert result.ml_boost == 2.0
        assert result.ml_status == "enabled"
        assert result.ml_probability == 0.65

    def test_score_clamps_to_range(self, mock_scoring_deps):
        """Test that scores are clamped to 0-100."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        # Mock extreme values
        mock_scoring_deps["tech"].return_value = 110.0  # Over 100
        
        scorer = UnifiedScorer(config={"enable_ml": False})
        
        result = scorer.score(
            ticker_data={"Close": 100.0},
            technical_indicators={},
            fundamental_data={}
        )
        
        # Should be clamped to 100
        assert result.final_conviction <= 100.0

    def test_ml_boost_clamped_to_max(self, mock_scoring_deps):
        """Test that ML boost is clamped to max configured."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        # Mock ML returning huge boost
        mock_scoring_deps["ml"].return_value = (80.0, {"ml_probability": 0.9, "ml_boost": 25.0, "ml_status": "enabled"})
        
        scorer = UnifiedScorer(config={"enable_ml": True, "ml_max_boost_pct": 5.0})
        
        result = scorer.score(
            ticker_data={"Close": 100.0},
            technical_indicators={},
            fundamental_data={}
        )
        
        # ML boost should be clamped to 5.0
        assert result.ml_boost == 5.0

    def test_score_from_row(self, mock_scoring_deps):
        """Test scoring from a DataFrame row."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        scorer = UnifiedScorer(config={"enable_ml": False})
        
        row = pd.Series({
            "Close": 150.0,
            "Volume": 1000000,
            "Ticker": "AAPL",
            "RSI": 55.0,
            "ATR_Pct": 0.025,
            "Return_20d": 0.05,
            "roe": 0.25,
            "pe": 28.0,
        })
        
        result = scorer.score_from_row(row)
        
        assert isinstance(result.final_conviction, float)
        assert 0 <= result.final_conviction <= 100


class TestUnifiedScorerDataQuality:
    """Tests for data quality assessment."""
    
    def test_assess_data_quality_high(self):
        """Test high data quality assessment."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        scorer = UnifiedScorer()
        
        ticker_data = {
            "Close": 100.0,
            "Volume": 1000000,
            "data_sources": ["yfinance", "polygon", "finnhub"]
        }
        fundamental_data = {"Fundamental_Coverage_Pct": 90}
        
        quality = scorer._assess_data_quality(ticker_data, fundamental_data)
        
        assert quality == "high"

    def test_assess_data_quality_low(self):
        """Test low data quality assessment."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        scorer = UnifiedScorer()
        
        ticker_data = {}
        fundamental_data = {}
        
        quality = scorer._assess_data_quality(ticker_data, fundamental_data)
        
        assert quality == "low"

    def test_assess_data_quality_medium(self):
        """Test medium data quality assessment."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        scorer = UnifiedScorer()
        
        ticker_data = {
            "Close": 100.0,
            "data_sources": ["yfinance"]
        }
        fundamental_data = {"Fundamental_Coverage_Pct": 30}
        
        quality = scorer._assess_data_quality(ticker_data, fundamental_data)
        
        assert quality == "medium"


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @patch("core.scoring.unified_scorer.UnifiedScorer")
    def test_score_ticker(self, mock_scorer_class):
        """Test score_ticker convenience function."""
        from core.scoring.unified_scorer import score_ticker, ScoringResult
        
        mock_instance = MagicMock()
        mock_instance.score.return_value = ScoringResult(
            final_conviction=75.0,
            technical_score=80.0,
            fundamental_score=70.0,
            ml_probability=0.7,
            ml_boost=3.0,
            ml_status="enabled"
        )
        mock_scorer_class.return_value = mock_instance
        
        result = score_ticker(
            {"Close": 100.0},
            {"RSI": 50.0},
            {"roe": 0.2}
        )
        
        assert result.final_conviction == 75.0
        mock_instance.score.assert_called_once()

    @patch("core.scoring.unified_scorer.UnifiedScorer")
    def test_score_dataframe(self, mock_scorer_class):
        """Test score_dataframe convenience function."""
        from core.scoring.unified_scorer import score_dataframe, ScoringResult
        
        mock_instance = MagicMock()
        mock_instance.score_from_row.return_value = ScoringResult(
            final_conviction=75.0,
            technical_score=80.0,
            fundamental_score=70.0,
            ml_probability=0.7,
            ml_boost=3.0,
            ml_status="enabled",
            reliability_pct=85.0
        )
        mock_scorer_class.return_value = mock_instance
        
        df = pd.DataFrame({
            "Close": [100.0, 200.0],
            "RSI": [50.0, 60.0],
            "Ticker": ["AAPL", "MSFT"]
        })
        
        result_df = score_dataframe(df, score_column="Score")
        
        assert "Score" in result_df.columns
        assert "Score_Tech" in result_df.columns
        assert "Score_Fund" in result_df.columns
        assert "Score_ML" in result_df.columns
        assert len(result_df) == 2


class TestErrorHandling:
    """Tests for error handling in UnifiedScorer."""
    
    def test_technical_score_error_returns_neutral(self):
        """Test that technical scoring errors return neutral score."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        with patch("core.unified_logic.compute_tech_score_20d_v2") as mock_tech:
            mock_tech.side_effect = Exception("Technical scoring failed")
            
            scorer = UnifiedScorer(config={"enable_ml": False})
            tech_score = scorer._compute_technical_score({}, {})
            
            assert tech_score == 50.0

    def test_fundamental_score_error_returns_neutral(self):
        """Test that fundamental scoring errors return neutral score."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        with patch("core.scoring.fundamental.compute_fundamental_score_with_breakdown") as mock_fund:
            mock_fund.side_effect = Exception("Fundamental scoring failed")
            
            scorer = UnifiedScorer()
            fund_score, breakdown = scorer._compute_fundamental_score({})
            
            assert fund_score == 50.0
            assert breakdown == {}

    def test_ml_error_returns_no_boost(self):
        """Test that ML errors return zero boost."""
        from core.scoring.unified_scorer import UnifiedScorer
        
        with patch("core.ml_integration.integrate_ml_with_conviction") as mock_ml:
            mock_ml.side_effect = Exception("ML failed")
            
            scorer = UnifiedScorer(config={"enable_ml": True})
            prob, boost, status = scorer._apply_ml_boost(70.0, {}, {}, {})
            
            assert prob is None
            assert boost == 0.0
            assert "error" in status


class TestImportExport:
    """Tests for module exports."""
    
    def test_import_from_scoring_module(self):
        """Test importing from core.scoring module."""
        from core.scoring import (
            UnifiedScorer,
            ScoringResult,
            score_ticker,
            score_dataframe,
        )
        
        assert UnifiedScorer is not None
        assert ScoringResult is not None
        assert score_ticker is not None
        assert score_dataframe is not None

    def test_import_from_unified_scorer(self):
        """Test importing from unified_scorer directly."""
        from core.scoring.unified_scorer import (
            UnifiedScorer,
            ScoringResult,
            score_ticker,
            score_dataframe,
        )
        
        assert UnifiedScorer is not None
        assert ScoringResult is not None
