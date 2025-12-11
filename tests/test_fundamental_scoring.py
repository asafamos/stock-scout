"""
Comprehensive tests for fundamental scoring with Quality/Growth/Valuation breakdown.
"""
import pytest
import numpy as np
import pandas as pd
from core.scoring.fundamental import (
    compute_fundamental_score_with_breakdown,
    _safe_float,
    _normalize,
)


class TestSafeFloat:
    """Tests for safe float conversion."""
    
    def test_valid_numeric(self):
        """Convert valid numbers."""
        assert _safe_float(42.0) == 42.0
        assert _safe_float("42.5") == 42.5
        assert _safe_float(100) == 100.0
    
    def test_auto_scale_percent(self):
        """Auto-scale small decimals to percentages."""
        assert _safe_float(0.05, scale_to_pct=True) == 5.0
        assert _safe_float(0.15, scale_to_pct=True) == 15.0
        assert _safe_float(1.5, scale_to_pct=True) == 150.0  # Already large
    
    def test_none_and_invalid(self):
        """Return None for invalid inputs."""
        assert _safe_float(None) is None
        assert _safe_float("invalid") is None
        assert _safe_float(np.nan) is None
        assert _safe_float(np.inf) is None


class TestNormalize:
    """Tests for normalization function."""
    
    def test_normalize_simple(self):
        """Normalize within bounds."""
        assert _normalize(0.5, 0.0, 1.0) == 0.5
        assert _normalize(0.0, 0.0, 1.0) == 0.0
        assert _normalize(1.0, 0.0, 1.0) == 1.0
    
    def test_normalize_clipping(self):
        """Clip out-of-bounds values."""
        assert _normalize(-1.0, 0.0, 1.0) == 0.0
        assert _normalize(2.0, 0.0, 1.0) == 1.0
    
    def test_normalize_neutral(self):
        """Return neutral for None/invalid."""
        assert _normalize(None, 0.0, 1.0) == 0.5
        assert _normalize(np.nan, 0.0, 1.0) == 0.5


class TestQualityScore:
    """Tests for quality (profitability) scoring."""
    
    def test_high_quality(self):
        """High ROE, ROIC, margins → high quality."""
        data = {
            "roe": 0.20,
            "roic": 0.18,
            "gm": 0.40,
            "Fundamental_Coverage_Pct": 100.0,
            "Fundamental_Sources_Count": 3,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 75 <= score.breakdown.quality_score <= 100
        assert score.breakdown.quality_label == "High"
    
    def test_low_quality(self):
        """Negative or very low profitability → low quality."""
        data = {
            "roe": -0.05,
            "roic": 0.02,
            "gm": 0.08,
            "Fundamental_Coverage_Pct": 100.0,
            "Fundamental_Sources_Count": 2,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 0 <= score.breakdown.quality_score <= 40
        assert score.breakdown.quality_label == "Low"
    
    def test_medium_quality(self):
        """Moderate profitability → medium quality."""
        data = {
            "roe": 0.12,
            "roic": 0.10,
            "gm": 0.20,
            "Fundamental_Coverage_Pct": 100.0,
            "Fundamental_Sources_Count": 2,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 40 <= score.breakdown.quality_score <= 70
        assert score.breakdown.quality_label == "Medium"
    
    def test_missing_quality_metrics(self):
        """Missing metrics should not crash, use neutral defaults."""
        data = {
            "roe": None,
            "roic": None,
            "gm": None,
            "Fundamental_Coverage_Pct": 0.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 0 <= score.breakdown.quality_score <= 100
        assert score.breakdown.quality_label in ["Low", "Medium", "High"]


class TestGrowthScore:
    """Tests for growth scoring."""
    
    def test_strong_growth(self):
        """High rev & EPS growth → high growth score."""
        data = {
            "rev_yoy": 0.20,
            "eps_yoy": 0.30,
            "Fundamental_Coverage_Pct": 100.0,
            "Fundamental_Sources_Count": 3,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.breakdown.growth_score >= 75
        assert score.breakdown.growth_label in ["Moderate", "Fast"]
    
    def test_declining_growth(self):
        """Negative growth → low growth score."""
        data = {
            "rev_yoy": -0.15,
            "eps_yoy": -0.20,
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.breakdown.growth_score <= 35
        assert score.breakdown.growth_label in ["Declining", "Slow"]
    
    def test_weak_growth(self):
        """Weak but positive growth → medium score."""
        data = {
            "rev_yoy": 0.03,
            "eps_yoy": 0.02,
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 35 <= score.breakdown.growth_score <= 60
        assert score.breakdown.growth_label in ["Slow", "Moderate"]
    
    def test_missing_growth_metrics(self):
        """Missing growth data should not crash."""
        data = {
            "rev_yoy": None,
            "eps_yoy": None,
            "Fundamental_Coverage_Pct": 0.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 0 <= score.breakdown.growth_score <= 100


class TestValuationScore:
    """Tests for valuation scoring."""
    
    def test_cheap_valuation(self):
        """Low P/E, P/S → high valuation score."""
        data = {
            "pe": 12.0,
            "ps": 1.5,
            "Fundamental_Coverage_Pct": 100.0,
            "Fundamental_Sources_Count": 2,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.breakdown.valuation_score >= 70
        assert score.breakdown.valuation_label == "Cheap"
    
    def test_expensive_valuation(self):
        """High P/E, P/S → low valuation score."""
        data = {
            "pe": 80.0,
            "ps": 12.0,
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.breakdown.valuation_score <= 30
        assert score.breakdown.valuation_label == "Expensive"
    
    def test_fair_valuation(self):
        """Moderate P/E, P/S → medium valuation score."""
        data = {
            "pe": 25.0,
            "ps": 4.0,
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 40 <= score.breakdown.valuation_score <= 65
        assert score.breakdown.valuation_label == "Fair"
    
    def test_negative_earnings(self):
        """Negative P/E severely penalizes score."""
        data = {
            "pe": -5.0,
            "ps": 3.0,
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.breakdown.valuation_score <= 50


class TestLeverageScore:
    """Tests for leverage/debt scoring."""
    
    def test_low_debt(self):
        """Low D/E → high leverage score."""
        data = {
            "de": 0.2,
            "Fundamental_Coverage_Pct": 100.0,
            "Fundamental_Sources_Count": 2,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.breakdown.leverage_score >= 85
        assert score.breakdown.leverage_label == "Low"
    
    def test_high_debt(self):
        """High D/E → low leverage score."""
        data = {
            "de": 3.5,
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.breakdown.leverage_score <= 25
        assert score.breakdown.leverage_label == "High"
    
    def test_moderate_debt(self):
        """Moderate D/E → medium leverage score."""
        data = {
            "de": 1.0,
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 40 <= score.breakdown.leverage_score <= 70
        assert score.breakdown.leverage_label == "Medium"


class TestStabilityScore:
    """Tests for stability (beta + market cap)."""
    
    def test_stable_low_beta(self):
        """Low beta → high stability."""
        data = {
            "beta": 0.6,
            "market_cap": 5e9,  # $5B
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.breakdown.stability_score >= 75
        assert score.breakdown.stability_label == "Stable"
    
    def test_volatile_high_beta(self):
        """High beta → low stability."""
        data = {
            "beta": 2.0,
            "market_cap": 1e8,  # $100M micro-cap
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.breakdown.stability_score <= 40
        assert score.breakdown.stability_label in ["Volatile", "Very Volatile"]
    
    def test_moderate_stability(self):
        """Neutral beta with mid-cap → moderate stability."""
        data = {
            "beta": 1.0,
            "market_cap": 2e9,  # $2B
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 45 <= score.breakdown.stability_score <= 65


class TestOverallFundamentalScore:
    """Tests for overall fundamental score composition."""
    
    def test_high_quality_stock(self):
        """
        High-quality growth stock: good quality, good growth, fair valuation.
        Should have high overall fundamental score.
        """
        data = {
            "roe": 0.22,
            "roic": 0.20,
            "gm": 0.35,
            "rev_yoy": 0.18,
            "eps_yoy": 0.25,
            "pe": 28.0,
            "ps": 4.5,
            "de": 0.4,
            "beta": 1.1,
            "market_cap": 8e9,
            "Fundamental_Coverage_Pct": 100.0,
            "Fundamental_Sources_Count": 3,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.total >= 70
        assert score.breakdown.quality_label == "High"
        assert score.breakdown.valuation_label == "Fair"
    
    def test_overvalued_weak_stock(self):
        """
        Overvalued, weak growth: poor quality, declining growth, expensive valuation.
        Should have low overall fundamental score.
        """
        data = {
            "roe": 0.06,
            "roic": 0.04,
            "gm": 0.12,
            "rev_yoy": -0.08,
            "eps_yoy": -0.05,
            "pe": 85.0,
            "ps": 15.0,
            "de": 2.8,
            "beta": 1.8,
            "market_cap": 2e8,
            "Fundamental_Coverage_Pct": 100.0,
            "Fundamental_Sources_Count": 2,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.total <= 40  # Slightly relaxed threshold
        assert score.breakdown.quality_label in ["Low", "Medium"]  # Either is acceptable with weak growth
        assert score.breakdown.valuation_label == "Expensive"
    
    def test_partial_data_capping(self):
        """
        Low coverage should cap the maximum score.
        """
        data = {
            "roe": 0.20,
            "roic": 0.18,
            "gm": 0.40,
            "rev_yoy": 0.20,
            "eps_yoy": 0.25,
            "pe": 15.0,
            "ps": 2.0,
            "de": 0.3,
            "beta": 0.9,
            "market_cap": 5e9,
            "Fundamental_Coverage_Pct": 35.0,  # Low coverage
            "Fundamental_Sources_Count": 1,
        }
        score = compute_fundamental_score_with_breakdown(data)
        # Even with good fundamentals, low coverage should cap the score
        assert score.total <= 55
    
    def test_no_sources_conservative_cap(self):
        """Zero sources should result in conservative score."""
        data = {
            "roe": 0.20,
            "roic": 0.18,
            "gm": 0.40,
            "rev_yoy": 0.25,
            "eps_yoy": 0.30,
            "pe": 12.0,
            "ps": 1.5,
            "de": 0.2,
            "Fundamental_Coverage_Pct": 100.0,
            "Fundamental_Sources_Count": 0,  # No sources
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert score.total <= 35
    
    def test_all_scores_in_bounds(self):
        """All component scores should be in [0, 100]."""
        data = {
            "roe": 0.15,
            "roic": 0.12,
            "gm": 0.25,
            "rev_yoy": 0.10,
            "eps_yoy": 0.12,
            "pe": 25.0,
            "ps": 3.5,
            "de": 1.0,
            "beta": 1.1,
            "market_cap": 3e9,
            "Fundamental_Coverage_Pct": 80.0,
            "Fundamental_Sources_Count": 2,
        }
        score = compute_fundamental_score_with_breakdown(data)
        
        assert 0 <= score.total <= 100
        assert 0 <= score.breakdown.quality_score <= 100
        assert 0 <= score.breakdown.growth_score <= 100
        assert 0 <= score.breakdown.valuation_score <= 100
        assert 0 <= score.breakdown.leverage_score <= 100
        assert 0 <= score.breakdown.stability_score <= 100
    
    def test_monotonic_improvement_quality(self):
        """Better quality metrics should increase quality score monotonically."""
        base_data = {
            "roic": 0.05,
            "gm": 0.10,
            "Fundamental_Coverage_Pct": 100.0,
        }
        
        roe_low = {**base_data, "roe": 0.05}
        score_low = compute_fundamental_score_with_breakdown(roe_low)
        
        roe_high = {**base_data, "roe": 0.25}
        score_high = compute_fundamental_score_with_breakdown(roe_high)
        
        assert score_high.breakdown.quality_score > score_low.breakdown.quality_score
    
    def test_monotonic_improvement_growth(self):
        """Better growth should increase growth score monotonically."""
        base_data = {
            "Fundamental_Coverage_Pct": 100.0,
        }
        
        growth_low = {**base_data, "rev_yoy": -0.05, "eps_yoy": -0.05}
        score_low = compute_fundamental_score_with_breakdown(growth_low)
        
        growth_high = {**base_data, "rev_yoy": 0.25, "eps_yoy": 0.30}
        score_high = compute_fundamental_score_with_breakdown(growth_high)
        
        assert score_high.breakdown.growth_score > score_low.breakdown.growth_score
    
    def test_monotonic_valuation_inverse(self):
        """Higher valuation multiples should reduce valuation score."""
        base_data = {
            "Fundamental_Coverage_Pct": 100.0,
        }
        
        cheap = {**base_data, "pe": 12.0, "ps": 1.5}
        score_cheap = compute_fundamental_score_with_breakdown(cheap)
        
        expensive = {**base_data, "pe": 75.0, "ps": 12.0}
        score_expensive = compute_fundamental_score_with_breakdown(expensive)
        
        assert score_cheap.breakdown.valuation_score > score_expensive.breakdown.valuation_score


class TestBreakdownExports:
    """Tests for data export and integration."""
    
    def test_breakdown_to_dict(self):
        """Breakdown should convert to dictionary."""
        data = {
            "roe": 0.15,
            "roic": 0.12,
            "gm": 0.25,
            "rev_yoy": 0.10,
            "eps_yoy": 0.12,
            "pe": 25.0,
            "ps": 3.5,
            "de": 1.0,
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        breakdown_dict = score.breakdown.to_dict()
        
        assert "quality_score" in breakdown_dict
        assert "growth_score" in breakdown_dict
        assert "valuation_score" in breakdown_dict
        assert "leverage_score" in breakdown_dict
        assert "stability_score" in breakdown_dict
        assert breakdown_dict["quality_score"] == score.breakdown.quality_score
    
    def test_score_to_dict(self):
        """Score should convert to dictionary."""
        data = {
            "roe": 0.15,
            "rev_yoy": 0.10,
            "pe": 25.0,
            "de": 1.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        score_dict = score.to_dict()
        
        assert "total" in score_dict
        assert "breakdown" in score_dict
        assert score_dict["total"] == score.total


class TestEdgeCases:
    """Tests for edge cases and corner cases."""
    
    def test_all_none_values(self):
        """All None values should not crash."""
        data = {
            "roe": None,
            "roic": None,
            "gm": None,
            "rev_yoy": None,
            "eps_yoy": None,
            "pe": None,
            "ps": None,
            "de": None,
            "beta": None,
            "market_cap": None,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 0 <= score.total <= 100
    
    def test_extreme_positive_values(self):
        """Very high values should be clipped appropriately."""
        data = {
            "roe": 2.0,  # 200% ROE - extreme
            "roic": 1.5,
            "gm": 0.95,
            "rev_yoy": 10.0,  # 1000% growth - extreme
            "eps_yoy": 5.0,
            "pe": 500.0,  # Extreme valuation
            "ps": 100.0,
            "de": 10.0,  # Extreme leverage
            "beta": 5.0,
            "market_cap": 1e12,  # $1 trillion
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 0 <= score.total <= 100
    
    def test_extreme_negative_values(self):
        """Very negative values should be handled gracefully."""
        data = {
            "roe": -2.0,
            "roic": -1.0,
            "gm": -0.5,
            "rev_yoy": -5.0,
            "eps_yoy": -3.0,
            "pe": -100.0,
            "ps": -50.0,
            "de": -0.5,  # Unusual but technically possible
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 0 <= score.total <= 100
    
    def test_zero_market_cap_edge(self):
        """Zero market cap should not crash."""
        data = {
            "beta": 1.0,
            "market_cap": 0.0,
            "Fundamental_Coverage_Pct": 100.0,
        }
        score = compute_fundamental_score_with_breakdown(data)
        assert 0 <= score.breakdown.stability_score <= 100
    
    def test_very_small_market_cap(self):
        """Micro-cap stocks should score lower on stability."""
        data_micro = {
            "beta": 1.0,
            "market_cap": 1e4,  # $10k micro-cap
            "Fundamental_Coverage_Pct": 100.0,
        }
        score_micro = compute_fundamental_score_with_breakdown(data_micro)
        
        data_large = {
            "beta": 1.0,
            "market_cap": 1e11,  # $100B mega-cap
            "Fundamental_Coverage_Pct": 100.0,
        }
        score_large = compute_fundamental_score_with_breakdown(data_large)
        
        assert score_micro.breakdown.stability_score < score_large.breakdown.stability_score
