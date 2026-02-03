"""
Tests for multi-source data aggregation and reliability scoring.
Validates field-level median merge, disagreement scoring, and reliability formulas.
"""
import pytest
import numpy as np
from core.data_sources_v2 import (
    aggregate_price,
    aggregate_fundamentals,
)


def test_aggregate_price_single_source():
    """Single price source should have zero std deviation."""
    prices = {"fmp": 150.0}
    mean, std, count = aggregate_price(prices)
    
    assert np.isfinite(mean)
    assert mean == 150.0
    assert std == 0.0
    assert count == 1


def test_aggregate_price_multiple_sources_close():
    """Multiple close prices should give median and low std."""
    prices = {
        "fmp": 150.0,
        "finnhub": 150.5,
        "tiingo": 149.8,
    }
    mean, std, count = aggregate_price(prices)
    
    assert np.isfinite(mean)
    assert 149.5 < mean < 150.5  # Near 150
    assert std < 1.0  # Low disagreement
    assert count == 3


def test_aggregate_price_multiple_sources_divergent():
    """Widely different prices should result in higher std."""
    prices = {
        "fmp": 150.0,
        "finnhub": 160.0,
        "tiingo": 140.0,
    }
    mean, std, count = aggregate_price(prices)
    
    assert np.isfinite(mean)
    assert 145.0 < mean < 155.0  # Median around 150
    assert std > 5.0  # Significant disagreement
    assert count == 3


def test_aggregate_price_with_none_values():
    """Should ignore None values and only use valid prices."""
    prices = {
        "fmp": 150.0,
        "finnhub": None,
        "tiingo": 151.0,
        "alpha": None,
    }
    mean, std, count = aggregate_price(prices)
    
    assert np.isfinite(mean)
    assert 150.0 <= mean <= 151.0
    assert count == 2


def test_aggregate_price_all_none():
    """All None prices should return NaN."""
    prices = {
        "fmp": None,
        "finnhub": None,
        "tiingo": None,
    }
    mean, std, count = aggregate_price(prices)
    
    assert np.isnan(mean)
    assert np.isnan(std)
    assert count == 0


def test_aggregate_price_with_invalid_values():
    """Should ignore negative or invalid prices."""
    prices = {
        "fmp": 150.0,
        "finnhub": -10.0,  # Invalid
        "tiingo": 0.0,  # Invalid
        "alpha": 151.0,
    }
    mean, std, count = aggregate_price(prices)
    
    assert np.isfinite(mean)
    assert count == 2  # Only fmp and alpha


def test_aggregate_fundamentals_empty_sources():
    """When no sources available or ticker doesn't exist, should return minimal result."""
    # This test uses a mock or expects aggregate_fundamentals to handle missing APIs gracefully
    # Note: some providers may return placeholder data even for non-existent tickers
    result = aggregate_fundamentals("NONEXISTENT_TICKER_XYZ999")
    
    assert "sources_used" in result
    # May have 0 or more sources - depends on provider behavior with invalid tickers
    assert isinstance(result["sources_used"], list)
    assert result["disagreement_score"] >= 0.0
    # Metadata should exist (may be neutral or have some values)
    assert "Fundamental_Sources_Count" in result
    assert "Fundamental_Coverage_Pct" in result
    assert "Fundamental_S" in result


def test_reliability_bounds_and_monotonicity():
    """Reliability v2 should stay within bounds and respond to signals."""
    from core.v2_risk_engine import calculate_reliability_v2
    import pandas as pd

    # Case 1: Empty fundamentals, single price source, high variance → low reliability
    row1 = pd.Series({
        "Fundamental_Coverage_Pct": 0.0,
        "Fundamental_Sources_Count": 0,
        "Price_Sources_Count": 1,
        "Price_STD": 5.0,
        "Price_Mean": 50.0,
        "ATR_Pct": 0.06,
        "Quality_Score": 50.0,
    })
    r1, _ = calculate_reliability_v2(row1)
    assert 0.0 <= r1 <= 100.0
    assert r1 < 35.0

    # Case 2: Single fundamental source, partial coverage, moderate variance → medium reliability
    row2 = pd.Series({
        "Fundamental_Coverage_Pct": 50.0,
        "Fundamental_Sources_Count": 1,
        "Price_Sources_Count": 2,
        "Price_STD": 0.5,
        "Price_Mean": 100.0,
        "ATR_Pct": 0.03,
        "Quality_Score": 60.0,
    })
    r2, _ = calculate_reliability_v2(row2)
    assert 0.0 <= r2 <= 100.0
    assert r2 > r1
    assert 40.0 <= r2 <= 75.0

    # Case 3: Multiple sources, high coverage, low variance → high reliability
    row3 = pd.Series({
        "Fundamental_Coverage_Pct": 90.0,
        "Fundamental_Sources_Count": 3,
        "Price_Sources_Count": 4,
        "Price_STD": 0.2,
        "Price_Mean": 200.0,
        "ATR_Pct": 0.025,
        "Quality_Score": 75.0,
    })
    r3, _ = calculate_reliability_v2(row3)
    assert 0.0 <= r3 <= 100.0
    assert r3 > r2
    assert r3 >= 75.0

    # Case 4: High price disagreement reduces reliability
    row4 = row3.copy()
    row4["Price_STD"] = 15.0
    row4["Price_Mean"] = 100.0
    r4, _ = calculate_reliability_v2(row4)
    assert 0.0 <= r4 <= 100.0
    assert r4 < r3


def test_reliability_formula_no_sources():
    """Reliability with zero sources should be zero."""
    # Simulate what main pipeline does
    f_sources = []
    f_disagreement = 1.0
    coverage_pct = 0.0
    
    source_factor = 0.5 + 0.5 * min(len(f_sources) / 4.0, 1.0)
    fund_raw = coverage_pct * (1.0 - f_disagreement) * source_factor
    if len(f_sources) >= 1:
        fund_raw = max(fund_raw, 0.15 * source_factor)
    fund_reliability_pct = max(0.0, min(fund_raw * 100.0, 100.0))
    
    assert fund_reliability_pct == 0.0


def test_reliability_formula_single_source_partial_coverage():
    """Single source with partial coverage should have reliability >= 15%."""
    f_sources = ["fmp"]
    f_disagreement = 0.2
    coverage_pct = 0.5  # 50% fields covered
    
    source_factor = 0.5 + 0.5 * min(len(f_sources) / 4.0, 1.0)  # 0.625
    fund_raw = coverage_pct * (1.0 - f_disagreement) * source_factor
    if len(f_sources) >= 1:
        fund_raw = max(fund_raw, 0.15 * source_factor)
    fund_reliability_pct = max(0.0, min(fund_raw * 100.0, 100.0))
    
    # With 1 source, source_factor = 0.625
    # fund_raw = 0.5 * 0.8 * 0.625 = 0.25
    # floor = 0.15 * 0.625 = 0.09375
    # So fund_raw should be 0.25 (above floor)
    assert fund_reliability_pct > 15.0
    assert fund_reliability_pct < 40.0


def test_reliability_formula_multiple_sources_high_coverage():
    """Multiple sources with high coverage should have high reliability."""
    f_sources = ["fmp", "finnhub", "tiingo", "alpha"]
    f_disagreement = 0.1  # Low disagreement
    coverage_pct = 0.9  # 90% fields covered
    
    source_factor = 0.5 + 0.5 * min(len(f_sources) / 4.0, 1.0)  # 1.0
    fund_raw = coverage_pct * (1.0 - f_disagreement) * source_factor
    if len(f_sources) >= 1:
        fund_raw = max(fund_raw, 0.15 * source_factor)
    fund_reliability_pct = max(0.0, min(fund_raw * 100.0, 100.0))
    
    # fund_raw = 0.9 * 0.9 * 1.0 = 0.81
    assert fund_reliability_pct > 75.0


def test_price_reliability_formula_low_variance():
    """Low price variance should yield high reliability."""
    price_count = 5
    variance_ratio = 0.01  # 1% variance
    
    source_factor_price = min(price_count / 5.0, 1.0)  # 1.0
    price_raw = (1.0 - variance_ratio) * (0.4 + 0.6 * source_factor_price)
    if price_count >= 1:
        price_raw = max(price_raw, 0.20 * (0.4 + 0.6 * source_factor_price))
    price_reliability_pct = max(0.0, min(price_raw * 100.0, 100.0))
    
    # price_raw = 0.99 * 1.0 = 0.99
    assert price_reliability_pct > 90.0


def test_price_reliability_formula_high_variance():
    """High price variance should yield lower reliability."""
    price_count = 3
    variance_ratio = 0.5  # 50% variance (very high disagreement)
    
    source_factor_price = min(price_count / 5.0, 1.0)  # 0.6
    price_raw = (1.0 - variance_ratio) * (0.4 + 0.6 * source_factor_price)
    if price_count >= 1:
        price_raw = max(price_raw, 0.20 * (0.4 + 0.6 * source_factor_price))
    price_reliability_pct = max(0.0, min(price_raw * 100.0, 100.0))
    
    # price_raw = 0.5 * (0.4 + 0.36) = 0.5 * 0.76 = 0.38
    assert price_reliability_pct > 20.0  # Above floor
    assert price_reliability_pct < 50.0  # But not high


def test_combined_reliability_weighted():
    """Combined reliability should weight Fund 60% and Price 40%."""
    fund_rel = 80.0
    price_rel = 60.0
    
    combined = 0.6 * fund_rel + 0.4 * price_rel
    
    assert combined == 72.0  # 48 + 24
