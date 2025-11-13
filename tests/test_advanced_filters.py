import pandas as pd
import numpy as np
from advanced_filters import (
    compute_relative_strength,
    detect_volume_surge,
    detect_consolidation,
    check_ma_alignment,
    find_support_resistance,
    compute_momentum_quality,
    should_reject_ticker
)


def test_relative_strength_positive():
    """Test RS calculation when stock outperforms benchmark"""
    ticker_df = pd.DataFrame({
        "Close": [100, 105, 110, 115, 120]
    })
    bench_df = pd.DataFrame({
        "Close": [100, 101, 102, 103, 104]
    })
    
    rs = compute_relative_strength(ticker_df, bench_df, periods=[4])
    assert "rs_4d" in rs
    assert rs["rs_4d"] > 0  # Stock should be outperforming


def test_volume_surge_detection():
    """Test volume surge detection"""
    df = pd.DataFrame({
        "Volume": [1000] * 20 + [3000] * 5,
        "Close": list(range(100, 125))
    })
    
    result = detect_volume_surge(df, lookback=20)
    assert result["volume_surge"] >= 2.0  # Should detect significant surge
    assert "pv_correlation" in result


def test_consolidation_squeeze():
    """Test volatility squeeze detection"""
    # Create data with narrowing range
    df = pd.DataFrame({
        "High": [102, 101.5, 101.2, 101.1, 101.0] + [100.8] * 50,
        "Low": [98, 98.5, 98.8, 98.9, 99.0] + [99.2] * 50,
        "Close": [100] * 55
    })
    
    squeeze = detect_consolidation(df, short_period=10, long_period=50)
    assert squeeze <= 1.0  # Should indicate tightening or stable
    assert np.isfinite(squeeze)


def test_ma_alignment_bullish():
    """Test MA alignment for bullish setup"""
    # Create uptrending data
    close_prices = list(range(100, 300))
    df = pd.DataFrame({"Close": close_prices})
    
    result = check_ma_alignment(df, periods=[10, 20, 50, 100])
    assert result["ma_aligned"] is True
    assert result["alignment_score"] == 1.0


def test_ma_alignment_bearish():
    """Test MA alignment for bearish setup"""
    # Create downtrending data
    close_prices = list(range(300, 100, -1))
    df = pd.DataFrame({"Close": close_prices})
    
    result = check_ma_alignment(df, periods=[10, 20, 50, 100])
    assert result["ma_aligned"] is False


def test_support_resistance_levels():
    """Test support/resistance calculation"""
    # Create data with clear levels
    df = pd.DataFrame({
        "High": [105] * 10 + [110] * 10 + [115] * 10,
        "Low": [95] * 10 + [100] * 10 + [105] * 10,
        "Close": [100] * 10 + [105] * 10 + [110] * 10
    })
    
    result = find_support_resistance(df, window=5)
    assert "distance_to_support" in result
    assert "distance_to_resistance" in result
    assert np.isfinite(result["distance_to_support"])


def test_momentum_consistency():
    """Test momentum quality assessment"""
    # Create consistent uptrend
    df = pd.DataFrame({
        "Close": [100 + i * 0.5 for i in range(100)]
    })
    
    result = compute_momentum_quality(df)
    assert result["momentum_consistency"] > 0.7  # High consistency
    assert 0 <= result["momentum_consistency"] <= 1.0


def test_rejection_criteria_underperforming():
    """Test rejection for underperforming stocks"""
    signals = {
        "rs_63d": -0.15,  # Underperforming by 15%
        "momentum_consistency": 0.5,
        "risk_reward_ratio": 2.0,
        "alignment_score": 0.8
    }
    
    should_reject, reason = should_reject_ticker(signals)
    assert should_reject is True
    assert "Underperforming" in reason


def test_rejection_criteria_weak_momentum():
    """Test rejection for weak momentum"""
    signals = {
        "rs_63d": 0.05,
        "momentum_consistency": 0.2,  # Very weak
        "risk_reward_ratio": 2.0,
        "alignment_score": 0.8
    }
    
    should_reject, reason = should_reject_ticker(signals)
    assert should_reject is True
    assert "momentum" in reason.lower()


def test_rejection_criteria_poor_rr():
    """Test rejection for poor risk/reward"""
    signals = {
        "rs_63d": 0.05,
        "momentum_consistency": 0.7,
        "risk_reward_ratio": 0.5,  # Poor R/R
        "alignment_score": 0.8
    }
    
    should_reject, reason = should_reject_ticker(signals)
    assert should_reject is True
    assert "Risk/Reward" in reason


def test_acceptance_criteria_good_setup():
    """Test that good setups pass all filters"""
    signals = {
        "rs_63d": 0.08,  # Outperforming
        "momentum_consistency": 0.75,  # Strong momentum
        "risk_reward_ratio": 2.5,  # Good R/R
        "alignment_score": 0.9  # Bullish alignment
    }
    
    should_reject, reason = should_reject_ticker(signals)
    assert should_reject is False
    assert reason == ""


def test_volume_surge_with_insufficient_data():
    """Test volume surge with insufficient data"""
    df = pd.DataFrame({
        "Volume": [1000] * 10,
        "Close": list(range(100, 110))
    })
    
    result = detect_volume_surge(df, lookback=20)
    assert result["volume_surge"] >= 0.0
    assert result["pv_correlation"] == 0.0


def test_consolidation_with_insufficient_data():
    """Test consolidation detection with insufficient data"""
    df = pd.DataFrame({
        "High": [101, 102, 103],
        "Low": [99, 98, 97],
        "Close": [100, 100, 100]
    })
    
    squeeze = detect_consolidation(df, short_period=20, long_period=50)
    assert np.isnan(squeeze)
