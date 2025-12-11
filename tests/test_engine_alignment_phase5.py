"""
Phase 5: Integration and Consistency Tests for Engine Alignment

Tests that verify all three execution modes produce identical results.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.config import get_config
from core.unified_logic import (
    build_technical_indicators,
    apply_technical_filters,
    compute_technical_score,
    fetch_stock_data,
)
from core.scoring_engine import evaluate_rr_unified, compute_overall_score
from core.debug_utils import validate_ticker_consistency, validate_consistency_batch


class TestEngineAlignment:
    """Test suite validating that core functions are single sources of truth."""
    
    def test_build_indicators_deterministic(self):
        """Test that build_technical_indicators produces consistent results."""
        # Fetch data once
        df = fetch_stock_data('AAPL', '2024-09-01', '2024-12-31')
        assert df is not None and not df.empty, "Failed to fetch test data"
        
        # Build indicators twice
        ind1 = build_technical_indicators(df)
        ind2 = build_technical_indicators(df)
        
        # Compare results
        assert ind1.shape == ind2.shape, "Indicator shape mismatch"
        
        # Check key columns are present
        expected_cols = ['RSI', 'ATR', 'MA20', 'MA50', 'Overext', 'RR', 'MomCons']
        for col in expected_cols:
            assert col in ind1.columns, f"Missing column: {col}"
        
        # Check consistency (last row should be identical)
        row1 = ind1.iloc[-1]
        row2 = ind2.iloc[-1]
        for col in expected_cols:
            v1 = float(row1.get(col, np.nan))
            v2 = float(row2.get(col, np.nan))
            if np.isfinite(v1) and np.isfinite(v2):
                assert abs(v1 - v2) < 1e-6, f"Value mismatch for {col}: {v1} vs {v2}"
    
    def test_technical_score_reproducible(self):
        """Test that compute_technical_score produces consistent results."""
        config = get_config()
        df = fetch_stock_data('MSFT', '2024-10-01', '2024-12-31')
        assert df is not None and not df.empty
        
        tech_ind = build_technical_indicators(df)
        row = tech_ind.iloc[-1]
        weights = dict(config.weights) if hasattr(config.weights, 'items') else config.weights
        
        # Compute score twice
        score1 = compute_technical_score(row, weights=weights)
        score2 = compute_technical_score(row, weights=weights)
        
        # Should be identical
        assert abs(score1 - score2) < 1e-6, f"Score mismatch: {score1} vs {score2}"
        assert 0 <= score1 <= 100, f"Score {score1} out of range"
    
    def test_filters_deterministic(self):
        """Test that apply_technical_filters produces consistent results."""
        df = fetch_stock_data('GOOGL', '2024-10-01', '2024-12-31')
        assert df is not None and not df.empty
        
        tech_ind = build_technical_indicators(df)
        row = tech_ind.iloc[-1]
        
        # Apply filters multiple times
        result1_core = apply_technical_filters(row, strict=True)
        result1_spec = apply_technical_filters(row, strict=False)
        result1_relax = apply_technical_filters(row, strict=False, relaxed=True)
        
        result2_core = apply_technical_filters(row, strict=True)
        result2_spec = apply_technical_filters(row, strict=False)
        result2_relax = apply_technical_filters(row, strict=False, relaxed=True)
        
        # Should be identical
        assert result1_core == result2_core, "Core filter inconsistency"
        assert result1_spec == result2_spec, "Speculative filter inconsistency"
        assert result1_relax == result2_relax, "Relaxed filter inconsistency"
    
    def test_rr_scoring_unified(self):
        """Test that evaluate_rr_unified is consistent."""
        test_ratios = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, np.nan, None]
        
        for ratio in test_ratios:
            score1, adj_ratio1, band1 = evaluate_rr_unified(ratio)
            score2, adj_ratio2, band2 = evaluate_rr_unified(ratio)
            
            # Should be identical
            assert score1 == score2, f"RR score mismatch for ratio {ratio}"
            assert adj_ratio1 == adj_ratio2, f"RR ratio mismatch for input {ratio}"
            assert band1 == band2, f"RR band mismatch for ratio {ratio}"
    
    def test_consistency_checker_valid(self):
        """Test that consistency checker validates correctly."""
        # Use a recent date with real data
        result = validate_ticker_consistency(
            ticker='AAPL',
            event_date=datetime(2024, 12, 15),
            pre_days=20,
            verbose=False
        )
        
        # Should have all required fields
        assert 'is_consistent' in result
        assert 'technical_score' in result
        assert 'passes_filters' in result
        assert 'validation_status' in result
        
        # If successful, score should be in valid range
        if result['validation_status'] in ['PASS', 'WARN']:
            assert 0 <= result['technical_score'] <= 100
    
    def test_consistency_batch_multiple_tickers(self):
        """Test batch consistency checking."""
        test_cases = [
            {'ticker': 'AAPL', 'event_date': pd.Timestamp('2024-12-10')},
            {'ticker': 'MSFT', 'event_date': pd.Timestamp('2024-12-10')},
            {'ticker': 'GOOGL', 'event_date': pd.Timestamp('2024-12-10')},
        ]
        
        batch_result = validate_consistency_batch(test_cases, verbose=False)
        
        assert batch_result['total_cases'] == 3
        assert batch_result['consistency_rate'] >= 0.0
        assert batch_result['consistency_rate'] <= 100.0
        assert len(batch_result['results']) == 3
    
    def test_overall_score_deterministic(self):
        """Test that compute_overall_score is deterministic."""
        test_row = pd.Series({
            'Fundamental_S': 70.0,
            'Technical_S': 75.0,
            'RR_Score': 65.0,
            'Reliability_v2': 80.0,
            'ML_Probability': 0.60,
            'RR': 2.0,
            'RiskMeter': 50.0,
        })
        
        score1, breakdown1 = compute_overall_score(test_row)
        score2, breakdown2 = compute_overall_score(test_row)
        
        # Should be identical
        assert score1 == score2, f"Overall score mismatch: {score1} vs {score2}"
        assert 0 <= score1 <= 100, f"Score {score1} out of range"


class TestDataConsistency:
    """Test data format and consistency across pipelines."""
    
    def test_indicators_column_names(self):
        """Verify expected indicator columns are present."""
        df = fetch_stock_data('SPY', '2024-11-01', '2024-12-31')
        assert df is not None and not df.empty
        
        ind = build_technical_indicators(df)
        
        # Check expected columns
        required_cols = {
            'Close', 'High', 'Low', 'Volume',  # Price data
            'MA20', 'MA50', 'MA200',  # Moving averages
            'RSI', 'ATR', 'ATR_Pct',  # Volatility
            'Overext', 'Near52w',  # Price position
            'MomCons', 'VolSurge', 'RR',  # Momentum/RR
        }
        
        for col in required_cols:
            assert col in ind.columns, f"Missing required column: {col}"
    
    def test_indicators_value_ranges(self):
        """Verify indicator values are in expected ranges."""
        df = fetch_stock_data('NVDA', '2024-11-01', '2024-12-31')
        assert df is not None and not df.empty
        
        ind = build_technical_indicators(df)
        row = ind.iloc[-1]
        
        # RSI should be 0-100
        rsi = row.get('RSI')
        if pd.notna(rsi):
            assert 0 <= rsi <= 100, f"RSI {rsi} out of range [0, 100]"
        
        # Overext should be >= 0
        overext = row.get('Overext')
        if pd.notna(overext):
            assert overext >= 0, f"Overext {overext} should be non-negative"
        
        # MomCons should be 0-1
        momc = row.get('MomCons')
        if pd.notna(momc):
            assert 0 <= momc <= 1, f"MomCons {momc} out of range [0, 1]"


class TestConfigConsistency:
    """Test that all entry points use the same configuration."""
    
    def test_get_config_returns_consistent_object(self):
        """Verify get_config returns same config values repeatedly."""
        config1 = get_config()
        config2 = get_config()
        
        # Check key attributes are identical
        assert config1.universe_limit == config2.universe_limit
        assert config1.lookback_days == config2.lookback_days
        assert config1.ma_short == config2.ma_short
        assert config1.ma_long == config2.ma_long
        assert config1.min_price == config2.min_price
    
    def test_config_has_required_attributes(self):
        """Verify config object has all required attributes."""
        config = get_config()
        
        required_attrs = [
            'universe_limit', 'lookback_days', 'ma_short', 'ma_long',
            'rsi_bounds', 'weights', 'min_price', 'min_avg_volume',
            'fundamental_enabled', 'beta_filter_enabled',
        ]
        
        for attr in required_attrs:
            assert hasattr(config, attr), f"Missing config attribute: {attr}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
