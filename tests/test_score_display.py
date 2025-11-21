"""
Test that the pretty score display doesn't affect ranking order.

The pretty score is only for display (60-90 range).
The raw score is used for all internal logic including sorting.
This test verifies that the relative ranking is preserved.
"""
import pytest
import numpy as np
import pandas as pd


def test_pretty_score_preserves_ranking():
    """Verify that pretty score transformation preserves stock ranking order"""
    # Create sample data with varying raw scores
    raw_scores = np.array([25.5, 45.2, 67.8, 89.1, 12.3, 55.0, 78.4])
    
    df = pd.DataFrame({
        'Ticker': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'overall_score_raw': raw_scores
    })
    
    # Apply the same transformation as in stock_scout.py
    s_min = float(df['overall_score_raw'].min())
    s_max = float(df['overall_score_raw'].max())
    
    if s_max == s_min:
        df['overall_score_pretty'] = 75.0
    else:
        normalized = (df['overall_score_raw'] - s_min) / (s_max - s_min)
        pretty = 60.0 + normalized * 30.0
        df['overall_score_pretty'] = pretty.clip(0, 100)
    
    # Sort by raw scores
    df_sorted_raw = df.sort_values('overall_score_raw', ascending=False).reset_index(drop=True)
    
    # Sort by pretty scores
    df_sorted_pretty = df.sort_values('overall_score_pretty', ascending=False).reset_index(drop=True)
    
    # Rankings should be identical
    assert df_sorted_raw['Ticker'].tolist() == df_sorted_pretty['Ticker'].tolist(), \
        "Ranking order changed after pretty score transformation!"
    
    # Verify pretty scores are in 60-90 range
    assert df['overall_score_pretty'].min() >= 60.0, "Pretty scores below 60"
    assert df['overall_score_pretty'].max() <= 90.0, "Pretty scores above 90"
    
    # Verify the highest raw score gets 90, lowest gets 60
    assert abs(df.loc[df['overall_score_raw'].idxmax(), 'overall_score_pretty'] - 90.0) < 0.01
    assert abs(df.loc[df['overall_score_raw'].idxmin(), 'overall_score_pretty'] - 60.0) < 0.01


def test_pretty_score_with_identical_values():
    """Verify graceful handling when all scores are identical"""
    df = pd.DataFrame({
        'Ticker': ['A', 'B', 'C'],
        'overall_score_raw': [50.0, 50.0, 50.0]
    })
    
    s_min = float(df['overall_score_raw'].min())
    s_max = float(df['overall_score_raw'].max())
    
    if s_max == s_min:
        df['overall_score_pretty'] = 75.0
    else:
        normalized = (df['overall_score_raw'] - s_min) / (s_max - s_min)
        pretty = 60.0 + normalized * 30.0
        df['overall_score_pretty'] = pretty.clip(0, 100)
    
    # All pretty scores should be 75
    assert all(df['overall_score_pretty'] == 75.0), \
        "When all raw scores are identical, pretty score should be 75"


def test_pretty_score_linear_mapping():
    """Verify that the mapping is truly linear"""
    df = pd.DataFrame({
        'overall_score_raw': [10.0, 30.0, 50.0, 70.0, 90.0]
    })
    
    s_min = float(df['overall_score_raw'].min())
    s_max = float(df['overall_score_raw'].max())
    
    normalized = (df['overall_score_raw'] - s_min) / (s_max - s_min)
    df['overall_score_pretty'] = 60.0 + normalized * 30.0
    
    # Check that normalized values are correct
    expected_normalized = [0.0, 0.25, 0.5, 0.75, 1.0]
    actual_normalized = ((df['overall_score_raw'] - s_min) / (s_max - s_min)).tolist()
    
    for exp, act in zip(expected_normalized, actual_normalized):
        assert abs(exp - act) < 0.01, f"Normalization incorrect: {exp} vs {act}"
    
    # Check pretty scores
    expected_pretty = [60.0, 67.5, 75.0, 82.5, 90.0]
    actual_pretty = df['overall_score_pretty'].tolist()
    
    for exp, act in zip(expected_pretty, actual_pretty):
        assert abs(exp - act) < 0.01, f"Pretty score incorrect: {exp} vs {act}"


def test_raw_score_unchanged():
    """Verify that raw score is preserved and not modified"""
    original_raw = np.array([15.0, 25.0, 35.0, 45.0, 55.0])
    
    df = pd.DataFrame({
        'overall_score_raw': original_raw.copy()
    })
    
    # Apply transformation
    s_min = float(df['overall_score_raw'].min())
    s_max = float(df['overall_score_raw'].max())
    normalized = (df['overall_score_raw'] - s_min) / (s_max - s_min)
    df['overall_score_pretty'] = 60.0 + normalized * 30.0
    
    # Raw scores should be unchanged
    assert np.allclose(df['overall_score_raw'].values, original_raw), \
        "Raw scores were modified during transformation!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
