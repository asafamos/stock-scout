"""
Test overall score computation with explicit component weights and penalties.

Validates:
1. Score range: 0-100 bounds
2. Component weights: 35% fund + 35% tech + 15% RR + 15% reliability
3. ML delta: bounded to ±10%
4. Penalties: RR, risk, reliability, missing data
5. Spread: 30+ point difference between high/low quality opportunities
"""
import pytest
import numpy as np
import pandas as pd
from core.scoring_engine import compute_overall_score, calculate_quality_score


def test_overall_score_bounds():
    """Overall score must be between 0 and 100"""
    # Worst case: all scores at 0
    row_worst = pd.Series({
        'Fundamental_S': 0.0,
        'Technical_S': 0.0,
        'RR_Score': 0.0,
        'Reliability_v2': 0.0,
        'ML_Probability': 0.0,
        'RR': 0.5,  # Poor RR
        'RiskMeter': 100.0,  # High risk
        'PE_f': np.nan,
        'ROE_f': np.nan,
        'GM_f': np.nan,
        'DE_f': np.nan,
        'RevG_f': np.nan,
    })
    
    score, components = compute_overall_score(row_worst)
    assert 0 <= score <= 100, f"Score {score} out of bounds"
    assert score < 30, f"Worst case score {score} should be below 30"
    
    # Best case: all scores at 100
    row_best = pd.Series({
        'Fundamental_S': 100.0,
        'Technical_S': 100.0,
        'RR_Score': 100.0,
        'Reliability_v2': 100.0,
        'ML_Probability': 1.0,
        'RR': 3.5,  # Excellent RR
        'RiskMeter': 20.0,  # Low risk
        'PE_f': 12.0,
        'ROE_f': 0.20,
        'GM_f': 0.35,
        'DE_f': 0.3,
        'RevG_f': 0.25,
    })
    
    score, components = compute_overall_score(row_best)
    assert 0 <= score <= 100, f"Score {score} out of bounds"
    assert score > 70, f"Best case score {score} should be above 70"


def test_component_weights():
    """Verify 35/35/15/15 weight formula"""
    row = pd.Series({
        'Fundamental_S': 80.0,
        'Technical_S': 60.0,
        'RR_Score': 70.0,
        'Reliability_v2': 90.0,
        'ML_Probability': None,  # No ML to isolate weights
        'RR': 2.5,  # Good RR, no penalty
        'RiskMeter': 50.0,  # Medium risk, no penalty
        'PE_f': 15.0,
        'ROE_f': 0.15,
        'GM_f': 0.25,
        'DE_f': 0.5,
        'RevG_f': 0.15,
    })
    
    score, components = compute_overall_score(row)
    
    # Check individual components
    fund_comp = components['fund_component']
    tech_comp = components['tech_component']
    rr_comp = components['rr_component']
    rel_comp = components['reliability_component']
    
    # Verify weights (allow 0.1 tolerance for floating point)
    expected_fund = 80.0 * 0.35
    expected_tech = 60.0 * 0.35
    expected_rr = 70.0 * 0.15
    expected_rel = 90.0 * 0.15
    
    assert abs(fund_comp - expected_fund) < 0.1, f"Fund component {fund_comp} != {expected_fund}"
    assert abs(tech_comp - expected_tech) < 0.1, f"Tech component {tech_comp} != {expected_tech}"
    assert abs(rr_comp - expected_rr) < 0.1, f"RR component {rr_comp} != {expected_rr}"
    assert abs(rel_comp - expected_rel) < 0.1, f"Rel component {rel_comp} != {expected_rel}"
    
    # Base score should be sum of components (before penalties)
    expected_base = expected_fund + expected_tech + expected_rr + expected_rel
    assert abs(components['base_score'] - expected_base) < 0.1, \
        f"Base score {components['base_score']} != {expected_base}"


def test_ml_delta_bounded():
    """ML adjustment must be bounded to ±10%"""
    base_row = pd.Series({
        'Fundamental_S': 70.0,
        'Technical_S': 70.0,
        'RR_Score': 70.0,
        'Reliability_v2': 70.0,
        'RR': 2.0,
        'RiskMeter': 50.0,
        'PE_f': 15.0,
        'ROE_f': 0.15,
        'GM_f': 0.25,
        'DE_f': 0.5,
        'RevG_f': 0.15,
    })
    
    # Test ML prob = 0.0 (should give -10% penalty)
    row_low = base_row.copy()
    row_low['ML_Probability'] = 0.0
    score_low, components_low = compute_overall_score(row_low)
    assert -10.5 <= components_low['ml_delta'] <= -9.5, \
        f"ML delta {components_low['ml_delta']} should be ~-10"
    
    # Test ML prob = 1.0 (should give +10% boost)
    row_high = base_row.copy()
    row_high['ML_Probability'] = 1.0
    score_high, components_high = compute_overall_score(row_high)
    assert 9.5 <= components_high['ml_delta'] <= 10.5, \
        f"ML delta {components_high['ml_delta']} should be ~+10"
    
    # Test ML prob = 0.5 (should give 0 adjustment)
    row_neutral = base_row.copy()
    row_neutral['ML_Probability'] = 0.5
    score_neutral, components_neutral = compute_overall_score(row_neutral)
    assert abs(components_neutral['ml_delta']) < 0.1, \
        f"ML delta {components_neutral['ml_delta']} should be ~0 at prob=0.5"


def test_penalties_applied():
    """Verify penalties reduce score"""
    # Base row with no penalties
    row_clean = pd.Series({
        'Fundamental_S': 70.0,
        'Technical_S': 70.0,
        'RR_Score': 70.0,
        'Reliability_v2': 80.0,
        'ML_Probability': 0.5,
        'RR': 2.5,  # Good RR
        'RiskMeter': 50.0,  # Medium risk
        'PE_f': 15.0,
        'ROE_f': 0.15,
        'GM_f': 0.25,
        'DE_f': 0.5,
        'RevG_f': 0.15,
    })
    
    score_clean, components_clean = compute_overall_score(row_clean)
    
    # Row with poor RR (should have penalty)
    row_poor_rr = row_clean.copy()
    row_poor_rr['RR'] = 0.8
    score_poor_rr, components_poor_rr = compute_overall_score(row_poor_rr)
    
    assert score_poor_rr < score_clean, "Poor RR should reduce score"
    assert components_poor_rr['penalty_total'] > 0, "Penalty should be applied for RR < 1.0"
    
    # Row with high risk (should have penalty)
    row_high_risk = row_clean.copy()
    row_high_risk['RiskMeter'] = 85.0
    score_high_risk, components_high_risk = compute_overall_score(row_high_risk)
    
    assert score_high_risk < score_clean, "High risk should reduce score"
    assert components_high_risk['penalty_total'] > 0, "Penalty should be applied for RiskMeter > 65"
    
    # Row with low reliability (should have penalty)
    row_low_rel = row_clean.copy()
    row_low_rel['Reliability_v2'] = 50.0
    score_low_rel, components_low_rel = compute_overall_score(row_low_rel)
    
    assert score_low_rel < score_clean, "Low reliability should reduce score"
    assert components_low_rel['penalty_total'] > 0, "Penalty should be applied for reliability < 75"
    
    # Row with missing data (should have penalty)
    row_missing = row_clean.copy()
    row_missing['PE_f'] = np.nan
    row_missing['ROE_f'] = np.nan
    row_missing['GM_f'] = np.nan
    score_missing, components_missing = compute_overall_score(row_missing)
    
    assert score_missing < score_clean, "Missing data should reduce score"
    assert components_missing['penalty_total'] > 0, "Penalty should be applied for missing data"


def test_score_spread_30_points():
    """Verify 30+ point spread between high and low quality opportunities"""
    # Core opportunity: high quality, good fundamentals
    row_core = pd.Series({
        'Fundamental_S': 85.0,
        'Technical_S': 80.0,
        'RR_Score': 90.0,
        'Reliability_v2': 90.0,
        'ML_Probability': 0.85,
        'RR': 3.0,
        'RiskMeter': 30.0,
        'PE_f': 12.0,
        'ROE_f': 0.25,
        'GM_f': 0.40,
        'DE_f': 0.2,
        'RevG_f': 0.30,
    })
    
    # Problematic opportunity: low quality, poor fundamentals
    row_problematic = pd.Series({
        'Fundamental_S': 35.0,
        'Technical_S': 40.0,
        'RR_Score': 30.0,
        'Reliability_v2': 50.0,
        'ML_Probability': 0.45,
        'RR': 0.9,  # Poor RR
        'RiskMeter': 80.0,  # High risk
        'PE_f': np.nan,
        'ROE_f': np.nan,
        'GM_f': np.nan,
        'DE_f': 3.0,  # High debt
        'RevG_f': -0.05,  # Negative growth
    })
    
    score_core, _ = compute_overall_score(row_core)
    score_problematic, _ = compute_overall_score(row_problematic)
    
    spread = score_core - score_problematic
    assert spread >= 30, f"Score spread {spread:.1f} should be >= 30 points"
    
    # Verify score separation (core should be significantly higher)
    assert score_core > 75, f"Core score {score_core:.1f} should be > 75"
    assert score_problematic < 50, f"Problematic score {score_problematic:.1f} should be < 50"
    
    # Verify the spread is meaningful
    assert score_core / score_problematic > 2.0, "Core score should be 2x+ problematic score"


def test_quality_score_levels():
    """Verify quality score 3-level system"""
    # High quality
    row_high = pd.Series({
        'ROE_f': 0.25,  # 25%
        'GM_f': 0.40,  # 40%
        'ProfitMargin': 0.15,  # 15%
        'RevG_f': 0.30,  # 30%
        'EPSG_f': 0.35,  # 35%
        'DE_f': 0.3,  # Low debt
    })
    
    score_high, level_high = calculate_quality_score(row_high)
    assert 0 <= score_high <= 1, "Quality score must be 0-1"
    assert level_high == "High", f"Expected 'High' but got '{level_high}'"
    assert score_high >= 0.7, f"High quality score {score_high:.2f} should be >= 0.7"
    
    # Medium quality
    row_medium = pd.Series({
        'ROE_f': 0.12,  # 12%
        'GM_f': 0.20,  # 20%
        'ProfitMargin': 0.05,  # 5%
        'RevG_f': 0.10,  # 10%
        'EPSG_f': 0.08,  # 8%
        'DE_f': 1.0,  # Moderate debt
    })
    
    score_medium, level_medium = calculate_quality_score(row_medium)
    assert level_medium == "Medium", f"Expected 'Medium' but got '{level_medium}'"
    assert 0.4 <= score_medium < 0.7, f"Medium quality score {score_medium:.2f} should be 0.4-0.69"
    
    # Low quality
    row_low = pd.Series({
        'ROE_f': 0.03,  # 3%
        'GM_f': 0.08,  # 8%
        'ProfitMargin': -0.02,  # -2%
        'RevG_f': -0.05,  # -5%
        'EPSG_f': -0.10,  # -10%
        'DE_f': 2.5,  # High debt
    })
    
    score_low, level_low = calculate_quality_score(row_low)
    assert level_low == "Low", f"Expected 'Low' but got '{level_low}'"
    assert score_low < 0.4, f"Low quality score {score_low:.2f} should be < 0.4"


def test_quality_score_with_missing_data():
    """Quality score should handle missing data gracefully"""
    row_empty = pd.Series({
        'ROE_f': np.nan,
        'GM_f': np.nan,
        'ProfitMargin': np.nan,
        'RevG_f': np.nan,
        'EPSG_f': np.nan,
        'DE_f': np.nan,
    })
    
    score, level = calculate_quality_score(row_empty)
    assert 0 <= score <= 1, "Quality score must be 0-1 even with missing data"
    assert level == "Medium", f"Expected 'Medium' for missing data but got '{level}'"
    assert 0.4 <= score <= 0.6, f"Missing data should yield neutral score, got {score:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
