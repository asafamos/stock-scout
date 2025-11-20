import pytest
import numpy as np
import pandas as pd

from core import v2_risk_engine as v2


def test_apply_v2_conviction_mix_rr_ml():
    # base_conviction 80, perfect reliability, full risk, rr=2, ml_prob=0.6
    base_conv = 80.0
    reliability_v2 = 100.0
    risk_penalty = 1.0
    rr_ratio = 2.0
    ml_prob = 0.6

    final, ml_boost, details = v2.apply_v2_conviction_adjustments(
        base_conv, reliability_v2, risk_penalty, rr_ratio, ml_prob, enable_ml_boost=True
    )

    # Expected calculation (manual):
    # reliability_factor = 0.5 + 0.5*(100/100)=1.0 -> after_rel = 80
    # after_risk = 80 * 1 = 80
    # rr_component = clip(2/2,0,1)*100 = 100
    # mixed = 0.85*80 + 0.15*100 = 68 + 15 = 83
    # ml_boost = (0.6 - 0.5) * 20 = 2.0 -> final = 85
    assert pytest.approx(ml_boost, rel=1e-6) == 2.0
    assert pytest.approx(final, rel=1e-6) == 85.0


def test_calculate_risk_gate_block_and_no_fund_sources():
    # Case 1: RR < 1 -> should be blocked
    status, penalty, details = v2.calculate_risk_gate_v2(0.8, 50.0, fund_sources_count=1, quality_score=50.0)
    assert status == 'blocked'
    assert penalty == 0.0

    # Case 2: no fund sources -> immediate block regardless of rr/reliability
    status2, penalty2, details2 = v2.calculate_risk_gate_v2(5.0, 90.0, fund_sources_count=0, quality_score=80.0)
    assert status2 == 'blocked'
    assert penalty2 == 0.0


def test_calculate_position_size_caps_and_minimum():
    base_alloc = 2000.0
    budget = 5000.0
    # risk_gate_penalty 0.6, reliability 80 -> reliability_factor = 0.5+0.5*0.8=0.9
    amount_core, details_core = v2.calculate_position_size_v2(
        base_alloc, 'reduced', 0.6, 'core', 80.0, budget, min_position=50.0
    )
    # Compute expected: 2000*0.6=1200 -> *0.9 =1080 -> core cap = 15%*5000 = 750 -> amount = 750
    assert pytest.approx(amount_core, rel=1e-6) == 750.0

    # Speculative cap: max 3% -> 150
    amount_spec, details_spec = v2.calculate_position_size_v2(
        base_alloc, 'reduced', 0.6, 'speculative', 80.0, budget, min_position=50.0
    )
    assert amount_spec <= 150.0
    # If amount_spec > 0 enforce minimum
    if amount_spec > 0:
        assert amount_spec >= 50.0


def test_score_ticker_v2_enhanced_end_to_end():
    # Build a row containing many fundamental fields so reliability is high
    row = pd.Series({
        'PE_f': 10.0,
        'PS_f': 2.0,
        'PBRatio': 1.0,
        'ROE_f': 10.0,
        'ROIC_f': 5.0,
        'GM_f': 0.25,
        'ProfitMargin': 0.1,
        'DE_f': 0.5,
        'RevG_f': 0.05,
        'EPSG_f': 0.02,
        'RevenueGrowthYoY': 0.05,
        'EPSGrowthYoY': 0.02,
        'Fundamental_S': 80.0,
        'Quality_Score_F': 40.0,
        'Score_Tech': 60.0,
        'RewardRisk': 3.0,
        'Fund_from_FMP': True,
        'Fund_from_Alpha': False,
        'Fund_from_Finnhub': False,
        'Fund_from_SimFin': False,
        'Fund_from_EODHD': False,
        'Price_Sources_Count': 3,
        'Price_STD': 0.5,
        'Price_Mean': 100.0,
        'סכום קנייה ($)': 100.0,
        'Unit_Price': 10.0,
        'Risk_Level': 'core'
    })

    out = v2.score_ticker_v2_enhanced('TST', row, budget_total=5000.0, min_position=50.0, enable_ml=False)

    # Expect risk gate full and a buy amount consistent with calculations
    assert out['risk_gate_status_v2'] == 'full'
    # buy amount should be non-zero and less than core cap (750) but > min position
    assert out['buy_amount_v2'] > 0
    assert out['buy_amount_v2'] <= 750.0
    assert out['buy_amount_v2'] >= 50.0
    # Shares to buy reflect floor division
    assert out['shares_to_buy_v2'] == int(out['buy_amount_v2'] / row['Unit_Price'])
