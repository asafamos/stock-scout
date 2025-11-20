import math
import pandas as pd
import numpy as np
import pytest

from stock_scout import calculate_rr


def test_rr_clipped_when_reward_very_large():
    entry = 100.0
    target = 600.0
    atr = 2.0
    # reward = 500, risk = max(atr*2=4, entry*0.01=1) = 4 -> rr = 125 -> clipped to 5
    rr = calculate_rr(entry, target, atr)
    assert isinstance(rr, float)
    assert math.isclose(rr, 5.0, rel_tol=1e-9)


def test_rr_fallback_price_used_when_atr_missing():
    entry = 100.0
    target = 150.0
    atr = float('nan')
    # simulate estimated ATR from history and pass it via fallback_price
    est_atr = 40.0
    # reward = 50, risk = max(est_atr*2=80, entry*0.01=1)=80 -> rr = 50/80 = 0.625
    rr = calculate_rr(entry, target, atr, fallback_price=est_atr)
    assert pytest.approx(rr, rel=1e-6) == 0.625


def test_rr_zero_when_target_not_greater():
    entry = 100.0
    target = 90.0
    atr = 5.0
    rr = calculate_rr(entry, target, atr)
    assert rr == 0.0


def test_rr_handles_nan_atr_by_entry_pct_fallback():
    entry = 200.0
    target = 201.5
    atr = float('nan')
    # No fallback provided -> atr fallback = max(entry*0.01, 1e-6) = 2.0
    # risk = max(atr*2=4.0, entry*0.01=2.0) = 4.0 -> reward = 1.5 -> rr = 0.375
    rr = calculate_rr(entry, target, atr)
    assert pytest.approx(rr, rel=1e-6) == 0.375


def test_rr_non_numeric_inputs_return_nan():
    assert math.isnan(calculate_rr(None, 150, 2))
    assert math.isnan(calculate_rr(100, None, 2))
    assert math.isnan(calculate_rr("notanumber", 150, 2))
