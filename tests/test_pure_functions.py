import math
import numpy as np
import pandas as pd
import pytest

from indicators import rsi, atr, macd_line
from core.scoring.fundamental import compute_fundamental_score_with_breakdown, fundamental_score_legacy
from core.portfolio import _normalize_weights, allocate_budget


def test_rsi_basic_trend():
    # Upward trend should produce high RSI (>70) near end
    prices = pd.Series(np.linspace(100, 120, 50))
    rs = rsi(prices, period=14)
    assert rs.iloc[-1] > 70
    assert 0 <= rs.dropna().min() >= 0
    assert rs.dropna().max() <= 100


def test_atr_constant_range():
    # Constant daily range leads ATR equal to that range after warmup
    highs = pd.Series([10]*30)
    lows = pd.Series([9]*30)
    closes = pd.Series([9.5]*30)
    df = pd.DataFrame({"High": highs, "Low": lows, "Close": closes})
    a = atr(df, period=5)
    # After period warmup ATR should approximate range=1
    assert abs(a.iloc[-1] - 1.0) < 1e-6


def test_macd_zero_when_flat():
    flat = pd.Series([50]*40)
    macd, signal, hist = macd_line(flat)
    # All should be ~0 for constant price series
    assert macd.abs().max() < 1e-9
    assert signal.abs().max() < 1e-9
    assert hist.abs().max() < 1e-9


def test_fundamental_score_breakdown_consistency():
    data = {
        "roe": 0.15,
        "roic": 0.12,
        "gm": 0.40,
        "rev_g_yoy": 0.20,
        "eps_g_yoy": 0.35,
        "pe": 15,
        "ps": 3,
        "de": 0.5,
    }
    score = compute_fundamental_score_with_breakdown(data)
    # Total score bounded and breakdown pieces present
    assert 0 <= score.total <= 100
    b = score.breakdown
    for attr in ["quality_score","growth_score","valuation_score","leverage_score"]:
        assert 0 <= getattr(b, attr) <= 100
    # Labels not empty
    for attr in ["quality_label","growth_label","valuation_label","leverage_label"]:
        assert getattr(b, attr)
    # Legacy compatibility (0-1 scale)
    legacy = fundamental_score_legacy(data)
    assert 0 <= legacy <= 1
    # Monotonic check: improvement should raise total
    better = data.copy(); better["roe"] = 0.20
    better_score = compute_fundamental_score_with_breakdown(better)
    assert better_score.total >= score.total


def test_normalize_weights_basic_and_allocate():
    w = {"a":2, "b":3, "c":5}
    nw = _normalize_weights(w)
    assert pytest.approx(sum(nw.values()), 1e-9) == 1.0
    assert all(v > 0 for v in nw.values())
    # Ratio preserved roughly
    assert nw["a"] < nw["c"]
    df = pd.DataFrame({"Ticker":["A","B","C"],"Score":[90,60,30]})
    out = allocate_budget(df, total=600, min_pos=0, max_pos_pct=60)
    assert out["סכום קנייה ($)"].sum() == 600


def test_normalize_weights_invalid_total():
    w = {"a":0, "b":0}
    nw = _normalize_weights(w)
    assert nw["a"] == nw["b"] == 0.5

