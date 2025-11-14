import numpy as np
import pandas as pd

from indicators import rsi, atr, macd_line
from core.portfolio import _normalize_weights
from core.portfolio import allocate_budget


def test_rsi_basic():
    prices = pd.Series([1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5, 6, 5])
    res = rsi(prices, period=5)
    assert isinstance(res, pd.Series)
    assert not res.isnull().all()


def test_atr_basic():
    df = pd.DataFrame({
        "High": [2, 3, 4, 5, 6, 7],
        "Low": [1, 2, 3, 4, 5, 6],
        "Close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    })
    res = atr(df, period=3)
    assert isinstance(res, pd.Series)


def test_normalize_weights_and_allocate():
    w = dict(ma=1, mom=1, rsi=0)
    nw = _normalize_weights(w)
    assert abs(sum(nw.values()) - 1.0) < 1e-6

    df = pd.DataFrame({
        "Ticker": ["A", "B", "C"],
        "Score": [90, 80, 10],
        "Unit_Price": [10, 20, 5],
    })
    out = allocate_budget(df, total=1000, min_pos=100, max_pos_pct=50)
    assert "סכום קנייה ($)" in out.columns
    assert out["סכום קנייה ($)"].sum() <= 1000 + 1
