import pandas as pd

from core.unified_logic import compute_technical_score


def test_compute_technical_score_all_string_numbers():
    row = pd.Series({
        "MA_Aligned": 1.0,
        "Momentum_Consistency": 1.0,
        "RSI": 50,
        "VolSurge": 1.0,
        "Overext": 0.05,
        "Near52w": 10.0,
        "RR": 2.0,
        "ATR_Pct": 0.03,
        "MACD_Pos": True,
        "ADX14": 25.0,
    })
    weights = {"rsi": "0.2", "ma": 0.3, "mom": "0.5"}
    score = compute_technical_score(row, weights=weights)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def test_compute_technical_score_invalid_strings_graceful():
    row = pd.Series({
        "MA_Aligned": 1.0,
        "Momentum_Consistency": 1.0,
        "RSI": 50,
        "VolSurge": 1.0,
    })
    weights = {"rsi": "abc", "ma": "-1", "mom": None}
    score = compute_technical_score(row, weights=weights)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0
