import numpy as np
from ui.components.cards import build_clean_card

def test_build_clean_card_nan():
    row = {
        "Ticker": "TEST",
        "FinalScore_20d": np.nan,
        "Score": np.nan,
        "Target_Price": np.nan,
        "Entry_Price": np.nan,
    }
    html = build_clean_card(row)
    assert "N/A" in html
    assert "TypeError" not in html
