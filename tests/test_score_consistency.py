import pandas as pd
import numpy as np
from core.scoring_config import get_canonical_score

def test_score_consistency():
    row = pd.Series({
        "FinalScore_20d": 88.0,
        "Score": 88.0,
        "overall_score_20d": 88.0,
        "overall_score": 88.0,
        "overall_score_pretty": 88.0,
    })
    score = get_canonical_score(row)
    assert score == 88.0

    # Edge case: missing canonical, fallback to alias
    row2 = pd.Series({
        "Score": 77.0,
        "overall_score_20d": np.nan,
    })
    score2 = get_canonical_score(row2)
    assert score2 == 77.0

    # Edge case: all missing
    row3 = pd.Series({})
    score3 = get_canonical_score(row3)
    assert np.isnan(score3)
