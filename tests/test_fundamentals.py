import pandas as pd
import numpy as np
from fundamentals import compute_bucket_scores, compute_fundamental_score

def test_fundamentals_handles_missing_and_renormalizes():
    idx = ["A", "B", "C"]
    df = pd.DataFrame({
        "roic": [10.0, np.nan, 5.0],
        "fcf_margin": [5.0, 2.0, np.nan],
        "oper_margin": [np.nan, 4.0, 3.0],
        "gross_margin_stability": [1.0, 1.0, 1.0],
        "rev_yoy": [0.1, 0.2, np.nan],
        "eps_yoy": [0.2, np.nan, 0.05],
        "fcf_yoy": [0.05, 0.02, np.nan],
        "fwd_pe": [15.0, np.nan, 8.0],
        "ev_ebitda": [10.0, 12.0, np.nan],
        "ps": [2.0, np.nan, 1.5],
        "peg": [1.0, np.nan, np.nan],
        "debt_to_equity": [0.5, 1.0, np.nan],
        "net_debt_ebitda": [1.0, np.nan, 0.3],
        "interest_cover": [5.0, 3.0, np.nan],
        "share_dilution_yoy": [0.0, 0.01, np.nan],
    }, index=idx)
    meta = pd.DataFrame({"sector": ["Tech", "Tech", "Health"], "market_cap": [1e9, 5e9, 20e9]}, index=idx)
    buckets = compute_bucket_scores(df, meta)
    f = compute_fundamental_score(buckets)
    assert f.min() >= 0.0 and f.max() <= 100.0