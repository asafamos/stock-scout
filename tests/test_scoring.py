import pandas as pd
from scoring import normalize_series, final_score

def test_normalize_series_constant():
    s = pd.Series([1.0, 1.0, 1.0])
    out = normalize_series(s)
    assert (out == 0.0).all()

def test_final_score_bounds():
    tech = pd.DataFrame({"momentum": [0.1, 0.9], "trend": [0.2, 0.8], "overext_penalty": [0.0, 0.0]}, index=["A", "B"])
    bucket_scores = pd.DataFrame({"q_score": [50, 60], "g_score": [20, 30], "v_score": [30, 20], "l_score": [10, 10]}, index=["A", "B"])
    penalties = pd.DataFrame({"earnings": [0.0, -2.0]}, index=["A", "B"])
    # build minimal meta/metrics for final_score: final_score wrapper uses compute_fundamental_score internally,
    # so we call directly final_score scaffold with tech & synthetic metrics/meta
    metrics = pd.DataFrame(index=["A","B"])
    meta = pd.DataFrame({"sector":["X","X"], "market_cap":[1e9,2e9]}, index=["A","B"])
    f = final_score(tech, metrics, meta, penalties=penalties)
    assert f.min() >= 0 and f.max() <= 100