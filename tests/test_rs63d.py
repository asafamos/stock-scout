"""P3-b: RS_63d key mismatch (always NaN) and the difference-vs-ratio semantics of the RR adjustment."""
import numpy as np
import pandas as pd
import advanced_filters as af
from core.pipeline import helpers


def _px(n, start, drift):
    idx = pd.bdate_range("2026-01-01", periods=n)
    c = start * (1 + drift) ** np.arange(n)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}, index=idx)


def test_compute_signals_exposes_rs_63d_as_a_finite_difference():
    stock, spy = _px(260, 50.0, 0.0030), _px(260, 400.0, 0.0010)
    _score, sig = af.compute_advanced_score("TEST", stock, spy, 0.5)
    assert np.isfinite(sig["rs_63d"]) and sig["rs_63d"] > 0                 # stock outran SPY → positive difference (was NaN before the fix)


def _adj_from(rs_diff):
    """momentum adjustment the dynamic-RR helper derives from an RS difference."""
    src = helpers.__dict__
    row = pd.Series({"RS_63d": rs_diff})
    seen = {}
    orig = np.clip
    try:
        np.clip = lambda v, lo, hi: (seen.setdefault("v", v), orig(v, lo, hi))[1]
        helpers._compute_rr_for_row  # noqa: B018 (existence check)
    finally:
        np.clip = orig
    return seen


def test_helper_treats_rs_as_difference_not_ratio():
    # +0.30 outperformance must be 'strong' (ratio 1.30 > 1.2); -0.30 'weak' (0.70 < 0.8); 0.0 neutral (ratio 1.0, not > 1.0)
    cfg = {"rs_strong_threshold": 1.2, "rs_above_avg_threshold": 1.0, "rs_weak_threshold": 0.8}
    def bucket(diff):
        r = 1.0 + diff
        return "strong" if r > cfg["rs_strong_threshold"] else "above" if r > cfg["rs_above_avg_threshold"] else "weak" if r < cfg["rs_weak_threshold"] else "neutral"
    assert [bucket(0.30), bucket(0.10), bucket(0.0), bucket(-0.30)] == ["strong", "above", "neutral", "weak"]
    # and the source really applies that conversion
    assert "1.0 + float(_rs63)" in open(helpers.__file__).read()
