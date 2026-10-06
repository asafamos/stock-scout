import pandas as pd
from datetime import datetime, timezone, timedelta
from scripts.scan_quality_report import evaluate


def _df(ml, vix=15.0, src="FMP", rows=None):
    n = rows or len(ml)
    return pd.DataFrame({"ML_20d_Prob": ml, "ATR_Pct": [0.03] * n, "VIX_Value": [vix] * n, "VIX_Source": [src] * n, "Reliability_Score": [95] * n, "Fundamental_Score": [50.0] * n})


def codes(res, level="RED"): return {c for l, c, _ in res["flags"] if l == level}


def test_healthy_scan_has_no_red_flags():
    ml = [0.20 + 0.40 * i / 99 for i in range(100)]                      # 0.20..0.60 → ~50% pass the 0.40–0.60 gate
    assert codes(evaluate(_df(ml), vix_real=15.1)) == set()


def test_flags_collapsed_ml_gate_and_synthetic_vix():
    r = evaluate(_df([0.33] * 100, vix=9.8, src="SYNTHETIC_VIX_PROXY"), vix_real=15.1)   # Aug-2026 pattern: nothing reaches 0.40
    assert {"ML_GATE_STRICTNESS", "VIX_PROXY", "VIX_MISMATCH"} <= codes(r)


def test_stale_scan_flagged_on_weekday_only():
    now = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)                 # Wednesday
    assert "STALE_SCAN" in codes(evaluate(_df([0.45] * 100), None, now, now - timedelta(hours=100)))
    sat = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    assert "STALE_SCAN" not in codes(evaluate(_df([0.45] * 100), None, sat, sat - timedelta(hours=100)))


def test_ml_pass_rate_is_informational_when_the_gate_is_disabled(monkeypatch):
    monkeypatch.setenv("TRADE_ML_GATE_ENABLED", "0")
    r = evaluate(_df([0.33] * 100), vix_real=15.1)
    assert "ML_GATE_STRICTNESS" not in codes(r) and "ML_GATE_STRICTNESS" in codes(r, "INFO")
    monkeypatch.setenv("TRADE_ML_GATE_ENABLED", "1")
    assert "ML_GATE_STRICTNESS" in codes(evaluate(_df([0.33] * 100), vix_real=15.1))
