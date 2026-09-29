"""NaN in a scan row must behave like MISSING data, never like a passing value (2026-09-29 audit)."""
import math
from types import SimpleNamespace

from core.trading import policy

CFG = SimpleNamespace(
    min_score_to_trade=73, max_score_to_trade=85, min_rr_to_trade=2.5, max_rr_to_trade=5.0,
    min_ml_prob=0.40, max_ml_prob=0.60, min_atr_pct=0.03, min_fundamental_score=45,
    max_volume_surge=1.5, min_reliability=50, blocked_sectors_list=[], blocked_regimes_list=["PANIC"],
    adaptive_gates_enabled=False, min_addv_usd=0, confidence_regime_relax=False,
    min_confidence="High",
)
GOOD = {"Ticker": "AAA", "FinalScore_20d": 78, "RewardRisk": 3.5, "ML_20d_Prob": 0.5, "Sector": "Technology",
        "SignalQuality": "High", "Market_Regime": "SIDEWAYS", "Reliability_Score": 90,
        "ATR_Pct": 0.05, "Fundamental_Score": 60, "Volume_Surge": 1.0}


def _gate(**over):
    row = dict(GOOD, **over)
    return policy.evaluate_static_gates(row, cfg=CFG, state={}, held_tickers=set())


def test_baseline_row_passes():
    r = _gate()
    assert r.would_buy, r.gates_failed


def test_nan_score_rr_ml_fail_their_gates():
    for col in ("FinalScore_20d", "RewardRisk", "ML_20d_Prob"):
        r = _gate(**{col: math.nan})
        assert not r.would_buy, f"NaN {col} must not pass"


def test_inf_is_missing_too():
    assert not _gate(RewardRisk=math.inf).would_buy


def test_nan_atr_is_missing_passthrough_like_zero():
    assert _gate(ATR_Pct=math.nan).would_buy == _gate(ATR_Pct=0).would_buy
