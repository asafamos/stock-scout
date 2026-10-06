"""P2-A2 (2026-10-07): ML decoupled from decisions. Defaults = ML has no influence; flags restore the legacy behaviour."""
import numpy as np
import pandas as pd
import pytest

import core.scoring_engine as se
from core.trading.config import TradingConfig
from core.trading.policy import evaluate_static_gates


@pytest.fixture
def cfg():
    c = TradingConfig()
    c.min_score_to_trade, c.max_score_to_trade, c.min_rr_to_trade = 73.0, 95.0, 2.0
    c.min_ml_prob, c.max_ml_prob, c.min_confidence, c.min_reliability = 0.40, 0.60, "High", 50.0
    c.blocked_sectors, c.blocked_regimes = "Consumer Defensive", "PANIC,CORRECTION"
    return c


def _row(ml):
    return {"Ticker": "TEST", "FinalScore_20d": 80.0, "Score": 80.0, "RewardRisk": 3.0, "ML_20d_Prob": ml, "Sector": "Technology",
            "SignalQuality": "High", "Market_Regime": "SIDEWAYS", "Reliability_Score": 90.0, "Entry_Price": 50.0, "Stop_Loss": 46.0,
            "Target_Price": 58.0, "ATR_Pct": 0.04, "Fundamental_Score": 55.0, "TechScore_20d": 70.0, "Volume_Surge": 0.9}


def _ml_failures(res):
    return [g for g in res.gates_failed if g.startswith("ML ")]


def test_gate_flag_off_ignores_the_ml_value(cfg):
    cfg.ml_gate_enabled = False
    for ml in (0.05, 0.34, 0.95):                                     # far outside [0.40, 0.60]
        assert not _ml_failures(evaluate_static_gates(_row(ml), cfg=cfg))
    assert any("ML gate disabled" in p for p in evaluate_static_gates(_row(0.3), cfg=cfg).gates_passed)


def test_gate_flag_on_keeps_legacy_window(cfg):
    cfg.ml_gate_enabled = True
    assert _ml_failures(evaluate_static_gates(_row(0.30), cfg=cfg))      # below the floor → blocked
    assert not _ml_failures(evaluate_static_gates(_row(0.50), cfg=cfg))


def _score_row(ml):
    # minimal row for compute_final_score_20d
    return pd.Series({"TechScore_20d": 60.0, "Fundamental_Score": 55.0, "RewardRisk": 2.5, "Reliability_Score": 90.0, "ML_20d_Prob": ml,
                      "Ticker": "TEST", "ATR_Pct": 0.04})


def test_score_ignores_ml_by_default():
    assert se.ML_IN_DECISIONS is False
    lo, hi = se.compute_final_score_20d(_score_row(0.10), return_breakdown=True), se.compute_final_score_20d(_score_row(0.90), return_breakdown=True)
    (s_lo, bd_lo), (s_hi, bd_hi) = lo, hi
    assert s_lo == pytest.approx(s_hi)                                   # ML 0.10 vs 0.90 must not move the score
    assert bd_lo.get("ml_delta", 0.0) == 0.0 and "ml_gate_penalty" not in bd_lo and "ml_gate_bonus" not in bd_hi


def test_score_uses_ml_when_flag_restored(monkeypatch):
    monkeypatch.setattr(se, "ML_IN_DECISIONS", True)
    s_lo, bd_lo = se.compute_final_score_20d(_score_row(0.10), return_breakdown=True)
    s_hi, bd_hi = se.compute_final_score_20d(_score_row(0.90), return_breakdown=True)
    assert s_hi > s_lo and bd_lo.get("ml_delta", 0.0) < 0 < bd_hi.get("ml_delta", 0.0)
