"""P3-a: the score must use the REAL reward/risk (it used a placeholder 2.0 because scoring ran before the RR stage)."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

import core.pipeline.runner as runner
from core.scoring_engine import compute_final_score_20d


def _row(rr):
    return {"Ticker": "T", "TechScore_20d": 60.0, "Fundamental_Score": 55.0, "Reliability_Score": 90.0, "ML_20d_Prob": 0.4, "ATR_Pct": 0.04,
            "RR": rr, "FinalScore_20d": 50.0, "Score": 50.0, "ScoreBreakdown": "{}"}


def test_real_rr_changes_the_score_and_is_recorded():
    ctx = SimpleNamespace(results=pd.DataFrame([_row(1.2), _row(4.0), _row(2.0)]))
    assert runner._rescore_with_real_rr(ctx) == 3
    s = ctx.results["FinalScore_20d"].tolist()
    assert s[1] > s[2] > s[0]                                           # better RR -> higher score; RR 2.0 sits in between
    bd = [json.loads(x) for x in ctx.results["ScoreBreakdown"]]
    assert [b["rr_ratio"] for b in bd] == [1.2, 4.0, 2.0] and (ctx.results["Score"] == ctx.results["FinalScore_20d"]).all()


def test_missing_rr_keeps_first_pass_score_instead_of_being_punished():
    ctx = SimpleNamespace(results=pd.DataFrame([_row(np.nan), _row(-1.0), _row(3.0)]))
    assert runner._rescore_with_real_rr(ctx) == 1
    assert ctx.results.loc[0, "FinalScore_20d"] == 50.0 and ctx.results.loc[1, "FinalScore_20d"] == 50.0 and ctx.results.loc[2, "FinalScore_20d"] != 50.0


def test_flag_off_restores_old_behaviour(monkeypatch):
    monkeypatch.setattr(runner, "RR_REAL_IN_SCORE", False)
    ctx = SimpleNamespace(results=pd.DataFrame([_row(4.0)]))
    assert runner._rescore_with_real_rr(ctx) == 0 and ctx.results.loc[0, "FinalScore_20d"] == 50.0
