"""Shadow selector: logger idempotency, S1 rule, window measurement, report verdict gating."""
import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import shadow_log, shadow_report, shadow_resolve


def _scan(tmp_path, d="2026-09-29"):
    rows = []
    for i, (t, fund, sector) in enumerate([("AAA", 70, "Technology"), ("BBB", 65, "Energy"),
                                           ("CCC", 60, "Utilities"), ("DDD", 55, "Technology"),
                                           ("EEE", 50, "Industrials"), ("FFF", 40, "Technology")]):
        rows.append({"Ticker": t, "Fundamental_Score": fund, "Sector": sector, "Close": 50.0 + i,
                     "FinalScore_20d": 78.0, "ML_20d_Prob": 0.5, "RewardRisk": 3.0, "ATR_Pct": 0.05,
                     "Market_Regime": "SIDEWAYS", "SignalQuality": "High", "As_Of_Date": d,
                     "Reliability_Score": 90})
    p = tmp_path / "scan.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return p


CFG = SimpleNamespace(
    min_score_to_trade=73, max_score_to_trade=85, min_rr_to_trade=2.5, max_rr_to_trade=5.0, min_ml_prob=0.4,
    max_ml_prob=0.6, min_atr_pct=0.03, min_fundamental_score=45, max_volume_surge=1.5, min_reliability=50,
    blocked_sectors_list=["Utilities"], blocked_regimes_list=["PANIC"], adaptive_gates_enabled=False,
    min_addv_usd=0, min_confidence="High")


def test_s1_picks_top3_by_fund_skipping_blocked_and_low(tmp_path):
    p = _scan(tmp_path)
    picks, scans = tmp_path / "picks.jsonl", tmp_path / "scans.jsonl"
    n = shadow_log.log_scan(p, picks, scans, cfg=CFG)
    assert n == 6
    rows = [json.loads(x) for x in picks.read_text().splitlines()]
    ranks = {r["ticker"]: r["s1_rank"] for r in rows}
    assert ranks == {"AAA": 1, "BBB": 2, "CCC": None, "DDD": 3, "EEE": None, "FFF": None}
    assert json.loads(scans.read_text())["s1_picks"] == ["AAA", "BBB", "DDD"]


def test_second_scan_same_day_is_ignored(tmp_path):
    p = _scan(tmp_path)
    picks, scans = tmp_path / "picks.jsonl", tmp_path / "scans.jsonl"
    shadow_log.log_scan(p, picks, scans, cfg=CFG)
    assert shadow_log.log_scan(p, picks, scans, cfg=CFG) == 0
    assert len(picks.read_text().splitlines()) == 6


def _bars(n, start="2026-08-04", open0=100.0, step=0.5):
    idx = pd.bdate_range(start, periods=n)
    o = [open0 + step * i for i in range(n)]
    return pd.DataFrame({"Open": o, "High": [x + 1 for x in o], "Low": [x - 1 for x in o],
                         "Close": [x + 0.25 for x in o]}, index=idx)


def test_window_return_needs_all_20_sessions_and_uses_next_open():
    assert shadow_resolve.window_return(_bars(19), date(2026, 8, 3)) is None
    r = shadow_resolve.window_return(_bars(25), date(2026, 8, 3))
    assert r["bars"] == 20 and r["entry_open"] == 100.0
    assert r["exit_close"] == pytest.approx(100.0 + 0.5 * 19 + 0.25)


def test_bar_on_scan_date_is_not_used_for_entry():
    b = _bars(25, start="2026-08-03")          # first bar IS the scan date
    r = shadow_resolve.window_return(b, date(2026, 8, 3))
    assert r["entry_open"] == 100.5             # second bar's open


def test_report_refuses_a_verdict_with_few_dates():
    picks = [{"scan_date": "2026-08-03", "ticker": "AAA", "s1_rank": 1, "live_gate_pass": True}]
    outs = [{"scan_date": "2026-08-03", "ticker": "AAA", "ret_pct": 10.0, "spy_ret_pct": 1.0}]
    txt = shadow_report.report(picks, outs, 0.5)
    assert "VERDICT: CONTINUE" in txt and "1/60" in txt


def test_report_can_pass_with_enough_consistent_dates():
    picks, outs = [], []
    for i in range(70):
        d = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
        for t, rank, ret in (("AAA", 1, 6.0), ("BBB", None, 1.0), ("CCC", None, 0.5)):
            picks.append({"scan_date": d, "ticker": t, "s1_rank": rank, "live_gate_pass": t == "CCC"})
            outs.append({"scan_date": d, "ticker": t, "ret_pct": ret + (i % 3) * 0.1, "spy_ret_pct": 0.5})
    txt = shadow_report.report(picks, outs, 0.5)
    assert "S1 PASSES" in txt
