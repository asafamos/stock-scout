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


def test_s3_is_logged_and_matches_the_live_selector(tmp_path):
    rows = []
    for i, (t, atr, mc, va) in enumerate([("WILD", 0.08, 5e8, 3e5), ("CALM", 0.02, 9e10, 5e6), ("MID", 0.05, 3e9, 4e5),
                                          ("ILLQ", 0.09, 1e8, 1e3)]):
        rows.append({"Ticker": t, "Close": 20.0, "ATR_Pct": atr, "market_cap": mc, "vol_vol": 0, "vol_avg": va,
                     "Sector": "Technology", "Fundamental_Score": 50.0, "As_Of_Date": "2026-09-30",
                     "FinalScore_20d": 78.0, "ML_20d_Prob": 0.5, "RewardRisk": 3.0, "Market_Regime": "SIDEWAYS",
                     "SignalQuality": "High", "Reliability_Score": 90})
    p = tmp_path / "s.parquet"
    pd.DataFrame(rows).to_parquet(p)
    picks, scans = tmp_path / "p.jsonl", tmp_path / "s.jsonl"
    shadow_log.log_scan(p, picks, scans, cfg=CFG)
    logged = {json.loads(x)["ticker"]: json.loads(x) for x in picks.read_text().splitlines()}
    assert logged["WILD"]["s3_rank"] == 1 and logged["ILLQ"]["s3_rank"] is None   # illiquid excluded
    assert json.loads(scans.read_text())["s3_picks"][0] == "WILD"


def test_s2_prefers_volatile_small_caps_and_skips_blocked(tmp_path):
    rows = [
        {"ticker": "BIGCALM", "close": 50.0, "atr_pct": 0.01, "market_cap": 5e11, "sector": "Technology"},
        {"ticker": "SMALLWILD", "close": 5.0, "atr_pct": 0.09, "market_cap": 3e8, "sector": "Technology"},
        {"ticker": "MIDWILD", "close": 20.0, "atr_pct": 0.07, "market_cap": 5e9, "sector": "Energy"},
        {"ticker": "SMALLCALM", "close": 9.0, "atr_pct": 0.02, "market_cap": 4e8, "sector": "Technology"},
        {"ticker": "UTILWILD", "close": 9.0, "atr_pct": 0.10, "market_cap": 1e8, "sector": "Utilities"},
        {"ticker": "NOCAP", "close": 9.0, "atr_pct": 0.10, "market_cap": None, "sector": "Technology"},
    ]
    ranks = shadow_log.select_s2(rows, {"Utilities"})
    assert list(ranks)[0] == "SMALLWILD" and "UTILWILD" not in ranks and "NOCAP" not in ranks
    assert len(ranks) == 3


def test_report_refuses_a_verdict_with_few_dates():
    picks = [{"scan_date": "2026-08-03", "ticker": "AAA", "s1_rank": 1, "live_gate_pass": True}]
    outs = [{"scan_date": "2026-08-03", "ticker": "AAA", "ret_pct": 10.0, "spy_ret_pct": 1.0}]
    txt = shadow_report.report(picks, outs, 0.5)
    assert "S1: CONTINUE" in txt and "1/60" in txt


def test_report_can_pass_with_enough_consistent_dates():
    picks, outs = [], []
    for i in range(70):
        d = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
        for t, rank, ret in (("AAA", 1, 6.0), ("BBB", None, 1.0), ("CCC", None, 0.5)):
            picks.append({"scan_date": d, "ticker": t, "s1_rank": rank, "live_gate_pass": t == "CCC"})
            outs.append({"scan_date": d, "ticker": t, "ret_pct": ret + (i % 3) * 0.1, "spy_ret_pct": 0.5})
    txt = shadow_report.report(picks, outs, 0.5)
    assert "S1: PASSES" in txt


def _bars_dl(tickers, lo, hi):
    """Fake downloader: 120 flat-then-rising sessions from 2026-06-01 for every ticker."""
    idx = pd.bdate_range("2026-06-01", periods=120)
    o = [100 + 0.2 * i for i in range(120)]
    df = pd.DataFrame({"Open": o, "High": [x + 1.5 for x in o], "Low": [x - 1.5 for x in o],
                       "Close": [x + 0.3 for x in o]}, index=idx)
    return {t: df for t in tickers}


def test_exit_resolver_writes_finished_policies_once(tmp_path, monkeypatch):
    picks = tmp_path / "picks.jsonl"
    picks.write_text(json.dumps({"scan_date": "2026-06-12", "ticker": "AAA", "s1_rank": 1, "s2_rank": None,
                                 "live_gate_pass": True}) + "\n")
    out = tmp_path / "exits.jsonl"
    monkeypatch.setattr("scripts.shadow_resolve.date", type("D", (date,), {"today": staticmethod(lambda: date(2026, 11, 20))}))
    n = shadow_resolve.resolve_exits(29, 100, picks, out, downloader=_bars_dl)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    pols = {r["policy"] for r in rows}
    assert n == len(rows) and {"LEGACY", "HOLD20", "CANARY", "HOLD60"} <= pols
    assert all(r["ret_pct"] > 0 for r in rows), "a steadily rising stock is a winner under every policy"
    assert shadow_resolve.resolve_exits(29, 100, picks, out, downloader=_bars_dl) == 0, "idempotent"


def test_exit_report_pairs_against_legacy():
    picks = [{"scan_date": f"2026-0{1 + i // 28}-{1 + i % 28:02d}", "ticker": "AAA", "s1_rank": 1, "s2_rank": None,
              "live_gate_pass": True} for i in range(20)]
    exits = []
    for p in picks:
        exits.append({"scan_date": p["scan_date"], "ticker": "AAA", "policy": "LEGACY", "ret_pct": 0.5, "days": 10, "reason": "stop"})
        exits.append({"scan_date": p["scan_date"], "ticker": "AAA", "policy": "CANARY", "ret_pct": 2.5, "days": 25, "reason": "time"})
    txt = shadow_report.exit_report(picks, exits, 0.5)
    assert "CANARY" in txt and "vs LEGACY +2.00pp" in txt
