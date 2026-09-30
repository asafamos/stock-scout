"""v2 sleeve selector (S3_v1) + self-kill bookkeeping."""
import json

import pytest

from core.trading import v2_selector as v2


def R(t, close=20.0, atr=0.05, mc=2e9, va=1e6, sector="Technology"):
    return {"ticker": t, "close": close, "atr_pct": atr, "market_cap": mc, "vol_avg": va, "sector": sector}


def test_prefers_volatile_small_liquid_names():
    rows = [R("BIGCALM", atr=0.01, mc=5e11), R("SMALLWILD", atr=0.09, mc=4e8), R("MIDWILD", atr=0.07, mc=5e9),
            R("SMALLCALM", atr=0.02, mc=4e8)]
    ranks = v2.select_s3(rows, set())
    assert list(ranks)[0] == "SMALLWILD" and ranks["SMALLWILD"] == 1
    assert "BIGCALM" not in list(ranks)[:2]


def test_liquidity_price_sector_and_data_filters():
    rows = [R("GOOD"),
            R("ILLIQ", va=50_000),                 # ADDV 1M < 5M
            R("PENNY", close=3.0, va=5e6),         # price < 5
            R("BLOCKED", sector="Utilities"),
            R("NOCAP", mc=None), R("NOATR", atr=0), R("NOVOL", va=None)]
    assert list(v2.select_s3(rows, {"Utilities"})) == ["GOOD"]


def test_deterministic_ties_and_top_n():
    rows = [R(t) for t in ("CCC", "AAA", "BBB", "DDD")]      # identical scores
    assert list(v2.select_s3(rows, set(), top_n=3)) == ["AAA", "BBB", "CCC"]


def test_enabled_flag(monkeypatch):
    monkeypatch.delenv("TRADE_V2_SLEEVE", raising=False)
    assert not v2.enabled()
    monkeypatch.setenv("TRADE_V2_SLEEVE", "1")
    assert v2.enabled()


def test_sleeve_health_kills_on_cumulative_loss_and_on_bad_mean(tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "STATE_DIR", tmp_path)
    for i in range(20):
        v2.record_entry(f"T{i}", 1, 100.0, ts=f"2026-10-{i + 1:02d}T14:00:00+00:00")
    def trip(i, pnl):
        return {"ticker": f"T{i}", "realized_pnl": pnl, "entry_price": 100.0, "shares": 1,
                "exit_time": f"2026-10-{i + 1:02d}T18:00:00+00:00"}
    other = {"ticker": "LEGACY", "realized_pnl": -500.0, "entry_price": 100.0, "shares": 1, "exit_time": "2026-10-05T18:00:00+00:00"}
    # -$75 over 3 trades is now tolerated (limit $100); a non-sleeve loss never counts
    assert v2.sleeve_health([trip(0, -25.0), trip(1, -25.0), trip(2, -25.0), other])["ok"]
    # 4 trades losing $27 each = -$108 -> stop on cumulative loss
    h = v2.sleeve_health([trip(i, -27.0) for i in range(4)])
    assert not h["ok"] and h["n"] == 4 and "cumulative" in h["reason"]
    # 14 small losers averaging -3% (cum -$42): not enough closes for the mean test yet
    assert v2.sleeve_health([trip(i, -3.0) for i in range(14)])["ok"]
    # 15 of them -> mean test fires
    h = v2.sleeve_health([trip(i, -3.0) for i in range(15)])
    assert not h["ok"] and h["n"] == 15 and "mean" in h["reason"]
    # healthy
    assert v2.sleeve_health([trip(0, 4.0), trip(1, -2.0)])["ok"]


def test_disable_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "STATE_DIR", tmp_path)
    assert v2.disabled_reason() is None
    v2.disable("test")
    assert v2.disabled_reason() == "test"
