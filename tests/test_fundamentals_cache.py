"""Fundamentals cache (2026-11-15): TTL skip, stale fallback during a provider outage, corrupt-file safety."""
import json
import time

import pytest

from core import fundamentals_cache as fc


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FUND_CACHE", "1")
    monkeypatch.delenv("FUND_CACHE_TTL_DAYS", raising=False)
    monkeypatch.delenv("FUND_CACHE_STALE_MAX_DAYS", raising=False)
    fc.reset_for_tests(str(tmp_path / "fc.json"))
    yield
    fc.reset_for_tests(str(tmp_path / "fc2.json"))


def test_put_get_roundtrip_and_ttl(monkeypatch):
    fc.put("AAA", "fmp", {"pe": 12.0, "roe": 0.2})
    assert fc.get("AAA", "fmp") == {"pe": 12.0, "roe": 0.2}
    assert fc.get("AAA", "finnhub") is None
    # older than TTL but younger than the stale limit
    fc._load()["AAA"]["fmp"]["ts"] = time.time() - 10 * 86400
    assert fc.get("AAA", "fmp") is None
    assert fc.get("AAA", "fmp", fc.stale_max_days()) == {"pe": 12.0, "roe": 0.2}
    fc._load()["AAA"]["fmp"]["ts"] = time.time() - 60 * 86400
    assert fc.get("AAA", "fmp", fc.stale_max_days()) is None


def test_save_and_reload(tmp_path):
    fc.put("BBB", "tiingo", {"pe": 9.0})
    fc.save()
    p = tmp_path / "fc.json"
    assert p.exists() and "BBB" in json.loads(p.read_text())
    fc.reset_for_tests(str(p))
    assert fc.get("BBB", "tiingo") == {"pe": 9.0}


def test_corrupt_file_does_not_crash(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    fc.reset_for_tests(str(p))
    assert fc.get("X", "fmp") is None   # starts empty instead of raising
    fc.put("X", "fmp", {"pe": 1.0})
    assert fc.get("X", "fmp") == {"pe": 1.0}


def test_disabled(monkeypatch):
    monkeypatch.setenv("FUND_CACHE", "0")
    fc.put("CCC", "fmp", {"pe": 5.0})
    assert fc.get("CCC", "fmp") is None


def test_aggregate_uses_cache_then_stale_on_outage(monkeypatch):
    from core import data_sources_v2 as ds

    calls = {"n": 0, "fail": False}

    def fake_fmp(ticker, provider_status=None):
        calls["n"] += 1
        if calls["fail"]:
            raise RuntimeError("HTTP 429")
        return {"pe": 15.0, "ps": 2.0, "pb": 3.0, "roe": 0.15, "margin": 0.1, "market_cap": 1e9}

    monkeypatch.setattr(ds, "get_prioritized_fetch_funcs", lambda ps: [("fmp", fake_fmp)])
    r1 = ds.aggregate_fundamentals("ZZZ")
    assert calls["n"] == 1 and "fmp" in r1.get("sources_used", ["fmp"])
    # fresh cache -> no second API call
    ds.aggregate_fundamentals("ZZZ")
    assert calls["n"] == 1
    # age the entry past the TTL, provider now rate-limited -> stale value still used
    fc._load()["ZZZ"]["fmp"]["ts"] = time.time() - 10 * 86400
    calls["fail"] = True
    r3 = ds.aggregate_fundamentals("ZZZ")
    assert calls["n"] == 2
    assert not r3.get("Fundamental_Missing"), "stale cache should have prevented the missing-data fallback"


def test_alpha_premium_notice_disables_alpha_fundamentals(monkeypatch):
    """Alpha OVERVIEW is premium for our key: after one notice, no more calls / no 12 s pacer waits."""
    from core import data_sources_v2 as ds

    monkeypatch.setattr(ds, "ALPHA_VANTAGE_API_KEY", "k")
    monkeypatch.setattr(ds, "_get_from_cache", lambda k: None)
    monkeypatch.setattr(ds, "_rate_limit", lambda s: None)
    calls = {"n": 0}

    def fake_http(*a, **k):
        calls["n"] += 1
        return {"Information": "Thank you ... This is a premium endpoint. You may subscribe ..."}

    monkeypatch.setattr(ds, "_http_get_with_retry", fake_http)
    monkeypatch.setattr(ds, "DISABLED_PROVIDERS", set())
    assert ds.fetch_fundamentals_alpha("AAA") is None
    assert "alpha:fundamentals" in ds.DISABLED_PROVIDERS
    assert ds.fetch_fundamentals_alpha("BBB") is None
    assert calls["n"] == 1


def test_unusable_source_is_auto_disabled_but_finnhub_is_not(monkeypatch):
    from core import data_sources_v2 as ds

    monkeypatch.setattr(ds, "DISABLED_PROVIDERS", set())
    monkeypatch.setattr(ds, "_FUND_STREAK", {})
    monkeypatch.setattr(ds, "_FUND_DISABLE_AFTER", 3)
    for _ in range(3):
        ds._note_fund_outcome("tiingo", False)
        ds._note_fund_outcome("finnhub", False)
    assert "tiingo:fundamentals" in ds.DISABLED_PROVIDERS
    assert "finnhub:fundamentals" not in ds.DISABLED_PROVIDERS
    # a success resets the streak
    ds._note_fund_outcome("simfin", False); ds._note_fund_outcome("simfin", False)
    ds._note_fund_outcome("simfin", True); ds._note_fund_outcome("simfin", False)
    assert "simfin:fundamentals" not in ds.DISABLED_PROVIDERS


def test_index_series_uses_yfinance_for_etf_and_vix(monkeypatch):
    import pandas as pd
    from core import data_sources_v2 as ds

    df = pd.DataFrame({"date": pd.date_range("2026-08-01", periods=30), "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": 1.0, "volume": 0})
    monkeypatch.setattr(ds, "_fetch_index_yfinance", lambda sym, a, b: df)
    monkeypatch.setattr(ds, "_get_from_cache", lambda k: None)
    monkeypatch.setattr(ds, "_put_in_cache", lambda k, v: None)
    for sym in ("XLK", "^VIX"):
        out = ds.get_index_series(sym, "2026-08-01", "2026-11-13", provider_status={"fmp": False, "polygon": True})
        assert out is df and ds.get_last_index_source(sym) == "YFINANCE"
