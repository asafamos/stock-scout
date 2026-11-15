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
