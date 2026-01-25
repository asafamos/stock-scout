import types
import time as _time
from typing import List, Dict

import pytest

from core.provider_guard import get_provider_guard
from core.data_sources_v2 import _http_get_with_retry, aggregate_fundamentals, fetch_fundamentals_batch, get_prioritized_fetch_funcs


class DummyResponse:
    def __init__(self, status_code: int, json_obj=None):
        self.status_code = status_code
        self._json = json_obj or {}
        self.content = b"{}"

    def json(self):
        return self._json


def test_http_429_no_sleep_and_cooldown(monkeypatch):
    calls = {"sleep": 0, "get": 0}

    def fake_sleep(seconds):
        calls["sleep"] += 1
        # If sleep is called, test should fail later

    def fake_get(*args, **kwargs):
        calls["get"] += 1
        # Simulate 429 from provider
        return DummyResponse(429)

    monkeypatch.setattr("core.data_sources_v2.time.sleep", fake_sleep)
    monkeypatch.setattr("core.data_sources_v2.requests.get", fake_get)

    guard = get_provider_guard()
    # Ensure no prior cooldown
    allowed, _r, _d = guard.allow("TIINGO", "price")
    assert allowed

    # Call HTTP helper
    out = _http_get_with_retry("https://api.tiingo.com/tiingo/daily/XYZ/prices", headers={}, timeout=1, provider="TIINGO", capability="price")
    assert out is None
    assert calls["get"] == 1
    # Critically: we should not sleep on 429
    assert calls["sleep"] == 0

    snap = guard.snapshot()
    assert "TIINGO" in snap
    # Cooldown_until should be set (not None)
    assert snap["TIINGO"]["cooldown_until"] is not None


def test_preflight_blocked_provider_never_called(monkeypatch):
    urls: List[str] = []

    def fake_get(url, *args, **kwargs):
        urls.append(url)
        # Return 200 empty so callers move on
        return DummyResponse(200, json_obj={})

    monkeypatch.setattr("core.data_sources_v2.requests.get", fake_get)

    provider_status: Dict = {
        "TIINGO": {"status": "no_key", "can_fund": False},
        "FMP": {"ok": True, "can_fund": True},
        "FINNHUB": {"ok": True, "can_fund": True},
        "ALPHAVANTAGE": {"ok": True, "can_fund": True},
    }

    # Build prioritized functions and ensure Tiingo is excluded
    funcs = get_prioritized_fetch_funcs(provider_status)
    names = [name for name, _ in funcs]
    assert "tiingo" not in names

    # Run aggregation; ensure any requests to Tiingo endpoints aren't made
    _ = aggregate_fundamentals("AAPL", provider_status=provider_status)
    # Verify that no Tiingo URL appeared
    assert not any("tiingo" in u.lower() for u in urls)


def test_fetch_fundamentals_batch_imports_ok(monkeypatch):
    # Stub aggregate_fundamentals to avoid network and validate ThreadPoolExecutor path works
    def stub_agg(tkr, *args, **kwargs):
        return {"ticker": tkr, "Fundamental_Sources_Count": 0, "Fundamental_Coverage_Pct": 0.0}

    monkeypatch.setattr("core.data_sources_v2.aggregate_fundamentals", stub_agg)

    df = fetch_fundamentals_batch(["AAPL", "MSFT"], provider_status={"FMP": {"ok": True}})
    assert df is not None
    assert not df.empty
    assert list(df.index) == ["AAPL", "MSFT"]
