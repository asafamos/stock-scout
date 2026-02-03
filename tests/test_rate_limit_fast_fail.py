import time
import types
import requests
from core.data_sources_v2 import _http_get_with_retry, DISABLED_PROVIDERS, fetch_price_multi_source
from core.provider_guard import get_provider_guard

def test_rate_limit_429_fast_fail(monkeypatch):
    # Reset provider guard state before test
    guard = get_provider_guard()
    guard.reset()
    
    # Track sleeps
    sleep_calls = {"count": 0}
    def fake_sleep(s):
        sleep_calls["count"] += 1
    monkeypatch.setattr(time, "sleep", fake_sleep)

    class Resp:
        def __init__(self, code):
            self.status_code = code
        def json(self):
            return {"ok": True}

    call_count = {"get": 0}
    def fake_get(url, params=None, headers=None, timeout=3):
        # Simulate 429 for Tiingo once
        call_count["get"] += 1
        if "tiingo" in url:
            return Resp(429)
        return Resp(200)

    monkeypatch.setattr(requests, "get", fake_get)

    # Call helper directly: should NOT sleep and should put provider on cooldown
    res = _http_get_with_retry(
        "https://api.tiingo.com/tiingo/daily/AAPL/prices",
        headers={},
        timeout=3,
        provider="TIINGO",
        capability="price",
    )
    assert res is None
    assert sleep_calls["count"] == 0
    
    # Check that guard is blocking TIINGO:price now (on cooldown)
    allowed, reason, _ = guard.allow("TIINGO", "price")
    assert not allowed, f"Expected TIINGO:price to be blocked, but got allowed with reason: {reason}"

    # Fallback: multi-source price fetch should skip Tiingo and not crash
    # Second call should be short-circuited by pre-check without invoking requests.get
    res2 = _http_get_with_retry(
        "https://api.tiingo.com/tiingo/daily/AAPL/prices",
        headers={},
        timeout=3,
        provider="TIINGO",
        capability="price",
    )
    # Result should be blocked sentinel dict
    assert res2 is None or (isinstance(res2, dict) and res2.get("__blocked__"))
    # requests.get should not be called again for tiingo due to pre-check; call_count remains 1
    assert call_count["get"] == 1
    prices = fetch_price_multi_source("AAPL", provider_status={}, telemetry=None)
    assert isinstance(prices, dict)
