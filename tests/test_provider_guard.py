import os
import types
import pytest

from core.provider_guard import get_provider_guard, ProviderDecision
from core.data_sources_v2 import fetch_price_multi_source


def test_cooldown_on_429(monkeypatch):
    # Reset guard state before test
    guard = get_provider_guard()
    guard.reset()
    
    # Force TIINGO as the only candidate provider
    monkeypatch.setenv("TIINGO_API_KEY", "test_token")
    # Block other providers via provider_status
    provider_status = {
        "POLYGON": {"can_price": False, "status": "disabled"},
        "EODHD": {"can_price": False, "status": "disabled"},
        "FMP": {"can_price": False, "status": "disabled"},
        "FINNHUB": {"can_price": False, "status": "disabled"},
        "MARKETSTACK": {"can_price": False, "status": "disabled"},
    }
    
    # Update guard with preflight info
    guard.update_from_preflight(provider_status)

    # Prevent any sleeps from rate limiter
    import core.data_sources_v2 as dsv2
    monkeypatch.setattr(dsv2, "_rate_limit", lambda *_args, **_kw: None)

    # Track requests.get calls for TIINGO
    call_counter = {"tiingo": 0}

    class FakeResp:
        def __init__(self, status_code=200, json_data=None):
            self.status_code = status_code
            self._json_data = json_data or {}
        def json(self):
            return self._json_data

    def fake_requests_get(url, *args, **kwargs):
        if "tiingo" in url:
            call_counter["tiingo"] += 1
            # First call: simulate 429 rate limit
            if call_counter["tiingo"] == 1:
                return FakeResp(status_code=429)
            # Subsequent calls should not happen if guard works
            return FakeResp(status_code=200, json_data=[{"close": 123.45}])
        # Default safe response
        return FakeResp(status_code=200, json_data={})

    monkeypatch.setattr(dsv2.requests, "get", fake_requests_get)

    # Also ensure time.sleep is not used by HTTP helper when provider is known
    def no_sleep(_seconds):
        raise AssertionError("sleep should not be called on guarded 429")
    monkeypatch.setattr(dsv2.time, "sleep", no_sleep)

    # First call triggers 429 and guard cooldown
    prices1 = fetch_price_multi_source("AAPL", provider_status=provider_status, telemetry=None)
    assert isinstance(prices1, dict)
    # Second call should NOT hit TIINGO requests.get again due to cooldown
    prices2 = fetch_price_multi_source("AAPL", provider_status=provider_status, telemetry=None)
    assert call_counter["tiingo"] == 1, "TIINGO should be skipped after 429 by ProviderGuard"


def test_permanent_disable_on_401():
    guard = get_provider_guard()
    guard.reset()  # Reset state before test
    # Record an auth failure (401)
    guard.record_failure("ALPHAVANTAGE", 401, "auth_error")
    allowed, reason, decision = guard.allow("ALPHAVANTAGE", "fundamentals")
    assert not allowed
    assert decision == ProviderDecision.BLOCK_PRECHECK
    assert reason in ("auth_error", "disabled_permanent")


def test_preflight_blocks_no_key():
    guard = get_provider_guard()
    guard.reset()  # Reset state before test
    guard.update_from_preflight({"FINNHUB": {"status": "no_key"}})
    allowed, reason, decision = guard.allow("FINNHUB", "price")
    assert not allowed
    assert decision == ProviderDecision.BLOCK_PRECHECK
    assert reason == "no_key"
