"""Tests for the API preflight module's tri-state status logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_check(status_code=200, timeout=False, exception=None, key_present=True):
    """Call _check_provider with controlled mocks."""
    with patch("core.api_preflight._key_present", return_value=key_present):
        from core.api_preflight import _check_provider

        if not key_present:
            return _check_provider("TEST", "http://test.example.com", key_envs=("TEST_KEY",))

        if timeout:
            with patch("core.api_preflight.requests.get", side_effect=requests.Timeout("timed out")):
                return _check_provider("TEST", "http://test.example.com", key_envs=("TEST_KEY",))

        if exception:
            with patch("core.api_preflight.requests.get", side_effect=exception):
                return _check_provider("TEST", "http://test.example.com", key_envs=("TEST_KEY",))

        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = "response body"
        with patch("core.api_preflight.requests.get", return_value=mock_resp):
            return _check_provider("TEST", "http://test.example.com", key_envs=("TEST_KEY",))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckProviderTriState:
    """Verify the tri-state (up/degraded/down) classification."""

    def test_200_returns_ok_up(self):
        result = _call_check(status_code=200)
        assert result["ok"] is True
        assert result["level"] == "up"
        assert result["status"] == "ok"

    def test_500_returns_not_ok_degraded(self):
        result = _call_check(status_code=500)
        assert result["ok"] is False
        assert result["level"] == "degraded"
        assert result["status"] == "transient_error"

    def test_503_returns_not_ok_degraded(self):
        result = _call_check(status_code=503)
        assert result["ok"] is False
        assert result["level"] == "degraded"

    def test_timeout_returns_not_ok_degraded(self):
        result = _call_check(timeout=True)
        assert result["ok"] is False
        assert result["level"] == "degraded"
        assert result["status"] == "transient_error"
        assert "timeout" in result["reason"]

    def test_429_returns_ok_degraded(self):
        """Rate limit means provider IS reachable, just busy."""
        result = _call_check(status_code=429)
        assert result["ok"] is True
        assert result["level"] == "degraded"
        assert result["status"] == "rate_limit"

    def test_401_returns_not_ok_down(self):
        result = _call_check(status_code=401)
        assert result["ok"] is False
        assert result["level"] == "down"
        assert result["status"] == "auth_error"

    def test_403_returns_not_ok_down(self):
        result = _call_check(status_code=403)
        assert result["ok"] is False
        assert result["level"] == "down"
        assert result["status"] == "auth_error"

    def test_no_key_returns_not_ok_down(self):
        result = _call_check(key_present=False)
        assert result["ok"] is False
        assert result["level"] == "down"
        assert result["status"] == "no_key"

    def test_generic_exception_returns_degraded(self):
        result = _call_check(exception=ConnectionError("DNS failure"))
        assert result["ok"] is False
        assert result["level"] == "degraded"

    def test_key_present_no_url(self):
        """Provider with key but no URL (e.g. NASDAQ) → up."""
        with patch("core.api_preflight._key_present", return_value=True):
            from core.api_preflight import _check_provider
            result = _check_provider("NASDAQ", None, key_envs=("NASDAQ_API_KEY",))
        assert result["ok"] is True
        assert result["level"] == "up"

    def test_other_http_code_returns_degraded(self):
        """Non-standard HTTP codes (e.g. 418) → degraded."""
        result = _call_check(status_code=418)
        assert result["ok"] is False
        assert result["level"] == "degraded"

    def test_all_results_have_level_field(self):
        """Every return path must include the 'level' field."""
        for sc in [200, 401, 403, 429, 500, 418]:
            result = _call_check(status_code=sc)
            assert "level" in result, f"Missing 'level' for status_code={sc}"
        # Timeout
        result = _call_check(timeout=True)
        assert "level" in result
        # No key
        result = _call_check(key_present=False)
        assert "level" in result
