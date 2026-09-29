"""The suite must be incapable of messaging the owner (see tests/conftest.py, 2026-09-29)."""
import pytest
import requests


def test_notification_helpers_are_stubbed_under_test():
    from core.trading import notifications as n
    assert n.notify_error("ctx", "body <b>") is None  # returns cleanly, sends nothing


def test_direct_telegram_http_is_blocked():
    with pytest.raises(AssertionError, match="REAL Telegram"):
        requests.post("https://api.telegram.org/botX/sendMessage", json={"text": "x"})
