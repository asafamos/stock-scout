"""IB Gateway 10.52+ device-selection dialog auto-confirm (2026-11-08).

No real Telegram / docker: _docker_exec, send_message and time.sleep are stubbed.
"""
import importlib

import pytest


@pytest.fixture
def bot(monkeypatch):
    monkeypatch.setenv("TRADE_TELEGRAM_TOKEN", "x")
    monkeypatch.setenv("TRADE_TELEGRAM_CHAT_ID", "1")
    mod = importlib.import_module("scripts.telegram_status_bot")
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    mod._DEVICE_DIALOG_STATE["last_ts"] = 0.0
    sent = []
    monkeypatch.setattr(mod, "send_message", lambda t: sent.append(t))
    mod._sent = sent
    return mod


def _fake_exec(calls, present_after_key=False, wid="4194500"):
    state = {"keyed": False}

    def run(cmd):
        calls.append(cmd)
        if "search --name" in cmd:
            if state["keyed"] and not present_after_key:
                return ""
            return wid + "\n"
        if "key Return" in cmd:
            state["keyed"] = True
            return ""
        if "getwindowgeometry" in cmd:
            return "WINDOW=1\nX=100\nY=200\nWIDTH=500\nHEIGHT=100\nSCREEN=0\n"
        return ""
    return run


def test_no_dialog_does_nothing(bot, monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "_docker_exec", lambda c: calls.append(c) or "")
    assert bot._gw_handle_device_dialog() is False
    assert bot._sent == []
    assert not any("key Return" in c for c in calls)


def test_dialog_confirmed_with_return_and_owner_notified(bot, monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "_docker_exec", _fake_exec(calls))
    assert bot._gw_handle_device_dialog() is True
    assert any("key Return" in c for c in calls)
    assert not any("click 1" in c for c in calls)  # fallback not needed
    assert len(bot._sent) == 1 and "IB Key" in bot._sent[0]


def test_falls_back_to_geometry_click_when_dialog_persists(bot, monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "_docker_exec", _fake_exec(calls, present_after_key=True))
    bot._gw_handle_device_dialog()
    clicks = [c for c in calls if "click 1" in c]
    assert len(clicks) == 1
    # X=100 + 500*0.567 = 383 ; Y=200 + 100*0.80 = 280
    assert "mousemove 383 280 click 1" in clicks[0]


def test_cooldown_prevents_repeat_confirms(bot, monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "_docker_exec", _fake_exec(calls, present_after_key=True))
    bot._gw_handle_device_dialog()
    n = len(bot._sent)
    assert bot._gw_handle_device_dialog() is True  # dialog still there, cooling down
    assert len(bot._sent) == n
