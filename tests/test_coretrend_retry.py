"""2026-10-19: the 13:45 UTC CoreTrend run failed (IB gateway waiting for weekly 2FA). A second attempt (11:45 ET) retries ONLY if the
first run did not complete cleanly — it must never repeat/extend a phase that already ran (SELL today, BUY after settlement)."""
import types
from datetime import datetime, timezone

import pytest

from scripts import run_coretrend as rc


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "DONE", tmp_path / "coretrend_last_ok.json")
    monkeypatch.setattr(rc, "KILL", tmp_path / "no_kill.json")
    monkeypatch.setenv("TRADE_CORETREND", "1")
    monkeypatch.delenv("TRADE_LIVE_CONFIRMED", raising=False)
    return tmp_path


def _today():
    return datetime.now(timezone.utc).date()


def test_marker_roundtrip_and_date_scoped(env):
    assert not rc._done_today(_today())
    rc._mark_ok(_today(), "noop")
    assert rc._done_today(_today())
    assert not rc._done_today(_today().replace(year=_today().year - 1))


def test_retry_exits_before_touching_ib_when_todays_run_completed(env, monkeypatch, capsys):
    rc._mark_ok(_today(), "acted")
    monkeypatch.setattr(rc, "current_signal", lambda *_: (_ for _ in ()).throw(AssertionError("must not run")))
    assert rc.main(["--retry"]) == 0
    assert "already completed" in capsys.readouterr().out


def test_retry_proceeds_when_nothing_completed_today(env, monkeypatch):
    monkeypatch.setattr(rc, "current_signal", lambda *_: (_ for _ in ()).throw(RuntimeError("proceeded")))
    with pytest.raises(RuntimeError, match="proceeded"):
        rc.main(["--retry"])


def test_plain_run_is_not_blocked_by_the_marker(env, monkeypatch):
    rc._mark_ok(_today(), "acted")
    monkeypatch.setattr(rc, "current_signal", lambda *_: (_ for _ in ()).throw(RuntimeError("proceeded")))
    with pytest.raises(RuntimeError, match="proceeded"):
        rc.main([])            # manual / first run is never skipped by the marker


class _FakeClient:
    status = "Filled"
    def __init__(self, *a, **k): pass
    def connect(self): return True
    def disconnect(self): pass
    _ib = types.SimpleNamespace(positions=lambda: [])
    def get_cash_balance(self): return 500.0
    def get_net_liquidation(self): return 800.0
    def get_open_buy_symbols(self): return []
    def get_live_price(self, s): return 100.0
    def buy_with_bracket(self, *a, **k): return {"buy": types.SimpleNamespace(status=_FakeClient.status)}


def _live(monkeypatch, plan, status="Filled"):
    from core.trading import coretrend_exec as ex, ibkr_client, live_quote, market_hours, notifications, order_manager
    _FakeClient.status = status
    monkeypatch.setenv("TRADE_LIVE_CONFIRMED", "1")
    monkeypatch.setattr(rc, "current_signal", lambda *_: ({"state": "RISK_ON", "close": 740.0, "sma": 670.0, "gap_pct": 10.3}, _today()))
    monkeypatch.setattr(ibkr_client, "IBKRClient", _FakeClient)
    monkeypatch.setattr(order_manager, "_acquire_trade_lock", lambda: object())
    monkeypatch.setattr(order_manager, "_release_trade_lock", lambda l: None)
    monkeypatch.setattr(market_hours, "is_regular_session", lambda *a, **k: True)
    monkeypatch.setattr(live_quote, "get_realtime_price", lambda s: (100.0, "test"))
    monkeypatch.setattr(ex, "plan", lambda *a, **k: plan)
    sent = []
    monkeypatch.setattr(notifications, "_send", lambda m, *a, **k: sent.append(m) or True)
    monkeypatch.setattr(notifications, "notify_error", lambda *a, **k: sent.append("ERR"))
    return sent


def test_noop_live_run_writes_the_marker(env, monkeypatch):
    _live(monkeypatch, plan=[])
    assert rc.main([]) == 0
    assert rc._done_today(_today())


def test_successful_action_writes_the_marker(env, monkeypatch):
    _live(monkeypatch, plan=[{"action": "BUY", "symbol": "QQQM", "qty": 1, "why": "test"}])
    assert rc.main([]) == 0
    assert rc._done_today(_today())


def test_failed_action_leaves_no_marker_so_the_retry_runs(env, monkeypatch):
    sent = _live(monkeypatch, plan=[{"action": "BUY", "symbol": "QQQM", "qty": 1, "why": "test"}], status="Error")
    assert rc.main([]) == 0
    assert "ERR" in sent
    assert not rc._done_today(_today())
