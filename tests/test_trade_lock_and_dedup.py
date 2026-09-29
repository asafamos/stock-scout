"""Regression tests: trade-run lock + open-BUY-order dedup (2026-09-29 audit).

Overlapping runs (manual run_auto_trade, monitor opportunistic buy, pipeline) could both pass
the held-ticker check before either order filled -> double buy. The lock lives in
OrderManager.execute_recommendations; open BUY orders (any clientId) are filtered out too.
"""
from types import SimpleNamespace

import pytest

from core.trading import order_manager as om
from core.trading import ibkr_client as ic


def test_lock_is_exclusive_and_releasable(tmp_path):
    p = str(tmp_path / "t.lock")
    first = om._acquire_trade_lock(p)
    assert first is not None
    assert om._acquire_trade_lock(p) is None, "second concurrent run must be refused"
    om._release_trade_lock(first)
    again = om._acquire_trade_lock(p)
    assert again is not None, "lock must be reusable after release"
    om._release_trade_lock(again)


def _manager(calls):
    mgr = om.OrderManager.__new__(om.OrderManager)
    mgr.cfg = SimpleNamespace(dry_run=False)
    mgr._execute_recommendations_locked = lambda scan_df=None, _adaptive_retry=False: calls.append(_adaptive_retry) or ["ran"]
    return mgr


def test_second_run_is_aborted_while_lock_held(tmp_path, monkeypatch):
    p = str(tmp_path / "t.lock")
    real = om._acquire_trade_lock
    monkeypatch.setattr(om, "_acquire_trade_lock", lambda: real(p))
    calls = []
    mgr = _manager(calls)
    held = real(p)
    assert mgr.execute_recommendations() == [] and calls == []  # refused, inner never ran
    # the adaptive retry runs INSIDE the outer call and must not be blocked by its own lock
    assert mgr.execute_recommendations(_adaptive_retry=True) == ["ran"] and calls == [True]
    om._release_trade_lock(held)
    assert mgr.execute_recommendations() == ["ran"]  # free again -> runs, then releases
    assert mgr.execute_recommendations() == ["ran"]


def test_dry_run_never_takes_the_lock(tmp_path, monkeypatch):
    p = str(tmp_path / "t.lock")
    real = om._acquire_trade_lock
    monkeypatch.setattr(om, "_acquire_trade_lock", lambda: real(p))
    calls = []
    mgr = _manager(calls)
    mgr.cfg.dry_run = True
    held = real(p)
    assert mgr.execute_recommendations() == ["ran"]  # simulations are harmless -> not blocked
    om._release_trade_lock(held)


class _T:
    def __init__(self, sym, action, status):
        self.contract = SimpleNamespace(symbol=sym)
        self.order = SimpleNamespace(action=action)
        self.orderStatus = SimpleNamespace(status=status)


def _client(ib):
    c = ic.IBKRClient.__new__(ic.IBKRClient)
    c._ib = ib
    return c


def test_open_buy_symbols_active_buys_only():
    ib = SimpleNamespace(reqAllOpenOrders=lambda: [
        _T("PBF", "BUY", "Submitted"), _T("XYZ", "SELL", "Submitted"),
        _T("ABC", "BUY", "Cancelled"), _T("DEF", "BUY", "PreSubmitted"),
    ])
    assert _client(ib).get_open_buy_symbols() == {"PBF", "DEF"}


def test_open_buy_symbols_falls_back_to_openTrades():
    def boom():
        raise RuntimeError("no reqAllOpenOrders")
    ib = SimpleNamespace(reqAllOpenOrders=boom, openTrades=lambda: [_T("PBF", "BUY", "Submitted")])
    assert _client(ib).get_open_buy_symbols() == {"PBF"}
