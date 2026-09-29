"""Exit-side price handling (2026-09-29 audit): real-time quote first, IB mark as fallback.

The buy path already used a real-time quote; exits (target-hit, peak/ratchet, sell limit) read
the delayed IB mark (~15 min lag), so targets were declared late and a sell limit
`mark*0.995` sat ABOVE the market after a >0.5% drop (never filling; sub-$2k has no MKT fallback).
"""
from types import SimpleNamespace

import pytest

from core.trading import live_quote as lq
from core.trading import ibkr_client as ic


def test_exit_price_prefers_realtime(monkeypatch):
    monkeypatch.setattr(lq, "get_realtime_price", lambda t: (101.5, "FMP+Finnhub"))
    assert lq.get_exit_price("X", 99.0) == (101.5, "FMP+Finnhub")


@pytest.mark.parametrize("behaviour", ["none", "disagree", "boom"])
def test_exit_price_never_blocks_falls_back_to_mark(monkeypatch, behaviour):
    def fake(t):
        if behaviour == "none":
            return None
        if behaviour == "disagree":
            raise lq.QuoteDisagreement("X: FMP $100 vs Finnhub $98")
        raise RuntimeError("network down")
    monkeypatch.setattr(lq, "get_realtime_price", fake)
    assert lq.get_exit_price("X", 99.0) == (99.0, "IB-mark")  # an exit is never refused


def test_monitor_fresh_price_helper(monkeypatch):
    import scripts.monitor_positions as mp
    monkeypatch.setattr(lq, "get_realtime_price", lambda t: (55.0, "FMP"))
    assert mp._fresh_price("X", 50.0) == 55.0
    monkeypatch.setattr(lq, "get_realtime_price", lambda t: None)
    assert mp._fresh_price("X", 50.0) == 50.0


class _Trade:
    def __init__(self, order):
        self.order = order
        self.orderStatus = SimpleNamespace(status="Filled", avgFillPrice=order.lmtPrice)


class _SellIB:
    def __init__(self):
        self.placed, self.tickers_calls = [], 0

    def qualifyContracts(self, c):
        return [c]

    def reqMarketDataType(self, n):
        pass

    def reqTickers(self, c):  # the DELAYED snapshot — must NOT be needed when real-time exists
        self.tickers_calls += 1
        raise AssertionError("delayed snapshot must be skipped when a real-time quote exists")

    def placeOrder(self, contract, order):
        self.placed.append(order)
        return _Trade(order)

    def sleep(self, s):
        pass


def _client(ib):
    c = ic.IBKRClient.__new__(ic.IBKRClient)
    c._ib = ib
    c.cfg = SimpleNamespace(dry_run=False)
    return c


def test_sell_limit_priced_from_realtime_and_skips_delayed_snapshot(monkeypatch):
    monkeypatch.setattr(lq, "get_realtime_price", lambda t: (80.00, "FMP"))
    ib = _SellIB()
    res = _client(ib)._sell_market("PBF", 8)
    assert ib.tickers_calls == 0
    assert res.status == "Filled"
    assert ib.placed[0].lmtPrice == round(80.00 * 0.995, 2)  # 79.60 — floor under the REAL price


def test_sell_falls_back_to_delayed_snapshot_when_no_realtime(monkeypatch):
    monkeypatch.setattr(lq, "get_realtime_price", lambda t: None)
    ib = _SellIB()
    ib.reqTickers = lambda c: [SimpleNamespace(marketPrice=lambda: 70.0, last=70.0)]
    res = _client(ib)._sell_market("PBF", 8)
    assert res.status == "Filled"
    assert ib.placed[0].lmtPrice == round(70.0 * 0.995, 2)  # legacy behaviour preserved
