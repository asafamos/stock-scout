"""Risk breakers must see broker-truth closes (2026-09-29 audit).

In ledger mode (default) trail-fired / limit closes go through position_tracker.drop_metadata,
which writes NO CLOSE row to trade_log; the loss also leaves ib.portfolio(). The daily-loss and
drawdown breakers read only trade_log CLOSE rows, so a trail loss was invisible to both.
The ledger (executions.jsonl, IB commissionReport.realizedPNL) is the source of truth.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.trading import ledger
from core.trading.risk_manager import RiskManager


def _rm(trade_log=None, net=800.0, ledger_enabled=True):
    client = SimpleNamespace(
        get_net_liquidation=lambda: net,
        _ib=SimpleNamespace(portfolio=lambda: []),  # nothing open -> no unrealized
    )
    tracker = SimpleNamespace(get_trade_log=lambda: trade_log or [])
    cfg = SimpleNamespace(max_daily_loss_pct=5.0, max_drawdown_pct=10.0,
                          max_position_size=450, max_open_positions=3,
                          ledger_enabled=ledger_enabled)
    return RiskManager(client, tracker, cfg)


def test_daily_loss_breaker_sees_ledger_trail_loss(monkeypatch):
    # -$60 realized today via a trail fill: in the ledger, NOT in trade_log
    monkeypatch.setattr(ledger, "realized_today", lambda cfg=None: -60.0)
    ok, reason = _rm(trade_log=[]).check_daily_loss_breaker()
    assert ok is False and "Daily loss" in reason, "-$60 on $800 (7.5%) must trip the 5% breaker"


def test_daily_loss_breaker_still_allows_when_no_loss(monkeypatch):
    monkeypatch.setattr(ledger, "realized_today", lambda cfg=None: 0.0)
    ok, _ = _rm(trade_log=[]).check_daily_loss_breaker()
    assert ok is True


def test_daily_loss_breaker_uses_worse_of_ledger_and_trade_log(monkeypatch):
    today = datetime.now().date().isoformat()
    log = [{"action": "CLOSE", "timestamp": today + "T14:00:00", "pnl": -55.0}]
    monkeypatch.setattr(ledger, "realized_today", lambda cfg=None: 0.0)  # ledger hasn't ingested yet
    ok, _ = _rm(trade_log=log).check_daily_loss_breaker()
    assert ok is False, "a legacy CLOSE row alone must still count"


def _trips(*pnls):
    return [{"ticker": f"T{i}", "realized_pnl": p, "exit_time": f"2026-09-{i+1:02d}T15:00:00+00:00"}
            for i, p in enumerate(pnls)]


def test_drawdown_reduces_size_between_cap_and_halt(monkeypatch):
    # +60 peak then -120 -> $120 below peak; on ~$800 NetLiq that is 13% (>10% cap, <25% halt)
    monkeypatch.setattr(ledger, "closed_round_trips", lambda cfg=None: _trips(60.0, -120.0))
    rm = _rm(trade_log=[], net=800.0)
    ok, _ = rm.check_drawdown_breaker()
    assert ok is True and rm._dd_size_mult == 0.5, "must keep trading, at half size"


def test_drawdown_halts_beyond_halt_pct(monkeypatch):
    monkeypatch.setattr(ledger, "closed_round_trips", lambda cfg=None: _trips(60.0, -300.0))
    ok, reason = _rm(trade_log=[], net=650.0).check_drawdown_breaker()  # $300/(650+300)=31.6%
    assert ok is False and "halt" in reason


def test_drawdown_recovers_after_deposit_and_matches_real_ledger_shape(monkeypatch):
    monkeypatch.setattr(ledger, "closed_round_trips", lambda cfg=None: _trips(60.0, -120.0))
    rm = _rm(trade_log=[], net=2000.0)  # deposit lifts NetLiq: 120/2120 = 5.7%
    ok, _ = rm.check_drawdown_breaker()
    assert ok is True and rm._dd_size_mult == 1.0


def test_drawdown_size_mult_reaches_the_sizing_multiplier(monkeypatch):
    monkeypatch.setattr(ledger, "closed_round_trips", lambda cfg=None: _trips(60.0, -120.0))
    rm = _rm(trade_log=[], net=800.0)
    rm.check_drawdown_breaker()
    assert min(1.0, getattr(rm, "_dd_size_mult", 1.0)) == 0.5
