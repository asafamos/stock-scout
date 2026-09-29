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


def test_drawdown_breaker_sees_ledger_round_trips(monkeypatch):
    trips = [
        {"ticker": "A", "realized_pnl": 60.0, "exit_time": "2026-09-01T15:00:00+00:00"},
        {"ticker": "B", "realized_pnl": -120.0, "exit_time": "2026-09-05T15:00:00+00:00"},
    ]
    monkeypatch.setattr(ledger, "closed_round_trips", lambda cfg=None: trips)
    ok, reason = _rm(trade_log=[], net=800.0).check_drawdown_breaker()
    assert ok is False and "Drawdown" in reason, "a -$120 loss after a +$60 peak on ~$800 is >10% DD"
