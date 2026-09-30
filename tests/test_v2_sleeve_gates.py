"""can_open_position(sleeve='v2'): performance gates skipped, SAFETY gates kept."""
from types import SimpleNamespace

import pytest

from core.trading.risk_manager import RiskManager
from core.trading import policy


def _rm(cfg_over=None, held=(), net=821.0, cash=800.0):
    cfg = SimpleNamespace(
        max_daily_loss_pct=5.0, max_drawdown_pct=10.0, max_drawdown_halt_pct=25.0, drawdown_size_mult=0.5,
        max_position_size=450, max_open_positions=3, max_daily_buys=3, max_portfolio_exposure=1350,
        ledger_enabled=False, throttle_enabled=False, dry_run=True, earnings_gate_enabled=False,
        min_score_to_trade=73, max_score_to_trade=85, min_rr_to_trade=2.5, max_rr_to_trade=5.0,
        min_ml_prob=0.40, max_ml_prob=0.60, min_atr_pct=0.03, min_fundamental_score=45, max_volume_surge=1.5,
        min_reliability=50, blocked_sectors_list=["Utilities"], blocked_regimes_list=["PANIC"],
        adaptive_gates_enabled=False, min_addv_usd=0, min_confidence="High", cash_reserve=0,
        ml_sizing_enabled=False, reduce_regimes_list=[], max_sector_positions=2,
    )
    for k, v in (cfg_over or {}).items():
        setattr(cfg, k, v)
    client = SimpleNamespace(
        get_net_liquidation=lambda: net, get_cash_balance=lambda: cash, is_market_open=lambda: True,
        _ib=SimpleNamespace(portfolio=lambda: [], fills=lambda: [], executions=lambda: []),
        get_todays_sells=lambda: [], get_executions_today=lambda: [],
    )
    tracker = SimpleNamespace(
        get_trade_log=lambda: [], get_open_positions=lambda: [{"ticker": t} for t in held],
        is_holding=lambda t: t in held, open_count=len(held), daily_buy_count=lambda: 0, total_exposure=0.0,
    )
    return RiskManager(client, tracker, cfg)


def _call(rm, sleeve, **kw):
    base = dict(ticker="AAA", price=20.0, score=0.0, rr=0.0, sector="Technology", atr_pct=0.05,
                stop_loss=16.0, target_price=32.0, market_regime="SIDEWAYS", ml_prob=0.0,
                signal_quality="", reliability_score=100.0, fundamental_score=-1.0)
    base.update(kw)
    return rm.can_open_position(sleeve=sleeve, **base)


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch):
    monkeypatch.setattr("core.trading.notifications.notify_error", lambda *a, **k: None)
    monkeypatch.setattr(policy, "_load_blocked_tickers", lambda: set())


def test_legacy_rejects_a_zero_score_row_but_the_sleeve_does_not_care():
    rm = _rm()
    ok_legacy, why = _call(rm, "")
    assert not ok_legacy                               # Score/ML/RR gates reject it
    ok_v2, why2 = _call(rm, "v2")
    assert ok_v2, why2


def test_sleeve_keeps_the_safety_gates():
    rm = _rm()
    assert not _call(rm, "v2", sector="Utilities")[0]             # blocked sector
    assert not _call(rm, "v2", market_regime="PANIC")[0]          # blocked regime
    assert not _call(_rm(held=("AAA",)), "v2")[0]                 # already holding
    full = _rm(held=("X", "Y", "Z"))
    assert not _call(full, "v2", ticker="AAA")[0]                 # max open positions


def test_sleeve_respects_the_drawdown_halt(monkeypatch):
    rm = _rm()
    monkeypatch.setattr(rm, "check_drawdown_breaker", lambda: (False, "Drawdown breaker: halt"))
    ok, why = _call(rm, "v2")
    assert not ok and "Drawdown" in why


def test_market_closed_blocks_the_sleeve():
    rm = _rm(cfg_over={"dry_run": False})
    rm.client.is_market_open = lambda: False
    ok, why = _call(rm, "v2")
    assert not ok
