"""A passive ETF held in the same account must be invisible to the bot's bookkeeping."""
from types import SimpleNamespace

from core.trading import ignore_list as il
from core.trading import ledger


def test_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("TRADE_IGNORE_TICKERS", raising=False)
    assert il.is_ignored("spy") and il.is_ignored("VOO") and not il.is_ignored("AAPL")
    monkeypatch.setenv("TRADE_IGNORE_TICKERS", "XYZ, abc")
    assert il.ignored() == {"XYZ", "ABC"} and not il.is_ignored("SPY")
    monkeypatch.setenv("TRADE_IGNORE_TICKERS", "")
    assert il.ignored() == set()


def test_ledger_ignores_the_etf_round_trip_but_not_bot_trades(monkeypatch):
    monkeypatch.delenv("TRADE_IGNORE_TICKERS", raising=False)
    rows = [
        {"ticker": "SPY", "side": "BUY", "shares": 1, "price": 600.0, "time": "2026-10-01T14:00:00+00:00", "realized_pnl": 0.0, "commission": 1.0, "exec_id": "a"},
        {"ticker": "SPY", "side": "SELL", "shares": 1, "price": 640.0, "time": "2026-11-01T14:00:00+00:00", "realized_pnl": 38.0, "commission": 1.0, "exec_id": "b"},
        {"ticker": "AAA", "side": "BUY", "shares": 2, "price": 50.0, "time": "2026-10-02T14:00:00+00:00", "realized_pnl": 0.0, "commission": 1.0, "exec_id": "c"},
        {"ticker": "AAA", "side": "SELL", "shares": 2, "price": 45.0, "time": "2026-10-09T14:00:00+00:00", "realized_pnl": -11.0, "commission": 1.0, "exec_id": "d"},
    ]
    monkeypatch.setattr(ledger, "load", lambda cfg=None: rows)
    trips = ledger.closed_round_trips()
    assert [t["ticker"] for t in trips] == ["AAA"]
    assert ledger.realized_pnl() == -11.0


def test_client_position_getters_skip_ignored(monkeypatch):
    from core.trading import ibkr_client as ic
    pos = [SimpleNamespace(contract=SimpleNamespace(symbol="SPY"), position=2.0, avgCost=600.0),
           SimpleNamespace(contract=SimpleNamespace(symbol="AAA"), position=3.0, avgCost=50.0),
           SimpleNamespace(contract=SimpleNamespace(symbol="BBB"), position=0.0, avgCost=10.0)]
    c = ic.IBKRClient.__new__(ic.IBKRClient)
    c.cfg = SimpleNamespace(dry_run=False)
    c._ib = SimpleNamespace(positions=lambda: pos)
    monkeypatch.delenv("TRADE_IGNORE_TICKERS", raising=False)
    assert [p.ticker for p in c.get_positions()] == ["AAA"]
    assert [p.ticker for p in c.get_positions_strict()] == ["AAA"]
