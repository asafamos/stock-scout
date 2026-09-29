"""Buy-side dedup must FAIL CLOSED (2026-09-29 audit).

get_open_positions() returned [] on a corrupt file and IBKRClient.get_positions() returned []
on any IB error — the BUY path read both as "we hold nothing, all slots free" (duplicate buys).
"""
from types import SimpleNamespace

import pytest

from core.trading import ibkr_client as ic
from core.trading import order_manager as om
from core.trading.position_tracker import PositionTracker, TrackerUnreadable
from core.trading.risk_manager import RiskManager


def _tracker(tmp_path, text=None):
    t = PositionTracker.__new__(PositionTracker)
    t._positions_path = tmp_path / "open_positions.json"
    if text is not None:
        t._positions_path.write_text(text)
    return t


def test_corrupt_tracker_lenient_says_empty_strict_raises(tmp_path):
    t = _tracker(tmp_path, "{not json")
    assert t.get_open_positions() == []          # legacy behaviour kept for the monitor
    with pytest.raises(TrackerUnreadable):
        t.get_open_positions_strict()


def test_missing_tracker_file_is_a_legitimate_empty(tmp_path):
    assert _tracker(tmp_path).get_open_positions_strict() == []


def test_healthy_tracker_strict_returns_rows(tmp_path):
    assert _tracker(tmp_path, '[{"ticker": "AAA"}]').get_open_positions_strict() == [{"ticker": "AAA"}]


def test_risk_manager_refuses_when_tracker_unreadable(tmp_path):
    rm = RiskManager(SimpleNamespace(), _tracker(tmp_path, "{oops"), SimpleNamespace())
    ok, why = rm.check_tracker_readable()
    assert ok is False and "unreadable" in why.lower()
    healthy = _tracker(tmp_path, "[]")
    assert RiskManager(SimpleNamespace(), healthy, SimpleNamespace()).check_tracker_readable() == (True, "")


def _client_with(ib_positions):
    c = ic.IBKRClient.__new__(ic.IBKRClient)
    c.cfg = SimpleNamespace(dry_run=False)
    c._ib = SimpleNamespace(positions=ib_positions)
    return c


def test_ib_positions_lenient_swallows_strict_raises():
    def boom():
        raise RuntimeError("IB down")
    c = _client_with(boom)
    assert c.get_positions() == []               # legacy: swallowed
    with pytest.raises(RuntimeError):
        c.get_positions_strict()                 # buy path: must know


def test_strict_held_tickers_paths(tmp_path):
    good_ib = lambda: [SimpleNamespace(contract=SimpleNamespace(symbol="PBF"), position=5.0, avgCost=70.0),
                       SimpleNamespace(contract=SimpleNamespace(symbol="ZERO"), position=0.0, avgCost=1.0)]
    mgr = om.OrderManager.__new__(om.OrderManager)
    mgr.tracker, mgr.client = _tracker(tmp_path, "[]"), _client_with(good_ib)
    assert mgr._strict_held_tickers() == {"PBF"}          # zero-qty residuals ignored

    mgr.tracker = _tracker(tmp_path, "{corrupt")
    with pytest.raises(TrackerUnreadable):
        mgr._strict_held_tickers()

    def boom():
        raise RuntimeError("IB down")
    mgr.tracker, mgr.client = _tracker(tmp_path, "[]"), _client_with(boom)
    with pytest.raises(RuntimeError):
        mgr._strict_held_tickers()


def test_daily_buy_count_uses_utc_date(tmp_path):
    """Tracker timestamps are UTC; the day boundary must be too (was local date.today())."""
    from datetime import datetime
    now = datetime.utcnow()
    t = _tracker(tmp_path, "[]")
    t.get_trade_log = lambda: [
        {"action": "OPEN", "timestamp": now.isoformat()},
        {"action": "OPEN", "timestamp": "2000-01-01T00:00:00"},
        {"action": "CLOSE", "timestamp": now.isoformat()},
    ]
    assert PositionTracker.daily_buy_count(t) == 1
