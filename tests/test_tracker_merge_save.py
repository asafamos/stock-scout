"""merge_save must not clobber concurrent tracker changes (2026-09-29 audit: lost-update race)."""
import json

import pytest

from core.trading.position_tracker import PositionTracker


def _tracker(tmp_path, rows):
    t = PositionTracker.__new__(PositionTracker)
    t._positions_path = tmp_path / "open_positions.json"
    t._positions_path.write_text(json.dumps(rows))
    return t


def test_keeps_position_added_meanwhile(tmp_path):
    t = _tracker(tmp_path, [{"ticker": "AAA", "trailing_stop_pct": 9.0}])
    snapshot = t.get_open_positions()            # monitor reads
    # pipeline adds BBB while monitor is busy with IB calls
    cur = json.loads(t._positions_path.read_text()); cur.append({"ticker": "BBB"})
    t._positions_path.write_text(json.dumps(cur))
    snapshot[0]["trailing_stop_pct"] = 5.5       # monitor mutates + saves
    t.merge_save(snapshot)
    out = {p["ticker"]: p for p in json.loads(t._positions_path.read_text())}
    assert set(out) == {"AAA", "BBB"}, "new position must survive"
    assert out["AAA"]["trailing_stop_pct"] == 5.5


def test_does_not_resurrect_position_closed_meanwhile(tmp_path):
    t = _tracker(tmp_path, [{"ticker": "AAA"}, {"ticker": "CCC"}])
    snapshot = t.get_open_positions()
    t._positions_path.write_text(json.dumps([{"ticker": "AAA"}]))   # CCC closed elsewhere
    t.merge_save(snapshot)
    assert [p["ticker"] for p in json.loads(t._positions_path.read_text())] == ["AAA"]


def test_add_position_records_exit_profile(tmp_path):
    t = PositionTracker.__new__(PositionTracker)
    t._positions_path = tmp_path / "open_positions.json"
    t._positions_path.write_text("[]")
    t._log_trade = lambda *a, **k: None
    t.add_position("AAA", 3, 50.0, 40.0, 80.0, target_date="2026-11-10", trailing_stop_pct=12.0, score=78.0,
                   order_ids={}, exit_profile="atr_wide")
    t.add_position("BBB", 3, 50.0, 40.0, 80.0, trailing_stop_pct=9.0)
    rows = {p["ticker"]: p for p in json.loads(t._positions_path.read_text())}
    assert rows["AAA"]["exit_profile"] == "atr_wide" and "exit_profile" not in rows["BBB"]


def test_add_position_logs_execution_cost_vs_real_time_reference(tmp_path):
    t = PositionTracker.__new__(PositionTracker)
    t._positions_path = tmp_path / "open_positions.json"
    t._positions_path.write_text("[]")
    logged = []
    t._log_trade = lambda action, ticker, qty, price, extra=None: logged.append(extra)
    t.add_position("AAA", 3, 50.5, 40.0, 80.0, trailing_stop_pct=9.0, scan_price=49.0, ref_price=50.25)
    x = logged[0]
    assert x["slippage_pct"] == pytest.approx((50.5 - 49.0) / 49.0 * 100, abs=1e-3)       # vs planned scan entry
    assert x["slippage_vs_ref_pct"] == pytest.approx((50.5 - 50.25) / 50.25 * 100, abs=1e-3)  # true execution cost
