"""merge_save must not clobber concurrent tracker changes (2026-09-29 audit: lost-update race)."""
import json

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
