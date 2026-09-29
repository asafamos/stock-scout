"""ATR-wide exit profile + the shared daily-bar exit simulator (2026-09-29 canary)."""
import math

import pytest

from core.trading import exit_profile as xp
from core.trading import exit_sim as xs


def test_scan_atr_is_a_fraction_and_is_converted():
    assert xp.atr_percent(0.034) == pytest.approx(3.4)
    assert xp.atr_percent(3.4) == pytest.approx(3.4)      # already percent
    assert xp.atr_percent(float("nan")) == 0.0 and xp.atr_percent(0) == 0.0


def test_wide_trail_is_atr_scaled_and_clamped():
    assert xp.wide_trail_pct(0.03) == 12.0                # 4 x 3%
    assert xp.wide_trail_pct(0.015) == 8.0                # floor
    assert xp.wide_trail_pct(0.08) == 20.0                # cap
    assert xp.wide_trail_pct(0.0) == 14.0                 # unknown ATR -> mid band, never tiny


def test_risk_cap_shrinks_size_for_wide_stops():
    # NetLiq 800 -> max loss $32 ; price 50, trail 16% => $8/share => 4 shares
    assert xp.risk_capped_qty(50.0, 16.0, 800.0, 9) == 4
    assert xp.risk_capped_qty(50.0, 8.0, 800.0, 9) == 8      # $4/share -> 8 shares
    assert xp.risk_capped_qty(50.0, 8.0, 800.0, 3) == 3      # never grows the order
    assert xp.risk_capped_qty(500.0, 20.0, 800.0, 5) == 0    # $100/share > $32 budget -> skip


def test_profile_default_is_legacy(monkeypatch):
    monkeypatch.delenv("TRADE_EXIT_PROFILE", raising=False)
    assert xp.profile() == "legacy"
    monkeypatch.setenv("TRADE_EXIT_PROFILE", "atr_wide")
    assert xp.profile() == "atr_wide"
    monkeypatch.setenv("TRADE_EXIT_PROFILE", "garbage")
    assert xp.profile() == "legacy"
    assert xp.is_wide({"exit_profile": "atr_wide"}) and not xp.is_wide({}) and not xp.is_wide(None)


def _bars(highs, lows, opens=None, closes=None):
    n = len(highs)
    return (opens or [100.0] * n), highs, lows, (closes or [100.0] * n)


def test_hold_exits_at_close_of_last_session():
    o, h, l, c = _bars([101] * 25, [99] * 25, closes=[100 + i * 0.1 for i in range(25)])
    r = xs.simulate(o, h, l, c, 0, xs.POLICIES["HOLD20"], cost_pct=0.5)
    assert r["finished"] and r["days"] == 20 and r["reason"] == "time"
    assert r["ret_pct"] == pytest.approx((c[19] / 100 - 1) * 100 - 0.5)


def test_trail_stops_on_low_and_gap_fills_at_open():
    # peak 110 after bar 1; stop 9% => 100.1 ; bar 3 gaps to open 95 => fill at 95, not at the stop
    o = [100, 105, 108, 95, 90, 90]
    h = [101, 110, 109, 96, 91, 91]
    l = [99, 104, 107, 94, 89, 89]
    c = [100, 109, 108, 95, 90, 90]
    r = xs.simulate(o, h, l, c, 0, {"kind": "pct", "p": 9.0, "max": 20})
    assert r["reason"] == "stop" and r["days"] == 4 and r["ret_pct"] == pytest.approx(-5.0)


def test_bar_cannot_raise_peak_and_stop_on_its_own_high():
    # bar 1 spikes to 130 with low 99 (> the 91 stop from the old peak 100): it must NOT stop out on
    # its own spike; the raised peak (stop 118.3) only bites from bar 2.
    o = [100, 100, 100]
    h = [100, 130, 100]
    l = [100, 99, 99]
    c = [100, 100, 100]
    r = xs.simulate(o, h, l, c, 0, {"kind": "pct", "p": 9.0, "max": 3})
    assert r["reason"] == "stop" and r["days"] == 3


def test_unfinished_when_data_runs_out():
    o = [100] * 5
    r = xs.simulate(o, [101] * 5, [99] * 5, [100] * 5, 0, xs.POLICIES["HOLD20"])
    assert r == {"finished": False}


def test_canary_policy_mirrors_live_constants_and_uses_entry_atr():
    p = xs.POLICIES["CANARY"]
    assert (p["K"], p["lo"], p["hi"], p["max"]) == (xp.ATR_MULT, xp.TRAIL_MIN, xp.TRAIL_MAX, xp.MAX_HOLD_SESSIONS)
    # 20 flat-ish bars with a 4-point true range on ~100 => ATR% ~4 => trail 16%
    n = 40
    o = [100.0] * n; h = [102.0] * n; l = [98.0] * n; c = [100.0] * n
    assert xs._pct_for(p, 5, xs.atr_pct_at(h, l, c, 20) / o[20] * 100) == pytest.approx(16.0)
