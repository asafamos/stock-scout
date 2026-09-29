"""Resolver v2 (2026-09-29 measurement audit): full window required, splits rebased, costs explicit."""
import pandas as pd

from scripts import track_scan_outcomes as t


def _hist(n, start="2026-08-04", px=100.0, split_at=None, ratio=2.0, tz=True):
    idx = pd.bdate_range(start, periods=n)
    if tz:
        idx = idx.tz_localize("America/New_York")
    px_list, splits = [], []
    for i in range(n):
        p = px * (1 + 0.001 * i)
        if split_at is not None:
            p = p / ratio   # yfinance serves ALL bars on today's (post-split) share basis
        px_list.append(p)
        splits.append(ratio if (split_at is not None and i == split_at) else 0.0)
    return pd.DataFrame({"Open": px_list, "High": [x * 1.01 for x in px_list], "Low": [x * 0.99 for x in px_list],
                         "Close": px_list, "Stock Splits": splits}, index=idx)


REC = {"ticker": "AAA", "scan_date": "2026-08-03", "holding_days": 20,
       "entry_price": 100.0, "target_price": 130.0, "stop_loss": 80.0}


def test_short_window_is_not_resolved():
    out = t._resolve_one(dict(REC), hist=_hist(14))
    assert out["resolved"] is False and "insufficient_bars" in out["resolve_error"]


def test_full_window_resolves_and_reports_net():
    out = t._resolve_one(dict(REC), hist=_hist(25))
    assert out["resolved"] and out["trading_days_data"] == 20 and out["resolver_version"] == 2
    assert out["net_final_return_pct"] == round(out["final_return_pct"] - t.MODEL_COST_PCT, 2)


def test_split_inside_window_is_not_a_fake_crash():
    out = t._resolve_one(dict(REC), hist=_hist(25, split_at=8, ratio=2.0))
    assert out["resolved"]
    assert out["split_factor"] == 2.0
    assert abs(out["final_return_pct"]) < 5, "a 2:1 split must not show up as -50%"
    assert out["outcome"] == "time_expired"
