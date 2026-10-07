"""P3-d: Yahoo's debtToEquity is a percent; the scoring layer expects a ratio."""
import sys, types
import core.data_sources_v2 as ds
from core.scoring.fundamental import compute_fundamental_score_with_breakdown


def _fake_yf(info):
    mod = types.ModuleType("yfinance")
    mod.Ticker = lambda t: types.SimpleNamespace(info=info)
    return mod


def test_yfinance_debt_to_equity_is_converted_to_a_ratio(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf({"trailingPE": 25.0, "debtToEquity": 78.445, "marketCap": 3e12}))
    monkeypatch.setattr(ds, "_get_from_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(ds, "_put_in_cache", lambda *_a, **_k: None)
    r = ds.fetch_fundamentals_yfinance("AAPL")
    assert abs(r["debt_equity"] - 0.78445) < 1e-9


def test_missing_debt_to_equity_stays_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf({"trailingPE": 25.0, "marketCap": 3e12}))
    monkeypatch.setattr(ds, "_get_from_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(ds, "_put_in_cache", lambda *_a, **_k: None)
    assert ds.fetch_fundamentals_yfinance("AAPL")["debt_equity"] is None


def test_score_penalises_unconverted_percent_but_not_the_ratio():
    base = {"pe": 18.0, "roe": 0.20, "margin": 0.18, "rev_yoy": 0.10}
    ratio = compute_fundamental_score_with_breakdown({**base, "debt_equity": 0.78}).total
    pct = compute_fundamental_score_with_breakdown({**base, "debt_equity": 78.4}).total          # what the bug fed
    assert ratio > pct + 3                                           # the bug cost several fundamental points per affected row
