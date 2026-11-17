"""Universe fallback (2026-11-17): when the FMP screener is unavailable, use the yfinance screener, not the S&P 500."""
import types

from core.pipeline import universe as u


def _fake_screen_factory(n):
    def screen(q, sortField=None, sortAsc=None, size=250, offset=0):
        quotes = [{"symbol": f"T{i}", "quoteType": "EQUITY", "marketCap": 1e9} for i in range(offset, min(offset + size, n))]
        return {"quotes": quotes}
    return screen


def test_yfinance_universe_used_when_fmp_down(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "screen", _fake_screen_factory(2200))
    monkeypatch.setattr(u, "get_secret", lambda k, d="": "")          # FMP path skipped
    out = u.fetch_top_us_tickers_by_market_cap(limit=2000)
    assert len(out) == 2000 and u.LAST_UNIVERSE_PROVIDER == "YFINANCE_SCREENER"


def test_partial_yfinance_answer_falls_through(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "screen", _fake_screen_factory(300))      # far too few for limit=2000
    monkeypatch.setattr(u, "get_secret", lambda k, d="": "")
    out = u.fetch_top_us_tickers_by_market_cap(limit=2000)
    assert u.LAST_UNIVERSE_PROVIDER != "YFINANCE_SCREENER" and out    # local S&P 500 (or later) fallback


def test_can_be_disabled(monkeypatch):
    import yfinance as yf

    monkeypatch.setenv("UNIVERSE_YFINANCE", "0")
    monkeypatch.setattr(yf, "screen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    monkeypatch.setattr(u, "get_secret", lambda k, d="": "")
    assert u.fetch_top_us_tickers_by_market_cap(limit=2000)
