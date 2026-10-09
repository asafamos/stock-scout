"""fetch_history_bulk must log a visible shortfall when most tickers come back empty (2026-10-09 overnight-scan diagnostics)."""
import logging
import pandas as pd
from core.pipeline import market_data as md


def test_shortfall_is_logged_as_warning(monkeypatch, caplog):
    monkeypatch.setattr(md.yf, "download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(md.time, "sleep", lambda *_: None)
    with caplog.at_level(logging.INFO):
        out = md.fetch_history_bulk(["AAA", "BBB", "CCC"], 250, 200)
    assert out == {}
    recs = [r for r in caplog.records if "[HISTORY]" in r.getMessage()]
    assert recs and recs[0].levelno == logging.WARNING
    assert "requested=3 usable=0" in recs[0].getMessage()
