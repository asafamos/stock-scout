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


def test_history2_flags_a_nan_last_close(monkeypatch, caplog):
    idx = pd.date_range("2025-10-01", periods=260, freq="B")
    good = pd.DataFrame({"Close": 10.0, "Volume": 1.0}, index=idx)
    bad = good.copy(); bad.iloc[-1, bad.columns.get_loc("Close")] = float("nan")     # a bar whose Close is missing survives dropna(how="all")
    frame = pd.concat({"AAA": good, "BBB": bad}, axis=1)
    monkeypatch.setattr(md.yf, "download", lambda *a, **k: frame)
    monkeypatch.setattr(md.time, "sleep", lambda *_: None)
    with caplog.at_level(logging.INFO):
        out = md.fetch_history_bulk(["AAA", "BBB"], 250, 200)
    assert set(out) == {"AAA", "BBB"}
    rec = [r for r in caplog.records if "[HISTORY2]" in r.getMessage()]
    assert rec and rec[0].levelno == logging.WARNING and "last_close_nan=1" in rec[0].getMessage()


def test_history2_is_quiet_when_everything_is_normal(monkeypatch, caplog):
    idx = pd.date_range("2025-10-01", periods=260, freq="B")
    good = pd.DataFrame({"Close": 10.0, "Volume": 1.0}, index=idx)
    frame = pd.concat({"AAA": good, "BBB": good}, axis=1)
    monkeypatch.setattr(md.yf, "download", lambda *a, **k: frame)
    monkeypatch.setattr(md.time, "sleep", lambda *_: None)
    with caplog.at_level(logging.INFO):
        md.fetch_history_bulk(["AAA", "BBB"], 250, 200)
    rec = [r for r in caplog.records if "[HISTORY2]" in r.getMessage()]
    assert rec and rec[0].levelno == logging.INFO
