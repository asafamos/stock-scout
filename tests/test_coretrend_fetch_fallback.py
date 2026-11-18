"""CoreTrend price fetch (2026-11-18): FMP first, yfinance when FMP fails — the 11-16/11-17 runs crashed on 429."""
from datetime import date

import pytest

from scripts import coretrend_paper as cp


def test_uses_fmp_when_it_works(monkeypatch):
    monkeypatch.setattr(cp, "_fetch_daily_fmp", lambda s, st: [(date(2026, 11, 17), 650.0)])
    monkeypatch.setattr(cp, "_fetch_daily_yfinance", lambda s, st: (_ for _ in ()).throw(AssertionError("not needed")))
    assert cp.fetch_daily("QQQ") == [(date(2026, 11, 17), 650.0)]


def test_falls_back_to_yfinance_on_429(monkeypatch):
    def boom(s, st):
        raise RuntimeError("HTTP 429")
    monkeypatch.setattr(cp, "_fetch_daily_fmp", boom)
    monkeypatch.setattr(cp, "_fetch_daily_yfinance", lambda s, st: [(date(2026, 11, 17), 656.95)])
    assert cp.fetch_daily("QQQ")[-1] == (date(2026, 11, 17), 656.95)


def test_empty_fmp_answer_also_falls_back(monkeypatch):
    monkeypatch.setattr(cp, "_fetch_daily_fmp", lambda s, st: [])
    monkeypatch.setattr(cp, "_fetch_daily_yfinance", lambda s, st: [(date(2026, 11, 17), 1.0)])
    assert cp.fetch_daily("IEF") == [(date(2026, 11, 17), 1.0)]


def test_raises_when_both_sources_fail(monkeypatch):
    monkeypatch.setattr(cp, "_fetch_daily_fmp", lambda s, st: [])
    monkeypatch.setattr(cp, "_fetch_daily_yfinance", lambda s, st: [])
    with pytest.raises(RuntimeError):
        cp.fetch_daily("SPY")
