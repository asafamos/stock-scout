"""get_fundamentals_safe without FMP (2026-11-18): Finnhub+yfinance fallback, no Alpha pacer, FMP circuit, unit-consistent Finnhub."""
import time

from core import data_sources_v2 as ds


class _Resp:
    def __init__(self, code): self.status_code = code
    def json(self): return {}


def test_fallback_used_when_fmp_429_and_alpha_never_called(monkeypatch):
    monkeypatch.setattr(ds, "_get_from_cache", lambda k: None)
    monkeypatch.setattr(ds, "_put_in_cache", lambda k, v: None)
    monkeypatch.setattr(ds, "_rate_limit", lambda s: (_ for _ in ()).throw(AssertionError(f"rate limiter used for {s}")) if s == "alpha" else None)
    monkeypatch.setattr(ds.requests, "get", lambda *a, **k: _Resp(429))
    monkeypatch.setattr(ds, "FMP_API_KEY", "k")
    monkeypatch.setattr(ds, "fetch_fundamentals_finnhub", lambda t, provider_status=None: {"pe": 20.0, "pb": 5.0, "roe": 0.5, "beta": 1.2, "market_cap": 1e9, "debt_equity": 1.0})
    monkeypatch.setattr(ds, "fetch_fundamentals_yfinance", lambda t, provider_status=None: {"pe": 21.0, "sector": "Industrials", "industry": "Machinery", "vol_avg": 3e6, "peg": 1.1, "price_backup": 50.0, "last_div": 0.01})
    ds._FMP_SAFE_BLOCKED_UNTIL[0] = 0.0
    out = ds.get_fundamentals_safe("CAT")
    assert out["Sector"] == "Industrials" and out["PE_Ratio"] == 20.0 and out["Beta"] == 1.2 and out["Vol_Avg"] == 3e6
    assert ds._FMP_SAFE_BLOCKED_UNTIL[0] > time.time() + 500          # circuit opened for ~10 min


def test_circuit_skips_fmp_calls(monkeypatch):
    monkeypatch.setattr(ds, "_get_from_cache", lambda k: None)
    monkeypatch.setattr(ds, "_put_in_cache", lambda k, v: None)
    monkeypatch.setattr(ds, "FMP_API_KEY", "k")
    monkeypatch.setattr(ds.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("FMP must not be called")))
    monkeypatch.setattr(ds, "fetch_fundamentals_finnhub", lambda t, provider_status=None: {"pe": 10.0})
    monkeypatch.setattr(ds, "fetch_fundamentals_yfinance", lambda t, provider_status=None: None)
    ds._FMP_SAFE_BLOCKED_UNTIL[0] = time.time() + 300
    try:
        assert ds.get_fundamentals_safe("X") == {"PE_Ratio": 10.0}
    finally:
        ds._FMP_SAFE_BLOCKED_UNTIL[0] = 0.0


def test_finnhub_percent_fields_become_decimals(monkeypatch):
    monkeypatch.setattr(ds, "_get_from_cache", lambda k: None)
    monkeypatch.setattr(ds, "_put_in_cache", lambda k, v: None)
    monkeypatch.setattr(ds, "_rate_limit", lambda s: None)
    monkeypatch.setattr(ds, "FINNHUB_API_KEY", "k")
    metric = {"metric": {"peBasicExclExtraTTM": 21.9, "psAnnual": 3.8, "pbAnnual": 12.6, "roeTTM": 58.53,
                         "netProfitMarginTTM": 15.21, "marketCapitalization": 258925.0, "beta": 1.65,
                         "totalDebt/totalEquityAnnual": 2.03, "revenueGrowthTTMYoy": 20.26, "epsGrowthTTMYoy": 30.81}}
    monkeypatch.setattr(ds, "_http_get_with_retry", lambda *a, **k: metric)
    r = ds.fetch_fundamentals_finnhub("CAT")
    assert abs(r["roe"] - 0.5853) < 1e-9 and abs(r["margin"] - 0.1521) < 1e-9
    assert abs(r["rev_yoy"] - 0.2026) < 1e-9 and abs(r["eps_yoy"] - 0.3081) < 1e-9
    assert r["pe"] == 21.9 and r["debt_equity"] == 2.03             # ratios untouched


def test_unusable_fundamentals_providers_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FUND_DISABLED_DEFAULT", raising=False)
    ds.reset_disabled_providers()
    for p in ("alpha", "eodhd", "simfin"):
        assert f"{p}:fundamentals" in ds.DISABLED_PROVIDERS
    names = [n for n, _ in ds.get_prioritized_fetch_funcs(None)]
    assert "alpha" not in names and "simfin" not in names and "eodhd" not in names
    monkeypatch.setenv("FUND_DISABLED_DEFAULT", "")
    ds.reset_disabled_providers()
    assert "alpha:fundamentals" not in ds.DISABLED_PROVIDERS
    monkeypatch.delenv("FUND_DISABLED_DEFAULT", raising=False)
    ds.reset_disabled_providers()
