"""Root cause of the synthetic VIX in scans (2026-10-06): the FMP symbol lost its caret and the /light endpoint (no OHLC) was rejected."""
import pandas as pd
import core.data_sources_v2 as ds


class _Resp:
    def __init__(self, data, code=200): self._d, self.status_code = data, code
    def json(self): return self._d


def test_vix_is_fetched_from_fmp_with_caret_and_full_endpoint(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None, **kw):
        seen["url"] = url
        if "%5EVIX" in url and "historical-price-eod/full" in url:
            return _Resp([{"symbol": "^VIX", "date": "2026-10-05", "open": 15.9, "high": 16.0, "low": 15.3, "close": 15.52, "volume": 0},
                          {"symbol": "^VIX", "date": "2026-10-06", "open": 15.5, "high": 15.54, "low": 14.96, "close": 15.06, "volume": 0}])
        return _Resp([])                       # what FMP really returns for symbol=VIX (no caret)

    monkeypatch.setattr(ds, "FMP_API_KEY", "test-key")
    monkeypatch.setattr(ds.requests, "get", fake_get)
    monkeypatch.setattr(ds, "_rate_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(ds, "record_api_call", lambda *_a, **_k: None)
    monkeypatch.setattr(ds, "_get_from_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(ds, "_put_in_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(ds, "_PROVIDER_DISABLED", {})
    df = ds.get_index_series("^VIX", "2026-10-01", "2026-10-06")
    assert df is not None and float(df["close"].iloc[-1]) == 15.06                 # real VIX, not the 9.8 proxy
    assert ds.get_last_index_source("^VIX") == "FMP"
    assert "%5EVIX" in seen["url"] and "/full" in seen["url"]
    assert set(["open", "high", "low", "close", "volume"]) <= set(df.columns)
