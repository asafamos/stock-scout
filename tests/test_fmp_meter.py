"""FMP meter (2026-11-15): endpoint normalisation (no key/query/symbol), call/byte/429 accounting, never raises."""
import json
import urllib.error

import pytest

from core import fmp_meter as fm


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_METER_FILE", str(tmp_path / "u.jsonl"))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    fm._by_ep.clear()
    yield
    fm._by_ep.clear()


def test_endpoint_strips_query_key_and_symbol():
    assert fm.endpoint_of("https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey=SECRET") == "/stable/profile"
    assert fm.endpoint_of("https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?apikey=SECRET") == "/api/v3/historical-price-full"
    assert fm.endpoint_of("https://financialmodelingprep.com/api/v4/price-target?symbol=X") == "/api/v4/price-target"


def test_record_counts_classes_and_bytes():
    u = "https://financialmodelingprep.com/stable/quote?symbol=A&apikey=SECRET"
    fm.record(u, 200, 1000)
    fm.record(u, 429, 0)
    fm.record(u, 500, 0)
    fm.record(u, None, 0)
    s = fm.snapshot()
    d = s["by_endpoint"]["/stable/quote"]
    assert (d["calls"], d["ok"], d["e429"], d["e_other"], d["bytes"]) == (4, 1, 1, 2, 1000)
    assert "SECRET" not in json.dumps(s)


def test_flush_writes_jsonl_and_resets(tmp_path):
    fm.record("https://financialmodelingprep.com/stable/profile?symbol=A", 200, 50)
    fm.flush()
    rows = (tmp_path / "u.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["calls"] == 1
    assert fm.snapshot()["calls"] == 0


def test_urllib_wrapper_counts_ok_and_http_error(monkeypatch):
    import urllib.request

    class FakeResp:
        status = 200
        def read(self, *a): return b"x" * 123
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def close(self): pass

    def fake_open(url, *a, **k):
        if "bad" in str(url):
            raise urllib.error.HTTPError(str(url), 429, "Too Many", {}, None)
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    monkeypatch.setattr(fm, "_installed", False)
    fm._wrap_urllib()
    with urllib.request.urlopen("https://financialmodelingprep.com/stable/profile?symbol=A&apikey=K") as r:
        assert r.read() == b"x" * 123
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen("https://financialmodelingprep.com/stable/profile?bad=1&apikey=K")
    urllib.request.urlopen("https://example.com/other")          # not FMP -> not counted
    d = fm.snapshot()["by_endpoint"]["/stable/profile"]
    assert (d["calls"], d["ok"], d["e429"], d["bytes"]) == (2, 1, 1, 123)
