import types
import requests
import pytest

from scoring import _normalize_weights, allocate_budget
from stock_scout import http_get_retry


def test_normalize_weights_zero_sum():
    w = {"ma": 0, "mom": 0}
    nw = _normalize_weights(w)
    # when all input weights are zero, normalized weights should be all zeros
    assert abs(sum(nw.values()) - 0.0) < 1e-9


def test_allocate_budget_zero_total():
    import pandas as pd

    df = pd.DataFrame({"Ticker": ["A"], "Score": [10], "Unit_Price": [10]})
    out = allocate_budget(df, total=0, min_pos=100, max_pos_pct=50)
    assert out["סכום קנייה ($)"].sum() == 0


def test_http_get_retry_timeout(monkeypatch):
    class FakeExc(Exception):
        pass

    def fake_get(url, timeout, headers=None):
        raise requests.RequestException("fail")

    monkeypatch.setattr(requests, "get", fake_get)
    r = http_get_retry("http://example.invalid", tries=2, timeout=0.01)
    assert r is None
