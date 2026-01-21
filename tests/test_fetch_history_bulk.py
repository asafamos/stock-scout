#!/usr/bin/env python
"""Test fetch_history_bulk from pipeline_runner"""

from core.pipeline_runner import fetch_history_bulk


def test_fetch_history_bulk_smoke():
    tickers = ["AAPL", "MSFT", "JPM"]
    period_days = 30
    ma_long = 50
    data_map = fetch_history_bulk(tickers, period_days, ma_long)
    assert isinstance(data_map, dict)
