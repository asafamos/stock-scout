"""Measure which data providers are actually usable and how fast (2026-11-15).

    FUND_CACHE=0 python -m scripts.provider_benchmark [--n 12]

For a fixed sample of tickers it times every provider per capability and reports usable-rate and median
latency, so the routing order / disable rules are set from data, not assumptions.
Capabilities: fundamentals (each fetch_fundamentals_*), multi-source price, next-earnings date, index series.
Never prints keys.
"""
from __future__ import annotations

import argparse
import logging
import statistics
import time

logging.disable(logging.CRITICAL)

SAMPLE = ["AAPL", "MSFT", "JNJ", "XOM", "CAT", "HD", "PG", "KO", "DIS", "BA", "NKE", "MCD", "V", "WMT", "CVX",
          "IBM", "INTC", "CSCO", "PFE", "MRK", "SEZL", "PATH", "HNGE", "PACS"]
KEYS = ("pe", "ps", "pb", "roe", "margin", "market_cap", "beta")


def _row(name, lat, ok, n):
    med = statistics.median(lat) if lat else 0
    mx = max(lat) if lat else 0
    print(f"  {name:<14} usable {ok:>2}/{n:<2}  median {med:5.2f}s  max {mx:5.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    tick = SAMPLE[: a.n]
    from core import data_sources_v2 as d

    print(f"Sample: {len(tick)} tickers\n\nFUNDAMENTALS (per provider, called alone)")
    for name in ("fmp", "finnhub", "tiingo", "alpha", "eodhd", "simfin", "yfinance"):
        f = getattr(d, f"fetch_fundamentals_{name}")
        lat, ok = [], 0
        for t in tick:
            t0 = time.time()
            try:
                r = f(t)
                ok += bool(r) and any(r.get(k) is not None for k in KEYS)
            except Exception:
                pass
            lat.append(time.time() - t0)
        _row(name, lat, ok, len(tick))

    print("\nPRICE (fetch_price_multi_source: all sources per call)")
    lat, srcs = [], {}
    for t in tick:
        t0 = time.time()
        try:
            r = d.fetch_price_multi_source(t) or {}
        except Exception:
            r = {}
        lat.append(time.time() - t0)
        for k, v in r.items():
            if v:
                srcs[k] = srcs.get(k, 0) + 1
    _row("multi-price", lat, sum(1 for _ in lat), len(tick))
    print("  sources that returned a price:", dict(sorted(srcs.items(), key=lambda kv: -kv[1])))

    print("\nNEXT EARNINGS DATE")
    lat, ok = [], 0
    for t in tick:
        t0 = time.time()
        try:
            ok += bool(d.get_next_earnings_date(t))
        except Exception:
            pass
        lat.append(time.time() - t0)
    _row("earnings", lat, ok, len(tick))

    print("\nINDEX / ETF SERIES (sector ETFs the scan needs each run)")
    for sym in ("SPY", "XLK", "XLF", "XLE", "^VIX"):
        t0 = time.time()
        try:
            s = d.get_index_series(sym, "2026-08-01", "2026-11-13")
            n = 0 if s is None else len(s)
        except Exception:
            n = -1
        print(f"  {sym:<6} rows {n:>4}  {time.time() - t0:5.2f}s  source={d.get_last_index_source(sym)}")


if __name__ == "__main__":
    main()
