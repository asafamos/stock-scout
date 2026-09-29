"""Real-time pre-buy price from the data providers we already pay for.

2026-09-29: the pre-buy price refresh used ONLY IB's free delayed feed
(reqMarketDataType(3), ~15 min lag — no IB market-data subscription). On a
rising stock the marketable limit (delayed price + 0.30%) sat BELOW the real
market and never filled: PBF live was $75.70 (FMP/Finnhub, age <1 min) while
the IB-delayed reference was $75.33 → two unfilled orders. It also made the 3%
slippage guard look weaker than designed (PBF +2.6% delayed vs +3.2% real).

Sources, in order: FMP, Finnhub. Each quote must carry a provider timestamp no
older than MAX_AGE_SEC, else it is rejected (pre/post-market or a stale symbol
would otherwise look "live"). When BOTH return a fresh quote they are
cross-checked: if they disagree by more than MAX_DISAGREE_PCT the trade is
skipped (QuoteDisagreement) rather than priced off a doubtful number; when they
agree the mean is used. With one source available it is used alone; with none
the caller falls back to IB delayed.

Kill switch: TRADE_REALTIME_QUOTE_ENABLED=0.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MAX_AGE_SEC = int(os.getenv("TRADE_REALTIME_QUOTE_MAX_AGE_SEC", "180"))
MAX_DISAGREE_PCT = float(os.getenv("TRADE_REALTIME_QUOTE_MAX_DISAGREE_PCT", "0.5"))


class QuoteDisagreement(Exception):
    """Two fresh real-time sources disagree beyond tolerance — do not trade off either."""


def _enabled() -> bool:
    return os.getenv("TRADE_REALTIME_QUOTE_ENABLED", "1").strip() not in ("0", "false", "False", "no")


def _get_json(url: str, timeout: float = 6.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _fmp(ticker: str) -> Optional[float]:
    key = os.getenv("FMP_API_KEY")
    if not key:
        return None
    sym = urllib.parse.quote(ticker.replace(".", "-"))
    d = _get_json(f"https://financialmodelingprep.com/stable/quote?symbol={sym}&apikey={key}")
    x = d[0] if isinstance(d, list) and d else None
    if not isinstance(x, dict):
        return None
    price, ts = x.get("price"), x.get("timestamp")
    if not price or not ts or time.time() - float(ts) > MAX_AGE_SEC:
        return None
    return float(price)


def _finnhub(ticker: str) -> Optional[float]:
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        return None
    d = _get_json(f"https://finnhub.io/api/v1/quote?symbol={urllib.parse.quote(ticker)}&token={key}")
    price, ts = d.get("c"), d.get("t")
    if not price or not ts or time.time() - float(ts) > MAX_AGE_SEC:
        return None
    return float(price)


def get_realtime_price(ticker: str) -> Optional[Tuple[float, str]]:
    """Return (price, source) from fresh real-time quotes, or None if none available.

    Raises QuoteDisagreement when two fresh sources differ by > MAX_DISAGREE_PCT.
    """
    if not _enabled():
        return None
    fresh = {}
    for name, fn in (("FMP", _fmp), ("Finnhub", _finnhub)):
        try:
            p = fn(ticker)
            if p and p > 0:
                fresh[name] = p
        except Exception as e:  # network / auth / parse — the other source may still work
            logger.warning("realtime quote %s failed for %s: %s", name, ticker, e)
    if not fresh:
        return None
    if len(fresh) == 1:
        (name, p), = fresh.items()
        return p, name
    lo, hi = min(fresh.values()), max(fresh.values())
    spread_pct = (hi - lo) / lo * 100
    if spread_pct > MAX_DISAGREE_PCT:
        raise QuoteDisagreement(
            f"{ticker}: " + ", ".join(f"{n} ${v:.2f}" for n, v in fresh.items())
            + f" differ by {spread_pct:.2f}% (> {MAX_DISAGREE_PCT}%)"
        )
    return sum(fresh.values()) / len(fresh), "+".join(fresh)
