"""Tickers the bot must NOT manage, count, alert on or learn from (2026-09-30).

The owner may hold a passive broad-market ETF in the same IB account (buy-and-hold, bought manually). The bot's
machinery assumes every IB position is one of ITS trades: the drift check would alert "UNTRACKED POSITION" every
cycle, the position counters / dedup would treat it as a trade slot, and its eventual sale would be scored as a
closed trade by the drawdown breaker / performance guard / daily-loss breaker. Tickers in this list are invisible to
all of that. The bot's own universe is single stocks, so ETFs here can never collide with a real bot position.

Override / extend with env TRADE_IGNORE_TICKERS (comma separated). Default: the common broad-market ETFs.
"""
from __future__ import annotations

import os

DEFAULT = "SPY,VOO,IVV,VTI,QQQ,QQQM,SCHB,ITOT,VT,IWM,DIA,VXUS,BND"


def ignored() -> set:
    raw = os.getenv("TRADE_IGNORE_TICKERS")
    raw = DEFAULT if raw is None else raw
    return {t.strip().upper() for t in raw.split(",") if t.strip()}


def is_ignored(ticker: str) -> bool:
    return str(ticker or "").upper() in ignored()
