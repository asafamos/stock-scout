"""
Fundamentals data mode and backtest-awareness helpers.

DataMode distinguishes between LIVE and BACKTEST contexts.
- In BACKTEST mode, fundamentals providers typically return only current
  snapshot (point-in-time support is rare). Using snapshot fundamentals for
  historical evaluation can introduce lookahead bias.
- Callers should pass DataMode.BACKTEST to fundamentals fetchers during
  backtests. The aggregation layer will log a warning and mark results with
  `Fundamental_Backtest_Unsafe=True` so downstream can surface this risk.
"""
from __future__ import annotations
from enum import Enum


class DataMode(str, Enum):
    LIVE = "live"
    BACKTEST = "backtest"
