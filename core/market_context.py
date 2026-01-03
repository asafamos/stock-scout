"""
Market context features for better ML predictions.
Captures regime, sentiment, and relative strength.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import functools
from typing import Optional

from core.data_sources_v2 import get_index_series


@functools.lru_cache(maxsize=8)
def get_benchmark_series(symbol: str = "SPY", period: str = "6mo") -> pd.DataFrame:
    """
    Canonical benchmark fetcher with caching.
    - Tries multi-source provider via core.data_sources_v2.get_index_series
      using a derived start/end from `period`.
    - Falls back to yfinance if providers unavailable.
    - Returns DataFrame with columns: ['date','open','high','low','close','volume'].
    Caches results to avoid redundant calls and rate limits.
    """
    def _period_to_days(p: str) -> int:
        try:
            if p.endswith("d"):
                return int(p[:-1])
            if p.endswith("mo"):
                return int(p[:-2]) * 30
            if p.endswith("y"):
                return int(p[:-1]) * 365
        except Exception:
            pass
        return 180

    # Attempt multi-source first
    try:
        days = _period_to_days(period)
        now = datetime.utcnow()
        start = (now - timedelta(days=days + 5)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        df = get_index_series(symbol, start, end)
        if df is not None and len(df) > 0:
            return df
    except Exception:
        pass

    # Fallback: yfinance
    try:
        df = yf.Ticker(symbol).history(period=period)
        if not df.empty:
            out = (
                df.reset_index()
                  .rename(columns={
                      "Date": "date",
                      "Open": "open",
                      "High": "high",
                      "Low": "low",
                      "Close": "close",
                      "Volume": "volume",
                  })[["date", "open", "high", "low", "close", "volume"]]
            )
            return out
    except Exception:
        pass

    # Empty safe default
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def fetch_spy_context(lookback_days: int = 90) -> dict:
    """Fetch SPY market context features using cached benchmark series."""
    try:
        period = f"{lookback_days + 60}d"
        df = get_benchmark_series("SPY", period=period)
        if len(df) < 20:
            return {}

        close = df["close"] if "close" in df.columns else df["Close"]
        # Market trend
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
        current = close.iloc[-1]

        # Volatility regime
        returns = close.pct_change()
        vol_20d = returns.tail(20).std() * np.sqrt(252)

        return {
            'market_trend': 1.0 if current > sma20 > sma50 else 0.0,
            'market_volatility': float(min(vol_20d, 0.5)),  # Cap at 50%
            'spy_rsi': compute_simple_rsi(close, 14),
        }
    except Exception:
        return {}


def compute_simple_rsi(series: pd.Series, period: int = 14) -> float:
    """Quick RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).tail(period).mean()
    loss = -delta.where(delta < 0, 0.0).tail(period).mean()
    
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_relative_strength_vs_spy(ticker_df: pd.DataFrame, spy_df: pd.DataFrame, period: int = 20) -> float:
    """
    Relative strength = stock outperformance vs SPY.
    Positive = beating market = good.
    """
    if len(ticker_df) < period or len(spy_df) < period:
        return 0.0
    
    # Align dates
    ticker_ret = (ticker_df['Close'].iloc[-1] / ticker_df['Close'].iloc[-period]) - 1
    spy_ret = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-period]) - 1
    
    return ticker_ret - spy_ret


def compute_sector_momentum(ticker: str) -> float:
    """
    Sector momentum proxy: compare to sector ETF.
    Returns percentile (0-1) of sector performance.
    
    Simplified: just return 0.5 for now (can enhance later with sector mapping).
    """
    # TODO: Map ticker → sector → sector ETF → compute relative strength
    # For now return neutral
    return 0.5


def get_market_cap_decile(ticker: str) -> int:
    """
    Market cap bucket (1-10).
    1 = mega cap, 10 = small cap.
    Larger caps often more stable.
    """
    try:
        info = yf.Ticker(ticker).info
        market_cap = info.get('marketCap', 0)
        
        if market_cap > 200e9:
            return 1  # Mega cap
        elif market_cap > 50e9:
            return 2  # Large cap
        elif market_cap > 10e9:
            return 3  # Mid cap
        elif market_cap > 2e9:
            return 5  # Small cap
        else:
            return 8  # Micro cap
    except:
        return 5  # Default to mid


def compute_price_distance_from_52w_high(df: pd.DataFrame) -> float:
    """
    Distance from 52-week high.
    -0.5 = 50% below high (deeply beaten), 0.0 = at high.
    """
    if len(df) < 252:
        high_52w = df['High'].max()
    else:
        high_52w = df['High'].tail(252).max()
    
    current = df['Close'].iloc[-1]
    return (current / high_52w) - 1


def engineer_context_features(ticker: str, ticker_df: pd.DataFrame) -> dict:
    """
    Compute all market context features.
    These add information ML model doesn't have from technicals alone.
    """
    spy_context = fetch_spy_context()

    # Fetch SPY data for relative strength via cached benchmark
    try:
        spy_df = get_benchmark_series("SPY", period='3mo')
        # Map columns to expected yfinance-like capitalization for compatibility
        spy_df_cap = spy_df.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        rel_strength = compute_relative_strength_vs_spy(ticker_df, spy_df_cap)
    except Exception:
        rel_strength = 0.0
    
    features = {
        'market_trend': spy_context.get('market_trend', 0.5),
        'market_volatility': spy_context.get('market_volatility', 0.2),
        'spy_rsi': spy_context.get('spy_rsi', 50.0),
        'relative_strength_20d': rel_strength,
        'sector_momentum': compute_sector_momentum(ticker),
        'market_cap_decile': get_market_cap_decile(ticker),
        'dist_from_52w_high': compute_price_distance_from_52w_high(ticker_df),
    }
    
    return features
