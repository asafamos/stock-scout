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

from core.data_sources_v2 import get_index_series, get_last_index_source

# Global cache for per-date market breadth
_MARKET_BREADTH_CACHE = {}
_GLOBAL_INDEX_CACHE: dict[str, pd.DataFrame] = {}
_MARKET_CONTEXT_INITIALIZED: bool = False

def initialize_market_context(symbols: Optional[list[str]] = None, period_days: int = 180) -> None:
    """Prefetch and cache global market series (SPY + sector ETFs).

    Fetches once at process start to avoid per-stock rate limits.
    Strictly uses primary provider via get_index_series.
    """
    default_symbols = ["SPY", "^VIX", "XLK", "XLV", "XLF", "XLY", "XLP", "XLI", "XLU", "XLE", "XLB", "XLRE"]
    # Avoid re-initialization to preserve cached results and reduce rate limits
    global _MARKET_CONTEXT_INITIALIZED
    if _MARKET_CONTEXT_INITIALIZED and all(sym in _GLOBAL_INDEX_CACHE for sym in ["SPY", "^VIX"]):
        return
    syms = symbols or default_symbols
    end_dt = datetime.utcnow().strftime('%Y-%m-%d')
    start_dt = (datetime.utcnow() - timedelta(days=period_days)).strftime('%Y-%m-%d')
    required = {"SPY", "^VIX"}
    for sym in syms:
        try:
            df = get_index_series(sym, start_dt, end_dt)
            if df is None or df.empty:
                if sym in required:
                    raise RuntimeError(f"Missing global series for {sym}")
                else:
                    # Non-critical (sector ETF) — continue; breadth will fallback to yfinance
                    continue
            _GLOBAL_INDEX_CACHE[sym] = df.copy()
        except Exception as e:
            if sym in required:
                raise RuntimeError(f"initialize_market_context failed for {sym}: {e}")
            # Non-critical failure, proceed
            continue
    _MARKET_CONTEXT_INITIALIZED = True


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

    # Prefer global cache if available
    if symbol in _GLOBAL_INDEX_CACHE:
        return _GLOBAL_INDEX_CACHE[symbol].copy()

    # Strict fetch via primary provider
    days = _period_to_days(period)
    now = datetime.utcnow()
    start = (now - timedelta(days=days + 5)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    df = get_index_series(symbol, start, end)
    if df is None or df.empty:
        raise RuntimeError(f"Benchmark series unavailable for {symbol}")
    return df


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
    
    # Compute proxy market breadth (fraction of sector ETFs above MA20)
    market_breadth = get_market_breadth()

    features = {
        'market_trend': spy_context.get('market_trend', 0.5),
        'market_volatility': spy_context.get('market_volatility', 0.2),
        'spy_rsi': spy_context.get('spy_rsi', 50.0),
        'relative_strength_20d': rel_strength,
        'sector_momentum': compute_sector_momentum(ticker),
        'market_cap_decile': get_market_cap_decile(ticker),
        'dist_from_52w_high': compute_price_distance_from_52w_high(ticker_df),
        'market_breadth': float(market_breadth),
    }
    
    return features


def get_market_breadth(date_str: Optional[str] = None, etfs: Optional[list[str]] = None) -> float:
    """Compute proxy market breadth using sector ETFs, cached by date.

    Returns fraction in [0,1] of ETFs currently above their 20-day MA.

    Args:
        date_str: Optional date key (YYYY-MM-DD). If provided and in cache, returns cached value.
        etfs: Optional list of sector ETF symbols; defaults to SPDR sectors.

    Behavior:
        - Tries multi-source index fetch; if unavailable, falls back to yfinance directly.
        - Uses period="1mo" to minimize data transfer while enabling MA20.
        - Stores result in a global per-date cache to avoid repeated computation.
    """
    default_etfs = ['XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLI', 'XLU', 'XLE', 'XLB', 'XLRE']
    symbols = etfs or default_etfs

    # Normalize date key
    if date_str is None:
        date_key = datetime.utcnow().strftime('%Y-%m-%d')
    else:
        try:
            date_key = pd.to_datetime(date_str).strftime('%Y-%m-%d')
        except Exception:
            date_key = str(date_str)

    # Cache check
    cached = _MARKET_BREADTH_CACHE.get(date_key)
    if cached is not None and np.isfinite(cached):
        return float(cached)

    counts = 0
    above = 0
    for sym in symbols:
        try:
            # Use global cache or strict provider fetch
            end_dt = datetime.utcnow().strftime('%Y-%m-%d')
            start_dt = (datetime.utcnow() - timedelta(days=35)).strftime('%Y-%m-%d')
            df = _GLOBAL_INDEX_CACHE.get(sym) or get_index_series(sym, start_dt, end_dt)
            # Fallback to yfinance for sector ETFs if providers unavailable
            if df is None or df.empty or 'close' not in df.columns:
                try:
                    yf_df = yf.Ticker(sym).history(period='2mo')
                    if yf_df is not None and not yf_df.empty:
                        yf_df = yf_df.reset_index()
                        yf_df = yf_df.rename(columns={
                            'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                        })
                        df = yf_df[['date', 'close']].copy()
                    else:
                        raise RuntimeError(f"Breadth missing for {sym}")
                except Exception:
                    raise RuntimeError(f"Breadth missing for {sym}")
            close = df['close']

            ma20 = close.rolling(20).mean()
            if len(close) < 20 or pd.isna(ma20.iloc[-1]):
                continue
            counts += 1
            if float(close.iloc[-1]) > float(ma20.iloc[-1]):
                above += 1
        except Exception:
            continue

    if counts == 0:
        raise RuntimeError("Market breadth unavailable (no valid sector data)")
    breadth = float(above / counts)
    _MARKET_BREADTH_CACHE[date_key] = breadth
    return breadth
