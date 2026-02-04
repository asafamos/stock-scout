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
import logging

from core.data_sources_v2 import get_index_series, get_last_index_source
from core.config import get_secret
import requests

logger = logging.getLogger(__name__)

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
            # Prefer Tiingo for sector ETFs (XL*) first; if it fails, try Polygon after a short delay
            df = None
            if sym.upper().startswith("XL"):
                tiingo_key = get_secret("TIINGO_API_KEY", "")
                if tiingo_key:
                    try:
                        tiingo_symbol = sym.replace('^', '$')
                        url = f"https://api.tiingo.com/tiingo/daily/{tiingo_symbol}/prices"
                        headers = {"Content-Type": "application/json", "Authorization": f"Token {tiingo_key}"}
                        params = {"startDate": start_dt, "endDate": end_dt}
                        resp = requests.get(url, params=params, headers=headers, timeout=4)
                        if resp.status_code == 200:
                            data = resp.json()
                            if isinstance(data, list) and len(data) > 0:
                                tdf = pd.DataFrame(data)
                                tdf["date"] = pd.to_datetime(tdf["date"]) if "date" in tdf.columns else pd.to_datetime(tdf.index)
                                tdf = tdf.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
                                cols = ["date", "open", "high", "low", "close", "volume"]
                                if all(c in tdf.columns for c in cols):
                                    df = tdf[cols].sort_values("date").reset_index(drop=True)
                        else:
                            df = None
                    except Exception:
                        df = None
                # If Tiingo failed, try Polygon with a 2-second delay and FMP disabled to avoid delays
                if df is None:
                    try:
                        import time as _t
                        _t.sleep(2)
                        df = get_index_series(sym, start_dt, end_dt, provider_status={"fmp": False, "polygon": True, "tiingo": True, "alpha": True})
                    except Exception:
                        df = None
            if df is None:
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
    Dual-Phase Relative Strength vs SPY.

    Returns a single float representing a weighted blend of:
    - Standard 20d RS: stock 20-day return minus SPY 20-day return
    - Resilience RS: if SPY's cumulative 10-day return is negative, blend in the
      stock's compounded return on the specific SPY down-days (last 10 trading days)
      with a 1.5x multiplier

    NaN/length safety: returns 0.0 if inputs are insufficient.
    """
    try:
        # Normalize close column casing ('Close' preferred, fallback to 'close')
        def _get_close(df: pd.DataFrame) -> Optional[pd.Series]:
            if df is None or df.empty:
                return None
            if 'Close' in df.columns:
                return df['Close']
            if 'close' in df.columns:
                return df['close']
            return None

        t_close = _get_close(ticker_df)
        s_close = _get_close(spy_df)
        if t_close is None or s_close is None:
            return 0.0

        if len(t_close) < period or len(s_close) < period:
            return 0.0

        # Standard 20d returns and RS
        t0, t20 = float(t_close.iloc[-1]), float(t_close.iloc[-period])
        s0, s20 = float(s_close.iloc[-1]), float(s_close.iloc[-period])
        if not np.isfinite(t0) or not np.isfinite(t20) or t20 == 0:
            return 0.0
        if not np.isfinite(s0) or not np.isfinite(s20) or s20 == 0:
            return 0.0

        stock_20d_ret = (t0 / t20) - 1.0
        spy_20d_ret = (s0 / s20) - 1.0
        standard_rs = float(stock_20d_ret - spy_20d_ret)

        # Resilience RS: only applies if SPY 10-day cumulative return is negative
        resilience_rs = 0.0

        # Build aligned daily return frame over recent window (last ~12 days for pct_change)
        # Use 'Date' column if present; else derive from index
        def _with_date_close(df: pd.DataFrame) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return None
            date_col = None
            if 'Date' in df.columns:
                date_col = 'Date'
            elif 'date' in df.columns:
                date_col = 'date'
            else:
                # try index as date
                try:
                    idx = pd.to_datetime(df.index)
                    tmp = pd.DataFrame({'Date': idx})
                    # attach close series with normalized casing
                    close_series = _get_close(df)
                    if close_series is None:
                        return None
                    tmp['Close'] = close_series.values
                    return tmp[['Date', 'Close']]
                except Exception:
                    return None
            # build from explicit column
            close_series = _get_close(df)
            if close_series is None:
                return None
            return df[[date_col]].rename(columns={date_col: 'Date'}).assign(Close=close_series.values)

        t_df = _with_date_close(ticker_df)
        s_df = _with_date_close(spy_df)
        if t_df is None or s_df is None or t_df.empty or s_df.empty:
            return standard_rs

        # Keep only the last 12 rows to compute recent daily returns robustly
        t_df = t_df.dropna().copy()
        s_df = s_df.dropna().copy()
        t_df = t_df.sort_values('Date').tail(12)
        s_df = s_df.sort_values('Date').tail(12)

        merged = pd.merge(t_df, s_df, on='Date', suffixes=('_stock', '_spy'))
        if len(merged) < 2:
            return standard_rs

        # Daily returns
        merged['ret_stock'] = merged['Close_stock'].pct_change()
        merged['ret_spy'] = merged['Close_spy'].pct_change()
        recent = merged.tail(10)

        # Determine whether SPY 10-day cumulative return is negative
        try:
            spy_10d_cum = (float(recent['Close_spy'].iloc[-1]) / float(recent['Close_spy'].iloc[0])) - 1.0
        except Exception:
            spy_10d_cum = float(recent['ret_spy'].add(1.0).prod() - 1.0)

        if np.isfinite(spy_10d_cum) and spy_10d_cum < 0:
            down_mask = recent['ret_spy'] < 0
            stock_on_down = recent.loc[down_mask, 'ret_stock'].dropna()
            if len(stock_on_down) > 0:
                # compounded return on SPY down-days
                resilience_rs = float(stock_on_down.add(1.0).prod() - 1.0)

        # Increase resilience emphasis in down markets (2.5x multiplier)
        final_rs = float(standard_rs + 2.5 * resilience_rs)
        # NaN safety
        if not np.isfinite(final_rs):
            return 0.0
        return final_rs
    except Exception:
        return 0.0


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
    except Exception as e:
        logger.debug(f"Failed to get market cap for {ticker}: {e}")
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
