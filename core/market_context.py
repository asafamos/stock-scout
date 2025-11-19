"""
Market context features for better ML predictions.
Captures regime, sentiment, and relative strength.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_spy_context(lookback_days: int = 90) -> dict:
    """Fetch SPY market context features."""
    try:
        spy = yf.Ticker('SPY')
        end = datetime.now()
        start = end - timedelta(days=lookback_days + 30)
        df = spy.history(start=start, end=end)
        
        if len(df) < 20:
            return {}
        
        # Market trend
        sma20 = df['Close'].rolling(20).mean().iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma20
        current = df['Close'].iloc[-1]
        
        # Volatility regime
        returns = df['Close'].pct_change()
        vol_20d = returns.tail(20).std() * np.sqrt(252)
        
        return {
            'market_trend': 1.0 if current > sma20 > sma50 else 0.0,
            'market_volatility': min(vol_20d, 0.5),  # Cap at 50%
            'spy_rsi': compute_simple_rsi(df['Close'], 14),
        }
    except:
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
    
    # Fetch SPY data for relative strength
    try:
        spy = yf.Ticker('SPY')
        spy_df = spy.history(period='3mo')
        rel_strength = compute_relative_strength_vs_spy(ticker_df, spy_df)
    except:
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
