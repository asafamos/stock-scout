"""
Timing filters to reduce early entries.
Helps identify when setup is RIPE vs just FORMING.
"""

import pandas as pd
import numpy as np


def check_volume_breakout(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    Volume breakout = confirmation signal.
    Today's volume > 1.5x average = institutions entering.
    """
    if len(df) < lookback + 1:
        return False
    
    recent_vol = df['Volume'].iloc[-1]
    avg_vol = df['Volume'].iloc[-lookback-1:-1].mean()
    
    return recent_vol > (avg_vol * 1.5)


def check_price_breakout(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    Price breaking above recent resistance = timing signal.
    Close > 90th percentile of last 20 days.
    """
    if len(df) < lookback + 1:
        return False
    
    current_close = df['Close'].iloc[-1]
    recent_high = df['Close'].iloc[-lookback-1:-1].quantile(0.90)
    
    return current_close > recent_high


def check_momentum_acceleration(df: pd.DataFrame) -> bool:
    """
    Momentum INCREASING = better timing than flat momentum.
    Last 5 days momentum > previous 5 days.
    """
    if len(df) < 11:
        return False
    
    # Compare recent vs previous 5-day returns
    recent_ret = (df['Close'].iloc[-1] / df['Close'].iloc[-6]) - 1
    previous_ret = (df['Close'].iloc[-6] / df['Close'].iloc[-11]) - 1
    
    return recent_ret > previous_ret


def check_fresh_setup(df: pd.DataFrame, rsi_series: pd.Series) -> bool:
    """
    Fresh oversold = better timing.
    RSI was >40 recently (last 10 days), now <35 = fresh dip.
    Avoid stocks stuck in basement for weeks.
    """
    if len(rsi_series) < 11:
        return False
    
    current_rsi = rsi_series.iloc[-1]
    recent_max_rsi = rsi_series.iloc[-10:-1].max()
    
    # Fresh dip: was healthy recently, now oversold
    return current_rsi < 35 and recent_max_rsi > 40


def compute_timing_score(df: pd.DataFrame, rsi_series: pd.Series) -> dict:
    """
    Compute timing readiness score (0-100).
    Higher = better entry timing RIGHT NOW.
    
    Returns dict with score and breakdown.
    """
    signals = {
        'volume_breakout': check_volume_breakout(df),
        'price_breakout': check_price_breakout(df),
        'momentum_acceleration': check_momentum_acceleration(df),
        'fresh_setup': check_fresh_setup(df, rsi_series),
    }
    
    # Weight the signals
    weights = {
        'volume_breakout': 40,      # Most important = institutions entering
        'price_breakout': 30,        # Breaking resistance
        'momentum_acceleration': 20, # Trend strengthening
        'fresh_setup': 10,          # Not dead money
    }
    
    score = sum(weights[k] for k, v in signals.items() if v)
    
    return {
        'timing_score': score,
        'signals': signals,
        'is_ready': score >= 50,  # Need at least 2 major signals
    }
