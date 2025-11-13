"""
Advanced filtering and scoring mechanisms for stock selection.
Implements multi-layered quality checks and timing signals.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import yfinance as yf
from datetime import datetime, timedelta


def compute_relative_strength(
    ticker_df: pd.DataFrame, 
    benchmark_df: pd.DataFrame,
    periods: list[int] = [21, 63, 126]
) -> Dict[str, float]:
    """
    Calculate relative strength vs benchmark across multiple timeframes.
    Returns dict with RS scores for each period.
    """
    rs_scores = {}
    
    for period in periods:
        if len(ticker_df) < period or len(benchmark_df) < period:
            rs_scores[f"rs_{period}d"] = np.nan
            continue
            
        ticker_return = (ticker_df["Close"].iloc[-1] / ticker_df["Close"].iloc[-period] - 1)
        bench_return = (benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[-period] - 1)
        
        # Relative strength: positive means outperforming
        rs_scores[f"rs_{period}d"] = ticker_return - bench_return
    
    return rs_scores


def detect_volume_surge(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """
    Detect abnormal volume spikes indicating institutional interest.
    Returns surge ratio and price-volume correlation.
    """
    if len(df) < lookback + 5:
        return {"volume_surge": 0.0, "pv_correlation": 0.0}
    
    recent_vol = df["Volume"].tail(5).mean()
    avg_vol = df["Volume"].tail(lookback).mean()
    
    surge_ratio = recent_vol / avg_vol if avg_vol > 0 else 0.0
    
    # Price-volume correlation (positive = healthy)
    price_change = df["Close"].pct_change().tail(lookback)
    vol_change = df["Volume"].pct_change().tail(lookback)
    pv_corr = price_change.corr(vol_change) if len(price_change.dropna()) > 5 else 0.0
    
    return {
        "volume_surge": surge_ratio,
        "pv_correlation": pv_corr if np.isfinite(pv_corr) else 0.0
    }


def detect_consolidation(df: pd.DataFrame, short_period: int = 20, long_period: int = 50) -> float:
    """
    Detect price consolidation (volatility squeeze) - precursor to breakout.
    Returns ratio < 1.0 indicates tightening, potential energy building.
    """
    if len(df) < long_period:
        return np.nan
    
    # Calculate ATR for both periods
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    atr_short = tr.tail(short_period).mean()
    atr_long = tr.tail(long_period).mean()
    
    if atr_long == 0 or not np.isfinite(atr_long):
        return np.nan
    
    squeeze_ratio = atr_short / atr_long
    return squeeze_ratio


def check_ma_alignment(df: pd.DataFrame, periods: list[int] = [10, 20, 50, 200]) -> Dict[str, any]:
    """
    Check moving average alignment and trend strength.
    Perfect bullish: MA10 > MA20 > MA50 > MA200
    """
    if len(df) < max(periods):
        return {"ma_aligned": False, "alignment_score": 0.0, "trend_strength": 0.0}
    
    mas = {}
    for p in periods:
        mas[f"ma_{p}"] = df["Close"].rolling(p).mean().iloc[-1]
    
    # Check if properly ordered (bullish)
    ma_values = [mas[f"ma_{p}"] for p in periods]
    aligned = all(ma_values[i] > ma_values[i+1] for i in range(len(ma_values)-1))
    
    # Alignment score: how close to perfect order
    if all(np.isfinite(v) for v in ma_values):
        diffs = [ma_values[i] - ma_values[i+1] for i in range(len(ma_values)-1)]
        alignment_score = sum(1.0 if d > 0 else 0.0 for d in diffs) / len(diffs)
        
        # Trend strength: slope of longest MA
        ma_200_series = df["Close"].rolling(periods[-1]).mean().tail(20)
        trend_strength = np.polyfit(range(len(ma_200_series)), ma_200_series, 1)[0] if len(ma_200_series) >= 10 else 0.0
    else:
        alignment_score = 0.0
        trend_strength = 0.0
    
    return {
        "ma_aligned": aligned,
        "alignment_score": alignment_score,
        "trend_strength": trend_strength if np.isfinite(trend_strength) else 0.0
    }


def find_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """
    Identify key support and resistance levels using swing highs/lows.
    Returns distance to nearest levels.
    """
    if len(df) < window * 2:
        return {"distance_to_support": np.nan, "distance_to_resistance": np.nan}
    
    # Find local highs and lows
    highs = df["High"].rolling(window, center=True).max()
    lows = df["Low"].rolling(window, center=True).min()
    
    current_price = df["Close"].iloc[-1]
    
    # Recent swing levels
    recent_highs = highs.tail(window * 3).dropna().unique()
    recent_lows = lows.tail(window * 3).dropna().unique()
    
    # Find nearest support (below current price)
    supports = [l for l in recent_lows if l < current_price]
    nearest_support = max(supports) if supports else current_price * 0.9
    
    # Find nearest resistance (above current price)
    resistances = [h for h in recent_highs if h > current_price]
    nearest_resistance = min(resistances) if resistances else current_price * 1.1
    
    dist_support = (current_price - nearest_support) / current_price if nearest_support > 0 else np.nan
    dist_resistance = (nearest_resistance - current_price) / current_price if nearest_resistance > 0 else np.nan
    
    return {
        "distance_to_support": dist_support,
        "distance_to_resistance": dist_resistance,
        "support_level": nearest_support,
        "resistance_level": nearest_resistance
    }


def compute_momentum_quality(df: pd.DataFrame) -> Dict[str, float]:
    """
    Assess momentum quality: consistency, acceleration, breadth.
    High quality momentum = sustainable move.
    """
    if len(df) < 60:
        return {"momentum_consistency": 0.0, "momentum_acceleration": 0.0}
    
    # Calculate returns over multiple periods
    returns_1w = df["Close"].pct_change(5).tail(12)
    returns_1m = df["Close"].pct_change(21).tail(12)
    
    # Consistency: what % of recent periods were positive
    consistency_1w = (returns_1w > 0).sum() / len(returns_1w) if len(returns_1w) > 0 else 0.0
    consistency_1m = (returns_1m > 0).sum() / len(returns_1m) if len(returns_1m) > 0 else 0.0
    
    momentum_consistency = (consistency_1w + consistency_1m) / 2
    
    # Acceleration: is momentum increasing
    recent_returns = df["Close"].pct_change(21).tail(3)
    acceleration = 1.0 if len(recent_returns) >= 2 and recent_returns.iloc[-1] > recent_returns.iloc[0] else 0.0
    
    return {
        "momentum_consistency": momentum_consistency,
        "momentum_acceleration": acceleration
    }


def calculate_risk_reward_ratio(df: pd.DataFrame, atr_period: int = 14) -> Dict[str, float]:
    """
    Calculate risk/reward based on ATR, support/resistance levels.
    Higher ratio = better setup.
    """
    if len(df) < atr_period * 2:
        return {"risk_reward_ratio": np.nan, "potential_reward": np.nan, "potential_risk": np.nan}
    
    # ATR for risk measure
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.tail(atr_period).mean()
    
    current_price = df["Close"].iloc[-1]
    
    # Get support/resistance
    sr_levels = find_support_resistance(df)
    resistance = sr_levels.get("resistance_level", current_price * 1.1)
    support = sr_levels.get("support_level", current_price * 0.95)
    
    # Potential reward (to resistance)
    potential_reward = (resistance - current_price) if resistance > current_price else atr * 2
    
    # Potential risk (to support or 2x ATR)
    potential_risk = max((current_price - support), atr * 2) if support < current_price else atr * 2
    
    risk_reward = potential_reward / potential_risk if potential_risk > 0 else 0.0
    
    return {
        "risk_reward_ratio": risk_reward if np.isfinite(risk_reward) else 0.0,
        "potential_reward": potential_reward,
        "potential_risk": potential_risk
    }


def compute_advanced_score(
    ticker: str,
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    base_score: float
) -> Tuple[float, Dict[str, any]]:
    """
    Compute enhanced score with all advanced filters.
    Returns (enhanced_score, signals_dict)
    """
    signals = {}
    
    # 1. Relative Strength
    rs_scores = compute_relative_strength(df, benchmark_df)
    signals.update(rs_scores)
    rs_boost = 0.0
    rs_63d_val = rs_scores.get("rs_63d", np.nan)
    if np.isfinite(rs_63d_val):
        # Boost if outperforming in medium term
        rs_boost = 10.0 if rs_63d_val > 0.05 else 5.0 if rs_63d_val > 0 else 0.0
    
    # 2. Volume Analysis
    vol_data = detect_volume_surge(df)
    signals.update(vol_data)
    vol_boost = 0.0
    if vol_data["volume_surge"] > 1.5 and vol_data["pv_correlation"] > 0.3:
        vol_boost = 8.0
    elif vol_data["volume_surge"] > 1.2:
        vol_boost = 4.0
    
    # 3. Consolidation Detection
    squeeze = detect_consolidation(df)
    signals["consolidation_ratio"] = squeeze
    consolidation_boost = 0.0
    if np.isfinite(squeeze) and 0.6 < squeeze < 0.85:
        consolidation_boost = 6.0  # Tight range before breakout
    
    # 4. MA Alignment
    ma_data = check_ma_alignment(df)
    signals.update(ma_data)
    ma_boost = 0.0
    if ma_data["ma_aligned"]:
        ma_boost = 12.0
    elif ma_data["alignment_score"] > 0.66:
        ma_boost = 6.0
    
    # 5. Support/Resistance
    sr_data = find_support_resistance(df)
    signals.update(sr_data)
    sr_boost = 0.0
    dist_support = sr_data.get("distance_to_support", np.nan)
    if np.isfinite(dist_support) and 0.02 < dist_support < 0.05:
        sr_boost = 5.0  # Near support = good entry
    
    # 6. Momentum Quality
    mom_data = compute_momentum_quality(df)
    signals.update(mom_data)
    mom_boost = 0.0
    if mom_data["momentum_consistency"] > 0.7:
        mom_boost = 8.0
    elif mom_data["momentum_consistency"] > 0.5:
        mom_boost = 4.0
    
    # 7. Risk/Reward
    rr_data = calculate_risk_reward_ratio(df)
    signals.update(rr_data)
    rr_boost = 0.0
    rr_val = rr_data.get("risk_reward_ratio", np.nan)
    if np.isfinite(rr_val):
        if rr_val > 3.0:
            rr_boost = 10.0
        elif rr_val > 2.0:
            rr_boost = 6.0
        elif rr_val > 1.5:
            rr_boost = 3.0
    
    # Calculate total boost (max 50 points)
    total_boost = min(50.0, 
        rs_boost + vol_boost + consolidation_boost + 
        ma_boost + sr_boost + mom_boost + rr_boost
    )
    
    # Enhanced score
    enhanced_score = min(100.0, base_score + total_boost)
    
    # Add quality flags
    signals["quality_score"] = total_boost
    signals["high_confidence"] = (
        ma_data["ma_aligned"] and 
        vol_data["volume_surge"] > 1.2 and
        mom_data["momentum_consistency"] > 0.6 and
        np.isfinite(rr_val) and rr_val > 1.5
    )
    
    return enhanced_score, signals


def should_reject_ticker(signals: Dict[str, any]) -> Tuple[bool, str]:
    """
    Hard rejection criteria - eliminate poor setups.
    Returns (should_reject, reason)
    """
    # Reject if underperforming market significantly
    rs_63d = signals.get("rs_63d", np.nan)
    if np.isfinite(rs_63d) and rs_63d < -0.10:
        return True, "Underperforming market by >10%"
    
    # Reject if weak momentum consistency
    mom_consistency = signals.get("momentum_consistency", 0.0)
    if mom_consistency < 0.3:
        return True, "Weak momentum consistency"
    
    # Reject if poor risk/reward
    rr = signals.get("risk_reward_ratio", np.nan)
    if np.isfinite(rr) and rr < 1.0:
        return True, "Risk/Reward < 1.0"
    
    # Reject if MA trend is bearish
    alignment_score = signals.get("alignment_score", 0.0)
    if alignment_score < 0.3:
        return True, "Bearish MA alignment"
    
    return False, ""


def fetch_benchmark_data(benchmark: str = "SPY", days: int = 400) -> pd.DataFrame:
    """
    Fetch benchmark data for relative strength calculations.
    Cached to avoid repeated downloads.
    """
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        bench_df = yf.download(benchmark, start=start, end=end, auto_adjust=True, progress=False)
        return bench_df if not bench_df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
