"""
Meteor-oriented feature engineering helpers for ML training/inference.

Adds:
- dist_from_52w_high
- rs_rank_vs_spy (percentile rank expected to be supplied externally)
- volume_surge_ratio (current vs 20d avg)
- adr_pct (Average Daily Range percentage)
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def compute_dist_from_52w_high(df: pd.DataFrame) -> float:
    df = df.rename(columns={
        'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    look = min(len(df), 252)
    if look < 20:
        return np.nan
    high_52w = float(df['High'].tail(look).max())
    close = float(df['Close'].iloc[-1])
    if high_52w <= 0:
        return np.nan
    return float((close / high_52w) - 1.0)


def compute_volume_surge_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    df = df.rename(columns={
        'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    if len(df) < lookback + 5:
        return np.nan
    recent = float(df['Volume'].tail(5).mean())
    avg = float(df['Volume'].tail(lookback).mean())
    return float(recent / avg) if avg > 0 else np.nan


def compute_adr_pct(df: pd.DataFrame, period: int = 14) -> float:
    df = df.rename(columns={
        'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    close = df['Close']
    if len(close) == 0 or pd.isna(atr.iloc[-1]) or close.iloc[-1] <= 0:
        return np.nan
    return float(atr.iloc[-1] / close.iloc[-1])

"""
ML Features V3 - Enhanced Feature Engineering for 20d Model

This module contains advanced feature engineering for the 20d ML model:
- Relative strength (vs SPY, vs sector)
- Volatility classification and context
- Sequential patterns (streaks, pullbacks, extensions)
- Earnings proximity (if available)

All functions are NaN-safe and avoid future peeking.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_multi_period_returns(df: pd.DataFrame, close_col: str = 'Close') -> pd.DataFrame:
    """
    Add multi-period returns to a price DataFrame.
    
    Adds columns: Return_5d, Return_10d, Return_20d, Return_60d, Return_120d
    
    Args:
        df: DataFrame with price history (must have close_col)
        close_col: Name of close price column
    
    Returns:
        DataFrame with added return columns
    """
    result = df.copy()
    
    close = result[close_col]
    
    result['Return_5d'] = close.pct_change(5, fill_method=None)
    result['Return_10d'] = close.pct_change(10, fill_method=None)
    result['Return_20d'] = close.pct_change(20, fill_method=None)
    result['Return_60d'] = close.pct_change(60, fill_method=None)
    result['Return_120d'] = close.pct_change(120, fill_method=None)
    
    return result


def compute_relative_strength_features(
    row: pd.Series,
    spy_context: Optional[pd.DataFrame] = None,
    sector_context: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Compute relative strength features vs SPY and sector.
    
    Args:
        row: Stock data row with Return_20d, Return_60d, As_Of_Date, Sector
        spy_context: DataFrame with date, SPY_20d_ret, SPY_60d_ret
        sector_context: DataFrame with date, Sector, Sector_20d_ret, Sector_60d_ret
    
    Returns:
        Dict with RS_SPY_20d, RS_SPY_60d, RS_Sector_20d, RS_Sector_60d
    """
    features = {
        'RS_SPY_20d': np.nan,
        'RS_SPY_60d': np.nan,
        'RS_Sector_20d': np.nan,
        'RS_Sector_60d': np.nan
    }
    
    # Get stock returns
    stock_ret_20d = row.get('Return_20d', np.nan)
    stock_ret_60d = row.get('Return_60d', np.nan)
    as_of_date = row.get('As_Of_Date')
    sector = row.get('Sector', 'Unknown')
    
    # Compute RS vs SPY
    if spy_context is not None and as_of_date is not None:
        try:
            spy_row = spy_context[spy_context['date'] == pd.to_datetime(as_of_date)]
            if not spy_row.empty:
                spy_ret_20d = float(spy_row.iloc[0]['SPY_20d_ret'])
                spy_ret_60d = float(spy_row.iloc[0]['SPY_60d_ret'])
                
                if pd.notna(stock_ret_20d) and pd.notna(spy_ret_20d):
                    features['RS_SPY_20d'] = stock_ret_20d - spy_ret_20d
                
                if pd.notna(stock_ret_60d) and pd.notna(spy_ret_60d):
                    features['RS_SPY_60d'] = stock_ret_60d - spy_ret_60d
        except Exception as e:
            logger.debug(f"RS vs SPY computation failed: {e}")
    
    # Compute RS vs Sector
    if sector_context is not None and as_of_date is not None and sector != 'Unknown':
        try:
            sector_row = sector_context[
                (sector_context['date'] == pd.to_datetime(as_of_date)) &
                (sector_context['Sector'] == sector)
            ]
            if not sector_row.empty:
                sector_ret_20d = float(sector_row.iloc[0]['Sector_20d_ret'])
                sector_ret_60d = float(sector_row.iloc[0]['Sector_60d_ret'])
                
                if pd.notna(stock_ret_20d) and pd.notna(sector_ret_20d):
                    features['RS_Sector_20d'] = stock_ret_20d - sector_ret_20d
                
                if pd.notna(stock_ret_60d) and pd.notna(sector_ret_60d):
                    features['RS_Sector_60d'] = stock_ret_60d - sector_ret_60d
        except Exception as e:
            logger.debug(f"RS vs Sector computation failed: {e}")
    
    return features


def compute_volatility_context_features(
    atr_pct: float,
    atr_pct_universe: pd.Series
) -> Dict[str, float]:
    """
    Compute volatility context features based on ATR percentile within universe.
    
    Args:
        atr_pct: Stock's ATR as % of price
        atr_pct_universe: Series of ATR_Pct values for all stocks in universe
    
    Returns:
        Dict with:
        - ATR_Pct_percentile: percentile rank (0-1)
        - Vol_Class: discrete class (0=very low, 1=low-mid, 2=mid-high, 3=extreme)
        - Vol_SweetSpot_Flag: 1 if in "good" ATR band (0.4-0.8 percentile)
        - Vol_Extreme_Flag: 1 if in extreme band (>0.9 percentile)
    """
    features = {
        'ATR_Pct_percentile': 0.5,  # Default neutral
        'Vol_Class': 1,  # Default low-mid
        'Vol_SweetSpot_Flag': 0,
        'Vol_Extreme_Flag': 0
    }
    
    if pd.isna(atr_pct) or atr_pct_universe is None or len(atr_pct_universe) == 0:
        return features
    
    try:
        # Compute percentile within universe
        percentile = (atr_pct_universe <= atr_pct).sum() / len(atr_pct_universe)
        features['ATR_Pct_percentile'] = float(percentile)
        
        # Discrete volatility class
        if percentile < 0.2:
            features['Vol_Class'] = 0  # very low
        elif percentile < 0.5:
            features['Vol_Class'] = 1  # low-mid
        elif percentile < 0.8:
            features['Vol_Class'] = 2  # mid-high
        else:
            features['Vol_Class'] = 3  # extreme
        
        # Sweet spot flag (mid-to-high volatility, but not extreme)
        if 0.4 <= percentile <= 0.8:
            features['Vol_SweetSpot_Flag'] = 1
        
        # Extreme volatility flag
        if percentile > 0.9:
            features['Vol_Extreme_Flag'] = 1
            
    except Exception as e:
        logger.debug(f"Volatility context computation failed: {e}")
    
    return features


def build_ml_20d_features(history_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Build the minimal ML 20d feature frame expected by strict tests.

        Inputs:
            - history_df: DataFrame with columns [Date, Open, High, Low, Close, Volume]

        Outputs:
            - X: single-row DataFrame with columns [Volume_Surge_Ratio, Dist_52w_High, ADR_Pct]
            - names: ordered feature list

        Notes:
            - This function is NaN-safe and uses recent windows for robustness.
        """
        try:
                df = history_df.copy()
                if 'Date' in df.columns:
                        df = df.sort_values('Date')
                # Compute features using helpers (expect NaN-safe behavior)
                vsr = compute_volume_surge_ratio(df)
                dist = compute_dist_from_52w_high(df)
                adr = compute_adr_pct(df)
                names = ["Volume_Surge_Ratio", "Dist_52w_High", "ADR_Pct"]
                X = pd.DataFrame([{names[0]: vsr, names[1]: dist, names[2]: adr}])
                # Replace infs and ensure numeric dtype
                X = X.replace([np.inf, -np.inf], np.nan).astype(float)
                return X, names
        except Exception as e:
                logger.warning(f"build_ml_20d_features failed: {e}")
                names = ["Volume_Surge_Ratio", "Dist_52w_High", "ADR_Pct"]
                return pd.DataFrame([{n: np.nan for n in names}]), names


def compute_sequential_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sequential pattern features to a price DataFrame.
    
    Computes:
    - UpStreak_Days, DownStreak_Days: consecutive up/down days
    - High_20d, Low_20d: rolling 20d high/low
    - PullbackFromHigh_20d, DistanceFromLow_20d: price position metrics
    - DaysSince20dHigh, DaysSince20dLow: time since local extremes
    - OvernightGap_Pct: (open - prev_close) / prev_close (if open available)
    - Range_Pct: (high - low) / close (intraday range)
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with added sequential pattern columns
    """
    result = df.copy()
    close = result['Close']
    high = result['High']
    low = result['Low']
    
    # Streaks: count consecutive up/down closes
    daily_change = close.diff()
    
    def compute_streak(series, condition):
        """Compute consecutive streak length."""
        streak = []
        count = 0
        for val in series:
            if condition(val):
                count += 1
            else:
                count = 0
            streak.append(count)
        return pd.Series(streak, index=series.index)
    
    result['UpStreak_Days'] = compute_streak(daily_change, lambda x: x > 0)
    result['DownStreak_Days'] = compute_streak(daily_change, lambda x: x < 0)
    
    # Rolling 20d high/low
    result['High_20d'] = high.rolling(20).max()
    result['Low_20d'] = low.rolling(20).min()
    
    # Pullback / extension metrics
    result['PullbackFromHigh_20d'] = (close - result['High_20d']) / result['High_20d']
    result['DistanceFromLow_20d'] = (close - result['Low_20d']) / result['Low_20d']
    
    # Days since 20d high/low
    def days_since_extreme(series, close_series):
        """Compute days since last occurrence of extreme value."""
        days_since = []
        for i in range(len(series)):
            if pd.isna(series.iloc[i]):
                days_since.append(np.nan)
                continue
            
            # Look back in window
            lookback = min(i + 1, 20)
            window = close_series.iloc[max(0, i - lookback + 1):i + 1]
            
            if window.empty:
                days_since.append(np.nan)
                continue
            
            # Find days since max/min
            if series.iloc[i] == window.max():
                days_since.append(0)
            else:
                max_idx = window.idxmax()
                days_since.append(i - window.index.get_loc(max_idx))
        
        return pd.Series(days_since, index=series.index)
    
    result['DaysSince20dHigh'] = days_since_extreme(result['High_20d'], high)
    result['DaysSince20dLow'] = days_since_extreme(result['Low_20d'], low)
    
    # Overnight gap (if open prices available)
    if 'Open' in result.columns:
        prev_close = close.shift(1)
        result['OvernightGap_Pct'] = (result['Open'] - prev_close) / prev_close
    else:
        result['OvernightGap_Pct'] = 0.0
    
    # Intraday range
    result['Range_Pct'] = (high - low) / close
    
    return result


def compute_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute breakout-oriented features commonly used to anticipate upward moves:
    - Bollinger/Keltner squeeze: BB inside KC -> potential energy buildup
    - Volatility contraction: 20d std vs 60d std ratio
    - Relative volume: today's volume vs 20d average
    - MA slopes: short vs long moving averages slopes over recent days

    Returns a DataFrame with added columns:
    - BB_Width, KC_Width, Squeeze_On_Flag
    - Vol_Contraction_Ratio
    - Volume_Relative_20d
    - MA_Slope_S, MA_Slope_L
    """
    result = df.copy()
    close = result.get('Close')
    high = result.get('High', close)
    low = result.get('Low', close)
    vol = result.get('Volume', pd.Series(np.nan, index=result.index))

    # Bollinger Bands (20, 2)
    ma20 = close.rolling(20, min_periods=10).mean()
    std20 = close.rolling(20, min_periods=10).std(ddof=0)
    bb_up = ma20 + 2.0 * std20
    bb_lo = ma20 - 2.0 * std20
    bb_width = (bb_up - bb_lo) / close.replace(0, np.nan)
    result['BB_Width'] = bb_width.replace([np.inf, -np.inf], np.nan)

    # ATR(14)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=7).mean()

    # Keltner Channels (EMA20 ± 2*ATR14)
    ema20 = close.ewm(span=20, adjust=False).mean()
    kc_up = ema20 + 2.0 * atr14
    kc_lo = ema20 - 2.0 * atr14
    kc_width = (kc_up - kc_lo) / close.replace(0, np.nan)
    result['KC_Width'] = kc_width.replace([np.inf, -np.inf], np.nan)

    # Squeeze: BB inside KC
    result['Squeeze_On_Flag'] = ((bb_up <= kc_up) & (bb_lo >= kc_lo)).astype(int)

    # Volatility contraction ratio (lower implies contraction)
    ret = close.pct_change(fill_method=None)
    std20r = ret.rolling(20, min_periods=10).std(ddof=0)
    std60r = ret.rolling(60, min_periods=20).std(ddof=0)
    vol_ratio = std20r / std60r.replace(0, np.nan)
    result['Vol_Contraction_Ratio'] = vol_ratio.replace([np.inf, -np.inf], np.nan).clip(0.0, 2.0)

    # Relative volume (20d)
    vol_ma20 = vol.rolling(20, min_periods=10).mean()
    result['Volume_Relative_20d'] = (vol / vol_ma20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # MA slopes (short/long)
    ma_short = close.rolling(10, min_periods=5).mean()
    ma_long = close.rolling(50, min_periods=25).mean()
    # 5-day slope normalized by price
    slope_s = (ma_short - ma_short.shift(5)) / 5.0
    slope_l = (ma_long - ma_long.shift(5)) / 5.0
    result['MA_Slope_S'] = (slope_s / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    result['MA_Slope_L'] = (slope_l / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    return result

def compute_anchored_vwap_features(df: pd.DataFrame, anchor_days: int = 60) -> pd.DataFrame:
    """
    Compute anchored VWAP since an anchor point (e.g., last 60 trading days) and proximity to 52w high.
    Adds:
    - AnchoredVWAP_60d: VWAP from t-anchor to t
    - Price_vs_AnchoredVWAP_Pct: (Close / AnchoredVWAP - 1)
    - High_252d: rolling 252d high
    - Near52wHigh_Pct: (Close / High_252d - 1)
    """
    result = df.copy()
    close = result.get('Close')
    vol = result.get('Volume', pd.Series(np.nan, index=result.index))
    n = len(result)
    anchor_idx = max(0, n - anchor_days)
    # Cumulative sums for VWAP from anchor
    price_vol = (close * vol).astype(float)
    cum_pv = price_vol.cumsum()
    cum_v = vol.astype(float).cumsum()
    pv_anchor = cum_pv.shift(anchor_idx)
    v_anchor = cum_v.shift(anchor_idx)
    vwap = (cum_pv - (pv_anchor.fillna(0))) / (cum_v - (v_anchor.fillna(0)))
    result['AnchoredVWAP_60d'] = vwap.replace([np.inf, -np.inf], np.nan)
    result['Price_vs_AnchoredVWAP_Pct'] = (close / result['AnchoredVWAP_60d'] - 1.0).replace([np.inf, -np.inf], np.nan)
    # 52-week high proximity (≈252 trading days)
    high = result.get('High', close)
    result['High_252d'] = high.rolling(252, min_periods=60).max()
    result['Near52wHigh_Pct'] = (close / result['High_252d'] - 1.0).replace([np.inf, -np.inf], np.nan)
    return result

def compute_pivot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute recent pivot highs/lows and proximity/breakout flags.
    Adds:
    - PivotHigh_20d, PivotLow_20d: local extreme values (rolling window)
    - Dist_to_PivotHigh_Pct, Dist_to_PivotLow_Pct: proximity metrics
    - BreakoutAbovePivot_Flag: 1 if Close > PivotHigh_20d
    - ReboundFromPivotLow_Flag: 1 if Close > PivotLow_20d and UpStreak_Days >= 2
    """
    result = df.copy()
    close = result.get('Close')
    high = result.get('High', close)
    low = result.get('Low', close)

    # Rolling pivots using 10-day window inside 20d scope
    win = 10
    pivot_high = high.rolling(win, min_periods=5).max()
    pivot_low = low.rolling(win, min_periods=5).min()
    result['PivotHigh_20d'] = pivot_high
    result['PivotLow_20d'] = pivot_low

    # Distances to pivots
    result['Dist_to_PivotHigh_Pct'] = (close / pivot_high - 1.0).replace([np.inf, -np.inf], np.nan)
    result['Dist_to_PivotLow_Pct'] = (close / pivot_low - 1.0).replace([np.inf, -np.inf], np.nan)

    # Breakout/rebound flags
    result['BreakoutAbovePivot_Flag'] = (close > pivot_high).astype(int)
    # Use existing sequential streak computation if present; fallback to 0
    if 'UpStreak_Days' not in result.columns:
        tmp = compute_sequential_pattern_features(result)
        result['UpStreak_Days'] = tmp['UpStreak_Days']
    result['ReboundFromPivotLow_Flag'] = ((close > pivot_low) & (result['UpStreak_Days'] >= 2)).astype(int)

    return result


def compute_earnings_proximity_features(
    ticker: str,
    as_of_date: pd.Timestamp,
    earnings_calendar: Optional[Dict[str, pd.Timestamp]] = None
) -> Dict[str, float]:
    """
    Compute earnings proximity features.
    
    Args:
        ticker: Stock ticker
        as_of_date: Current date
        earnings_calendar: Dict mapping ticker -> next_earnings_date
    
    Returns:
        Dict with:
        - DaysToNextEarnings: days until next earnings (NaN if unknown)
        - In_Earnings_Window_Flag: 1 if within 5 days before earnings
    """
    features = {
        'DaysToNextEarnings': np.nan,
        'In_Earnings_Window_Flag': 0
    }
    
    if earnings_calendar is None or ticker not in earnings_calendar:
        return features
    
    try:
        next_earnings = earnings_calendar[ticker]
        if pd.notna(next_earnings):
            days_to = (next_earnings - as_of_date).days
            features['DaysToNextEarnings'] = float(days_to)
            
            # Flag if within 5 days before earnings
            if 0 <= days_to <= 5:
                features['In_Earnings_Window_Flag'] = 1
    except Exception as e:
        logger.debug(f"Earnings proximity computation failed for {ticker}: {e}")
    
    return features
