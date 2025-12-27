"""
ML Features V3 - Enhanced Feature Engineering for 20d Model

This module contains advanced feature engineering for the 20d ML model:
- Relative strength (vs SPY, vs sector)
- Volatility classification and context
- Sequential patterns (streaks, pullbacks, extensions)
- Earnings proximity (if available)

All functions are NaN-safe and avoid future peeking.
"""
from __future__ import annotations
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
