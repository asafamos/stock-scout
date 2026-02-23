"""
Technical Indicator Computation for Stock Scout.

Single source of truth for all technical indicator calculations.
All entry points (live app, backtest, time-test) must use
:func:`build_technical_indicators` to ensure deterministic, consistent results.

Functions:
    compute_rsi         — Relative Strength Index (14-period EMA)
    compute_atr         — Average True Range
    compute_momentum_consistency — Fraction of up days in window
    compute_volume_surge — Volume relative to N-day average
    compute_reward_risk  — Trend-following reward/risk ratio
    build_technical_indicators — Master function: OHLCV → 40+ features
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Elementary indicator functions
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index using exponential moving average.

    RSI measures momentum by comparing average gains vs. average losses.
    - RSI > 70: Overbought (potential pullback)
    - RSI < 30: Oversold (potential bounce)
    - RSI 40-60: Neutral

    Args:
        series: Closing prices.
        period: Look-back period (default 14 per Wilder's standard).

    Returns:
        Series with RSI values in [0, 100].
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR) — volatility indicator.

    Formula: TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|);
             ATR = SMA(TR, *period*).

    Args:
        df: DataFrame with High, Low, Close columns.
        period: Look-back window (default 14).

    Returns:
        Series with ATR values in price units.
    """
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    true_range = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    return true_range.rolling(window=period).mean()


def compute_momentum_consistency(close: pd.Series, lookback: int = 14) -> pd.Series:
    """Fraction of up-days in a rolling window — measures trend strength.

    Returns:
        Series in [0, 1]; closer to 1.0 = strong uptrend.
    """
    up_days = (close.diff() > 0).astype(int)
    return up_days.rolling(window=lookback).mean()


def compute_volume_surge(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Current volume relative to *lookback*-day average.

    Returns:
        Series; values > 1.0 = above-average volume.
    """
    avg_volume = volume.rolling(window=lookback).mean()
    return volume / avg_volume


def compute_reward_risk(
    close: pd.Series,
    low: pd.Series,
    high: pd.Series,
    lookback: int = 14,
) -> pd.Series:
    """Trend-following Reward/Risk using ATR.

    Assumes breakout strategy: Reward = 3×ATR, Risk = 1.5×ATR.
    Clipped to [0.5, 5.0]; NaN filled with neutral 2.0.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=lookback).mean()

    rr = (3.0 * atr) / (1.5 * atr).replace(0, np.nan)
    return rr.clip(lower=0.5, upper=5.0).fillna(2.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_streak(change_series: pd.Series, is_up: bool) -> pd.Series:
    """Consecutive streak length (up or down days)."""
    streak_list: list[int] = []
    count = 0
    for val in change_series:
        if pd.notna(val):
            if (is_up and val > 0) or (not is_up and val < 0):
                count += 1
            else:
                count = 0
        else:
            count = 0
        streak_list.append(count)
    return pd.Series(streak_list, index=change_series.index)


# ---------------------------------------------------------------------------
# Master indicator builder
# ---------------------------------------------------------------------------

def build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Build all technical indicators from OHLCV data.

    **Single source of truth** for indicator computation.

    Args:
        df: DataFrame with OHLCV columns (may have yfinance MultiIndex).

    Returns:
        DataFrame with 40+ indicator columns.
    """
    result = pd.DataFrame(index=df.index)

    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        dff = df.copy()
        dff.columns = dff.columns.get_level_values(-1)
    else:
        dff = df

    close = dff["Close"]
    high = dff["High"]
    low = dff["Low"]
    volume = dff["Volume"]

    # ── Moving Averages ──────────────────────────────────────────
    result["MA20"] = close.rolling(window=20).mean()
    result["MA50"] = close.rolling(window=50).mean()
    result["MA200"] = close.rolling(window=200).mean()

    # ── Core indicators ──────────────────────────────────────────
    result["RSI"] = compute_rsi(close, period=14)
    result["ATR"] = compute_atr(dff, period=14)
    result["ATR_Pct"] = result["ATR"] / close
    result["ADR_Pct"] = result["ATR_Pct"]  # alias for ML features

    # ── Tightness metrics ────────────────────────────────────────
    try:
        atr_5 = compute_atr(dff, period=5)
        atr_20 = compute_atr(dff, period=20)
        result["RangeRatio_5_20"] = (atr_5 / atr_20).replace([np.inf, -np.inf], np.nan)
    except (TypeError, ValueError, ZeroDivisionError):
        result["RangeRatio_5_20"] = np.nan

    try:
        std_5 = close.rolling(5).std()
        std_20 = close.rolling(20).std()
        result["Tightness_Ratio"] = (std_5 / std_20).replace([np.inf, -np.inf], np.nan)
    except (TypeError, ValueError, ZeroDivisionError):
        result["Tightness_Ratio"] = np.nan

    # ── VCP (Volatility Contraction Pattern) — Minervini-inspired ─
    atr_10 = compute_atr(dff, period=10)
    atr_30 = compute_atr(dff, period=30)
    ratio = atr_10 / atr_30

    # 1. Volatility contraction
    volatility_score = (1.0 - ratio.clip(lower=0.0, upper=1.5)).where(
        (atr_10 < atr_30) & ratio.notna(), 0.0
    ).clip(lower=0.0, upper=1.0)

    # 2. Price near 52w high
    try:
        hi_52w_vcp = close.rolling(window=min(len(close), 252)).max()
        price_position = close / hi_52w_vcp
        price_score = ((price_position - 0.75) / 0.25).clip(lower=0.0, upper=1.0)
    except Exception:
        price_score = pd.Series(0.5, index=close.index)

    # 3. Volume drying up
    try:
        vol_recent_5 = volume.rolling(5).mean()
        vol_old_20 = volume.rolling(20).mean()
        volume_dry_score = (1.0 - (vol_recent_5 / vol_old_20).clip(lower=0.5, upper=1.5)).clip(
            lower=0.0, upper=1.0
        )
    except Exception:
        volume_dry_score = pd.Series(0.5, index=close.index)

    # 4. Tightness
    try:
        range_5d = close.rolling(5).max() - close.rolling(5).min()
        range_20d = close.rolling(20).max() - close.rolling(20).min()
        tightness_score = (1.0 - (range_5d / range_20d.replace(0, np.nan)).clip(
            lower=0.0, upper=1.0
        )).clip(lower=0.0, upper=1.0).fillna(0.5)
    except Exception:
        tightness_score = pd.Series(0.5, index=close.index)

    # Bonus: within 2% of MA20 while ATR contracting
    near_ma20 = ((close / result["MA20"]) - 1.0).abs() <= 0.02
    ma20_bonus = near_ma20.astype(float) * 0.1

    vcp_composite = (
        0.35 * volatility_score
        + 0.25 * price_score
        + 0.20 * volume_dry_score
        + 0.20 * tightness_score
        + ma20_bonus
    ).clip(lower=0.0, upper=1.0)

    result["Volatility_Contraction_Score"] = vcp_composite.astype(float)
    result["Coil_Bonus"] = (vcp_composite > 0.6).astype(int)

    # ── Price-based features ─────────────────────────────────────
    result["Overext"] = (close / result["MA50"]) - 1
    result["Near52w"] = (close / close.rolling(window=252).max()) * 100
    try:
        hi_52w = close.rolling(window=min(len(close), 252)).max()
        result["Dist_52w_High"] = (close / hi_52w) - 1.0
    except Exception:
        result["Dist_52w_High"] = np.nan

    # ── Momentum & Volume ────────────────────────────────────────
    result["MomCons"] = compute_momentum_consistency(close, lookback=14)
    result["VolSurge"] = compute_volume_surge(volume, lookback=20)
    try:
        recent_vol = volume.rolling(5).mean()
        avg_vol20 = volume.rolling(20).mean()
        result["Volume_Surge_Ratio"] = recent_vol / avg_vol20
    except Exception:
        result["Volume_Surge_Ratio"] = np.nan
    result["RR"] = compute_reward_risk(close, low, high, lookback=14)

    # ── Derived ML features ──────────────────────────────────────
    result["RR_MomCons"] = result["RR"] * result["MomCons"]
    result["RSI_Neutral"] = (result["RSI"] - 50).abs()
    result["RSI_Squared"] = result["RSI"] ** 2
    result["Risk_Score"] = result["Overext"].abs() + result["ATR_Pct"]
    result["Vol_Mom"] = result["VolSurge"] * result["MomCons"]
    result["Overext_Mom_Div"] = result["Overext"] * result["MomCons"]
    result["RR_Risk_Adj"] = result["RR"] / (1 + result["Overext"].abs())

    # ATR regime (simplified binning)
    result["ATR_Regime"] = pd.cut(
        result["ATR_Pct"], bins=[0, 0.02, 0.04, 1.0], labels=[1, 2, 3]
    ).astype(float)

    # ── Timing indicators ────────────────────────────────────────
    result["Vol_Breakout"] = volume > (volume.rolling(20).mean() * 1.5)
    result["Price_Breakout"] = close > close.rolling(20).quantile(0.90)
    result["Mom_Acceleration"] = close.pct_change(5, fill_method=None) > close.shift(5).pct_change(
        5, fill_method=None
    )

    # ── Multi-period returns ─────────────────────────────────────
    result["Return_1m"] = close.pct_change(20, fill_method=None)
    result["Return_3m"] = close.pct_change(60, fill_method=None)
    result["Return_6m"] = close.pct_change(120, fill_method=None)
    result["Return_5d"] = close.pct_change(5, fill_method=None)
    result["Return_10d"] = close.pct_change(10, fill_method=None)

    # MA50 slope for trend component
    result["MA50_Slope"] = result["MA50"].pct_change(10, fill_method=None)

    # ── Streaks ──────────────────────────────────────────────────
    daily_change = close.diff()
    result["UpStreak_Days"] = _compute_streak(daily_change, is_up=True)
    result["DownStreak_Days"] = _compute_streak(daily_change, is_up=False)

    # ── Rolling high/low & pullback metrics ──────────────────────
    result["High_20d"] = high.rolling(20).max()
    # VCP bonus: tight coiling near 20d high
    try:
        near_20d_high = close >= (result["High_20d"] * 0.97)
        tight_mask = (result["Tightness_Ratio"] < 0.6) & near_20d_high
        vcp_with_bonus = result["Volatility_Contraction_Score"] + (tight_mask.astype(float) * 0.25)
        result["Volatility_Contraction_Score"] = vcp_with_bonus.clip(upper=1.0)
    except Exception:
        pass

    result["Low_20d"] = low.rolling(20).min()
    result["PullbackFromHigh_20d"] = (close - result["High_20d"]) / result["High_20d"]
    result["DistanceFromLow_20d"] = (close - result["Low_20d"]) / result["Low_20d"]
    result["DaysSince20dHigh"] = 0.0  # placeholder
    result["DaysSince20dLow"] = 0.0  # placeholder

    # ── Intraday range ───────────────────────────────────────────
    result["Range_Pct"] = (high - low) / close

    # ── OHLCV reference ──────────────────────────────────────────
    result["Close"] = close
    result["Volume"] = volume
    result["High"] = high
    result["Low"] = low

    return result
