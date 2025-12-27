
from __future__ import annotations
import pandas as pd
import numpy as np
from core.classification import apply_classification
from core.scoring_engine import compute_overall_score

def compute_big_winner_signal_20d(row: pd.Series) -> dict:
    """
    Compute a purely technical 'big winner' signal for a 20d horizon.

    Inputs (from build_technical_indicators):
        - row["TechScore_20d"]
        - row["RSI"]
        - row["ATR_Pct"]
        - row["RR"]        # optional, used lightly
        - row["MomCons"]   # optional
        - row["VolSurge"]  # optional

    Returns a dict with:
        - "BigWinnerScore_20d": float in [0, 100]
        - "BigWinnerFlag_20d": int (1 = strong candidate, 0 = else)
    """
    score = 0.0
    try:
        tech_score = row.get("TechScore_20d", np.nan)
        rsi = row.get("RSI", np.nan)
        atr = row.get("ATR_Pct", np.nan)
        rr = row.get("RR", np.nan)
        # Optionals
        mom = row.get("MomCons", np.nan)
        vol = row.get("VolSurge", np.nan)

        # If any core field is missing, return 0,0
        if np.isnan(tech_score) or np.isnan(rsi) or np.isnan(atr):
            return {"BigWinnerScore_20d": 0.0, "BigWinnerFlag_20d": 0}

        # --- Thresholds (easy to tune) ---
        ATR_MIN = 0.03
        ATR_BONUS = 0.05
        SCORE_MID_LOW = 40
        SCORE_MID_HIGH = 80
        SCORE_WEAK_LOW = 30
        SCORE_WEAK_HIGH = 90
        RSI_MAX = 70
        RSI_BONUS = 60
        RR_MIN = 1.2
        RR_MAX = 3.0

        # --- Scoring logic ---
        # ATR filter
        if atr >= ATR_MIN:
            score += 40
            if atr >= ATR_BONUS:
                score += 20

        # TechScore band
        if SCORE_MID_LOW <= tech_score < SCORE_MID_HIGH:
            score += 30
        elif SCORE_WEAK_LOW <= tech_score < SCORE_WEAK_HIGH:
            score += 15

        # RSI filter
        if rsi < RSI_MAX:
            score += 10
            if rsi < RSI_BONUS:
                score += 5

        # RR bonus (optional)
        if not np.isnan(rr) and RR_MIN <= rr <= RR_MAX:
            score += 5

        # Optionally, could add small bonuses for MomCons/VolSurge if desired

        # Clip to [0, 100]
        score = float(np.clip(score, 0, 100))
        flag = 1 if score >= 60 else 0
        return {"BigWinnerScore_20d": score, "BigWinnerFlag_20d": flag}
    except Exception:
        return {"BigWinnerScore_20d": 0.0, "BigWinnerFlag_20d": 0}


def compute_recommendation_scores(
    row: pd.Series,
    ticker: str | None = None,
    as_of_date: datetime | None = None,
    enable_ml: bool = True,
    use_multi_source: bool = True,
) -> pd.Series:
    """Compute all recommendation scores and labels for a single stock row.

    This MUST be the single source of truth used by:
    - the live Streamlit app
    - experiments/offline_recommendation_audit.py

    It should:
    1) Call score_ticker_v2 to get the full v2 scoring + breakdown
    2) Merge all original row fields (technical indicators) into the result
    3) Add As_Of_Date if provided
    4) Run apply_classification(...) to compute Risk_Level, Data_Quality, Confidence_Level, Should_Display, Classification_Warnings
    5) Call compute_overall_score(rec_row) to populate:
       - rec_row["Score"]
       - rec_row["Score_Breakdown"]
    and return rec_row as a pandas Series.
    """
    base_ticker = ticker or row.get("Ticker")
    v2_result = score_ticker_v2(
        ticker=base_ticker,
        row=row,
        historical_df=None,
        enable_ml=enable_ml,
        use_multi_source=use_multi_source,
    )

    # Start from v2_result
    rec_row = pd.Series(v2_result, dtype="object")

    # Preserve original indicators from the input row
    for k, v in row.items():
        if k not in rec_row:
            rec_row[k] = v

    # Store As_Of_Date for traceability
    if as_of_date is not None:
        rec_row["As_Of_Date"] = pd.to_datetime(as_of_date)

    # Classification (risk, quality, display flags)
    classified = apply_classification(pd.DataFrame([rec_row])).iloc[0]
    for col in [
        "Risk_Level",
        "Data_Quality",
        "Confidence_Level",
        "Should_Display",
        "Classification_Warnings",
    ]:
        rec_row[col] = classified.get(col)

    # Unified overall score (for backwards compatibility with old UI)
    score, breakdown = compute_overall_score(rec_row)
    rec_row["Score"] = score
    rec_row["Score_Breakdown"] = breakdown

    return rec_row

def compute_overall_score_20d(row):
    """
    Compute a 20-day technical score targeting big-winner setups, using only technical features.
    Returns a float in [0, 100].
    """
    rsi = row.get("RSI", np.nan)
    atr = row.get("ATR_Pct", np.nan)
    rr = row.get("RR", np.nan)
    vs = row.get("VolSurge", np.nan)
    mom = row.get("MomCons", np.nan)

    def _score_atr(atr_val: float) -> float:
        if pd.isna(atr_val):
            return 0.5
        if atr_val < 0.01 or atr_val > 0.06:
            return 0.2
        if 0.013 <= atr_val <= 0.03:
            return 1.0
        if 0.01 <= atr_val < 0.013 or 0.03 < atr_val <= 0.05:
            return 0.7
        return 0.5

    def _score_rr(rr_val: float) -> float:
        if pd.isna(rr_val):
            return 0.5
        if rr_val < 1.0:
            return 0.2
        elif rr_val < 2.0:
            return 0.4
        elif rr_val < 4.0:
            return 0.7
        elif rr_val <= 7.0:
            return 1.0
        else:
            return 0.8

    def _score_rsi(rsi_val: float) -> float:
        if pd.isna(rsi_val):
            return 0.5
        if 36.0 <= rsi_val <= 50.0:
            return 1.0
        if 50.0 < rsi_val <= 58.0:
            return 0.7
        if 30.0 <= rsi_val < 36.0:
            return 0.6
        if 58.0 < rsi_val <= 65.0:
            return 0.4
        if rsi_val < 30.0:
            return 0.5
        return 0.2

    def _score_mom(mom_val: float) -> float:
        if pd.isna(mom_val):
            return 0.5
        if 0.25 <= mom_val <= 0.55:
            return 1.0
        if 0.15 <= mom_val < 0.25:
            return 0.7
        if 0.55 < mom_val <= 0.70:
            return 0.5
        if mom_val < 0.10:
            return 0.2
        if mom_val > 0.70:
            return 0.3
        return 0.5

    def _score_vol_surge(vs_val: float) -> float:
        if pd.isna(vs_val):
            return 0.5
        if 1.0 <= vs_val <= 1.6:
            return 1.0
        if 0.7 <= vs_val < 1.0:
            return 0.7
        if 1.6 < vs_val <= 2.5:
            return 0.6
        if vs_val < 0.7:
            return 0.2
        return 0.3

    atr_score = _score_atr(atr)
    rr_score = _score_rr(rr)
    rsi_score = _score_rsi(rsi)
    mom_score = _score_mom(mom)
    vol_score = _score_vol_surge(vs)

    w_atr = 0.25
    w_rr = 0.30
    w_rsi = 0.20
    w_mom = 0.15
    w_vol = 0.10

    base = (
        w_atr * atr_score +
        w_rr * rr_score +
        w_rsi * rsi_score +
        w_mom * mom_score +
        w_vol * vol_score
    )

    # Synergy: strong RR + not overbought RSI
    if rr_score >= 0.8 and rsi is not None and not pd.isna(rsi) and rsi <= 45.0:
        base += 0.05

    # Synergy: nice volatility + good momentum consistency
    if atr_score >= 0.8 and mom is not None and not pd.isna(mom) and 0.25 <= mom <= 0.55:
        base += 0.05

    base = float(np.clip(base, 0.0, 1.0))
    return base * 100.0


def compute_tech_score_20d_v2_components(row: pd.Series) -> dict:
    """
    Compute TechScore_20d_v2 component scores based on technical analysis summary recommendations.
    
    Returns 4 component scores (each 0-1) plus the combined raw score:
    - TrendScore: 40% weight (price vs MA50, MA50 vs MA200, MA50 slope)
    - MomentumScore: 35% weight (1m/3m/6m returns, de-emphasize extremes)
    - VolatilityScore: 15% weight (ATR_Pct sweet-spot in mid-quantiles)
    - LocationScore: 10% weight (penalize extreme RSI and near-highs)
    
    Note: These scores are RAW [0, 1] before normalization. 
    To get final TechScore_20d_v2, normalize per date using percent-rank.
    """
    # --- TrendScore (40%): Reward clean uptrends ---
    trend_score = 0.5  # Default neutral
    
    price_vs_ma50 = row.get('Overext', np.nan)  # (Price / MA50) - 1
    ma50 = row.get('MA50', np.nan)
    ma200 = row.get('MA200', np.nan)
    ma50_slope = row.get('MA50_Slope', np.nan)
    
    if pd.notna(price_vs_ma50) and pd.notna(ma50) and pd.notna(ma200):
        # Price above MA50
        if price_vs_ma50 > 0:
            trend_score = 0.6
            # MA50 above MA200 (golden cross territory)
            if ma50 > ma200:
                trend_score = 0.8
                # Positive MA50 slope
                if pd.notna(ma50_slope) and ma50_slope > 0:
                    trend_score = 1.0
        # Downtrend penalty
        elif price_vs_ma50 < -0.05:  # Price > 5% below MA50
            trend_score = 0.3
            if pd.notna(ma50) and pd.notna(ma200) and ma50 < ma200:
                trend_score = 0.1  # Death cross territory
    
    # --- MomentumScore (35%): Moderate positive momentum, de-emphasize parabolic ---
    momentum_score = 0.5  # Default neutral
    
    ret_1m = row.get('Return_1m', np.nan)
    ret_3m = row.get('Return_3m', np.nan)
    ret_6m = row.get('Return_6m', np.nan)
    
    # Compute average momentum (if available)
    rets = [r for r in [ret_1m, ret_3m, ret_6m] if pd.notna(r)]
    if rets:
        avg_ret = np.mean(rets)
        # Sweet spot: +5% to +25% (good momentum, not parabolic)
        if 0.05 <= avg_ret <= 0.25:
            momentum_score = 1.0
        elif 0.0 <= avg_ret < 0.05:
            momentum_score = 0.7
        elif 0.25 < avg_ret <= 0.50:
            momentum_score = 0.6  # Too hot, de-emphasize
        elif avg_ret > 0.50:
            momentum_score = 0.3  # Parabolic, likely to mean-revert
        elif -0.10 <= avg_ret < 0.0:
            momentum_score = 0.4  # Slight negative
        else:
            momentum_score = 0.2  # Strong negative
    
    # --- VolatilityScore (15%): ATR_Pct sweet-spot (mid-to-high quantiles) ---
    volatility_score = 0.5  # Default neutral
    
    atr_pct = row.get('ATR_Pct', np.nan)
    if pd.notna(atr_pct):
        # Sweet spot: 1.5% - 4.5% (active but not casino)
        if 0.015 <= atr_pct <= 0.045:
            volatility_score = 1.0
        elif 0.01 <= atr_pct < 0.015:
            volatility_score = 0.6  # Slightly low
        elif 0.045 < atr_pct <= 0.08:
            volatility_score = 0.7  # Slightly high but acceptable
        elif atr_pct < 0.01:
            volatility_score = 0.2  # Dead stock
        else:
            volatility_score = 0.3  # Too volatile (casino)
    
    # --- LocationScore (10%): Penalize chasing (extreme RSI, near highs) ---
    location_score = 0.5  # Default neutral
    
    rsi = row.get('RSI', np.nan)
    near_52w = row.get('Near52w', np.nan)
    
    # Penalize extreme overbought
    if pd.notna(rsi):
        if rsi >= 75:
            location_score = 0.2  # Very overbought
        elif rsi >= 65:
            location_score = 0.5  # Moderately overbought
        elif 40 <= rsi < 65:
            location_score = 0.8  # Healthy zone
        elif 30 <= rsi < 40:
            location_score = 0.7  # Slightly oversold (can be good)
        elif rsi < 30:
            location_score = 0.5  # Very oversold (risky)
    
    # Penalize near 52-week high (chasing)
    if pd.notna(near_52w) and near_52w > 95:
        location_score *= 0.7  # Reduce score if very near highs
    
    # Compute raw combined score
    raw_score = (
        0.40 * trend_score +
        0.35 * momentum_score +
        0.15 * volatility_score +
        0.10 * location_score
    )
    
    return {
        'TrendScore': trend_score,
        'MomentumScore': momentum_score,
        'VolatilityScore': volatility_score,
        'LocationScore': location_score,
        'TechScore_20d_v2_raw': raw_score
    }


def compute_tech_score_20d_v2(row: pd.Series) -> float:
    """
    Compute TechScore_20d_v2 using the hybrid technical formula.
    
    Returns a raw score in [0, 1] that should be normalized to [0, 100] 
    using percent-rank within the daily universe (grouped by As_Of_Date).
    
    This is a more balanced technical score compared to the legacy TechScore_20d,
    based on empirical analysis in experiments/outputs/technical_logic/summary.txt
    """
    components = compute_tech_score_20d_v2_components(row)
    return components['TechScore_20d_v2_raw']


from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from core.config import get_config
from core.scoring.fundamental import compute_fundamental_score_with_breakdown
from core.classification import apply_classification


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) using exponential moving average.

    RSI measures momentum by comparing average gains vs. average losses.
    - RSI > 70: Overbought (potential pullback)
    - RSI < 30: Oversold (potential bounce)
    - RSI 40-60: Neutral

    Args:
        series: Series of closing prices
        period: Look-back period (default 14 per Wilders standard)

    Returns:
        Series with RSI values in range [0, 100]
        NaN values at beginning due to look-back requirement
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR) - volatility indicator.
    
    ATR measures true range (highest high-low swing including gaps).
    Used to assess market volatility and set stop-loss distances.
    
    Formula:
        TR = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = SMA(TR, period)
    
    Args:
        df: DataFrame with High, Low, Close columns
        period: Look-back period (default 14)
    
    Returns:
        Series with ATR values in same units as price
        NaN values at beginning due to look-back requirement
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    range1 = high - low
    range2 = np.abs(high - prev_close)
    range3 = np.abs(low - prev_close)
    
    true_range = np.maximum(range1, np.maximum(range2, range3))
    atr = true_range.rolling(window=period).mean()
    return atr


def compute_momentum_consistency(close: pd.Series, lookback: int = 14) -> pd.Series:
    """
    Compute momentum consistency - fraction of up days in rolling window.
    
    Measures trend strength by counting days with positive close-to-close change.
    Values closer to 1.0 indicate strong uptrend; closer to 0.0 indicate downtrend.
    
    Args:
        close: Series of closing prices
        lookback: Rolling window size (default 14)
    
    Returns:
        Series with consistency values in range [0, 1]
    """
    up_days = (close.diff() > 0).astype(int)
    consistency = up_days.rolling(window=lookback).mean()
    return consistency


def compute_volume_surge(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Compute volume surge ratio - current volume relative to average.
    
    Ratio of current volume to N-day moving average.
    Values > 1.0 indicate above-average volume (potential momentum confirmation).
    Values < 1.0 indicate below-average volume (potential weakness).
    
    Args:
        volume: Series of trading volumes
        lookback: Period for average calculation (default 20)
    
    Returns:
        Series with surge values (typically in range [0.2, 3.0])
    """
    avg_volume = volume.rolling(window=lookback).mean()
    surge = volume / avg_volume
    return surge


def compute_reward_risk(close: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Compute reward/risk ratio - upside opportunity vs downside risk.

    Quantifies risk/reward setup:
        RR = (high_20d - current_price) / (current_price - low_20d)

    - RR > 2.0: Favorable setup (2:1 or better reward)
    - RR 1.0-2.0: Neutral setup
    - RR < 1.0: Unfavorable (risk exceeds reward)
    - Capped at 10 to avoid extreme outliers

    Args:
        close: Series of closing prices
        lookback: Period for high/low range (default 20)

    Returns:
        Series with RR values, capped at 10.0
    """
    high_20d = close.rolling(window=lookback).max()
    low_20d = close.rolling(window=lookback).min()
    
    reward = high_20d - close
    risk = close - low_20d
    
    rr = reward / risk.replace(0, np.nan)
    rr = rr.clip(upper=10)
    return rr


def build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all technical indicators from OHLCV data.
    
    This is the SINGLE SOURCE OF TRUTH for technical indicator calculation.
    All entry points (live app, backtest, time-test) must call this function
    to ensure deterministic, consistent results.
    
    Args:
        df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume).
            May have MultiIndex columns from yfinance (ticker level).
    
    Returns:
        DataFrame with original OHLCV plus computed indicators:
        - Moving Averages: MA20, MA50, MA200
        - RSI: RSI (14-period)
        - Volatility: ATR (14-period), ATR_Pct (ATR as % of price)
        - Price Position: Overext (price vs MA50), Near52w (% of 52-week high)
        - Momentum: MomCons (fraction of up days, 14-period), VolSurge (volume ratio)
        - Risk/Reward: RR (reward/risk ratio, 20-period, capped at 10)
        - Derived ML Features: RR_MomCons, RSI_Neutral, RSI_Squared, Risk_Score,
          Vol_Mom, Overext_Mom_Div, RR_Risk_Adj, ATR_Regime
        - Timing Signals: Vol_Breakout, Price_Breakout, Mom_Acceleration
        - Reference: Close, Volume, High, Low (for external use)
    
    Raises:
        Gracefully handles missing data; NaN values propagate to output.
    
    Note:
        All calculations are deterministic and independent of external state.
    """
    result = pd.DataFrame(index=df.index)
    
    # Extract series (handle multi-level columns from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        volume = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
    else:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
    
    # Moving averages
    result['MA20'] = close.rolling(window=20).mean()
    result['MA50'] = close.rolling(window=50).mean()
    result['MA200'] = close.rolling(window=200).mean()
    
    # Technical indicators
    result['RSI'] = compute_rsi(close, period=14)
    result['ATR'] = compute_atr(df, period=14)
    result['ATR_Pct'] = result['ATR'] / close
    
    # Price-based features
    result['Overext'] = (close / result['MA50']) - 1
    result['Near52w'] = (close / close.rolling(window=252).max()) * 100
    
    # Momentum and volume
    result['MomCons'] = compute_momentum_consistency(close, lookback=14)
    result['VolSurge'] = compute_volume_surge(volume, lookback=20)
    result['RR'] = compute_reward_risk(close, lookback=20)
    
    # Derived features for ML
    result['RR_MomCons'] = result['RR'] * result['MomCons']
    result['RSI_Neutral'] = (result['RSI'] - 50).abs()
    result['RSI_Squared'] = result['RSI'] ** 2
    result['Risk_Score'] = result['Overext'].abs() + result['ATR_Pct']
    result['Vol_Mom'] = result['VolSurge'] * result['MomCons']
    result['Overext_Mom_Div'] = result['Overext'] * result['MomCons']
    result['RR_Risk_Adj'] = result['RR'] / (1 + result['Overext'].abs())
    
    # ATR regime (simplified binning)
    result['ATR_Regime'] = pd.cut(result['ATR_Pct'], bins=[0, 0.02, 0.04, 1.0], labels=[1, 2, 3]).astype(float)
    
    # Timing indicators
    result['Vol_Breakout'] = volume > (volume.rolling(20).mean() * 1.5)
    result['Price_Breakout'] = close > close.rolling(20).quantile(0.90)
    result['Mom_Acceleration'] = (close.pct_change(5, fill_method=None) > close.shift(5).pct_change(5, fill_method=None))
    
    # Multi-period momentum returns (for TechScore_20d_v2)
    # Note: fill_method=None to avoid FutureWarning in pandas 2.1+
    result['Return_1m'] = close.pct_change(20, fill_method=None)  # ~1 month (20 trading days)
    result['Return_3m'] = close.pct_change(60, fill_method=None)  # ~3 months
    result['Return_6m'] = close.pct_change(120, fill_method=None)  # ~6 months
    
    # Additional returns for ML features v3
    result['Return_5d'] = close.pct_change(5, fill_method=None)
    result['Return_10d'] = close.pct_change(10, fill_method=None)
    
    # MA50 slope for trend component
    result['MA50_Slope'] = result['MA50'].pct_change(10, fill_method=None)  # 10-day slope
    
    # Sequential pattern features (streaks, pullbacks, etc.)
    # Streaks: consecutive up/down closes
    daily_change = close.diff()
    
    def compute_streak_column(change_series, is_up: bool):
        """Compute consecutive streak length efficiently."""
        streak_list = []
        count = 0
        for change_val in change_series:
            if pd.notna(change_val):
                if (is_up and change_val > 0) or (not is_up and change_val < 0):
                    count += 1
                else:
                    count = 0
            else:
                count = 0
            streak_list.append(count)
        return pd.Series(streak_list, index=change_series.index)
    
    result['UpStreak_Days'] = compute_streak_column(daily_change, is_up=True)
    result['DownStreak_Days'] = compute_streak_column(daily_change, is_up=False)
    
    # Rolling 20d high/low for pullback/extension metrics
    result['High_20d'] = high.rolling(20).max()
    result['Low_20d'] = low.rolling(20).min()
    result['PullbackFromHigh_20d'] = (close - result['High_20d']) / result['High_20d']
    result['DistanceFromLow_20d'] = (close - result['Low_20d']) / result['Low_20d']
    
    # Days since 20d high/low (simplified version)
    result['DaysSince20dHigh'] = 0.0  # Placeholder - can be computed with more complex logic if needed
    result['DaysSince20dLow'] = 0.0   # Placeholder
    
    # Intraday range as volatility proxy
    result['Range_Pct'] = (high - low) / close
    
    # Copy price/volume for reference
    result['Close'] = close
    result['Volume'] = volume
    result['High'] = high
    result['Low'] = low
    
    return result


def apply_technical_filters(row: pd.Series, strict: bool = True, relaxed: bool = False) -> bool:
    """
    Apply technical filters to determine if a stock qualifies for trading.
    
    This is the SINGLE SOURCE OF TRUTH for filter logic.
    All entry points must call this function with the same parameters
    to ensure consistent pass/fail decisions.
    
    Three filter tiers (mutually exclusive):
    1. **strict=True, relaxed=False** (Core): Conservative filters for stable picks
         - RSI: 25-75
         - Overextension: <=20% above MA50
         - ATR: <=12% of price
         - RR: >=-0.5
         - Momentum Consistency: >=40% up days

     2. **strict=False, relaxed=False** (Speculative): Relaxed tier for aggressive picks
         - RSI: 20-85
         - Overextension: <=30%
         - ATR: <=22%
         - RR: >=-1.0
         - Momentum Consistency: >=20% up days

     3. **relaxed=True** (Momentum-First): Ultra-aggressive, momentum-driven
         - RSI: 15-90
         - Overextension: <=40%
         - ATR: <=28%
         - RR: >=-1.5
         - Momentum Consistency: >=10% up days
    
    Args:
        row: Series containing technical indicator columns (RSI, Overext, ATR_Pct, RR, MomCons).
             Typically a single row from the output of build_technical_indicators().
        strict: If True, apply Core filters (conservative); if False, apply Speculative filters.
        relaxed: If True, override strict and apply Momentum-First filters (ultra-relaxed).
        
    Returns:
        True if the stock passes all filter thresholds, False otherwise.
        Returns True if a filter value is missing (NaN) to avoid false rejections.
    
    Raises:
        No exceptions. Missing values are treated as passing the filter.
    
    Examples:
        >>> row = pd.Series({'RSI': 45, 'Overext': 0.10, 'ATR_Pct': 0.08, 'RR': 0.5, 'MomCons': 0.50})
        >>> apply_technical_filters(row, strict=True)  # Core filters
        True
        >>> apply_technical_filters(row, strict=False)  # Speculative filters
        True
        >>> apply_technical_filters(row, relaxed=True)  # Momentum-First filters
        True
    """
    if relaxed:
        # Ultra-relaxed (Momentum-first) mode
        rsi_min, rsi_max = 15, 90
        max_overext = 0.40
        max_atr_pct = 0.28
        min_rr = -1.5
        min_mom_cons = 0.10
    elif strict:
        # Core filters (loosened)
        rsi_min, rsi_max = 25, 75
        max_overext = 0.20
        max_atr_pct = 0.12
        min_rr = -0.5
        min_mom_cons = 0.40
    else:
        # Speculative filters (relaxed tier)
        rsi_min, rsi_max = 20, 85
        max_overext = 0.30
        max_atr_pct = 0.22
        min_rr = -1.0
        min_mom_cons = 0.20
    
    # RSI check
    rsi = row.get('RSI', np.nan)
    if pd.notna(rsi) and (rsi < rsi_min or rsi > rsi_max):
        return False
    
    # Overextension check
    overext = row.get('Overext', np.nan)
    if pd.notna(overext) and abs(overext) > max_overext:
        return False
    
    # ATR/Price check
    atr_pct = row.get('ATR_Pct', np.nan)
    if pd.notna(atr_pct) and atr_pct > max_atr_pct:
        return False
    
    # Reward/Risk check
    rr = row.get('RR', np.nan)
    if pd.notna(rr) and rr < min_rr:
        return False
    
    # Momentum consistency check
    mom_cons = row.get('MomCons', np.nan)
    if pd.notna(mom_cons) and mom_cons < min_mom_cons:
        return False
    
    return True


def score_with_ml_model(row: pd.Series, model_data: Optional[Dict] = None) -> float:
    """
    Score stock with ML model - probability of positive 20-day return.
    
    Uses retrained XGBoost model with 14 engineered features.
    Model performance: AUC 0.555 (slight alpha vs 0.50 baseline).
    
    Features Used:
        Base (6): RSI, ATR_Pct, Overext, RR, MomCons, VolSurge
        Derived (8): RR_MomCons, RSI_Neutral, RSI_Squared, Risk_Score,
                     Vol_Mom, Overext_Mom_Div, RR_Risk_Adj, ATR_Regime
    
    Args:
        row: Series with technical indicator columns from build_technical_indicators()
        model_data: Dict with keys:
                   - 'model': Fitted XGBoost model object
                   - 'feature_names': List of 14 feature names in model order
                   If None or empty, returns neutral 0.5
    
    Returns:
        Float in range [0.0, 1.0] representing probability of positive 20d return.
        0.5 = neutral (no prediction)
        > 0.5 = bullish prediction
        < 0.5 = bearish prediction
    
    Raises:
        No exceptions. On any error, returns neutral 0.5.
    
    Note:
        Model is optional for scoring. Live app can function without ML model.
        Missing features are filled with sensible defaults (0.5 or 1.0).
    """
    if model_data is None or model_data.get('model') is None:
        return 0.5  # Neutral if no model
    
    try:
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Extract base features from row
        base_features = ['RSI', 'ATR_Pct', 'Overext', 'RR', 'MomCons', 'VolSurge']
        features = {}
        for fname in base_features:
            if fname in row.index:
                features[fname] = row[fname]
            else:
                # Provide sensible defaults
                features[fname] = 0.5 if fname == 'MomCons' else 1.0
        
        # Engineer derived features (must match train_recommender.py)
        X = pd.DataFrame([features])
        
        # RR_MomCons
        if 'RR' in X.columns and 'MomCons' in X.columns:
            X['RR_MomCons'] = X['RR'] * X['MomCons']
        
        # RSI_Neutral, RSI_Squared
        if 'RSI' in X.columns:
            X['RSI_Neutral'] = (X['RSI'] - 50).abs()
            X['RSI_Squared'] = X['RSI'] ** 2
        
        # Risk_Score
        if 'Overext' in X.columns and 'ATR_Pct' in X.columns:
            X['Risk_Score'] = X['Overext'].abs() + X['ATR_Pct']
        
        # Vol_Mom
        if 'VolSurge' in X.columns and 'MomCons' in X.columns:
            X['Vol_Mom'] = X['VolSurge'] * X['MomCons']
        
        # Overext_Mom_Div
        if 'Overext' in X.columns and 'MomCons' in X.columns:
            X['Overext_Mom_Div'] = X['Overext'] * X['MomCons']
        
        # RR_Risk_Adj
        if 'RR' in X.columns and 'Overext' in X.columns:
            X['RR_Risk_Adj'] = X['RR'] / (1 + X['Overext'].abs())
        
        # ATR_Regime
        if 'ATR_Pct' in X.columns:
            # Simple regime: low (<0.02), med (0.02-0.04), high (>0.04)
            atr_val = X['ATR_Pct'].iloc[0]
            if atr_val < 0.02:
                X['ATR_Regime'] = 1.0
            elif atr_val < 0.04:
                X['ATR_Regime'] = 2.0
            else:
                X['ATR_Regime'] = 3.0
        
        # Ensure all model features exist
        for fname in feature_names:
            if fname not in X.columns:
                X[fname] = 0.5  # fallback
        
        # Select features in correct order
        X_model = X[feature_names].fillna(0.5)
        
        # Predict probability
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(X_model.values)[0][1])
        else:
            # XGBoost API
            prob = float(model.predict_proba(X_model)[0][1])
        
        # Clip to valid range
        return max(0.0, min(1.0, prob))
        
    except Exception as e:
        # Fallback to neutral on any error
        return 0.5


def compute_forward_returns(
    df: pd.DataFrame, 
    date: pd.Timestamp, 
    horizons: List[int] = [5, 10, 20],
    benchmark_df: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Compute forward (future) returns from a specific date - for backtesting.
    
    Looks forward from the given date and calculates returns at multiple horizons.
    Optionally computes excess returns vs. a benchmark (e.g., SPY).
    
    Args:
        df: DataFrame with Close prices, indexed by date (DatetimeIndex)
        date: Starting date (must be in df.index)
        horizons: List of forward periods in trading days (default [5, 10, 20])
        benchmark_df: Optional DataFrame with Close prices for relative return calculation
    
    Returns:
        Dict with keys:
        - 'R_5d', 'R_10d', 'R_20d': Simple returns (%) at each horizon
        - 'Excess_5d', 'Excess_10d', 'Excess_20d': Return vs benchmark (%)
        All values are float or NaN if date is not found or beyond data end
    
    Raises:
        No exceptions. Missing dates return NaN values.
    
    Examples:
        >>> fwd = compute_forward_returns(df, pd.Timestamp('2024-01-15'), horizons=[5, 10, 20])
        >>> fwd['R_5d']  # Return 5 trading days forward
        2.35  # +2.35%
        >>> fwd['Excess_10d']  # 10-day excess vs benchmark
        1.80  # Outperformed benchmark by 1.80%
    
    Note:
        Handles potential Series vs scalar values from DataFrame indexing.
        Gracefully returns NaN for out-of-bounds dates.
    """
    results = {}
    
    if date not in df.index:
        return {f'R_{h}d': np.nan for h in horizons}
    
    idx = df.index.get_loc(date)
    
    # Extract close price (handle potential Series)
    close_val = df.loc[date, 'Close']
    if isinstance(close_val, pd.Series):
        price_start = float(close_val.iloc[0])
    else:
        price_start = float(close_val)
    
    for h in horizons:
        end_idx = idx + h
        if end_idx >= len(df):
            results[f'R_{h}d'] = np.nan
            results[f'Excess_{h}d'] = np.nan
            continue
        
        end_date = df.index[end_idx]
        
        # Extract end price (handle potential Series)
        close_end_val = df.loc[end_date, 'Close']
        if isinstance(close_end_val, pd.Series):
            price_end = float(close_end_val.iloc[0])
        else:
            price_end = float(close_end_val)
        
        # Stock return
        ret = (price_end / price_start - 1) * 100
        results[f'R_{h}d'] = float(ret)
        
        # Benchmark return
        if benchmark_df is not None and date in benchmark_df.index:
            bench_idx = benchmark_df.index.get_loc(date)
            bench_end_idx = bench_idx + h
            
            if bench_end_idx < len(benchmark_df):
                bench_end_date = benchmark_df.index[bench_end_idx]
                
                bench_start_val = benchmark_df.loc[date, 'Close']
                bench_end_val = benchmark_df.loc[bench_end_date, 'Close']
                
                # Handle potential Series
                if isinstance(bench_start_val, pd.Series):
                    bench_start = float(bench_start_val.iloc[0])
                else:
                    bench_start = float(bench_start_val)
                    
                if isinstance(bench_end_val, pd.Series):
                    bench_end = float(bench_end_val.iloc[0])
                else:
                    bench_end = float(bench_end_val)
                
                bench_ret = (bench_end / bench_start - 1) * 100
                results[f'Excess_{h}d'] = float(ret - bench_ret)
            else:
                results[f'Excess_{h}d'] = np.nan
        else:
            results[f'Excess_{h}d'] = np.nan
    
    return results


def compute_technical_score(row: pd.Series, weights: Optional[Dict[str, float]] = None) -> float:
    """Compute a deterministic technical score (0-100) from indicators in `row`.
    
    This is the SINGLE SOURCE OF TRUTH for technical scoring.
    All entry points must call this function with identical parameters
    to ensure consistent, reproducible results.
    
    Scoring formula (November 2025 - Volatility Adjusted):
    - Component scores (0-1): MA alignment, momentum, RSI, volume, overextension, 
      pullback strength, risk/reward, MACD, ADX
    - Weights: Configurable via `weights` parameter; defaults to balanced allocation
    - Volatility adjustment:
      * ATR > 6% (extreme volatility): 50% score penalty
      * ATR 5-6% (high volatility): 30% score penalty
      * ATR 2-4% (sweet spot): 20% score bonus
      * ATR < 2% (low volatility): 20% score penalty
    - Risk/reward is volatility-adjusted: RR_adjusted = RR * vol_multiplier
    - Final score clipped to [0, 100]
    
    Args:
        row: Series with technical indicator columns from build_technical_indicators().
             Expected columns: MA_Aligned, Momentum_Consistency, RSI, VolSurge,
             Overext, Near52w, RR, ATR_Pct, MACD_Pos, ADX14.
             Missing values default to neutral (0.5) or 0.
        weights: Optional dict mapping component names to weight values:
                 {'ma', 'mom', 'rsi', 'near_high_bell', 'vol', 'overext',
                  'pullback', 'risk_reward', 'macd', 'adx'}.
                 Defaults to balanced allocation if None.
                 Weights are normalized internally to sum to 1.0.
        
    Returns:
        Float between 0.0 and 100.0 representing technical conviction score.
        Higher scores indicate stronger technical alignment.
        
    Raises:
        No exceptions. Missing indicator values handled gracefully with defaults.
    
    Examples:
        >>> row = pd.Series({'MA_Aligned': 1.0, 'RSI': 50, 'Overext': 0.05, 
        ...                   'ATR_Pct': 0.03, 'VolSurge': 1.5, 'Momentum_Consistency': 0.6})
        >>> compute_technical_score(row)  # Uses default weights
        78.5
        >>> custom_w = {'ma': 0.4, 'mom': 0.3, ...}  # Custom weights
        >>> compute_technical_score(row, weights=custom_w)
        82.1
    
    Note:
        Volatility adjustment is ALWAYS applied, regardless of weights parameter.
        Extreme volatility (ATR > 6%) significantly reduces scores to penalize
        high-risk, high-uncertainty trading opportunities.
    """
    default_weights = {
        'ma': 0.2,
        'mom': 0.25,
        'rsi': 0.12,
        'near_high_bell': 0.10,
        'vol': 0.08,
        'overext': 0.06,
        'pullback': 0.05,
        'risk_reward': 0.06,
        'macd': 0.04,
        'adx': 0.04,
    }

    # Sanitize weights: coerce to float, clamp negatives, drop invalid to 0.0
    raw_weights = default_weights.copy()
    if weights is not None:
        raw_weights.update(weights)

    def _safe_weight(val: float) -> float:
        try:
            w = float(val)
            if not np.isfinite(w):
                return 0.0
            return max(w, 0.0)
        except Exception:
            return 0.0

    sanitized = {k: _safe_weight(v) for k, v in raw_weights.items()}
    total_w = sum(sanitized.values())

    # If everything zero/invalid, fall back to defaults
    if total_w <= 0:
        sanitized = default_weights
        total_w = sum(sanitized.values())

    # Normalize weights to sum 1
    norm_w = {k: (v / total_w) for k, v in sanitized.items()}

    # Extract components from row (use safe defaults)
    ma_ok = float(row.get('MA_Aligned', 0.0)) if pd.notna(row.get('MA_Aligned', np.nan)) else 0.0
    mom = float(row.get('Momentum_Consistency', 0.0)) if pd.notna(row.get('Momentum_Consistency', np.nan)) else 0.0
    
    # RSI score: closer to mid-band (50) is neutral; reward 50-40/60 range
    rsi = row.get('RSI', np.nan)
    if pd.isna(rsi):
        rsi_score = 0.5
    else:
        # Map RSI 0-100 to 0-1 with preference for 25-75
        if 25 <= rsi <= 75:
            rsi_score = 1.0
        else:
            rsi_score = max(0.0, 1.0 - (abs(rsi - 50) - 25) / 50.0)

    vol = float(row.get('VolSurge', 1.0)) if pd.notna(row.get('VolSurge', np.nan)) else 1.0
    vol_score = min(2.0, vol) / 2.0

    overext = float(row.get('Overext', 0.0)) if pd.notna(row.get('Overext', np.nan)) else 0.0
    overext_score = max(0.0, 1.0 - (overext / 0.2))

    pullback = float(row.get('Near52w', np.nan)) / 100.0 if pd.notna(row.get('Near52w', np.nan)) else 0.5

    # VOLATILITY-ADJUSTED RISK/REWARD
    # Penalize high volatility, reward stable movers
    rr = float(row.get('RR', np.nan)) if pd.notna(row.get('RR', np.nan)) else 1.0
    atr_pct = float(row.get('ATR_Pct', np.nan)) if pd.notna(row.get('ATR_Pct', np.nan)) else 0.03
    
    # Volatility penalty/bonus
    if atr_pct > 0.06:  # >6% = extreme volatility
        vol_adjustment = 0.5  # Heavy penalty
    elif atr_pct > 0.05:  # 5-6% = high volatility
        vol_adjustment = 0.7
    elif atr_pct < 0.02:  # <2% = too stable (low opportunity)
        vol_adjustment = 0.8
    elif 0.02 <= atr_pct <= 0.04:  # Sweet spot: 2-4%
        vol_adjustment = 1.2  # Bonus!
    else:  # 4-5% = acceptable
        vol_adjustment = 1.0
    
    # Risk-adjusted RR = RR * volatility adjustment
    adjusted_rr = rr * vol_adjustment
    rr_score = max(0.0, min(1.0, (adjusted_rr + 1.0) / 5.0))

    macd = 1.0 if row.get('MACD_Pos', False) else 0.0
    adx = float(row.get('ADX14', 0.0)) if pd.notna(row.get('ADX14', np.nan)) else 0.0
    adx_score = np.clip((adx - 15.0) / 20.0, 0.0, 1.0)

    tech = (
        norm_w['ma'] * ma_ok
        + norm_w['mom'] * mom
        + norm_w['rsi'] * rsi_score
        + norm_w['near_high_bell'] * pullback
        + norm_w['vol'] * vol_score
        + norm_w['overext'] * overext_score
        + norm_w['pullback'] * pullback
        + norm_w['risk_reward'] * rr_score  # Now volatility-adjusted!
        + norm_w['macd'] * macd
        + norm_w['adx'] * adx_score
    )

    # Additional volatility penalty for extreme cases
    if atr_pct > 0.06:
        tech *= 0.85  # 15% penalty to overall score
    
    # scale to 0-100
    return float(np.clip(tech * 100.0, 0.0, 100.0))


def compute_final_score(
    tech_score: float, 
    fundamental_score: Optional[float] = None, 
    ml_prob: Optional[float] = None
) -> float:
    """
    Compute unified final score combining technical, fundamental, and ML components.
    
    This is the SINGLE SOURCE OF TRUTH for final score calculation.
    All entry points must call this function to ensure consistency.
    
    Formula:
        final_score = 0.55 * tech_score + 0.25 * fund_score + 0.20 * (ml_prob * 100)
    
    Where:
    - **tech_score** (0-100): Technical analysis score from compute_technical_score()
    - **fund_score** (0-100): Fundamental valuation score from compute_fundamental_score_with_breakdown()
      If None or NaN, defaults to 0 (no fundamental penalty)
    - **ml_prob** (0-1): ML model prediction probability
      If None or NaN, defaults to 0.5 (neutral)
    - **final score** (0-100): Weighted combination, clipped to [0, 100]
    
    Weight Rationale:
    - Technical (55%): Primary signal; most actionable and immediate
    - Fundamental (25%): Secondary confirmation; reduces false positives
    - ML (20%): Tertiary enhancement; model confidence boost
    
    Args:
        tech_score: Technical score from build_technical_indicators() + compute_technical_score().
                    Must be in range [0, 100].
        fundamental_score: Optional fundamental valuation score [0, 100].
                          Defaults to 0 if None or NaN.
        ml_prob: Optional ML model probability [0, 1].
                 Defaults to 0.5 (neutral) if None or NaN.
    
    Returns:
        Float between 0.0 and 100.0 representing final conviction score.
        Clipped to valid range regardless of input values.
    
    Raises:
        No exceptions. Invalid inputs handled gracefully with defaults.
    
    Examples:
        >>> compute_final_score(tech_score=75.0, fundamental_score=65.0, ml_prob=0.65)
        72.0  # = 0.55*75 + 0.25*65 + 0.20*65
        
        >>> compute_final_score(tech_score=80.0)  # No fundamentals or ML
        44.0  # = 0.55*80 + 0.25*0 + 0.20*50 (uses defaults)
        
        >>> compute_final_score(tech_score=60.0, fundamental_score=np.nan, ml_prob=1.0)
        65.0  # NaN fund score treated as 0
    
    Note:
        Always returns a valid numeric value suitable for sorting/ranking stocks.
        No exception is raised for invalid inputs; defaults are applied instead.
    """
    fund = 0.0 if fundamental_score is None or pd.isna(fundamental_score) else float(fundamental_score)
    ml_score = 0.5 if ml_prob is None or pd.isna(ml_prob) else float(ml_prob)
    final = 0.55 * tech_score + 0.25 * fund + 0.20 * (ml_score * 100.0)
    return float(np.clip(final, 0.0, 100.0))


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data from Yahoo Finance.
    
    This is the preferred method for retrieving price history.
    Uses auto_adjust=True to handle splits/dividends automatically.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'SPY')
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
    
    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex
        Returns None if fetch fails or no data available
    
    Raises:
        No exceptions raised. Errors are caught and None is returned.
    
    Note:
        Data is auto-adjusted for splits and dividends by yfinance.
        May take a few seconds for large date ranges.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df.empty:
            return None
        return df
    except Exception:
        return None


# ============================================================================
# V2 SCORING ENGINE INTEGRATION (NON-BREAKING)
# ============================================================================

def score_ticker_v2(
    ticker: str,
    row: pd.Series,
    historical_df: Optional[pd.DataFrame] = None,
    enable_ml: bool = True,
    use_multi_source: bool = True
) -> Dict:
    """
    V2 scoring pipeline using multi-source data and unified scoring engine.
    
    This is a NON-BREAKING addition. V1 scoring remains unchanged.
    
    Args:
        ticker: Stock symbol
        row: Row from technical indicators DataFrame (contains RSI, ATR, etc.)
        historical_df: Full OHLCV DataFrame (optional, for additional calculations)
        enable_ml: Whether to use ML boost
        use_multi_source: Whether to fetch multi-source fundamentals
    
    Returns:
        Dict with all v2 scores and metadata:
        {
            "ticker": str,
            "fundamental_score_v2": float (0-100),
            "fundamental_confidence_v2": float (0-100),
            "technical_score_v2": float (0-100),
            "technical_confidence_v2": float (0-100),
            "rr_score_v2": float (0-100),
            "rr_confidence_v2": float (0-100),
            "reliability_score_v2": float (0-100),
            "conviction_v2_base": float (0-100),  # Before ML
            "conviction_v2_final": float (0-100),  # After ML
            "ml_probability": Optional[float],
            "ml_boost": float,
            "ml_status": str,
            "risk_meter_v2": float (0-100),
            "risk_label_v2": str,
            "warnings_v2": List[str],
            "sources_used": List[str],
            "disagreement_score": float,
            "breakdown": Dict  # Detailed component scores
        }
    """
    from core import scoring_engine
    from core import data_sources_v2
    from core import ml_integration
    
    result = {
        "ticker": ticker,
        "fundamental_score_v2": 50.0,
        "fundamental_confidence_v2": 0.0,
        "technical_score_v2": 50.0,
        "technical_confidence_v2": 0.0,
        "rr_score_v2": 50.0,
        "rr_confidence_v2": 0.0,
        "reliability_score_v2": 50.0,
        "conviction_v2_base": 50.0,
        "conviction_v2_final": 50.0,
        "ml_probability": None,
        "ml_boost": 0.0,
        "ml_status": "Not computed",
        "risk_meter_v2": 50.0,
        "risk_label_v2": "MODERATE",
        "warnings_v2": [],
        "sources_used": [],
        "disagreement_score": 0.0,
        "breakdown": {}
    }
    
    try:
        # ===== STEP 1: Fetch Multi-Source Fundamentals =====
        if use_multi_source:
            try:
                multi_source_data = data_sources_v2.fetch_multi_source_data(ticker)
                result["sources_used"] = multi_source_data.get("sources_used", [])
                result["disagreement_score"] = multi_source_data.get("disagreement_score", 0.0)
            except Exception as e:
                import logging
                logging.warning(f"Multi-source fetch failed for {ticker}: {e}")
                multi_source_data = {}
        else:
            multi_source_data = {}
        
        # ===== STEP 2: Calculate Fundamental Score =====
        fund_score, fund_confidence = scoring_engine.calculate_fundamental_score(
            pe=multi_source_data.get("pe"),
            ps=multi_source_data.get("ps"),
            pb=multi_source_data.get("pb"),
            roe=multi_source_data.get("roe"),
            margin=multi_source_data.get("margin"),
            rev_yoy=multi_source_data.get("rev_yoy"),
            eps_yoy=multi_source_data.get("eps_yoy"),
            debt_equity=multi_source_data.get("debt_equity")
        )
        
        result["fundamental_score_v2"] = fund_score
        result["fundamental_confidence_v2"] = fund_confidence
        
        # ===== STEP 3: Calculate Technical/Momentum Score =====
        # Extract technical indicators from row
        rsi = row.get("RSI", np.nan)
        atr = row.get("ATR", np.nan)
        close = row.get("Close", np.nan) if historical_df is None else historical_df['Close'].iloc[-1]
        atr_pct = (atr / close) if (np.isfinite(atr) and np.isfinite(close) and close > 0) else np.nan
        
        ma_aligned = row.get("MA_Aligned", False)
        overextension = row.get("OverextRatio", np.nan)
        volume_surge = row.get("Volume_Surge", np.nan)
        near_high = row.get("Near52w", np.nan)
        
        # Momentum periods (if available)
        mom_1m = row.get("Mom_1M", np.nan)
        mom_3m = row.get("Mom_3M", np.nan)
        mom_6m = row.get("Mom_6M", np.nan)
        
        tech_score, tech_confidence = scoring_engine.calculate_momentum_score(
            rsi=rsi,
            atr_pct=atr_pct,
            ma_aligned=ma_aligned,
            mom_1m=mom_1m,
            mom_3m=mom_3m,
            mom_6m=mom_6m,
            near_high=near_high,
            overextension=overextension,
            volume_surge=volume_surge
        )
        
        result["technical_score_v2"] = tech_score
        result["technical_confidence_v2"] = tech_confidence
        
        # ===== STEP 4: Calculate Risk/Reward Score =====
        rr_ratio = row.get("RewardRisk", np.nan)
        support = row.get("Support_20d", np.nan)
        resistance = row.get("Resistance_20d", np.nan)
        current_price = close
        
        rr_score, rr_confidence = scoring_engine.calculate_rr_score(
            rr_ratio=rr_ratio,
            atr=atr,
            support=support,
            resistance=resistance,
            current_price=current_price
        )
        
        result["rr_score_v2"] = rr_score
        result["rr_confidence_v2"] = rr_confidence
        
        # ===== STEP 5: Calculate Reliability Score =====
        price_sources = multi_source_data.get("price_sources", 1)
        fund_sources = len(result["sources_used"])
        price_std = multi_source_data.get("price_std", np.nan)
        price_mean = multi_source_data.get("price_mean", np.nan)
        
        # Data completeness: how many technical indicators are valid?
        tech_fields = ["RSI", "ATR", "MA_Aligned", "OverextRatio", "Volume_Surge", "Near52w", "RewardRisk"]
        valid_count = sum(1 for f in tech_fields if np.isfinite(row.get(f, np.nan)))
        data_completeness = (valid_count / len(tech_fields)) * 100.0
        
        reliability_score = scoring_engine.calculate_reliability_score(
            price_sources=price_sources,
            fund_sources=fund_sources,
            price_std=price_std,
            price_mean=price_mean,
            fundamental_confidence=fund_confidence,
            data_completeness=data_completeness
        )
        
        result["reliability_score_v2"] = reliability_score
        
        # ===== STEP 6: Calculate Base Conviction =====
        base_conviction, breakdown = scoring_engine.calculate_conviction_score(
            fundamental_score=fund_score,
            fundamental_confidence=fund_confidence,
            momentum_score=tech_score,
            momentum_confidence=tech_confidence,
            rr_score=rr_score,
            rr_confidence=rr_confidence,
            reliability_score=reliability_score,
            ml_probability=None  # Add ML separately
        )
        
        result["conviction_v2_base"] = base_conviction
        result["breakdown"] = breakdown
        
        # ===== STEP 7: Apply ML Boost (if enabled) =====
        if enable_ml:
            # Prepare data for ML
            ticker_data = {
                "near_high": near_high,
                "market_trend": row.get("Market_Trend", 0.0),
                "market_volatility": row.get("Market_Volatility", 0.02),
                "spy_rsi": row.get("SPY_RSI", 50.0),
                "relative_strength_20d": row.get("Relative_Strength_20d", 0.0),
                "dist_from_52w_high": row.get("Dist_From_52w_High", 0.1),
                "mom_acceleration": row.get("Mom_Acceleration", 0.0)
            }
            
            technical_indicators = {
                "rsi": rsi,
                "atr_pct": atr_pct,
                "overextension": overextension,
                "rr_ratio": rr_ratio,
                "momentum_consistency": row.get("Momentum_Consistency", 0.5),
                "volume_surge": volume_surge
            }
            
            fundamental_scores_dict = {
                "fund_score": fund_score,
                "fund_confidence": fund_confidence
            }
            
            final_conviction, ml_info = ml_integration.integrate_ml_with_conviction(
                base_conviction=base_conviction,
                ticker_data=ticker_data,
                technical_indicators=technical_indicators,
                fundamental_scores=fundamental_scores_dict,
                enable_ml=True
            )
            
            result["conviction_v2_final"] = final_conviction
            result["ml_probability"] = ml_info.get("ml_probability")
            result["ml_boost"] = ml_info.get("ml_boost", 0.0)
            result["ml_status"] = ml_info.get("ml_status", "Unknown")
        else:
            result["conviction_v2_final"] = base_conviction
            result["ml_status"] = "ML disabled"
        
        # ===== STEP 8: Calculate Risk Meter =====
        beta = multi_source_data.get("beta", np.nan)
        leverage = multi_source_data.get("debt_equity", np.nan)
        
        risk_meter, risk_label = scoring_engine.calculate_risk_meter(
            rr_ratio=rr_ratio,
            beta=beta,
            atr_pct=atr_pct,
            leverage=leverage
        )
        
        result["risk_meter_v2"] = risk_meter
        result["risk_label_v2"] = risk_label
        
        # ===== STEP 9: Generate Warnings =====
        warnings = scoring_engine.generate_warnings(
            rr_ratio=rr_ratio,
            fundamental_confidence=fund_confidence,
            beta=beta,
            atr_pct=atr_pct,
            reliability_score=reliability_score
        )
        
        result["warnings_v2"] = warnings
        
    except Exception as e:
        import logging
        logging.error(f"V2 scoring failed for {ticker}: {e}")
        result["warnings_v2"].append(f"⚠️ V2 scoring error: {str(e)[:100]}")
    
    return result


def batch_score_v2(
    tickers: List[str],
    indicators_df: pd.DataFrame,
    enable_ml: bool = True,
    use_multi_source: bool = True
) -> pd.DataFrame:
    """
    Batch v2 scoring for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        indicators_df: DataFrame with technical indicators (index = tickers)
        enable_ml: Whether to use ML
        use_multi_source: Whether to fetch multi-source data
    
    Returns:
        DataFrame with all v2 scores (one row per ticker)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    results = []
    
    for ticker in tickers:
        if ticker not in indicators_df.index:
            logger.warning(f"Ticker {ticker} not in indicators DataFrame")
            continue
        
        row = indicators_df.loc[ticker]
        
        try:
            v2_result = score_ticker_v2(
                ticker=ticker,
                row=row,
                enable_ml=enable_ml,
                use_multi_source=use_multi_source
            )
            results.append(v2_result)
        except Exception as e:
            logger.error(f"Failed to score {ticker}: {e}")
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df.set_index("ticker", inplace=True)
    
    return df


# ============================================================================
# MARKET CONTEXT & REGIME CLASSIFICATION
# ============================================================================

def build_market_context_table(
    start_date: str,
    end_date: str,
    provider_status: Optional[Dict[str, bool]] = None
) -> pd.DataFrame:
    """
    Build a market context table keyed by date with SPY/VIX features and regime classification.
    
    Computes for each trading date:
    - SPY returns (20d, 60d)
    - SPY drawdown from 60d high
    - VIX percentile (vs 1-year rolling window)
    - Market regime classification (TREND_UP, SIDEWAYS, CORRECTION, PANIC)
    - One-hot encoded regime flags
    
    Args:
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'  
        provider_status: Optional provider availability dict
    
    Returns:
        DataFrame with columns:
        - date: trading date
        - SPY_20d_ret, SPY_60d_ret: SPY returns
        - SPY_drawdown_60d: drawdown from 60d high
        - VIX_close, VIX_pct: VIX level and percentile
        - Market_Regime: categorical string
        - Regime_TrendUp, Regime_Sideways, Regime_Correction, Regime_Panic: binary flags
    """
    from core.data_sources_v2 import get_index_series
    import logging
    logger = logging.getLogger(__name__)
    
    # Extend date range to compute rolling windows
    start_dt = pd.to_datetime(start_date) - timedelta(days=400)  # ~1.5 years buffer
    end_dt = pd.to_datetime(end_date)
    
    start_extended = start_dt.strftime('%Y-%m-%d')
    end_extended = end_dt.strftime('%Y-%m-%d')
    
    # Fetch SPY data
    spy_df = get_index_series('SPY', start_extended, end_extended, provider_status)
    if spy_df is None or spy_df.empty:
        logger.error("Failed to fetch SPY data for market context")
        return pd.DataFrame()
    
    # Fetch VIX data (optional, use defaults if unavailable)
    vix_df = get_index_series('^VIX', start_extended, end_extended, provider_status)
    if vix_df is None or vix_df.empty:
        logger.warning("VIX data unavailable, using default neutral values")
        vix_df = pd.DataFrame({
            'date': spy_df['date'],
            'close': 20.0  # Neutral VIX level
        })
    
    # Prepare SPY features
    spy_df = spy_df.sort_values('date').reset_index(drop=True)
    spy_df['SPY_20d_ret'] = spy_df['close'].pct_change(20)
    spy_df['SPY_60d_ret'] = spy_df['close'].pct_change(60)
    spy_df['SPY_high_60d'] = spy_df['close'].rolling(60).max()
    spy_df['SPY_drawdown_60d'] = (spy_df['close'] / spy_df['SPY_high_60d']) - 1.0
    
    # Prepare VIX features
    vix_df = vix_df.sort_values('date').reset_index(drop=True)
    vix_df['VIX_close'] = vix_df['close']
    # VIX percentile over 252 days (~1 year)
    vix_df['VIX_pct'] = vix_df['VIX_close'].rolling(252, min_periods=60).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
    )
    
    # Merge SPY and VIX
    context_df = spy_df[['date', 'SPY_20d_ret', 'SPY_60d_ret', 'SPY_drawdown_60d']].copy()
    context_df = context_df.merge(
        vix_df[['date', 'VIX_close', 'VIX_pct']],
        on='date',
        how='left'
    )
    
    # Fill any missing VIX values
    context_df['VIX_close'] = context_df['VIX_close'].fillna(20.0)
    context_df['VIX_pct'] = context_df['VIX_pct'].fillna(0.5)
    
    # Classify market regime
    def classify_regime(row):
        """
        Classify market regime based on SPY performance and VIX.
        
        PANIC: SPY drawdown < -15% OR VIX percentile > 85%
        CORRECTION: SPY drawdown < -8% OR VIX percentile > 70%
        TREND_UP: SPY 60d return > +8% AND drawdown > -5%
        SIDEWAYS: everything else
        """
        dd = row.get('SPY_drawdown_60d', 0)
        ret_60d = row.get('SPY_60d_ret', 0)
        vix_pct = row.get('VIX_pct', 0.5)
        
        # Handle NaN values
        if pd.isna(dd) or pd.isna(ret_60d) or pd.isna(vix_pct):
            return 'SIDEWAYS'
        
        # Panic conditions
        if dd < -0.15 or vix_pct > 0.85:
            return 'PANIC'
        
        # Correction conditions
        if dd < -0.08 or vix_pct > 0.70:
            return 'CORRECTION'
        
        # Trend up conditions
        if ret_60d > 0.08 and dd > -0.05:
            return 'TREND_UP'
        
        # Default sideways
        return 'SIDEWAYS'
    
    context_df['Market_Regime'] = context_df.apply(classify_regime, axis=1)
    
    # One-hot encode regime flags
    context_df['Regime_TrendUp'] = (context_df['Market_Regime'] == 'TREND_UP').astype(int)
    context_df['Regime_Sideways'] = (context_df['Market_Regime'] == 'SIDEWAYS').astype(int)
    context_df['Regime_Correction'] = (context_df['Market_Regime'] == 'CORRECTION').astype(int)
    context_df['Regime_Panic'] = (context_df['Market_Regime'] == 'PANIC').astype(int)
    
    # Filter to requested date range
    context_df = context_df[
        (context_df['date'] >= pd.to_datetime(start_date)) &
        (context_df['date'] <= pd.to_datetime(end_date))
    ].copy()
    
    logger.info(f"✓ Built market context table: {len(context_df)} dates")
    logger.info(f"  Regime distribution: {context_df['Market_Regime'].value_counts().to_dict()}")
    
    return context_df
