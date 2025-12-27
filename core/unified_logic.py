
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import logging
import numpy as np
import pandas as pd

from core.classifier import apply_classification
from core.scoring_engine import (
    calculate_reliability_score,
    calculate_risk_meter,
    compute_overall_score,
)
from core.scoring_config import (
    ADVANCED_FILTER_DEFAULTS,
    ATR_RULES,
    FINAL_SCORE_WEIGHTS,
    TECH_WEIGHTS,
    REQUIRED_TECH_COLS,
    MultiSourceData,
)

logger = logging.getLogger(__name__)


def apply_technical_filters(row: pd.Series, strict: bool = True, relaxed: bool = False) -> bool:
    """Apply basic technical filters to determine if a stock passes initial screening.
    
    Args:
        row: Series with technical indicators
        strict: If True, apply strict volume/liquidity rules
        relaxed: If True, apply very lenient rules
        
    Returns:
        bool: True if stock passes filters, False otherwise
    """
    # Minimal volume check
    volume = row.get("Volume", 0)
    if volume < 100000 and not relaxed:  # Min 100k volume
        return False
    
    # Basic price sanity
    close = row.get("Close", 0)
    if close < 1.0 or close > 10000:  # Filter penny stocks and extreme prices
        return False
        
    return True


@dataclass
class RecommendationResult:
    """Structured representation of a single ticker recommendation."""

    ticker: str
    tech_score: float
    fundamental_score: float
    ml_prob: float
    final_score: float
    conviction_score: float
    reliability_score: float
    risk_label: str
    risk_meter: float
    should_display: bool
    extras: Dict[str, Any]

    def to_series(self) -> pd.Series:
        base = {
            "Ticker": self.ticker,
            "TechScore_20d": self.tech_score,
            "Fundamental_Score": self.fundamental_score,
            "ML_20d_Prob": self.ml_prob,
            "FinalScore_20d": self.final_score,
            "ConvictionScore": self.conviction_score,
            "Reliability_Score": self.reliability_score,
            "Risk_Label": self.risk_label,
            "Risk_Meter": self.risk_meter,
            "Should_Display": self.should_display,
            # Legacy alias
            "Score": self.final_score,
        }
        base.update(self.extras)
        return pd.Series(base, dtype="object")

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
    ml_prob_override: Optional[float] = None,
    multi_source_override: Optional[MultiSourceData] = None,
) -> pd.Series:
    """Compute all recommendation scores and labels for a single stock row.

    **Single source of truth** for all scoring throughout Stock Scout.
    Called by pipeline_runner, UI, CSV export, and backtests.

    Args:
        row: Technical indicators from build_technical_indicators (last row)
        ticker: Stock symbol (optional if row contains "Ticker" or "ticker")
        as_of_date: Optional timestamp for the recommendation
        enable_ml: Whether to compute ML_20d_Prob (requires ML model)
        use_multi_source: Whether to fetch multi-provider fundamentals
        ml_prob_override: Force a specific ML probability (testing/offline)
        multi_source_override: Force specific fundamental data (testing/offline)

    Returns:
        pd.Series with canonical columns:
            - Ticker: str
            - TechScore_20d: float (0-100) - Technical score for 20d horizon
            - Fundamental_Score: float (0-100) - Fundamental quality score
            - ML_20d_Prob: float (0-1) - ML model probability for 20d gain
            - FinalScore_20d: float (0-100) - Combined score (tech + fund + ML)
            - ConvictionScore: float (0-100) - Conviction metric
            - Reliability_Score: float (0-100) - Data quality/multi-source reliability
            - Risk_Label: str - Human-readable risk category
            - Risk_Meter: float (0-100) - Numeric risk level (higher = riskier)
            - Should_Display: bool - Whether to show in UI
            - Score: float (0-100) - Legacy alias for FinalScore_20d
            - Plus all input row columns preserved
            - Plus classification columns from apply_classification

    Raises:
        ValueError: If ticker cannot be determined from inputs
    """
    base_ticker = ticker or row.get("Ticker") or row.get("ticker")
    if base_ticker is None:
        raise ValueError("compute_recommendation_scores requires a ticker")

    # --- Multi-source fundamentals (optional) ---
    ms_data = multi_source_override
    if ms_data is None and use_multi_source:
        try:
            from core import data_sources_v2

            raw = data_sources_v2.fetch_multi_source_data(base_ticker)
            ms_data = MultiSourceData.from_dict(raw)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("multi-source fetch failed for %s: %s", base_ticker, exc)
            ms_data = MultiSourceData()
    elif ms_data is None:
        ms_data = MultiSourceData()

    # --- Technical component (prefer v2 raw 0-1 scaled to 0-100) ---
    try:
        tech_raw = compute_tech_score_20d_v2(row)
        tech_score = float(np.clip(tech_raw * 100.0, 0.0, 100.0))
    except Exception:
        tech_score = compute_technical_score(row)

    # --- Fundamental component ---
    from core.scoring import compute_fundamental_score_with_breakdown
    
    fund_input = asdict(ms_data)
    fundamental = compute_fundamental_score_with_breakdown(fund_input)
    fundamental_score = float(fundamental.total) if fundamental else 0.0

    # --- ML component ---
    ml_prob = ml_prob_override
    if ml_prob is None and enable_ml:
        ml_prob = row.get("ML_20d_Prob_raw") or row.get("ML_20d_Prob")
        if ml_prob is None:
            try:
                from core.ml_20d_inference import ML_20D_AVAILABLE, predict_20d_prob_from_row

                if ML_20D_AVAILABLE:
                    ml_prob = predict_20d_prob_from_row(row)
            except Exception:
                ml_prob = None
    if ml_prob is None:
        ml_prob = 0.5

    # --- Final & conviction ---
    final_score = compute_final_score(tech_score, fundamental_score, ml_prob)
    conviction_score = final_score

    # --- Reliability & risk ---
    tech_fields = [
        "RSI",
        "ATR",
        "MA_Aligned",
        "Overext",
        "VolSurge",
        "Near52w",
        "RR",
    ]
    valid_count = sum(1 for f in tech_fields if pd.notna(row.get(f, np.nan)))
    data_completeness = (valid_count / len(tech_fields)) * 100.0
    fundamental_confidence = data_completeness  # proxy when coverage not provided

    reliability_score = calculate_reliability_score(
        price_sources=ms_data.price_sources or 0,
        fund_sources=len(ms_data.sources_used),
        price_std=ms_data.price_std,
        price_mean=ms_data.price_mean,
        fundamental_confidence=fundamental_confidence,
        data_completeness=data_completeness,
    )

    risk_meter, risk_label = calculate_risk_meter(
        rr_ratio=row.get("RR"),
        beta=ms_data.beta,
        atr_pct=row.get("ATR_Pct"),
        leverage=ms_data.debt_to_equity,
    )

    rec = RecommendationResult(
        ticker=base_ticker,
        tech_score=tech_score,
        fundamental_score=fundamental_score,
        ml_prob=float(ml_prob),
        final_score=final_score,
        conviction_score=conviction_score,
        reliability_score=reliability_score,
        risk_label=risk_label,
        risk_meter=risk_meter,
        should_display=True,
        extras={
            "Fundamental_Breakdown": getattr(fundamental, "breakdown", None),
            "Sources_Used": ms_data.sources_used,
            "Price_STD": ms_data.price_std,
            "Price_Mean": ms_data.price_mean,
        },
    )

    rec_row = rec.to_series()

    # Preserve original indicators from the input row
    for k, v in row.items():
        if k not in rec_row:
            rec_row[k] = v

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

    # Legacy overall score/breakdown for backward compatibility
    score, breakdown = compute_overall_score(rec_row)
    rec_row["Score_Breakdown"] = breakdown
    
    # STRICT RULE: Score must always equal FinalScore_20d (set in to_series above)
    # Never override Score here - it's already set correctly

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
from core.classifier import apply_classification


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


def compute_technical_score(row: pd.Series, weights: Optional[Dict[str, float]] = None) -> float:
    """Compute a deterministic technical score (0–100) using shared weights.

    Combines 10 technical factors using weights from scoring_config.TECH_WEIGHTS:
    - MA alignment, momentum consistency, RSI, volume surge, overextension
    - Near 52w high (pullback), risk/reward ratio, MACD position, ADX strength
    
    ATR% is used to scale RR and apply volatility adjustments.
    
    Args:
        row: Series with technical indicators (from build_technical_indicators)
        weights: Optional weight overrides (uses TECH_WEIGHTS by default)
    
    Returns:
        float: Technical score 0-100
        
    Required columns (from REQUIRED_TECH_COLS):
        MA_Aligned, Momentum_Consistency, RSI, VolSurge, Overext, Near52w,
        RR, ATR_Pct, MACD_Pos, ADX14
    
    Note: Missing columns emit warnings and use neutral defaults.
    """

    missing = [c for c in REQUIRED_TECH_COLS if c not in row]
    if missing:
        logging.warning("compute_technical_score: missing columns %s", missing)

    # Merge user weights with defaults and normalize
    raw_weights = {**TECH_WEIGHTS, **(weights or {})}
    sanitized = {k: max(float(v), 0.0) if np.isfinite(v) else 0.0 for k, v in raw_weights.items()}
    total_w = sum(sanitized.values()) or 1.0
    norm_w = {k: v / total_w for k, v in sanitized.items()}

    def _get_float(col: str, default: float = np.nan) -> float:
        val = row.get(col, default)
        return float(val) if pd.notna(val) else float(default)

    ma_score = 1.0 if bool(row.get("MA_Aligned", False)) else 0.0
    mom_score = float(np.clip(_get_float("Momentum_Consistency", 0.0), 0.0, 1.0))

    rsi_val = _get_float("RSI", np.nan)
    if pd.isna(rsi_val):
        rsi_score = 0.5
    elif 25 <= rsi_val <= 75:
        rsi_score = 1.0
    else:
        rsi_score = max(0.0, 1.0 - (abs(rsi_val - 50) - 25) / 50.0)

    vol_surge = _get_float("VolSurge", 1.0)
    vol_score = float(np.clip(vol_surge / 2.0, 0.0, 1.0))

    overext = _get_float("Overext", 0.0)
    overext_score = float(np.clip(1.0 - (overext / 0.2), 0.0, 1.0))

    near_high = _get_float("Near52w", np.nan)
    pullback_score = 0.5 if pd.isna(near_high) else float(np.clip(1.0 - (near_high / 100.0), 0.0, 1.0))

    rr_raw = _get_float("RR", 1.0)
    atr_pct = _get_float("ATR_Pct", np.nan)

    # ATR scaling for RR and later overall adjustment
    atr_factor = 1.0
    if pd.notna(atr_pct):
        if atr_pct >= ATR_RULES["extreme_high"]["min"]:
            atr_factor = ATR_RULES["extreme_high"].get("factor", 1.0)
        elif ATR_RULES["high"]["min"] <= atr_pct < ATR_RULES["high"]["max"]:
            atr_factor = ATR_RULES["high"].get("factor", 1.0)
        elif ATR_RULES["sweet_spot"]["min"] <= atr_pct <= ATR_RULES["sweet_spot"]["max"]:
            atr_factor = ATR_RULES["sweet_spot"].get("factor", 1.0)
        elif atr_pct < ATR_RULES["low"]["max"]:
            atr_factor = ATR_RULES["low"].get("factor", 1.0)

    rr_score = float(np.clip((max(rr_raw, 0.0) * atr_factor) / 3.0, 0.0, 1.5))

    macd_pos = _get_float("MACD_Pos", 0.0)
    macd_score = 1.0 if macd_pos > 0 else 0.0

    adx_val = _get_float("ADX14", 0.0)
    adx_score = float(np.clip(adx_val / 50.0, 0.0, 1.0))

    base = (
        norm_w["ma"] * ma_score
        + norm_w["mom"] * mom_score
        + norm_w["rsi"] * rsi_score
        + norm_w["near_high_bell"] * pullback_score
        + norm_w["vol"] * vol_score
        + norm_w["overext"] * overext_score
        + norm_w["pullback"] * pullback_score
        + norm_w["risk_reward"] * rr_score
        + norm_w["macd"] * macd_score
        + norm_w["adx"] * adx_score
    )

    vol_adjust = 1.0
    if pd.notna(atr_pct):
        if atr_pct >= ATR_RULES["extreme_high"]["min"]:
            vol_adjust = 1.0 - ATR_RULES["extreme_high"].get("penalty", 0.0)
        elif ATR_RULES["sweet_spot"]["min"] <= atr_pct <= ATR_RULES["sweet_spot"].get("max", atr_pct):
            vol_adjust = ATR_RULES["sweet_spot"].get("factor", 1.0)
        elif atr_pct < ATR_RULES["low"]["max"]:
            vol_adjust = ATR_RULES["low"].get("factor", 1.0)

    return float(np.clip(base * vol_adjust * 100.0, 0.0, 100.0))


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
    ml_prob: Optional[float] = None,
) -> float:
    """Combine technical, fundamental, and ML components into a 0–100 score.
    
    Uses weights from scoring_config.FINAL_SCORE_WEIGHTS:
    - Technical: 55%
    - Fundamental: 25%
    - ML: 20%
    
    Args:
        tech_score: Technical score 0-100
        fundamental_score: Fundamental score 0-100 (optional, defaults to 0)
        ml_prob: ML probability 0-1 (optional, defaults to 0.5 neutral)
    
    Returns:
        float: Combined final score 0-100
    """

    weights = FINAL_SCORE_WEIGHTS

    fund = 0.0 if fundamental_score is None or pd.isna(fundamental_score) else float(fundamental_score)
    ml_score = 0.5 if ml_prob is None or pd.isna(ml_prob) else float(ml_prob)

    final = (
        weights["technical"] * float(tech_score)
        + weights["fundamental"] * fund
        + weights["ml"] * (ml_score * 100.0)
    )

    return float(np.clip(final, 0.0, 100.0))


def compute_final_score_with_patterns(
    tech_score: float,
    fundamental_score: Optional[float] = None,
    ml_prob: Optional[float] = None,
    big_winner_score: Optional[float] = None,
    pattern_score: Optional[float] = None,
    bw_weight: float = 0.10,
    pattern_weight: float = 0.10,
) -> Tuple[float, Dict[str, float]]:
    """
    Enhanced final score incorporating Big Winner signal and pattern matching.
    
    UPDATED (Dec 2025): Incorporates learnings from historical backtests.
    
    Weights:
    - Technical: 45% (down from 55%)
    - Fundamental: 20% (down from 25%)
    - ML: 15% (down from 20%)
    - Big Winner Signal: 10% (new - highly predictive 20d pattern)
    - Pattern Matching: 10% (new - reward historical winners)
    
    Args:
        tech_score: Technical score 0-100
        fundamental_score: Fundamental score 0-100
        ml_prob: ML probability 0-1
        big_winner_score: Big Winner signal 0-100 (optional)
        pattern_score: Pattern match score 0-100 (optional)
        bw_weight: Weight for Big Winner signal (default 0.10)
        pattern_weight: Weight for pattern matching (default 0.10)
    
    Returns:
        Tuple of (final_score 0-100, breakdown dict)
    """
    tech = float(tech_score) if tech_score is not None else 50.0
    fund = float(fundamental_score) if fundamental_score is not None and pd.notna(fundamental_score) else 50.0
    ml_score = float(ml_prob) if ml_prob is not None and pd.notna(ml_prob) else 0.5
    bw = float(big_winner_score) if big_winner_score is not None and pd.notna(big_winner_score) else 50.0
    patt = float(pattern_score) if pattern_score is not None and pd.notna(pattern_score) else 50.0
    
    # Base weights (must sum to 1.0)
    base_weights = {
        "technical": 0.45,
        "fundamental": 0.20,
        "ml": 0.15,
        "big_winner": bw_weight,
        "pattern": pattern_weight,
    }
    
    # Normalize to ensure sum = 1.0
    total_weight = sum(base_weights.values())
    normalized_weights = {k: v / total_weight for k, v in base_weights.items()}
    
    final = (
        normalized_weights["technical"] * tech
        + normalized_weights["fundamental"] * fund
        + normalized_weights["ml"] * (ml_score * 100.0)
        + normalized_weights["big_winner"] * bw
        + normalized_weights["pattern"] * patt
    )
    
    final = float(np.clip(final, 0.0, 100.0))
    
    breakdown = {
        "tech_component": normalized_weights["technical"] * tech,
        "fund_component": normalized_weights["fundamental"] * fund,
        "ml_component": normalized_weights["ml"] * (ml_score * 100.0),
        "bw_component": normalized_weights["big_winner"] * bw,
        "pattern_component": normalized_weights["pattern"] * patt,
        "final_score": final,
        "weights_used": normalized_weights,
    }
    
    return final, breakdown


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
    Legacy wrapper kept for backward compatibility. Delegates to
    compute_recommendation_scores for the unified scoring flow.
    """
    rec_series = compute_recommendation_scores(
        row=row,
        ticker=ticker,
        enable_ml=enable_ml,
        use_multi_source=use_multi_source,
    )

    return {
        "ticker": ticker,
        "fundamental_score_v2": float(rec_series.get("Fundamental_Score", 0.0)),
        "fundamental_confidence_v2": float(rec_series.get("Fundamental_Coverage", 50.0)),
        "technical_score_v2": float(rec_series.get("TechScore_20d", 0.0)),
        "technical_confidence_v2": 100.0,
        "rr_score_v2": float(rec_series.get("RR", 0.0)) * 20.0 if pd.notna(rec_series.get("RR", np.nan)) else 50.0,
        "rr_confidence_v2": 50.0,
        "reliability_score_v2": float(rec_series.get("Reliability_Score", 50.0)),
        "conviction_v2_base": float(rec_series.get("FinalScore_20d", 0.0)),
        "conviction_v2_final": float(rec_series.get("ConvictionScore", rec_series.get("FinalScore_20d", 0.0))),
        "ml_probability": rec_series.get("ML_20d_Prob"),
        "ml_boost": 0.0,
        "ml_status": "delegated",
        "risk_meter_v2": float(rec_series.get("Risk_Meter", 50.0)),
        "risk_label_v2": rec_series.get("Risk_Label", "UNKNOWN"),
        "warnings_v2": rec_series.get("Classification_Warnings", []) or [],
        "sources_used": rec_series.get("Sources_Used", []) or [],
        "disagreement_score": rec_series.get("Price_STD", 0.0) or 0.0,
        "breakdown": {},
    }


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
