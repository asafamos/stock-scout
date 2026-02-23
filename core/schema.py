"""
Canonical column names and DataFrame schema validation for Stock Scout.

All production code should reference column names via :class:`Columns`
instead of using string literals.  This eliminates the alias chaos
(e.g. "Score" vs "FinalScore_20d" vs "overall_score_20d") and provides
a single place to audit every column the system produces.

Usage::

    from core.schema import Col, validate_columns

    df[Col.FINAL_SCORE]            # instead of df["FinalScore_20d"]
    validate_columns(df, Col.PIPELINE_OUTPUT)
"""
from __future__ import annotations

import logging
from typing import Iterable, Set

import pandas as pd

logger = logging.getLogger(__name__)


class Col:
    """Canonical column name constants.

    Naming convention: UPPER_SNAKE_CASE for constants.
    Values match the canonical DataFrame column names used throughout
    the pipeline (pipeline_runner → UI → CSV export).
    """

    # ── Identifiers ────────────────────────────────────────────────
    TICKER = "Ticker"

    # ── Technical Indicators ───────────────────────────────────────
    RSI = "RSI"
    ATR = "ATR"
    ATR_PCT = "ATR_Pct"
    MA20 = "MA20"
    MA50 = "MA50"
    MA200 = "MA200"
    MA_ALIGNED = "MA_Aligned"
    OVEREXT = "Overext"
    NEAR_52W = "Near52w"
    MOM_CONS = "MomCons"
    VOL_SURGE = "VolSurge"
    RR = "RR"
    MACD_POS = "MACD_Pos"
    ADX14 = "ADX14"
    VCP_SCORE = "Volatility_Contraction_Score"
    COIL_BONUS = "Coil_Bonus"

    # ── Returns ────────────────────────────────────────────────────
    RETURN_5D = "Return_5d"
    RETURN_10D = "Return_10d"
    RETURN_1M = "Return_1m"
    RETURN_3M = "Return_3m"
    RETURN_6M = "Return_6m"

    # ── Scores ─────────────────────────────────────────────────────
    TECH_SCORE = "TechScore_20d"
    FUND_SCORE = "Fundamental_Score"
    ML_PROB = "ML_20d_Prob"
    ML_PROB_RAW = "ML_20d_Prob_raw"
    FINAL_SCORE = "FinalScore_20d"
    CONVICTION = "ConvictionScore"
    SCORE = "Score"  # Legacy alias — always == FINAL_SCORE

    # ── Big Winner ─────────────────────────────────────────────────
    BIG_WINNER_SCORE = "BigWinnerScore_20d"
    BIG_WINNER_FLAG = "BigWinnerFlag_20d"

    # ── Pattern ────────────────────────────────────────────────────
    PATTERN_SCORE = "Pattern_Score"
    PATTERN_COUNT = "Pattern_Count"

    # ── Reliability & Risk ─────────────────────────────────────────
    RELIABILITY = "Reliability_Score"
    RELIABILITY_LEGACY = "ReliabilityScore"  # Legacy alias
    RISK_METER = "Risk_Meter"
    RISK_LABEL = "Risk_Label"
    RISK_LEVEL = "Risk_Level"

    # ── Classification ─────────────────────────────────────────────
    DATA_QUALITY = "Data_Quality"
    CONFIDENCE_LEVEL = "Confidence_Level"
    SHOULD_DISPLAY = "Should_Display"
    CLASSIFICATION_WARNINGS = "Classification_Warnings"

    # ── ML Metadata ────────────────────────────────────────────────
    ML_CONFIDENCE_STATUS = "ML_Confidence_Status"
    ML_FEATURES_FALLBACK = "ML_Features_Fallback"

    # ── Market Context ─────────────────────────────────────────────
    MARKET_REGIME = "Market_Regime"
    DATA_INTEGRITY = "Data_Integrity"

    # ── Signal Quality ─────────────────────────────────────────────
    SIGNAL_REASONS = "SignalReasons"
    SIGNAL_REASONS_COUNT = "SignalReasons_Count"
    SIGNAL_QUALITY = "SignalQuality"

    # ── Fundamentals ───────────────────────────────────────────────
    SECTOR = "Sector"
    INDUSTRY = "Industry"
    FUND_BREAKDOWN = "Fundamental_Breakdown"
    SOURCES_USED = "Sources_Used"
    PRICE_STD = "Price_STD"
    PRICE_MEAN = "Price_Mean"
    SCORE_BREAKDOWN = "Score_Breakdown"

    # ── OHLCV ──────────────────────────────────────────────────────
    CLOSE = "Close"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    VOLUME = "Volume"

    # ── Date ───────────────────────────────────────────────────────
    AS_OF_DATE = "As_Of_Date"

    # ── Column groups (for validation) ─────────────────────────────
    PIPELINE_OUTPUT: Set[str] = {
        TICKER, TECH_SCORE, FUND_SCORE, ML_PROB, FINAL_SCORE,
        CONVICTION, RELIABILITY, RISK_METER, RISK_LABEL,
        SHOULD_DISPLAY, SCORE,
    }

    TECHNICAL_INDICATORS: Set[str] = {
        RSI, ATR, ATR_PCT, MA20, MA50, MA200,
        OVEREXT, NEAR_52W, MOM_CONS, VOL_SURGE, RR,
        CLOSE, VOLUME, HIGH, LOW,
    }

    SCORING_COLUMNS: Set[str] = {
        TECH_SCORE, FUND_SCORE, ML_PROB, FINAL_SCORE,
        CONVICTION, SCORE,
    }


def validate_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    *,
    strict: bool = False,
    label: str = "",
) -> bool:
    """Check that *df* contains all *required* columns.

    Args:
        df: DataFrame to validate.
        required: Column names that must be present.
        strict: If ``True``, raise ``KeyError`` on missing columns.
        label: Optional context label for log messages.

    Returns:
        ``True`` if all columns are present.
    """
    required_set = set(required)
    present = set(df.columns)
    missing = required_set - present

    if missing:
        msg = f"[{label}] Missing columns: {sorted(missing)}" if label else f"Missing columns: {sorted(missing)}"
        if strict:
            raise KeyError(msg)
        logger.warning(msg)
        return False
    return True
