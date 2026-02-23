"""
Recommendation Scoring — single source of truth for Stock Scout.

:func:`compute_recommendation_scores` is the **canonical entry point**
called by pipeline_runner, UI, CSV export, and backtests.  It combines
technical scoring, fundamental scoring, ML probability, pattern matching,
big-winner detection, reliability assessment, and risk classification
into a fully-populated recommendation row.

Classes:
    RecommendationResult — structured recommendation dataclass
Functions:
    compute_recommendation_scores — master scoring function
    apply_technical_filters — basic volume/price pre-filter
    score_with_ml_model — backward-compatible ML wrapper
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.classifier import apply_classification
from core.pattern_matcher import PatternMatcher
from core.schema import Col
from core.scoring.big_winner import compute_big_winner_signal_20d
from core.scoring.final import compute_final_score, compute_final_score_with_patterns
from core.scoring.technical import compute_tech_score_20d_v2, compute_technical_score
from core.scoring_config import (
    ML_PROB_THRESHOLD,
    TECH_STRONG_THRESHOLD,
    MultiSourceData,
)
from core.scoring_engine import (
    calculate_reliability_score,
    calculate_risk_meter,
    compute_overall_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
            Col.TICKER: self.ticker,
            Col.TECH_SCORE: self.tech_score,
            Col.FUND_SCORE: self.fundamental_score,
            Col.ML_PROB: self.ml_prob,
            Col.FINAL_SCORE: self.final_score,
            Col.CONVICTION: self.conviction_score,
            Col.RELIABILITY: self.reliability_score,
            Col.RISK_LABEL: self.risk_label,
            Col.RISK_METER: self.risk_meter,
            Col.SHOULD_DISPLAY: self.should_display,
            Col.SCORE: self.final_score,  # Legacy alias
        }
        base.update(self.extras)
        return pd.Series(base, dtype="object")


# ---------------------------------------------------------------------------
# Pre-filters
# ---------------------------------------------------------------------------

def apply_technical_filters(
    row: pd.Series,
    strict: bool = True,
    relaxed: bool = False,
) -> bool:
    """Basic volume/price pre-filter.

    Returns ``True`` if the stock passes.
    """
    volume = row.get("Volume", 0)
    if volume < 100_000 and not relaxed:
        return False
    close = row.get("Close", 0)
    if close < 1.0 or close > 10_000:
        return False
    return True


# ---------------------------------------------------------------------------
# ML helper (backward compat)
# ---------------------------------------------------------------------------

def score_with_ml_model(
    row: pd.Series,
    model_data: Optional[Dict] = None,
) -> float:
    """Return ML probability for 20d horizon.

    Checks existing row fields first, then falls back to live model prediction.
    Returns neutral 0.5 if unavailable.
    """
    try:
        val = row.get("ML_20d_Prob_raw") or row.get("ML_20d_Prob") or row.get("ML_Probability")
        if isinstance(val, (int, float)) and np.isfinite(val):
            return float(val)
    except Exception:
        pass

    try:
        from core.ml_20d_inference import ML_20D_AVAILABLE, predict_20d_prob_from_row

        if ML_20D_AVAILABLE:
            prob = predict_20d_prob_from_row(row)
            if isinstance(prob, (int, float)) and np.isfinite(prob):
                return float(prob)
    except Exception:
        pass

    return 0.5


# ---------------------------------------------------------------------------
# Master scoring function
# ---------------------------------------------------------------------------

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

    Returns:
        pd.Series with canonical columns (see :class:`Col`).
    """
    base_ticker = ticker or row.get("Ticker") or row.get("ticker")
    if base_ticker is None:
        raise ValueError("compute_recommendation_scores requires a ticker")

    # ── Multi-source fundamentals ────────────────────────────────
    ms_data = multi_source_override
    if ms_data is None and use_multi_source:
        try:
            from core import data_sources_v2

            as_of_for_fund = None
            if as_of_date is not None:
                try:
                    as_of_for_fund = pd.to_datetime(as_of_date).date()
                except (TypeError, ValueError):
                    pass
            raw = data_sources_v2.fetch_multi_source_data(base_ticker, as_of_date=as_of_for_fund)
            ms_data = MultiSourceData.from_dict(raw)
        except Exception as exc:
            logging.warning("multi-source fetch failed for %s: %s", base_ticker, exc)
            ms_data = MultiSourceData()
    elif ms_data is None:
        ms_data = MultiSourceData()

    # ── Technical component ──────────────────────────────────────
    try:
        tech_raw = compute_tech_score_20d_v2(row)
        tech_score = float(np.clip(tech_raw * 100.0, 0.0, 100.0))
    except (TypeError, ValueError, KeyError):
        tech_score = compute_technical_score(row)

    # ── Fundamental component ────────────────────────────────────
    from core.scoring import compute_fundamental_score_with_breakdown

    fund_input = asdict(ms_data)
    fundamental = compute_fundamental_score_with_breakdown(fund_input)
    fundamental_score = float(fundamental.total) if fundamental else 0.0

    # ── Market regime ────────────────────────────────────────────
    market_regime = None
    data_integrity = "OK"
    try:
        if as_of_date is not None:
            from core.unified_logic import build_market_context_table

            start_ctx = (pd.to_datetime(as_of_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
            end_ctx = (pd.to_datetime(as_of_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            ctx = build_market_context_table(start_ctx, end_ctx)
            if ctx is not None and not ctx.empty:
                market_regime = str(ctx["Market_Regime"].iloc[-1])
            else:
                data_integrity = "DATA_INCOMPLETE"
    except (TypeError, ValueError, KeyError, AttributeError):
        data_integrity = "DATA_INCOMPLETE"

    # ── ML component ─────────────────────────────────────────────
    ml_prob = ml_prob_override
    if ml_prob is None:
        if "ML_20d_Prob" in row.index and pd.notna(row.get("ML_20d_Prob")):
            try:
                ml_prob = float(row.get("ML_20d_Prob"))
            except (TypeError, ValueError):
                pass
        elif enable_ml:
            try:
                from core.ml_20d_inference import (
                    ML_20D_AVAILABLE,
                    calibrate_ml_20d_prob,
                    compute_ml_20d_probabilities_raw,
                )

                if ML_20D_AVAILABLE:
                    prob_raw = compute_ml_20d_probabilities_raw(row)
                    ml_prob = calibrate_ml_20d_prob(
                        prob_raw,
                        atr_pct_percentile=_safe_float(row.get("ATR_Pct_percentile")),
                        price_as_of=_safe_float(row.get("Price_As_Of_Date")),
                        reliability_factor=1.0,
                        market_regime=market_regime,
                        rsi=_safe_float(row.get("RSI")),
                    )
            except (ImportError, TypeError, ValueError):
                pass
    # Coerce
    try:
        if ml_prob is None or not np.isfinite(float(ml_prob)):
            ml_prob = 0.5
    except (TypeError, ValueError):
        ml_prob = 0.5

    # ── Pattern + Big Winner enhancements ────────────────────────
    # ── Extract RR for scoring gate ──────────────────────────────
    try:
        _rr = float(row.get("RR")) if row.get("RR") is not None and pd.notna(row.get("RR")) else None
    except (TypeError, ValueError):
        _rr = None

    try:
        bw_signal = compute_big_winner_signal_20d(row)
        patt_eval = PatternMatcher.evaluate_stock(row)
        final_score, breakdown = compute_final_score_with_patterns(
            tech_score=float(tech_score),
            fundamental_score=float(fundamental_score),
            ml_prob=float(ml_prob),
            big_winner_score=float(bw_signal.get("BigWinnerScore_20d", 0.0)),
            pattern_score=float(patt_eval.get("pattern_score", 0.0)),
            bw_weight=0.10,
            pattern_weight=0.10,
            market_regime=market_regime,
            rr_ratio=_rr,
        )
        conviction_score = final_score
    except (TypeError, ValueError, KeyError, AttributeError):
        final_score = compute_final_score(tech_score, fundamental_score, ml_prob, market_regime=market_regime, rr_ratio=_rr)
        conviction_score = final_score

    # ── ML confidence flagging ───────────────────────────────────
    try:
        ml_val = float(ml_prob) if ml_prob is not None else np.nan
    except Exception:
        ml_val = np.nan
    if pd.isna(ml_val):
        ml_conf_status = "NEUTRAL"
    elif ml_val < 0.15:
        ml_conf_status = "PENALIZED"
    elif ml_val > 0.62:
        ml_conf_status = "BOOSTED"
    else:
        ml_conf_status = "NEUTRAL"

    # ── Reliability & Risk ───────────────────────────────────────
    tech_fields = ["RSI", "ATR", "MA_Aligned", "Overext", "VolSurge", "Near52w", "RR"]
    valid_count = sum(1 for f in tech_fields if pd.notna(row.get(f, np.nan)))
    data_completeness = (valid_count / len(tech_fields)) * 100.0

    reliability_score = calculate_reliability_score(
        price_sources=ms_data.price_sources or 0,
        fund_sources=len(ms_data.sources_used),
        price_std=ms_data.price_std,
        price_mean=ms_data.price_mean,
        fundamental_confidence=data_completeness,
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
            Col.FUND_BREAKDOWN: getattr(fundamental, "breakdown", None),
            Col.SOURCES_USED: ms_data.sources_used,
            Col.PRICE_STD: ms_data.price_std,
            Col.PRICE_MEAN: ms_data.price_mean,
            Col.ML_CONFIDENCE_STATUS: ml_conf_status,
            Col.MARKET_REGIME: market_regime,
            Col.DATA_INTEGRITY: data_integrity,
        },
    )

    rec_row = rec.to_series()
    rec_row[Col.RELIABILITY_LEGACY] = rec_row[Col.RELIABILITY]

    # Preserve original indicators
    for k, v in row.items():
        if k not in rec_row:
            rec_row[k] = v

    if as_of_date is not None:
        rec_row[Col.AS_OF_DATE] = pd.to_datetime(as_of_date)

    # Classification
    classified = apply_classification(pd.DataFrame([rec_row])).iloc[0]
    for col in [Col.RISK_LEVEL, Col.DATA_QUALITY, Col.CONFIDENCE_LEVEL, Col.SHOULD_DISPLAY, Col.CLASSIFICATION_WARNINGS]:
        rec_row[col] = classified.get(col)

    # Legacy overall score/breakdown
    score, breakdown = compute_overall_score(rec_row)
    rec_row[Col.SCORE_BREAKDOWN] = breakdown

    # Pattern extras
    try:
        patt_eval = PatternMatcher.evaluate_stock(rec_row)
        rec_row[Col.PATTERN_SCORE] = float(patt_eval.get("pattern_score", 0.0))
        rec_row[Col.PATTERN_COUNT] = int(patt_eval.get("pattern_count", 0))
    except (TypeError, ValueError, KeyError, AttributeError):
        rec_row[Col.PATTERN_SCORE] = rec_row.get(Col.PATTERN_SCORE, np.nan)
        rec_row[Col.PATTERN_COUNT] = rec_row.get(Col.PATTERN_COUNT, np.nan)

    # Signal reasons & quality
    try:
        reasons = []
        ts = float(tech_score)
        if np.isfinite(ts) and ts >= float(TECH_STRONG_THRESHOLD):
            reasons.append("Strong technical momentum")
        mp = float(ml_prob) if ml_prob is not None else np.nan
        if np.isfinite(mp) and mp >= float(ML_PROB_THRESHOLD):
            reasons.append("High ML breakout probability")
        ps = float(rec_row.get(Col.PATTERN_SCORE, 0.0) or 0.0)
        if np.isfinite(ps) and ps > 0.0:
            reasons.append("Bullish pattern detected")
        reg = str(market_regime or "").upper()
        if reg in ("TREND_UP", "BULLISH", "NEUTRAL", "SIDEWAYS"):
            reasons.append("Supportive market regime")

        cnt = len(reasons)
        quality = "High" if cnt >= 3 else ("Medium" if cnt == 2 else "Speculative")
        rec_row[Col.SIGNAL_REASONS] = "; ".join(reasons)
        rec_row[Col.SIGNAL_REASONS_COUNT] = cnt
        rec_row[Col.SIGNAL_QUALITY] = quality
    except (ImportError, KeyError, TypeError):
        rec_row.setdefault(Col.SIGNAL_REASONS, "")
        rec_row.setdefault(Col.SIGNAL_REASONS_COUNT, 0)
        rec_row.setdefault(Col.SIGNAL_QUALITY, "Speculative")

    # ML features fallback flag
    try:
        from core.ml_20d_inference import BUNDLE_HAS_MISSING_METEOR_FEATURES

        rec_row[Col.ML_FEATURES_FALLBACK] = 1 if bool(BUNDLE_HAS_MISSING_METEOR_FEATURES) else 0
    except (ImportError, AttributeError):
        rec_row[Col.ML_FEATURES_FALLBACK] = 0

    return rec_row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    """Convert to float if not NaN/None, else return None."""
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
