"""Per-ticker scoring and parallel indicator computation for the pipeline."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.filters import apply_technical_filters
from core.market_context import (
    compute_relative_strength_vs_spy,
    get_benchmark_series,
)
from core.ml_20d_inference import ML_20D_AVAILABLE, get_ml_health_meta
from core.ml_feature_builder import build_all_ml_features_v3
from core.pattern_matcher import PatternMatcher
from core.scoring import build_technical_indicators
from core.scoring_config import ML_PROB_THRESHOLD, TECH_STRONG_THRESHOLD
from core.unified_logic import (
    compute_big_winner_signal_20d,
    compute_final_score_with_patterns,
    compute_recommendation_scores,
)

# V2 Bridge (ML + Risk integration)
try:
    from core.bridge import analyze_row_with_bridge

    V2_BRIDGE_AVAILABLE = True
except ImportError:
    V2_BRIDGE_AVAILABLE = False

from core.pipeline.fallback_tracking import _record_legacy_fallback
from core.pipeline.market_data import (
    _GLOBAL_MARKET_CONTEXT,
    _get_sector_context_for_ticker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-ticker processing
# ---------------------------------------------------------------------------

def _process_single_ticker(
    tkr: str, df: pd.DataFrame, skip_tech_filter: bool,
) -> Optional[pd.Series]:
    """Compute indicators, ML features, and scores for a single ticker.

    Returns a Series suitable for aggregation into a results DataFrame,
    or *None* if the ticker should be excluded.
    """
    if df.empty:
        return None
    try:
        tech_df = build_technical_indicators(df)
        # Work with a single-row DataFrame to avoid SettingWithCopy issues,
        # then convert back to Series once enrichment is done
        row_df = tech_df.iloc[[-1]].copy()
        row_df["Ticker"] = tkr
    except Exception as exc:
        logger.warning("indicator build failed for %s: %s", tkr, exc)
        return None

    # Relative Strength vs SPY (ensure working source via market_context)
    try:
        spy_df = get_benchmark_series("SPY", period='3mo')
        spy_df_cap = spy_df.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        rs_val = compute_relative_strength_vs_spy(
            tech_df.rename(columns=str.title), spy_df_cap,
        )
        row_df = row_df.set_index("Ticker")
        row_df.loc[tkr, "relative_strength_20d"] = float(rs_val)
    except (KeyError, TypeError, ValueError, IndexError):
        row_df = row_df.set_index("Ticker")
        row_df.loc[tkr, "relative_strength_20d"] = np.nan
    # Convert enriched single-row DataFrame back to Series for downstream logic
    row_indicators = row_df.reset_index(drop=False).iloc[0]

    # Enrich with all 34 ML features using the feature builder
    try:
        # Compute stock's 20d return for sector context
        stock_20d_return = 0.0
        try:
            close = df["Close"] if "Close" in df.columns else df.get("close", pd.Series())
            if len(close) >= 20:
                stock_20d_return = float(close.iloc[-1] / close.iloc[-20] - 1.0)
        except Exception:
            pass

        # Get sector context for this ticker
        sector_ctx = _get_sector_context_for_ticker(tkr, stock_20d_return)

        # Import globals at call time (they're mutated by _initialize_ml_context)
        from core.pipeline.market_data import _GLOBAL_MARKET_CONTEXT as _mkt_ctx

        ml_features = build_all_ml_features_v3(
            row=row_indicators,
            df_hist=df,
            market_context=_mkt_ctx if _mkt_ctx else None,
            sector_context=sector_ctx if sector_ctx else None,
        )
        # Add ML features to row_indicators
        for feat_name, feat_val in ml_features.items():
            row_indicators[feat_name] = feat_val
    except Exception as exc:
        logger.debug(f"ML feature building failed for {tkr}: {exc}")

    # Tier 2 may skip this filter because Tier 1 already applied OHLCV checks
    if not skip_tech_filter:
        if not apply_technical_filters(row_indicators, strict=False):
            return None

    # Use the last bar's date as the signal date
    try:
        as_of_dt = pd.to_datetime(df.index[-1])
    except (IndexError, TypeError, ValueError):
        as_of_dt = None

    used_v2 = False
    if V2_BRIDGE_AVAILABLE:
        try:
            rec_dict = analyze_row_with_bridge(ticker=tkr, row=row_indicators)
            # Verify success: ensure we have a valid score
            if "FinalScore_20d" in rec_dict and rec_dict["FinalScore_20d"] is not None:
                rec_series = pd.Series(rec_dict)
                used_v2 = True
            else:
                msg = f"Bridge returned no score for {tkr}"
                logger.warning(msg + ", falling back to legacy.")
                _record_legacy_fallback(f"BridgeNoScore: {msg}")
        except Exception as e:
            logger.warning(f"Bridge failed for {tkr}: {e}. Falling back to legacy.")
            _record_legacy_fallback(f"{e.__class__.__name__}: {e}")
            used_v2 = False

    if not used_v2:
        # Fallback to legacy logic (explicit, tracked)
        try:
            ml_meta = get_ml_health_meta()
            ml_enable = bool(ML_20D_AVAILABLE) and not bool(
                ml_meta.get("ml_bundle_version_warning")
            )
        except (AttributeError, KeyError, TypeError):
            ml_enable = bool(ML_20D_AVAILABLE)
        rec_series = compute_recommendation_scores(
            row=row_indicators,
            ticker=tkr,
            as_of_date=as_of_dt,
            enable_ml=ml_enable,
            use_multi_source=False,
        )

    # Ensure legacy alias for reliability is present regardless of source
    if ("ReliabilityScore" not in rec_series) and ("Reliability_Score" in rec_series):
        try:
            rec_series["ReliabilityScore"] = rec_series["Reliability_Score"]
        except (KeyError, TypeError):
            pass

    # Enhance with Big Winner signal + historical pattern matching
    try:
        bw_signal = compute_big_winner_signal_20d(row_indicators)
        patt_eval = PatternMatcher.evaluate_stock(row_indicators)

        final_score, breakdown = compute_final_score_with_patterns(
            tech_score=float(rec_series.get("TechScore_20d", 0.0)),
            fundamental_score=float(rec_series.get("Fundamental_Score", 0.0)),
            ml_prob=float(rec_series.get("ML_20d_Prob", 0.5)),
            big_winner_score=float(bw_signal or 0.0),
            pattern_score=float(patt_eval.get("pattern_score", 0.0)),
            bw_weight=0.10,
            pattern_weight=0.10,
        )

        rec_series["FinalScore_20d"] = float(final_score)
        rec_series["Pattern_Score"] = float(patt_eval.get("pattern_score", 0.0))
        rec_series["Pattern_Count"] = int(patt_eval.get("pattern_count", 0))
        rec_series["Big_Winner_Signal"] = float(bw_signal or 0.0)
        rec_series["Score_Breakdown_Patterns"] = breakdown
    except Exception as exc:
        logger.debug(f"Pattern/BW enhancement failed for {tkr}: {exc}")

    # Ensure SignalReasons / SignalQuality exist even when using bridge path
    try:
        reasons: List[str] = []
        try:
            ts_val = float(rec_series.get("TechScore_20d", np.nan))
            if np.isfinite(ts_val) and ts_val >= float(TECH_STRONG_THRESHOLD):
                reasons.append("Strong technical momentum")
        except (TypeError, ValueError):
            pass
        try:
            mlp_val = float(rec_series.get("ML_20d_Prob", np.nan))
            if np.isfinite(mlp_val) and mlp_val >= float(ML_PROB_THRESHOLD):
                reasons.append("High ML breakout probability")
        except (TypeError, ValueError):
            pass
        try:
            ps_val = float(rec_series.get("Pattern_Score", 0.0) or 0.0)
            if np.isfinite(ps_val) and ps_val > 0.0:
                reasons.append("Bullish pattern detected")
        except (TypeError, ValueError):
            pass
        try:
            reg_val = str(rec_series.get("Market_Regime") or "").upper()
            if reg_val in ("TREND_UP", "BULLISH", "NEUTRAL", "SIDEWAYS"):
                reasons.append("Supportive market regime")
        except (TypeError, ValueError, AttributeError):
            pass
        cnt = len(reasons)
        quality = "High" if cnt >= 3 else ("Medium" if cnt == 2 else "Speculative")
        if "SignalReasons" not in rec_series:
            rec_series["SignalReasons"] = "; ".join(reasons)
        if "SignalReasons_Count" not in rec_series:
            rec_series["SignalReasons_Count"] = cnt
        if "SignalQuality" not in rec_series:
            rec_series["SignalQuality"] = quality
    except (ImportError, KeyError, TypeError) as exc:
        logger.debug(f"SignalReasons computation fallback: {exc}")
        if "SignalReasons" not in rec_series:
            rec_series["SignalReasons"] = ""
        if "SignalReasons_Count" not in rec_series:
            rec_series["SignalReasons_Count"] = 0
        if "SignalQuality" not in rec_series:
            rec_series["SignalQuality"] = "Speculative"

    # Ensure Ticker column is present for downstream merges/filters
    try:
        rec_series["Ticker"] = tkr
    except (TypeError, KeyError):
        pass
    return rec_series


# ---------------------------------------------------------------------------
# Parallel scoring orchestration
# ---------------------------------------------------------------------------

def _step_compute_scores_with_unified_logic(
    data_map: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    status_callback: Optional[Callable[[str], None]],
    skip_tech_filter: bool = False,
) -> pd.DataFrame:
    """Score all tickers in *data_map* in parallel via ThreadPoolExecutor."""
    if status_callback:
        status_callback("Computing technical indicators (parallel)...")

    rows: List[pd.Series] = []
    max_workers = min(10, max(1, len(data_map)))
    batch_size = max(10, max_workers * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map: dict = {}
        items = list(data_map.items())
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            for tkr, df in batch:
                future = executor.submit(
                    _process_single_ticker, tkr, df, skip_tech_filter,
                )
                future_map[future] = tkr
            # small backoff between batches
            time.sleep(0.5)
        for future in as_completed(future_map):
            tkr = future_map[future]
            try:
                res = future.result()
                if res is not None:
                    rows.append(res)
            except Exception as exc:
                logger.warning(f"Ticker {tkr} failed in parallel scoring: {exc}")

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    # Ensure a canonical Ticker column exists
    try:
        if "Ticker" not in results.columns:
            if "ticker" in results.columns:
                results = results.rename(columns={"ticker": "Ticker"})
            elif results.index.name == "Ticker":
                results = results.reset_index()
    except (KeyError, AttributeError) as exc:
        logger.debug(f"Ticker column normalization skipped: {exc}")

    if "Score" not in results.columns and "FinalScore_20d" in results.columns:
        results["Score"] = results["FinalScore_20d"]

    # Ensure reliability column compatibility for downstream consumers/tests
    if (
        ("ReliabilityScore" not in results.columns)
        and ("Reliability_Score" not in results.columns)
        and ("reliability_v2" not in results.columns)
        and ("Reliability_v2" not in results.columns)
    ):
        def _calc_reliability(row):
            fund_sources = row.get("Fundamental_Sources_Count", 0)
            if pd.isna(fund_sources):
                fund_sources = 0
            fund_score = min(fund_sources, 4) * 15 + 20

            price_bonus = 0
            if pd.notna(row.get("Price_Yahoo", row.get("Close"))):
                price_bonus += 10
            if pd.notna(row.get("ATR")):
                price_bonus += 5
            if pd.notna(row.get("RSI")):
                price_bonus += 5

            return min(fund_score + price_bonus, 100)

        results["ReliabilityScore"] = results.apply(_calc_reliability, axis=1)
    else:
        if ("Reliability_Score" in results.columns) and (
            "ReliabilityScore" not in results.columns
        ):
            results["ReliabilityScore"] = results["Reliability_Score"]

    # Ensure momentum proxy exists (prefer TechScore_20d)
    if ("MomentumScore" not in results.columns) and (
        "TechScore_20d" not in results.columns
    ):
        if "FinalScore_20d" in results.columns:
            results["TechScore_20d"] = results["FinalScore_20d"]
        elif "Score" in results.columns:
            results["TechScore_20d"] = results["Score"]
        else:
            results["TechScore_20d"] = np.nan

    return results
