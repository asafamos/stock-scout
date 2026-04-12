"""Per-ticker scoring and parallel indicator computation for the pipeline."""

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.filters import apply_technical_filters
from core.market_context import (
    compute_relative_strength_vs_spy,
    get_benchmark_series,
)
from core.ml_20d_inference import ML_20D_AVAILABLE, get_ml_health_meta
from core.ml_feature_builder import build_all_ml_features_v3_6
from core.pattern_matcher import PatternMatcher
from core.scoring import build_technical_indicators
from core.scoring_config import ML_PROB_THRESHOLD, TECH_STRONG_THRESHOLD
from core.unified_logic import (
    compute_big_winner_signal_20d,
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

        ml_features = build_all_ml_features_v3_6(
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
            # Verify success: bridge provides component scores (TechScore_20d,
            # ML_20d_Prob etc.); FinalScore_20d is computed later by runner.py.
            if "TechScore_20d" in rec_dict and rec_dict.get("ML_20d_Prob") is not None:
                rec_series = pd.Series(rec_dict)
                used_v2 = True
            else:
                msg = f"Bridge returned incomplete components for {tkr}"
                logger.warning(msg + ", falling back to legacy.")
                _record_legacy_fallback(f"BridgeNoComponents: {msg}")
        except Exception as e:
            logger.warning(f"Bridge failed for {tkr}: {e}. Falling back to legacy.")
            _record_legacy_fallback(f"{e.__class__.__name__}: {e}")
            used_v2 = False

    if not used_v2:
        # Fallback to legacy logic (explicit, tracked)
        try:
            ml_meta = get_ml_health_meta()
            # Enable ML if the model loaded successfully. A sklearn version
            # mismatch is only a warning — the model still produces valid
            # predictions in virtually all cases.  Disabling ML here caused
            # every ticker to fall back to ML_20d_Prob = 0.5.
            ml_enable = bool(ML_20D_AVAILABLE)
            if ml_meta.get("ml_bundle_version_warning"):
                logger.info(
                    "ML bundle version mismatch detected (%s) — continuing "
                    "with ML enabled (predictions are usually still valid).",
                    ml_meta.get("ml_bundle_warning_reason", "unknown"),
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
        # BigWinner needs TechScore_20d which is only available in rec_series
        # (computed by bridge or legacy scoring), not in raw row_indicators.
        bw_row = row_indicators.copy()
        if "TechScore_20d" in rec_series:
            bw_row["TechScore_20d"] = rec_series["TechScore_20d"]
        bw_dict = compute_big_winner_signal_20d(bw_row)
        bw_score = float(bw_dict.get("BigWinnerScore_20d", 0.0)) if isinstance(bw_dict, dict) else 0.0
        bw_flag = int(bw_dict.get("BigWinnerFlag_20d", 0)) if isinstance(bw_dict, dict) else 0
        patt_eval = PatternMatcher.evaluate_stock(row_indicators)

        # Store pattern/BW columns for compute_final_score_20d() to read later.
        # NOTE: FinalScore_20d is NOT set here — runner.py:1290 is the single
        # authoritative computation via compute_final_score_20d().
        rec_series["Pattern_Score"] = float(patt_eval.get("pattern_score", 0.0))
        rec_series["Pattern_Count"] = int(patt_eval.get("pattern_count", 0))
        rec_series["Big_Winner_Signal"] = bw_score
        rec_series["BigWinnerFlag_20d"] = bw_flag
    except Exception as exc:
        logger.debug(f"Pattern/BW enhancement failed for {tkr}: {exc}")

    # NOTE: RR gate is applied in runner.py AFTER _compute_rr_for_row sets
    # the actual Risk/Reward ratio. At this point, row_indicators["RR"] is
    # a technical indicator, not the final R/R ratio.

    # Ensure SignalReasons / SignalQuality exist even when using bridge path
    # Reasons are weighted: swing-trade signals (momentum, pattern, R/R, volume)
    # count more than supporting signals (fundamentals, regime).
    try:
        reasons: List[str] = []
        swing_strength = 0.0  # Weighted signal strength for quality assessment
        # 1. Technical momentum — PRIMARY signal for swing trades (weight=1.0)
        try:
            ts_val = float(rec_series.get("TechScore_20d", np.nan))
            if np.isfinite(ts_val) and ts_val >= float(TECH_STRONG_THRESHOLD):
                reasons.append("Strong technical momentum")
                swing_strength += 1.0
            elif np.isfinite(ts_val) and ts_val >= 45.0:
                reasons.append("Positive technical setup")
                swing_strength += 0.5
        except (TypeError, ValueError):
            pass
        # 2. ML probability — swing signal (weight=0.75)
        try:
            mlp_val = float(rec_series.get("ML_20d_Prob", np.nan))
            if np.isfinite(mlp_val) and mlp_val >= float(ML_PROB_THRESHOLD):
                reasons.append("High ML breakout probability")
                swing_strength += 0.75
            elif np.isfinite(mlp_val) and mlp_val >= 0.50:
                reasons.append("Moderate ML breakout probability")
                swing_strength += 0.25
        except (TypeError, ValueError):
            pass
        # 3. Pattern signals — PRIMARY signal for swing trades (weight=1.0)
        try:
            ps_val = float(rec_series.get("Pattern_Score", 0.0) or 0.0)
            if np.isfinite(ps_val) and ps_val > 0.0:
                reasons.append("Bullish pattern detected")
                swing_strength += 1.0
        except (TypeError, ValueError):
            pass
        # 4. Favorable risk/reward — PRIMARY for any trade (weight=1.0)
        try:
            rr_val = float(rec_series.get("RR", rec_series.get("RR_Ratio", np.nan)))
            if np.isfinite(rr_val) and rr_val >= 2.0:
                reasons.append("Favorable risk/reward ratio")
                swing_strength += 1.0
        except (TypeError, ValueError):
            pass
        # 5. Volume confirmation — swing signal (weight=0.75)
        try:
            vol_surge = float(rec_series.get("VolSurge", rec_series.get("Volume_Surge_Ratio", np.nan)))
            if np.isfinite(vol_surge) and vol_surge >= 1.3:
                reasons.append("Volume surge confirmation")
                swing_strength += 0.75
        except (TypeError, ValueError):
            pass
        # 6. Market regime — supporting context (weight=0.25)
        try:
            reg_val = str(rec_series.get("Market_Regime") or "").upper()
            if reg_val in ("TREND_UP", "BULLISH", "NEUTRAL", "SIDEWAYS"):
                reasons.append("Supportive market regime")
                swing_strength += 0.25
        except (TypeError, ValueError, AttributeError):
            pass
        # 7. Fundamental support — minor quality check (weight=0.25)
        #    Only 10% of score, so this is context, not a driver
        try:
            fund_val = float(rec_series.get("Fundamental_S", rec_series.get("Fundamental_Score", np.nan)))
            if np.isfinite(fund_val) and fund_val >= 70.0:
                reasons.append("Fundamental support")
                swing_strength += 0.25
        except (TypeError, ValueError):
            pass
        # Quality based on weighted swing strength, not just reason count
        cnt = len(reasons)
        quality = "High" if swing_strength >= 3.0 else ("Medium" if swing_strength >= 1.5 else "Speculative")
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
    max_workers = min(4, max(1, len(data_map)))  # Conservative for Streamlit Cloud (1GB RAM)
    batch_size = 20  # Process in small batches to limit peak memory
    items = list(data_map.items())
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map: dict = {}
            for tkr, df in batch:
                future = executor.submit(
                    _process_single_ticker, tkr, df, skip_tech_filter,
                )
                future_map[future] = tkr
            try:
                for future in as_completed(future_map, timeout=120):
                    tkr = future_map[future]
                    try:
                        res = future.result()
                        if res is not None:
                            rows.append(res)
                    except Exception as exc:
                        logger.warning(f"Ticker {tkr} failed in parallel scoring: {exc}")
            except FuturesTimeoutError:
                pending = [future_map[f] for f in future_map if not f.done()]
                logger.warning(f"Ticker scoring batch timed out after 120s. Skipping pending tickers: {pending}")
        # Free memory between batches
        gc.collect()
        if status_callback and i > 0 and i % 100 == 0:
            status_callback(f"Scored {i}/{len(items)} tickers...")

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

    if "Score" not in results.columns:
        if "FinalScore_20d" in results.columns:
            results["Score"] = results["FinalScore_20d"]
        elif "TechScore_20d" in results.columns:
            results["Score"] = results["TechScore_20d"]

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
    # NOTE: We do NOT fallback TechScore_20d to FinalScore_20d because they
    # measure different things. TechScore_20d is a technical momentum score;
    # FinalScore_20d is a composite. Mixing them hides scoring failures.
    if ("MomentumScore" not in results.columns) and (
        "TechScore_20d" not in results.columns
    ):
        logger.warning(
            "TechScore_20d missing from %d scored results — "
            "setting to NaN (not masking with FinalScore_20d)",
            len(results),
        )
        results["TechScore_20d"] = np.nan

    return results
