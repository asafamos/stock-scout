"""Pipeline orchestration — phase functions and top-level scan runners.

This is the main module of the ``core.pipeline`` package.  It wires together
sub-modules (market_data, ticker_scoring, universe, helpers, …) and external
``core.*`` modules into the multi-phase scan pipeline.
"""

import dataclasses
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- core modules --------------------------------------------------------
from core.allocation import allocate_budget
from core.api_preflight import run_preflight
from core.classifier import apply_classification
from core.config import get_config
from core.contracts import (
    CoverageMetrics,
    Diagnostics,
    DisagreementStats,
    FallbackCounts,
    MarketContextStatus,
    MLStatus,
    Recommendation,
    RiskSizing,
    ScanMetadata,
    ScanResult,
    Targets,
)
from core.data import (
    fetch_fundamentals_batch,
    fetch_price_multi_source,
    get_fundamentals_safe,
)
from core.data_sources_v2 import (
    disable_provider_category,
    get_last_index_source,
    get_prioritized_fetch_funcs,
    set_default_provider_status,
)
from core.filters import (
    apply_technical_filters,
    compute_advanced_score,
    fetch_benchmark_data,
    should_reject_ticker,
)
from core.market_context import initialize_market_context
from core.ml_20d_inference import ML_20D_AVAILABLE, get_ml_health_meta
from core.provider_guard import get_provider_guard
from core.scoring import compute_fundamental_score_with_breakdown
from core.scoring_config import BYPASS_DISABLED_ABOVE_MIN_SCORE, ML_PROB_THRESHOLD, PATTERN_MIN_SCORE, REGIME_MIN_SCORE, SIGNAL_MIN_SCORE, TOP_SIGNAL_K
from core.scoring_engine import compute_final_score_20d
from core.sector_mapping import get_stock_sector
from core.telemetry import Telemetry

# --- sibling sub-modules --------------------------------------------------
from core.pipeline.config_norm import _normalize_config
from core.pipeline.fallback_tracking import (
    _LEGACY_FALLBACK_REASONS,
    _LEGACY_FALLBACK_USED,
    _LEGACY_LOCK,
)
from core.pipeline.helpers import (
    _canon_column_name,
    _compute_rr_for_row,
    _quantile_safe,
    _t2_pass_and_reasons,
    _to_float,
    check_earnings_blackout,
)
from core.pipeline.market_data import (
    _initialize_ml_context,
    _step_fetch_and_prepare_base_data,
    fetch_beta_vs_benchmark,
)
from core.pipeline.news_sentiment import (
    analyze_sentiment_openai,
    fetch_latest_company_news,
)
from core.pipeline.ticker_scoring import (
    _step_compute_scores_with_unified_logic,
)
from core.pipeline.universe import (
    LAST_UNIVERSE_PROVIDER,
    fetch_top_us_tickers_by_market_cap,
    preflight_check,
)

# Optional: relative-strength helper from root-level module
try:
    from advanced_filters import compute_relative_strength
except ImportError:  # pragma: no cover
    compute_relative_strength = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline context & meta builder
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _PipelineContext:
    """Mutable state bag threaded through pipeline phases."""

    config: Dict[str, Any]
    universe: List[str]
    status_callback: Optional[Callable[[str], None]]
    data_map: Dict[str, pd.DataFrame]
    diagnostics: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    telemetry: Telemetry = dataclasses.field(default_factory=Telemetry)
    provider_status: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    run_mode: str = "OK"
    benchmark_df: Optional[pd.DataFrame] = None
    benchmark_status: str = "OK"
    postfilter_mode: str = "strict"
    filtered_df: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    tier1_pass: List[str] = dataclasses.field(default_factory=list)
    results: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    fundamentals_status: str = "not_requested"
    start_universe: int = 0


def _build_pipeline_meta(ctx: _PipelineContext, **overrides) -> Dict[str, Any]:
    """Build the standard meta dict returned alongside scan results."""
    meta: Dict[str, Any] = {
        "engine_version": "pipeline_v2",
        "engine_mode": "SIGNAL_ONLY",
        "used_legacy_fallback": bool(_LEGACY_FALLBACK_USED),
        "fallback_reason": (
            ", ".join(sorted(set(_LEGACY_FALLBACK_REASONS)))
            if _LEGACY_FALLBACK_REASONS
            else None
        ),
        "sources_used": ctx.telemetry.export(),
        "run_timestamp_utc": datetime.utcnow().isoformat(),
        "postfilter_mode": ctx.postfilter_mode,
        "run_mode": (
            "DEGRADED_TECH_ONLY"
            if ctx.run_mode == "OK"
            and ctx.fundamentals_status in ("not_requested", "requested_empty")
            else ctx.run_mode
        ),
        "benchmark_status": ctx.benchmark_status,
    }
    meta.update(overrides)
    try:
        meta.update(get_ml_health_meta())
        if not ML_20D_AVAILABLE:
            meta["ml_mode"] = "DISABLED_NO_MODEL"
        elif meta.get("ml_bundle_version_warning"):
            # Model loaded and producing predictions despite version mismatch.
            # Mark as degraded hybrid, not disabled — ML IS working.
            meta["ml_mode"] = "HYBRID_DEGRADED"
        else:
            meta["ml_mode"] = "HYBRID"
    except (ImportError, AttributeError, TypeError, KeyError):
        pass
    return meta


# ---------------------------------------------------------------------------
# Phase functions
# ---------------------------------------------------------------------------

def _phase_init_context(ctx: _PipelineContext) -> Optional[Dict[str, Any]]:
    """Phases 0-4: config, telemetry, market context, ML, preflight, gating.

    Returns an early-exit result dict when the scan must abort, else *None*.
    """
    try:
        logger.info(
            "\U0001f30c Starting pipeline with universe size: %d tickers",
            len(ctx.universe),
        )
    except (TypeError, ValueError):
        pass

    ctx.config = _normalize_config(ctx.config)
    ctx.start_universe = len(ctx.universe)

    if ctx.status_callback:
        ctx.status_callback(f"Starting pipeline for {len(ctx.universe)} tickers...")

    ctx.telemetry = Telemetry()
    try:
        ctx.telemetry.set_value("universe_provider", LAST_UNIVERSE_PROVIDER)
    except (AttributeError, TypeError):
        pass

    # ---- Market context (SPY + VIX) ----
    try:
        initialize_market_context(symbols=["SPY", "^VIX"], period_days=250)
        logger.info("[PIPELINE] Global index cache initialized (SPY, VIX)")
        try:
            spy_src = get_last_index_source("SPY")
            vix_src = get_last_index_source("^VIX") or get_last_index_source("VIX")
            idx_map: Dict[str, Any] = {}
            if spy_src:
                idx_map["SPY"] = spy_src
                try:
                    if str(spy_src).upper() == "POLYGON":
                        ctx.telemetry.mark_used("price", "POLYGON")
                except (AttributeError, TypeError):
                    pass
            if vix_src:
                idx_map["VIX"] = vix_src
            ctx.telemetry.set_value("index", idx_map)
            if vix_src and "Synthetic" in vix_src:
                ctx.telemetry.record_fallback(
                    "index",
                    "provider_chain",
                    "SYNTHETIC_VIX_PROXY",
                    "constructed proxy from SPY volatility",
                )
        except (AttributeError, TypeError, KeyError):
            pass
    except Exception as e:
        logger.debug("[PIPELINE] Market context init skipped: %s", e)

    # ---- ML context ----
    try:
        _initialize_ml_context()
    except Exception as e:
        logger.debug("[PIPELINE] ML context init skipped: %s", e)

    # ---- API Preflight ----
    try:
        ctx.provider_status = run_preflight()
        ctx.run_mode = ctx.provider_status.get("SCAN_STATUS", "OK")
        try:
            ctx.telemetry.set_value(
                "preflight_status", ctx.provider_status.get("ACTIVE_COUNTS", {})
            )
        except (AttributeError, TypeError):
            pass
        try:
            set_default_provider_status(ctx.provider_status)
            try:
                guard = get_provider_guard()
                guard.update_from_preflight(ctx.provider_status)
                ctx.telemetry.set_value("provider_states", guard.snapshot())
            except (ImportError, AttributeError, TypeError):
                pass
            if not ctx.provider_status.get("FMP_INDEX", {"ok": True}).get("ok", True):
                disable_provider_category("fmp", "index")
            try:
                funcs = get_prioritized_fetch_funcs(ctx.provider_status)
                name_map = {
                    "fmp": "FMP",
                    "finnhub": "FINNHUB",
                    "tiingo": "TIINGO",
                    "alpha": "ALPHAVANTAGE",
                    "eodhd": "EODHD",
                    "simfin": "SIMFIN",
                }
                parts = []
                for internal_name, _func in funcs:
                    up = name_map.get(internal_name, internal_name.upper())
                    p_meta = ctx.provider_status.get(up, {})
                    lat = p_meta.get("latency")
                    if isinstance(lat, (int, float)):
                        parts.append(f"{up} ({lat:.2f}s)")
                    else:
                        parts.append(f"{up} (n/a)")
                if parts:
                    logger.info("Smart Routing: " + " -> ".join(parts))
            except (ImportError, KeyError, TypeError, AttributeError):
                pass
        except (ImportError, AttributeError):
            pass
    except (ImportError, KeyError) as exc:
        logger.debug("API preflight skipped: %s", exc)
        ctx.provider_status = {}
        ctx.run_mode = "OK"

    # Reset fallback trackers
    try:
        with _LEGACY_LOCK:
            import core.pipeline.fallback_tracking as _ft

            _ft._LEGACY_FALLBACK_USED = False
            _ft._LEGACY_FALLBACK_REASONS = []
    except (RuntimeError, NameError):
        pass

    # ---- Gating ----
    try:
        if ctx.run_mode == "BLOCKED":
            logger.error(
                "[PIPELINE] Preflight BLOCKED: no active price providers; aborting scan"
            )
            meta = _build_pipeline_meta(ctx, postfilter_mode="blocked", run_mode="BLOCKED")
            return {
                "result": {
                    "results_df": pd.DataFrame(),
                    "data_map": {},
                    "diagnostics": {},
                },
                "meta": meta,
            }
        elif ctx.run_mode == "DEGRADED_TECH_ONLY":
            ctx.config["fundamental_enabled"] = False
            logger.warning(
                "[PIPELINE] Preflight DEGRADED_TECH_ONLY: fundamentals disabled for this run"
            )
    except (KeyError, TypeError):
        pass

    return None


def _phase_fetch_and_tier1(ctx: _PipelineContext) -> None:
    """Phases 5-7: fetch historical data, RS ranking, Tier 1 OHLCV filter.

    Populates *ctx.data_map*, *ctx.benchmark_df*, *ctx.tier1_pass*,
    *ctx.filtered_df*, *ctx.benchmark_status*, *ctx.postfilter_mode*.
    """
    ctx.data_map, ctx.benchmark_df = _step_fetch_and_prepare_base_data(
        ctx.universe, ctx.config, ctx.status_callback, ctx.data_map,
    )

    # Benchmark status
    ctx.benchmark_status = "OK"
    try:
        if ctx.benchmark_df is None or (
            hasattr(ctx.benchmark_df, "empty") and ctx.benchmark_df.empty
        ):
            ctx.benchmark_status = "UNAVAILABLE"
            logger.warning(
                "[PIPELINE] Benchmark unavailable; skipping RS/Beta dependent steps"
            )
    except (AttributeError, TypeError):
        ctx.benchmark_status = "UNAVAILABLE"

    try:
        small = bool(
            ctx.start_universe < 50
            or bool(ctx.config.get("smoke_mode", False))
        )
    except (TypeError, AttributeError):
        small = bool(ctx.start_universe < 50)
    ctx.postfilter_mode = "lenient_small_universe" if small else "strict"

    # Minimal price telemetry sample
    try:
        if ctx.universe:
            tkr0 = str(ctx.universe[0])
            _ = fetch_price_multi_source(
                tkr0,
                provider_status=ctx.provider_status,
                telemetry=ctx.telemetry,
            )
    except (ImportError, IndexError, TypeError) as exc:
        logger.debug("Price telemetry sample skipped: %s", exc)

    # ---- RS blended ranking (>100 tickers, env-gated) ----
    rs_enabled_env = os.getenv("RS_RANKING", os.getenv("RS_RANKING_ENABLED", "1"))
    rs_enabled = bool(rs_enabled_env == "1")
    if rs_enabled and ctx.start_universe > 100 and compute_relative_strength is not None:
        if ctx.status_callback:
            ctx.status_callback("Ranking universe by blended RS (21/63d)...")
        try:
            bench_df = fetch_benchmark_data("SPY", days=200)
            if bench_df is None or bench_df.empty:
                raise ValueError("Benchmark DataFrame is empty; skipping RS blended ranking")
            rs_records: List[Dict[str, Any]] = []
            for tkr, df in (ctx.data_map or {}).items():
                try:
                    if df is None or df.empty:
                        rs_records.append({"Ticker": tkr, "RS_blend": np.nan})
                        continue
                    rs = compute_relative_strength(
                        df.rename(columns=str.title),
                        bench_df.rename(columns=str.title),
                        periods=[21, 63],
                    )
                    rs63 = rs.get("rs_63d", rs.get("RS_63d", np.nan))
                    rs21 = rs.get("rs_21d", rs.get("RS_21d", np.nan))
                    if pd.notna(rs63) and pd.notna(rs21):
                        rs_blend = 0.7 * float(rs63) + 0.3 * float(rs21)
                    else:
                        rs_blend = rs63 if pd.notna(rs63) else rs21
                    rs_records.append({"Ticker": tkr, "RS_blend": rs_blend})
                except (KeyError, TypeError, ValueError):
                    rs_records.append({"Ticker": tkr, "RS_blend": np.nan})
            if rs_records:
                rs_df = pd.DataFrame(rs_records)
                rs_df["RS_blend_Pctl"] = rs_df["RS_blend"].rank(pct=True, ascending=True)
                logger.info(
                    "[PIPELINE] RS blended ranking computed for %d tickers "
                    "(no hard filter applied)",
                    len(rs_records),
                )
        except Exception as e:
            logger.warning("RS blended ranking failed (continuing without): %s", e)

    # ---- Tier 1: Fast OHLCV filter ----
    if ctx.status_callback:
        ctx.status_callback("Tier 1: applying OHLCV filters...")

    ctx.tier1_pass = []
    filtered_rows: List[Dict[str, Any]] = []
    for tkr, df in (ctx.data_map or {}).items():
        try:
            if df is None or df.empty:
                ctx.diagnostics.setdefault(
                    tkr, {"tier1_reasons": [], "tier2_reasons": []}
                )
                ctx.diagnostics[tkr]["tier1_reasons"].append(
                    {"rule": "EMPTY_HISTORY", "message": "No historical OHLCV data"}
                )
                ctx.diagnostics[tkr]["last_price"] = np.nan
                ctx.diagnostics[tkr]["last_volume"] = np.nan
                filtered_rows.append(
                    {
                        "Ticker": tkr,
                        "Tier1_Reasons": "EMPTY_HISTORY",
                        "last_price": np.nan,
                        "last_volume": np.nan,
                    }
                )
                continue

            try:
                cols_lower = {_canon_column_name(c): c for c in df.columns}
            except (AttributeError, TypeError):
                cols_lower = {}

            close_series = None
            vol_series = None
            if "close" in cols_lower:
                close_series = pd.to_numeric(df[cols_lower["close"]], errors="coerce")
            elif "adj close" in cols_lower:
                close_series = pd.to_numeric(
                    df[cols_lower["adj close"]], errors="coerce"
                )
            if "volume" in cols_lower:
                vol_series = pd.to_numeric(df[cols_lower["volume"]], errors="coerce")
            elif "v" in cols_lower:
                vol_series = pd.to_numeric(df[cols_lower["v"]], errors="coerce")

            close = (
                close_series.dropna().iloc[-1]
                if isinstance(close_series, pd.Series) and not close_series.dropna().empty
                else np.nan
            )
            volume = (
                vol_series.dropna().iloc[-1]
                if isinstance(vol_series, pd.Series) and not vol_series.dropna().empty
                else np.nan
            )

            reasons: List[Dict[str, Any]] = []
            if pd.isna(volume):
                reasons.append(
                    {"rule": "MISSING_VOLUME_DATA", "message": "Missing Volume on last bar"}
                )
            elif volume < 100_000:
                reasons.append(
                    {
                        "rule": "VOLUME_MIN",
                        "message": "Volume below minimum",
                        "value": float(volume) if pd.notna(volume) else None,
                        "threshold": 100_000,
                    }
                )
            if pd.isna(close):
                reasons.append(
                    {"rule": "MISSING_PRICE_DATA", "message": "Missing Close price on last bar"}
                )
            elif close < 1.0:
                reasons.append(
                    {
                        "rule": "PRICE_MIN",
                        "message": "Price below $1",
                        "value": float(close),
                        "threshold": 1.0,
                    }
                )
            if pd.notna(close) and close > 10000:
                reasons.append(
                    {
                        "rule": "PRICE_MAX",
                        "message": "Price above $10,000",
                        "value": float(close),
                        "threshold": 10000.0,
                    }
                )

            row_for_filter = pd.Series({"Volume": volume, "Close": close})
            if (not reasons) and apply_technical_filters(row_for_filter, relaxed=False):
                ctx.tier1_pass.append(tkr)
                ctx.diagnostics.setdefault(
                    tkr, {"tier1_reasons": [], "tier2_reasons": []}
                )
                ctx.diagnostics[tkr]["last_price"] = (
                    float(close) if pd.notna(close) else np.nan
                )
                ctx.diagnostics[tkr]["last_volume"] = (
                    float(volume) if pd.notna(volume) else np.nan
                )
            else:
                ctx.diagnostics.setdefault(
                    tkr, {"tier1_reasons": [], "tier2_reasons": []}
                )
                if not reasons:
                    reasons.append(
                        {"rule": "TECH_FILTER_FAIL", "message": "Failed Tier 1 technical filter"}
                    )
                ctx.diagnostics[tkr]["tier1_reasons"].extend(reasons)
                ctx.diagnostics[tkr]["last_price"] = (
                    float(close) if pd.notna(close) else np.nan
                )
                ctx.diagnostics[tkr]["last_volume"] = (
                    float(volume) if pd.notna(volume) else np.nan
                )
                try:
                    joined = ";".join(
                        [str(r.get("rule")) for r in reasons]
                    )
                except (TypeError, AttributeError, KeyError):
                    joined = ";".join([str(r) for r in reasons])
                filtered_rows.append(
                    {
                        "Ticker": tkr,
                        "Tier1_Reasons": joined,
                        "last_price": float(close) if pd.notna(close) else np.nan,
                        "last_volume": float(volume) if pd.notna(volume) else np.nan,
                    }
                )
        except Exception as exc:
            logger.debug("Tier1 filter failed for %s: %s", tkr, exc)
            ctx.diagnostics.setdefault(tkr, {"tier1_reasons": [], "tier2_reasons": []})
            ctx.diagnostics[tkr]["tier1_reasons"].append(
                {
                    "rule": "EXCEPTION",
                    "message": f"Tier1 error: {exc.__class__.__name__}",
                }
            )
            ctx.diagnostics[tkr]["last_price"] = np.nan
            ctx.diagnostics[tkr]["last_volume"] = np.nan
            filtered_rows.append(
                {
                    "Ticker": tkr,
                    "Tier1_Reasons": "EXCEPTION",
                    "last_price": np.nan,
                    "last_volume": np.nan,
                }
            )
            continue

    filtered_count = ctx.start_universe - len(ctx.tier1_pass)
    logger.info(
        "[PIPELINE] Tier 1: scanned=%d, passed=%d, filtered=%d",
        ctx.start_universe,
        len(ctx.tier1_pass),
        filtered_count,
    )

    try:
        ctx.filtered_df = pd.DataFrame(filtered_rows)
    except (TypeError, ValueError):
        ctx.filtered_df = pd.DataFrame()


def _phase_score_and_filter(ctx: _PipelineContext) -> Optional[Dict[str, Any]]:
    """Phases 8-11: Tier 2 scoring, beta, advanced filters, meteor.

    Populates *ctx.results*.  Returns early-exit dict on empty results, else *None*.
    """
    # Build Tier 2 input from Tier 1 output
    tier2_map: Dict[str, pd.DataFrame] = {
        t: ctx.data_map[t] for t in ctx.tier1_pass if t in ctx.data_map
    }
    if not tier2_map and isinstance(ctx.data_map, dict) and ctx.data_map:
        logger.warning(
            "[PIPELINE] Tier 1 yielded no candidates; "
            "falling back to relaxed Tier 2 on full universe"
        )
        tier2_map = dict(ctx.data_map)

    # ---- Tier 2: indicators + ML ----
    ctx.results = _step_compute_scores_with_unified_logic(
        tier2_map, ctx.config, ctx.status_callback, skip_tech_filter=True,
    )

    # Verify Tier 2 I/O
    tier2_input_set = set(tier2_map.keys())
    try:
        if not ctx.results.empty:
            if "Ticker" in ctx.results.columns:
                tier2_output_set = set(
                    str(x) for x in ctx.results["Ticker"].tolist() if pd.notna(x)
                )
            elif ctx.results.index.name == "Ticker":
                tier2_output_set = set(
                    str(x) for x in ctx.results.index.tolist() if pd.notna(x)
                )
            elif "ticker" in ctx.results.columns:
                tier2_output_set = set(
                    str(x).upper().replace(".", "-").replace("/", "-")
                    for x in ctx.results["ticker"].tolist()
                    if pd.notna(x)
                )
            else:
                tier2_output_set = set()
        else:
            tier2_output_set = set()
    except (KeyError, AttributeError, TypeError):
        tier2_output_set = set()
    if tier2_input_set == tier2_output_set:
        logger.info(
            "[PIPELINE] Tier 2 verification OK: input=%d equals output tickers",
            len(tier2_input_set),
        )
    else:
        missing = sorted(tier2_input_set - tier2_output_set)
        extra = sorted(tier2_output_set - tier2_input_set)
        logger.warning(
            "[PIPELINE] Tier 2 verification mismatch: input=%d, output=%d; "
            "missing=%d, extra=%d",
            len(tier2_input_set),
            len(tier2_output_set),
            len(missing),
            len(extra),
        )

    # Empty guard
    if ctx.results.empty:
        try:
            ctx.telemetry.set_value("fundamentals_status", "not_requested")
        except (AttributeError, TypeError):
            pass
        meta = _build_pipeline_meta(ctx)
        try:
            if isinstance(ctx.data_map, dict):
                ctx.data_map = dict(ctx.data_map)
                ctx.data_map["filtered_tier1_df"] = ctx.filtered_df
        except (TypeError, KeyError):
            pass
        return {
            "result": {
                "results_df": ctx.results,
                "data_map": ctx.data_map,
                "diagnostics": ctx.diagnostics,
            },
            "meta": meta,
        }

    if "Score" not in ctx.results.columns:
        if "FinalScore_20d" in ctx.results.columns:
            ctx.results["Score"] = ctx.results["FinalScore_20d"]
        elif "TechScore_20d" in ctx.results.columns:
            ctx.results["Score"] = ctx.results["TechScore_20d"]
        else:
            ctx.results["Score"] = 0.0

    # ---- Beta filter ----
    if ctx.config.get("beta_filter_enabled"):
        if ctx.status_callback:
            ctx.status_callback("Applying Beta filter...")
        beta_max = float(ctx.config.get("beta_max_allowed", 1.5))
        top_k = int(ctx.config.get("beta_top_k", 50))
        ctx.results = ctx.results.sort_values("Score", ascending=False)
        to_check = ctx.results.head(top_k).index
        for idx in to_check:
            tkr = ctx.results.at[idx, "Ticker"]
            b = fetch_beta_vs_benchmark(tkr, ctx.config.get("beta_benchmark", "SPY"))
            ctx.results.at[idx, "Beta"] = b
        ctx.results = ctx.results[
            ~((ctx.results["Beta"].notna()) & (ctx.results["Beta"] > beta_max))
        ]
        logger.info("[PIPELINE] After beta filter: %d remain", len(ctx.results))

    # ---- Advanced filters (penalties) ----
    if ctx.status_callback:
        ctx.status_callback("Applying advanced filters...")

    signals_store: List[Tuple[Any, Dict, float]] = []
    for idx, row in ctx.results.iterrows():
        tkr = row["Ticker"]
        if tkr in ctx.data_map:
            df = ctx.data_map[tkr]
            base_score = row["Score"]
            df_title = df.rename(columns=str.title)
            if "Date" not in df_title.columns:
                try:
                    df_title = df_title.reset_index()
                    if "Date" not in df_title.columns and df_title.columns[0] not in (
                        "Date",
                        "date",
                    ):
                        first = df_title.columns[0]
                        df_title = df_title.rename(columns={first: "Date"})
                except (KeyError, IndexError, AttributeError):
                    pass
            bench_title = ctx.benchmark_df.rename(columns=str.title)
            enhanced, sig = compute_advanced_score(
                tkr, df_title, bench_title, base_score / 100.0
            )
            signals_store.append((idx, sig, enhanced))

    rs_vals = [
        s.get("rs_63d")
        for _, s, _ in signals_store
        if s.get("rs_63d") is not None
    ]
    mom_vals = [
        s.get("momentum_consistency")
        for _, s, _ in signals_store
        if s.get("momentum_consistency") is not None
    ]
    rr_vals = [
        s.get("risk_reward_ratio")
        for _, s, _ in signals_store
        if s.get("risk_reward_ratio") is not None
    ]

    num_stocks = len(signals_store)
    if num_stocks < 20:
        rs_thresh = -0.40
        mom_thresh = 0.10
        rr_thresh = 0.30
        logger.info(
            "[PIPELINE] Using fixed lenient thresholds for %d stocks", num_stocks
        )
    else:
        rs_thresh = min(_quantile_safe(rs_vals, 0.02, -0.40), -0.30)
        mom_thresh = min(_quantile_safe(mom_vals, 0.05, 0.10), 0.12)
        rr_thresh = min(_quantile_safe(rr_vals, 0.05, 0.30), 0.40)
        logger.info(
            "[PIPELINE] Using dynamic thresholds for %d stocks", num_stocks
        )

    dyn_thresh = {
        "rs_63d": rs_thresh,
        "momentum_consistency": mom_thresh,
        "risk_reward_ratio": rr_thresh,
    }
    logger.info(
        "[PIPELINE] Thresholds: RS=%.3f, Mom=%.3f, RR=%.3f",
        rs_thresh,
        mom_thresh,
        rr_thresh,
    )

    for idx, sig, enhanced in signals_store:
        catastrophic, reason = should_reject_ticker(sig, dynamic=dyn_thresh)

        ctx.results.at[idx, "RS_63d"] = sig.get("rs_63d")
        ctx.results.at[idx, "Volume_Surge"] = sig.get("volume_surge")
        ctx.results.at[idx, "MA_Aligned"] = sig.get("ma_aligned")
        ctx.results.at[idx, "Signal_Boost_Score"] = sig.get("signal_boost_score")
        ctx.results.at[idx, "RR_Ratio"] = sig.get("risk_reward_ratio")
        ctx.results.at[idx, "Momentum_Consistency"] = sig.get("momentum_consistency")

        if catastrophic:
            ctx.results.at[idx, "AdvFilter_Score"] = float(enhanced)
            ctx.results.at[idx, "RejectionReason"] = reason
            try:
                tkr = (
                    str(ctx.results.at[idx, "Ticker"])
                    if "Ticker" in ctx.results.columns
                    else None
                )
                if tkr:
                    ctx.diagnostics.setdefault(
                        tkr, {"tier1_reasons": [], "tier2_reasons": []}
                    )
                    ctx.diagnostics[tkr]["tier2_reasons"].append(
                        {
                            "rule": "ADVANCED_REJECT",
                            "message": str(reason) if reason else "Advanced filters rejection",
                        }
                    )
            except (KeyError, TypeError, AttributeError):
                pass
        else:
            # Penalty for below-threshold metrics (each 0.05-0.10 on 0-1 scale)
            # Before fix: penalties divided by 100 → negligible 0.01-0.035 impact.
            # Now: meaningful 5-10% reduction per weak metric.
            penalty = 0.0
            if sig.get("rs_63d", 0) < rs_thresh:
                penalty += 0.05   # weak relative strength
            if sig.get("momentum_consistency", 0) < mom_thresh:
                penalty += 0.05   # weak momentum
            if sig.get("risk_reward_ratio", 0) < rr_thresh:
                penalty += 0.10   # poor risk/reward
            ctx.results.at[idx, "AdvPenalty"] = penalty
            ctx.results.at[idx, "AdvFilter_Score"] = max(
                0.01, float(enhanced) - penalty
            )
            try:
                tkr = (
                    str(ctx.results.at[idx, "Ticker"])
                    if "Ticker" in ctx.results.columns
                    else None
                )
                if tkr:
                    ctx.diagnostics.setdefault(
                        tkr, {"tier1_reasons": [], "tier2_reasons": []}
                    )
                    if (
                        sig.get("rs_63d", None) is not None
                        and sig.get("rs_63d") < rs_thresh
                    ):
                        ctx.diagnostics[tkr]["tier2_reasons"].append(
                            {
                                "rule": "RS_BELOW_THRESH",
                                "message": "RS_63d below threshold",
                                "value": float(sig.get("rs_63d")),
                                "threshold": float(rs_thresh),
                            }
                        )
                    if (
                        sig.get("momentum_consistency", None) is not None
                        and sig.get("momentum_consistency") < mom_thresh
                    ):
                        ctx.diagnostics[tkr]["tier2_reasons"].append(
                            {
                                "rule": "MOM_BELOW_THRESH",
                                "message": "Momentum consistency below threshold",
                                "value": float(sig.get("momentum_consistency")),
                                "threshold": float(mom_thresh),
                            }
                        )
                    if (
                        sig.get("risk_reward_ratio", None) is not None
                        and sig.get("risk_reward_ratio") < rr_thresh
                    ):
                        ctx.diagnostics[tkr]["tier2_reasons"].append(
                            {
                                "rule": "RR_BELOW_THRESH",
                                "message": "Risk/Reward below threshold",
                                "value": float(sig.get("risk_reward_ratio")),
                                "threshold": float(rr_thresh),
                            }
                        )
            except (KeyError, TypeError, AttributeError):
                pass

    # Scale AdvFilter_Score to 0-100 (diagnostic only; FinalScore_20d set by W6)
    if "AdvFilter_Score" in ctx.results.columns:
        ctx.results["AdvFilter_Score"] = ctx.results["AdvFilter_Score"] * 100.0
    logger.info(
        "[PIPELINE] Advanced filters applied without score-zeroing; total stocks: %d",
        len(ctx.results),
    )

    # ---- Meteor mode (optional) ----
    meteor_mode = bool(
        ctx.config.get("meteor_mode", bool(os.getenv("METEOR_MODE", "0") == "1"))
    )
    if meteor_mode and not ctx.results.empty:
        if ctx.status_callback:
            ctx.status_callback("Applying Meteor filters (VCP + RS + Pocket Pivots)...")
        try:
            bench_df = fetch_benchmark_data("SPY", days=200)
            if bench_df is None or bench_df.empty:
                logger.warning("Benchmark DataFrame empty; skipping Meteor filters")
            else:
                kept_rows = []
                for _, row in ctx.results.iterrows():
                    tkr = str(row.get("Ticker"))
                    df = tier2_map.get(tkr)
                    if df is None or (hasattr(df, "empty") and df.empty):
                        df = ctx.data_map.get(tkr)
                    if df is None or (hasattr(df, "empty") and df.empty):
                        continue
                    base_score = float(
                        row.get("FinalScore_20d", row.get("Score", 0.0))
                    )
                    new_score, details = compute_advanced_score(
                        tkr,
                        df.rename(columns=str.title),
                        bench_df.rename(columns=str.title),
                        base_score,
                    )
                    if details.get("passed"):
                        row = row.copy()
                        row["Meteor_Score"] = float(new_score)
                        row["Meteor_Passed"] = True
                        row["Meteor_Reason"] = details.get("reason")
                        kept_rows.append(row)
                ctx.results = pd.DataFrame(kept_rows)
                logger.info(
                    "[PIPELINE] Meteor Mode: %d candidates after filters",
                    len(ctx.results),
                )
        except Exception as e:
            logger.warning("Meteor filter application failed: %s", e)

    return None


def _phase_enrich_fundamentals(ctx: _PipelineContext) -> None:
    """Phase 12-14: fetch fundamentals, merge, sector mapping, score recalc.

    Updates *ctx.results* in place and sets *ctx.fundamentals_status*.
    """
    if not ctx.config.get("fundamental_enabled", True):
        ctx.results["Sector"] = "Unknown"
        ctx.results["Fundamental_S"] = 50.0
        try:
            ctx.telemetry.set_value("fundamentals_status", "not_requested")
            ctx.fundamentals_status = "not_requested"
        except (AttributeError, TypeError):
            pass
        _apply_sector_mapping(ctx)
        return

    if ctx.status_callback:
        ctx.status_callback("Fetching fundamentals & sector data...")

    # Signal date for fundamentals
    fund_as_of = None
    try:
        if "As_Of_Date" in ctx.results.columns and len(ctx.results) > 0:
            fund_as_of = pd.to_datetime(ctx.results["As_Of_Date"].iloc[0]).date()
        else:
            dates = []
            for df in (ctx.data_map or {}).values():
                if df is not None and not df.empty:
                    try:
                        dates.append(pd.to_datetime(df.index[-1]).date())
                    except (IndexError, TypeError, ValueError):
                        pass
            if dates:
                fund_as_of = max(dates)
    except (KeyError, TypeError, ValueError):
        fund_as_of = None

    # Score/cap thresholds
    # NOTE: Threshold lowered from 60 → 35 (2026-02-15) to ensure speculative
    # stocks (classified as SPEC at score >= 40) also receive fundamental data.
    # Previously, SPEC stocks never got fundamentals, resulting in N/A for
    # Quality/Growth/Valuation/Leverage and a reliability floor of 40.
    try:
        score_thr = float(
            ctx.config.get(
                "fundamentals_score_threshold",
                float(os.getenv("FUNDAMENTALS_SCORE_THRESHOLD", "35")),
            )
        )
    except (TypeError, ValueError):
        score_thr = 35.0
    try:
        top_n_cap = int(
            ctx.config.get(
                "fundamentals_top_n_cap",
                int(os.getenv("FUNDAMENTALS_TOP_N_CAP", "100")),
            )
        )
    except (TypeError, ValueError):
        top_n_cap = 100
    if top_n_cap > 0:
        logger.info(
            "\u26a1 Fetching fundamentals for Top-%d stocks (score > %s)",
            top_n_cap,
            score_thr,
        )
    else:
        logger.info(
            "\u26a1 Fetching fundamentals for ALL stocks with score > %s (no cap)",
            score_thr,
        )

    # Ensure scoring columns exist
    try:
        if "Score" not in ctx.results.columns and "FinalScore_20d" in ctx.results.columns:
            ctx.results["Score"] = ctx.results["FinalScore_20d"]
        if "TechScore_20d" not in ctx.results.columns:
            if "Score" in ctx.results.columns:
                ctx.results["TechScore_20d"] = ctx.results["Score"]
            elif "FinalScore_20d" in ctx.results.columns:
                ctx.results["TechScore_20d"] = ctx.results["FinalScore_20d"]
    except (KeyError, TypeError):
        pass

    score_col = (
        "TechScore_20d"
        if "TechScore_20d" in ctx.results.columns
        else ("Score" if "Score" in ctx.results.columns else None)
    )
    if score_col is None or ctx.results.empty:
        eligible = pd.DataFrame(columns=ctx.results.columns)
    else:
        eligible = ctx.results[ctx.results[score_col] > score_thr].sort_values(
            score_col, ascending=False
        )
    if top_n_cap > 0:
        eligible = eligible.head(top_n_cap)

    try:
        if "Ticker" in eligible.columns:
            winners = eligible["Ticker"].tolist()
        elif eligible.index.name == "Ticker":
            winners = eligible.index.tolist()
        elif "ticker" in eligible.columns:
            winners = [
                str(x).upper().replace(".", "-").replace("/", "-")
                for x in eligible["ticker"].tolist()
            ]
        else:
            winners = []
    except (KeyError, AttributeError, TypeError):
        winners = []

    # Fetch
    if not winners:
        logger.info(
            "No winners with Score > %.0f; skipping fundamentals fetch", score_thr
        )
        fund_df = pd.DataFrame()
        try:
            ctx.telemetry.set_value("fundamentals_status", "requested_empty")
            ctx.fundamentals_status = "requested_empty"
        except (AttributeError, TypeError):
            pass
    else:
        logger.info(
            "Fetching fundamentals for %d winners (%s > %.0f, TopN=%d)",
            len(winners),
            score_col,
            score_thr,
            top_n_cap,
        )
        logger.info(
            "[PIPELINE] Sending %d tickers to fetch_fundamentals_batch",
            len(winners),
        )
        fund_df = fetch_fundamentals_batch(
            winners,
            provider_status=ctx.provider_status,
            as_of_date=fund_as_of,
            telemetry=ctx.telemetry,
        )
        try:
            status_val = (
                "used"
                if (isinstance(fund_df, pd.DataFrame) and not fund_df.empty)
                else "requested_empty"
            )
            ctx.telemetry.set_value("fundamentals_status", status_val)
            ctx.fundamentals_status = status_val
        except (AttributeError, TypeError):
            pass

    # Normalize fund_df index
    if isinstance(fund_df.index, pd.Index):
        fund_df = fund_df.reset_index()
        if "ticker" in fund_df.columns:
            fund_df = fund_df.rename(columns={"ticker": "Ticker"})
        elif "index" in fund_df.columns and "Ticker" not in fund_df.columns:
            fund_df = fund_df.rename(columns={"index": "Ticker"})
    if "Ticker" not in fund_df.columns and len(fund_df) > 0:
        if len(fund_df.columns) > 0:
            first_col = fund_df.columns[0]
            if fund_df[first_col].dtype == "object":
                fund_df = fund_df.rename(columns={first_col: "Ticker"})

    # Merge
    if "Ticker" in fund_df.columns and len(fund_df) > 0:
        ctx.results = pd.merge(
            ctx.results, fund_df, on="Ticker", how="left", suffixes=("", "_fund")
        )
    else:
        logger.warning("Fundamental data has no Ticker column, skipping merge")
        try:
            ctx.telemetry.set_value("fundamentals_status", "requested_empty")
            ctx.fundamentals_status = "requested_empty"
        except (AttributeError, TypeError):
            pass

    # Map aggregated columns to UI names + fill missing
    try:
        _col_map = {
            "market_cap": "Market_Cap",
            "beta": "Beta",
            "sector": "Sector",
            "pe": "PE_Ratio",
            "peg": "PEG_Ratio",
            "debt_equity": "Debt_to_Equity",
            "roe": "ROE",
            "margin": "Margin",
            "ps": "PS_Ratio",
            "pb": "PB_Ratio",
        }
        for src, dst in _col_map.items():
            if src in ctx.results.columns:
                # Overwrite uppercase version if it's all NaN (shadow fix)
                if dst in ctx.results.columns and ctx.results[dst].isna().all():
                    ctx.results[dst] = ctx.results[src]
                elif dst not in ctx.results.columns:
                    ctx.results[dst] = ctx.results[src]

        for col in ["Market_Cap", "PE_Ratio", "PEG_Ratio", "Beta", "Debt_to_Equity"]:
            if col in ctx.results.columns:
                ctx.results[col] = pd.to_numeric(ctx.results[col], errors="coerce")

        # Fill missing via get_fundamentals_safe (winners only)
        ui_cols = [
            "Market_Cap",
            "PE_Ratio",
            "PEG_Ratio",
            "PB_Ratio",
            "Beta",
            "Sector",
            "Industry",
            "Debt_to_Equity",
            "ROE",
            "Vol_Avg",
            "Dividend",
            "Price",
        ]
        winners_set = set(winners) if winners else set()
        fill_count = 0
        for idx, row in ctx.results.iterrows():
            tkr = row.get("Ticker")
            if not tkr or tkr not in winners_set:
                continue
            need_fill = any(pd.isna(row.get(c)) for c in ui_cols)
            if not need_fill:
                continue
            safe = get_fundamentals_safe(str(tkr))
            if not safe:
                continue
            fill_count += 1
            for c in ui_cols:
                if pd.isna(row.get(c)) and (c in safe):
                    ctx.results.at[idx, c] = safe.get(c)
        if fill_count > 0:
            logger.debug("Filled missing fundamentals for %d winners", fill_count)

        # Valuation / Quality / Leverage
        if "PE_Ratio" in ctx.results.columns:
            ctx.results["Valuation"] = pd.to_numeric(
                ctx.results["PE_Ratio"], errors="coerce"
            )
        else:
            ctx.results["Valuation"] = np.nan
        if "ROE" in ctx.results.columns:
            ctx.results["Quality"] = pd.to_numeric(ctx.results["ROE"], errors="coerce")
        elif "roe" in ctx.results.columns:
            ctx.results["Quality"] = pd.to_numeric(ctx.results["roe"], errors="coerce")
        else:
            ctx.results["Quality"] = np.nan
        if "Debt_to_Equity" in ctx.results.columns:
            ctx.results["Leverage"] = pd.to_numeric(
                ctx.results["Debt_to_Equity"], errors="coerce"
            )
        else:
            ctx.results["Leverage"] = np.nan
    except Exception as e:
        logger.debug("UI mapping/Valuation-Leverage setup skipped: %s", e)

    # Raw metric aliases
    try:
        _alias_map = {
            "pe": "PE",
            "peg": "PEG",
            "roe": "ROE",
            "debt_equity": "Debt_Equity",
        }
        for src, dst in _alias_map.items():
            if src in ctx.results.columns and dst not in ctx.results.columns:
                ctx.results[dst] = ctx.results[src]
    except Exception as e:
        logger.debug("Valuation/Quality column creation skipped: %s", e)

    # Compute fundamental scores
    for idx, row in ctx.results.iterrows():
        try:
            fund_data = row.to_dict()
            has_fund_data = any(
                pd.notna(fund_data.get(f))
                for f in ["pe", "roe", "pb", "margin", "debt_equity",
                          "rev_yoy", "eps_yoy", "roic", "ps"]
            )
            if not has_fund_data:
                # Penalize missing data — 35 is below-average, not neutral 50
                ctx.results.at[idx, "Fundamental_S"] = 35.0
                continue
            fund_score_obj = compute_fundamental_score_with_breakdown(fund_data)
            ctx.results.at[idx, "Fundamental_S"] = fund_score_obj.total
            ctx.results.at[idx, "Quality_Score_F"] = fund_score_obj.breakdown.quality_score
            ctx.results.at[idx, "Growth_Score_F"] = fund_score_obj.breakdown.growth_score
            ctx.results.at[idx, "Valuation_Score_F"] = fund_score_obj.breakdown.valuation_score
        except Exception as e:
            ctx.results.at[idx, "Fundamental_S"] = 35.0
            logger.debug("Fundamental scoring failed for %s: %s", row.get("Ticker"), e)

    # Sync the display column to the correctly-computed per-stock score
    if "Fundamental_S" in ctx.results.columns:
        ctx.results["Fundamental_Score"] = ctx.results["Fundamental_S"]

    # Card-compatible aliases for fundamental raw metrics
    # The UI cards expect DE_f, Price_STD, Price_Sources_Count, etc.
    try:
        if "debt_equity" in ctx.results.columns and "DE_f" not in ctx.results.columns:
            ctx.results["DE_f"] = pd.to_numeric(ctx.results["debt_equity"], errors="coerce")
        elif "Debt_to_Equity" in ctx.results.columns and "DE_f" not in ctx.results.columns:
            ctx.results["DE_f"] = pd.to_numeric(ctx.results["Debt_to_Equity"], errors="coerce")
        if "price_std" in ctx.results.columns and "Price_STD" not in ctx.results.columns:
            ctx.results["Price_STD"] = pd.to_numeric(ctx.results["price_std"], errors="coerce")
        if "price_mean" in ctx.results.columns and "Price_Mean" not in ctx.results.columns:
            ctx.results["Price_Mean"] = pd.to_numeric(ctx.results["price_mean"], errors="coerce")
    except Exception as e:
        logger.debug("Card alias mapping skipped: %s", e)

    # Sector from fundamentals
    if "sector" in ctx.results.columns:
        ctx.results["Sector"] = ctx.results["sector"].fillna("Unknown")
    elif "Sector" not in ctx.results.columns:
        ctx.results["Sector"] = "Unknown"

    # Sector mapping fallback
    _apply_sector_mapping(ctx)

    # NOTE: FinalScore_20d is computed ONCE after reliability recomputation below.
    # Do NOT add a compute_final_score_20d() call here — it was removed to
    # eliminate duplication (the result was immediately overwritten anyway).

    # Source metadata
    try:
        if "sources_used" in ctx.results.columns:
            ctx.results["fund_sources_used_v2"] = ctx.results["sources_used"].apply(
                lambda x: len(x) if isinstance(x, list) else 0,
            )
            ctx.results["Fundamental_Sources_Count"] = ctx.results[
                "fund_sources_used_v2"
            ]
        if "price_sources" in ctx.results.columns:
            ctx.results["price_sources_used_v2"] = (
                ctx.results["price_sources"].fillna(0).astype(int)
            )
            ctx.results["Price_Sources_Count"] = ctx.results["price_sources_used_v2"]
    except Exception as e:
        logger.debug("Source metadata mapping skipped due to error: %s", e)

    # ── Reliability recomputation (2026-02-15) ──────────────────────────
    # Reliability was originally computed in Tier 2 (ticker_scoring) BEFORE
    # fundamentals were available, so Fundamental_Sources_Count was always 0
    # and ReliabilityScore was capped at ~40. Now that we have real
    # fundamental data merged, recompute reliability with actual source counts.
    # This also unlocks the full ML boost (reliability >= 60 → full ±10).
    try:
        def _recompute_reliability(row):
            """Recompute reliability with post-enrichment data."""
            fund_sources = row.get("Fundamental_Sources_Count", 0)
            if pd.isna(fund_sources):
                fund_sources = 0
            fund_sources = int(fund_sources)
            # Base from fundamental sources: 20 (floor) + up to 60 (4 sources × 15)
            fund_score = min(fund_sources, 4) * 15 + 20

            price_bonus = 0
            if pd.notna(row.get("Price_Yahoo", row.get("Close"))):
                price_bonus += 10
            if pd.notna(row.get("ATR")):
                price_bonus += 5
            if pd.notna(row.get("RSI")):
                price_bonus += 5

            # Bonus for having actual fundamental data (not just default 50)
            fund_data_bonus = 0
            fund_s = row.get("Fundamental_S", 50.0)
            has_real_fund = any(
                pd.notna(row.get(f))
                for f in ["pe", "roe", "pb", "margin", "debt_equity",
                           "PE_Ratio", "ROE", "PB_Ratio", "Debt_to_Equity"]
            )
            if has_real_fund and fund_s != 50.0:
                fund_data_bonus = 10  # Real fundamental data available

            return min(fund_score + price_bonus + fund_data_bonus, 100)

        ctx.results["ReliabilityScore"] = ctx.results.apply(
            _recompute_reliability, axis=1
        )
        # Sync aliases
        ctx.results["Reliability_Score"] = ctx.results["ReliabilityScore"]

        logger.info(
            "[PIPELINE] Reliability recomputed post-enrichment: "
            "mean=%.1f, min=%.0f, max=%.0f",
            ctx.results["ReliabilityScore"].mean(),
            ctx.results["ReliabilityScore"].min(),
            ctx.results["ReliabilityScore"].max(),
        )
    except Exception as e:
        logger.warning("Reliability recomputation failed: %s", e)

    # ── Compute FinalScore_20d (SINGLE authoritative computation) ────────
    # All enrichment is complete: fundamentals, reliability, ML inference.
    # This is the ONE AND ONLY place FinalScore_20d is set from components.
    try:
        import json as _bd_json
        for idx, row in ctx.results.iterrows():
            try:
                new_score, breakdown = compute_final_score_20d(row, return_breakdown=True)
                ctx.results.at[idx, "FinalScore_20d"] = float(new_score)
                ctx.results.at[idx, "Score"] = float(new_score)
                ctx.results.at[idx, "ScoreBreakdown"] = _bd_json.dumps(breakdown, default=str)
            except Exception as e:
                logger.debug(
                    "FinalScore_20d computation failed for %s: %s",
                    row.get("Ticker"), e,
                )
        # Log breakdown for top 5 stocks for observability
        if not ctx.results.empty:
            _top5 = ctx.results.nlargest(5, "FinalScore_20d")
            for _, _r in _top5.iterrows():
                logger.info(
                    "[SCORE] %s = %.1f | %s",
                    _r.get("Ticker"), _r.get("FinalScore_20d", 0),
                    _r.get("ScoreBreakdown", "{}"),
                )
        logger.info(
            "[PIPELINE] FinalScore_20d computed (single pass) for %d stocks",
            len(ctx.results),
        )
    except Exception as e:
        logger.warning("FinalScore_20d computation skipped: %s", e)


def _apply_sector_mapping(ctx: _PipelineContext) -> None:
    """Apply sector_mapping fallback for unknown sectors."""
    try:
        for idx, row in ctx.results.iterrows():
            ticker = row.get("Ticker", "")
            mapped_sector = get_stock_sector(ticker)
            if mapped_sector != "Unknown":
                ctx.results.at[idx, "Sector"] = mapped_sector
    except Exception as e:
        logger.debug("Sector mapping fallback failed: %s", e)


def _phase_finalize(ctx: _PipelineContext) -> Dict[str, Any]:
    """Phases 15-22: classification, earnings, RR, signal filter, persist, meta.

    Returns the final pipeline output dict.
    """
    # Classification & Allocation
    if ctx.status_callback:
        ctx.status_callback("Classifying & Allocating...")
    ctx.results = apply_classification(ctx.results)

    # Earnings blackout
    if ctx.config.get("EARNINGS_BLACKOUT_DAYS", 0) > 0:
        topk = int(ctx.config.get("EARNINGS_CHECK_TOPK", 30))
        blackout_days = int(ctx.config.get("EARNINGS_BLACKOUT_DAYS", 7))
        if ctx.status_callback:
            ctx.status_callback(f"Checking earnings blackout (top {topk})...")
        try:
            top_indices = ctx.results.nlargest(topk, "Score").index
            for idx in top_indices:
                ticker = ctx.results.at[idx, "Ticker"]
                if check_earnings_blackout(ticker, blackout_days):
                    logger.info(
                        "[EARNINGS] %s has earnings within %d days - reducing allocation",
                        ticker,
                        blackout_days,
                    )
                    if "buy_amount_v2" in ctx.results.columns:
                        ctx.results.at[idx, "buy_amount_v2"] *= 0.5
        except Exception as e:
            logger.warning("Earnings blackout check failed: %s", e)

    # Allocation-Free Signal Engine — budget allocation disabled (signal-only mode)

    # Score = FinalScore_20d (strict)
    if "FinalScore_20d" in ctx.results.columns:
        ctx.results["Score"] = ctx.results["FinalScore_20d"]
        logger.info(
            "[PIPELINE] Final check: Score column set to FinalScore_20d for all %d results",
            len(ctx.results),
        )

    # Tier debug columns
    try:
        ctx.results["Tier1_Passed"] = True
        ctx.results["Tier1_Reasons"] = ""
        t2_vals = ctx.results.apply(
            lambda row: _t2_pass_and_reasons(row, ctx.diagnostics),
            axis=1,
            result_type="expand",
        )
        if isinstance(t2_vals, pd.DataFrame) and t2_vals.shape[1] == 2:
            ctx.results["Tier2_Passed"] = t2_vals.iloc[:, 0]
            ctx.results["Tier2_Reasons"] = t2_vals.iloc[:, 1]
        else:
            ctx.results["Tier2_Passed"] = True
            ctx.results["Tier2_Reasons"] = ""
    except Exception as e:
        logger.debug("Tier debug columns setup skipped: %s", e)

    # Dynamic RR
    try:
        # Determine market regime for regime-aware target calculation
        _rr_regime = "neutral"
        _NUMERIC_REGIME_MAP = {1.0: "bullish", 0.0: "neutral", -1.0: "bearish"}
        _WYCKOFF_TO_ATR = {
            "trend_up": "bullish", "moderate_up": "bullish",
            "sideways": "neutral",
            "distribution": "bearish", "correction": "bearish", "panic": "bearish",
        }
        try:
            if "Market_Regime" in ctx.results.columns and not ctx.results.empty:
                raw_regime = ctx.results["Market_Regime"].mode().iloc[0]
                if isinstance(raw_regime, (int, float)):
                    _rr_regime = _NUMERIC_REGIME_MAP.get(float(raw_regime), "neutral")
                else:
                    _lower = str(raw_regime).lower()
                    _rr_regime = _WYCKOFF_TO_ATR.get(_lower, _lower)
        except Exception:
            pass
        rr_updates = ctx.results.apply(
            lambda row: _compute_rr_for_row(row, ctx.data_map, market_regime=_rr_regime),
            axis=1,
            result_type="expand",
        )
        for col in [
            "Entry_Price",
            "Target_Price",
            "Stop_Loss",
            "RewardRisk",
            "RR_Ratio",
            "RR",
            "Target_Source",
            "Stop_Source",
            "Volume_UpDown_Ratio",
        ]:
            if col in rr_updates.columns:
                ctx.results[col] = rr_updates[col]

        # ── Distribution volume penalty ─────────────────────────────────
        # Penalize stocks in distribution where up-day volume is weak relative
        # to down-day volume (rallies lack conviction).
        try:
            from core.scoring_config import DISTRIBUTION_VOLUME_PENALTY as _dvp_cfg
            # Check regime using raw Wyckoff phase (not ATR-mapped name)
            _dvp_raw_regime = ""
            try:
                if "Market_Regime" in ctx.results.columns and not ctx.results.empty:
                    _dvp_raw_regime = str(ctx.results["Market_Regime"].mode().iloc[0]).lower()
            except Exception:
                pass
            if (
                _dvp_cfg.get("enabled", False)
                and "Volume_UpDown_Ratio" in ctx.results.columns
                and _dvp_raw_regime in [r.lower() for r in _dvp_cfg.get("regimes", [])]
            ):
                _vol_thresh = float(_dvp_cfg.get("up_down_volume_ratio_threshold", 0.85))
                _vol_penalty = float(_dvp_cfg.get("penalty_points", 3.0))
                _sc_col = "FinalScore_20d" if "FinalScore_20d" in ctx.results.columns else "Score"
                for _vi in ctx.results.index:
                    _vr = ctx.results.at[_vi, "Volume_UpDown_Ratio"]
                    if isinstance(_vr, (int, float)) and np.isfinite(_vr) and _vr < _vol_thresh:
                        if _sc_col in ctx.results.columns:
                            _old_sc = float(ctx.results.at[_vi, _sc_col])
                            ctx.results.at[_vi, _sc_col] = max(0.0, _old_sc - _vol_penalty)
                            ctx.results.at[_vi, "Volume_Penalty"] = _vol_penalty
                            logger.info(
                                "[VOLUME] %s: weak rally volume (ratio=%.2f < %.2f) → -%.0f pts",
                                ctx.results.at[_vi, "Ticker"] if "Ticker" in ctx.results.columns else _vi,
                                _vr, _vol_thresh, _vol_penalty,
                            )
        except Exception as _dvp_exc:
            logger.debug("Distribution volume penalty skipped: %s", _dvp_exc)

        # NOTE: Multiplicative RR gate REMOVED (2026-02-27) to eliminate double-gating.
        # RR is already included as 20% of the conviction score inside
        # compute_final_score_20d() via evaluate_rr_unified(). Applying a second
        # multiplicative gate on top was double-penalizing poor RR and double-
        # rewarding good RR. The RR component in the scoring function is sufficient.
        if "FinalScore_20d" in ctx.results.columns:
            ctx.results["FinalScore"] = ctx.results["FinalScore_20d"]
            ctx.results["Score"] = ctx.results["FinalScore_20d"]

        # ── Post-RR safety re-check ──────────────────────────────────────
        # Classification ran BEFORE dynamic RR (which may overwrite RR_Ratio).
        # If the early RR was NaN the safety filter skipped the check.  Now
        # that we have the final RR values, re-apply the hard R:R minimum
        # and update SafetyBlocked accordingly.
        try:
            from core.scoring_config import HARD_FILTERS as _hf_rr, get_vix_min_rr
            # Extract VIX from the global index cache (already fetched by market_regime)
            _vix_val = None
            try:
                from core.market_context import _GLOBAL_INDEX_CACHE
                _vix_df = _GLOBAL_INDEX_CACHE.get("^VIX")
                if _vix_df is not None and not _vix_df.empty:
                    _vix_col = "close" if "close" in _vix_df.columns else ("Close" if "Close" in _vix_df.columns else _vix_df.columns[0])
                    _vix_val = float(_vix_df[_vix_col].iloc[-1])
            except Exception:
                pass
            # Also try regime data
            if _vix_val is None:
                try:
                    _vix_val = ctx.market_regime_data.get("vix_value") if hasattr(ctx, "market_regime_data") and ctx.market_regime_data else None
                except Exception:
                    pass
            _min_rr_base = float(_hf_rr.get("min_rr", 0.0))
            _min_rr = get_vix_min_rr(_vix_val) if _vix_val is not None else _min_rr_base
            # Regime R:R floor: in adverse regimes, override VIX-only threshold
            try:
                from core.scoring_config import REGIME_RR_FLOOR
                # Recompute raw Wyckoff regime from results (don't rely on
                # raw_regime variable which may be out of scope)
                _regime_upper = ""
                try:
                    if "Market_Regime" in ctx.results.columns and not ctx.results.empty:
                        _regime_upper = str(ctx.results["Market_Regime"].mode().iloc[0]).upper()
                except Exception:
                    pass
                _regime_floor = float(REGIME_RR_FLOOR.get(_regime_upper, 0.0))
                if _regime_floor > _min_rr:
                    logger.info(
                        "[PIPELINE] Regime R:R floor: %s → %.1f (overrides VIX-based %.2f)",
                        _regime_upper, _regime_floor, _min_rr,
                    )
                    _min_rr = _regime_floor
            except Exception as _rf_exc:
                logger.debug("Regime R:R floor skipped: %s", _rf_exc)
            logger.info("[PIPELINE] Effective min R:R = %.2f (VIX=%.1f, regime=%s)", _min_rr, _vix_val or 0.0, _rr_regime)
            # Store VIX observability columns
            if not ctx.results.empty:
                ctx.results["VIX_Value"] = _vix_val if _vix_val is not None else np.nan
                ctx.results["VIX_Min_RR"] = _min_rr
            if _min_rr > 0 and not ctx.results.empty:
                for _idx in ctx.results.index:
                    _rr_val = None
                    for _rr_col in ("RR", "RR_Ratio", "RewardRisk"):
                        _v = ctx.results.at[_idx, _rr_col] if _rr_col in ctx.results.columns else None
                        if _v is not None and isinstance(_v, (int, float)) and np.isfinite(_v):
                            _rr_val = float(_v)
                            break
                    if _rr_val is not None and _rr_val < _min_rr:
                        if "SafetyBlocked" in ctx.results.columns:
                            ctx.results.at[_idx, "SafetyBlocked"] = True
                        if "RiskClass" in ctx.results.columns:
                            ctx.results.at[_idx, "RiskClass"] = "REJECT"
                        logger.info(
                            "[PIPELINE] Post-RR filter: %s blocked (RR=%.2f < %.1f)",
                            ctx.results.at[_idx, "Ticker"] if "Ticker" in ctx.results.columns else _idx,
                            _rr_val,
                            _min_rr,
                        )
        except Exception as _rr_exc:
            logger.debug("Post-RR safety re-check skipped: %s", _rr_exc)

    except Exception as e:
        logger.warning("[PIPELINE] Dynamic RR computation failed: %s", e)

    # ── Per-stock Holding_Days and Target_Date ────────────────────────────
    # Compute individual holding periods based on ATR/volatility.
    # Low-vol stocks get longer holding periods; high-vol stocks get shorter.
    try:
        if "ATR_Pct" in ctx.results.columns and not ctx.results.empty:
            _atr_vals = ctx.results["ATR_Pct"].dropna()
            _median_atr = float(_atr_vals.median()) if len(_atr_vals) > 0 else 0.025

            def _dynamic_holding(row):
                atr = row.get("ATR_Pct", _median_atr)
                if not isinstance(atr, (int, float)) or pd.isna(atr) or atr <= 0:
                    atr = _median_atr
                # Low vol (< 2%) → 25 days; Med (2-4%) → 20; High (> 4%) → 12
                if atr < 0.015:
                    return 28
                if atr < 0.025:
                    return 22
                if atr < 0.04:
                    return 18
                return 12

            ctx.results["Holding_Days"] = ctx.results.apply(_dynamic_holding, axis=1)
        elif "Holding_Days" not in ctx.results.columns:
            ctx.results["Holding_Days"] = 20

        # Compute per-stock Target_Date from As_Of_Date + Holding_Days
        if "Holding_Days" in ctx.results.columns:
            _now = pd.Timestamp.now()

            def _target_dt(row):
                try:
                    hd = int(row.get("Holding_Days", 20))
                    as_of = row.get("As_Of_Date", None)
                    if as_of is not None and not pd.isna(as_of):
                        base = pd.Timestamp(as_of)
                    else:
                        base = _now
                    return base + pd.offsets.BDay(hd)
                except Exception:
                    return pd.NaT

            ctx.results["Target_Date"] = ctx.results.apply(_target_dt, axis=1)
    except Exception as e:
        logger.debug("Holding_Days/Target_Date computation: %s", e)

    # ── Analyst consensus cross-check ────────────────────────────────
    # Fetch analyst price targets from FMP (already implemented in sentiment_data.py
    # but never wired into the pipeline). Penalize stocks where the system target
    # significantly exceeds analyst consensus or where analysts are bearish.
    try:
        from core.scoring_config import ANALYST_TARGET_PENALTY as _apt_cfg
        if _apt_cfg.get("enabled", False) and not ctx.results.empty:
            from core.sentiment_data import fetch_analyst_ratings_fmp
            _score_col = "FinalScore_20d" if "FinalScore_20d" in ctx.results.columns else "Score"
            for _a_idx in ctx.results.index:
                _a_ticker = str(ctx.results.at[_a_idx, "Ticker"]) if "Ticker" in ctx.results.columns else ""
                if not _a_ticker:
                    continue
                try:
                    _analyst = fetch_analyst_ratings_fmp(_a_ticker)
                    _pt_upside = _analyst.get("price_target_upside", None)
                    ctx.results.at[_a_idx, "Price_Target_Upside"] = (
                        float(_pt_upside) if _pt_upside is not None else np.nan
                    )
                    ctx.results.at[_a_idx, "Analyst_Count"] = float(_analyst.get("analyst_count", 0))
                    # Apply score penalty for significant divergence
                    if _pt_upside is not None and np.isfinite(float(_pt_upside)):
                        _pt_upside_f = float(_pt_upside)
                        _current_score = ctx.results.at[_a_idx, _score_col] if _score_col in ctx.results.columns else 0
                        _penalty = 0.0
                        if _pt_upside_f < 0:
                            # Analysts think stock is overvalued (PT < current price)
                            _penalty = float(_apt_cfg.get("negative_upside_penalty", 8.0))
                            logger.info(
                                "[ANALYST] %s: negative PT upside (%.1f%%) → -%.0f pts",
                                _a_ticker, _pt_upside_f * 100, _penalty,
                            )
                        else:
                            # Check if system target overshoots analyst target
                            _entry = ctx.results.at[_a_idx, "Entry_Price"] if "Entry_Price" in ctx.results.columns else np.nan
                            _target = ctx.results.at[_a_idx, "Target_Price"] if "Target_Price" in ctx.results.columns else np.nan
                            if np.isfinite(_entry) and np.isfinite(_target) and _entry > 0:
                                _sys_upside = (_target - _entry) / _entry
                                _thresh = float(_apt_cfg.get("overestimate_threshold", 0.20))
                                if _sys_upside > _pt_upside_f + _thresh:
                                    _penalty = float(_apt_cfg.get("penalty_points", 5.0))
                                    logger.info(
                                        "[ANALYST] %s: system upside %.1f%% >> analyst %.1f%% → -%.0f pts",
                                        _a_ticker, _sys_upside * 100, _pt_upside_f * 100, _penalty,
                                    )
                        if _penalty > 0 and _score_col in ctx.results.columns:
                            _new_score = max(0.0, float(_current_score) - _penalty)
                            ctx.results.at[_a_idx, _score_col] = _new_score
                            ctx.results.at[_a_idx, "Analyst_Penalty"] = _penalty
                            if "FinalScore" in ctx.results.columns:
                                ctx.results.at[_a_idx, "FinalScore"] = _new_score
                            if "Score" in ctx.results.columns:
                                ctx.results.at[_a_idx, "Score"] = _new_score
                except Exception as _ae:
                    logger.debug("Analyst data for %s: %s", _a_ticker, _ae)
    except Exception as _apt_exc:
        logger.debug("Analyst consensus cross-check skipped: %s", _apt_exc)

    # No-Trade Signal: warn when market regime is unfavorable
    try:
        if "Market_Regime" in ctx.results.columns and not ctx.results.empty:
            _regime_val = str(ctx.results["Market_Regime"].mode().iloc[0]).upper()
            _caution_regimes = {"DISTRIBUTION", "CORRECTION", "PANIC", "BEARISH"}
            if _regime_val in _caution_regimes:
                ctx.results["No_Trade_Signal"] = True
                ctx.results["Market_Caution"] = (
                    f"CAUTION: Market regime is {_regime_val}. "
                    "Historically unfavorable for new positions. "
                    "Consider reduced position sizes or waiting for regime change."
                )
                logger.warning(
                    "[NO-TRADE SIGNAL] Market regime %s — flagging all %d recommendations with caution",
                    _regime_val, len(ctx.results),
                )
            else:
                ctx.results["No_Trade_Signal"] = False
                ctx.results["Market_Caution"] = ""
    except Exception as e:
        logger.debug("No-trade signal computation: %s", e)

    # Signal-first filtering & ranking
    try:
        orig_len = len(ctx.results)
        score_col = (
            "FinalScore_20d"
            if "FinalScore_20d" in ctx.results.columns
            else ("Score" if "Score" in ctx.results.columns else None)
        )
        if score_col is not None and not ctx.results.empty:
            sc = pd.to_numeric(ctx.results[score_col], errors="coerce")
            mlp = pd.to_numeric(
                ctx.results.get(
                    "ML_20d_Prob", pd.Series(index=ctx.results.index)
                ),
                errors="coerce",
            )
            patt = pd.to_numeric(
                ctx.results.get(
                    "Pattern_Score", pd.Series(index=ctx.results.index)
                ),
                errors="coerce",
            )
            # Respect safety filters: exclude stocks blocked by classification
            safety_ok = ~ctx.results.get(
                "SafetyBlocked", pd.Series(False, index=ctx.results.index)
            ).astype(bool)

            # Determine regime-aware minimum score threshold
            _regime_for_threshold = "SIDEWAYS"  # safe default
            if "Market_Regime" in ctx.results.columns and not ctx.results.empty:
                try:
                    _regime_for_threshold = str(
                        ctx.results["Market_Regime"].mode().iloc[0]
                    ).upper()
                except Exception:
                    pass
            effective_min_score = float(
                REGIME_MIN_SCORE.get(_regime_for_threshold, SIGNAL_MIN_SCORE)
            )
            logger.info(
                "[PIPELINE] Regime=%s → effective min score=%.1f (base=%.1f)",
                _regime_for_threshold,
                effective_min_score,
                SIGNAL_MIN_SCORE,
            )

            # In strict regimes (DISTRIBUTION+), disable ML/pattern bypasses —
            # stocks MUST meet the regime score threshold.  Without this guard,
            # ML_prob >= 0.62 or any Pattern_Score > 0 would let every stock
            # through, making REGIME_MIN_SCORE ineffective.
            if effective_min_score >= BYPASS_DISABLED_ABOVE_MIN_SCORE:
                mask = safety_ok & (sc >= effective_min_score)
                logger.info(
                    "[PIPELINE] Strict regime (%.0f >= %.0f): ML/pattern bypasses DISABLED",
                    effective_min_score, BYPASS_DISABLED_ABOVE_MIN_SCORE,
                )
            else:
                mask = (
                    safety_ok
                    & (
                        (sc >= effective_min_score)
                        | (mlp >= float(ML_PROB_THRESHOLD))
                        | ((patt.fillna(0.0) > 0.0) & (sc >= float(PATTERN_MIN_SCORE)))
                    )
                )
            filtered = (
                ctx.results[mask].copy() if isinstance(mask, pd.Series) else ctx.results.copy()
            )

            sort_cols = ["_score_numeric"]
            asc = [False]
            if "SignalReasons_Count" in filtered.columns:
                sort_cols.append("SignalReasons_Count")
                asc.append(False)
            sort_cols.append("ML_20d_Prob")
            asc.append(False)
            filtered = (
                filtered.assign(
                    _score_numeric=pd.to_numeric(filtered[score_col], errors="coerce")
                )
                .sort_values(by=sort_cols, ascending=asc)
                .drop(columns=["_score_numeric"])
            )

            topn = int(ctx.config.get("topn_results", TOP_SIGNAL_K))
            if filtered.empty:
                # In strict regimes (DISTRIBUTION+), do NOT fall back — show empty
                if effective_min_score >= 75.0:
                    ctx.results = pd.DataFrame(columns=ctx.results.columns)
                    logger.info(
                        "[PIPELINE] No candidates meet regime-aware threshold (%.0f) "
                        "in %s market — returning empty (no recommendations)",
                        effective_min_score,
                        _regime_for_threshold,
                    )
                else:
                    sort_cols_fb = ["_score_numeric"]
                    asc_fb = [False]
                    if "SignalReasons_Count" in ctx.results.columns:
                        sort_cols_fb.append("SignalReasons_Count")
                        asc_fb.append(False)
                    sort_cols_fb.append("ML_20d_Prob")
                    asc_fb.append(False)
                    # Fallback also respects safety filters
                    safe_results = ctx.results[safety_ok] if safety_ok.any() else ctx.results
                    fallback = (
                        safe_results.assign(
                            _score_numeric=pd.to_numeric(
                                safe_results[score_col], errors="coerce"
                            )
                        )
                        .sort_values(by=sort_cols_fb, ascending=asc_fb)
                        .drop(columns=["_score_numeric"])
                    )
                    ctx.results = fallback.head(topn).reset_index(drop=True)
                    logger.info(
                        "[PIPELINE] Signal thresholds yielded no candidates; "
                        "using top-%d by score as fallback",
                        topn,
                    )
            else:
                ctx.results = filtered.head(topn).reset_index(drop=True)
                logger.info(
                    "[PIPELINE] Signal-First ranking applied: kept %d of %d; "
                    "regime=%s, threshold=%.0f, ml>=%s",
                    len(ctx.results),
                    orig_len,
                    _regime_for_threshold,
                    effective_min_score,
                    ML_PROB_THRESHOLD,
                )
            ctx.postfilter_mode = "signal_only"
        else:
            logger.info("[PIPELINE] No score column for signal filter; keeping all")
    except Exception as e:
        logger.warning("[PIPELINE] Signal-first filtering failed: %s", e)

    # Persist latest results
    try:
        to_save = ctx.results.copy()
        # Drop non-serializable object columns (e.g. dataclass instances, dicts,
        # and complex structs) that cause Arrow/Parquet conversion failures
        obj_cols_to_drop = []
        for col in to_save.columns:
            if to_save[col].dtype == object:
                sample = to_save[col].dropna().head(1)
                if sample.empty:
                    continue
                val = sample.iloc[0]
                # Drop: dataclass instances, dicts (Parquet struct issues), other non-primitives
                if isinstance(val, dict):
                    obj_cols_to_drop.append(col)
                elif not isinstance(val, (str, list, int, float, bool)):
                    obj_cols_to_drop.append(col)
        # Also drop known problematic columns
        for known_bad in ["coverage", "FundamentalBreakdown", "prices_by_source"]:
            if known_bad in to_save.columns and known_bad not in obj_cols_to_drop:
                obj_cols_to_drop.append(known_bad)
        if obj_cols_to_drop:
            logger.info("[PIPELINE] Dropping non-serializable columns for Parquet: %s", obj_cols_to_drop)
            to_save = to_save.drop(columns=obj_cols_to_drop, errors="ignore")
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        to_save.to_json(
            data_dir / "latest_scan_live.json", orient="records", date_format="iso"
        )
        to_save.to_parquet(data_dir / "latest_scan_live.parquet", index=False)
        # Also persist to data/scans/ so the Streamlit app and CI find the same files
        scans_dir = data_dir / "scans"
        scans_dir.mkdir(parents=True, exist_ok=True)
        to_save.to_parquet(scans_dir / "latest_scan_live.parquet", index=False)
        to_save.to_json(
            scans_dir / "latest_scan_live.json", orient="records", date_format="iso"
        )
        # Timestamped backup for scan history
        from datetime import datetime as _dt2
        _ts_str = _dt2.utcnow().strftime("%Y%m%d_%H%M")
        to_save.to_parquet(scans_dir / f"scan_live_{_ts_str}.parquet", index=False)
        # Save scan metadata alongside results
        import json as _json
        from datetime import datetime as _dt
        _regime = "unknown"
        if "Market_Regime" in to_save.columns and not to_save.empty:
            try:
                _regime = str(to_save["Market_Regime"].mode().iloc[0])
            except Exception:
                pass
        _scan_meta = {
            "results_count": len(to_save),
            "timestamp": _dt.utcnow().isoformat(),
            "scoring_mode": ctx.postfilter_mode if hasattr(ctx, "postfilter_mode") else "signal_only",
            "ml_enabled": "ML_20d_Prob" in to_save.columns,
            "fundamental_enabled": "Fundamental_S" in to_save.columns,
            "market_regime": _regime,
            "universe_size": ctx.config.get("universe_size", len(to_save)),
            "features_used": len([c for c in to_save.columns if c.startswith(("ML_", "VCP", "RS_"))]),
            "ml_model_version": ctx.config.get("ml_model_version", "v3.1"),
            "total_tickers": len(to_save),
        }
        (scans_dir / "latest_scan_live.meta.json").write_text(
            _json.dumps(_scan_meta, indent=2)
        )
        logger.info(
            "\u2705 Pipeline Finalized: Saved Top %d recommendations to data/ and data/scans/",
            len(to_save),
        )
    except Exception as e:
        logger.warning("[PIPELINE] Failed to persist latest scan files: %s", e)

    # Persist to DuckDB + register for outcome tracking
    try:
        from core.db.store import get_scan_store
        from core.db.outcome_tracker import OutcomeTracker

        store = get_scan_store()
        scan_id = store.generate_scan_id()
        scan_meta = {
            "universe_name": ctx.config.get("universe_name", "unknown"),
            "universe_size": ctx.config.get("universe_size", len(ctx.results)),
            "market_regime": ctx.results["Market_Regime"].mode().iloc[0]
            if "Market_Regime" in ctx.results.columns and not ctx.results.empty
            else None,
            "total_scored": ctx.config.get("total_scored"),
            "ml_model_version": ctx.config.get("ml_model_version"),
        }
        n_saved = store.save_scan(scan_id, ctx.results, ctx.config, scan_meta)
        tracker = OutcomeTracker(store)
        n_registered = tracker.register_recommendations(scan_id, ctx.results)
        logger.info(
            "[PIPELINE] DuckDB: saved %d recs, registered %d for outcome tracking (scan %s)",
            n_saved, n_registered, scan_id,
        )
    except Exception as e:
        logger.warning("[PIPELINE] DuckDB persistence skipped: %s", e)

    # Attach Tier 1 filtered summary
    try:
        if isinstance(ctx.data_map, dict):
            ctx.data_map = dict(ctx.data_map)
            ctx.data_map["filtered_tier1_df"] = ctx.filtered_df
    except (TypeError, KeyError):
        pass

    meta = _build_pipeline_meta(ctx)
    return {
        "result": {
            "results_df": ctx.results,
            "data_map": ctx.data_map,
            "diagnostics": ctx.diagnostics,
        },
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Public scan entry points
# ---------------------------------------------------------------------------

def run_scan_pipeline(
    universe: List[str],
    config: Dict[str, Any],
    status_callback: Optional[Callable[[str], None]] = None,
    data_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Unified pipeline runner for Stock Scout.

    **Pipeline flow:**
    1. Fetch historical data (yfinance bulk download)
    2. Compute technical indicators for each ticker
    3. Apply basic technical filters (volume, liquidity)
    4. **Call compute_recommendation_scores (unified scoring) for each ticker**
    5. Apply beta filter (optional, top-K only)
    6. Apply advanced filters (RS, momentum, RR penalties)
    7. Enrich with fundamentals & sector data
    8. Run classification & allocation
    9. Check earnings blackout (optional, top-K only)

    Returns:
        Dict with ``result`` (results_df, data_map, diagnostics) and ``meta``.
    """
    try:
        logger.info(
            "\U0001f30c Starting pipeline with universe size: %d tickers",
            len(universe),
        )
    except (TypeError, ValueError):
        pass

    ctx = _PipelineContext(
        config=config,
        universe=list(universe),
        status_callback=status_callback,
        data_map=dict(data_map) if isinstance(data_map, dict) else {},
    )

    # Phase 1 — init, market context, preflight, gating
    early = _phase_init_context(ctx)
    if early is not None:
        return early

    # Phase 2 — fetch data, RS ranking, Tier 1 OHLCV filter
    _phase_fetch_and_tier1(ctx)

    # Phase 3 — Tier 2 scoring, beta, advanced filters, meteor
    early = _phase_score_and_filter(ctx)
    if early is not None:
        return early

    # Phase 4 — fundamentals, sector mapping, score recalc
    _phase_enrich_fundamentals(ctx)

    # Phase 5 — classification, RR, signal filter, persist, meta
    return _phase_finalize(ctx)


def run_scan(*args, **kwargs):
    """Backward-compatible wrapper that returns a ``ScanResult`` contract.

    Delegates to ``run_scan_pipeline`` and converts the result dict into
    strongly-typed contract objects.
    """
    out = run_scan_pipeline(*args, **kwargs)
    # If pipeline returns wrapper, convert to contracts for backward consumers
    if isinstance(out, dict) and ("result" in out):
        payload = out.get("result")
        if isinstance(payload, dict) and ("results_df" in payload):
            _results_df = payload.get("results_df")
            results_df = _results_df if _results_df is not None else pd.DataFrame()
            data_map = payload.get("data_map") or {}
        else:
            try:
                results_df, data_map = payload
            except (TypeError, ValueError):
                results_df = (
                    payload if isinstance(payload, pd.DataFrame) else pd.DataFrame()
                )
                data_map = {}
    elif isinstance(out, ScanResult):
        return out
    else:
        return out

    try:
        now = datetime.utcnow()
    except (TypeError, ValueError):
        now = None
    metadata = ScanMetadata(
        scan_id=f"scan-{int(time.time())}",
        logic_version="pipeline_v2",
        timestamp=now,
        data_provider_stats={},
        warnings=[],
    )
    diagnostics = Diagnostics(
        coverage=CoverageMetrics(
            universe_size=int(
                len(results_df) if hasattr(results_df, "__len__") else 0
            ),
            fetched_history_count=int(
                len(data_map) if isinstance(data_map, dict) else 0
            ),
            fundamentals_count=int(
                results_df.get("Fundamental_S", pd.Series()).notna().sum()
            )
            if isinstance(results_df, pd.DataFrame)
            else 0,
            external_price_verified_count=0,
            earnings_known_count=0,
        ),
        fallbacks=FallbackCounts(
            alpha_to_finnhub=0, polygon_to_tiingo=0, other_fallbacks={}
        ),
        disagreement=DisagreementStats(
            price_std_mean=None, high_disagreement_count=0, agreement_pct=None
        ),
        ml_status=MLStatus(
            enabled=ML_20D_AVAILABLE,
            model_name=None,
            model_version=None,
            loaded=True,
            inference_ok=True,
            inference_count=int(len(results_df))
            if isinstance(results_df, pd.DataFrame)
            else 0,
            last_trained_ts=None,
            metrics=None,
        ),
        market_context_status=MarketContextStatus(
            ok=True,
            regime=str(results_df.get("Market_Regime").iloc[0])
            if isinstance(results_df, pd.DataFrame)
            and "Market_Regime" in results_df.columns
            and len(results_df) > 0
            else None,
            computed_at=now,
            vix_level=None,
            beta_mean=None,
            spx_trend=None,
            note=None,
        ),
    )

    recs: List[Recommendation] = []
    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        for _, row in results_df.iterrows():
            ticker = (
                str(row.get("Ticker")) if row.get("Ticker") is not None else None
            )
            final_score = float(row.get("FinalScore_20d", row.get("Score", 0.0)))
            rr_val = row.get("RR", row.get("RR_Ratio"))
            beta_val = row.get("Beta")
            targets = Targets(
                entry=_to_float(row.get("Entry_Price")),
                target_20d=_to_float(row.get("Target_Price")),
                stop_loss=_to_float(row.get("Stop_Loss")),
                take_profit=None,
            )
            pos_usd = _to_float(row.get("buy_amount_v2"))
            if pos_usd is None:
                pos_usd = 0.0
            risk_sizing = RiskSizing(
                position_size_usd=pos_usd,
                position_size_shares=None,
                max_risk_pct=None,
                atr_shares=None,
                risk_notes=None,
            )
            breakdown: Dict[str, float] = {}
            for key in [
                "ML_20d_Prob",
                "Fundamental_Score",
                "Fundamental_S",
                "TechScore_20d",
                "ReliabilityScore",
                "Reliability_Score",
                "reliability_v2",
                "Reliability_v2",
                "MomentumScore",
            ]:
                if key in results_df.columns:
                    val = row.get(key)
                    if val is not None and np.isfinite(_to_float(val) or float("nan")):
                        breakdown[key] = float(val)
            reasons: List[str] = []
            if row.get("RejectionReason"):
                try:
                    reasons.append(str(row.get("RejectionReason")))
                except (TypeError, ValueError):
                    pass
            risk_class = (
                row.get("RiskClass") if "RiskClass" in results_df.columns else None
            )
            safety_blocked = (
                row.get("SafetyBlocked")
                if "SafetyBlocked" in results_df.columns
                else None
            )
            safety_reasons = (
                row.get("SafetyReasons")
                if "SafetyReasons" in results_df.columns
                else None
            )
            rec = Recommendation(
                ticker=ticker or "",
                scores_breakdown=breakdown,
                final_score_20d=float(final_score),
                rr=_to_float(rr_val),
                beta=_to_float(beta_val),
                targets=targets,
                risk_sizing=risk_sizing,
                reasons=reasons,
                risk_class=risk_class,
                safety_blocked=bool(safety_blocked)
                if safety_blocked is not None
                else None,
                safety_reasons=str(safety_reasons)
                if safety_reasons is not None
                else None,
                risk_level=row.get("Risk_Level")
                if "Risk_Level" in results_df.columns
                else None,
                data_quality=row.get("Data_Quality")
                if "Data_Quality" in results_df.columns
                else None,
                confidence_level=row.get("Confidence_Level")
                if "Confidence_Level" in results_df.columns
                else None,
                should_display=bool(row.get("Should_Display"))
                if "Should_Display" in results_df.columns
                else None,
                consolidation_ratio=_to_float(row.get("Consolidation_Ratio"))
                if "Consolidation_Ratio" in results_df.columns
                else None,
                pocket_pivot_ratio=_to_float(row.get("Pocket_Pivot_Ratio"))
                if "Pocket_Pivot_Ratio" in results_df.columns
                else None,
                vcp_ratio=_to_float(row.get("VCP_Ratio"))
                if "VCP_Ratio" in results_df.columns
                else None,
            )
            recs.append(rec)

    return ScanResult(
        metadata=metadata, diagnostics=diagnostics, recommendations=recs
    )


def run_scan_smoke() -> Dict[str, Any]:
    """Run a tiny smoke test on a single ticker and print meta."""
    logging.basicConfig(level=logging.INFO)
    try:
        import core.pipeline.universe as _u

        _u.LAST_UNIVERSE_PROVIDER = "smoke/manual"
    except (ImportError, AttributeError):
        pass
    try:
        cfg = _normalize_config(get_config())
        cfg["fundamental_enabled"] = False
        cfg["beta_filter_enabled"] = False
        cfg["meteor_mode"] = False
        cfg["smoke_mode"] = True
    except (ImportError, TypeError, KeyError):
        cfg = {"fundamental_enabled": False, "beta_filter_enabled": False}
    out = run_scan_pipeline(["AAPL"], cfg, status_callback=None)
    try:
        meta = out.get("meta", {})
        try:
            meta["smoke_overrides"] = {"meteor_mode_forced_off": True}
        except (TypeError, KeyError):
            pass
        print("Smoke Meta:", meta)
    except (AttributeError, TypeError):
        pass
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry for production scan: 1500-2000 US stocks, strict mode."""
    logging.basicConfig(level=logging.INFO)
    preflight_check()

    logger.info("Initializing Market Context...")
    initialize_market_context()
    try:
        src = get_last_index_source("SPY") or "Unknown"
        logger.info("[SUCCESS via %s] Market context initialized", src)
    except (ImportError, AttributeError):
        logger.info("[SUCCESS] Market context initialized")

    cfg_probe = get_config()
    universe = fetch_top_us_tickers_by_market_cap(
        limit=int(getattr(cfg_probe, "universe_limit", 2000))
    )
    logger.info("Fetched %d tickers", len(universe))

    cfg = _normalize_config(get_config())
    cfg["fundamental_enabled"] = True
    cfg["beta_filter_enabled"] = False

    wrapper = run_scan_pipeline(universe, cfg)
    payload = wrapper.get("result") if isinstance(wrapper, dict) else wrapper
    if isinstance(payload, dict) and ("results_df" in payload):
        _results_df = payload.get("results_df")
        results = _results_df if _results_df is not None else pd.DataFrame()
        data_map = payload.get("data_map") or {}
    elif isinstance(payload, tuple) and len(payload) >= 1:
        results = payload[0]
        data_map = payload[1] if len(payload) > 1 else {}
    else:
        results = pd.DataFrame()
        data_map = {}
    if results.empty:
        logger.error("No results from pipeline")
        return

    filtered = results[
        (results["FinalScore_20d"] > 70.0)
        & (results["Market_Regime"].str.upper() == "TREND_UP")
    ].copy()
    filtered = filtered.sort_values("FinalScore_20d", ascending=False)
    top10 = filtered.head(10)

    output = []
    for _, row in top10.iterrows():
        tkr = str(row.get("Ticker"))
        news_items = fetch_latest_company_news(tkr, count=5)
        headlines = [
            n.get("headline") or n.get("title") or "" for n in news_items
        ]
        headlines = [h for h in headlines if h]
        sentiment = analyze_sentiment_openai(headlines)
        output.append(
            {
                "Ticker": tkr,
                "FinalScore_20d": float(row.get("FinalScore_20d", 0.0)),
                "TechScore_20d": float(row.get("TechScore_20d", 0.0)),
                "Fundamental_Score": float(
                    row.get("Fundamental_Score", row.get("Fundamental_S", 0.0))
                ),
                "ML_20d_Prob": float(row.get("ML_20d_Prob", 0.5)),
                "Market_Regime": str(row.get("Market_Regime")),
                "Sector": str(row.get("Sector", "Unknown")),
                "Sentiment": sentiment,
            }
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
