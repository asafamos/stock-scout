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
from core.scoring_config import ML_PROB_THRESHOLD, SIGNAL_MIN_SCORE, TOP_SIGNAL_K
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
        if meta.get("ml_bundle_version_warning"):
            meta["ml_mode"] = "DISABLED_VERSION_MISMATCH"
        elif not ML_20D_AVAILABLE:
            meta["ml_mode"] = "DISABLED_NO_MODEL"
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
            elif volume < 100000:
                reasons.append(
                    {
                        "rule": "VOLUME_MIN",
                        "message": "Volume below minimum",
                        "value": float(volume) if pd.notna(volume) else None,
                        "threshold": 100000,
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

    if "Score" not in ctx.results.columns and "FinalScore_20d" in ctx.results.columns:
        ctx.results["Score"] = ctx.results["FinalScore_20d"]

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
        ctx.results.at[idx, "Quality_Score"] = sig.get("quality_score")
        ctx.results.at[idx, "RR_Ratio"] = sig.get("risk_reward_ratio")
        ctx.results.at[idx, "Momentum_Consistency"] = sig.get("momentum_consistency")

        if catastrophic:
            ctx.results.at[idx, "FinalScore_20d"] = float(enhanced)
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
            penalty = 0.0
            if sig.get("rs_63d", 0) < rs_thresh:
                penalty += 1.0
            if sig.get("momentum_consistency", 0) < mom_thresh:
                penalty += 1.0
            if sig.get("risk_reward_ratio", 0) < rr_thresh:
                penalty += 1.5
            normalized_penalty = penalty / 100.0
            ctx.results.at[idx, "AdvPenalty"] = penalty
            ctx.results.at[idx, "FinalScore_20d"] = max(
                0.01, float(enhanced) - float(normalized_penalty)
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

    # Scale FinalScore_20d back to 0-100
    ctx.results["FinalScore_20d"] = ctx.results["FinalScore_20d"] * 100.0
    if "FinalScore_20d" in ctx.results.columns:
        ctx.results["Score"] = ctx.results["FinalScore_20d"]
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
                    df = tier2_map.get(tkr) or ctx.data_map.get(tkr)
                    if df is None or df.empty:
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
                        row["FinalScore_20d"] = float(new_score)
                        row["Score"] = float(new_score)
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
    try:
        score_thr = float(
            ctx.config.get(
                "fundamentals_score_threshold",
                float(os.getenv("FUNDAMENTALS_SCORE_THRESHOLD", "60")),
            )
        )
    except (TypeError, ValueError):
        score_thr = 60.0
    try:
        top_n_cap = int(
            ctx.config.get(
                "fundamentals_top_n_cap",
                int(os.getenv("FUNDAMENTALS_TOP_N_CAP", "50")),
            )
        )
    except (TypeError, ValueError):
        top_n_cap = 50
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
        }
        for src, dst in _col_map.items():
            if src in ctx.results.columns and dst not in ctx.results.columns:
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
                for f in ["pe", "roe", "pb", "margin", "debt_equity"]
            )
            if not has_fund_data:
                ctx.results.at[idx, "Fundamental_S"] = 50.0
                continue
            fund_score_obj = compute_fundamental_score_with_breakdown(fund_data)
            ctx.results.at[idx, "Fundamental_S"] = fund_score_obj.total
            ctx.results.at[idx, "Quality_Score_F"] = fund_score_obj.breakdown.quality_score
            ctx.results.at[idx, "Growth_Score_F"] = fund_score_obj.breakdown.growth_score
            ctx.results.at[idx, "Valuation_Score_F"] = fund_score_obj.breakdown.valuation_score
        except Exception as e:
            ctx.results.at[idx, "Fundamental_S"] = 50.0
            logger.debug("Fundamental scoring failed for %s: %s", row.get("Ticker"), e)

    # Sector from fundamentals
    if "sector" in ctx.results.columns:
        ctx.results["Sector"] = ctx.results["sector"].fillna("Unknown")
    elif "Sector" not in ctx.results.columns:
        ctx.results["Sector"] = "Unknown"

    # Sector mapping fallback
    _apply_sector_mapping(ctx)

    # Recalculate FinalScore_20d with fundamentals
    try:
        for idx, row in ctx.results.iterrows():
            try:
                new_score = compute_final_score_20d(row)
                ctx.results.at[idx, "FinalScore_20d"] = float(new_score)
                ctx.results.at[idx, "Score"] = float(new_score)
            except Exception as e:
                logger.debug(
                    "FinalScore recalc failed for %s: %s", row.get("Ticker"), e
                )
        logger.info(
            "[PIPELINE] Recalculated FinalScore_20d with fundamentals for %d stocks",
            len(ctx.results),
        )
    except Exception as e:
        logger.warning("FinalScore recalc skipped: %s", e)

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

    # Allocation-Free Signal Engine
    if False and "buy_amount_v2" not in ctx.results.columns:
        ctx.results = allocate_budget(
            ctx.results,
            ctx.config.get("BUDGET_TOTAL", 5000),
            ctx.config.get("MIN_POSITION", 500),
            ctx.config.get("MAX_POSITION_PCT", 0.2),
        )

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
        rr_updates = ctx.results.apply(
            lambda row: _compute_rr_for_row(row, ctx.data_map),
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
        ]:
            if col in rr_updates.columns:
                ctx.results[col] = rr_updates[col]

        if "FinalScore_20d" in ctx.results.columns and "RR" in ctx.results.columns:
            low_mask = pd.to_numeric(ctx.results["RR"], errors="coerce") < 1.5
            mid_mask = (~low_mask) & (
                pd.to_numeric(ctx.results["RR"], errors="coerce") < 2.0
            )
            ctx.results.loc[low_mask, "FinalScore_20d"] = (
                ctx.results.loc[low_mask, "FinalScore_20d"] - 8.0
            )
            ctx.results.loc[mid_mask, "FinalScore_20d"] = (
                ctx.results.loc[mid_mask, "FinalScore_20d"] - 3.0
            )
            ctx.results["FinalScore"] = ctx.results["FinalScore_20d"]
            ctx.results["Score"] = ctx.results["FinalScore_20d"]
    except Exception as e:
        logger.warning("[PIPELINE] Dynamic RR computation failed: %s", e)

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
            mask = (
                (sc >= float(SIGNAL_MIN_SCORE))
                | (mlp >= float(ML_PROB_THRESHOLD))
                | (patt.fillna(0.0) > 0.0)
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
                sort_cols_fb = ["_score_numeric"]
                asc_fb = [False]
                if "SignalReasons_Count" in ctx.results.columns:
                    sort_cols_fb.append("SignalReasons_Count")
                    asc_fb.append(False)
                sort_cols_fb.append("ML_20d_Prob")
                asc_fb.append(False)
                fallback = (
                    ctx.results.assign(
                        _score_numeric=pd.to_numeric(
                            ctx.results[score_col], errors="coerce"
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
                    "thresholds: score>=%s, ml>=%s",
                    len(ctx.results),
                    orig_len,
                    SIGNAL_MIN_SCORE,
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
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        to_save.to_json(
            data_dir / "latest_scan_live.json", orient="records", date_format="iso"
        )
        to_save.to_parquet(data_dir / "latest_scan_live.parquet", index=False)
        logger.info(
            "\u2705 Pipeline Finalized: Saved strict Top %d recommendations",
            len(to_save),
        )
    except Exception as e:
        logger.warning("[PIPELINE] Failed to persist latest scan files: %s", e)

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
