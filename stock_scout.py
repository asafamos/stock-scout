# -*- coding: utf-8 -*-
"""
Stock Scout — Streamlit UI entrypoint.
CONFIG is imported from app_config (built from core.config dataclass).
"""

# ── stdlib ──────────────────────────────────────────────────────────
import os
import io
import time
import json
import logging
import warnings
import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── third-party ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objs as go
import yfinance as yf
from dotenv import load_dotenv, find_dotenv

# ── project: config ─────────────────────────────────────────────────
from app_config import CONFIG, empty_fund_row as _empty_fund_row
from core.risk import calculate_rr

# ── project: indicators ─────────────────────────────────────────────
from indicators import rsi

# ── project: core helpers / UI ──────────────────────────────────────
from core.ui_helpers import (
    StatusManager,
    get_pipeline_stages,
    create_debug_expander,
    SourcesOverview,
)
from core.market_regime import detect_market_regime, adjust_target_for_regime
from core.scan_io import (
    load_latest_scan,
    load_precomputed_scan_with_fallback,
    save_scan as save_scan_helper,
)
from core.pipeline_runner import (
    run_scan_pipeline,
    fetch_top_us_tickers_by_market_cap,
    LAST_UNIVERSE_PROVIDER,
)
from core.data_sources_v2 import clear_cache, reset_disabled_providers

# ── optional: OpenAI ────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── NullStatus fallback (safe no-op status context) ────────────────
class NullStatus:
    """No-op status placeholder when st.status context is unavailable."""
    def write(self, *a, **kw): pass
    def update(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


# Fundamentals schema imported from app_config (single source of truth)




def render_data_sources_overview(provider_status: dict, provider_usage: dict, results: pd.DataFrame) -> None:
    """
    Render a dynamic, compact data sources table showing which providers were actually used in this run.
    Uses emoji indicators and Hebrew labels with RTL layout and avoids any raw HTML inside the dataframe.
    """

    # Canonical provider names; map to internal usage labels if needed
    synonyms = {
        "Alpha Vantage": "Alpha",
        "Nasdaq": "NasdaqDL",
        "Yahoo": "Yahoo",
    }

    table_rows = []
    for provider_name, status_info in provider_status.items():
        ok = bool(status_info.get("ok", False))
        status_icon = "🟢" if ok else "🔴"
        status_text = "פעיל" if ok else "תקלה / חסום"

        usage_key = provider_name if provider_name in provider_usage else synonyms.get(provider_name, provider_name)
        usage_info = provider_usage.get(usage_key, {})

        used_price = bool(usage_info.get("used_price"))
        used_fund = bool(usage_info.get("used_fundamentals"))
        used_ml = bool(usage_info.get("used_ml"))
        implemented = bool(usage_info.get("implemented", True))

        if not implemented:
            status_icon = "⚪"
            status_text = "לא רלוונטי בריצה זו"

        if used_price or used_fund or used_ml:
            used_icon = "🟢"
            used_text = "בשימוש"
        else:
            used_icon = "⚪"
            used_text = "לא בשימוש"

        details_parts = []
        if used_price:
            details_parts.append("מחיר")
        if used_fund:
            details_parts.append("פונדמנטלי")
        if used_ml:
            details_parts.append("ML")

        usage_detail = " | ".join(details_parts) if details_parts else "—"

        table_rows.append(
            {
                "ספק": provider_name,
                "סטטוס": f"{status_icon} {status_text}",
                "שימוש": f"{used_icon} {used_text}",
                "פרטים": usage_detail,
            }
        )

    if not table_rows:
        return

    df_sources = pd.DataFrame(table_rows)
    df_sources["ספק"] = df_sources["ספק"].astype(str)

    styled = (
        df_sources.style
        .set_properties(
            subset=["ספק"],
            **{
                "direction": "ltr",
                "text-align": "left",
                "font-size": "14px",
                "white-space": "nowrap",
            }
        )
        .set_properties(
            subset=["סטטוס", "שימוש", "פרטים"],
            **{
                "text-align": "center",
                "font-size": "14px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("text-align", "center"), ("font-size", "15px")],
                }
            ]
        )
    )

    st.markdown("### 🔌 מקורות נתונים")
    st.dataframe(styled, width='stretch', hide_index=True)

    used_count = 0
    for provider_name in provider_status.keys():
        usage_key = provider_name if provider_name in provider_usage else synonyms.get(provider_name, provider_name)
        info = provider_usage.get(usage_key, {})
        if info.get("used_price") or info.get("used_fundamentals") or info.get("used_ml"):
            used_count += 1
    st.caption(f"סה\"כ ספקים פעילים: {used_count} / {len(provider_status)}")


# Load environment variables
warnings.simplefilter("ignore", FutureWarning)

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Original load (kept for backwards compatibility); `_force_load_env` above already tried explicit paths.
load_dotenv(find_dotenv(usecwd=True))

# Display ML 20d readiness using unified inference loader
try:
    from core.ml_20d_inference import ML_20D_AVAILABLE, FEATURE_COLS_20D, PREFERRED_SCORING_MODE_20D
    if ML_20D_AVAILABLE:
        try:
            st.info(f"✓ ML 20d ready (features: {len(FEATURE_COLS_20D)}; mode: {PREFERRED_SCORING_MODE_20D})")
        except Exception as e:
            logger.debug("unknown: %s", e)
    else:
        st.info("ML 20d model not found; ML scoring will be neutral.")
except Exception as _e:
    st.warning(f"ML 20d loader issue: {_e}")





# ==================== Environment helper ====================
def _env(key: str) -> Optional[str]:
    """Get environment variable or Streamlit secret (supports nested sections)."""
    # Try Streamlit secrets first (Cloud deployment)
    try:
        if hasattr(st, "secrets"):
            # Try direct access (top-level key)
            try:
                val = st.secrets[key]
                if val:  # Ensure it's not empty
                    return str(val)
            except (KeyError, FileNotFoundError) as e:
                logger.debug("_env: %s", e)

            # Try nested sections (api_keys, keys, secrets, tokens)
            for section in ("api_keys", "keys", "secrets", "tokens"):
                try:
                    val = st.secrets[section][key]
                    if val:
                        return str(val)
                except (KeyError, FileNotFoundError, AttributeError):
                    continue
    except Exception as e:
        # Log for debugging in cloud
        if hasattr(st, "warning"):
            st.warning(f"⚠️ Secret access error for {key}: {e}")

    # Fallback to environment variable (local .env)
    return os.getenv(key)



# Earnings date fetching now in core/data_sources_v2.get_next_earnings_date




# Connectivity checks now handled by core.api_preflight.run_preflight
from core.api_preflight import run_preflight as _run_preflight


# ==================== Fundamentals (Alpha -> Finnhub) ====================
# calculate_rr imported from core.risk (single source of truth)



# Beta calculation now in core/pipeline_runner.fetch_beta_vs_benchmark




# External prices now handled by core.price_verify.fetch_prices_for_ticker
from core.price_verify import (
    fetch_prices_for_ticker as _fetch_prices_unified,
    compute_price_stats,
    format_source_badges as _format_source_badges,
)




# ==================== UI ====================
st.set_page_config(
    page_title="סקאוט מניות — 2026", page_icon="📈", layout="wide"
)

# === HEBREW RTL STYLING WITH LTR ENGLISH TEXT ===
st.markdown("""
<style>
/* Global RTL direction */
body, .stApp, .main, .block-container {
    direction: rtl;
    text-align: right;
}
/* Streamlit overrides */
/* RTL text alignment */
h1, h2, h3, h4, h5, h6 { text-align: right; }
/* Force LTR for English text, tickers, numbers, provider names */
span.ltr, .ltr, .stMetricDelta, [class*="st-emotion"] {
    direction: ltr !important;
    text-align: left !important;
    unicode-bidi: embed;
}


</style>
""", unsafe_allow_html=True)

st.title("📈 סקאוט מניות — 2026 אסף")
st.caption("🇮🇱 סקאן מניות אישי בעברית | כלי למחקר בלבד. לא ייעוץ השקעות.")

st.markdown("### 📊 Read-Only Dashboard")
st.session_state["enable_openai_targets"] = False

# Force ML always on (no visible toggle)
st.session_state["ENABLE_ML"] = True
st.session_state["USE_FINAL_SCORE_SORT"] = True

st.markdown("---")



# Status table via unified preflight
@st.cache_data(ttl=300)
def _cached_preflight():
    return _run_preflight(timeout=3.0)

_preflight = _cached_preflight()
alpha_ok = bool(_preflight.get("ALPHAVANTAGE", {}).get("ok", False))
finn_ok = bool(_preflight.get("FINNHUB", {}).get("ok", False))
poly_ok = bool(_preflight.get("POLYGON", {}).get("ok", False))
tiin_ok = bool(_preflight.get("TIINGO", {}).get("ok", False))
fmp_ok = bool(_preflight.get("FMP", {}).get("ok", False))
st.session_state["_alpha_ok"] = alpha_ok
simfin_key = _env("SIMFIN_API_KEY") if CONFIG.get("ENABLE_SIMFIN") else None
marketstack_key = (
    _env("MARKETSTACK_API_KEY") if CONFIG.get("ENABLE_MARKETSTACK") else None
)
nasdaq_key = (
    (_env("NASDAQ_API_KEY") or _env("NASDAQ_DL_API_KEY"))
    if CONFIG.get("ENABLE_NASDAQ_DL")
    else None
)
eodhd_key = (
    (_env("EODHD_API_KEY") or _env("EODHD_TOKEN"))
    if CONFIG.get("ENABLE_EODHD")
    else None
)

# Critical API keys presence check (non-blocking).
# Consider it CRITICAL only if neither key is present at all.
alpha_key_present = bool(_env("ALPHA_VANTAGE_API_KEY") or _env("ALPHAVANTAGE_API_KEY"))
finn_key_present = bool(_env("FINNHUB_API_KEY"))

if not (alpha_key_present or finn_key_present):
    st.error(
        f"""
**CRITICAL: Missing API Keys**

At least one of the following is required for fundamentals:
`ALPHA_VANTAGE_API_KEY` or `FINNHUB_API_KEY`

**Streamlit Cloud:** Add either Alpha Vantage or Finnhub key (or both) under Settings -> Secrets.
Do not include quote characters around the value.

**Local:** Put keys in `.env` and ensure `load_dotenv()` runs.

The pipeline will continue with technical-only scoring.
"""
    )
elif not (alpha_ok or finn_ok):
    st.warning(
        "⚠️ Providers detected but currently unavailable (rate limits or network). Falling back to technical-only scoring and any other price sources."
    )
elif not alpha_ok and finn_ok:
    st.warning(
        "⚠️ Alpha Vantage unavailable (rate limits or config) - falling back to Finnhub and other providers. Signals will still be generated."
    )

# Store provider status in session state for connectivity checks
st.session_state["_alpha_vantage_ok"] = alpha_ok
st.session_state["_finnhub_ok"] = finn_ok
st.session_state["_polygon_ok"] = poly_ok
st.session_state["_tiingo_ok"] = tiin_ok
st.session_state["_fmp_ok"] = fmp_ok

# Initialize centralized status manager
status_manager = StatusManager(get_pipeline_stages())

# Map pipeline detail messages to status stage advancements
_stage_triggers = [
    ("Fetching historical data", "Historical Data Fetch"),
    ("Computing technical indicators", "Technical Indicators"),
    ("Applying Beta filter", "Beta Filter"),
    ("Applying advanced filters", "Advanced Filters"),
    ("Fetching fundamentals", "Fundamentals Enrichment"),
    ("Classifying & Allocating", "Signal Evaluation"),
]
_completed_stages: Set[str] = set()



# timers
def t_start() -> float:
    return time.perf_counter()


def t_end(t0: float) -> float:
    return time.perf_counter() - t0


phase_times: Dict[str, float] = {}
if "av_calls" not in st.session_state:
    st.session_state.av_calls = 0

# ==================== DATA SOURCE MODE ====================
st.markdown("---")
st.markdown("### ⚡ מצב נתונים")

# One-shot Live Scan button:
# - By default, the app prefers precomputed scan (if it exists).
# - If the user clicks the button, we force a single live run and ignore the snapshot for this run only.
if st.button("🔄 הרץ לייב סריקה עכשיו", key="live_scan_button"):
    st.session_state["force_live_scan_once"] = True
    st.session_state["skip_pipeline"] = False  # ensure live path
    st.rerun()

force_live_scan_once = st.session_state.get("force_live_scan_once", False)

# scan_io already imported at top of file

def save_latest_scan_from_results(results_df: pd.DataFrame, metadata: Optional[Dict] = None) -> None:
    """Helper to save scan results using scan_io.save_scan with proper paths.
    Saves even empty DataFrames to avoid missing snapshot state."""
    if results_df is None:
        logger.warning("Cannot save results: DataFrame is None")
        return
    
    # Ensure parquet-safe types (convert complex objects to strings)
    results_to_save = results_df.copy()
    for col in results_to_save.columns:
        if results_to_save[col].dtype == "object":
            results_to_save[col] = results_to_save[col].apply(
                lambda v: v
                if isinstance(v, (str, int, float, bool, np.bool_, np.integer, np.floating)) or v is None
                else str(v)
            )
    
    output_dir = Path(__file__).parent / "data" / "scans"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save live runs under a separate filename to avoid shadowing autoscan snapshots
    path_latest = output_dir / "latest_scan_live.parquet"
    
    # Prepare metadata
    meta = metadata.copy() if metadata else {}
    meta["total_tickers"] = len(results_df)
    meta["scan_type"] = "live_streamlit"
    # Add build commit for parity with autoscan artifacts
    try:
        import subprocess
        meta["build_commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        )
    except Exception as e:
        logger.debug("save_latest_scan_from_results: %s", e)
    
    try:
        save_scan_helper(
            results_df=results_to_save,
            config=CONFIG,
            path_latest=path_latest,
            path_timestamped=None,  # Don't create timestamped backup for live scans
            metadata=meta
        )
        logger.info(f"✅ Saved live scan results: {len(results_df)} tickers to {path_latest}")
    except Exception as e:
        logger.error(f"❌ Failed to save scan: {e}")
        raise

# Attempt to load precomputed scan (internal, no user dropdown)
precomputed_df = None
precomputed_meta = None
use_precomputed = False

# Canonical implementation moved to core.scan_io; keep alias for backward compat
_load_precomputed_scan_with_fallback = load_precomputed_scan_with_fallback


scan_dir = Path(__file__).parent / "data" / "scans"
t0_precomputed = time.perf_counter()
try:
    status_manager.update_detail("Loading precomputed scan from disk...")
    precomputed_df, precomputed_meta, scan_path = _load_precomputed_scan_with_fallback(scan_dir)
    t1_precomputed = time.perf_counter()
    load_time = t1_precomputed - t0_precomputed
    logger.info(f"[PERF] Precomputed scan load time: {load_time:.3f}s (path={scan_path})")
except Exception as exc:
    logger.warning(f"Precomputed scan load failed: {exc}")
    scan_path = scan_dir / "latest_scan.parquet"
    t1_precomputed = time.perf_counter()
    load_time = t1_precomputed - t0_precomputed

# Optionally prefer remote autoscan artifacts from GitHub if newer than local/live
def _parse_iso(ts: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except Exception as exc:
        logger.debug(f"ISO parse failed for '{ts}': {exc}")
        return None

if CONFIG.get("USE_REMOTE_AUTOSCAN", True):
    try:
        repo = CONFIG.get("REMOTE_AUTOSCAN_REPO", "asafamos/stock-scout")
        branch = CONFIG.get("REMOTE_AUTOSCAN_BRANCH", "main")
        base = f"https://raw.githubusercontent.com/{repo}/{branch}/data/scans"
        # Fetch remote metadata first
        r_meta = requests.get(f"{base}/latest_scan.json", timeout=8)
        if r_meta.ok:
            meta_remote = r_meta.json()
            ts_remote = _parse_iso(meta_remote.get("timestamp", ""))

            # Determine current local timestamp (auto/live whichever was chosen)
            ts_current = None
            if precomputed_meta and precomputed_meta.get("timestamp"):
                ts_current = _parse_iso(precomputed_meta.get("timestamp"))
            if not ts_current:
                try:
                    mtime_source = (scan_path.with_suffix('.json') if (scan_path and scan_path.with_suffix('.json').exists()) else scan_path)
                    ts_current = datetime.datetime.fromtimestamp(mtime_source.stat().st_mtime)
                except Exception as exc:
                    logger.debug(f"Scan mtime read failed: {exc}")
                    ts_current = None

            # Only prefer remote if it's newer than current
            if ts_remote and (ts_current is None or ts_remote > ts_current) and not force_live_scan_once:
                r_pq = requests.get(f"{base}/latest_scan.parquet", timeout=20)
                if r_pq.ok:
                    try:
                        df_remote = pd.read_parquet(io.BytesIO(r_pq.content), engine="pyarrow")
                        meta_remote.setdefault("total_tickers", len(df_remote))
                        precomputed_df, precomputed_meta = df_remote, meta_remote
                        scan_path = Path("REMOTE:latest_scan.parquet")
                        logger.info("Using remote autoscan artifacts from GitHub (remote is newer)")
                    except Exception as e:
                        logger.warning(f"Failed to parse remote parquet: {e}")
    except Exception as e:
        logger.warning(f"Remote autoscan fetch skipped/failed: {e}")

timestamp_str = "unknown"
universe_size = 0
scan_age_hours = None
scan_too_old = False

if precomputed_meta is not None:
    timestamp_str = precomputed_meta.get("timestamp", "unknown")
    universe_size = precomputed_meta.get("total_tickers", precomputed_meta.get("universe_size", 0))
    
    # Check scan age (12 hour limit)
    try:
        # Parse metadata timestamp (may be naive or UTC)
        scan_time_meta = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        # Normalize to naive for comparison
        if scan_time_meta.tzinfo is not None:
            scan_time_meta = scan_time_meta.replace(tzinfo=None)
        
        # Also consider file modification time (parquet/json), use the freshest
        try:
            meta_path = scan_path.with_suffix('.json')
            mtime_source = meta_path if meta_path.exists() else scan_path
            scan_time_file = datetime.datetime.fromtimestamp(mtime_source.stat().st_mtime)
        except Exception as exc:
            logger.debug(f"Scan mtime fallback: {exc}")
            scan_time_file = scan_time_meta

        # Prefer the most recent timestamp available (both naive now)
        scan_time = max(scan_time_meta, scan_time_file)
        scan_age_hours = (datetime.datetime.now() - scan_time).total_seconds() / 3600
        scan_too_old = scan_age_hours > 12
    except Exception as exc:
        logger.debug(f"Scan age check failed, assuming stale: {exc}")
        scan_too_old = True

# Helper: show snapshot provenance banner
def _render_snapshot_banner(meta: Dict[str, Any], path_obj: Path, age_hours: Optional[float]) -> None:
    src = "Remote autoscan (GitHub)" if str(path_obj).startswith("REMOTE:") else (
        "Local latest" if path_obj.name == "latest_scan.parquet" else "Local backup"
    )
    ts = meta.get("timestamp", "unknown")
    commit = meta.get("build_commit", "unknown")
    saved = meta.get("total_tickers")
    uni = meta.get("universe_size")
    age_txt = f"{age_hours:.1f} שעות" if isinstance(age_hours, (int, float)) else "unknown"
    if saved is not None and uni is not None:
        st.caption(f"🧾 מקור: {src} • קומיט: {commit} • זמן: {ts} • גיל: {age_txt} • מניות נשמרו: {saved} מתוך נסקרו: {uni}")
    else:
        size = saved or uni or "unknown"
        st.caption(f"🧾 מקור: {src} • קומיט: {commit} • זמן: {ts} • גיל: {age_txt} • מניות: {size}")

if force_live_scan_once:
    # User explicitly forced a live run: ignore any snapshot age/status
    st.info("🔄 סריקה חיה נכפית - מתעלם מסריקה אוטומטית.")
    st.caption(f"📊 סריקה אוטומטית מ-{timestamp_str} מתעלמת עבור הרצה זו.")
    use_precomputed = False
    st.session_state["skip_pipeline"] = False
elif precomputed_df is not None and precomputed_meta is not None and not scan_too_old:
    # Successfully loaded and NOT too old -> use precomputed snapshot
    status_manager.advance(
        f"Precomputed scan loaded: {universe_size} tickers (last updated: {timestamp_str})"
    )
    age_display = f"{scan_age_hours:.1f}" if isinstance(scan_age_hours, (int, float)) else "unknown"
    st.success(f"✅ נתונים עדכניים מסריקה אוטומטית ({age_display} שעות)")
    _render_snapshot_banner(precomputed_meta, scan_path, scan_age_hours)
    st.caption(f"📊 {universe_size} מניות נותחו | ⏰ סריקה אוטומטית פעמיים ביום (8:00 + 20:00 UTC)")

    st.session_state["skip_pipeline"] = True
    st.session_state["precomputed_results"] = precomputed_df
    try:
        st.session_state["universe_size"] = int(universe_size)
    except Exception as e:
        logger.debug("_render_snapshot_banner: %s", e)
    logger.info(f"[PERF] Precomputed scan: DataFrame shape {precomputed_df.shape}")
    use_precomputed = True
else:
    # Either no snapshot exists, or scan is too old
    if scan_too_old and precomputed_df is not None:
        age_display = f"{scan_age_hours:.1f}" if isinstance(scan_age_hours, (int, float)) else "unknown"
        st.warning(f"⚠️ הסריקה הקיימת ישנה מדי ({age_display} שעות) - מחכה לסריקה אוטומטית הבאה")
        _render_snapshot_banner(precomputed_meta, scan_path, scan_age_hours)
        st.info("💡 סריקה אוטומטית חדשה תתבצע תוך מספר שעות (פעמיים ביום: 8:00 + 20:00 UTC)")
        # Use old scan anyway but warn user
        st.session_state["skip_pipeline"] = True
        st.session_state["precomputed_results"] = precomputed_df
        try:
            st.session_state["universe_size"] = int(universe_size)
        except Exception as e:
            logger.debug("unknown: %s", e)
        use_precomputed = True
    else:
        st.info("📊 אין סריקה זמינה - מחכה לסריקה אוטומטית הבאה.")
        st.caption("💡 סריקות אוטומטיות רצות פעמיים ביום דרך GitHub Actions.")
        use_precomputed = False
        st.session_state["skip_pipeline"] = False
    
# Reset the one-shot flag (always)
st.session_state["force_live_scan_once"] = False

# ==================== MAIN PIPELINE ====================
st.markdown("---")
st.markdown("### 🚀 Pipeline Execution")

# Check if we should skip pipeline (precomputed mode)
skip_pipeline = st.session_state.get("skip_pipeline", False) and use_precomputed

# Always show market regime
t0_regime = t_start()
if not skip_pipeline:
    status_manager.update_detail("Analyzing SPY/QQQ/VIX trends...")

@st.cache_data(ttl=3600)
def _cached_detect_market_regime(lookback_days: int = 60):
    return detect_market_regime(lookback_days=lookback_days)

market_regime_data = _cached_detect_market_regime(lookback_days=60)
regime = market_regime_data.get("regime", "neutral")
regime_confidence = market_regime_data.get("confidence", 50)
phase_times["market_regime"] = t_end(t0_regime)

# Show regime with color coding
regime_emoji = {"bullish": "📈", "neutral": "➡️", "bearish": "📉"}
regime_color = {"bullish": "#16a34a", "neutral": "#f59e0b", "bearish": "#dc2626"}

st.markdown(
    f"""<div style='background:{regime_color[regime]};color:white;padding:12px;border-radius:8px;margin:10px 0'>
    <strong>{regime_emoji[regime]} Market Regime: {regime.upper()}</strong> (confidence: {regime_confidence}%)<br>
    <small>{market_regime_data.get('details', '')}</small>
    </div>""",
    unsafe_allow_html=True
)

# Store in session state for use in target calculations
st.session_state['market_regime'] = market_regime_data

if skip_pipeline:
    # Use precomputed results from full pipeline
    t0_stage3 = time.perf_counter()
    results = st.session_state.get("precomputed_results")
    logger.info(f"[PRECOMPUTED] Loaded {len(results)} tickers from full pipeline scan")
    
    # Add 'Score' alias for overall_score_20d to maintain compatibility
    if "overall_score_20d" in results.columns and "Score" not in results.columns:
        results["Score"] = results["overall_score_20d"]
    elif "FinalScore_20d" in results.columns and "Score" not in results.columns:
        results["Score"] = results["FinalScore_20d"]
    
    # UI now shows all pipeline results; no adaptive percentile or top-N capping
    logger.info(f"[PRECOMPUTED] Final display: {len(results)} stocks (no adaptive/top-N filter applied)")
    
    # IMPORTANT: Update session state with filtered results so they persist
    st.session_state["precomputed_results"] = results.copy()
    
    try:
        status_manager.update_detail(f"Precomputed scan: {len(results)} top stocks")
        status_manager.set_progress(1.0)
    except Exception as e:
        logger.debug("_cached_detect_market_regime: %s", e)
    
    # Show summary to user
    try:
        original_count = int(st.session_state.get("universe_size", 0))
        if not original_count:
            # Fall back to computed universe_size (from metadata) or current results length
            try:
                original_count = int(universe_size)
            except Exception as exc:
                logger.debug(f"universe_size parse: {exc}")
                original_count = len(results)
    except Exception as exc:
        logger.debug(f"original_count fallback: {exc}")
        original_count = len(results)
    st.info(f"⚡ **{len(results)} מניות מובילות** מתוך {original_count} שעברו סריקה מלאה")
    st.caption("✅ כולל: ML model, Technical scoring, Fundamental data, Risk assessment, Classification")
    
    t1_stage3 = time.perf_counter()
    logger.info(f"[PERF] Precomputed stage 3/10 (post-load to recommendations) time: {t1_stage3-t0_stage3:.3f}s")
    
else:
    # Live scan execution fallback
    with st.status("🚀 Running Live Scan...", expanded=True) as status:
        status = status or NullStatus()
        status.write("Initializing pipeline...")

        # Clear cache and reset providers for consistent results
        clear_cache()
        reset_disabled_providers()
        status.write("Cache cleared, providers reset")

        # 1. Fetch Universe
        universe = fetch_top_us_tickers_by_market_cap(limit=CONFIG["UNIVERSE_LIMIT"])
        status.write(f"Fetched universe: {len(universe)} tickers")
        
        # 2. Run Pipeline (now returns wrapper {result, meta})
        wrapper = run_scan_pipeline(
            universe,
            CONFIG,
            status_callback=status.write if status else None
        )
        meta = wrapper.get("meta", {}) if isinstance(wrapper, dict) else {}
        payload = wrapper.get("result") if isinstance(wrapper, dict) else wrapper
        # Prefer new dict schema
        if isinstance(payload, dict) and ("results_df" in payload):
            _results_df = payload.get("results_df")
            results = _results_df if _results_df is not None else pd.DataFrame()
            data_map = payload.get("data_map") or None
        # Backward: tuple/list
        elif isinstance(payload, tuple) and len(payload) >= 1:
            results = payload[0]
            data_map = payload[1] if len(payload) > 1 else None
        # Backward: bare DataFrame
        elif isinstance(payload, pd.DataFrame):
            results = payload
            data_map = None
        else:
            results = pd.DataFrame()
            data_map = None
        # Pull diagnostics (filter reasons) if available
        diagnostics = {}
        try:
            diagnostics = (payload.get("diagnostics") if isinstance(payload, dict) else {}) or {}
        except Exception as exc:
            logger.debug(f"Diagnostics payload extract failed: {exc}")
            diagnostics = {}

        # Display meta prominently
        try:
            engine_version = meta.get("engine_version", "unknown")
            engine_mode = meta.get("engine_mode")
            used_fb = bool(meta.get("used_legacy_fallback", False))
            fb_reason = meta.get("fallback_reason")
            sources_used = meta.get("sources_used")
            run_ts = meta.get("run_timestamp_utc")
            # Top banner
            st.caption(f"Engine: {engine_version} · Run: {run_ts or 'n/a'}")
            if engine_mode == "SIGNAL_ONLY":
                st.caption("Signal-based scan (no allocation, no position sizing)")
            # Save mode for later UI decisions
            st.session_state["engine_mode"] = engine_mode
            if sources_used:
                try:
                    # Universe provider
                    src_line = sources_used.get("universe_provider") or sources_used
                    st.caption(f"Universe Source: {src_line}")
                    # Compact telemetry summary
                    price_used = sorted(list((sources_used.get("price") or {}).keys()))
                    fund_used = sorted(list((sources_used.get("fundamentals") or {}).keys()))
                    fb_count = len(sources_used.get("fallback_events") or [])
                    if price_used:
                        st.caption(f"Price Providers: {', '.join(price_used)}")
                    if fund_used:
                        st.caption(f"Fund Providers: {', '.join(fund_used)}")
                    if fb_count:
                        st.caption(f"Fallback Events: {fb_count}")
                except Exception as e:
                    logger.debug("unknown: %s", e)
            # Show Tier 1 filtered reasons (diagnostics)
            try:
                if diagnostics:
                    filtered_rows = []
                    for tkr, rec in diagnostics.items():
                        tier1 = rec.get("tier1_reasons") or []
                        if tier1:
                            def _fmt_reason(r: dict) -> str:
                                try:
                                    base = f"{r.get('rule')}: {r.get('message')}"
                                    val = r.get('value')
                                    thr = r.get('threshold')
                                    if val is not None or thr is not None:
                                        return f"{base} (val={val}, thr={thr})"
                                    return base
                                except Exception as exc:
                                    logger.debug(f"Tier1 reason format: {exc}")
                                    return str(r)
                            joined = "; ".join([_fmt_reason(r) for r in tier1])
                            filtered_rows.append({
                                "Ticker": tkr,
                                "Reasons": joined,
                                "last_price": rec.get("last_price"),
                                "last_volume": rec.get("last_volume"),
                            })
                    if filtered_rows:
                        with st.expander("Filtered Out (Tier 1)", expanded=False):
                            st.dataframe(pd.DataFrame(filtered_rows), use_container_width=True)
            except Exception as e:
                logger.debug("_fmt_reason: %s", e)
            if used_fb:
                st.warning(f"Legacy fallback engaged — {fb_reason or 'reason unavailable'}")
            # ML health warnings
            if bool(meta.get("ml_bundle_version_warning")):
                st.warning(f"ML version mismatch — {meta.get('ml_bundle_warning_reason')}")
            if bool(meta.get("ml_degraded")):
                missing = meta.get("ml_missing_features") or []
                if missing:
                    st.warning(f"ML degraded — missing features: {', '.join(missing)}")
                else:
                    st.warning("ML degraded — limited features or model unavailable")
        except Exception as e:
            logger.debug("_fmt_reason: %s", e)
        
        # 3. Mark scan as ready (actual save happens AFTER sector cap is applied below)
        if not results.empty:
            # Store in session state for sector cap processing, final save happens at line ~3435
            st.session_state["precomputed_results"] = results
            st.success(f"✅ Scan complete: {len(results)} results found (will apply sector cap)")
        else:
            st.error("❌ Live scan returned 0 results. Check logs/filtering.")

# Debug logging if enabled
create_debug_expander({
    "regime": regime,
    "confidence": regime_confidence,
    "spy_trend": market_regime_data.get("spy_trend", 0),
    "qqq_trend": market_regime_data.get("qqq_trend", 0),
    "vix_level": market_regime_data.get("vix", 0),
}, title="📊 Market Regime Details")

# API keys and universe provider diagnostics
try:
    api_status = {
        "FMP_API_KEY": "OK" if bool(_env("FMP_API_KEY")) else "MISSING",
        "POLYGON_API_KEY": "OK" if bool(_env("POLYGON_API_KEY")) else "MISSING",
        "EODHD_API_KEY": "OK" if bool(_env("EODHD_API_KEY")) else "MISSING",
        "FINNHUB_API_KEY": "OK" if bool(_env("FINNHUB_API_KEY")) else "MISSING",
        "ALPHA_VANTAGE_API_KEY": "OK" if bool(_env("ALPHA_VANTAGE_API_KEY")) else "MISSING",
        "TIINGO_API_KEY": "OK" if bool(_env("TIINGO_API_KEY")) else "MISSING",
        "Universe_Provider": LAST_UNIVERSE_PROVIDER,
    }
    create_debug_expander(api_status, title="🔑 API Keys & Universe Provider")
except Exception as e:
    logger.debug("unknown: %s", e)

# Initialize sources tracker
sources_overview = SourcesOverview()

## Read-only mode: live scan execution removed. Always use precomputed results loaded above.

# External price verification (Top-K) — using core.price_verify
t0 = t_start()
if CONFIG.get("EXTERNAL_PRICE_VERIFY", False):
    results["Price_Alpha"] = np.nan
    results["Price_Finnhub"] = np.nan
    results["Price_Polygon"] = np.nan
    results["Price_Tiingo"] = np.nan
    results["Price_Mean"] = np.nan
    results["Price_STD"] = np.nan
    results["Source_List"] = "🟡Yahoo"
    results["Historical_StdDev"] = np.nan


# Build provider_status dict for core.price_verify from preflight
_pv_status = {
    "alpha": alpha_ok,
    "finnhub": finn_ok,
    "polygon": poly_ok,
    "tiingo": tiin_ok,
    "fmp": fmp_ok,
}

# External price verification — run if ANY provider is available
any_price_provider = finn_ok or poly_ok

if CONFIG["EXTERNAL_PRICE_VERIFY"] and any_price_provider and "Price_Yahoo" in results.columns:
    subset_idx = list(results.head(int(CONFIG["TOP_VALIDATE_K"])).index)
    with st.spinner(f"💵 Verifying prices across {len(subset_idx)} top stocks..."):
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [
                ex.submit(
                    _fetch_prices_unified,
                    results.loc[idx, "Ticker"],
                    float(results.loc[idx, "Price_Yahoo"]),
                    _pv_status,
                )
                for idx in subset_idx
            ]
            for f in as_completed(futures):
                try:
                    vals, srcs = f.result()
                except Exception as exc:
                    logger.debug(f"External price verify future failed: {exc}")
                    continue
                # Identify ticker from vals (Yahoo price matches)
                tkr = None
                for si in subset_idx:
                    t = results.loc[si, "Ticker"]
                    yp = float(results.loc[si, "Price_Yahoo"])
                    if abs(vals.get("Yahoo", 0) - yp) < 0.01:
                        tkr = t
                        idx = si
                        break
                if tkr is None:
                    continue
                pmean, pstd, count = compute_price_stats(vals)
                results.loc[
                    idx,
                    [
                        "Price_Alpha",
                        "Price_Finnhub",
                        "Price_Polygon",
                        "Price_Tiingo",
                        "Price_Mean",
                        "Price_STD",
                        "Source_List",
                    ],
                ] = [
                    vals.get("Alpha", np.nan),
                    vals.get("Finnhub", np.nan),
                    vals.get("Polygon", np.nan),
                    vals.get("Tiingo", np.nan),
                    pmean,
                    pstd,
                    _format_source_badges(srcs),
                ]
                # Compute historical std dev for this ticker (only for verified subset)
                ticker = results.loc[idx, "Ticker"]
                if not skip_pipeline and ticker in data_map:
                    hist = data_map[ticker]
                    if len(hist) >= 5:  # Minimum 5 candles
                        recent = hist["Close"].tail(min(30, len(hist)))
                        if len(recent) >= 5:
                            results.at[idx, "Historical_StdDev"] = float(recent.std())
    # Price reliability metric (enhanced with better spread: 0.1-1.0)
    results["Price_Reliability"] = np.nan
    for i, row in results.iterrows():
        pmean = row.get("Price_Mean", np.nan)
        pstd = row.get("Price_STD", np.nan)
        providers = (
            str(row.get("Source_List", "")).split(" - ")
            if isinstance(row.get("Source_List"), str)
            else []
        )
        providers = [p for p in providers if p]
        count = len(providers)

        if np.isfinite(pmean) and pmean > 0 and np.isfinite(pstd):
            pct_std = pstd / pmean

            # Source count component (0-4 sources -> 0.2-1.0)
            if count == 1:
                source_score = 0.2
            elif count == 2:
                source_score = 0.5
            elif count == 3:
                source_score = 0.75
            else:  # 4+
                source_score = 1.0

            # Variance component (lower variance = higher reliability)
            # 0% variance -> 1.0, 5% variance -> 0.5, 10%+ variance -> 0.2
            if pct_std < 0.01:  # <1%
                variance_score = 1.0
            elif pct_std < 0.03:  # 1-3%
                variance_score = 0.85
            elif pct_std < 0.05:  # 3-5%
                variance_score = 0.65
            elif pct_std < 0.10:  # 5-10%
                variance_score = 0.4
            else:  # >10%
                variance_score = 0.2

            # Combined: 60% source count, 40% variance
            reliability = (source_score * 0.6) + (variance_score * 0.4)
            results.at[i, "Price_Reliability"] = round(
                np.clip(reliability, 0.1, 1.0), 4
            )
        else:
            # Fallback based on source count only
            if count == 1:
                results.at[i, "Price_Reliability"] = 0.15
            elif count == 2:
                results.at[i, "Price_Reliability"] = 0.35
            elif count == 3:
                results.at[i, "Price_Reliability"] = 0.55
            else:
                results.at[i, "Price_Reliability"] = 0.75

    # Price sources count column
    results["Price_Sources_Count"] = results["Source_List"].apply(
        lambda s: len(str(s).split(" - ")) if isinstance(s, str) and s else 0
    )

    # Fundamental reliability metric
    # Simplified Fundamental reliability metric based on number of distinct
    # fundamental providers that contributed any field. Mapping:
    #   0 -> 0.0
    #   1 -> 0.33
    #   2 -> 0.66
    #  >=3 -> 1.0
    def _map_sources_to_reliability(n: int) -> float:
        if n <= 0:
            return 0.0
        if n == 1:
            return 0.33
        if n == 2:
            return 0.66
        return 1.0

    if "Fund_Coverage_Pct" in results.columns:
        results["Fundamental_Reliability"] = 0.0
        fund_flags = [
            "from_fmp_full",
            "from_fmp",
            "from_simfin",
            "from_eodhd",
            "from_alpha",
            "from_finnhub",
            "from_tiingo",
        ]
        for i, row in results.iterrows():
            # Count providers that truly contributed (flags are written earlier)
            provider_count = int(sum(bool(row.get(f)) for f in fund_flags))
            # If coverage indicates data but flags are zero, treat as 1 source
            cov = row.get("Fund_Coverage_Pct", np.nan)
            if (
                provider_count == 0
                and isinstance(cov, (int, float))
                and np.isfinite(cov)
                and cov > 0
            ):
                provider_count = 1
            final_rel = _map_sources_to_reliability(provider_count)
            results.at[i, "Fundamental_Reliability"] = round(final_rel, 4)
    else:
        results["Fundamental_Reliability"] = 0.0

    # Fundamental sources count column (flags) — exact count (0 allowed if truly none)
    fund_flags = [
        "from_fmp_full",
        "from_fmp",
        "from_simfin",
        "from_eodhd",
        "from_alpha",
        "from_finnhub",
        "from_tiingo",
    ]

    def _fund_count(row: pd.Series) -> int:
        return int(sum(bool(row.get(f)) for f in fund_flags))

    results["Fundamental_Sources_Count"] = results.apply(_fund_count, axis=1)

    # Expose canonical fund reliability field using the simple mapping helper
    try:
        from core.scoring.fundamental import compute_fund_reliability
    except Exception as exc:
        logger.debug(f"compute_fund_reliability import failed: {exc}")
        compute_fund_reliability = None

    if compute_fund_reliability is not None:
        results["fund_reliability"] = results["Fundamental_Sources_Count"].apply(
            lambda n: float(compute_fund_reliability(n))
        )
        # Keep legacy column name in sync
        results["Fundamental_Reliability"] = results["fund_reliability"].copy()
    else:
        # Fallback: copy existing value or 0.0
        results["fund_reliability"] = results.get("Fundamental_Reliability", 0.0)

    # Combined reliability score
    if (
        "Price_Reliability" in results.columns
        and "Fundamental_Reliability" in results.columns
    ):
        results["Reliability_Score"] = (
            0.4 * results["Price_Reliability"].fillna(0)
            + 0.6 * results["Fundamental_Reliability"].fillna(0)
        ).round(4)
    else:
        results["Reliability_Score"] = np.nan
phase_times["price_verification"] = t_end(t0)
status_manager.update_detail(f"Price verification: {len(results)} validated")
status_manager.advance("Price Verification")

# Update ReliabilityScore in results (sync with Reliability_Score if computed)
# Only overwrite if pipeline hasn't already provided a valid ReliabilityScore
_has_pipeline_reliability = (
    "ReliabilityScore" in results.columns
    and results["ReliabilityScore"].notna().any()
    and (results["ReliabilityScore"] > 0).any()
)
if not _has_pipeline_reliability:
    if "Reliability_Score" in results.columns and results["Reliability_Score"].notna().any():
        # Convert from 0-1 to 0-100 scale and update ReliabilityScore
        results["ReliabilityScore"] = (results["Reliability_Score"].fillna(0.5) * 100).round(2)
    elif "Price_Reliability" in results.columns:
        # Fallback: use price reliability alone if fundamental not available
        results["ReliabilityScore"] = (results["Price_Reliability"].fillna(0.5) * 100).round(2)

# NOTE: Save happens AFTER sector cap is applied (at line ~3435) to ensure consistent reload counts


# Horizon heuristic
def infer_horizon(row: pd.Series) -> str:
    rsi_v = row.get("RSI", np.nan)
    near = row.get("Near52w", np.nan)
    score = row.get("Score", 0)
    if (
        isinstance(near, (int, float))
        and near >= 90
        and isinstance(rsi_v, (int, float))
        and 55 <= rsi_v <= 75
    ):
        return "Short (1-3 months)"
    if isinstance(rsi_v, (int, float)) and rsi_v <= 40 and score >= 60:
        return "Medium (3-6 months)"
    if score >= 70:
        return "Medium-Long (6-12 months)"
    return "Long (>12 months)"


results["Holding_Horizon"] = results.apply(infer_horizon, axis=1)


# Sector cap
def apply_sector_cap(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if not CONFIG["SECTOR_CAP_ENABLED"]:
        return df
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"
    # Vectorized: rank within each sector, keep rank <= cap
    df["_rank"] = df.groupby("Sector", sort=False).cumcount() + 1
    result = df[df["_rank"] <= cap].drop("_rank", axis=1).reset_index(drop=True)
    return result


score_candidates = [
    "overall_score_20d",
    "Score",
    "overall_score",
    "conviction_v2_final",
    "overall_score_pretty",
]
score_col = None
for c in score_candidates:
    if c in results.columns:
        score_col = c
        break

# Choose a safe ticker column if available
ticker_candidates = ["Ticker", "symbol", "Symbol", "ticker"]
ticker_col = None
for t in ticker_candidates:
    if t in results.columns:
        ticker_col = t
        break

sort_by = []
ascending = []
if score_col is not None:
    sort_by.append(score_col)
    ascending.append(False)
if ticker_col is not None:
    sort_by.append(ticker_col)
    ascending.append(True)

if sort_by:
    sorted_results = results.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
else:
    # No valid sort keys; keep original order
    sorted_results = results.reset_index(drop=True)

results = apply_sector_cap(
    sorted_results,
    int(CONFIG["SECTOR_CAP_MAX"]),
)

# Save FINAL filtered results (after sector cap) ONLY for live scans
# Don't re-save on precomputed reloads to avoid compounding filters
is_live_scan = not st.session_state.get("skip_pipeline", False)
if is_live_scan and not results.empty:
    try:
        meta_final = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",  # UTC with Z suffix for consistency
            "scan_type": "live_streamlit_final",
            "total_tickers": len(results),  # Count AFTER sector cap
            "sector_cap_applied": True,
        }
        save_latest_scan_from_results(results, metadata=meta_final)
        # Also update session state with the final filtered results
        st.session_state["precomputed_results"] = results.copy()
        logger.info(f"✅ Saved FINAL results after sector cap: {len(results)} tickers")
    except Exception as e:
        logger.warning(f"Failed to save final scan: {e}")


# Source badges now use core.price_verify.format_source_badges
def source_badges(row: pd.Series) -> str:
    """Build badges from provider flags and Source_List."""
    badges: list[str] = []
    if row.get("from_fmp_full") or row.get("from_fmp"):
        badges.append("🟣FMP")
    if row.get("from_simfin"):
        badges.append("🧪SimFin")
    if row.get("from_eodhd"):
        badges.append("📘EODHD")
    if row.get("from_alpha"):
        badges.append("🟣Alpha")
    if row.get("from_finnhub"):
        badges.append("🔵Finnhub")
    price_sources = row.get("Source_List")
    if isinstance(price_sources, str) and price_sources:
        for provider in price_sources.split(" - "):
            if provider and provider not in badges:
                badges.append(provider)
    if not badges:
        badges.append("🟡Yahoo")
    return _format_source_badges(badges)


results["Price_Sources"] = results.apply(source_badges, axis=1)
results["Price_Mean_Rounded"] = results["Price_Mean"].round(2)
results["Price_STD_Rounded"] = results["Price_STD"].round(4)
results["Unit_Price"] = np.where(
    results["Price_Mean_Rounded"].notna(),
    results["Price_Mean_Rounded"],
    results.get("Price_Yahoo", results.get("Close", np.nan)),
)
results["Unit_Price"] = pd.to_numeric(results["Unit_Price"], errors="coerce")

# Show ALL stocks that passed filters (no limit)
TOPN = len(results)

# UI does not apply ML confidence thresholds or perform allocation; pipeline output is authoritative

alloc_df = results.reset_index(drop=True).copy()

# Defensive: ensure allocation column exists even if upstream function changed
if "סכום קנייה ($)" not in results.columns:
    if "buy_amount_v2" in results.columns:
        results["סכום קנייה ($)"] = pd.to_numeric(results["buy_amount_v2"], errors="coerce")
    elif "Buy Amount ($)" in results.columns:
        results["סכום קנייה ($)"] = pd.to_numeric(results["Buy Amount ($)"], errors="coerce")
    else:
        results["סכום קנייה ($)"] = np.nan

results["מניות לקנייה"] = np.floor(
    np.where(
        (results["Unit_Price"] > 0) & (results["סכום קנייה ($)"].fillna(0) > 0),
        results["סכום קנייה ($)"].fillna(0) / results["Unit_Price"],
        0,
    )
).astype(int)
results["עודף ($)"] = np.round(
    np.where(
        pd.notna(results["סכום קנייה ($)"]) & (results["מניות לקנייה"] > 0),
        results["סכום קנייה ($)"] - results["מניות לקנייה"] * results["Unit_Price"],
        0.0,
    ),
    2,
)

# Timings and provider call count are displayed elsewhere; skip budget enforcement KPIs
    
# For precomputed mode: results variable already went through sector cap and sorting above.
# We just need to apply column renames for UI compatibility, but NOT reload from session state
# (which would discard the sector cap filtering).
if st.session_state.get("precomputed_results") is not None and st.session_state.get("skip_pipeline", False):
    # Apply column renames for UI compatibility (in-place on the already-filtered results)
    # Only rename Overall_Score→Score if Score doesn't already exist (avoid silent overwrite)
    column_renames = {
        'Close': 'Price_Yahoo',
        'Technical_Score': 'Score_Tech',
        'Fundamental_Score': 'Fundamental_S',
        'Fund_Sources_Count': 'fund_sources_used_v2',
    }
    if 'Score' not in results.columns:
        column_renames['Overall_Score'] = 'Score'
    results = results.rename(columns=column_renames)
    # Ensure both reliability aliases exist for UI fallbacks
    if 'Reliability_v2' not in results.columns and 'reliability_v2' in results.columns:
        results['Reliability_v2'] = results['reliability_v2']
    if 'reliability_pct' not in results.columns and 'Reliability_v2' in results.columns:
        results['reliability_pct'] = results['Reliability_v2']
    # Sector alias
    if 'Sector' not in results.columns and 'sector' in results.columns:
        results['Sector'] = results['sector']
    
    data_map = {}
    phase_times = phase_times if 'phase_times' in locals() else {}
    logger.info(f"Rendering precomputed scan with {len(results)} tickers (after sector cap)")
    try:
        status_manager.update_detail("Precomputed scan loaded — using cached results")
        status_manager.set_progress(1.0)
    except Exception as e:
        logger.debug("unknown: %s", e)
    st.info("⚡ Rendering using precomputed scan (no live pipeline run)")

# Note: Auto-save is now handled in the live pipeline section above
# No need for manual save button - results are saved automatically

# Mark pipeline completion at UI level
try:
    status_manager.advance("Signal Evaluation")
    status_manager.complete("✅ Pipeline complete")
except Exception as e:
    logger.debug("unknown: %s", e)

# Close the skip_pipeline conditional block
# (The if not skip_pipeline block ends here)







st.subheader("📊 Market Scan Results")
st.caption("Signal candidates shown. This is not investment advice.")

# Sidebar filters
# Sidebar removed - all controls moved to top bar above

# Read session state values set by top control bar
universe_size = int(st.session_state.get("universe_size", 2000))
fast_mode = bool(st.session_state.get("fast_mode", False))
total_budget = float(st.session_state.get("total_budget", CONFIG["BUDGET_TOTAL"]))
min_position = float(st.session_state.get("min_position", max(50.0, round(total_budget * 0.10))))
max_position_pct = float(st.session_state.get("max_position_pct", CONFIG["MAX_POSITION_PCT"]))
alloc_style_idx = int(st.session_state.get("alloc_style_idx", 0))
enable_multi_source = bool(st.session_state.get("enable_multi_source", True))
enable_ml_boost = bool(st.session_state.get("enable_ml_boost", True))
ml_threshold = int(st.session_state.get("ml_threshold", 0))
enable_openai_targets = bool(st.session_state.get("enable_openai_targets", False))

# Legacy variables for backward compatibility
RELAXED_MODE = bool(st.session_state.get("RELAXED_MODE", False))
FAST_MODE = bool(st.session_state.get("FAST_MODE", False))
DEBUG_SKIP_PIPELINE = bool(st.session_state.get("DEBUG_SKIP_PIPELINE", False))
use_full_export = bool(st.session_state.get("use_full_export", False))

# Initialize view filters from session state or defaults
risk_filter = ["core", "speculative"]
quality_filter = ["high", "medium", "low"]
sector_filter = []

# Continue with main app flow
st.markdown("---")

# Initialize view parameters for read-only display
show_debug_attr = False
compact_mode = False

initial_rec_count = len(results)
rec_df = results.copy()

## Do not apply local thresholds or filters; pipeline output is authoritative

## Skip risk/quality/sector/action filtering in UI

logger.info(f"[FILTER] Final candidates after pipeline: {len(rec_df)} stocks (started with {initial_rec_count})")

## Do not reshuffle candidates; display pipeline order and content only

## Remove pass/fail metrics derived from local thresholds

## Do not display local filtering removal captions

# --- DEBUG: Show top 5 with canonical 20d ML columns ---
if not rec_df.empty and "FinalScore_20d" in rec_df.columns:
    debug_cols = ["Ticker", "TechScore_20d", "ML_20d_Prob", "FinalScore_20d", "FinalScore"]
    available_debug_cols = [c for c in debug_cols if c in rec_df.columns]
    if available_debug_cols:
        st.info("🔍 Top 5 signal candidates (20d ML scoring):")
        top5_debug = rec_df.head(5)[available_debug_cols].copy()
        st.dataframe(top5_debug, width='stretch', hide_index=True)

rec_df = rec_df.copy()


# Responsive recommendation grid + card styles (full-width cards, auto-fit columns)
# (CSS and markdown blocks should be inside st.markdown or string, not as stray lines)

## Do not reorder; preserve pipeline ranking

# Empty-state messaging based solely on pipeline results
if rec_df.empty:
    st.info("No strong signals found in this scan.")
else:
    rec_df["Fallback_Display"] = False

# Provider usage tracking (aggregate from source lines)
# Build accurate provider usage tracker using session markers and per-row flags
providers_meta = {
    "Yahoo": {"env": None, "implemented": True, "label": "Yahoo"},
    "Alpha": {"env": "ALPHA_VANTAGE_API_KEY", "implemented": True, "label": "Alpha"},
    "Finnhub": {"env": "FINNHUB_API_KEY", "implemented": True, "label": "Finnhub"},
    "Tiingo": {"env": "TIINGO_API_KEY", "implemented": True, "label": "Tiingo"},
    "Polygon": {"env": "POLYGON_API_KEY", "implemented": True, "label": "Polygon"},
    "FMP": {"env": "FMP_API_KEY", "implemented": True, "label": "FMP"},
    "OpenAI": {"env": "OPENAI_API_KEY", "implemented": True, "label": "OpenAI"},
    # Mark optional providers as implemented by default; connectivity/key presence will reflect availability
    "SimFin": {"env": "SIMFIN_API_KEY", "implemented": True, "label": "SimFin"},
    "Marketstack": {"env": "MARKETSTACK_API_KEY", "implemented": True, "label": "Marketstack"},
    "EODHD": {"env": "EODHD_API_KEY", "implemented": True, "label": "EODHD"},
    "NasdaqDL": {"env": "NASDAQ_API_KEY", "implemented": True, "label": "NasdaqDL"},
}

# session-level usage markers updated by data fetch helpers (see _fetch_external_for and mark_provider_usage)
session_usage = st.session_state.get("provider_usage", {}) or {}

provider_usage = {}
for p, meta in providers_meta.items():
    key_present = False
    if meta["env"]:
        try:
            key_present = bool(_env(meta["env"]))
        except Exception as exc:
            logger.debug(f"Provider key check failed for {p}: {exc}")
            key_present = False
    else:
        # Providers that don't need a key (Yahoo)
        key_present = True

    implemented = bool(meta.get("implemented", True))

    # used flags: check session markers and per-row flags in results
    used_price = False
    used_fund = False
    used_ml = False

    # Session usage (set in fetch helpers)
    if p in session_usage:
        cats = session_usage.get(p) or set()
        used_price = used_price or ("price" in cats)
        used_fund = used_fund or ("fundamentals" in cats)
        used_ml = used_ml or ("ml" in cats)

    # Per-row indicators (fundamentals)
    try:
        if p == "Alpha" and "Fund_from_Alpha" in results.columns and results["Fund_from_Alpha"].any():
            used_fund = True
        if p == "Tiingo" and "Fund_from_Tiingo" in results.columns and results["Fund_from_Tiingo"].any():
            used_fund = True
        if p == "Finnhub" and "Fund_from_Finnhub" in results.columns and results["Fund_from_Finnhub"].any():
            used_fund = True
        if p == "SimFin" and "Fund_from_SimFin" in results.columns and results["Fund_from_SimFin"].any():
            used_fund = True
        if p == "EODHD" and "Fund_from_EODHD" in results.columns and results["Fund_from_EODHD"].any():
            used_fund = True
        if p == "Alpha" and ("Price_Alpha" in results.columns and results["Price_Alpha"].notna().any()):
            used_price = True
    except Exception as e:
        logger.debug("unknown: %s", e)

    # Price sources from Source_List column
    try:
        if "Source_List" in results.columns:
            # join all source_list strings and search for provider label appearance
            all_sources = " ".join([str(x) for x in results["Source_List"].fillna("") if x])
            if meta["label"].lower() in all_sources.lower():
                used_price = True
    except Exception as e:
        logger.debug("unknown: %s", e)

    # OpenAI/ML usage heuristic
    try:
        if p == "OpenAI":
            if st.session_state.get("enable_openai_targets", False):
                # if any ML probabilities exist, consider ML used
                if "ML_Probability" in results.columns and results["ML_Probability"].notna().any():
                    used_ml = True
                # also consider session_usage
                if "OpenAI" in session_usage and "ml" in (session_usage.get("OpenAI") or set()):
                    used_ml = True
    except Exception as e:
        logger.debug("unknown: %s", e)

    provider_usage[p] = {
        "key_present": bool(key_present),
        "used_price": bool(used_price),
        "used_fundamentals": bool(used_fund),
        "used_ml": bool(used_ml),
        "implemented": bool(implemented),
    }

# Count used providers (any usage)
used_count = sum(1 for v in provider_usage.values() if v.get("used_price") or v.get("used_fundamentals") or v.get("used_ml"))

# Render data sources overview (single dynamic table, no HTML)
# Build comprehensive provider status map for overview table
openai_key = _env("OPENAI_API_KEY")
simfin_ok = bool(simfin_key)
marketstack_ok = bool(marketstack_key)
nasdaq_ok = bool(nasdaq_key)
eodhd_ok = bool(eodhd_key)

# Build provider status map from canonical list in SourcesOverview
status_lookup = {
    "Yahoo": True,
    "Alpha Vantage": bool(alpha_ok),
    "Finnhub": bool(finn_ok),
    "Polygon": bool(poly_ok),
    "Tiingo": bool(tiin_ok),
    "FMP": bool(fmp_ok),
    "OpenAI": bool(openai_key),
    "SimFin": bool(simfin_ok),
    "Marketstack": bool(marketstack_ok),
    "EODHD": bool(eodhd_ok),
    "Nasdaq": bool(nasdaq_ok),
}
provider_status_map = {name: {"ok": bool(status_lookup.get(name, False))} for name in SourcesOverview.PROVIDERS.keys()}

render_data_sources_overview(
    provider_status=provider_status_map,
    provider_usage=provider_usage,
    results=results
)

# Calculate target prices and dates WITH OPTIONAL OPENAI ENHANCEMENT
# (datetime and timedelta already imported at top of file)


@st.cache_data(ttl=3600)
def get_openai_target_prediction(
    ticker: str, current_price: float, fundamentals: dict, technicals: dict
) -> Optional[Tuple[float, int]]:
    """
    Use OpenAI to predict realistic target price AND holding period based on fundamentals and technicals.
    Returns (target_price, days_to_target) tuple or None if API unavailable or request fails.
    """
    if not OPENAI_AVAILABLE:
        return None

    openai_key = _env("OPENAI_API_KEY")
    if not openai_key:
        return None

    try:
        client = OpenAI(api_key=openai_key)

        # Build context from fundamentals
        fund_str = ", ".join(
            [
                f"{k}: {v}"
                for k, v in fundamentals.items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
        )
        tech_str = ", ".join(
            [
                f"{k}: {v:.2f}"
                for k, v in technicals.items()
                if isinstance(v, (int, float)) and np.isfinite(v)
            ]
        )

        prompt = (
            f"You are a financial analyst. Based on the following data for {ticker}:\n"
            f"Current Price: USD {current_price:.2f}\n"
            f"Fundamentals: {fund_str}\n"
            f"Technical Indicators: {tech_str}\n"
            "Provide TWO predictions as a JSON object:\n"
            "1. Target Price: realistic price target considering growth trends, valuation, momentum, and risk/reward\n"
            "2. Days to Target: estimated holding period in days to reach this target (typically 7-180 days based on momentum and catalysts)\n"
            "Return ONLY a JSON object with this exact format:\n"
            '{"target_price": <number>, "days_to_target": <integer>}\n'
            "JSON:"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )

        # Extract JSON from response
        answer = response.choices[0].message.content.strip()
        import re

        # Try to extract JSON from response
        json_match = re.search(r"\{[^}]+\}", answer)
        if json_match:
            data = json.loads(json_match.group(0))
            target = float(data.get("target_price", 0))
            days = int(data.get("days_to_target", 20))
            # Sanity checks
            if (
                current_price * 0.5 <= target <= current_price * 3.0
                and 7 <= days <= 365
            ):
                return (target, days)
    except Exception as e:
        logger.warning(f"OpenAI target prediction failed for {ticker}: {e}")

    return None


def calculate_targets(row):
    """Calculate entry price, target price, and target date based on ATR, RR, and optional OpenAI prediction"""
    ticker = row.get("Ticker", "")
    current_price = row.get("Unit_Price", row.get("Price_Yahoo", np.nan))
    atr = row.get("ATR", np.nan)
    rr = row.get("RewardRisk", np.nan)
    rsi = row.get("RSI", np.nan)
    momentum = row.get("Momentum_63d", np.nan)
    sector = row.get("Sector", "")
    # Use canonical ML probability when available
    ml_prob = row.get("ML_20d_Prob", row.get("ml_probability", 0.5))

    if np.isfinite(current_price) and current_price > 0:
        # Entry price: current - 0.5*ATR (wait for slight pullback)
        if np.isfinite(atr):
            entry_price = current_price - (0.5 * atr)
        else:
            entry_price = current_price * 0.98  # 2% below if no ATR

        # Calculate volatility factor for date variability
        atr_pct = (
            (atr / current_price) if (np.isfinite(atr) and current_price > 0) else 0.02
        )
        volatility_factor = np.clip(atr_pct / 0.03, 0.5, 2.5)  # 0.5x to 2.5x multiplier

        # Sector-based offset (defensive sectors slower, growth sectors faster)
        sector_offsets = {
            "Utilities": 1.3,
            "Consumer Defensive": 1.2,
            "Real Estate": 1.15,
            "Financials": 1.1,
            "Healthcare": 1.0,
            "Industrials": 0.95,
            "Energy": 0.9,
            "Consumer Cyclical": 0.85,
            "Technology": 0.75,
            "Communication Services": 0.8,
        }
        sector_mult = sector_offsets.get(sector, 1.0)

        # ML probability influence (higher confidence = shorter timeline)
        ml_mult = 1.0
        if isinstance(ml_prob, (int, float)) and np.isfinite(ml_prob):
            ml_mult = 1.2 - (ml_prob * 0.4)  # 0.5→1.0, 1.0→0.8 (high conf = faster)

        # Calculate fallback days from multiple factors (more dynamic)
        if np.isfinite(rr):
            # Base days from RR
            base_days = 20 + (rr * 10)

            # Adjust based on RSI: oversold (< 40) = faster, overbought (> 70) = slower
            if np.isfinite(rsi):
                if rsi < 40:
                    base_days *= 0.75  # Strong momentum, faster target
                elif rsi > 70:
                    base_days *= 1.3  # Overbought, slower target

            # Adjust based on momentum: strong momentum = faster
            if np.isfinite(momentum) and momentum > 0.05:
                base_days *= 0.9  # Strong uptrend, faster
            elif np.isfinite(momentum) and momentum < -0.05:
                base_days *= 1.2  # Weak trend, slower

            # Apply volatility, sector, and ML multipliers
            base_days *= volatility_factor * sector_mult * ml_mult

            # Add ticker-specific variance (hash-based to keep consistent per ticker)
            ticker_seed = sum(ord(c) for c in ticker) % 20
            base_days += ticker_seed

            days = int(min(180, max(14, base_days)))
        else:
            # Fallback: use volatility factor for diverse dates (30-180 days)
            base_days = 60 * volatility_factor * sector_mult * ml_mult
            ticker_seed = sum(ord(c) for c in ticker) % 30
            days = int(min(180, max(30, base_days + ticker_seed)))

        # Try OpenAI-enhanced target (returns both price and days)
        ai_result = None
        # Skip OpenAI calls in precomputed mode for faster loading
        skip_openai = st.session_state.get("skip_pipeline", False)
        if not skip_openai and st.session_state.get("enable_openai_targets", False):
            fundamentals = {
                "PE": row.get("PERatio", np.nan),
                "PB": row.get("PBRatio", np.nan),
                "ROE": row.get("ROE", np.nan),
                "Margin": row.get("ProfitMargin", np.nan),
                "RevenueGrowth": row.get("RevenueGrowthYoY", np.nan),
            }
            technicals = {
                "RSI": rsi,
                "Momentum_63d": momentum,
                "RewardRisk": rr,
                "ATR": atr,
            }
            try:
                ai_result = get_openai_target_prediction(
                    ticker, current_price, fundamentals, technicals
                )
            except Exception as e:
                logger.warning(f"OpenAI call failed for {ticker}: {e}")
                ai_result = None

        if ai_result is not None:
            # Use AI prediction for both target price AND timing
            target_price, days = ai_result
            target_source = "AI"
        elif np.isfinite(atr) and np.isfinite(rr):
            # Fallback to technical calculation: entry + (RR * ATR)
            base_target_pct = rr * (atr / current_price) if current_price > 0 else 0.10
            
            # REGIME-AWARE ADJUSTMENT
            reliability = row.get("Reliability_v2", row.get("reliability_pct", 50.0))
            risk_meter = row.get("risk_meter_v2", row.get("RiskMeter", 50.0))
            regime_data = st.session_state.get('market_regime', {"regime": "neutral", "confidence": 50})
            
            adjusted_target_pct, adjustment_explanation = adjust_target_for_regime(
                base_target_pct, 
                reliability, 
                risk_meter, 
                regime_data
            )
            
            target_price = entry_price * (1 + adjusted_target_pct)
            target_source = "AI"  # Mark as AI to show regime-aware calculation
            
            # Debug logging
            if CONFIG.get("DEBUG_MODE") and adjustment_explanation != "no adjustments":
                logger.debug(f"{ticker}: Target adjusted from {base_target_pct*100:.1f}% to {adjusted_target_pct*100:.1f}% ({adjustment_explanation})")
            
            # days already calculated above from RR + RSI + momentum
        else:
            # Conservative default: 10% above entry
            target_price = entry_price * 1.10
            target_source = "Default"
            # days already set to 30

        # Target date: today + holding horizon (now from AI or calculated)
        target_date = (datetime.datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

        return entry_price, target_price, target_date, target_source
    else:
        return current_price, np.nan, "N/A", "N/A"


"""Respect pipeline targets: only compute if missing."""
# Determine if targets already exist and are populated
target_cols = ["Entry_Price", "Target_Price", "Stop_Loss"]
has_target_cols = all(c in rec_df.columns for c in target_cols)
has_any_targets = False
if has_target_cols:
    try:
        vals = pd.to_numeric(rec_df["Target_Price"], errors="coerce")
        has_any_targets = bool(np.isfinite(vals).any())
    except Exception as exc:
        logger.debug(f"Target column check: {exc}")
        has_any_targets = False

if not has_any_targets:
    with st.spinner(f"🎯 Calculating targets for {len(rec_df)} stocks..."):
        if not rec_df.empty:
            entry_prices = []
            target_prices = []
            target_dates = []
            target_sources = []

            for idx, row in rec_df.iterrows():
                entry, target, date, source = calculate_targets(row)
                entry_prices.append(entry)
                target_prices.append(target)
                target_dates.append(date)
                target_sources.append(source)

            rec_df["Entry_Price"] = entry_prices
            rec_df["Target_Price"] = target_prices
            rec_df["Target_Date"] = target_dates
            rec_df["Target_Source"] = target_sources
    logger.info(f"Calculated targets for {len(rec_df)} stocks (UI fallback)")
else:
    logger.info(f"Using pipeline targets for {len(rec_df)} stocks")


# _calculate_rr_with_fallback removed — use calculate_rr directly from core.risk
_calculate_rr_with_fallback = calculate_rr  # alias kept for existing callsites


# Recalculate Reward/Risk only if pipeline RR not present; otherwise respect pipeline RR
if not rec_df.empty:
    rr_present = False
    try:
        if "RR" in rec_df.columns:
            rr_present = pd.to_numeric(rec_df["RR"], errors="coerce").notna().any()
        elif "RR_Ratio" in rec_df.columns or "RewardRisk" in rec_df.columns:
            base_rr = rec_df.get("RR_Ratio", rec_df.get("RewardRisk", pd.Series([np.nan] * len(rec_df), index=rec_df.index)))
            rr_present = pd.to_numeric(base_rr, errors="coerce").notna().any()
    except Exception as exc:
        logger.debug(f"RR presence check: {exc}")
        rr_present = False

if not rec_df.empty and not rr_present:
    # ATR value may be stored under 'ATR' or 'ATR14' or 'ATR_Price'
    def _compute_rr_row(r):
        entry = r.get("Entry_Price", r.get("Unit_Price", r.get("Price_Yahoo", np.nan)))
        target = r.get("Target_Price", np.nan)
        atr = r.get("ATR", r.get("ATR14", r.get("ATR14", np.nan)))
        # fallback ATR in price terms (ATR_Price)
        atr_price = r.get("ATR_Price", np.nan)
        rr_val = _calculate_rr_with_fallback(entry, target, atr, fallback_price=atr_price)
        return rr_val

    rec_df["RewardRisk"] = rec_df.apply(
        lambda r: (
            (v := _compute_rr_row(r)),
            round(v, 2) if np.isfinite(v) else np.nan,
        )[-1],
        axis=1,
    )
    # Also update RR_Ratio alias used in classification
    rec_df["RR_Ratio"] = rec_df["RewardRisk"]

    # Propagate recalculated RewardRisk back into the main `results` frame so
    # downstream fields (rr alias, rr_score_v2, conviction) use the updated values.
    try:
        rr_map = rec_df.set_index("Ticker")["RewardRisk"].to_dict()
        results["RewardRisk"] = (
            results["Ticker"].map(rr_map).fillna(results.get("RewardRisk", np.nan))
        )
        results["RR_Ratio"] = results["RewardRisk"]
        results["rr"] = results["RewardRisk"]

        # Recompute normalized rr_score_v2 (0-100) for any updated rr values
        # Recompute rr_score_v2 and rr_band using evaluate_rr_unified (tiered, consistent)
        from core.scoring_engine import evaluate_rr_unified

        def _rr_eval_local(row_rr):
            try:
                score, ratio_adj, band = evaluate_rr_unified(float(row_rr))
                return pd.Series({"rr_score_v2": score, "rr_band": band})
            except Exception as exc:
                logger.debug(f"RR eval: {exc}")
                return pd.Series({"rr_score_v2": np.nan, "rr_band": "N/A"})

        rr_results = results["rr"].apply(_rr_eval_local)
        results["rr_score_v2"] = rr_results["rr_score_v2"]
        results["rr_band"] = rr_results["rr_band"]
        # Keep canonical RR_Score alias in sync if present
        results["RR_Score"] = results["rr_score_v2"].copy()
        results["RR"] = results["rr"].copy()

        # Sync updated RR fields back to rec_df (recommendation dataframe)
        if not rec_df.empty:
            for tkr_idx in rec_df.index:
                tkr_val = rec_df.at[tkr_idx, "Ticker"]
                if tkr_val in rr_map:
                    rec_df.at[tkr_idx, "rr"] = rr_map[tkr_val]
                    rec_df.at[tkr_idx, "RewardRisk"] = rr_map[tkr_val]
                    rec_df.at[tkr_idx, "RR_Ratio"] = rr_map[tkr_val]
            # Recompute rr_band for rec_df
            rec_df["rr_band"] = rec_df["rr"].apply(_rr_eval_local)
    except Exception as e:
        logger.warning(f"RR sync failed: {e}")

# Add new export fields for 2025 improvements
if not rec_df.empty:
    # Market regime info
    regime_data = st.session_state.get('market_regime', {"regime": "neutral", "confidence": 50})
    rec_df["Market_Regime"] = regime_data.get("regime", "neutral")
    rec_df["Regime_Confidence"] = regime_data.get("confidence", 50)
    
    # Reliability band (High/Medium/Low based on reliability score)
    def _get_reliability_band(reliability_val):
        if pd.notna(reliability_val) and isinstance(reliability_val, (int, float)):
            if reliability_val >= 75:
                return "High"
            elif reliability_val >= 40:
                return "Medium"
            else:
                return "Low"
        return "Unknown"
    
    # Get reliability values, defaulting to 50 if columns don't exist
    reliability_series = rec_df.get("Reliability_v2", rec_df.get("reliability_pct", pd.Series([50] * len(rec_df), index=rec_df.index)))
    rec_df["Reliability_Band"] = reliability_series.apply(_get_reliability_band)
    
    # Reliability components summary
    def _get_reliability_components(row):
        fund_rel = row.get("Fundamental_Reliability_v2", 0)
        price_rel = row.get("Price_Reliability_v2", 0)
        fund_sources = row.get("fund_sources_used_v2", 0)
        price_sources = row.get("price_sources_used_v2", 0)
        return f"F:{fund_rel:.0f}%(n={fund_sources}),P:{price_rel:.0f}%(n={price_sources})"
    
    rec_df["Reliability_Components"] = rec_df.apply(_get_reliability_components, axis=1)
    
    # Risk band (based on risk_meter_v2)
    rec_df["Risk_Band"] = rec_df.get("risk_band", "Unknown")
    
    # Fundamental coverage percentage
    if "Fund_Coverage_Pct" in rec_df.columns:
        rec_df["Fund_Coverage_Pct"] = rec_df["Fund_Coverage_Pct"].fillna(0)
    else:
        rec_df["Fund_Coverage_Pct"] = 0
    
    # Volatility penalty (from reliability calculation)
    atr_price = rec_df.get("ATR_Price")
    if atr_price is not None:
        rec_df["Volatility_Penalty"] = atr_price.apply(
            lambda x: "High" if x > 0.08 else ("Moderate" if x > 0.04 else "Low") if pd.notna(x) else "Unknown"
        )
    else:
        rec_df["Volatility_Penalty"] = "Unknown"
    
    # Safety caps applied (if overall score was capped)
    rec_df["Safety_Caps_Applied"] = "No"  # Placeholder - would need to track during scoring
    
    # Debug logging for sample stocks
    if CONFIG.get("DEBUG_MODE") and len(rec_df) > 0:
        sample_tickers = rec_df["Ticker"].head(3).tolist()
        for ticker in sample_tickers:
            row = rec_df[rec_df["Ticker"] == ticker].iloc[0]
            def _fmt_num(val, fmt):
                try:
                    return format(float(val), fmt) if val is not None and str(val) not in ("N/A", "nan") else str(val)
                except Exception as exc:
                    logger.debug(f"Debug fmt: {exc}")
                    return str(val)
            logger.info(
                f"DEBUG: {ticker} Breakdown:\n"
                f"  Overall Score: {row.get('overall_score_pretty', row.get('Score', 'N/A'))}\n"
                f"  Fundamentals: {_fmt_num(row.get('Fundamental_S', 'N/A'), '.0f')} (coverage: {_fmt_num(row.get('Fund_Coverage_Pct', 0), '.1f')}%)\n"
                f"  Technical: {_fmt_num(row.get('Technical_S', 'N/A'), '.0f')}\n"
                f"  RR: {_fmt_num(row.get('rr_score_v2', 'N/A'), '.0f')} (ratio: {_fmt_num(row.get('rr', 'N/A'), '.2f')})\n"
                f"  Reliability: {_fmt_num(row.get('Reliability_v2', 'N/A'), '.0f')} ({row.get('Reliability_Band', 'N/A')})\n"
                f"  Risk: {_fmt_num(row.get('risk_meter_v2', 'N/A'), '.0f')} ({row.get('risk_band', 'N/A')})\n"
                f"  Classification: {row.get('Risk_Level', 'N/A')}\n"
                f"  Market Regime: {regime_data.get('regime', 'N/A')} ({regime_data.get('confidence', 0)}%)"
            )

# CSS now loaded from design_system.py - no need for separate CARD_CSS



if rec_df.empty:
    st.info("No strong signals found in this scan.")
else:
    # Split into Core and Speculative
    if "Risk_Level" in rec_df.columns:
        levels = rec_df["Risk_Level"].astype(str).str.lower()
        core_df = rec_df[levels == "core"].copy()
        spec_df = rec_df[levels == "speculative"].copy()
    else:
        core_df = rec_df.copy()
        spec_df = pd.DataFrame()

    # Summary info
    total_candidates = len(core_df) + len(spec_df)
    # UI does not compute or display allocation-based funding metrics
    st.info(
        f"📊 Showing {total_candidates} candidates — {len(core_df)} Core, {len(spec_df)} Speculative"
    )

    # Legend for ML badge thresholds
    st.caption("ML badge legend: 🟢 >60% · 🟡 40–60% · 🔴 <40%")

    # Determine score label based on schema
    score_label = "FinalScore_20d" if "FinalScore_20d" in rec_df.columns else "Score"

    # Small helpers for compact card rendering (headline vs details)
    def _to_float(val) -> float:
        try:
            # Coerce common non-numeric placeholders safely
            if val in (None, "", "N/A", "nan"):
                return np.nan
            return float(val)
        except Exception as exc:
            logger.debug(f"_to_float coerce: {exc}")
            return np.nan


    def _normalize_prob(p):
        try:
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return None
            pv = float(p)
            if not np.isfinite(pv):
                return None
            if pv > 1 and pv <= 100:
                pv = pv / 100.0
            elif pv > 100:
                return '—'
            pv = max(0.0, min(1.0, pv))
            return pv
        except Exception as exc:
            logger.debug(f"_normalize_prob: {exc}")
            return None

    def _ml_badge(p) -> str:
        norm = _normalize_prob(p)
        if norm is None:
            return "—"
        if norm == '—':
            return '—'
        if norm > 0.60:
            return f"🟢 {norm*100:.0f}%"
        if norm >= 0.40:
            return f"🟡 {norm*100:.0f}%"
        return f"🔴 {norm*100:.0f}%"

    def _get_ml_prob_from_row(r):
        for k in ["ML_20d_Prob_live_v3","ML_20d_Prob","ML_20d_Prob_raw","ML_Probability"]:
            v = r.get(k, np.nan)
            if v is not None and pd.notna(v):
                try:
                    return float(v)
                except Exception as exc:
                    logger.debug(f"ML prob parse for key {k}: {exc}")
                    continue
        return np.nan

    def _risk_class(row: pd.Series) -> str:
        rc = row.get("RiskClass")
        if isinstance(rc, str) and rc:
            return rc
        # Fallback to legacy Risk_Level
        rl = str(row.get("Risk_Level", "speculative")).lower()
        return "CORE" if rl == "core" else ("SPEC" if rl else "SPEC")

    def _headline_story(row: pd.Series) -> str:
        # Headline fields: FinalScore_20d, ML_20d_Prob, RiskClass, RR, ReliabilityScore
        fund = _to_float(row.get("FundamentalScore", row.get("Fundamental_S", np.nan)))
        mom = _to_float(row.get("MomentumScore", row.get("TechScore_20d", np.nan)))
        rr = _to_float(row.get("RR", row.get("RR_Ratio", row.get("RewardRisk", np.nan))))
        rel = _to_float(row.get("ReliabilityScore", row.get("Reliability_Score", row.get("Reliability_v2", np.nan))))

        parts = []
        # Quality / fundamentals
        if isinstance(fund, float) and np.isfinite(fund):
            if fund >= 70:
                parts.append("Quality business")
            elif fund >= 50:
                parts.append("Decent fundamentals")
            else:
                parts.append("Weak fundamentals")
        # Momentum
        if isinstance(mom, float) and np.isfinite(mom):
            if mom >= 70:
                parts.append("strong momentum")
            elif mom >= 50:
                parts.append("moderate momentum")
            else:
                parts.append("weak momentum")
        # RR
        if isinstance(rr, float) and np.isfinite(rr):
            if rr >= 2.5:
                parts.append("excellent RR")
            elif rr >= 1.5:
                parts.append("good RR")
            else:
                parts.append("poor RR")
        # Reliability
        if isinstance(rel, float) and np.isfinite(rel):
            # Normalize legacy 0-1
            rel_val = rel
            if rel_val <= 1.0:
                rel_val *= 100.0
            if rel_val >= 75:
                parts.append("high data reliability")
            elif rel_val >= 40:
                parts.append("medium reliability")
            else:
                parts.append("low reliability")
        return ", ".join(parts[:4])

    def _fmt_num(val, fmt, na="N/A"):
        try:
            v = _to_float(val)
            return format(v, fmt) if isinstance(v, float) and np.isfinite(v) else na
        except Exception as exc:
            logger.debug(f"_fmt_num: {exc}")
            return na

    # UI displays exactly what the pipeline provided (no extra slicing)
    st.write(f"Showing {len(rec_df)} candidates")

    show_ml_debug = st.checkbox("Show ML Debug", value=False)

    # Core recommendations
    if not core_df.empty:
        st.markdown("### 🛡️ Core Stocks — Lower Relative Risk")
        st.caption(f"Showing {len(core_df)} candidates")

        @st.cache_data(ttl=3600)
        def _fallback_sector_yf(ticker: str) -> str:
            try:
                info = yf.Ticker(ticker).info
                sec = info.get('sector') or info.get('industry')
                return sec or 'Unknown'
            except Exception as exc:
                logger.debug(f"yfinance sector lookup failed for {ticker}: {exc}")
                return 'Unknown'

        for idx, r in core_df.iterrows():
            final_score = r.get("FinalScore_20d", r.get("Score", np.nan))
            ml_prob = _get_ml_prob_from_row(r)
            if show_ml_debug and idx < 10:
                ml_prob_live_v3 = r.get("ML_20d_Prob_live_v3", None)
                ml_prob_raw = r.get("ML_20d_Prob_raw", None)
                ml_prob_prob = r.get("ML_Probability", None)
                ml_rank_20d = r.get("ML_rank_20d", None)
                st.sidebar.write(f"[ML BADGE DEBUG] Ticker: {r.get('Ticker','N/A')} | ml_prob_final: {ml_prob} | ML_20d_Prob: {r.get('ML_20d_Prob', None)} | ML_Probability: {ml_prob_prob} | ML_20d_Prob_live_v3: {ml_prob_live_v3} | ML_20d_Prob_raw: {ml_prob_raw} | ML_rank_20d: {ml_rank_20d}")
            # (move the rest of the card rendering here, using ml_prob as before)
            ticker = r.get("Ticker", "N/A")
            sector = r.get("Sector", r.get("sector", "Unknown"))
            company = r.get("shortName", r.get("Company", r.get("Name", "")))
            if sector in (None, "", "Unknown") and isinstance(ticker, str) and ticker and ticker != "N/A":
                sector = _fallback_sector_yf(ticker)


            final_score = r.get("FinalScore_20d", r.get("Score", np.nan))
            # --- DEBUG: Print ML badge source info for first 10 rows ---
            ml_prob = r.get("ML_20d_Prob", r.get("ML_Probability", np.nan))
            if idx < 10:
                ml_prob_live_v3 = r.get("ML_20d_Prob_live_v3", None)
                ml_prob_raw = r.get("ML_20d_Prob_raw", None)
                ml_prob_prob = r.get("ML_Probability", None)
                ml_rank_20d = r.get("ML_rank_20d", None)
                print(f"[ML BADGE DEBUG] Ticker: {r.get('Ticker','N/A')} | ml_prob: {ml_prob} | Source: 'ML_20d_Prob'→'ML_Probability' | ML_20d_Prob: {r.get('ML_20d_Prob', None)} | ML_Probability: {ml_prob_prob} | ML_20d_Prob_live_v3: {ml_prob_live_v3} | ML_20d_Prob_raw: {ml_prob_raw} | ML_rank_20d: {ml_rank_20d}")
            rr = r.get("RR", r.get("RR_Ratio", r.get("RewardRisk", np.nan)))
            rel = r.get("ReliabilityScore", r.get("Reliability_Score", r.get("Reliability_v2", np.nan)))
            risk_c = _risk_class(r)

            # Headline compact view
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
                with c1:
                    title = ticker if not company else f"{ticker} · {company}"
                    st.subheader(title)
                    st.caption(f"Sector: {sector}")
                with c2:
                    st.metric(score_label, _fmt_num(final_score, '.0f'))
                with c3:
                    st.metric("Risk", risk_c)
                with c4:
                    st.metric("ML", _ml_badge(ml_prob))
                with c5:
                    rr_val = _to_float(rr)
                    rr_fmt = f"{rr_val:.2f}x" if isinstance(rr_val, float) and np.isfinite(rr_val) else "N/A"
                    st.metric("R/R", rr_fmt)

                # Short storyline
                storyline = _headline_story(r)
                if storyline:
                    st.caption(storyline)

                # Details expander: deep-dive fields
                with st.expander("Details", expanded=False):
                    # Technical indicators
                    t1, t2, t3, t4 = st.columns(4)
                    with t1:
                        st.text("RSI")
                        st.code(_fmt_num(r.get('RSI', np.nan), '.1f'))
                    with t2:
                        atrv = r.get('ATR_Price', r.get('ATR_Pct', np.nan))
                        st.text("ATR/Price")
                        st.code(_fmt_num(atrv, '.3f'))
                    with t3:
                        st.text("Momentum (Tech)")
                        momv = r.get('MomentumScore', r.get('TechScore_20d', np.nan))
                        st.code(_fmt_num(momv, '.0f'))
                    with t4:
                        st.text("ML Prob")
                        st.code(_fmt_num(ml_prob, '.3f'))

                    # Fundamentals breakdown
                    f1, f2, f3, f4 = st.columns(4)
                    with f1:
                        st.text("Fundamental")
                        fsv = r.get('FundamentalScore', r.get('Fundamental_S', np.nan))
                        st.code(_fmt_num(fsv, '.0f'))
                    with f2:
                        st.text("Quality/Growth")
                        q_score = r.get('Quality_Score_F', np.nan)
                        g_score = r.get('Growth_Score_F', np.nan)
                        st.code(f"{_fmt_num(q_score, '.0f')} / {_fmt_num(g_score, '.0f')}")
                    with f3:
                        st.text("Valuation")
                        st.code(_fmt_num(r.get('Valuation_Score_F', np.nan), '.0f'))
                    with f4:
                        st.text("Leverage (D/E)")
                        st.code(_fmt_num(r.get('DE_f', r.get('debt_to_equity', np.nan)), '.2f'))

                    # Reliability breakdown
                    rel1, rel2, rel3, rel4 = st.columns(4)
                    with rel1:
                        st.text("Reliability")
                        st.code(_fmt_num(rel, '.0f'))
                    with rel2:
                        st.text("Fund sources")
                        st.code(_fmt_num(r.get('Fundamental_Sources_Count', r.get('fund_sources_used_v2', np.nan)), '.0f'))
                    with rel3:
                        st.text("Price sources")
                        st.code(_fmt_num(r.get('Price_Sources_Count', r.get('price_sources_used_v2', np.nan)), '.0f'))
                    with rel4:
                        st.text("Price STD")
                        st.code(_fmt_num(r.get('Price_STD', r.get('price_std', np.nan)), '.2f'))

    # Speculative candidates
    if not spec_df.empty:
        st.markdown("### 🚀 Speculative Stocks — Higher Risk/Reward")
        st.caption(f"Showing {len(spec_df)} candidates")
        for idx, r in spec_df.iterrows():
            final_score = r.get("FinalScore_20d", r.get("Score", np.nan))
            ml_prob = _get_ml_prob_from_row(r)
            rr = r.get("RR", r.get("RR_Ratio", r.get("RewardRisk", np.nan)))
            rel = r.get("ReliabilityScore", r.get("Reliability_Score", r.get("Reliability_v2", np.nan)))
            if show_ml_debug and idx < 10:
                ml_prob_live_v3 = r.get("ML_20d_Prob_live_v3", None)
                ml_prob_raw = r.get("ML_20d_Prob_raw", None)
                ml_prob_prob = r.get("ML_Probability", None)
                ml_rank_20d = r.get("ML_rank_20d", None)
                st.sidebar.write(f"[ML BADGE DEBUG] Ticker: {r.get('Ticker','N/A')} | ml_prob_final: {ml_prob} | ML_20d_Prob: {r.get('ML_20d_Prob', None)} | ML_Probability: {ml_prob_prob} | ML_20d_Prob_live_v3: {ml_prob_live_v3} | ML_20d_Prob_raw: {ml_prob_raw} | ML_rank_20d: {ml_rank_20d}")
            # (move the rest of the card rendering here, using ml_prob as before)
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.text("RSI")
                st.code(_fmt_num(r.get('RSI', np.nan), '.1f'))
            with t2:
                atrv = r.get('ATR_Price', r.get('ATR_Pct', np.nan))
                st.text("ATR/Price")
                st.code(_fmt_num(atrv, '.3f'))
            with t3:
                st.text("Momentum (Tech)")
                momv = r.get('MomentumScore', r.get('TechScore_20d', np.nan))
                st.code(_fmt_num(momv, '.0f'))
            with t4:
                st.text("ML Prob")
                st.code(_fmt_num(ml_prob, '.3f'))

            f1, f2, f3, f4 = st.columns(4)
            with f1:
                st.text("Fundamental")
                fsv = r.get('FundamentalScore', r.get('Fundamental_S', np.nan))
                st.code(_fmt_num(fsv, '.0f'))
            with f2:
                st.text("Quality/Growth")
                q_score = r.get('Quality_Score_F', np.nan)
                g_score = r.get('Growth_Score_F', np.nan)
                st.code(f"{_fmt_num(q_score, '.0f')} / {_fmt_num(g_score, '.0f')}")
            with f3:
                st.text("Valuation")
                st.code(_fmt_num(r.get('Valuation_Score_F', np.nan), '.0f'))
            with f4:
                st.text("Leverage (D/E)")

                st.code(_fmt_num(r.get('DE_f', r.get('debt_to_equity', np.nan)), '.2f'))

            rel1, rel2, rel3, rel4 = st.columns(4)
            with rel1:
                st.text("Reliability")
                st.code(_fmt_num(rel, '.0f'))
            with rel2:
                st.text("Fund sources")
                st.code(_fmt_num(r.get('Fundamental_Sources_Count', r.get('fund_sources_used_v2', np.nan)), '.0f'))
            with rel3:
                st.text("Price sources")
                st.code(_fmt_num(r.get('Price_Sources_Count', r.get('price_sources_used_v2', np.nan)), '.0f'))
            with rel4:
                st.text("Price STD")
                st.code(_fmt_num(r.get('Price_STD', r.get('price_std', np.nan)), '.2f'))

    # Export section (single, unified)
show_order = [
    "Ticker",
    "TechScore_20d",
    "ML_20d_Prob",
    "ML_Features_Fallback",
    "FinalScore_20d",
    "FinalScore",
    "overall_score_20d",  # 20-day conviction score
    "Sector",
    "Risk Level",
    "Data Quality",
    "ML Probability",
    "ML Confidence",
    "Confidence Level",
    "Reliability Score",
    "reliability_v2",  # raw V2 reliability (0-100)
    "Fund Reliability",
    "Price Reliability",
    "Fundamental_Reliability_v2",  # New multi-source fund reliability
    "Price_Reliability_v2",  # New multi-source price reliability
    "fund_sources_used_v2",  # Number of fundamental sources
    "price_sources_used_v2",  # Number of price sources
    "fund_disagreement_score_v2",  # Disagreement between sources
    "price_variance_score_v2",  # Price variance ratio
    "fund_coverage_pct_v2",  # Field coverage percentage
    "Price_Mean_v2",  # Multi-source mean price
    "Price_STD_v2",  # Multi-source price std deviation
    "Sources Count",
    "risk_gate_status_v2",  # blocked / severely_reduced / reduced / full
    "risk_gate_penalty_v2",
    "reliability_score_v2",
    "Score",
    "overall_score",
    "overall_score_raw",  # True model score (internal logic)
    "overall_score_pretty",  # Display score (60-90 range)
    "reliability_pct",  # Reliability percentage (0-100)
    "reliability_band",  # High/Medium/Low
    "risk_band",  # Low/Medium/High/Very High
    "ml_conf_band",  # ML confidence band (Low/Medium/High)
    "fund_score",
    "tech_score",
    "rr_score",
    "reliability_score",
    "ml_delta",
    "conviction_v2_base",
    "conviction_v2_final",
    "ml_boost_v2",
    "reward_risk_v2",
    # Human-friendly V2 export labels (also include raw keys above for robustness)
    "Conviction v2 Base",
    "Conviction v2 Final",
    "Reliability Score v2",
    "Risk Gate Status v2",
    "Risk Gate Reason v2",
    "Reward/Risk v2",
    "Buy Amount v2",
    "Shares to Buy v2",
    "Fund Sources Used",
    "Price Sources Used",
    "Fund Disagreement Score",
    "Price Variance Score",
    "Quality Score",
    "Average Price",
    "Unit Price (calc)",
    "Buy Amount ($)",
    "buy_amount_v2",  # v2 raw dollar amount pre-hebrew mapping
    "Shares to Buy",
    "Leftover ($)",
    "Price Sources",
    "Price Marketstack",
    "Price NasdaqDL",
    "Price EODHD",
    "Tech Score",
    "Fundamental Score",
    "RSI",
    "Blended_RS_Value",
    "Tightness_Ratio",
    "Growth_Boost",
    "MarketCap_B",
    "Market vs (3M) (%)",
    "Volume Surge (x)",
    "MA Aligned",
    "Risk/Reward",
    "Momentum Consistency (%)",
    "Near 52w High (%)",
    "Volume / 20d Avg",
    "Overextension vs MA_L",
    "ATR/Price",
    "Reward/Risk (≈R)",
    "Beta",
    "Holding Horizon",
    "P/E",
    "P/S",
    "ROE",
    "ROIC",
    "Margin",
    "Debt/Equity",
    "Revenue YoY",
    "EPS YoY",
    # NEW FIELDS (2025 improvements)
    "Market_Regime",
    "Regime_Confidence",
    "Reliability_Band",
    "Reliability_Components",
    "Risk_Band",
    "Fund_Coverage_Pct",
    "Volatility_Penalty",
    "Safety_Caps_Applied",
    # ML live_v3 debug columns
    "ML_20d_Prob_live_v3",
    "ML_20d_Prob_raw",
    "TechScore_20d_v2",
    "ATR_Pct_percentile",
    "Price_As_Of_Date",
    "ReliabilityFactor",
]
# --- Hebrew column mapping for CSV export ---
hebrew_cols = {
    "Ticker": "סימול",
    "Sector": "סקטור",
    "Risk Level": "רמת סיכון",
    "Data Quality": "איכות נתונים",
    "ML Probability": "הסתברות ML",
    "ML Confidence": "רמת ביטחון ML",
    "ML_Features_Fallback": "מאפייני ML חלופיים",
    "Confidence Level": "רמת ביטחון",
    "Reliability Score": "ציון אמינות",
    "Score": "ציון כולל",
    "Buy Amount v2": "סכום קנייה ($)",
    "Shares to Buy v2": "מניות לקנייה",
    "Buy Amount ($)": "סכום קנייה ($)",
    "Shares to Buy": "מניות לקנייה",
    "Leftover ($)": "עודף ($)",
    "Entry_Price": "מחיר כניסה",
    "Target_Price": "מחיר יעד",
    "RR": "סיכון/סיכוי",
    "Reliability_Band": "רצועת אמינות",
    "Market_Regime": "מצב שוק",
    "Regime_Confidence": "ביטחון מצב שוק",
    "Fundamental_S": "ציון יסודות",
    "Technical_S": "ציון טכני",
    "overall_score_20d": "ציון 20 יום",
    "Blended_RS_Value": "RS משולב",
    "Tightness_Ratio": "יחס הידוק",
    "Growth_Boost": "בוסט צמיחה",
    "MarketCap_B": "שווי שוק (מיליארדים)",
    # Add more mappings as needed for full export
}

# Ensure aliases for UI columns
try:
    if "Blended_RS_Value" not in rec_df.columns:
        rec_df["Blended_RS_Value"] = rec_df.get("market_rs_blended", rec_df.get("RS_63d", np.nan))
    # Tightness alias
    if "Tightness_Ratio" not in rec_df.columns:
        rec_df["Tightness_Ratio"] = rec_df.get("tightness_ratio", rec_df.get("VCP_Tightness", np.nan))
    # Growth boost alias
    if "Growth_Boost" not in rec_df.columns:
        rec_df["Growth_Boost"] = rec_df.get("growth_boost", rec_df.get("growth_accel_bonus", np.nan))
    # MarketCap in Billions
    if "MarketCap_B" not in rec_df.columns:
        mc = pd.to_numeric(rec_df.get("MarketCap", rec_df.get("market_cap", np.nan)), errors="coerce")
        rec_df["MarketCap_B"] = (mc / 1e9).where(mc.notna(), np.nan)
except Exception as e:
    logger.debug("unknown: %s", e)

csv_df = rec_df.rename(columns=hebrew_cols)


# Ensure column names unique after rename (pandas JSON export requires uniqueness)
def _make_unique(names):
    counts = {}
    out = []
    for n in names:
        if n not in counts:
            counts[n] = 1
            out.append(n)
        else:
            counts[n] += 1
            out.append(f"{n}_{counts[n]}")
    return out


csv_df.columns = _make_unique(list(csv_df.columns))
# Build unique ordered columns for export referencing updated names
cols_for_export = []
seen_cols = set()
for c in show_order:
    # select first matching column (since duplicates now suffixed)
    matches = [col for col in csv_df.columns if col == c or col.startswith(f"{c}_")]
    if matches:
        first = matches[0]
        if first not in seen_cols:
            cols_for_export.append(first)
            seen_cols.add(first)

# Deduplicate show_order for display to avoid Arrow errors
show_order_unique = []
seen_show = set()
for c in show_order:
    if c not in seen_show:
        show_order_unique.append(c)
        seen_show.add(c)

lean_export_fields = [
    c for c in cols_for_export if c in {
        "Ticker","Score","Overall_Rank","Rank","Entry_Price","Target_Price","RR","Risk_Level","Reliability_Band","Market_Regime","Regime_Confidence","Fundamental_S","Technical_S"
    }
]
full_export_fields = cols_for_export  # preserve full order

# Sidebar checkbox stored earlier under dev options (create if missing)
use_full_export = bool(st.session_state.get("use_full_export", False))
export_fields = full_export_fields if use_full_export else lean_export_fields
csv_bytes = csv_df[export_fields].to_csv(index=False).encode("utf-8-sig")

# Download buttons side by side
col_csv, col_json = st.columns(2)
with col_csv:
    st.download_button(
        "📥 Download Results (CSV)",
        data=csv_bytes,
        file_name=f"stock_scout_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
with col_json:
    # JSON export for API/automation
    json_data = csv_df[export_fields].to_json(
        orient="records", force_ascii=False, indent=2
    )
    st.download_button(
        "📊 Download Results (JSON)",
        data=json_data.encode("utf-8"),
        file_name=f"stock_scout_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

# Force button text color override with inline styles
st.markdown(
    """
    <style>
    button, button p, button div, button span {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.dataframe(
    csv_df[[c for c in show_order_unique if c in csv_df.columns]],
    width='stretch',
    hide_index=True,
)

# ==================== Quick chart ====================
st.subheader("🔍 Chart Ticker from Results")
# Choose a safe ticker column for charting
_ticker_candidates = ["Ticker", "symbol", "Symbol", "ticker"]
_tcol = None
for _tc in _ticker_candidates:
    if _tc in rec_df.columns:
        _tcol = _tc
        break

if _tcol is None or rec_df.empty:
    st.caption("No ticker column available for charting.")
else:
    choices = ["(Select)"] + rec_df[_tcol].astype(str).tolist()
    choice = st.selectbox("Select ticker", choices, index=0)
    if choice and choice != "(Select)" and isinstance(data_map, dict) and choice in data_map:
        dfv = data_map[choice].copy()
        dfv["MA_S"] = dfv["Close"].rolling(int(CONFIG["MA_SHORT"])).mean()
        dfv["MA_L"] = dfv["Close"].rolling(int(CONFIG["MA_LONG"])).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=dfv.index,
                open=dfv["Open"],
                high=dfv["High"],
                low=dfv["Low"],
                close=dfv["Close"],
                name="Price",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dfv.index,
                y=dfv["MA_S"],
                mode="lines",
                name=f"MA{int(CONFIG['MA_SHORT'])}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dfv.index, y=dfv["MA_L"], mode="lines", name=f"MA{int(CONFIG['MA_LONG'])}"
            )
        )
        fig.update_layout(
            height=480, xaxis_rangeslider_visible=False, legend_orientation="h"
        )
        st.plotly_chart(fig, width='stretch')
        dfv["RSI"] = rsi(dfv["Close"], 14)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dfv.index, y=dfv["RSI"], mode="lines", name="RSI14"))
        fig2.add_hrect(
            y0=CONFIG["RSI_BOUNDS"][0],
            y1=CONFIG["RSI_BOUNDS"][1],
            fillcolor="LightGreen",
            opacity=0.2,
            line_width=0,
        )
        fig2.update_layout(height=220, legend_orientation="h")
        st.plotly_chart(fig2, width='stretch')

# ==================== Notes ====================
with st.expander("ℹ️ Methodology (Summary)"):
        st.markdown(
"""
- **Historical Data**: Yahoo Finance (`yfinance`). Price verification: Alpha Vantage, Finnhub, Polygon, Tiingo.
- **Technical Scoring**: Moving averages, momentum (1/3/6 months with Sigmoid), RSI range, Near-High bell curve, 
    Overextension vs MA_L, Pullback detection, ATR/Price ratio, Reward/Risk, MACD/ADX.
- **Fundamental Scoring**: Alpha Vantage OVERVIEW with Finnhub fallback (Growth/Quality/Valuation metrics + high leverage penalty).
- **Filters & Rules**: Minimum dollar-volume, ATR/Price and Overextension caps (hard disqualification), earnings blackout period, 
    beta filter, sector diversification caps, and budget allocation with maximum position size limits.
"""
        )
