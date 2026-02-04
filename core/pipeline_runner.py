import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any, Tuple
import os
import requests
import yfinance as yf
from pathlib import Path
from threading import Lock

from core.config import get_config
from core.scoring import build_technical_indicators, compute_fundamental_score_with_breakdown
from core.filters import (
    apply_technical_filters,
    compute_advanced_score,
    should_reject_ticker,
    fetch_benchmark_data,
)
from core.data import (
    fetch_fundamentals_batch,
    aggregate_fundamentals,
    fetch_price_multi_source,
    get_fundamentals_safe,
)
from core.allocation import allocate_budget
from core.classifier import apply_classification
from core.unified_logic import (
    compute_recommendation_scores,
    compute_final_score_with_patterns,
    compute_big_winner_signal_20d,
)
from core.pattern_matcher import PatternMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.market_context import get_benchmark_series, compute_relative_strength_vs_spy
from core.telemetry import Telemetry
from advanced_filters import compute_relative_strength
from core.scoring_config import SIGNAL_MIN_SCORE, ML_PROB_THRESHOLD, TOP_SIGNAL_K

# For backward compatibility with code that checks ML availability
from core.ml_20d_inference import ML_20D_AVAILABLE, get_ml_health_meta

# Contracts (minimal types used for wrapper output)
from core.contracts import (
    ScanResult,
    ScanMetadata,
    Diagnostics,
    CoverageMetrics,
    FallbackCounts,
    DisagreementStats,
    MLStatus,
    MarketContextStatus,
    Recommendation,
    Targets,
    RiskSizing,
)

# V2 Bridge (ML + Risk integration)
try:
    from core.bridge import analyze_row_with_bridge
    V2_BRIDGE_AVAILABLE = True
except ImportError:
    V2_BRIDGE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Public diagnostic: last provider used for universe construction
LAST_UNIVERSE_PROVIDER: str = "Unknown"

# --- Fallback tracking (for meta) ---
# Tracks whether any legacy/bridge fallback occurred during the run
_LEGACY_FALLBACK_USED: bool = False
_LEGACY_FALLBACK_REASONS: List[str] = []
_LEGACY_LOCK: Lock = Lock()

def _record_legacy_fallback(reason: str) -> None:
    """Record and log fallback from V2 Bridge to legacy scoring.
    
    IMPORTANT: This should be visible to users - silent fallbacks hide bugs.
    """
    try:
        with _LEGACY_LOCK:
            global _LEGACY_FALLBACK_USED, _LEGACY_FALLBACK_REASONS
            _LEGACY_FALLBACK_USED = True
            if reason:
                _LEGACY_FALLBACK_REASONS.append(str(reason))
        # Log prominently so it's not hidden
        logger.warning(f"⚠️ FALLBACK TO LEGACY SCORING: {reason}")
        logger.warning("   ML/Risk Bridge failed - using older scoring logic. Results may differ.")
    except Exception:
        # Best-effort; do not raise
        pass


def get_fallback_status() -> dict:
    """Get status of fallback usage for current run - exposed for UI/API."""
    with _LEGACY_LOCK:
        return {
            "fallback_used": _LEGACY_FALLBACK_USED,
            "fallback_count": len(_LEGACY_FALLBACK_REASONS),
            "reasons": list(_LEGACY_FALLBACK_REASONS[-10:]),  # Last 10 reasons
        }

# --- Helper Functions ---

def preflight_check() -> None:
    """Assert that critical API keys are present for production scan.

    Required:
    - FMP_API_KEY: Financial Modeling Prep (indices/ETFs, stock list)
    - POLYGON_API_KEY: Polygon (real-time price verification)
    - FINNHUB_API_KEY: Finnhub (earnings, news)

    Raises:
        RuntimeError: When any required key is missing.
    """
    missing = []
    for key in ["FMP_API_KEY", "POLYGON_API_KEY", "FINNHUB_API_KEY"]:
        if not os.getenv(key):
            missing.append(key)
    if missing:
        raise RuntimeError(f"Preflight failed: missing environment keys {missing}. Set them in .env or CI secrets.")

def fetch_top_us_tickers_by_market_cap(limit: int = 2000) -> List[str]:
    """Fetch US tickers ordered by market cap with robust fallbacks.

    Priority:
    1) FMP stock/list (fast, preferred)
    2) Polygon v3/reference/tickers (paginate via cursor, filter US common stocks)
    3) Local/Gist fallback: S&P 500 list from data/sp500_tickers.txt or hardcoded subset

    Args:
        limit: Hard cap of 2000 tickers (defaults to 2000)

    Returns:
        List of ticker symbols
    """
    global LAST_UNIVERSE_PROVIDER

    # --- Primary: FMP company screener (stable API) ---
    try:
        from core.config import get_secret
        api_key = get_secret("FMP_API_KEY", "")
        if api_key:
            url = "https://financialmodelingprep.com/stable/company-screener"
            try:
                min_cap = int(os.getenv("MIN_MCAP", "300000000"))  # $300M minimum
            except Exception:
                min_cap = 300_000_000
            try:
                # Allow full range up to mega-caps for comprehensive scanning
                max_cap = int(os.getenv("MAX_MCAP", "10000000000000"))  # $10T (effectively no limit)
            except Exception:
                max_cap = 10_000_000_000_000
            params = {
                "marketCapMoreThan": max(min_cap, 0),
                "marketCapLowerThan": max_cap,
                "isActivelyTrading": True,
                "isEtf": False,
                "isFund": False,
                "limit": min(limit * 2, 3000),  # Request extra to filter
                "apikey": api_key,
            }
            resp = requests.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                data = resp.json() or []
                rows = []
                for it in data:
                    sym = it.get("symbol")
                    mcap = it.get("marketCap") or it.get("marketCapitalization") or 0.0
                    if sym and isinstance(mcap, (int, float)) and float(mcap) > 0:
                        rows.append((sym, float(mcap)))
                rows.sort(key=lambda x: x[1], reverse=True)
                out = _normalize_symbols([s for s, _ in rows][:limit])
                if out:
                    logger.info(f"✓ Universe from FMP screener: {len(out)} tickers")
                    LAST_UNIVERSE_PROVIDER = "FMP"
                    return out
            elif resp.status_code == 403:
                logger.warning("FMP screener 403; falling back immediately to Polygon")
            else:
                logger.warning(f"FMP screener failed: HTTP {resp.status_code}")
        else:
            logger.warning("FMP_API_KEY missing; skipping FMP universe fetch")
    except Exception as e:
        logger.warning(f"FMP universe fetch errored: {e}")

    # --- Fallback 1: Polygon v3/reference/tickers ---
    # NOTE: Polygon returns alphabetical unsorted list, skip it and go to local fallback
    # which is sorted by market cap for better stock selection
    logger.info("Skipping Polygon (returns alphabetical list); using local market-cap-sorted fallback")


    # --- Fallback 2: Local S&P 500 (sorted by market cap) - PREFERRED ---
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        # Prefer sorted file (by market cap), then fall back to alphabetical
        candidates = [
            os.path.join(base_dir, "sp500_tickers_sorted.txt"),  # Sorted by market cap
            os.path.join(base_dir, "sp500_tickers.txt"),
            os.path.join(base_dir, "data", "sp500_tickers.txt"),
        ]
        for local_path in candidates:
            if os.path.exists(local_path):
                with open(local_path, "r") as f:
                    syms = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
                if syms:
                    logger.info(f"✓ Using local S&P 500 fallback: {len(syms)} tickers from {os.path.relpath(local_path, base_dir)}")
                    LAST_UNIVERSE_PROVIDER = "Local_SP500"
                    return syms[:min(limit, len(syms))]
    except Exception as e:
        logger.debug(f"Local S&P 500 read failed: {e}")

    # --- Fallback 3: EODHD or Nasdaq (API fallback) ---
    try:
        from core.config import get_secret
        eod_key = get_secret("EODHD_API_KEY", "")
        if eod_key:
            url = "https://eodhd.com/api/exchange-symbol-list/US"
            params = {"api_token": eod_key, "fmt": "json"}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json() or []
                syms = []
                for it in data:
                    sym = it.get("Code") or it.get("code") or it.get("Symbol")
                    type_ = (it.get("Type") or it.get("type") or "").upper()
                    # Keep common stocks only
                    if sym and ("COMMON" in type_ or type_ in ("COMMON STOCK", "CS", "ETF")):
                        syms.append(sym)
                out = _normalize_symbols(syms[:limit])
                if out:
                    logger.info(f"✓ Universe from EODHD: {len(out)} tickers")
                    LAST_UNIVERSE_PROVIDER = "EODHD"
                    return out
        nasdaq_key = get_secret("NASDAQ_API_KEY", "")
        if nasdaq_key:
            # Placeholder: Attempt Nasdaq symbols endpoint if available
            # If unsupported, skip silently
            pass
    except Exception as e:
        logger.warning(f"EODHD/Nasdaq universe fetch errored: {e}")

    # --- Fallback 4: Hardcoded Top 100 US stocks by market cap (Jan 2026 approx, for API fallback)
    try:
        top100_by_mcap = [
            # Mega-cap (>$500B)
            "AAPL","MSFT","NVDA","AMZN","GOOG","GOOGL","META","BRK-B","TSLA","AVGO",
            "LLY","V","JPM","UNH","XOM","MA","COST","WMT","JNJ","HD",
            # Large-cap ($100B-$500B)
            "ORCL","PG","MRK","ABBV","CVX","BAC","CRM","AMD","KO","NFLX",
            "PEP","TMO","CSCO","ACN","MCD","ABT","LIN","ADBE","DIS","WFC",
            "PM","INTC","INTU","TXN","QCOM","CMCSA","VZ","NEE","DHR","RTX",
            "AMGN","NKE","HON","SPGI","LOW","PFE","IBM","UNP","COP","BA",
            "CAT","GS","AMAT","BKNG","AXP","BLK","ELV","ISRG","SYK","MS",
            "DE","MDLZ","T","GILD","VRTX","NOW","MMC","REGN","ADI","LRCX",
            "C","SBUX","PANW","MU","BMY","PGR","TJX","CI","CB","SO",
            "ADP","SCHW","BSX","KLAC","DUK","MO","ZTS","SNPS","PLD","FI",
        ]
        out = _normalize_symbols(top100_by_mcap)
        logger.warning(f"Using hardcoded top-100 by market cap fallback: {len(out)} tickers")
        LAST_UNIVERSE_PROVIDER = "Hardcoded_Top100"
        return out[:min(limit, len(out))]
    except Exception:
        return top100_by_mcap

def _normalize_symbols(symbols: List[str]) -> List[str]:
    """Normalize ticker symbols for consistency across providers/yfinance.
    - Convert dots/slashes to dashes (e.g., BRK.B → BRK-B, BRK/B → BRK-B)
    - Uppercase symbols
    - De-duplicate while preserving order
    """
    seen = set()
    out: List[str] = []
    for s in symbols:
        if not s:
            continue
        t = str(s).upper().replace('.', '-').replace('/', '-')
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def fetch_latest_company_news(symbol: str, count: int = 5) -> List[Dict[str, Any]]:
    """Fetch latest company news via Finnhub.

    Args:
        symbol: Ticker symbol
        count: Number of headlines to return
    """
    from core.config import get_secret
    token = get_secret("FINNHUB_API_KEY", "")
    if not token:
        return []
    # Use last 7 days window
    to_dt = datetime.utcnow().date()
    from_dt = to_dt - timedelta(days=7)
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": str(from_dt), "to": str(to_dt), "token": token}
    try:
        r = requests.get(url, params=params, timeout=6)
        if r.status_code != 200:
            return []
        items = r.json() or []
        # Sort by datetime descending and take top count
        items.sort(key=lambda x: x.get("datetime", 0), reverse=True)
        return items[:count]
    except Exception:
        return []

def analyze_sentiment_openai(headlines: List[str]) -> Dict[str, Any]:
    """Call OpenAI Chat Completions to score sentiment for a set of headlines.

    Returns a dict with overall sentiment and per-headline scores.
    If OPENAI_API_KEY missing, returns a neutral placeholder.
    """
    from core.config import get_secret
    api_key = get_secret("OPENAI_API_KEY", "")
    model = get_secret("OPENAI_API_MODEL", "gpt-4o-mini")
    if not api_key or not headlines:
        return {"overall": "NEUTRAL", "confidence": 0.0, "details": []}
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt = (
            "You are an equity news analyst. Given the following 5 headlines, "
            "return a JSON with fields: overall in {POSITIVE, NEGATIVE, NEUTRAL}, confidence (0-1), "
            "and details per headline with sentiment and rationale (short). Keep it concise."
        )
        messages = [
            {"role": "system", "content": "Analyze equity news sentiment succinctly."},
            {"role": "user", "content": prompt + "\n\n" + "\n".join(f"- {h}" for h in headlines)},
        ]
        body = {"model": model, "messages": messages, "temperature": 0.2}
        r = requests.post(url, headers=headers, json=body, timeout=15)
        if r.status_code != 200:
            return {"overall": "NEUTRAL", "confidence": 0.0, "details": []}
        js = r.json()
        content = js.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        # Try to parse as JSON; if it isn't pure JSON, return as text blob
        try:
            import json
            return json.loads(content)
        except Exception:
            return {"overall": "NEUTRAL", "confidence": 0.0, "details": [{"raw": content}]}
    except Exception:
        return {"overall": "NEUTRAL", "confidence": 0.0, "details": []}

def _normalize_config(config: Any) -> Dict[str, Any]:
    """Normalize incoming config into a canonical dict.

    Supported inputs:
    - None: returns {}
    - dict: shallow-copied
    - objects with .to_dict(): use that
    - dataclasses: converted via dataclasses.asdict
    Otherwise: raise TypeError to avoid silent misconfiguration.
    Also applies selective environment overrides (e.g., METEOR_MODE, SMART_SCAN)
    after base normalization.
    """
    import dataclasses  # local import to avoid global pollution
    if config is None:
        normalized = {}
    elif isinstance(config, dict):
        normalized = dict(config)
    else:
        # Object types
        try:
            if hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
                normalized = dict(config.to_dict())
            elif dataclasses.is_dataclass(config):
                normalized = dataclasses.asdict(config)
            else:
                raise TypeError(f"Unsupported config type: {type(config).__name__}. Provide dict or dataclass/Config.")
        except Exception:
            raise
    key_map = {
        "FUNDAMENTAL_ENABLED": "fundamental_enabled",
        "BETA_FILTER_ENABLED": "beta_filter_enabled",
        "BETA_MAX_ALLOWED": "beta_max_allowed",
        "BETA_TOP_K": "beta_top_k",
        "BETA_BENCHMARK": "beta_benchmark",
    }
    for old_key, new_key in key_map.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized[old_key]
    # Enforce softened volume constraint for Tier 1 and downstream risk filters
    try:
        if "min_avg_volume" not in normalized or not isinstance(normalized.get("min_avg_volume"), (int, float)):
            normalized["min_avg_volume"] = 100_000
        else:
            if float(normalized.get("min_avg_volume", 100_000)) > 100_000:
                normalized["min_avg_volume"] = 100_000
    except Exception:
        normalized["min_avg_volume"] = 100_000

    # Environment overrides (post-normalization). Only apply to expected keys.
    try:
        def _env_bool(name: str) -> Optional[bool]:
            import os as _os
            val = _os.getenv(name)
            if val is None:
                return None
            s = str(val).strip().lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
            return None
        # METEOR_MODE -> meteor_mode
        mm = _env_bool("METEOR_MODE")
        if mm is not None:
            normalized["meteor_mode"] = bool(mm)
        # SMART_SCAN -> smart_scan
        ss = _env_bool("SMART_SCAN")
        if ss is not None:
            normalized["smart_scan"] = bool(ss)
        # EXTERNAL_PRICE_VERIFY -> external_price_verify
        epv = _env_bool("EXTERNAL_PRICE_VERIFY")
        if epv is not None:
            normalized["external_price_verify"] = bool(epv)
    except Exception:
        pass

    return normalized

def fetch_history_bulk(tickers: List[str], period_days: int, ma_long: int) -> Dict[str, pd.DataFrame]:
    # Phase 14: Hard override for lookback – ignore args and fetch 365-calendar days
    days_to_fetch = 365
    end = datetime.utcnow()
    start = end - timedelta(days=days_to_fetch)
    data_map = {}
    # Relax minimum rows requirement: allow proceeding with 60 rows
    min_rows = 60

    # Batch in groups of 50 to mitigate timeouts; sleep 1s between batches
    CHUNK = 50
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i+CHUNK]
        try:
            df_all = yf.download(
                chunk,
                start=start,
                end=end,
                group_by='ticker',
                progress=False,
                threads=True,
                auto_adjust=True,
            )
            if len(chunk) == 1:
                tkr = chunk[0]
                # For single ticker, yfinance with group_by='ticker' still returns MultiIndex
                # Access via ticker key to get flat columns
                try:
                    df = df_all[tkr].dropna(how='all')
                except (KeyError, TypeError):
                    # Fallback: flatten MultiIndex columns if direct access fails
                    df = df_all.copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(-1)
                    df = df.dropna(how='all')
                if not df.empty and len(df) >= min_rows:
                    data_map[tkr] = df
                    logger.info(f"Fetched {len(df)} rows for {tkr}")
                else:
                    logger.warning(f"Insufficient data for {tkr}: {len(df)} rows < {min_rows} required")
            else:
                for tkr in chunk:
                    try:
                        df = df_all[tkr].dropna(how='all')
                        if len(df) >= min_rows:
                            data_map[tkr] = df
                            logger.info(f"Fetched {len(df)} rows for {tkr}")
                        else:
                            logger.warning(f"Insufficient data for {tkr}: {len(df)} rows < {min_rows} required")
                    except KeyError:
                        logger.warning(f"No data for {tkr} in bulk download")
        except Exception as e:
            logger.warning(f"Batch fetch failed for {len(chunk)} tickers: {e}")
        # Sleep between batches to ease provider load
        time.sleep(1.0)
    return data_map

def fetch_beta_vs_benchmark(ticker: str, bench: str = "SPY", days: int = 252) -> float:
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days + 30)
        df_t = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        df_b = yf.download(bench, start=start, end=end, progress=False, auto_adjust=True)
        
        if df_t.empty or df_b.empty: return np.nan
        
        # Handle MultiIndex columns from newer yfinance versions
        if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)
        if isinstance(df_b.columns, pd.MultiIndex):
            df_b.columns = df_b.columns.get_level_values(0)
        
        j = pd.concat([df_t["Close"].pct_change(), df_b["Close"].pct_change()], axis=1).dropna()
        j.columns = ["rt", "rb"]
        if len(j) < 40: return np.nan
        
        slope = np.polyfit(j["rb"].to_numpy(), j["rt"].to_numpy(), 1)[0]
        return float(slope)
    except: return np.nan

def calculate_rr(entry_price: float, target_price: float, atr_value: float, history_df: pd.DataFrame = None) -> float:
    try:
        if not (np.isfinite(entry_price) and np.isfinite(target_price)): return 0.0
        atr = atr_value if np.isfinite(atr_value) else np.nan
        
        if np.isnan(atr) and history_df is not None:
            try:
                last = history_df.tail(14)
                est_atr = (last["High"] - last["Low"]).abs().mean()
                if np.isfinite(est_atr): atr = float(est_atr)
            except Exception as e:
                logger.debug(f"ATR estimation failed: {e}")
            
        risk = max(atr * 2.0, entry_price * 0.01) if np.isfinite(atr) else max(entry_price * 0.01, 0.01)
        reward = max(0.0, float(target_price) - float(entry_price))
        return float(np.clip(reward / max(risk, 1e-9), 0.0, 5.0))
    except: return 0.0


# --- Pipeline Steps ---

def _step_fetch_and_prepare_base_data(
    universe: List[str],
    config: Dict[str, Any],
    status_callback: Optional[Callable[[str], None]],
    data_map: Optional[Dict[str, pd.DataFrame]],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    if data_map is None:
        if status_callback:
            status_callback("Fetching historical data...")
        # Phase 13: Hard-code minimum lookback to 250 days for VCP/52-week calculations
        data_map = fetch_history_bulk(universe, 250, config.get("ma_long", 200))

    # Ensure benchmark also uses at least ~250 days (252 acceptable); keep existing default
    benchmark_df = fetch_benchmark_data(config.get("beta_benchmark", "SPY"), max(250, int(config.get("lookback_days", 252))))
    return data_map, benchmark_df


def _process_single_ticker(tkr: str, df: pd.DataFrame, skip_tech_filter: bool) -> Optional[pd.Series]:
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
        rs_val = compute_relative_strength_vs_spy(tech_df.rename(columns=str.title), spy_df_cap)
        # Use .loc with 'Ticker' index to avoid SettingWithCopyWarning
        row_df = row_df.set_index("Ticker")
        row_df.loc[tkr, "relative_strength_20d"] = float(rs_val)
    except Exception:
        # Ensure the column exists even on failure
        row_df = row_df.set_index("Ticker")
        row_df.loc[tkr, "relative_strength_20d"] = np.nan
    # Convert enriched single-row DataFrame back to Series for downstream logic
    row_indicators = row_df.reset_index(drop=False).iloc[0]

    # Compute sector features (Sector_RS, Sector_Momentum, Sector_Rank)
    sector_context = None
    try:
        from core.market_context import compute_sector_features
        sector_context = compute_sector_features(tkr, df)
        # Also add to row_indicators for downstream use
        for k, v in sector_context.items():
            row_indicators[k] = v
    except Exception as exc:
        logger.debug(f"Sector feature computation failed for {tkr}: {exc}")

    # Enrich with all 34 ML features using the feature builder
    try:
        from core.ml_feature_builder import build_all_ml_features_v3
        ml_features = build_all_ml_features_v3(
            row=row_indicators,
            df_hist=df,
            market_context=None,  # Will use defaults
            sector_context=sector_context,  # Now passing real sector context!
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
    except Exception:
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
                # Explicitly record fallback and warn loudly
                msg = f"Bridge returned no score for {tkr}"
                logger.warning(msg + ", falling back to legacy.")
                _record_legacy_fallback(f"BridgeNoScore: {msg}")
        except Exception as e:
            logger.warning(f"Bridge failed for {tkr}: {e}. Falling back to legacy.")
            _record_legacy_fallback(f"{e.__class__.__name__}: {e}")
            used_v2 = False

    if not used_v2:
        # Fallback to legacy logic (explicit, tracked)
        # ML enable flag with version-mismatch guard
        try:
            ml_meta = get_ml_health_meta()
            ml_enable = bool(ML_20D_AVAILABLE) and not bool(ml_meta.get("ml_bundle_version_warning"))
        except Exception:
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
        except Exception:
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

    # Ensure SignalReasons/SignalQuality exist even when using bridge path
    try:
        from core.scoring_config import ML_PROB_THRESHOLD, TECH_STRONG_THRESHOLD
        # Compute reasons based on available fields
        reasons = []
        try:
            ts_val = float(rec_series.get("TechScore_20d", np.nan))
            if np.isfinite(ts_val) and ts_val >= float(TECH_STRONG_THRESHOLD):
                reasons.append("Strong technical momentum")
        except Exception:
            pass
        try:
            mlp_val = float(rec_series.get("ML_20d_Prob", np.nan))
            if np.isfinite(mlp_val) and mlp_val >= float(ML_PROB_THRESHOLD):
                reasons.append("High ML breakout probability")
        except Exception:
            pass
        try:
            ps_val = float(rec_series.get("Pattern_Score", 0.0) or 0.0)
            if np.isfinite(ps_val) and ps_val > 0.0:
                reasons.append("Bullish pattern detected")
        except Exception:
            pass
        try:
            reg_val = str(rec_series.get("Market_Regime") or "").upper()
            if reg_val in ("TREND_UP", "BULLISH", "NEUTRAL", "SIDEWAYS"):
                reasons.append("Supportive market regime")
        except Exception:
            pass
        cnt = len(reasons)
        quality = "High" if cnt >= 3 else ("Medium" if cnt == 2 else "Speculative")
        if "SignalReasons" not in rec_series:
            rec_series["SignalReasons"] = "; ".join(reasons)
        if "SignalReasons_Count" not in rec_series:
            rec_series["SignalReasons_Count"] = cnt
        if "SignalQuality" not in rec_series:
            rec_series["SignalQuality"] = quality
    except Exception:
        # Default-safe: create columns if missing
        if "SignalReasons" not in rec_series:
            rec_series["SignalReasons"] = ""
        if "SignalReasons_Count" not in rec_series:
            rec_series["SignalReasons_Count"] = 0
        if "SignalQuality" not in rec_series:
            rec_series["SignalQuality"] = "Speculative"

    # Ensure Ticker column is present for downstream merges/filters
    try:
        rec_series["Ticker"] = tkr
    except Exception:
        pass
    return rec_series


def _step_compute_scores_with_unified_logic(
    data_map: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    status_callback: Optional[Callable[[str], None]],
    skip_tech_filter: bool = False,
) -> pd.DataFrame:
    if status_callback:
        status_callback("Computing technical indicators (parallel)...")

    rows: List[pd.Series] = []
    # Parallel processing with max_workers=10
    # Threading backoff: submit in small batches to avoid provider rate limits
    max_workers = min(10, max(1, len(data_map)))
    batch_size = max(10, max_workers * 2)  # reasonable batch size
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        items = list(data_map.items())
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            for tkr, df in batch:
                future = executor.submit(_process_single_ticker, tkr, df, skip_tech_filter)
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
    except Exception:
        pass
    if "Score" not in results.columns and "FinalScore_20d" in results.columns:
        results["Score"] = results["FinalScore_20d"]
    # Ensure reliability column compatibility for downstream consumers/tests
    if (
        ("ReliabilityScore" not in results.columns)
        and ("Reliability_Score" not in results.columns)
        and ("reliability_v2" not in results.columns)
        and ("Reliability_v2" not in results.columns)
    ):
        # Create a conservative default reliability metric when missing
        results["ReliabilityScore"] = 50.0
    else:
        # Provide legacy alias if canonical underscore variant exists
        if ("Reliability_Score" in results.columns) and ("ReliabilityScore" not in results.columns):
            results["ReliabilityScore"] = results["Reliability_Score"]
    # Ensure momentum proxy exists (prefer TechScore_20d)
    if ("MomentumScore" not in results.columns) and ("TechScore_20d" not in results.columns):
        if "FinalScore_20d" in results.columns:
            results["TechScore_20d"] = results["FinalScore_20d"]
        elif "Score" in results.columns:
            results["TechScore_20d"] = results["Score"]
        else:
            results["TechScore_20d"] = np.nan
    return results

# --- Main Pipeline Runner ---

def run_scan_pipeline(
    universe: List[str],
    config: Dict[str, Any],
    status_callback: Optional[Callable[[str], None]] = None,
    data_map: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict[str, Any]:
    """
    Unified pipeline runner for Stock Scout.
    
    **Pipeline flow:**
    1. Fetch historical data (yfinance bulk download)
    2. Compute technical indicators for each ticker
    3. Apply basic technical filters (volume, liquidity)
    4. **Call compute_recommendation_scores (unified scoring) for each ticker**
       - Technical score (0-100)
       - Fundamental score with multi-source data (0-100)
       - ML 20d probability (0-1)
       - Final combined score (0-100)
       - Risk meter & reliability metrics
       - Classification flags
    5. Apply beta filter (optional, top-K only)
    6. Apply advanced filters (RS, momentum, RR penalties)
    7. Enrich with fundamentals & sector data
    8. Run classification & allocation
    9. Check earnings blackout (optional, top-K only)
    
    Args:
        universe: List of ticker symbols to scan
        config: Configuration dict from get_config()
        status_callback: Optional progress callback function
        data_map: Optional pre-fetched historical data
    
    Returns:
        Tuple of (results_df, data_map) where:
        - results_df: DataFrame with all scores, metrics, and allocations
        - data_map: Dict of ticker -> historical DataFrame
    
    Key output columns:
        - FinalScore_20d: Main ranking score (0-100)
        - TechScore_20d: Technical component (0-100)
        - Fundamental_Score: Fundamental component (0-100)
            winners = results.loc[results["Score"] > 65.0, "Ticker"].tolist()
        - ConvictionScore: Conviction metric (0-100)
                logger.info("No winners with Score > 65; skipping fundamentals fetch")
        - Reliability_Score: Data quality (0-100)
                logger.info(f"Fetching fundamentals for {len(winners)} winners (Score > 65)")
        - Score: Legacy alias for FinalScore_20d
    """
    # Log scope at entry
    try:
        logger.info(f"🌌 Starting pipeline with universe size: {len(universe)} tickers")
    except Exception:
        pass

    # Normalize config keys (support legacy uppercase keys)
    config = _normalize_config(config)
    # Initialize diagnostics store for filter reasons
    diagnostics: Dict[str, Dict[str, Any]] = {}


    if status_callback:
        status_callback(f"Starting pipeline for {len(universe)} tickers...")

    # Initialize telemetry
    telemetry = Telemetry()
    try:
        telemetry.set_value("universe_provider", LAST_UNIVERSE_PROVIDER)
    except Exception:
        pass

    # Initialize global market context caches (SPY + VIX) to avoid repeated provider hits
    try:
        from core.market_context import initialize_market_context
        # Enforce minimum market context window of 250 days
        initialize_market_context(symbols=["SPY", "^VIX"], period_days=250)
        logger.info("[PIPELINE] Global index cache initialized (SPY, VIX)")
        # Record index provider telemetry (SPY, VIX)
        try:
            from core.data_sources_v2 import get_last_index_source
            spy_src = get_last_index_source('SPY')
            vix_src = get_last_index_source('^VIX') or get_last_index_source('VIX')
            idx_map = {}
            if spy_src:
                idx_map['SPY'] = spy_src
                # Also mark price domain if Polygon was used for SPY
                try:
                    if str(spy_src).upper() == 'POLYGON':
                        telemetry.mark_used('price', 'POLYGON')
                except Exception:
                    pass
            if vix_src:
                idx_map['VIX'] = vix_src
            telemetry.set_value('index', idx_map)
            if vix_src and 'Synthetic' in vix_src:
                telemetry.record_fallback('index', 'provider_chain', 'SYNTHETIC_VIX_PROXY', 'constructed proxy from SPY volatility')
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"[PIPELINE] Market context init skipped: {e}")

    # Run API preflight to determine active providers for this scan
    try:
        from core.api_preflight import run_preflight
        provider_status: Dict[str, Dict[str, Any]] = run_preflight()
        run_mode = provider_status.get("SCAN_STATUS", "OK")
        try:
            telemetry.set_value("preflight_status", provider_status.get("ACTIVE_COUNTS", {}))
        except Exception:
            pass
        # Apply global default provider status to v2 data layer
        try:
            from core.data_sources_v2 import set_default_provider_status, disable_provider_category, get_prioritized_fetch_funcs
            set_default_provider_status(provider_status)
            # Feed ProviderGuard with preflight baseline
            try:
                from core.provider_guard import get_provider_guard
                guard = get_provider_guard()
                guard.update_from_preflight(provider_status)
                # Snapshot into telemetry for meta visibility
                telemetry.set_value("provider_states", guard.snapshot())
            except Exception:
                pass
            # If FMP index endpoint is not OK, disable for session
            if not provider_status.get("FMP_INDEX", {"ok": True}).get("ok", True):
                disable_provider_category("fmp", "index")
            # Log dynamic smart routing order for fundamentals
            try:
                funcs = get_prioritized_fetch_funcs(provider_status)
                # Map internal names to uppercase provider keys
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
                    meta = provider_status.get(up, {})
                    lat = meta.get("latency")
                    if isinstance(lat, (int, float)):
                        parts.append(f"{up} ({lat:.2f}s)")
                    else:
                        parts.append(f"{up} (n/a)")
                if parts:
                    logger.info("Smart Routing: " + " -> ".join(parts))
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        provider_status = {}
        run_mode = "OK"

    # Reset fallback trackers for a clean run
    try:
        with _LEGACY_LOCK:
            global _LEGACY_FALLBACK_USED, _LEGACY_FALLBACK_REASONS
            _LEGACY_FALLBACK_USED = False
            _LEGACY_FALLBACK_REASONS = []
    except Exception:
        pass

    # Enforce preflight gating policy
    try:
        if run_mode == "BLOCKED":
            meta = {
                "engine_version": "pipeline_v2",
                "sources_used": telemetry.export(),
                "run_timestamp_utc": datetime.utcnow().isoformat(),
                "postfilter_mode": "blocked",
                "run_mode": "BLOCKED",
            }
            try:
                meta.update(get_ml_health_meta())
            except Exception:
                pass
            logger.error("[PIPELINE] Preflight BLOCKED: no active price providers; aborting scan")
            return {"result": {"results_df": pd.DataFrame(), "data_map": {}, "diagnostics": {}}, "meta": meta}
        elif run_mode == "DEGRADED_TECH_ONLY":
            config["fundamental_enabled"] = False
            logger.warning("[PIPELINE] Preflight DEGRADED_TECH_ONLY: fundamentals disabled for this run")
    except Exception:
        pass

    start_universe = len(universe)
    data_map, benchmark_df = _step_fetch_and_prepare_base_data(universe, config, status_callback, data_map)
    # Benchmark status meta
    benchmark_status = "OK"
    try:
        if benchmark_df is None or (hasattr(benchmark_df, "empty") and benchmark_df.empty):
            benchmark_status = "UNAVAILABLE"
            logger.warning("[PIPELINE] Benchmark unavailable; skipping RS/Beta dependent steps")
    except Exception:
        benchmark_status = "UNAVAILABLE"

    # Decide postfilter mode early and keep it consistent across all exits
    try:
        small_universe_lenient = bool(start_universe < 50 or bool(config.get("smoke_mode", False)))
    except Exception:
        small_universe_lenient = bool(start_universe < 50)
    postfilter_mode_global = "lenient_small_universe" if small_universe_lenient else "strict"

    # Minimal price telemetry for per-ticker providers (mark for first ticker to avoid overhead)
    try:
        if universe:
            tkr0 = str(universe[0])
            from core.data import fetch_price_multi_source
            _ = fetch_price_multi_source(tkr0, provider_status=provider_status, telemetry=telemetry)
    except Exception:
        pass

    # Early RS percentile ranking on full universe (Weighted RS = 0.7*RS_63d + 0.3*RS_21d)
    # Controlled by env RS_RANKING/RS_RANKING_ENABLED and sample size (>100)
    rs_enabled_env = os.getenv("RS_RANKING", os.getenv("RS_RANKING_ENABLED", "1"))
    rs_enabled = bool(rs_enabled_env == "1")
    if rs_enabled and start_universe > 100:
        if status_callback:
            status_callback("Ranking universe by blended RS (21/63d)...")
        try:
            bench_df = fetch_benchmark_data("SPY", days=200)
            # Explicit DataFrame check to avoid ambiguous truth errors
            if bench_df is None or bench_df.empty:
                raise ValueError("Benchmark DataFrame is empty; skipping RS blended ranking")
            rs_records = []
            for tkr, df in (data_map or {}).items():
                try:
                    if df is None or df.empty:
                        rs_records.append({"Ticker": tkr, "RS_blend": np.nan})
                        continue
                    rs = compute_relative_strength(df.rename(columns=str.title), bench_df.rename(columns=str.title), periods=[21, 63])
                    # Support both lower-case and upper-case keys
                    rs63 = rs.get("rs_63d", rs.get("RS_63d", np.nan))
                    rs21 = rs.get("rs_21d", rs.get("RS_21d", np.nan))
                    if pd.notna(rs63) and pd.notna(rs21):
                        rs_blend = 0.7 * float(rs63) + 0.3 * float(rs21)
                    else:
                        rs_blend = rs63 if pd.notna(rs63) else rs21
                    rs_records.append({"Ticker": tkr, "RS_blend": rs_blend})
                except Exception:
                    rs_records.append({"Ticker": tkr, "RS_blend": np.nan})
            if rs_records:
                # Compute RS percentiles for diagnostics/sorting only; do NOT filter out names
                rs_df = pd.DataFrame(rs_records)
                rs_df["RS_blend_Pctl"] = rs_df["RS_blend"].rank(pct=True, ascending=True)
                logger.info(
                    f"[PIPELINE] RS blended ranking computed for {len(rs_records)} tickers (no hard filter applied)"
                )
        except Exception as e:
            logger.warning(f"RS blended ranking failed (continuing without): {e}")

    # --- Tier 1: Fast Scan (OHLCV-only, relaxed=False) ---
    if status_callback:
        status_callback("Tier 1: applying OHLCV filters...")

    tier1_pass: List[str] = []
    filtered_rows: List[Dict[str, Any]] = []
    for tkr, df in (data_map or {}).items():
        try:
            if df is None or df.empty:
                # Record empty history as a Tier 1 reason
                diagnostics.setdefault(tkr, {"tier1_reasons": [], "tier2_reasons": []})
                diagnostics[tkr]["tier1_reasons"].append({
                    "rule": "EMPTY_HISTORY",
                    "message": "No historical OHLCV data",
                })
                diagnostics[tkr]["last_price"] = np.nan
                diagnostics[tkr]["last_volume"] = np.nan
                filtered_rows.append({
                    "Ticker": tkr,
                    "Tier1_Reasons": "EMPTY_HISTORY",
                    "last_price": np.nan,
                    "last_volume": np.nan,
                })
                continue
            last = df.iloc[-1]
            # Apply fast technical filter using only OHLCV
            # Build structured Tier 1 reasons when excluded
            # Mirror core.unified_logic.apply_technical_filters checks
            # Compute last valid close/volume using canonical mapping
            try:
                def _canon(c):
                    try:
                        # Handle MultiIndex like ('AAPL','Close')
                        if isinstance(c, tuple) and len(c) >= 2:
                            return str(c[1]).lower()
                        return str(c).lower()
                    except Exception:
                        return str(c)
                cols_lower = {_canon(c): c for c in df.columns}
            except Exception:
                cols_lower = {}
            close_series = None
            vol_series = None
            if "close" in cols_lower:
                close_series = pd.to_numeric(df[cols_lower["close"]], errors="coerce")
            elif "adj close" in cols_lower:
                close_series = pd.to_numeric(df[cols_lower["adj close"]], errors="coerce")
            if "volume" in cols_lower:
                vol_series = pd.to_numeric(df[cols_lower["volume"]], errors="coerce")
            elif "v" in cols_lower:
                vol_series = pd.to_numeric(df[cols_lower["v"]], errors="coerce")

            close = (close_series.dropna().iloc[-1] if isinstance(close_series, pd.Series) and not close_series.dropna().empty else np.nan)
            volume = (vol_series.dropna().iloc[-1] if isinstance(vol_series, pd.Series) and not vol_series.dropna().empty else np.nan)
            reasons: List[Dict[str, Any]] = []
            # Missing data reasons take precedence and prevent min-threshold rules
            if pd.isna(volume):
                reasons.append({
                    "rule": "MISSING_VOLUME_DATA",
                    "message": "Missing Volume on last bar",
                })
            elif volume < 100000:
                reasons.append({
                    "rule": "VOLUME_MIN",
                    "message": "Volume below minimum",
                    "value": float(volume) if pd.notna(volume) else None,
                    "threshold": 100000,
                })
            # Price sanity
            if pd.isna(close):
                reasons.append({
                    "rule": "MISSING_PRICE_DATA",
                    "message": "Missing Close price on last bar",
                })
            elif close < 1.0:
                reasons.append({
                    "rule": "PRICE_MIN",
                    "message": "Price below $1",
                    "value": float(close),
                    "threshold": 1.0,
                })
            if pd.notna(close) and close > 10000:
                reasons.append({
                    "rule": "PRICE_MAX",
                    "message": "Price above $10,000",
                    "value": float(close),
                    "threshold": 10000.0,
                })

            # Determine pass: must have no reasons and pass technical filter
            # Use a surrogate row with canonical fields to avoid MultiIndex issues
            row_for_filter = pd.Series({"Volume": volume, "Close": close})
            if (not reasons) and apply_technical_filters(row_for_filter, relaxed=False):
                tier1_pass.append(tkr)
                diagnostics.setdefault(tkr, {"tier1_reasons": [], "tier2_reasons": []})
                diagnostics[tkr]["last_price"] = float(close) if pd.notna(close) else np.nan
                diagnostics[tkr]["last_volume"] = float(volume) if pd.notna(volume) else np.nan
            else:
                diagnostics.setdefault(tkr, {"tier1_reasons": [], "tier2_reasons": []})
                # If no specific reason matched, add a generic rule for transparency
                if not reasons:
                    reasons.append({
                        "rule": "TECH_FILTER_FAIL",
                        "message": "Failed Tier 1 technical filter",
                    })
                diagnostics[tkr]["tier1_reasons"].extend(reasons)
                diagnostics[tkr]["last_price"] = float(close) if pd.notna(close) else np.nan
                diagnostics[tkr]["last_volume"] = float(volume) if pd.notna(volume) else np.nan
                try:
                    joined = ";".join([str(r.get("rule")) for r in reasons])
                except Exception:
                    joined = ";".join([str(r) for r in reasons])
                filtered_rows.append({
                    "Ticker": tkr,
                    "Tier1_Reasons": joined,
                    "last_price": float(close) if pd.notna(close) else np.nan,
                    "last_volume": float(volume) if pd.notna(volume) else np.nan,
                })
        except Exception as exc:
            logger.debug(f"Tier1 filter failed for {tkr}: {exc}")
            diagnostics.setdefault(tkr, {"tier1_reasons": [], "tier2_reasons": []})
            diagnostics[tkr]["tier1_reasons"].append({
                "rule": "EXCEPTION",
                "message": f"Tier1 error: {exc.__class__.__name__}",
            })
            diagnostics[tkr]["last_price"] = np.nan
            diagnostics[tkr]["last_volume"] = np.nan
            filtered_rows.append({
                "Ticker": tkr,
                "Tier1_Reasons": "EXCEPTION",
                "last_price": np.nan,
                "last_volume": np.nan,
            })
            continue

    filtered_count = start_universe - len(tier1_pass)
    logger.info(
        f"[PIPELINE] Tier 1: scanned={start_universe}, passed={len(tier1_pass)}, filtered={filtered_count}"
    )

    # Build a separate small DataFrame for Tier 1 filtered tickers
    try:
        filtered_df = pd.DataFrame(filtered_rows)
    except Exception:
        filtered_df = pd.DataFrame()

    # Build Tier 2 input map strictly from Tier 1 output
    tier2_map: Dict[str, pd.DataFrame] = {t: data_map[t] for t in tier1_pass if t in data_map}
    # Fallback: if nothing passed Tier 1, process all available histories (relaxed)
    if not tier2_map and isinstance(data_map, dict) and data_map:
        logger.warning("[PIPELINE] Tier 1 yielded no candidates; falling back to relaxed Tier 2 on full universe")
        tier2_map = dict(data_map)

    # --- Tier 2: Deep Dive (indicators + ML, fundamentals later) ---
    results = _step_compute_scores_with_unified_logic(
        tier2_map, config, status_callback, skip_tech_filter=True
    )

    # Verify Tier 2 input list matches Tier 1 output (robust to missing Ticker column)
    tier2_input_set = set(tier2_map.keys())
    try:
        if not results.empty:
            if "Ticker" in results.columns:
                tier2_output_set = set([str(x) for x in results["Ticker"].tolist() if pd.notna(x)])
            elif results.index.name == "Ticker":
                tier2_output_set = set([str(x) for x in results.index.tolist() if pd.notna(x)])
            elif "ticker" in results.columns:
                tier2_output_set = set([str(x).upper().replace('.', '-').replace('/', '-') for x in results["ticker"].tolist() if pd.notna(x)])
            else:
                tier2_output_set = set()
        else:
            tier2_output_set = set()
    except Exception:
        tier2_output_set = set()
    if tier2_input_set == tier2_output_set:
        logger.info(
            f"[PIPELINE] Tier 2 verification OK: input={len(tier2_input_set)} equals output tickers"
        )
    else:
        missing = sorted(list(tier2_input_set - tier2_output_set))
        extra = sorted(list(tier2_output_set - tier2_input_set))
        logger.warning(
            f"[PIPELINE] Tier 2 verification mismatch: input={len(tier2_input_set)}, output={len(tier2_output_set)}; "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    if results.empty:
        # Return wrapped structure even on empty results, with meta
        try:
            telemetry.set_value("fundamentals_status", "not_requested")
        except Exception:
            pass
        meta = {
            "engine_version": "pipeline_v2",
            "used_legacy_fallback": bool(_LEGACY_FALLBACK_USED),
            "fallback_reason": ", ".join(sorted(set(_LEGACY_FALLBACK_REASONS))) if _LEGACY_FALLBACK_REASONS else None,
            "sources_used": telemetry.export(),
            "run_timestamp_utc": datetime.utcnow().isoformat(),
            "postfilter_mode": postfilter_mode_global,
            "run_mode": run_mode,
            "benchmark_status": benchmark_status,
        }
        try:
            meta.update(get_ml_health_meta())
            try:
                if meta.get("ml_bundle_version_warning"):
                    meta["ml_mode"] = "DISABLED_VERSION_MISMATCH"
                elif not ML_20D_AVAILABLE:
                    meta["ml_mode"] = "DISABLED_NO_MODEL"
                else:
                    meta["ml_mode"] = "HYBRID"
            except Exception:
                pass
        except Exception:
            pass
        # Include filtered_df summary in data_map
        try:
            if isinstance(data_map, dict):
                data_map = dict(data_map)
                data_map["filtered_tier1_df"] = filtered_df
        except Exception:
            pass
        return {"result": {"results_df": results, "data_map": data_map, "diagnostics": diagnostics}, "meta": meta}
    
    if "Score" not in results.columns and "FinalScore_20d" in results.columns:
        results["Score"] = results["FinalScore_20d"]
    
    # 3b. Beta Filter
    if config.get("beta_filter_enabled"):
        if status_callback: status_callback("Applying Beta filter...")
        beta_max = float(config.get("beta_max_allowed", 1.5))
        top_k = int(config.get("beta_top_k", 50))
        results = results.sort_values("Score", ascending=False)
        to_check = results.head(top_k).index
        for idx in to_check:
            tkr = results.at[idx, "Ticker"]
            b = fetch_beta_vs_benchmark(tkr, config.get("beta_benchmark", "SPY"))
            results.at[idx, "Beta"] = b
        results = results[~( (results["Beta"].notna()) & (results["Beta"] > beta_max) )]
        logger.info(f"[PIPELINE] After beta filter: {len(results)} remain")
    
    # 4. Advanced Filters (Penalties)
    if status_callback: status_callback("Applying advanced filters...")
    signals_store = []
    for idx, row in results.iterrows():
        tkr = row["Ticker"]
        if tkr in data_map:
            df = data_map[tkr]
            base_score = row["Score"]
            # advanced_filters expects Title-case OHLCV column names
            df_title = df.rename(columns=str.title)
            # Ensure Date column exists for advanced_filters expectations
            if "Date" not in df_title.columns:
                try:
                    df_title = df_title.reset_index()
                    if "Date" not in df_title.columns and df_title.columns[0] not in ("Date", "date"):
                        # Fallback: rename first column to Date
                        first = df_title.columns[0]
                        df_title = df_title.rename(columns={first: "Date"})
                except Exception:
                    pass
            bench_title = benchmark_df.rename(columns=str.title)
            enhanced, sig = compute_advanced_score(
                tkr,
                df_title,
                bench_title,
                base_score / 100.0,
            )
            signals_store.append((idx, sig, enhanced))
            
    rs_vals = [s.get("rs_63d") for _, s, _ in signals_store if s.get("rs_63d") is not None]
    mom_vals = [s.get("momentum_consistency") for _, s, _ in signals_store if s.get("momentum_consistency") is not None]
    rr_vals = [s.get("risk_reward_ratio") for _, s, _ in signals_store if s.get("risk_reward_ratio") is not None]
    
    def _q(vals, q, default):
        return float(np.quantile(vals, q)) if vals else default
    
    # Use more lenient percentiles to avoid over-filtering
    # If we have fewer stocks, use even more lenient thresholds
    num_stocks = len(signals_store)
    if num_stocks < 20:
        # Very small sample - use fixed defaults only
        rs_thresh = -0.40  # More lenient
        mom_thresh = 0.10
        rr_thresh = 0.30   # More lenient
        logger.info(f"[PIPELINE] Using fixed lenient thresholds for {num_stocks} stocks")
    else:
        # Use dynamic thresholds but more lenient percentiles
        rs_thresh = min(_q(rs_vals, 0.02, -0.40), -0.30)  # Cap at -0.30
        mom_thresh = min(_q(mom_vals, 0.05, 0.10), 0.12)  # Cap at 0.12
        rr_thresh = min(_q(rr_vals, 0.05, 0.30), 0.40)    # Cap at 0.40
        logger.info(f"[PIPELINE] Using dynamic thresholds for {num_stocks} stocks")
    
    dyn_thresh = {"rs_63d": rs_thresh, "momentum_consistency": mom_thresh, "risk_reward_ratio": rr_thresh}
    logger.info(f"[PIPELINE] Thresholds: RS={rs_thresh:.3f}, Mom={mom_thresh:.3f}, RR={rr_thresh:.3f}")
    
    for idx, sig, enhanced in signals_store:
        catastrophic, reason = should_reject_ticker(sig, dynamic=dyn_thresh)
        
        results.at[idx, "RS_63d"] = sig.get("rs_63d")
        results.at[idx, "Volume_Surge"] = sig.get("volume_surge")
        results.at[idx, "MA_Aligned"] = sig.get("ma_aligned")
        results.at[idx, "Quality_Score"] = sig.get("quality_score")
        results.at[idx, "RR_Ratio"] = sig.get("risk_reward_ratio")
        results.at[idx, "Momentum_Consistency"] = sig.get("momentum_consistency")
        
        if catastrophic:
            # Absolute Hunter Override (Phase 13): do NOT zero technical score
            # Preserve the true technical/VCP strength and annotate the reason
            results.at[idx, "FinalScore_20d"] = float(enhanced)
            results.at[idx, "RejectionReason"] = reason
            try:
                tkr = str(results.at[idx, "Ticker"]) if "Ticker" in results.columns else None
                if tkr:
                    diagnostics.setdefault(tkr, {"tier1_reasons": [], "tier2_reasons": []})
                    diagnostics[tkr]["tier2_reasons"].append({
                        "rule": "ADVANCED_REJECT",
                        "message": str(reason) if reason else "Advanced filters rejection",
                    })
            except Exception:
                pass
        else:
            # Penalties are in [0, 4.5] scale (0-100 range), normalize to [0, 0.045]
            penalty = 0.0
            if sig.get("rs_63d", 0) < rs_thresh: penalty += 1.0  # Reduced from 2.0
            if sig.get("momentum_consistency", 0) < mom_thresh: penalty += 1.0  # Reduced from 2.0
            if sig.get("risk_reward_ratio", 0) < rr_thresh: penalty += 1.5  # Reduced from 3.0
            
            normalized_penalty = penalty / 100.0  # Convert from [0, 4.5] to [0, 0.045]
            results.at[idx, "AdvPenalty"] = penalty
            # Apply a mild penalty to FinalScore_20d, but never hard-zero it
            # enhanced is in [0, 1] range; keep lower bound small but positive
            results.at[idx, "FinalScore_20d"] = max(0.01, float(enhanced) - float(normalized_penalty))
            # Optional Tier 2 diagnostics for threshold breaches
            try:
                tkr = str(results.at[idx, "Ticker"]) if "Ticker" in results.columns else None
                if tkr:
                    diagnostics.setdefault(tkr, {"tier1_reasons": [], "tier2_reasons": []})
                    if sig.get("rs_63d", None) is not None and sig.get("rs_63d") < rs_thresh:
                        diagnostics[tkr]["tier2_reasons"].append({
                            "rule": "RS_BELOW_THRESH",
                            "message": "RS_63d below threshold",
                            "value": float(sig.get("rs_63d")),
                            "threshold": float(rs_thresh),
                        })
                    if sig.get("momentum_consistency", None) is not None and sig.get("momentum_consistency") < mom_thresh:
                        diagnostics[tkr]["tier2_reasons"].append({
                            "rule": "MOM_BELOW_THRESH",
                            "message": "Momentum consistency below threshold",
                            "value": float(sig.get("momentum_consistency")),
                            "threshold": float(mom_thresh),
                        })
                    if sig.get("risk_reward_ratio", None) is not None and sig.get("risk_reward_ratio") < rr_thresh:
                        diagnostics[tkr]["tier2_reasons"].append({
                            "rule": "RR_BELOW_THRESH",
                            "message": "Risk/Reward below threshold",
                            "value": float(sig.get("risk_reward_ratio")),
                            "threshold": float(rr_thresh),
                        })
            except Exception:
                pass

    # Scale FinalScore_20d back to 0-100 for consistency with rest of system
    # (all other scoring is in 0-100 range, FinalScore_20d is canonical for display/filtering)
    results["FinalScore_20d"] = results["FinalScore_20d"] * 100.0

    # Ensure Score always matches the scaled FinalScore_20d after advanced filters
    if "FinalScore_20d" in results.columns:
        results["Score"] = results["FinalScore_20d"]
    
    # Do not filter out low scores here; the UI will label REJECTs but keep their scores visible
    logger.info(f"[PIPELINE] Advanced filters applied without score-zeroing; total stocks: {len(results)}")
    
    # Optional: Meteor Mode filter (VCP + RS + Pocket Pivots)
    # NOTE: Default to OFF - Meteor filters are very strict and may filter all stocks
    meteor_mode = bool(config.get("meteor_mode", bool(os.getenv("METEOR_MODE", "0") == "1")))
    if meteor_mode and not results.empty:
        if status_callback: status_callback("Applying Meteor filters (VCP + RS + Pocket Pivots)...")
        try:
            # Use core.filters version to avoid local import shadowing
            bench_df = fetch_benchmark_data("SPY", days=200)
            if bench_df is None or bench_df.empty:
                logger.warning("Benchmark DataFrame empty; skipping Meteor filters")
            else:
                kept_rows = []
                for _, row in results.iterrows():
                    tkr = str(row.get("Ticker"))
                    # Avoid ambiguous truth-value evaluation of DataFrames when using `or`
                    df = tier2_map.get(tkr)
                    if df is None:
                        df = data_map.get(tkr)
                    if df is None or df.empty:
                        continue
                    base_score = float(row.get("FinalScore_20d", row.get("Score", 0.0)))
                    new_score, details = compute_advanced_score(
                        tkr,
                        df.rename(columns=str.title),
                        bench_df.rename(columns=str.title),
                        base_score,
                    )
                    # Only keep Meteor passes
                    if details.get("passed"):
                        row = row.copy()
                        row["FinalScore_20d"] = float(new_score)
                        row["Score"] = float(new_score)
                        row["Meteor_Passed"] = True
                        row["Meteor_Reason"] = details.get("reason")
                        kept_rows.append(row)
                results = pd.DataFrame(kept_rows)
                logger.info(f"[PIPELINE] Meteor Mode: {len(results)} candidates after filters")
        except Exception as e:
            logger.warning(f"Meteor filter application failed: {e}")

    # 5. Fundamentals & Sector Enrichment (only for Tier 1 passed stocks)
    # Track fundamentals status locally to enable TECH_ONLY fallback in allocation/meta
    fundamentals_status_local = "not_requested"
    if config.get("fundamental_enabled", True):
        if status_callback: status_callback("Fetching fundamentals & sector data...")
        # Choose scan date as the signal date (not wall-clock)
        fund_as_of = None
        try:
            if "As_Of_Date" in results.columns and len(results) > 0:
                fund_as_of = pd.to_datetime(results["As_Of_Date"].iloc[0]).date()
            else:
                # Fallback: derive from latest history across universe
                dates = []
                for df in (data_map or {}).values():
                    if df is not None and not df.empty:
                        try:
                            dates.append(pd.to_datetime(df.index[-1]).date())
                        except Exception:
                            pass
                if dates:
                    fund_as_of = max(dates)
        except Exception:
            fund_as_of = None

        # IMPORTANT: Only fetch fundamentals for high-Score candidates with an optional Top-N cap
        try:
            score_thr = float(config.get("fundamentals_score_threshold", float(os.getenv("FUNDAMENTALS_SCORE_THRESHOLD", "60"))))
        except Exception:
            score_thr = 60.0
        # Top-N cap from config/env - default 50 for wide scans; set 0 to disable cap
        try:
            top_n_cap = int(config.get("fundamentals_top_n_cap", int(os.getenv("FUNDAMENTALS_TOP_N_CAP", "50"))))
        except Exception:
            top_n_cap = 50
        if top_n_cap > 0:
            logger.info(f"⚡ Fetching fundamentals for Top-{top_n_cap} stocks (score > {score_thr})")
        else:
            logger.info(f"⚡ Fetching fundamentals for ALL stocks with score > {score_thr} (no cap)")

        # Filter and cap by top-N, prioritizing highest technical scores
        # Ensure scoring columns exist to avoid KeyError in edge cases
        try:
            if ("Score" not in results.columns) and ("FinalScore_20d" in results.columns):
                results["Score"] = results["FinalScore_20d"]
            if "TechScore_20d" not in results.columns:
                if "Score" in results.columns:
                    results["TechScore_20d"] = results["Score"]
                elif "FinalScore_20d" in results.columns:
                    results["TechScore_20d"] = results["FinalScore_20d"]
        except Exception:
            pass
        # Pick safest available score column
        score_col = "TechScore_20d" if "TechScore_20d" in results.columns else ("Score" if "Score" in results.columns else None)
        if score_col is None or results.empty:
            eligible = pd.DataFrame(columns=results.columns)
        else:
            eligible = results[results[score_col] > score_thr].sort_values(score_col, ascending=False)
        if top_n_cap > 0:
            eligible = eligible.head(top_n_cap)
        try:
            if "Ticker" in eligible.columns:
                winners = eligible["Ticker"].tolist()
            elif eligible.index.name == "Ticker":
                winners = eligible.index.tolist()
            elif "ticker" in eligible.columns:
                winners = [str(x).upper().replace('.', '-').replace('/', '-') for x in eligible["ticker"].tolist()]
            else:
                winners = []
        except Exception:
            winners = []

        if not winners:
            logger.info(f"No winners with Score > {score_thr:.0f}; skipping fundamentals fetch")
            fund_df = pd.DataFrame()
            try:
                telemetry.set_value("fundamentals_status", "requested_empty")
                fundamentals_status_local = "requested_empty"
            except Exception:
                pass
        else:
            logger.info(f"Fetching fundamentals for {len(winners)} winners ({score_col} > {score_thr:.0f}, TopN={top_n_cap})")
            # Explicit progress log for batch size being sent to data layer
            logger.info(f"[PIPELINE] Sending {len(winners)} tickers to fetch_fundamentals_batch")
            # Pass provider_status so data layer strictly skips failed providers and promotes alternates
            fund_df = fetch_fundamentals_batch(winners, provider_status=provider_status, as_of_date=fund_as_of, telemetry=telemetry)
            try:
                status_val = "used" if (isinstance(fund_df, pd.DataFrame) and not fund_df.empty) else "requested_empty"
                telemetry.set_value("fundamentals_status", status_val)
                fundamentals_status_local = status_val
            except Exception:
                pass
        
        # Properly handle index/column ambiguity
        if isinstance(fund_df.index, pd.Index):
            fund_df = fund_df.reset_index()
            # Rename index column to Ticker if it exists
            if 'ticker' in fund_df.columns:
                fund_df = fund_df.rename(columns={'ticker': 'Ticker'})
            elif 'index' in fund_df.columns and 'Ticker' not in fund_df.columns:
                fund_df = fund_df.rename(columns={'index': 'Ticker'})
        
        # Ensure Ticker column exists
        if "Ticker" not in fund_df.columns and len(fund_df) > 0:
            # Last resort: use the first column as Ticker
            if len(fund_df.columns) > 0:
                first_col = fund_df.columns[0]
                if fund_df[first_col].dtype == 'object':  # String column
                    fund_df = fund_df.rename(columns={first_col: 'Ticker'})
            
        # Merge fundamentals (only if we have valid Ticker column)
        if "Ticker" in fund_df.columns and len(fund_df) > 0:
            results = pd.merge(results, fund_df, on="Ticker", how="left", suffixes=("", "_fund"))
        else:
            logger.warning("Fundamental data has no Ticker column, skipping merge")
            # Treat as fundamentals unavailable for this run
            try:
                telemetry.set_value("fundamentals_status", "requested_empty")
                fundamentals_status_local = "requested_empty"
            except Exception:
                pass
        
        # Map to UI-expected keys and fill missing via safe FMP fetch
        try:
            # First, derive canonical UI columns from aggregated fields when present
            # Lowercase agg -> UI uppercase names
            if "market_cap" in results.columns and "Market_Cap" not in results.columns:
                results["Market_Cap"] = results["market_cap"]
            if "beta" in results.columns and "Beta" not in results.columns:
                results["Beta"] = results["beta"]
            if "sector" in results.columns and "Sector" not in results.columns:
                results["Sector"] = results["sector"]
            if "pe" in results.columns and "PE_Ratio" not in results.columns:
                results["PE_Ratio"] = results["pe"]
            if "peg" in results.columns and "PEG_Ratio" not in results.columns:
                results["PEG_Ratio"] = results["peg"]
            if "debt_equity" in results.columns and "Debt_to_Equity" not in results.columns:
                results["Debt_to_Equity"] = results["debt_equity"]

            # Ensure numeric types or NaN
            for col in ["Market_Cap", "PE_Ratio", "PEG_Ratio", "Beta", "Debt_to_Equity"]:
                if col in results.columns:
                    results[col] = pd.to_numeric(results[col], errors="coerce")

            # Fill missing via get_fundamentals_safe per ticker
            ui_cols = [
                "Market_Cap", "PE_Ratio", "PEG_Ratio", "PB_Ratio",
                "Beta", "Sector", "Industry", "Debt_to_Equity",
                "ROE", "Vol_Avg", "Dividend", "Price"
            ]
            for idx, row in results.iterrows():
                tkr = row.get("Ticker")
                if not tkr:
                    continue
                need_fill = any(pd.isna(row.get(c)) for c in ui_cols)
                if not need_fill:
                    continue
                safe = get_fundamentals_safe(str(tkr))
                if not safe:
                    continue
                for c in ui_cols:
                    if pd.isna(row.get(c)) and (c in safe):
                        results.at[idx, c] = safe.get(c)

            # Add explicit Valuation, Quality, and Leverage for UI
            # Valuation: use PE_Ratio directly per spec
            if "PE_Ratio" in results.columns:
                results["Valuation"] = pd.to_numeric(results["PE_Ratio"], errors="coerce")
            else:
                results["Valuation"] = np.nan
            # Quality: use ROE directly per spec
            if "ROE" in results.columns:
                results["Quality"] = pd.to_numeric(results["ROE"], errors="coerce")
            elif "roe" in results.columns:
                results["Quality"] = pd.to_numeric(results["roe"], errors="coerce")
            else:
                results["Quality"] = np.nan
            # Leverage: use Debt_to_Equity
            if "Debt_to_Equity" in results.columns:
                results["Leverage"] = pd.to_numeric(results["Debt_to_Equity"], errors="coerce")
            else:
                results["Leverage"] = np.nan
        except Exception as e:
            logger.debug(f"UI mapping/Valuation-Leverage setup skipped: {e}")

        # Add explicit Valuation and Quality helper columns to avoid NaN in UI cards
        try:
            # Raw metrics aliases
            if "pe" in results.columns and "PE" not in results.columns:
                results["PE"] = results["pe"]
            if "peg" in results.columns and "PEG" not in results.columns:
                results["PEG"] = results["peg"]
            if "roe" in results.columns and "ROE" not in results.columns:
                results["ROE"] = results["roe"]
            if "debt_equity" in results.columns and "Debt_Equity" not in results.columns:
                results["Debt_Equity"] = results["debt_equity"]

            # Compute Valuation: favor lower PE and PEG; use PE/PEG when both present
            def _valuation_row(row: pd.Series) -> float:
                pe = row.get("pe")
                peg = row.get("peg")
                try:
                    if pd.notna(pe) and pd.notna(peg) and float(peg) > 0:
                        return float(pe) / float(peg)
                    elif pd.notna(pe):
                        return float(pe)
                    else:
                        return 0.0
                except Exception:
                    return 0.0

            # Compute Quality: emphasize ROE vs Debt/Equity
            def _quality_row(row: pd.Series) -> float:
                roe = row.get("roe")
                de = row.get("debt_equity")
                try:
                    base_roe = float(roe) if pd.notna(roe) else 0.0
                    base_de = float(de) if (pd.notna(de) and float(de) > 0) else 1.0
                    return float(base_roe) / float(base_de)
                except Exception:
                    return 0.0

            # Legacy Quality helper no longer used here; Quality set from ROE above
        except Exception as e:
            logger.debug(f"Valuation/Quality column creation skipped: {e}")
        
        # Compute fundamental scores - always recalculate based on merged fund_df data
        # The default 50.0 from fetch_fundamentals_batch needs to be replaced with actual calculation
        for idx, row in results.iterrows():
            try:
                fund_data = row.to_dict()
                # Only calculate if we have actual fundamental data (pe, roe, etc.)
                has_fund_data = any(pd.notna(fund_data.get(f)) for f in ['pe', 'roe', 'pb', 'margin', 'debt_equity'])
                if not has_fund_data:
                    results.at[idx, "Fundamental_S"] = 50.0
                    continue
                fund_score_obj = compute_fundamental_score_with_breakdown(fund_data)
                results.at[idx, "Fundamental_S"] = fund_score_obj.total
                results.at[idx, "Quality_Score_F"] = fund_score_obj.breakdown.quality_score
                results.at[idx, "Growth_Score_F"] = fund_score_obj.breakdown.growth_score
                results.at[idx, "Valuation_Score_F"] = fund_score_obj.breakdown.valuation_score
            except Exception as e:
                results.at[idx, "Fundamental_S"] = 50.0  # Neutral default
                logger.debug(f"Fundamental scoring failed for {row.get('Ticker')}: {e}")
        
        # Extract Sector from fundamentals (if available)
        if "sector" in results.columns:
            results["Sector"] = results["sector"].fillna("Unknown")
        elif "Sector" not in results.columns:
            results["Sector"] = "Unknown"
    else:
        results["Sector"] = "Unknown"
        results["Fundamental_S"] = 50.0
        try:
            telemetry.set_value("fundamentals_status", "not_requested")
            fundamentals_status_local = "not_requested"
        except Exception:
            pass

    # Apply sector mapping fallback for unknown or potentially incorrect sectors
    try:
        from core.sector_mapping import get_stock_sector
        for idx, row in results.iterrows():
            ticker = row.get("Ticker", "")
            current_sector = row.get("Sector", "Unknown")
            # Apply mapping if sector is Unknown or if we have a known mapping
            mapped_sector = get_stock_sector(ticker)
            if mapped_sector != "Unknown":
                # Use mapped sector (more reliable than API data)
                results.at[idx, "Sector"] = mapped_sector
            # Keep current sector if mapping returns Unknown and current is not Unknown
    except Exception as e:
        logger.debug(f"Sector mapping fallback failed: {e}")

    # CRITICAL: Recalculate FinalScore_20d with fundamentals NOW integrated
    # The scoring engine needs the Fundamental_S field which was just added
    try:
        from core.scoring_engine import compute_final_score_20d
        for idx, row in results.iterrows():
            try:
                new_score = compute_final_score_20d(row)
                results.at[idx, "FinalScore_20d"] = float(new_score)
                results.at[idx, "Score"] = float(new_score)
            except Exception as e:
                logger.debug(f"FinalScore recalc failed for {row.get('Ticker')}: {e}")
        logger.info(f"[PIPELINE] Recalculated FinalScore_20d with fundamentals for {len(results)} stocks")
    except Exception as e:
        logger.warning(f"FinalScore recalc skipped: {e}")
    
    # Preserve and map source metadata to canonical UI columns
    try:
        if "sources_used" in results.columns:
            results["fund_sources_used_v2"] = results["sources_used"].apply(lambda x: len(x) if isinstance(x, list) else 0)
            results["Fundamental_Sources_Count"] = results["fund_sources_used_v2"]
        if "price_sources" in results.columns:
            # price_sources is expected to be an integer count from the data layer
            results["price_sources_used_v2"] = results["price_sources"].fillna(0).astype(int)
            results["Price_Sources_Count"] = results["price_sources_used_v2"]
    except Exception as e:
        logger.debug(f"Source metadata mapping skipped due to error: {e}")
        
    # 7. Classification & Allocation
    if status_callback: status_callback("Classifying & Allocating...")
    results = apply_classification(results)
    
    # 8. Earnings Blackout Check (optional, for top candidates)
    if config.get("EARNINGS_BLACKOUT_DAYS", 0) > 0:
        topk = int(config.get("EARNINGS_CHECK_TOPK", 30))
        blackout_days = int(config.get("EARNINGS_BLACKOUT_DAYS", 7))
        if status_callback: status_callback(f"Checking earnings blackout (top {topk})...")
        
        # Import earnings check function
        try:
            
            def check_earnings_blackout(ticker: str, days: int) -> bool:
                """Check if earnings are within next N days"""
                try:
                    info = yf.Ticker(ticker).calendar
                    if info is not None and 'Earnings Date' in info:
                        earnings_dates = info['Earnings Date']
                        if earnings_dates is not None and len(earnings_dates) > 0:
                            next_date = pd.to_datetime(earnings_dates[0])
                            days_until = (next_date - datetime.now()).days
                            return 0 <= days_until <= days
                except Exception as e:
                    logger.debug(f"Earnings check failed for {ticker}: {e}")
                return False
            
            # Check top K stocks only (performance optimization)
            top_indices = results.nlargest(topk, "Score").index
            for idx in top_indices:
                ticker = results.at[idx, "Ticker"]
                if check_earnings_blackout(ticker, blackout_days):
                    logger.info(f"[EARNINGS] {ticker} has earnings within {blackout_days} days - reducing allocation")
                    # Reduce buy amount by 50% (conservative approach)
                    if "buy_amount_v2" in results.columns:
                        results.at[idx, "buy_amount_v2"] *= 0.5
        except Exception as e:
            logger.warning(f"Earnings blackout check failed: {e}")
    
    # Allocation-Free Signal Engine: skip allocation entirely
    if False and "buy_amount_v2" not in results.columns:
        results = allocate_budget(
            results,
            config.get("BUDGET_TOTAL", 5000),
            config.get("MIN_POSITION", 500),
            config.get("MAX_POSITION_PCT", 0.2),
        )
    
    # STRICT ENFORCEMENT: Score must always equal FinalScore_20d
    # This is the final safety check before returning results
    if "FinalScore_20d" in results.columns:
        results["Score"] = results["FinalScore_20d"]
        logger.info(f"[PIPELINE] Final check: Score column set to FinalScore_20d for all {len(results)} results")

    # --- Add Tier debug columns to results ---
    try:
        # Tier 1: all present rows passed Tier 1
        results["Tier1_Passed"] = True
        results["Tier1_Reasons"] = ""
        # Tier 2: mark as False only if ADVANCED_REJECT present in diagnostics
        def _t2_pass_and_reasons(row: pd.Series):
            tkr = str(row.get("Ticker"))
            rec = diagnostics.get(tkr, {}) if isinstance(diagnostics, dict) else {}
            t2 = rec.get("tier2_reasons") or []
            has_adv_reject = any((r.get("rule") == "ADVANCED_REJECT") for r in t2 if isinstance(r, dict))
            # Include RejectionReason text if present
            reasons_rules = [str(r.get("rule")) for r in t2 if isinstance(r, dict) and r.get("rule")]
            reasons_text = row.get("RejectionReason")
            joined = ";".join(reasons_rules) if reasons_rules else (str(reasons_text) if reasons_text else "")
            return (not has_adv_reject, joined)
        t2_vals = results.apply(_t2_pass_and_reasons, axis=1, result_type="expand")
        if isinstance(t2_vals, pd.DataFrame) and t2_vals.shape[1] == 2:
            results["Tier2_Passed"] = t2_vals.iloc[:,0]
            results["Tier2_Reasons"] = t2_vals.iloc[:,1]
        else:
            results["Tier2_Passed"] = True
            results["Tier2_Reasons"] = ""
    except Exception as e:
        logger.debug(f"Tier debug columns setup skipped: {e}")
    
    # --- Dynamic Risk/Reward based on resistance & Bollinger ---
    try:
        def _compute_rr_for_row(row: pd.Series) -> Dict[str, Any]:
            tkr = str(row.get("Ticker"))
            hist = data_map.get(tkr)
            if hist is None or len(hist) < 5:
                return {
                    "Entry_Price": np.nan,
                    "Target_Price": np.nan,
                    "Stop_Loss": np.nan,
                    "RewardRisk": np.nan,
                    "RR_Ratio": np.nan,
                    "RR": np.nan,
                    "Target_Source": "N/A",
                }
            try:
                # Ensure expected columns
                hdf = hist.copy()
                if "Close" not in hdf.columns or "High" not in hdf.columns or "Low" not in hdf.columns:
                    return {
                        "Entry_Price": np.nan,
                        "Target_Price": np.nan,
                        "Stop_Loss": np.nan,
                        "RewardRisk": np.nan,
                        "RR_Ratio": np.nan,
                        "RR": np.nan,
                        "Target_Source": "N/A",
                    }
                # Entry price = latest close
                entry = float(hdf["Close"].iloc[-1])
                # ATR(14) using simple true range approximation
                close_shift = hdf["Close"].shift(1)
                tr = pd.concat([
                    (hdf["High"] - hdf["Low"]),
                    (hdf["High"] - close_shift).abs(),
                    (hdf["Low"] - close_shift).abs()
                ], axis=1).max(axis=1)
                atr14 = float(tr.rolling(14, min_periods=5).mean().iloc[-1]) if len(tr) >= 5 else float((hdf["High"] - hdf["Low"]).tail(5).mean())
                atr14 = max(atr14, 1e-6)
                # Stop loss = min(Low of last 5 days, entry - 2*ATR)
                low_5 = float(hdf["Low"].tail(5).min())
                stop_price = float(min(low_5, entry - 2.0 * atr14))
                # Bollinger upper band (20, 2)
                ma20 = float(hdf["Close"].rolling(20, min_periods=5).mean().iloc[-1])
                std20 = float(hdf["Close"].rolling(20, min_periods=5).std(ddof=0).iloc[-1])
                bb_upper = ma20 + 2.0 * std20 if np.isfinite(ma20) and np.isfinite(std20) else float(hdf["High"].tail(20).max())
                # Resistance = highest high last 60 days
                res_60 = float(hdf["High"].tail(60).max())
                target = float(max(res_60, bb_upper))
                # Compute RR = (target - entry) / (entry - stop)
                risk = float(entry - stop_price)
                reward = float(target - entry)
                rr = np.nan
                if risk > 0 and reward > 0:
                    rr = float(np.clip(reward / risk, 0.0, 15.0))  # Increased cap from 10 to 15
                return {
                    "Entry_Price": entry,
                    "Target_Price": target,
                    "Stop_Loss": stop_price,
                    "RewardRisk": rr,
                    "RR_Ratio": rr,
                    "RR": rr,
                    "Target_Source": "Resistance/Bollinger",
                }
            except Exception:
                return {
                    "Entry_Price": np.nan,
                    "Target_Price": np.nan,
                    "Stop_Loss": np.nan,
                    "RewardRisk": np.nan,
                    "RR_Ratio": np.nan,
                    "RR": np.nan,
                    "Target_Source": "N/A",
                }

        # Apply to all rows
        rr_updates = results.apply(_compute_rr_for_row, axis=1, result_type="expand")
        for col in ["Entry_Price", "Target_Price", "Stop_Loss", "RewardRisk", "RR_Ratio", "RR", "Target_Source"]:
            if col in rr_updates.columns:
                results[col] = rr_updates[col]

        # Adjust score for low RR
        if "FinalScore_20d" in results.columns and "RR" in results.columns:
            low_mask = pd.to_numeric(results["RR"], errors="coerce") < 1.5
            mid_mask = (~low_mask) & (pd.to_numeric(results["RR"], errors="coerce") < 2.0)
            results.loc[low_mask, "FinalScore_20d"] = results.loc[low_mask, "FinalScore_20d"] - 8.0
            results.loc[mid_mask, "FinalScore_20d"] = results.loc[mid_mask, "FinalScore_20d"] - 3.0
            # Keep alias in sync
            results["FinalScore"] = results["FinalScore_20d"]
            results["Score"] = results["FinalScore_20d"]
    except Exception as e:
        logger.warning(f"[PIPELINE] Dynamic RR computation failed: {e}")

    # --- Signal-First Filtering & Ranking (Allocation-Free) ---
    # Track postfilter mode for diagnostics/meta (initialized from early decision)
    postfilter_mode: Optional[str] = postfilter_mode_global
    try:
        orig_len = len(results)
        score_col = "FinalScore_20d" if "FinalScore_20d" in results.columns else ("Score" if "Score" in results.columns else None)
        if score_col is not None and not results.empty:
            sc = pd.to_numeric(results[score_col], errors="coerce")
            mlp = pd.to_numeric(results.get("ML_20d_Prob", pd.Series(index=results.index)), errors="coerce")
            patt = pd.to_numeric(results.get("Pattern_Score", pd.Series(index=results.index)), errors="coerce")
            mask = (
                (sc >= float(SIGNAL_MIN_SCORE)) |
                (mlp >= float(ML_PROB_THRESHOLD)) |
                (patt.fillna(0.0) > 0.0)
            )
            filtered = results[mask].copy() if isinstance(mask, pd.Series) else results.copy()
            # Sort by final score desc, then ML prob desc
            # Prefer sorting by score, then by number of signal reasons, then ML prob
            sort_cols = ["_score_numeric"]
            asc = [False]
            if "SignalReasons_Count" in filtered.columns:
                sort_cols.append("SignalReasons_Count")
                asc.append(False)
            sort_cols.append("ML_20d_Prob")
            asc.append(False)
            filtered = (
                filtered.assign(_score_numeric=pd.to_numeric(filtered[score_col], errors="coerce"))
                        .sort_values(by=sort_cols, ascending=asc)
                        .drop(columns=["_score_numeric"])
            )
            topn = int(config.get("topn_results", TOP_SIGNAL_K))
            if filtered.empty:
                # Fallback: keep top-N by score only to avoid zero results
                # Fallback: sort by score, reasons count if present, and ML prob
                sort_cols_fb = ["_score_numeric"]
                asc_fb = [False]
                if "SignalReasons_Count" in results.columns:
                    sort_cols_fb.append("SignalReasons_Count")
                    asc_fb.append(False)
                sort_cols_fb.append("ML_20d_Prob")
                asc_fb.append(False)
                fallback = (
                    results.assign(_score_numeric=pd.to_numeric(results[score_col], errors="coerce"))
                           .sort_values(by=sort_cols_fb, ascending=asc_fb)
                           .drop(columns=["_score_numeric"])
                )
                results = fallback.head(topn).reset_index(drop=True)
                logger.info(f"[PIPELINE] Signal thresholds yielded no candidates; using top-{topn} by score as fallback")
            else:
                results = filtered.head(topn).reset_index(drop=True)
                logger.info(f"[PIPELINE] Signal-First ranking applied: kept {len(results)} of {orig_len}; thresholds: score>={SIGNAL_MIN_SCORE}, ml>={ML_PROB_THRESHOLD}")
            postfilter_mode = "signal_only"
        else:
            logger.info("[PIPELINE] No score column for signal filter; keeping all")
    except Exception as e:
        logger.warning(f"[PIPELINE] Signal-first filtering failed: {e}")

    # --- Persist latest results for Streamlit dashboard freshness ---
    try:
        # What-You-See-Is-What-You-Save: save exactly the final filtered results
        to_save = results.copy()

        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        latest_json = data_dir / "latest_scan_live.json"
        latest_parquet = data_dir / "latest_scan_live.parquet"

        # Save JSON (records, ISO dates)
        to_save.to_json(latest_json, orient="records", date_format="iso")
        # Save Parquet
        to_save.to_parquet(latest_parquet, index=False)
        logger.info(f"✅ Pipeline Finalized: Saved strict Top {len(to_save)} recommendations")
    except Exception as e:
        logger.warning(f"[PIPELINE] Failed to persist latest scan files: {e}")

    # Attach Tier 1 filtered summary to data_map for downstream/UI
    try:
        if isinstance(data_map, dict):
            data_map = dict(data_map)
            data_map["filtered_tier1_df"] = filtered_df
    except Exception:
        pass

    # --- Build meta wrapper ---
    # --- Build meta wrapper ---
    meta = {
        "engine_version": "pipeline_v2",
        "engine_mode": "SIGNAL_ONLY",
        "used_legacy_fallback": bool(_LEGACY_FALLBACK_USED),
        "fallback_reason": ", ".join(sorted(set(_LEGACY_FALLBACK_REASONS))) if _LEGACY_FALLBACK_REASONS else None,
        "sources_used": telemetry.export(),
        "run_timestamp_utc": datetime.utcnow().isoformat(),
        "postfilter_mode": postfilter_mode,
        # If fundamentals were unavailable, degrade run_mode for UI truthfulness
        "run_mode": ("DEGRADED_TECH_ONLY" if (run_mode == "OK" and fundamentals_status_local in ("not_requested", "requested_empty")) else run_mode),
        "benchmark_status": benchmark_status,
    }
    try:
        meta.update(get_ml_health_meta())
        try:
            if meta.get("ml_bundle_version_warning"):
                meta["ml_mode"] = "DISABLED_VERSION_MISMATCH"
            elif not ML_20D_AVAILABLE:
                meta["ml_mode"] = "DISABLED_NO_MODEL"
            else:
                meta["ml_mode"] = "HYBRID"
        except Exception:
            pass
    except Exception:
        pass

    # === PIPELINE COMPLETION SUMMARY (with fallback status) ===
    try:
        n_results = len(results) if results is not None else 0
        fb_status = get_fallback_status()
        if fb_status.get("fallback_used"):
            fb_count = fb_status.get("fallback_count", 0)
            logger.warning(f"⚠️ [PIPELINE COMPLETE] {n_results} stocks scored. FALLBACK USED: {fb_count} legacy scoring fallbacks occurred!")
            logger.warning(f"   Fallback reasons: {', '.join(fb_status.get('reasons', [])[:5])}")
        else:
            logger.info(f"✓ [PIPELINE COMPLETE] {n_results} stocks scored successfully. No fallbacks needed.")
    except Exception:
        pass

    return {"result": {"results_df": results, "data_map": data_map, "diagnostics": diagnostics}, "meta": meta}


def run_scan(*args, **kwargs):
    """Backward-compatible wrapper for legacy imports expecting `run_scan`.

    Delegates to `run_scan_pipeline`. If the pipeline returns a tuple
    `(results_df, data_map)`, convert it into a `ScanResult` contracts
    object for downstream consumers. If it already returns a `ScanResult`,
    pass it through unchanged.

    Signature is intentionally flexible to pass through positional and
    keyword arguments without altering pipeline behavior.
    """
    out = run_scan_pipeline(*args, **kwargs)
    # If pipeline returns wrapper, convert to contracts for backward consumers
    if isinstance(out, dict) and ("result" in out):
        payload = out.get("result")
        # New schema: dict with keys
        if isinstance(payload, dict) and ("results_df" in payload):
            _results_df = payload.get("results_df")
            results_df = _results_df if _results_df is not None else pd.DataFrame()
            data_map = payload.get("data_map") or {}
        else:
            # Backward: tuple/list
            try:
                results_df, data_map = payload
            except Exception:
                results_df = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame()
                data_map = {}
    elif isinstance(out, ScanResult):
        return out
    else:
        # Unknown shape; pass through unchanged
        return out

    # Build minimal metadata/diagnostics placeholders (not asserted in tests)
    try:
        now = datetime.utcnow()
    except Exception:
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
            universe_size=int(len(results_df) if hasattr(results_df, "__len") else 0),
            fetched_history_count=int(len(data_map) if isinstance(data_map, dict) else 0),
            fundamentals_count=int(results_df.get("Fundamental_S", pd.Series()).notna().sum()) if isinstance(results_df, pd.DataFrame) else 0,
            external_price_verified_count=0,
            earnings_known_count=0,
        ),
        fallbacks=FallbackCounts(alpha_to_finnhub=0, polygon_to_tiingo=0, other_fallbacks={}),
        disagreement=DisagreementStats(price_std_mean=None, high_disagreement_count=0, agreement_pct=None),
        ml_status=MLStatus(
            enabled=ML_20D_AVAILABLE,
            model_name=None,
            model_version=None,
            loaded=True,
            inference_ok=True,
            inference_count=int(len(results_df)) if isinstance(results_df, pd.DataFrame) else 0,
            last_trained_ts=None,
            metrics=None,
        ),
        market_context_status=MarketContextStatus(
            ok=True,
            regime=str(results_df.get("Market_Regime").iloc[0]) if isinstance(results_df, pd.DataFrame) and "Market_Regime" in results_df.columns and len(results_df) > 0 else None,
            computed_at=now,
            vix_level=None,
            beta_mean=None,
            spx_trend=None,
            note=None,
        ),
    )

    # Convert each row to a Recommendation contract
    recs: List[Recommendation] = []
    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        for _, row in results_df.iterrows():
            ticker = str(row.get("Ticker")) if row.get("Ticker") is not None else None
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
            # Build breakdown from known columns present on the row
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
                    if val is not None and np.isfinite(_to_float(val)):
                        breakdown[key] = float(val)
            # Reasons
            reasons: List[str] = []
            if row.get("RejectionReason"):
                try:
                    reasons.append(str(row.get("RejectionReason")))
                except Exception:
                    pass
            # Classification fields
            risk_class = row.get("RiskClass") if "RiskClass" in results_df.columns else None
            safety_blocked = row.get("SafetyBlocked") if "SafetyBlocked" in results_df.columns else None
            safety_reasons = row.get("SafetyReasons") if "SafetyReasons" in results_df.columns else None
            # Legacy classification fields (optional)
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
                safety_blocked=bool(safety_blocked) if safety_blocked is not None else None,
                safety_reasons=str(safety_reasons) if safety_reasons is not None else None,
                risk_level=row.get("Risk_Level") if "Risk_Level" in results_df.columns else None,
                data_quality=row.get("Data_Quality") if "Data_Quality" in results_df.columns else None,
                confidence_level=row.get("Confidence_Level") if "Confidence_Level" in results_df.columns else None,
                should_display=bool(row.get("Should_Display")) if "Should_Display" in results_df.columns else None,
                consolidation_ratio=_to_float(row.get("Consolidation_Ratio")) if "Consolidation_Ratio" in results_df.columns else None,
                pocket_pivot_ratio=_to_float(row.get("Pocket_Pivot_Ratio")) if "Pocket_Pivot_Ratio" in results_df.columns else None,
                vcp_ratio=_to_float(row.get("VCP_Ratio")) if "VCP_Ratio" in results_df.columns else None,
            )
            recs.append(rec)

    return ScanResult(metadata=metadata, diagnostics=diagnostics, recommendations=recs)


def run_scan_smoke() -> Dict[str, Any]:
    """Run a tiny smoke test on a single ticker and print meta.

    Returns the wrapper dict {"result": ..., "meta": ...} for inspection.
    """
    logging.basicConfig(level=logging.INFO)
    # Explicitly mark universe provider for smoke/manual runs
    try:
        global LAST_UNIVERSE_PROVIDER
        LAST_UNIVERSE_PROVIDER = "smoke/manual"
    except Exception:
        pass
    try:
        cfg = _normalize_config(get_config())
        cfg["fundamental_enabled"] = False
        cfg["beta_filter_enabled"] = False
        # Force deterministic smoke: Meteor off, and mark smoke_mode
        cfg["meteor_mode"] = False
        cfg["smoke_mode"] = True
    except Exception:
        cfg = {"fundamental_enabled": False, "beta_filter_enabled": False}
    out = run_scan_pipeline(["AAPL"], cfg, status_callback=None)
    try:
        meta = out.get("meta", {})
        # Record smoke overrides in meta for visibility
        try:
            meta["smoke_overrides"] = {"meteor_mode_forced_off": True}
        except Exception:
            pass
        print("Smoke Meta:", meta)
    except Exception:
        pass
    return out


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        val = float(x)
        if np.isfinite(val):
            return val
        return None
    except Exception:
        return None


def main():
    """Main entry for production scan: 1500-2000 US stocks, strict mode.

    Steps:
    - Preflight keys
    - Initialize global market context
    - Fetch top market cap universe via FMP
    - Run pipeline
    - Filter Final_Score > 70 AND Market_Regime == 'TREND_UP'
    - For Top 10, fetch Finnhub news and analyze sentiment with OpenAI
    - Print JSON summary to stdout
    """
    logging.basicConfig(level=logging.INFO)
    preflight_check()

    # Initialize market context early to populate global caches
    from core.market_context import initialize_market_context
    logger.info("Initializing Market Context...")
    initialize_market_context()
    try:
        from core.data_sources_v2 import get_last_index_source
        src = get_last_index_source('SPY') or 'Unknown'
        logger.info(f"[SUCCESS via {src}] Market context initialized")
    except Exception:
        logger.info("[SUCCESS] Market context initialized")

    # Universe (use configured limit)
    cfg_probe = get_config()
    universe = fetch_top_us_tickers_by_market_cap(limit=int(getattr(cfg_probe, 'universe_limit', 2000)))
    logger.info(f"Fetched {len(universe)} tickers")

    # Config
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

    # Apply final filter
    filtered = results[(results["FinalScore_20d"] > 70.0) & (results["Market_Regime"].str.upper() == "TREND_UP")].copy()
    filtered = filtered.sort_values("FinalScore_20d", ascending=False)
    top10 = filtered.head(10)

    # Enrich with sentiment
    output = []
    for _, row in top10.iterrows():
        tkr = str(row.get("Ticker"))
        news_items = fetch_latest_company_news(tkr, count=5)
        headlines = [n.get("headline") or n.get("title") or "" for n in news_items]
        headlines = [h for h in headlines if h]
        sentiment = analyze_sentiment_openai(headlines)
        output.append({
            "Ticker": tkr,
            "FinalScore_20d": float(row.get("FinalScore_20d", 0.0)),
            "TechScore_20d": float(row.get("TechScore_20d", 0.0)),
            "Fundamental_Score": float(row.get("Fundamental_Score", row.get("Fundamental_S", 0.0))),
            "ML_20d_Prob": float(row.get("ML_20d_Prob", 0.5)),
            "Market_Regime": str(row.get("Market_Regime")),
            "Sector": str(row.get("Sector", "Unknown")),
            "Sentiment": sentiment,
        })

    # Print JSON table
    import json
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()