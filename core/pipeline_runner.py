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

# For backward compatibility with code that checks ML availability
from core.ml_20d_inference import ML_20D_AVAILABLE

logger = logging.getLogger(__name__)

# Public diagnostic: last provider used for universe construction
LAST_UNIVERSE_PROVIDER: str = "Unknown"

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

    # --- Primary: FMP stock screener (more permissive) ---
    try:
        from core.config import get_secret
        api_key = get_secret("FMP_API_KEY", "")
        if api_key:
            url = "https://financialmodelingprep.com/api/v3/stock-screener"
            try:
                min_cap = int(os.getenv("MIN_MCAP", "300000000"))
            except Exception:
                min_cap = 300_000_000
            try:
                # Target pre-jump candidates: cap upper bound at ~$10B by default
                max_cap = int(os.getenv("MAX_MCAP", "10000000000"))
            except Exception:
                max_cap = 10_000_000_000
            params = {
                "marketCapMoreThan": max(min_cap, 0),
                "marketCapLowerThan": max_cap,
                "limit": 2000,
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
    try:
        poly_key = os.getenv("POLYGON_API_KEY", "")
        if poly_key:
            url = "https://api.polygon.io/v3/reference/tickers"
            base_params = {
                "market": "stocks",
                "type": "CS",
                "active": "true",
                "limit": 1000,
                "apiKey": poly_key,
            }
            tickers: List[str] = []
            next_url: Optional[str] = None
            attempts = 0
            while len(tickers) < limit and attempts < 15:
                if next_url:
                    # Ensure apiKey present on next_url
                    if "apiKey=" not in next_url:
                        sep = '&' if '?' in next_url else '?'
                        next_url = f"{next_url}{sep}apiKey={poly_key}"
                    r = requests.get(next_url, timeout=10)
                else:
                    r = requests.get(url, params=base_params, timeout=10)
                if r.status_code == 429:
                    time.sleep(1.0)
                    attempts += 1
                    continue
                if r.status_code != 200:
                    logger.warning(f"Polygon reference/tickers failed: HTTP {r.status_code}")
                    break
                payload = r.json() or {}
                results = payload.get("results") or []
                for it in results:
                    sym = it.get("ticker")
                    if sym:
                        tickers.append(sym)
                next_url = payload.get("next_url")
                attempts += 1
                if not next_url:
                    break
            out = _normalize_symbols(tickers[:limit])
            if out:
                logger.info(f"✓ Universe from Polygon: {len(out)} tickers")
                LAST_UNIVERSE_PROVIDER = "Polygon"
                return out
        else:
            logger.warning("POLYGON_API_KEY missing; skipping Polygon universe fetch")
    except Exception as e:
        logger.warning(f"Polygon universe fetch errored: {e}")

    # --- Fallback 2: EODHD or Nasdaq ---
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

    # --- Fallback 3: Local/Gist S&P 500 ---
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        # Prefer root-level file if present (typically contains full 500 list),
        # then fall back to data/sp500_tickers.txt
        candidates = [
            os.path.join(base_dir, "sp500_tickers.txt"),
            os.path.join(base_dir, "data", "sp500_tickers.txt"),
        ]
        for local_path in candidates:
            if os.path.exists(local_path):
                with open(local_path, "r") as f:
                    syms = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
                logger.warning(f"Using local S&P 500 fallback: {len(syms)} tickers from {os.path.relpath(local_path, base_dir)}")
                LAST_UNIVERSE_PROVIDER = "Local_SP500"
                return syms[:min(limit, len(syms))]
    except Exception as e:
        logger.debug(f"Local S&P 500 read failed: {e}")

    # Hardcoded minimal subset to ensure progress if file missing
    try:
        minimal = [
            "AAPL","MSFT","NVDA","AMZN","GOOG","META","TSLA","AVGO","BRK-B","UNH",
            "V","JPM","WMT","XOM","LLY","MA","HD","PG","COST","ORCL",
        ]
        out = _normalize_symbols(minimal)
        logger.warning(f"Using hardcoded minimal universe: {len(out)} tickers")
        LAST_UNIVERSE_PROVIDER = "Hardcoded_Minimal"
        return out[:min(limit, len(out))]
    except Exception:
        return minimal

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

def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy/uppercase config keys to canonical lowercase ones.
    Does not remove original keys; just ensures lowercase toggles exist.
    """
    if not isinstance(config, dict):
        return {}
    normalized = dict(config)
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
                # Reduce to 100k if higher
                if float(normalized.get("min_avg_volume", 100_000)) > 100_000:
                    normalized["min_avg_volume"] = 100_000
        except Exception:
            normalized["min_avg_volume"] = 100_000
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
                df = df_all
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
            except: pass
            
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
        row_indicators = tech_df.iloc[-1]
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
        row_indicators["relative_strength_20d"] = float(rs_val)
    except Exception:
        row_indicators["relative_strength_20d"] = np.nan

    # Tier 2 may skip this filter because Tier 1 already applied OHLCV checks
    if not skip_tech_filter:
        if not apply_technical_filters(row_indicators, strict=False):
            return None

    # Use the last bar's date as the signal date
    try:
        as_of_dt = pd.to_datetime(df.index[-1])
    except Exception:
        as_of_dt = None

    rec_series = compute_recommendation_scores(
        row=row_indicators,
        ticker=tkr,
        as_of_date=as_of_dt,
        enable_ml=ML_20D_AVAILABLE,
        use_multi_source=False,
    )

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
    if "Score" not in results.columns and "FinalScore_20d" in results.columns:
        results["Score"] = results["FinalScore_20d"]
    return results

# --- Main Pipeline Runner ---

def run_scan_pipeline(
    universe: List[str],
    config: Dict[str, Any],
    status_callback: Optional[Callable[[str], None]] = None,
    data_map: Optional[Dict[str, pd.DataFrame]] = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
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
    # Normalize config keys (support legacy uppercase keys)
    config = _normalize_config(config)

    if status_callback:
        status_callback(f"Starting pipeline for {len(universe)} tickers...")

    # Initialize global market context caches (SPY + VIX) to avoid repeated provider hits
    try:
        from core.market_context import initialize_market_context
        # Enforce minimum market context window of 250 days
        initialize_market_context(symbols=["SPY", "^VIX"], period_days=250)
        logger.info("[PIPELINE] Global index cache initialized (SPY, VIX)")
    except Exception as e:
        logger.debug(f"[PIPELINE] Market context init skipped: {e}")

    # Run API preflight to determine active providers for this scan
    try:
        from core.api_preflight import run_preflight
        provider_status: Dict[str, Dict[str, Any]] = run_preflight()
        # Apply global default provider status to v2 data layer
        try:
            from core.data_sources_v2 import set_default_provider_status, disable_provider_category
            set_default_provider_status(provider_status)
            # If FMP index endpoint is not OK, disable for session
            if not provider_status.get("FMP_INDEX", {"ok": True}).get("ok", True):
                disable_provider_category("fmp", "index")
        except Exception:
            pass
    except Exception:
        provider_status = {}

    start_universe = len(universe)
    data_map, benchmark_df = _step_fetch_and_prepare_base_data(universe, config, status_callback, data_map)

    # Early RS percentile ranking on full universe (Weighted RS = 0.7*RS_63d + 0.3*RS_21d)
    # Controlled by env RS_RANKING/RS_RANKING_ENABLED and sample size (>100)
    rs_enabled_env = os.getenv("RS_RANKING", os.getenv("RS_RANKING_ENABLED", "1"))
    rs_enabled = bool(rs_enabled_env == "1")
    if rs_enabled and start_universe > 100:
        if status_callback:
            status_callback("Ranking universe by blended RS (21/63d)...")
        try:
            from advanced_filters import compute_relative_strength, fetch_benchmark_data
            bench_df = fetch_benchmark_data("SPY", days=200)
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
    for tkr, df in (data_map or {}).items():
        try:
            if df is None or df.empty:
                continue
            last = df.iloc[-1]
            # Apply fast technical filter using only OHLCV
            if apply_technical_filters(last, relaxed=False):
                tier1_pass.append(tkr)
        except Exception as exc:
            logger.debug(f"Tier1 filter failed for {tkr}: {exc}")
            continue

    filtered_count = start_universe - len(tier1_pass)
    logger.info(
        f"[PIPELINE] Tier 1: scanned={start_universe}, passed={len(tier1_pass)}, filtered={filtered_count}"
    )

    # Build Tier 2 input map strictly from Tier 1 output
    tier2_map: Dict[str, pd.DataFrame] = {t: data_map[t] for t in tier1_pass if t in data_map}

    # --- Tier 2: Deep Dive (indicators + ML, fundamentals later) ---
    results = _step_compute_scores_with_unified_logic(
        tier2_map, config, status_callback, skip_tech_filter=True
    )

    # Verify Tier 2 input list matches Tier 1 output
    tier2_input_set = set(tier2_map.keys())
    tier2_output_set = set(results["Ticker"].tolist()) if not results.empty else set()
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
        return results, data_map
    
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

    # Scale FinalScore_20d back to 0-100 for consistency with rest of system
    # (all other scoring is in 0-100 range, FinalScore_20d is canonical for display/filtering)
    results["FinalScore_20d"] = results["FinalScore_20d"] * 100.0

    # Ensure Score always matches the scaled FinalScore_20d after advanced filters
    if "FinalScore_20d" in results.columns:
        results["Score"] = results["FinalScore_20d"]
    
    # Do not filter out low scores here; the UI will label REJECTs but keep their scores visible
    logger.info(f"[PIPELINE] Advanced filters applied without score-zeroing; total stocks: {len(results)}")
    
    # Optional: Meteor Mode filter (VCP + RS + Pocket Pivots)
    meteor_mode = bool(config.get("meteor_mode", bool(os.getenv("METEOR_MODE", "1") == "1")))
    if meteor_mode and not results.empty:
        if status_callback: status_callback("Applying Meteor filters (VCP + RS + Pocket Pivots)...")
        try:
            # Use core.filters version to avoid local import shadowing
            bench_df = fetch_benchmark_data("SPY", days=200)
            kept_rows = []
            for _, row in results.iterrows():
                tkr = str(row.get("Ticker"))
                df = tier2_map.get(tkr) or data_map.get(tkr)
                if df is None or df.empty:
                    continue
                base_score = float(row.get("FinalScore_20d", row.get("Score", 0.0)))
                new_score, details = compute_advanced_score(tkr, df.rename(columns=str.title), bench_df.rename(columns=str.title), base_score)
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
            score_thr = float(config.get("fundamentals_score_threshold", float(os.getenv("FUNDAMENTALS_SCORE_THRESHOLD", "65"))))
        except Exception:
            score_thr = 65.0
        try:
            top_n_cap = int(config.get("fundamentals_top_n", int(os.getenv("FUNDAMENTALS_TOP_N", "200"))))
        except Exception:
            top_n_cap = 200

        # Filter and cap by top-N, prioritizing highest technical scores
        score_col = "TechScore_20d" if "TechScore_20d" in results.columns else "Score"
        eligible = results[results[score_col] > score_thr].sort_values(score_col, ascending=False)
        if top_n_cap > 0:
            eligible = eligible.head(top_n_cap)
        winners = eligible["Ticker"].tolist()

        if not winners:
            logger.info(f"No winners with Score > {score_thr:.0f}; skipping fundamentals fetch")
            fund_df = pd.DataFrame()
        else:
            logger.info(f"Fetching fundamentals for {len(winners)} winners ({score_col} > {score_thr:.0f}, TopN={top_n_cap})")
            # Pass provider_status so data layer strictly skips failed providers and promotes alternates
            fund_df = fetch_fundamentals_batch(winners, provider_status=provider_status, as_of_date=fund_as_of)
        
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
        
        # Compute fundamental scores only if missing to avoid double-calculation
        for idx, row in results.iterrows():
            if pd.notna(row.get("Fundamental_S")):
                continue
            try:
                fund_data = row.to_dict()
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
            from datetime import datetime, timedelta
            import yfinance as yf
            
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
                except: pass
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
    
    if "buy_amount_v2" not in results.columns:
        results = allocate_budget(results, config.get("BUDGET_TOTAL", 5000), config.get("MIN_POSITION", 500), config.get("MAX_POSITION_PCT", 0.2))
    
    # STRICT ENFORCEMENT: Score must always equal FinalScore_20d
    # This is the final safety check before returning results
    if "FinalScore_20d" in results.columns:
        results["Score"] = results["FinalScore_20d"]
        logger.info(f"[PIPELINE] Final check: Score column set to FinalScore_20d for all {len(results)} results")
    
    # --- Persist latest results for Streamlit dashboard freshness ---
    try:
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        latest_json = data_dir / "latest_scan_live.json"
        latest_parquet = data_dir / "latest_scan_live.parquet"

        # Save JSON (records, ISO dates)
        results.to_json(latest_json, orient="records", date_format="iso")
        # Save Parquet
        results.to_parquet(latest_parquet, index=False)
        logger.info(f"[PIPELINE] Persisted latest scan to {latest_json} and {latest_parquet}")
    except Exception as e:
        logger.warning(f"[PIPELINE] Failed to persist latest scan files: {e}")

    return results, data_map


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

    # Universe
    universe = fetch_top_us_tickers_by_market_cap(limit=2000)
    logger.info(f"Fetched {len(universe)} tickers")

    # Config
    cfg = _normalize_config(get_config())
    cfg["fundamental_enabled"] = True
    cfg["beta_filter_enabled"] = False

    results, data_map = run_scan_pipeline(universe, cfg)
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