import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any, Tuple
import yfinance as yf

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
from core.unified_logic import compute_recommendation_scores

# For backward compatibility with code that checks ML availability
from core.ml_20d_inference import ML_20D_AVAILABLE

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def fetch_history_bulk(tickers: List[str], period_days: int, ma_long: int) -> Dict[str, pd.DataFrame]:
    end = datetime.utcnow()
    start = end - timedelta(days=period_days + 50)
    data_map = {}
    # More lenient: just need enough for MA calculation, not ma_long + 40
    min_rows = max(50, ma_long // 2)  # At least 50 rows or half of ma_long
    
    try:
        # Use threads=True for faster download
        df_all = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False, threads=True)
        
        if len(tickers) == 1:
            tkr = tickers[0]
            if not df_all.empty and len(df_all) >= min_rows:
                data_map[tkr] = df_all
                logger.info(f"Fetched {len(df_all)} rows for {tkr}")
            else:
                logger.warning(f"Insufficient data for {tkr}: {len(df_all)} rows < {min_rows} required")
        else:
            for tkr in tickers:
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
        logger.warning(f"Bulk fetch failed: {e}")
        
    return data_map

def fetch_beta_vs_benchmark(ticker: str, bench: str = "SPY", days: int = 252) -> float:
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days + 30)
        df_t = yf.download(ticker, start=start, end=end, progress=False)
        df_b = yf.download(bench, start=start, end=end, progress=False)
        
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
        data_map = fetch_history_bulk(universe, config.get("lookback_days", 200), config.get("ma_long", 200))

    benchmark_df = fetch_benchmark_data(config.get("beta_benchmark", "SPY"), config.get("lookback_days", 252))
    return data_map, benchmark_df


def _step_compute_scores_with_unified_logic(
    data_map: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    status_callback: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    if status_callback:
        status_callback("Computing technical indicators...")

    rows = []
    for tkr, df in data_map.items():
        if df.empty:
            continue

        try:
            tech_df = build_technical_indicators(df)
            row_indicators = tech_df.iloc[-1]
        except Exception as exc:
            logger.warning("indicator build failed for %s: %s", tkr, exc)
            continue

        if not apply_technical_filters(row_indicators, strict=False):
            continue

        rec_series = compute_recommendation_scores(
            row=row_indicators,
            ticker=tkr,
            enable_ml=ML_20D_AVAILABLE,
            use_multi_source=config.get("fundamental_enabled", True),
        )

        rows.append(rec_series)

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    # Ensure canonical columns exist
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
        - ML_20d_Prob: ML probability (0-1)
        - ConvictionScore: Conviction metric (0-100)
        - Risk_Meter: Risk level (0-100, higher = riskier)
        - Reliability_Score: Data quality (0-100)
        - buy_amount_v2: Allocated dollar amount
        - Score: Legacy alias for FinalScore_20d
    """
    if status_callback:
        status_callback(f"Starting pipeline for {len(universe)} tickers...")

    start_universe = len(universe)
    data_map, benchmark_df = _step_fetch_and_prepare_base_data(universe, config, status_callback, data_map)
    results = _step_compute_scores_with_unified_logic(data_map, config, status_callback)
    logger.info(f"[PIPELINE] Stage counts: universe={start_universe}, tech_pass={len(results)}")

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
            enhanced, sig = compute_advanced_score(tkr, df, benchmark_df, base_score / 100.0)
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
            # Apply penalty by reducing FinalScore_20d, not by overwriting Score
            # Set to minimum score (0.01 in normalized [0, 1] range, will be scaled to 1.0 later)
            results.at[idx, "FinalScore_20d"] = 0.01
            results.at[idx, "RejectionReason"] = reason
        else:
            # Penalties are in [0, 4.5] scale (0-100 range), normalize to [0, 0.045]
            penalty = 0.0
            if sig.get("rs_63d", 0) < rs_thresh: penalty += 1.0  # Reduced from 2.0
            if sig.get("momentum_consistency", 0) < mom_thresh: penalty += 1.0  # Reduced from 2.0
            if sig.get("risk_reward_ratio", 0) < rr_thresh: penalty += 1.5  # Reduced from 3.0
            
            normalized_penalty = penalty / 100.0  # Convert from [0, 4.5] to [0, 0.045]
            results.at[idx, "AdvPenalty"] = penalty
            # Apply penalty to FinalScore_20d, not Score
            # enhanced is in [0, 1] range, so penalty must be too
            results.at[idx, "FinalScore_20d"] = max(0.01, enhanced - normalized_penalty)

    # Ensure Score always matches FinalScore_20d after advanced filters
    if "FinalScore_20d" in results.columns:
        results["Score"] = results["FinalScore_20d"]
    
    # Scale FinalScore_20d back to 0-100 for consistency with rest of system
    # (all other scoring is in 0-100 range, FinalScore_20d is canonical for display/filtering)
    results["FinalScore_20d"] = results["FinalScore_20d"] * 100.0
    
    # Keep stocks with positive scores (allow FinalScore_20d >= 1.0 which is 0.01 in normalized)
    results = results[results["FinalScore_20d"] >= 1.0].copy()
    logger.info(f"[PIPELINE] After advanced filters: {len(results)} stocks with FinalScore_20d >= 1.0")
    
    # 5. Fundamentals & Sector Enrichment
    if config.get("FUNDAMENTAL_ENABLED", True):
        if status_callback: status_callback("Fetching fundamentals & sector data...")
        fund_df = fetch_fundamentals_batch(results["Ticker"].tolist())
        
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
        
    return results, data_map