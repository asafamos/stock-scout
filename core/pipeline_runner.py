import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any, Tuple
import yfinance as yf

from core.config import get_config
from core.unified_logic import (
    build_technical_indicators,
    apply_technical_filters,
    compute_technical_score,
    compute_tech_score_20d_v2,
)
from core.ml_20d_inference import (
    predict_20d_prob_from_row,
    apply_live_v3_adjustments,
    ML_20D_AVAILABLE,
)
from core.scoring_pipeline_20d import compute_final_scores_20d
from core.data_sources_v2 import (
    fetch_fundamentals_batch,
)
from core.scoring.fundamental import compute_fundamental_score_with_breakdown
from core.v2_risk_engine import score_ticker_v2_enhanced
from core.classification import apply_classification
from core.portfolio import allocate_budget
from advanced_filters import (
    compute_advanced_score,
    should_reject_ticker,
    fetch_benchmark_data,
)

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def fetch_history_bulk(tickers: List[str], period_days: int, ma_long: int) -> Dict[str, pd.DataFrame]:
    end = datetime.utcnow()
    start = end - timedelta(days=period_days + 50)
    data_map = {}
    min_rows = ma_long + 40
    
    try:
        # Use threads=True for faster download
        df_all = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False, threads=True)
        
        if len(tickers) == 1:
            tkr = tickers[0]
            if not df_all.empty and len(df_all) >= min_rows:
                data_map[tkr] = df_all
        else:
            for tkr in tickers:
                try:
                    df = df_all[tkr].dropna(how='all')
                    if len(df) >= min_rows:
                        data_map[tkr] = df
                except KeyError:
                    pass
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

# --- Main Pipeline Runner ---

def run_scan_pipeline(
    universe: List[str],
    config: Dict[str, Any],
    status_callback: Optional[Callable[[str], None]] = None,
    data_map: Optional[Dict[str, pd.DataFrame]] = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Unified pipeline runner for Stock Scout.
    """
    if status_callback: status_callback(f"Starting pipeline for {len(universe)} tickers...")
    
    # 1. Fetch History
    if data_map is None:
        if status_callback: status_callback("Fetching historical data...")
        data_map = fetch_history_bulk(universe, config.get("LOOKBACK_DAYS", 200), config.get("MA_LONG", 200))
    
    # 2. Technical Analysis
    if status_callback: status_callback("Computing technical indicators...")
    rows = []
    
    # Pre-fetch benchmark for advanced filters
    benchmark_df = fetch_benchmark_data(config.get("BETA_BENCHMARK", "SPY"), config.get("LOOKBACK_DAYS", 252))
    
    for tkr, df in data_map.items():
        if df.empty: continue
        
        try:
            tech_df = build_technical_indicators(df)
            row_indicators = tech_df.iloc[-1]
        except: continue
        
        if not apply_technical_filters(row_indicators, strict=True):
            continue
            
        tech_score = compute_technical_score(row_indicators, weights=config.get("WEIGHTS"))
        tech_v2 = compute_tech_score_20d_v2(row_indicators) * 100.0
        ml_prob = predict_20d_prob_from_row(row_indicators) if ML_20D_AVAILABLE else 0.5
        
        row = row_indicators.to_dict()
        row["Ticker"] = tkr
        row["Score_Tech"] = tech_score
        row["TechScore_20d_v2_raw"] = tech_v2
        row["ML_20d_Prob_raw"] = ml_prob
        
        # Initial RR
        price = row.get("Close")
        atr = row.get("ATR")
        if price and atr:
            target = price + (2 * atr)
            row["RewardRisk"] = calculate_rr(price, target, atr)
            
        rows.append(row)
        
    if not rows:
        return pd.DataFrame(), data_map
        
    results = pd.DataFrame(rows)
    
    # 3. Final Scores (ML + Tech Blend)
    if status_callback: status_callback("Computing final scores...")
    if "ATR_Pct" in results.columns:
        results["ATR_Pct_percentile"] = results["ATR_Pct"].rank(pct=True)
    results["ML_20d_Prob_live_v3"] = apply_live_v3_adjustments(results, prob_col="ML_20d_Prob_raw")
    results["ML_20d_Prob"] = results["ML_20d_Prob_live_v3"]
    results = compute_final_scores_20d(results, include_ml=True)
    results["Score"] = results["FinalScore"] # Base score for next steps
    
    # 3b. Beta Filter
    if config.get("BETA_FILTER_ENABLED"):
        if status_callback: status_callback("Applying Beta filter...")
        beta_max = float(config.get("BETA_MAX_ALLOWED", 1.5))
        top_k = int(config.get("BETA_TOP_K", 50))
        results = results.sort_values("Score", ascending=False)
        to_check = results.head(top_k).index
        for idx in to_check:
            tkr = results.at[idx, "Ticker"]
            b = fetch_beta_vs_benchmark(tkr, config.get("BETA_BENCHMARK", "SPY"))
            results.at[idx, "Beta"] = b
        results = results[~( (results["Beta"].notna()) & (results["Beta"] > beta_max) )]
    
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
        
    rs_thresh = _q(rs_vals, 0.05, -0.30)
    mom_thresh = _q(mom_vals, 0.10, 0.15)
    rr_thresh = _q(rr_vals, 0.10, 0.50)
    
    dyn_thresh = {"rs_63d": rs_thresh, "momentum_consistency": mom_thresh, "risk_reward_ratio": rr_thresh}
    
    for idx, sig, enhanced in signals_store:
        catastrophic, reason = should_reject_ticker(sig, dynamic=dyn_thresh)
        
        results.at[idx, "RS_63d"] = sig.get("rs_63d")
        results.at[idx, "Volume_Surge"] = sig.get("volume_surge")
        results.at[idx, "MA_Aligned"] = sig.get("ma_aligned")
        results.at[idx, "Quality_Score"] = sig.get("quality_score")
        results.at[idx, "RR_Ratio"] = sig.get("risk_reward_ratio")
        results.at[idx, "Momentum_Consistency"] = sig.get("momentum_consistency")
        
        if catastrophic:
            results.at[idx, "Score"] = 0.0
            results.at[idx, "RejectionReason"] = reason
        else:
            penalty = 0.0
            if sig.get("rs_63d", 0) < rs_thresh: penalty += 2.0
            if sig.get("momentum_consistency", 0) < mom_thresh: penalty += 2.0
            if sig.get("risk_reward_ratio", 0) < rr_thresh: penalty += 3.0
            
            results.at[idx, "AdvPenalty"] = penalty
            results.at[idx, "Score"] = max(0.0, enhanced - penalty)

    results = results[results["Score"] > 0].copy()
    
    # 5. Fundamentals & Sector Enrichment
    if config.get("FUNDAMENTAL_ENABLED", True):
        if status_callback: status_callback("Fetching fundamentals & sector data...")
        fund_df = fetch_fundamentals_batch(results["Ticker"].tolist())
        
        # Reset index and ensure Ticker column
        if isinstance(fund_df.index, pd.Index) and fund_df.index.name == 'ticker':
            fund_df = fund_df.reset_index()
            fund_df = fund_df.rename(columns={'ticker': 'Ticker'})
        elif "Ticker" not in fund_df.columns and len(fund_df) > 0:
            fund_df["Ticker"] = fund_df.index
            
        # Merge fundamentals
        results = pd.merge(results, fund_df, on="Ticker", how="left", suffixes=("", "_fund"))
        
        # Compute fundamental scores for each ticker
        for idx, row in results.iterrows():
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
        
    # 6. Risk Engine V2
    if status_callback: status_callback("Running Risk Engine V2...")
    v2_results = []
    for idx, row in results.iterrows():
        res = score_ticker_v2_enhanced(row["Ticker"], row, budget_total=config.get("BUDGET_TOTAL", 5000), min_position=config.get("MIN_POSITION", 500), enable_ml=True)
        v2_results.append(res)
        
    if v2_results:
        v2_df = pd.DataFrame(v2_results)
        v2_df["Ticker"] = results["Ticker"].values
        results = pd.merge(results, v2_df, on="Ticker", how="left", suffixes=("", "_v2"))
        if "conviction_v2_final" in results.columns:
            results["Score"] = results["conviction_v2_final"]
            
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
        
    return results, data_map