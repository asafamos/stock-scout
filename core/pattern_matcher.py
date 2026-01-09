"""
Pattern Matcher Engine: Evaluate how well each stock matches historical winning patterns.

Patterns identified from backtests:
- RSI 20-55: 62% coverage, decent win rate
- RR >= 2.0: 66% win rate, strong signal
- Momentum consistency >= 0.6: 65% win rate
- MA aligned (price > MA50 > MA200): Strong signal
- Volume surge >= 1.5: Institutional interest

This engine scores each stock against these patterns and weights them by success rate.
"""
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


class PatternMatcher:
    """Evaluate stock match against historical winning patterns."""
    
    # Historical pattern success rates (from backtests)
    PATTERNS = {
        "rsi_oversold": {
            "description": "RSI 20-40 (oversold bounce)",
            "success_rate": 0.70,
            "win_rate_pct": 2.37,
        },
        "rsi_neutral": {
            "description": "RSI 40-60 (neutral zone)",
            "success_rate": 0.62,
            "win_rate_pct": 1.80,
        },
        "rr_strong": {
            "description": "Risk/Reward >= 2.0",
            "success_rate": 0.66,
            "win_rate_pct": 1.66,
        },
        "rr_moderate": {
            "description": "Risk/Reward 1.5-2.0",
            "success_rate": 0.58,
            "win_rate_pct": 1.20,
        },
        "momentum_strong": {
            "description": "Momentum consistency >= 0.6",
            "success_rate": 0.65,
            "win_rate_pct": 1.35,
        },
        "ma_aligned": {
            "description": "MA alignment bullish (price > MA50 > MA200)",
            "success_rate": 0.68,
            "win_rate_pct": 1.92,
        },
        "volume_surge": {
            "description": "Volume surge >= 1.5 (institutional interest)",
            "success_rate": 0.60,
            "win_rate_pct": 1.15,
        },
        "atr_sweet_spot": {
            "description": "ATR 2-4% (sweet spot volatility)",
            "success_rate": 0.65,
            "win_rate_pct": 1.50,
        },
        "near_support": {
            "description": "Price 2-5% above support level",
            "success_rate": 0.61,
            "win_rate_pct": 1.25,
        },
        "narrow_range_5d": {
            "description": "Last 5d ATR vs 20d ATR < 0.7 (tight coil)",
            "success_rate": 0.72,
            "win_rate_pct": 1.80,
        },
    }
    
    @classmethod
    def evaluate_stock(cls, row: pd.Series) -> Dict[str, float]:
        """
        Evaluate how well a stock matches winning patterns.
        
        Returns:
            Dict with keys:
            - pattern_score: 0-100, aggregate match score
            - pattern_count: number of patterns matched
            - best_pattern: highest confidence match
            - patterns: detailed dict of each pattern match
        """
        matches = {}
        match_scores = []
        
        # Pattern 1: RSI Oversold
        rsi = row.get("RSI")
        if rsi is not None and pd.notna(rsi):
            if 20 <= rsi <= 40:
                matches["rsi_oversold"] = {
                    "match": True,
                    "value": float(rsi),
                    "score": 1.0,
                    "sr": 0.70
                }
                match_scores.append((0.70, 1.0))
            elif 40 < rsi <= 60:
                matches["rsi_neutral"] = {
                    "match": True,
                    "value": float(rsi),
                    "score": 0.8,
                    "sr": 0.62
                }
                match_scores.append((0.62, 0.8))
        
        # Pattern 2: Risk/Reward
        rr = row.get("RR") or row.get("RR_Ratio")
        if rr is not None and pd.notna(rr):
            if float(rr) >= 2.0:
                matches["rr_strong"] = {
                    "match": True,
                    "value": float(rr),
                    "score": 1.0,
                    "sr": 0.66
                }
                match_scores.append((0.66, 1.0))
            elif float(rr) >= 1.5:
                matches["rr_moderate"] = {
                    "match": True,
                    "value": float(rr),
                    "score": 0.7,
                    "sr": 0.58
                }
                match_scores.append((0.58, 0.7))
        
        # Pattern 3: Momentum Consistency
        mom_cons = row.get("MomCons") or row.get("Momentum_Consistency")
        if mom_cons is not None and pd.notna(mom_cons):
            if float(mom_cons) >= 0.6:
                matches["momentum_strong"] = {
                    "match": True,
                    "value": float(mom_cons),
                    "score": 1.0,
                    "sr": 0.65
                }
                match_scores.append((0.65, 1.0))
        
        # Pattern 4: MA Alignment
        ma_aligned = row.get("MA_Aligned")
        if ma_aligned is not None:
            if bool(ma_aligned):
                matches["ma_aligned"] = {
                    "match": True,
                    "value": 1.0,
                    "score": 1.0,
                    "sr": 0.68
                }
                match_scores.append((0.68, 1.0))
        
        # Pattern 5: Volume Surge
        vol_surge = row.get("VolSurge")
        if vol_surge is not None and pd.notna(vol_surge):
            if float(vol_surge) >= 1.5:
                matches["volume_surge"] = {
                    "match": True,
                    "value": float(vol_surge),
                    "score": 1.0,
                    "sr": 0.60
                }
                match_scores.append((0.60, 1.0))
        
        # Pattern 6: ATR Sweet Spot
        atr_pct = row.get("ATR_Pct")
        if atr_pct is not None and pd.notna(atr_pct):
            if 0.02 <= float(atr_pct) <= 0.04:
                matches["atr_sweet_spot"] = {
                    "match": True,
                    "value": float(atr_pct),
                    "score": 1.0,
                    "sr": 0.65
                }
                match_scores.append((0.65, 1.0))

        # Pattern 7: Narrow Range 5d (tight volatility coil)
        # Range Ratio = ATR_5 / ATR_20 computed upstream in indicators
        rr_5_20 = row.get("RangeRatio_5_20")
        if rr_5_20 is not None and pd.notna(rr_5_20):
            try:
                ratio_val = float(rr_5_20)
                if np.isfinite(ratio_val) and ratio_val < 0.7:
                    matches["narrow_range_5d"] = {
                        "match": True,
                        "value": ratio_val,
                        "score": 1.0,
                        "sr": 0.72,
                    }
                    match_scores.append((0.72, 1.0))
            except Exception:
                pass
        
        # Calculate aggregate pattern score
        if match_scores:
            # Weight by success rate * match quality
            weighted_scores = [sr * score for sr, score in match_scores]
            pattern_score = np.mean(weighted_scores) * 100.0
            best_pattern = max(matches.keys(), key=lambda k: matches[k]["sr"])
        else:
            pattern_score = 0.0
            best_pattern = None
        
        return {
            "pattern_score": float(np.clip(pattern_score, 0, 100)),
            "pattern_count": len(matches),
            "best_pattern": best_pattern,
            "patterns": matches,
        }
    
    @classmethod
    def score_universe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pattern matching to entire universe."""
        results = []
        for idx, row in df.iterrows():
            eval_result = cls.evaluate_stock(row)
            results.append({
                "ticker": row.get("Ticker"),
                "pattern_score": eval_result["pattern_score"],
                "pattern_count": eval_result["pattern_count"],
                "best_pattern": eval_result["best_pattern"],
                "num_patterns_matched": len(eval_result["patterns"]),
            })
        
        patterns_df = pd.DataFrame(results)
        return patterns_df


def blend_pattern_with_final_score(
    results_df: pd.DataFrame,
    tech_score_col: str = "TechScore_20d",
    fundamental_col: str = "Fundamental_S",
    ml_col: str = "ML_20d_Prob",
    pattern_weight: float = 0.15,
) -> pd.DataFrame:
    """
    Blend pattern matching score with final conviction score.
    
    Formula:
    FinalScore_Enhanced = (1 - pattern_weight) * FinalScore_20d + pattern_weight * PatternScore
    
    This ensures historical winning patterns are rewarded without overwhelming other signals.
    """
    if "pattern_score" not in results_df.columns:
        pattern_eval = PatternMatcher.score_universe(results_df)
        results_df = results_df.merge(
            pattern_eval[["ticker", "pattern_score"]].rename(columns={"ticker": "Ticker"}),
            on="Ticker",
            how="left"
        )
    
    # Blend pattern score into final score
    final_col = "FinalScore_20d"
    if final_col in results_df.columns:
        original_final = pd.to_numeric(results_df[final_col], errors="coerce").fillna(50)
        pattern_scores = pd.to_numeric(results_df["pattern_score"], errors="coerce").fillna(50)
        
        blended = (1 - pattern_weight) * original_final + pattern_weight * pattern_scores
        results_df["FinalScore_20d_PatternBlended"] = blended
        results_df["PatternWeightAdjustment"] = pattern_weight
    
    return results_df


if __name__ == "__main__":
    # Example usage
    print("🎯 Pattern Matcher Engine")
    print("=" * 60)
    print("\nSupported patterns:")
    for name, info in PatternMatcher.PATTERNS.items():
        print(f"  {name:20s} (SR: {info['success_rate']:.0%}, Avg win: {info['win_rate_pct']:.2f}%)")
    
    print("\n✅ Pattern Matcher ready for integration")
