"""
UnifiedScorer - Single Source of Truth for Stock Scoring.

All scoring in the application should go through this class.
This ensures consistent scores regardless of entry point.

Usage:
    from core.scoring.unified_scorer import UnifiedScorer, score_ticker
    
    # Create scorer with custom config
    scorer = UnifiedScorer(config={"enable_ml": True, "ml_max_boost_pct": 10.0})
    
    # Score a ticker
    result = scorer.score(ticker_data, technical_indicators, fundamental_data)
    
    # Access results
    print(f"Final Conviction: {result.final_conviction}")
    print(f"Technical: {result.technical_score}, Fundamental: {result.fundamental_score}")
    print(f"ML Boost: {result.ml_boost} ({result.ml_status})")
    
    # Or use convenience function
    result = score_ticker(ticker_data, indicators, fundamentals)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
import logging
import warnings

from core.scoring_config import BASE_SCORE_WEIGHTS

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """
    Complete scoring result with breakdown.
    
    Attributes:
        final_conviction: The main score (0-100), used for ranking
        technical_score: Technical analysis score (0-100)
        fundamental_score: Fundamental analysis score (0-100)
        ml_probability: ML win probability (0-1) or None if unavailable
        ml_boost: ML adjustment to base score (typically -10 to +10)
        ml_status: ML status message ("enabled", "disabled", "error", etc.)
        breakdown: Detailed breakdown of score components
        reliability_pct: Data reliability percentage (0-100)
        data_quality: Overall data quality ("high", "medium", "low")
    """
    # Final scores
    final_conviction: float  # 0-100, the main score

    # Component scores
    technical_score: float  # 0-100
    fundamental_score: float  # 0-100

    # ML adjustment
    ml_probability: Optional[float]  # 0-1 or None
    ml_boost: float  # typically -10 to +10
    ml_status: str  # "enabled", "disabled", "error"

    # Breakdown for transparency
    breakdown: Dict[str, Any] = field(default_factory=dict)

    # Quality indicators
    reliability_pct: float = 0.0
    data_quality: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame/JSON."""
        return {
            "final_conviction": self.final_conviction,
            "technical_score": self.technical_score,
            "fundamental_score": self.fundamental_score,
            "ml_probability": self.ml_probability,
            "ml_boost": self.ml_boost,
            "ml_status": self.ml_status,
            "reliability_pct": self.reliability_pct,
            "data_quality": self.data_quality,
            **self.breakdown,
        }

    def __repr__(self) -> str:
        return (
            f"ScoringResult(final={self.final_conviction:.1f}, "
            f"tech={self.technical_score:.1f}, fund={self.fundamental_score:.1f}, "
            f"ml_boost={self.ml_boost:+.1f})"
        )


class UnifiedScorer:
    """
    Single entry point for all scoring operations.
    
    This class consolidates all scoring logic to ensure consistent results
    regardless of which part of the application initiates scoring.
    
    Configuration options:
        enable_ml (bool): Enable ML boost (default: True)
        ml_max_boost_pct (float): Maximum ML boost as % of base (default: 10.0)
        technical_weight (float): Weight for technical score (default: 0.60)
        fundamental_weight (float): Weight for fundamental score (default: 0.40)
        use_v2_scoring (bool): Use V2 scoring algorithms (default: True)
    
    Example:
        scorer = UnifiedScorer(config={"enable_ml": True})
        result = scorer.score(ticker_data, indicators, fundamentals)
        print(f"Score: {result.final_conviction:.1f}")
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize UnifiedScorer with configuration.
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        
        # ML configuration
        self.ml_enabled = self.config.get("enable_ml", True)
        self.ml_max_boost = self.config.get("ml_max_boost_pct", 10.0)
        
        # Weight configuration - defaults from centralized scoring_config
        self.technical_weight = self.config.get("technical_weight", BASE_SCORE_WEIGHTS.get("technical", 0.69))
        self.fundamental_weight = self.config.get("fundamental_weight", BASE_SCORE_WEIGHTS.get("fundamental", 0.31))
        
        # Algorithm version
        self.use_v2_scoring = self.config.get("use_v2_scoring", True)
        
        # Validate weights sum to 1.0
        total_weight = self.technical_weight + self.fundamental_weight
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(
                f"Scoring weights sum to {total_weight:.3f}, not 1.0. "
                "Results may be unexpected."
            )

    def score(
        self,
        ticker_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        fundamental_data: Dict[str, Any],
    ) -> ScoringResult:
        """
        Compute complete scoring for a ticker.
        
        This is the main entry point for scoring. It computes technical and
        fundamental scores, applies ML boost if enabled, and returns a
        comprehensive ScoringResult.
        
        Args:
            ticker_data: Raw ticker data (price, volume, symbol, etc.)
            technical_indicators: Computed technical indicators (RSI, ATR, etc.)
            fundamental_data: Fundamental metrics (PE, ROE, margins, etc.)
        
        Returns:
            ScoringResult with all components and breakdown
        
        Example:
            ticker_data = {"Close": 150.0, "Volume": 1000000, "Ticker": "AAPL"}
            indicators = {"RSI": 55.0, "ATR_Pct": 0.025, "Return_20d": 0.05}
            fundamentals = {"roe": 0.25, "pe": 28.0, "margin": 0.35}
            
            result = scorer.score(ticker_data, indicators, fundamentals)
        """
        # 1. Compute technical score
        tech_score = self._compute_technical_score(ticker_data, technical_indicators)
        
        # 2. Compute fundamental score
        fund_score, fund_breakdown = self._compute_fundamental_score(fundamental_data)
        
        # 3. Compute base conviction (weighted average)
        base_conviction = (
            tech_score * self.technical_weight +
            fund_score * self.fundamental_weight
        )
        
        # 4. Apply ML boost (if enabled)
        ml_prob = None
        ml_boost = 0.0
        ml_status = "disabled"
        
        if self.ml_enabled:
            ml_prob, ml_boost, ml_status = self._apply_ml_boost(
                base_conviction, ticker_data, technical_indicators, fundamental_data
            )
        
        # 5. Compute final conviction
        final_conviction = float(np.clip(base_conviction + ml_boost, 0, 100))
        
        # 6. Compute reliability and data quality
        reliability = self._compute_reliability(ticker_data, fundamental_data)
        data_quality = self._assess_data_quality(ticker_data, fundamental_data)
        
        return ScoringResult(
            final_conviction=final_conviction,
            technical_score=float(tech_score),
            fundamental_score=float(fund_score),
            ml_probability=ml_prob,
            ml_boost=float(ml_boost),
            ml_status=ml_status,
            reliability_pct=float(reliability),
            data_quality=data_quality,
            breakdown={
                "base_conviction": float(base_conviction),
                "technical_weight": self.technical_weight,
                "fundamental_weight": self.fundamental_weight,
                "ml_max_boost": self.ml_max_boost,
                **fund_breakdown,
            }
        )

    def score_from_row(self, row: pd.Series) -> ScoringResult:
        """
        Score from a DataFrame row (convenience method).
        
        Extracts ticker_data, technical_indicators, and fundamental_data
        from a single DataFrame row.
        
        Args:
            row: DataFrame row with all required columns
            
        Returns:
            ScoringResult
        """
        # Extract data from row
        ticker_data = row.to_dict()
        
        # Technical indicators (typically already in row)
        technical_indicators = {
            k: row.get(k) for k in [
                "RSI", "ATR_Pct", "ATR", "Return_20d", "Return_10d", "Return_5d",
                "VolSurge", "MomCons", "RR", "MA50", "MA200", "MA50_Slope",
                "Overext", "Return_1m", "Return_3m", "Return_6m"
            ] if k in row.index
        }
        
        # Fundamental data
        fundamental_data = {
            k: row.get(k) for k in [
                "roe", "roic", "pe", "ps", "margin", "gm", "de", "debt_equity",
                "rev_yoy", "rev_g_yoy", "eps_yoy", "eps_g_yoy", "beta",
                "market_cap", "Fundamental_Coverage_Pct", "Fundamental_Sources_Count"
            ] if k in row.index
        }
        
        return self.score(ticker_data, technical_indicators, fundamental_data)

    def _compute_technical_score(
        self, 
        ticker_data: Dict, 
        indicators: Dict
    ) -> float:
        """
        Compute technical score from indicators.
        
        Uses the V2 scoring algorithm by default.
        """
        try:
            if self.use_v2_scoring:
                # Combine ticker_data and indicators into a row-like dict
                combined = {**ticker_data, **indicators}
                row = pd.Series(combined)
                
                from core.unified_logic import compute_tech_score_20d_v2
                return compute_tech_score_20d_v2(row)
            else:
                # Legacy scoring
                from core.unified_logic import compute_overall_score_20d
                combined = {**ticker_data, **indicators}
                return compute_overall_score_20d(combined)
        except Exception as e:
            logger.warning(f"Technical scoring failed: {e}")
            return 50.0  # Neutral default

    def _compute_fundamental_score(self, fundamentals: Dict) -> Tuple[float, Dict]:
        """
        Compute fundamental score with breakdown.
        
        Returns:
            Tuple of (score, breakdown_dict)
        """
        try:
            from core.scoring.fundamental import compute_fundamental_score_with_breakdown
            
            result = compute_fundamental_score_with_breakdown(fundamentals)
            
            # Extract breakdown for transparency
            breakdown = {}
            if hasattr(result, 'breakdown') and result.breakdown:
                bd = result.breakdown
                breakdown = {
                    "fund_quality_score": getattr(bd, 'quality_score', None),
                    "fund_growth_score": getattr(bd, 'growth_score', None),
                    "fund_valuation_score": getattr(bd, 'valuation_score', None),
                    "fund_leverage_score": getattr(bd, 'leverage_score', None),
                    "fund_stability_score": getattr(bd, 'stability_score', None),
                }
            
            return result.total, breakdown
        except Exception as e:
            logger.warning(f"Fundamental scoring failed: {e}")
            return 50.0, {}  # Neutral default

    def _apply_ml_boost(
        self, 
        base_conviction: float, 
        ticker_data: Dict,
        indicators: Dict, 
        fundamentals: Dict
    ) -> Tuple[Optional[float], float, str]:
        """
        Apply ML prediction boost.
        
        Returns:
            Tuple of (ml_probability, ml_boost, status_message)
        """
        try:
            from core.ml_integration import integrate_ml_with_conviction
            
            final, ml_info = integrate_ml_with_conviction(
                base_conviction, 
                ticker_data, 
                indicators, 
                fundamentals,
                enable_ml=True
            )
            
            ml_prob = ml_info.get("ml_probability")
            ml_boost = ml_info.get("ml_boost", 0.0)
            ml_status = ml_info.get("ml_status", "unknown")
            
            # Clamp boost to configured max
            if abs(ml_boost) > self.ml_max_boost:
                ml_boost = np.sign(ml_boost) * self.ml_max_boost
            
            return ml_prob, ml_boost, ml_status
            
        except Exception as e:
            logger.warning(f"ML boost failed: {e}")
            return None, 0.0, f"error: {str(e)}"

    def _compute_reliability(
        self, 
        ticker_data: Dict, 
        fundamental_data: Dict
    ) -> float:
        """
        Compute data reliability percentage.
        
        Based on:
        - Number of data sources
        - Fundamental coverage
        - Data freshness
        """
        try:
            # Try V2 reliability scoring
            from core.v2_risk_engine import calculate_reliability_v2
            
            combined = {**ticker_data, **fundamental_data}
            row = pd.Series(combined)
            result = calculate_reliability_v2(row)
            
            # calculate_reliability_v2 returns (score, details_dict)
            if isinstance(result, tuple):
                return float(result[0])
            return float(result)
        except Exception:
            pass
        
        # Fallback: simple heuristic
        reliability = 50.0
        
        # Boost for multiple data sources
        sources = ticker_data.get("data_sources", [])
        if isinstance(sources, list):
            reliability += min(len(sources) * 10, 30)
        
        # Boost for fundamental coverage
        coverage = fundamental_data.get("Fundamental_Coverage_Pct", 0)
        if coverage:
            reliability += min(float(coverage) / 4, 20)
        
        return min(reliability, 100.0)

    def _assess_data_quality(
        self, 
        ticker_data: Dict, 
        fundamental_data: Dict
    ) -> str:
        """
        Assess overall data quality level.
        
        Returns:
            "high", "medium", or "low"
        """
        score = 0
        
        # Check data sources
        sources = ticker_data.get("data_sources", [])
        if isinstance(sources, list):
            score += len(sources)
        
        # Check fundamental coverage
        coverage = fundamental_data.get("Fundamental_Coverage_Pct", 0)
        if coverage:
            if coverage >= 80:
                score += 3
            elif coverage >= 50:
                score += 2
            elif coverage > 0:
                score += 1
        
        # Check for key fields
        if ticker_data.get("Close"):
            score += 1
        if ticker_data.get("Volume"):
            score += 1
        
        if score >= 5:
            return "high"
        elif score >= 2:
            return "medium"
        return "low"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def score_ticker(
    ticker_data: Dict,
    technical_indicators: Dict,
    fundamental_data: Dict,
    config: Optional[Dict] = None
) -> ScoringResult:
    """
    Convenience function to score a single ticker.
    
    Creates a UnifiedScorer and scores the ticker in one call.
    
    Args:
        ticker_data: Raw ticker data
        technical_indicators: Technical indicators
        fundamental_data: Fundamental metrics
        config: Optional scorer configuration
        
    Returns:
        ScoringResult
    """
    scorer = UnifiedScorer(config)
    return scorer.score(ticker_data, technical_indicators, fundamental_data)


def score_dataframe(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
    score_column: str = "UnifiedScore"
) -> pd.DataFrame:
    """
    Score all rows in a DataFrame.
    
    Adds scoring columns to the DataFrame.
    
    Args:
        df: DataFrame with ticker data
        config: Optional scorer configuration
        score_column: Name for the main score column
        
    Returns:
        DataFrame with scoring columns added
    """
    scorer = UnifiedScorer(config)
    
    results = []
    for idx, row in df.iterrows():
        try:
            result = scorer.score_from_row(row)
            results.append({
                "index": idx,
                score_column: result.final_conviction,
                f"{score_column}_Tech": result.technical_score,
                f"{score_column}_Fund": result.fundamental_score,
                f"{score_column}_ML": result.ml_boost,
                f"{score_column}_Reliability": result.reliability_pct,
            })
        except Exception as e:
            logger.warning(f"Scoring failed for row {idx}: {e}")
            results.append({
                "index": idx,
                score_column: 50.0,
                f"{score_column}_Tech": 50.0,
                f"{score_column}_Fund": 50.0,
                f"{score_column}_ML": 0.0,
                f"{score_column}_Reliability": 0.0,
            })
    
    results_df = pd.DataFrame(results).set_index("index")
    return df.join(results_df)


# =============================================================================
# DEPRECATION HELPERS
# =============================================================================

def _deprecation_warning(old_name: str, new_name: str = "UnifiedScorer"):
    """Issue deprecation warning for old scoring functions."""
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )
