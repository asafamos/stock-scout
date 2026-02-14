"""
Risk management module - Sector concentration, liquidity filters, position sizing.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

from core.logging_config import get_logger
from core.config import get_config
from core.models import StockRecommendation

logger = get_logger("risk")


@dataclass
class RiskConstraints:
    """Risk management constraints."""
    min_market_cap: float
    min_price: float
    min_avg_volume: int
    min_dollar_volume: int
    max_sector_pct: float
    max_positions_per_sector: int
    max_position_pct: float
    min_position: float
    beta_max: float


class RiskManager:
    """Risk management and filtering."""
    
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or get_config()
    
    def get_constraints(self) -> RiskConstraints:
        """Get current risk constraints from config."""
        return RiskConstraints(
            min_market_cap=self.config.min_market_cap,
            min_price=self.config.min_price,
            min_avg_volume=self.config.min_avg_volume,
            min_dollar_volume=self.config.min_dollar_volume,
            max_sector_pct=self.config.max_sector_allocation_pct,
            max_positions_per_sector=self.config.max_positions_per_sector,
            max_position_pct=self.config.max_position_pct,
            min_position=self.config.min_position,
            beta_max=self.config.beta_max_allowed,
        )
    
    def apply_liquidity_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply liquidity and basic filters.
        
        Args:
            df: DataFrame with Price, Volume, Avg_Volume columns
            
        Returns:
            Filtered DataFrame
        """
        constraints = self.get_constraints()
        
        initial_count = len(df)
        
        # Price filter
        df = df[df["Price"] >= constraints.min_price].copy()
        logger.info(f"Price filter: {initial_count} → {len(df)} (≥${constraints.min_price})")
        
        # Volume filter
        if "Avg_Volume" in df.columns:
            df = df[df["Avg_Volume"] >= constraints.min_avg_volume].copy()
            logger.info(f"Volume filter: → {len(df)} (≥{constraints.min_avg_volume:,})")
        
        # Dollar volume filter
        if "Avg_Volume" in df.columns:
            df["Dollar_Volume"] = df["Price"] * df["Avg_Volume"]
            df = df[df["Dollar_Volume"] >= constraints.min_dollar_volume].copy()
            logger.info(f"Dollar volume filter: → {len(df)} (≥${constraints.min_dollar_volume:,})")
        
        # Market cap filter (if available)
        if "Market_Cap" in df.columns:
            df = df[
                (df["Market_Cap"].isna()) | 
                (df["Market_Cap"] >= constraints.min_market_cap)
            ].copy()
            logger.info(f"Market cap filter: → {len(df)} (≥${constraints.min_market_cap:,.0f})")
        
        return df
    
    def apply_beta_filter(self, df: pd.DataFrame, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Filter stocks by beta vs benchmark.
        
        Args:
            df: DataFrame with Beta column
            top_k: Keep only top K stocks by score
            
        Returns:
            Filtered DataFrame
        """
        constraints = self.get_constraints()
        
        if "Beta" not in df.columns:
            return df
        
        initial_count = len(df)
        
        # Filter by max beta
        df = df[
            (df["Beta"].isna()) | 
            (df["Beta"] <= constraints.beta_max)
        ].copy()
        
        logger.info(f"Beta filter: {initial_count} → {len(df)} (≤{constraints.beta_max})")
        
        # Optionally keep only top K
        if top_k and len(df) > top_k:
            df = df.nlargest(top_k, "Final_Score")
            logger.info(f"Top {top_k} filter: → {len(df)}")
        
        return df
    
    def apply_sector_concentration(
        self,
        df: pd.DataFrame,
        sector_col: str = "Sector"
    ) -> pd.DataFrame:
        """
        Apply sector concentration limits.
        
        Args:
            df: DataFrame with sector information
            sector_col: Name of sector column
            
        Returns:
            DataFrame with sector limits applied
        """
        constraints = self.get_constraints()
        
        if sector_col not in df.columns:
            logger.warning(f"Sector column '{sector_col}' not found, skipping concentration limits")
            return df
        
        initial_count = len(df)
        
        # Count positions per sector
        sector_counts = df[sector_col].value_counts()
        
        # Keep only top N per sector
        kept = []
        for sector in sector_counts.index:
            sector_df = df[df[sector_col] == sector]
            
            # Sort by score and keep top N
            sector_df = sector_df.nlargest(
                constraints.max_positions_per_sector,
                "Final_Score"
            )
            kept.append(sector_df)
        
        df = pd.concat(kept, ignore_index=True)
        
        logger.info(
            f"Sector concentration: {initial_count} → {len(df)} "
            f"(max {constraints.max_positions_per_sector} per sector)"
        )
        
        return df
    
    def calculate_position_sizes(
        self,
        recommendations: List[StockRecommendation],
        total_budget: float
    ) -> List[StockRecommendation]:
        """
        Calculate position sizes with risk constraints.
        
        Args:
            recommendations: List of stock recommendations
            total_budget: Total budget available
            
        Returns:
            Updated recommendations with allocation amounts
        """
        constraints = self.get_constraints()
        
        if not recommendations:
            return recommendations
        
        # Calculate weights based on scores
        total_score = sum(r.final_score for r in recommendations)
        
        if total_score <= 0:
            logger.warning("Total score is zero, using equal weights")
            weight = 1.0 / len(recommendations)
            weights = [weight] * len(recommendations)
        else:
            weights = [r.final_score / total_score for r in recommendations]
        
        # Apply position size constraints
        total_allocated = 0.0
        
        for i, rec in enumerate(recommendations):
            # Calculate raw allocation
            raw_allocation = total_budget * weights[i]
            
            # Apply max position size
            max_position = total_budget * (constraints.max_position_pct / 100.0)
            allocation = min(raw_allocation, max_position)
            
            # Apply min position size
            if allocation < constraints.min_position:
                allocation = 0.0
                logger.debug(
                    f"{rec.ticker}: allocation ${allocation:.2f} < min ${constraints.min_position}, "
                    f"setting to $0"
                )
            
            rec.allocation_amount = allocation
            rec.allocation_pct = (allocation / total_budget * 100.0) if total_budget > 0 else 0.0
            rec.shares = int(allocation / rec.current_price) if rec.current_price > 0 else 0
            
            total_allocated += allocation
        
        logger.info(
            f"Allocated ${total_allocated:,.2f} of ${total_budget:,.2f} "
            f"({total_allocated/total_budget*100:.1f}%)"
        )
        
        return recommendations
    
    def check_earnings_blackout(
        self,
        df: pd.DataFrame,
        blackout_days: int = 7
    ) -> pd.DataFrame:
        """
        Filter out stocks in earnings blackout period.
        
        Args:
            df: DataFrame with Next_Earnings column
            blackout_days: Days before earnings to exclude
            
        Returns:
            Filtered DataFrame
        """
        if "Next_Earnings" not in df.columns:
            return df
        
        initial_count = len(df)
        
        from datetime import datetime, timedelta
        today = datetime.utcnow()
        blackout_date = today + timedelta(days=blackout_days)
        
        # Keep stocks where earnings is None or beyond blackout period
        df = df[
            (df["Next_Earnings"].isna()) |
            (df["Next_Earnings"] > blackout_date)
        ].copy()
        
        filtered = initial_count - len(df)
        if filtered > 0:
            logger.info(
                f"Earnings blackout filter: {initial_count} → {len(df)} "
                f"({filtered} in blackout period)"
            )
        
        return df


# ---------------------------------------------------------------------------
# Unified Reward / Risk calculator (single source of truth)
# ---------------------------------------------------------------------------

def calculate_rr(
    entry_price: float,
    target_price: float,
    atr_value: float,
    history_df: pd.DataFrame = None,
    fallback_price: float = None,
) -> float:
    """Compute Reward/Risk ratio with ATR-based risk estimation.

    Reward = max(0, target − entry)
    Risk   = max(ATR × 2, entry × 1 %)
    Result is clamped to [0, 5].

    ATR fallback chain
    -------------------
    1. *atr_value* (explicit)
    2. Mean(High − Low) over last 14 bars of *history_df*
    3. *fallback_price* (e.g. ATR_Price column from another source)
    4. 1 % of *entry_price*

    Returns *np.nan* for invalid numeric inputs, 0.0 for unexpected errors.
    """
    try:
        if not (isinstance(entry_price, (int, float)) and np.isfinite(entry_price)):
            return np.nan
        if not (isinstance(target_price, (int, float)) and np.isfinite(target_price)):
            return np.nan

        atr = (
            atr_value
            if (isinstance(atr_value, (int, float)) and np.isfinite(atr_value))
            else np.nan
        )

        # Fallback 1: estimate from history_df
        if (not np.isfinite(atr)) and history_df is not None:
            try:
                last = history_df.tail(14)
                if not last.empty and "High" in last.columns and "Low" in last.columns:
                    est_atr = (last["High"] - last["Low"]).abs().dropna().mean()
                    if np.isfinite(est_atr) and est_atr > 0:
                        atr = float(est_atr)
            except Exception as e:
                logger.debug("calculate_rr ATR estimation: %s", e)

        # Fallback 2: use fallback_price (e.g. ATR_Price from another column)
        if (
            not np.isfinite(atr)
            and isinstance(fallback_price, (int, float))
            and np.isfinite(fallback_price)
        ):
            atr = fallback_price

        # Fallback 3: 1% of entry price
        if not np.isfinite(atr):
            atr = max(0.01 * float(entry_price), 1e-6)

        risk = max(atr * 2.0, float(entry_price) * 0.01)
        reward = max(0.0, float(target_price) - float(entry_price))
        rr = 0.0 if risk <= 0 else reward / risk
        return float(np.clip(rr, 0.0, 5.0))
    except Exception as e:
        logger.debug("calculate_rr unexpected: %s", e)
        return 0.0
