"""
Data models and structures for Stock Scout.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd


@dataclass
class StockData:
    """Historical stock price and volume data."""
    ticker: str
    data: pd.DataFrame  # OHLCV data
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.data.empty:
            self.start_date = self.data.index[0] if isinstance(self.data.index[0], datetime) else None
            self.end_date = self.data.index[-1] if isinstance(self.data.index[-1], datetime) else None


@dataclass
class TechnicalScore:
    """Technical analysis score breakdown."""
    total: float
    ma_score: float = 0.0
    momentum_score: float = 0.0
    rsi_score: float = 0.0
    near_high_score: float = 0.0
    volume_score: float = 0.0
    overextension_score: float = 0.0
    pullback_score: float = 0.0
    risk_reward_score: float = 0.0
    macd_score: float = 0.0
    adx_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "ma": self.ma_score,
            "momentum": self.momentum_score,
            "rsi": self.rsi_score,
            "near_high": self.near_high_score,
            "volume": self.volume_score,
            "overextension": self.overextension_score,
            "pullback": self.pullback_score,
            "risk_reward": self.risk_reward_score,
            "macd": self.macd_score,
            "adx": self.adx_score,
        }


@dataclass
class FundamentalBreakdown:
    """Fundamental analysis breakdown with human-friendly labels."""
    
    # Raw scores (0-100)
    quality_score: float = 0.0
    growth_score: float = 0.0
    valuation_score: float = 0.0
    leverage_score: float = 0.0
    stability_score: float = 0.0
    
    # Human-friendly labels
    quality_label: str = "Unknown"  # Low / Medium / High
    growth_label: str = "Unknown"  # Declining / Slow / Moderate / Fast
    valuation_label: str = "Unknown"  # Expensive / Fair / Cheap
    leverage_label: str = "Unknown"  # High / Medium / Low
    stability_label: str = "Unknown"  # Very Volatile / Volatile / Moderate / Stable
    
    # Raw metrics
    roe: Optional[float] = None
    roic: Optional[float] = None
    gross_margin: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    pe_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "quality_score": self.quality_score,
            "growth_score": self.growth_score,
            "valuation_score": self.valuation_score,
            "leverage_score": self.leverage_score,
            "stability_score": self.stability_score,
            "quality_label": self.quality_label,
            "growth_label": self.growth_label,
            "valuation_label": self.valuation_label,
            "leverage_label": self.leverage_label,
            "stability_label": self.stability_label,
            "roe": self.roe,
            "roic": self.roic,
            "gross_margin": self.gross_margin,
            "revenue_growth_yoy": self.revenue_growth_yoy,
            "eps_growth_yoy": self.eps_growth_yoy,
            "pe_ratio": self.pe_ratio,
            "ps_ratio": self.ps_ratio,
            "debt_to_equity": self.debt_to_equity,
        }


@dataclass
class FundamentalScore:
    """Fundamental score with detailed breakdown."""
    total: float
    breakdown: FundamentalBreakdown = field(default_factory=FundamentalBreakdown)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "breakdown": self.breakdown.to_dict(),
        }


@dataclass
class AdvancedSignals:
    """Advanced filtering signals."""
    rs_21d: float = 0.0
    rs_63d: float = 0.0
    volume_surge: float = 0.0
    pv_correlation: float = 0.0
    consolidation_ratio: float = 0.0
    ma_aligned: bool = False
    alignment_score: float = 0.0
    trend_strength: float = 0.0
    distance_to_support: float = 0.0
    distance_to_resistance: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0
    momentum_consistency: float = 0.0
    momentum_acceleration: float = 0.0
    risk_reward_ratio: float = 0.0
    potential_reward: float = 0.0
    potential_risk: float = 0.0
    high_confidence: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rs_21d": self.rs_21d,
            "rs_63d": self.rs_63d,
            "volume_surge": self.volume_surge,
            "pv_correlation": self.pv_correlation,
            "consolidation_ratio": self.consolidation_ratio,
            "ma_aligned": self.ma_aligned,
            "alignment_score": self.alignment_score,
            "trend_strength": self.trend_strength,
            "distance_to_support": self.distance_to_support,
            "distance_to_resistance": self.distance_to_resistance,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            "momentum_consistency": self.momentum_consistency,
            "momentum_acceleration": self.momentum_acceleration,
            "risk_reward_ratio": self.risk_reward_ratio,
            "potential_reward": self.potential_reward,
            "potential_risk": self.potential_risk,
            "high_confidence": self.high_confidence,
        }


@dataclass
class StockRecommendation:
    """Complete stock recommendation with all scores and data."""
    ticker: str
    final_score: float
    technical_score: TechnicalScore
    fundamental_score: Optional[FundamentalScore] = None
    advanced_signals: Optional[AdvancedSignals] = None
    
    # Market data
    current_price: float = 0.0
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    avg_volume: Optional[float] = None
    
    # Allocation
    allocation_amount: float = 0.0
    allocation_pct: float = 0.0
    shares: int = 0
    
    # Next earnings date
    next_earnings: Optional[datetime] = None
    
    # Rejection info
    rejected: bool = False
    rejection_reason: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "final_score": self.final_score,
            "technical_score": self.technical_score.to_dict() if self.technical_score else {},
            "fundamental_score": self.fundamental_score.to_dict() if self.fundamental_score else {},
            "advanced_signals": self.advanced_signals.to_dict() if self.advanced_signals else {},
            "current_price": self.current_price,
            "sector": self.sector,
            "market_cap": self.market_cap,
            "beta": self.beta,
            "avg_volume": self.avg_volume,
            "allocation_amount": self.allocation_amount,
            "allocation_pct": self.allocation_pct,
            "shares": self.shares,
            "next_earnings": self.next_earnings.isoformat() if self.next_earnings else None,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class Portfolio:
    """Portfolio allocation result."""
    recommendations: List[StockRecommendation]
    total_allocated: float
    total_budget: float
    allocation_pct: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "recommendations": [r.to_dict() for r in self.recommendations],
            "total_allocated": self.total_allocated,
            "total_budget": self.total_budget,
            "allocation_pct": self.allocation_pct,
        }
