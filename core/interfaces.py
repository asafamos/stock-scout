from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime


class Action(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    LIQUIDATE = "LIQUIDATE"
    REJECT = "REJECT"


class MarketRegime(str, Enum):
    BULLISH_TREND = "BULLISH_TREND"
    BEARISH_TREND = "BEARISH_TREND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class DataQuality(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class MarketContext:
    as_of_date: datetime
    spy_return_20d: float
    vix_level: float
    regime: MarketRegime
    market_breadth: float
    is_lookahead_safe: bool = True


@dataclass(frozen=True)
class TickerFeatures:
    # Identity
    ticker: str
    as_of_date: datetime

    # Provenance
    data_timestamp: datetime
    source_map: Dict[str, str]
    quality: DataQuality
    point_in_time_ok: bool

    # Model Input
    model_features: Dict[str, float]

    # Risk Metadata
    risk_metadata: Dict[str, Any]


@dataclass(frozen=True)
class ModelOutput:
    prediction_prob: float
    expected_return: float
    confidence_score: float  # Expected range 0-1
    calibration_factor: float
    model_version: str
    generation_time: float


@dataclass(frozen=True)
class TradeDecision:
    ticker: str
    action: Action
    quantity: int
    limit_price: Optional[float]
    stop_loss_price: float
    target_price: float
    conviction: float
    estimated_commission: float

    # Explainability
    primary_reason: str
    active_filters: List[str]
    risk_penalties: List[str]
    explain_id: Optional[str] = None
