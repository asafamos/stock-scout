"""
Core contracts for pipeline outputs.

These dataclasses define the single source of truth for the
shape of data produced by scans, diagnostics, and recommendations.
No business logic is included—types only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


# --- Provider and diagnostics primitives ---

@dataclass(frozen=True)
class DataProviderStats:
    """Summary statistics for a single external data provider.

    - provider: Identifier (e.g., "alpha_vantage", "finnhub", "polygon", "tiingo").
    - requests: Total requests made.
    - successes: Successful responses.
    - failures: Failed responses (HTTP errors, parse errors, etc.).
    - rate_limit_hits: Count of rate limit events encountered.
    - cached_hits: Responses served from cache.
    - avg_latency_ms: Optional average response latency in milliseconds.
    """
    provider: str
    requests: int
    successes: int
    failures: int
    rate_limit_hits: int
    cached_hits: int
    avg_latency_ms: Optional[float]


@dataclass(frozen=True)
class CoverageMetrics:
    """Coverage metrics for the scan pipeline."""
    universe_size: int
    fetched_history_count: int
    fundamentals_count: int
    external_price_verified_count: int
    earnings_known_count: int


@dataclass(frozen=True)
class FallbackCounts:
    """Counts of fallbacks used across providers."""
    alpha_to_finnhub: int
    polygon_to_tiingo: int
    other_fallbacks: Dict[str, int]


@dataclass(frozen=True)
class DisagreementStats:
    """Agreement/disagreement across external price providers."""
    price_std_mean: Optional[float]
    high_disagreement_count: int
    agreement_pct: Optional[float]


@dataclass(frozen=True)
class MLStatus:
    """Status of ML components used in the pipeline."""
    enabled: bool
    model_name: Optional[str]
    model_version: Optional[str]
    loaded: bool
    inference_ok: bool
    inference_count: int
    last_trained_ts: Optional[datetime]
    metrics: Optional[Dict[str, float]]


@dataclass(frozen=True)
class MarketContextStatus:
    """Status summary for computed market context/regime."""
    ok: bool
    regime: Optional[str]
    computed_at: Optional[datetime]
    vix_level: Optional[float]
    beta_mean: Optional[float]
    spx_trend: Optional[str]
    note: Optional[str]


# --- Metadata and diagnostics containers ---

@dataclass(frozen=True)
class ScanMetadata:
    """Metadata describing a scan run."""
    scan_id: str
    logic_version: str
    timestamp: datetime
    data_provider_stats: Dict[str, DataProviderStats]
    warnings: List[str]


@dataclass(frozen=True)
class Diagnostics:
    """Diagnostics collected during the scan."""
    coverage: CoverageMetrics
    fallbacks: FallbackCounts
    disagreement: DisagreementStats
    ml_status: MLStatus
    market_context_status: MarketContextStatus


# --- Recommendation primitives ---

@dataclass(frozen=True)
class Targets:
    """Price targets and risk guardrails for a recommendation."""
    entry: Optional[float]
    target_20d: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]


@dataclass(frozen=True)
class RiskSizing:
    """Position sizing information for a recommendation."""
    position_size_usd: Optional[float]
    position_size_shares: Optional[float]
    max_risk_pct: Optional[float]
    atr_shares: Optional[float]
    risk_notes: Optional[str]


@dataclass(frozen=True)
class Recommendation:
    """A single ticker recommendation with scoring details."""
    ticker: str
    scores_breakdown: Dict[str, float]
    final_score_20d: float
    rr: Optional[float]
    beta: Optional[float]
    targets: Optional[Targets]
    risk_sizing: Optional[RiskSizing]
    reasons: List[str]
    # Classification outputs (optional)
    risk_class: Optional[str] = None
    safety_blocked: Optional[bool] = None
    safety_reasons: Optional[str] = None
    # Legacy-compatible classification fields (optional)
    risk_level: Optional[str] = None
    data_quality: Optional[str] = None
    confidence_level: Optional[str] = None
    should_display: Optional[bool] = None
    # Meteor/advanced signals (optional, snake_case normalized)
    consolidation_ratio: Optional[float] = None
    pocket_pivot_ratio: Optional[float] = None
    vcp_ratio: Optional[float] = None


# --- Top-level scan result ---

@dataclass(frozen=True)
class ScanResult:
    """Top-level output of a scan: metadata, diagnostics, and recommendations."""
    metadata: ScanMetadata
    diagnostics: Diagnostics
    recommendations: List[Recommendation]


__all__ = [
    "DataProviderStats",
    "CoverageMetrics",
    "FallbackCounts",
    "DisagreementStats",
    "MLStatus",
    "MarketContextStatus",
    "ScanMetadata",
    "Diagnostics",
    "Targets",
    "RiskSizing",
    "Recommendation",
    "ScanResult",
]
