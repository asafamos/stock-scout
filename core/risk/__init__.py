"""
Unified Risk Management Package for Stock Scout.

Consolidates functionality from three legacy modules:
- ``core.risk`` — RiskManager (filters, sizing, sector caps)
- ``core.risk_engine`` — RiskEngine (conviction → action mapping)
- ``core.v2_risk_engine`` — V2 (reliability-gated, position risk factor)

Preferred usage::

    from core.risk import (
        RiskManager,
        RiskConstraints,
        calculate_rr,
        calculate_risk_meter,
        generate_warnings,
        compute_position_risk_factor,
    )
"""
from __future__ import annotations

# From legacy core/risk.py (filters, position sizing, RR calculation)
from core.risk_module import (  # noqa: F401
    RiskManager,
    RiskConstraints,
    calculate_rr,
)

# From scoring_engine (risk meter, warnings) - will move here in future
from core.scoring_engine import (  # noqa: F401
    calculate_risk_meter,
    generate_warnings,
)

# From v2_risk_engine (position risk factor)
from core.v2_risk_engine import (  # noqa: F401
    compute_position_risk_factor,
    calculate_reliability_v2,
    score_ticker_v2_enhanced,
)

# From risk_engine (conviction → action mapping)
from core.risk_engine import (  # noqa: F401
    RiskEngine,
)

__all__ = [
    "RiskManager",
    "RiskConstraints",
    "RiskEngine",
    "calculate_rr",
    "calculate_risk_meter",
    "generate_warnings",
    "compute_position_risk_factor",
    "calculate_reliability_v2",
    "score_ticker_v2_enhanced",
]
