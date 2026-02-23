"""
Stock Scout Application Configuration

Thin wrapper that exposes a backward-compatible CONFIG dict auto-generated
from :class:`core.config.Config`.  New code should import from
``core.config`` directly::

    from core.config import get_config
    cfg = get_config()
    cfg.budget_total   # preferred

Legacy code can continue to use::

    from app_config import CONFIG
    CONFIG["BUDGET_TOTAL"]
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List

from core.config import get_config

# ── CONFIG dict (auto-generated from Config dataclass) ─────────────
CONFIG: Dict[str, Any] = get_config().to_dict()

# ── Fundamentals schema ────────────────────────────────────────────
FUND_SCHEMA_FIELDS: List[str] = [
    "oper_margin",
    "roe",
    "roic",
    "gm",
    "ps",
    "pe",
    "de",
    "rev_g_yoy",
    "eps_g_yoy",
]

FUND_STRING_FIELDS: List[str] = ["sector", "industry"]


def empty_fund_row() -> Dict[str, Any]:
    """Create empty fundamentals dict for a ticker (all NaN / Unknown)."""
    out: Dict[str, Any] = {f: np.nan for f in FUND_SCHEMA_FIELDS}
    out["sector"] = "Unknown"
    out["industry"] = "Unknown"
    out["_sources"] = {}
    out["_sources_used"] = []
    out["Fund_Coverage_Pct"] = 0.0
    out["fundamentals_available"] = False
    return out
