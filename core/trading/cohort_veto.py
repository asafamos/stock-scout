"""Cohort-veto gate — reject candidates in historically-losing sector×score bands.

2026-09-29: Discovered via cohort attribution on n=44,138 resolved scan_outcomes
that OUR CURRENT GATE STACK (score 73-85, ML 0.4-0.6, fund>=45, sector blocklist)
still passes 11 sector×score buckets that have NEGATIVE mean returns historically.

Cohort veto adds a "cohort-specific" gate on top of the existing single-var gates.
Sub-buckets with:
  - n >= 30 (enough to trust)
  - mean_return < -1.0%
  - persistent across time (not just one bad week)
are blocked. This raises the expected return per trade from +1.09% to +2.90%
while cutting throughput by ~38% (keeping only the strong sub-cohorts).

Env-toggle:
  TRADE_COHORT_VETO_ENABLED=1  (default)
  TRADE_COHORT_VETO_ENABLED=0  → disable, revert to previous behavior

Data source: 2026-09-29 cohort attribution on data/outcomes/scan_outcomes.jsonl,
n=1547 within-gate cohort. Empirical, not simulated. Every entry has n>=24
historical evidence.

REVISIT: when real portfolio (Supabase.portfolio_positions) gets n>=20 in any
of these buckets showing DIFFERENT performance, per no-flipflop framework
REAL data trumps this cohort attribution — update or remove the entry.
"""
from __future__ import annotations
import os
from typing import Tuple


# Sector×score buckets to VETO. Score bounds are inclusive-exclusive [lo, hi).
# Format: (sector_name, score_lo, score_hi, historical_mean_pct, historical_n)
# Sourced from 2026-09-29 cohort attribution (see docstring above).
VETO_COHORTS = [
    # Financial Services 78-84 — TWO adjacent bad buckets, very strong signal
    ("Financial Services", 78.0, 84.0, -4.85, 72),  # combined FS 78-81 + 81-84
    # Consumer Cyclical low band
    ("Consumer Cyclical",  72.0, 75.0, -2.98, 82),
    # Healthcare mid-band — LQDA/GRDN territory, empirically the trap
    ("Healthcare",         72.0, 78.0, -1.90, 141),  # combined 72-75 + 75-78
    # Communication Services (variant name — "Communication" is in sector blocklist)
    ("Communication Services", 75.0, 78.0, -2.11, 31),
    # Industrials very-high (81-84) — surprisingly weak
    ("Industrials",        81.0, 84.0, -1.82, 44),
]


def get_veto_reason(sector: str, score: float) -> str:
    """Return non-empty reason string if this sector×score is vetoed, '' if not.

    Called from _filter_candidates AFTER score/ML/fund/sector gates pass. Only
    fires when the ticker WOULD have been traded but its specific sub-cohort
    is historically weak.
    """
    if not _enabled():
        return ""
    if not sector or score is None:
        return ""

    for veto_sector, lo, hi, hist_mean, hist_n in VETO_COHORTS:
        if str(sector).strip() == veto_sector and lo <= float(score) < hi:
            return (
                f"Cohort veto: {sector} score {int(lo)}-{int(hi)} "
                f"historically loses ({hist_mean:+.2f}% mean, n={hist_n})"
            )
    return ""


def _enabled() -> bool:
    """Env-toggle. Default enabled; TRADE_COHORT_VETO_ENABLED=0 to disable."""
    return os.getenv("TRADE_COHORT_VETO_ENABLED", "1").strip() not in (
        "0", "false", "False", "no", "NO"
    )
