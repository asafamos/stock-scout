"""Sector Champion — score×sector cohort lookup + ranking bonus.

Backtest evidence (402 real closed positions from Supabase, 2026-07-21):

    Sector        Score      n    Mean_Ret   Win_Rate
    Energy        70-75      12   +2.88%     75.0% 🏆
    Energy        75-80      14   +3.49%     78.6% 🏆
    Energy        80-85      23   +5.24%     73.9% 🏆
    Technology    70-75      10   +3.94%     70.0% 🏆
    Technology    75-80       8   +5.69%     75.0% 🏆
    Technology    80-85       7   +5.98%     71.4% 🏆
    Healthcare    70-75      13   +2.69%     84.6% 🏆
    Industrials   70-75      19   +3.49%     63.2%      (moderate)

Cohorts flagged 🏆 have WR>=70% AND mean_ret>+2% on real data.
Baseline (all 402 closes): mean=+1.06%, WR=51.0%.

Usage:
  * `get_cohort_stats(sector, score)` -> dict with n, mean_ret, wr, is_champion
  * `champion_bonus_mask(sector, score)` -> 1.0 if strong cohort, else 0.0
    Applied in ranking as `bonus_weight * mask` (env: TRADE_SECTOR_CHAMPION_WEIGHT)

Kill switch: TRADE_SECTOR_CHAMPION_WEIGHT=0 disables the ranking bonus.
Alert integration works regardless of the ranker bonus flag.
"""
from typing import Optional

# score_band, sector -> (n, mean_ret_pct, win_rate_pct, is_champion)
# is_champion = WR >= 70 AND mean_ret > 2.0 (on real Supabase data)
_COHORTS: dict = {
    # Energy
    ("Energy", (70, 75)): dict(n=12, mean_ret=2.88, wr=75.0, is_champion=True),
    ("Energy", (75, 80)): dict(n=14, mean_ret=3.49, wr=78.6, is_champion=True),
    ("Energy", (80, 85)): dict(n=23, mean_ret=5.24, wr=73.9, is_champion=True),
    ("Energy", (85, 90)): dict(n=8,  mean_ret=0.27, wr=50.0, is_champion=False),
    # Technology
    ("Technology", (65, 70)): dict(n=9,  mean_ret=-0.11, wr=55.6, is_champion=False),
    ("Technology", (70, 75)): dict(n=10, mean_ret=3.94, wr=70.0, is_champion=True),
    ("Technology", (75, 80)): dict(n=8,  mean_ret=5.69, wr=75.0, is_champion=True),
    ("Technology", (80, 85)): dict(n=7,  mean_ret=5.98, wr=71.4, is_champion=True),
    # Healthcare
    ("Healthcare", (70, 75)): dict(n=13, mean_ret=2.69, wr=84.6, is_champion=True),
    # Industrials (moderate — not champion, but positive)
    ("Industrials", (65, 70)): dict(n=6,  mean_ret=0.56, wr=33.3, is_champion=False),
    ("Industrials", (70, 75)): dict(n=19, mean_ret=3.49, wr=63.2, is_champion=False),
    ("Industrials", (75, 80)): dict(n=11, mean_ret=2.19, wr=45.5, is_champion=False),
    # Negative cohorts — informational only, no bonus
    ("Consumer Cyclical", (70, 75)): dict(n=14, mean_ret=-1.97, wr=21.4, is_champion=False),
    ("Financial Services", (70, 75)): dict(n=8,  mean_ret=0.07, wr=50.0, is_champion=False),
    ("Consumer Defensive", (70, 75)): dict(n=7,  mean_ret=-5.08, wr=28.6, is_champion=False),
    ("Utilities", (65, 70)): dict(n=6, mean_ret=-1.48, wr=16.7, is_champion=False),
    ("Utilities", (70, 75)): dict(n=5, mean_ret=-0.62, wr=60.0, is_champion=False),
}

# Baseline for comparison in alerts
BASELINE = dict(n=402, mean_ret=1.06, wr=51.0)

_SCORE_BANDS = [(65, 70), (70, 75), (75, 80), (80, 85), (85, 90)]


# ── Adaptive edges overlay (added 2026-08-25) ──────────────────────
# When TRADE_ADAPTIVE_EDGES_APPLY=1 and data/adaptive/current_edges.json
# exists, dynamic champions from live scan_outcomes MERGE with the
# hardcoded _COHORTS (union: adaptive additions OVERRIDE hardcoded
# for same sector×band). Recomputed nightly by
# scripts/compute_adaptive_edges.py — data-driven, no more stale hardcoded.
# Freeze-safe: env-toggleable, falls back to hardcoded on any error.
import os as _os
import json as _json
from pathlib import Path as _Path

_ADAPTIVE_CACHE: dict = {}
_ADAPTIVE_MTIME: float = 0.0
_ADAPTIVE_PATH = _Path(
    "/home/stockscout/stock-scout-2/data/adaptive/current_edges.json"
) if _Path("/home/stockscout").exists() else \
    _Path(__file__).resolve().parents[2] / "data" / "adaptive" / "current_edges.json"


def _load_adaptive_cohorts() -> dict:
    """Load champions from current_edges.json (cached by mtime).

    Returns dict keyed like _COHORTS: (sector, (lo, hi)) -> stats dict.
    Falls back to {} on any error — hardcoded _COHORTS still work.
    """
    global _ADAPTIVE_CACHE, _ADAPTIVE_MTIME
    if _os.getenv("TRADE_ADAPTIVE_EDGES_APPLY", "0") not in ("1", "true", "yes"):
        return {}
    try:
        mt = _ADAPTIVE_PATH.stat().st_mtime
        if mt == _ADAPTIVE_MTIME and _ADAPTIVE_CACHE:
            return _ADAPTIVE_CACHE
        with _ADAPTIVE_PATH.open() as f:
            edges = _json.load(f)
        out = {}
        for c in (edges.get("champions") or []):
            sec = c.get("sector") or ""
            lo = int(c.get("score_min", 0))
            hi = int(c.get("score_max", 0))
            if sec and lo < hi:
                out[(sec, (lo, hi))] = dict(
                    n=int(c.get("n", 0)),
                    mean_ret=float(c.get("mean", 0)),
                    wr=float(c.get("wr", 0)),
                    is_champion=True,  # by definition — passed champion filter
                    source="adaptive",
                )
        _ADAPTIVE_CACHE = out
        _ADAPTIVE_MTIME = mt
        return out
    except Exception:
        return {}


def _find_band(score: float) -> Optional[tuple]:
    """Find band matching hardcoded _SCORE_BANDS OR adaptive bands."""
    # First check adaptive bands (they use 5-point bands 0-5-10-... intervals)
    if score is not None:
        adaptive = _load_adaptive_cohorts()
        for (_sec, band) in adaptive.keys():
            lo, hi = band
            if lo <= score < hi:
                return band
    # Fallback to hardcoded bands
    for lo, hi in _SCORE_BANDS:
        if lo <= score < hi:
            return (lo, hi)
    return None


def get_cohort_stats(sector: str, score: float) -> Optional[dict]:
    """Return cohort stats — adaptive edges overlay hardcoded _COHORTS."""
    if not sector or score is None:
        return None
    band = _find_band(float(score))
    if band is None:
        return None
    # Adaptive first — overlays if newer data
    adaptive = _load_adaptive_cohorts()
    if (sector, band) in adaptive:
        return adaptive[(sector, band)]
    return _COHORTS.get((sector, band))


def is_sector_champion(sector: str, score: float) -> bool:
    """True if this (sector, score) combo is a historically-strong cohort
    (WR>=70%, mean_ret>+2% on real closes). Used for ranking bonus + alert."""
    stats = get_cohort_stats(sector, score)
    return bool(stats and stats.get("is_champion"))


def champion_bonus_mask(sector: str, score: float) -> float:
    """Ranker-side bonus mask. Returns 1.0 for champion cohorts, else 0.0.
    Multiplied by `TRADE_SECTOR_CHAMPION_WEIGHT` in the ranker."""
    return 1.0 if is_sector_champion(sector, score) else 0.0


def format_cohort_line(sector: str, score: float) -> str:
    """One-line summary for Telegram alerts. Empty string if no cohort match."""
    stats = get_cohort_stats(sector, score)
    if not stats:
        return ""
    marker = " 🏆" if stats.get("is_champion") else ""
    return (
        f"  Cohort: {sector} score-band → "
        f"n={stats['n']} WR={stats['wr']:.0f}% mean={stats['mean_ret']:+.2f}%{marker}"
    )
