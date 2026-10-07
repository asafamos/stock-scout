"""Sector label synonyms (2026-10-07, P3-c).

Providers (FMP / Yahoo / Finviz / the static map) label the SAME GICS sector differently, and the scan history shows 17 variants
('Consumer Staples' 36 rows vs 'Consumer Defensive' 144, 'Materials' vs 'Basic Materials', 'Financial' vs 'Financial Services', ...).
The blocklist and the per-sector position cap compare exact strings, so a synonym silently escaped the block ('Consumer Staples' vs the
blocked 'Consumer Defensive', which the data showed at -3.13%, p=0.006).

Only UNAMBIGUOUS synonyms are grouped. 'Communication' vs 'Communication Services' is deliberately NOT grouped: the owner blocked
'Communication' as a precaution while 'Communication Services' is his best real cohort (+2.25%, n=26) — a policy decision, not a typo.
"""
from __future__ import annotations

from typing import Iterable, List

# each tuple = one real-world sector, spelled the ways providers spell it
SYNONYM_GROUPS = (
    ("Consumer Defensive", "Consumer Staples"),
    ("Consumer Cyclical", "Consumer Discretionary"),
    ("Basic Materials", "Materials"),
    ("Financial Services", "Financial"),
    ("Technology", "Information Technology"),
    ("Healthcare", "Health Care"),
)
_CANON = {}
for _g in SYNONYM_GROUPS:
    for _name in _g:
        _CANON[_name.strip().lower()] = _g[0]


def canonical_sector(label: str) -> str:
    """Canonical spelling of a sector (unknown labels are returned stripped, unchanged)."""
    s = str(label or "").strip()
    return _CANON.get(s.lower(), s)


def same_sector(a: str, b: str) -> bool:
    return bool(str(a or "").strip()) and canonical_sector(a).lower() == canonical_sector(b).lower()


def expand_blocklist(blocked: Iterable[str]) -> List[str]:
    """Blocked labels plus every synonym of each blocked sector (order kept, no duplicates)."""
    out, seen = [], set()
    def add(x):
        k = x.strip().lower()
        if x.strip() and k not in seen:
            seen.add(k); out.append(x.strip())
    for b in blocked:
        add(b)
    for b in list(out):
        for g in SYNONYM_GROUPS:
            if b.strip().lower() in {n.lower() for n in g}:
                for n in g:
                    add(n)
    return out
