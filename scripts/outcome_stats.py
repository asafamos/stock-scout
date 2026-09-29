"""Date-clustered statistics for scan-outcome style data.

Picks made on the same day share the market's fate and 20-day windows overlap, so treating rows
as independent (as ad-hoc analyses did) inflates t-stats. Here the unit of inference is the SCAN
DATE: average within a date first, then bootstrap over dates (and report how many dates there are).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def date_means(dates: Sequence[str], values: Sequence[float]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for d, v in zip(dates, values):
        if v is None or not np.isfinite(v):
            continue
        acc.setdefault(d, []).append(float(v))
    return {d: float(np.mean(v)) for d, v in acc.items()}


def cluster_bootstrap_mean(per_date: Dict[str, float], n_boot: int = 4000, seed: int = 7) -> Dict[str, float]:
    """Mean of per-date means with a percentile bootstrap CI over dates."""
    vals = np.array(list(per_date.values()), dtype=float)
    n = len(vals)
    if n == 0:
        return {"n_dates": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_le_0": float("nan")}
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, n), replace=True).mean(axis=1)
    return {
        "n_dates": n, "mean": float(vals.mean()),
        "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
        "p_le_0": float((boots <= 0).mean()),
    }


def paired_diff(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """Bootstrap the mean of (a - b) over the dates present in BOTH."""
    common = sorted(set(a) & set(b))
    return cluster_bootstrap_mean({d: a[d] - b[d] for d in common})


def rank_ic(xs: Iterable[float], ys: Iterable[float]) -> float:
    x, y = np.asarray(list(xs), float), np.asarray(list(ys), float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    rx = x[m].argsort().argsort().astype(float)
    ry = y[m].argsort().argsort().astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])
