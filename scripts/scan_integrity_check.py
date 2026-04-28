"""Scan integrity check — fails the GH Actions step if the scan output
is broken in a way that would silently corrupt the pipeline.

Catches the failure modes we've actually seen in production:
- Empty parquet (0 rows) — caused 4 days of broken outcomes-record
- Missing critical columns — order_manager would crash later
- Wild score values (negative, > 100, NaN) — pipeline math breaks
- Market_Regime missing on most rows — regime-aware filters become useless
- JSON metadata corrupted with merge-conflict markers

Usage:
    python -m scripts.scan_integrity_check                # exit 0 if OK
    python -m scripts.scan_integrity_check --strict       # also fail on warnings

Run it AFTER the scan completes but BEFORE git commit + push, so a broken
scan never ends up on origin/main where the VPS would pull it.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "scans" / "latest_scan.parquet"
JSON_META = ROOT / "data" / "scans" / "latest_scan.json"

# Columns that order_manager / risk_manager actually read.
# Missing any of these → downstream KeyError or silent garbage.
REQUIRED_COLS = [
    "Ticker",
    "FinalScore_20d",
    "RewardRisk",
    "ML_20d_Prob",
    "Sector",
    "Market_Regime",
    "Entry_Price",
    "Stop_Loss",
    "Target_Price",
]

MIN_ROWS = 30           # Below this → scan likely truncated mid-run
MAX_ROWS = 5000         # Above this → universe filter broken
SCORE_MIN = 0.0
SCORE_MAX = 100.0
REGIME_MISSING_MAX_FRAC = 0.50  # >50% of rows with no regime → broadcast broken


def check_parquet() -> Tuple[List[str], List[str]]:
    """Return (errors, warnings)."""
    import pandas as pd
    errors, warnings = [], []

    if not PARQUET.exists():
        errors.append(f"Scan parquet missing: {PARQUET}")
        return errors, warnings

    size_bytes = PARQUET.stat().st_size
    if size_bytes < 1000:
        errors.append(f"Parquet absurdly small ({size_bytes}B) — likely empty schema-only")

    try:
        df = pd.read_parquet(PARQUET)
    except Exception as e:
        errors.append(f"Parquet unreadable: {e}")
        return errors, warnings

    n = len(df)
    if n == 0:
        errors.append("Parquet has 0 rows")
        return errors, warnings
    if n < MIN_ROWS:
        errors.append(f"Only {n} rows (< {MIN_ROWS} expected) — scan likely truncated")
    if n > MAX_ROWS:
        warnings.append(f"{n} rows (> {MAX_ROWS}) — universe filter may be broken")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Required columns missing: {missing}")

    # Score sanity
    if "FinalScore_20d" in df.columns:
        scores = pd.to_numeric(df["FinalScore_20d"], errors="coerce").dropna()
        if scores.empty:
            errors.append("FinalScore_20d column has no numeric values")
        else:
            if scores.min() < SCORE_MIN or scores.max() > SCORE_MAX:
                errors.append(
                    f"Scores out of range: min={scores.min():.1f}, max={scores.max():.1f} "
                    f"(expected [{SCORE_MIN}, {SCORE_MAX}])"
                )
            if scores.isna().any():
                warnings.append(f"{scores.isna().sum()} rows with NaN score")

    # Regime broadcast (caused real bug 2026-04-22..27)
    if "Market_Regime" in df.columns:
        regime_missing = df["Market_Regime"].isna() | (df["Market_Regime"].astype(str) == "")
        miss_frac = regime_missing.sum() / max(n, 1)
        if miss_frac > REGIME_MISSING_MAX_FRAC:
            errors.append(
                f"Market_Regime missing on {miss_frac*100:.0f}% of rows "
                f"(> {REGIME_MISSING_MAX_FRAC*100:.0f}% threshold) — "
                f"regime broadcast broken"
            )

    # ML probability sanity
    if "ML_20d_Prob" in df.columns:
        ml = pd.to_numeric(df["ML_20d_Prob"], errors="coerce").dropna()
        if not ml.empty:
            if ml.min() < 0 or ml.max() > 1:
                errors.append(
                    f"ML_20d_Prob out of [0,1]: min={ml.min():.3f}, max={ml.max():.3f}"
                )

    logger.info(
        "Parquet: %d rows, %d cols, score [%.1f, %.1f]",
        n, len(df.columns),
        df["FinalScore_20d"].min() if "FinalScore_20d" in df.columns else -1,
        df["FinalScore_20d"].max() if "FinalScore_20d" in df.columns else -1,
    )
    return errors, warnings


def check_json() -> Tuple[List[str], List[str]]:
    errors, warnings = [], []
    if not JSON_META.exists():
        warnings.append(f"JSON metadata missing: {JSON_META}")
        return errors, warnings
    text = JSON_META.read_text()
    # Catch git merge markers — root cause of "outcomes-record" crash on 2026-04-27
    for marker in ("<<<<<<<", "=======", ">>>>>>>"):
        if marker in text:
            errors.append(f"JSON contains git conflict marker: {marker!r}")
    try:
        data = json.loads(text)
    except Exception as e:
        errors.append(f"JSON unparseable: {e}")
        return errors, warnings
    if not isinstance(data, dict):
        errors.append(f"JSON is not a dict (got {type(data).__name__})")
        return errors, warnings
    expected_keys = {"timestamp", "results_count", "columns"}
    missing_keys = expected_keys - set(data.keys())
    if missing_keys:
        warnings.append(f"JSON missing keys: {missing_keys}")
    return errors, warnings


def notify_telegram(errors: List[str], warnings: List[str]):
    """Best-effort Telegram alert if integrity check fails. Non-blocking."""
    try:
        from core.trading.notifications import _send
        msg_lines = ["⚠️ <b>SCAN INTEGRITY FAIL</b>"]
        if errors:
            msg_lines.append("\n<b>Errors:</b>")
            msg_lines.extend(f"  • {e}" for e in errors)
        if warnings:
            msg_lines.append("\n<b>Warnings:</b>")
            msg_lines.extend(f"  • {w}" for w in warnings)
        msg_lines.append("\nScan was NOT pushed to origin.")
        _send("\n".join(msg_lines))
    except Exception as e:
        logger.warning("Telegram notify failed: %s", e)


def main() -> int:
    strict = "--strict" in sys.argv
    p_errors, p_warns = check_parquet()
    j_errors, j_warns = check_json()

    errors = p_errors + j_errors
    warnings_ = p_warns + j_warns

    if warnings_:
        for w in warnings_:
            logger.warning("WARN: %s", w)
    if errors:
        for e in errors:
            logger.error("FAIL: %s", e)
        notify_telegram(errors, warnings_)
        print("\n[FAIL] scan integrity check failed")
        return 2
    if strict and warnings_:
        notify_telegram([], warnings_)
        print("\n[FAIL] strict mode: warnings count as errors")
        return 1
    print("[OK] scan integrity check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
