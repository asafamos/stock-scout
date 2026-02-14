"""Fallback tracking for the scan pipeline.

Records whether legacy scoring was used instead of the V2 Bridge during pipeline runs.
"""

import logging
from threading import Lock
from typing import List

logger = logging.getLogger(__name__)

# --- Fallback tracking (for meta) ---
# Tracks whether any legacy/bridge fallback occurred during the run
_LEGACY_FALLBACK_USED: bool = False
_LEGACY_FALLBACK_REASONS: List[str] = []
_LEGACY_LOCK: Lock = Lock()


def _record_legacy_fallback(reason: str) -> None:
    """Record and log fallback from V2 Bridge to legacy scoring.

    IMPORTANT: This should be visible to users — silent fallbacks hide bugs.
    """
    try:
        with _LEGACY_LOCK:
            global _LEGACY_FALLBACK_USED, _LEGACY_FALLBACK_REASONS
            _LEGACY_FALLBACK_USED = True
            if reason:
                _LEGACY_FALLBACK_REASONS.append(str(reason))
        # Log prominently so it's not hidden
        logger.warning(f"⚠️ FALLBACK TO LEGACY SCORING: {reason}")
        logger.warning("   ML/Risk Bridge failed - using older scoring logic. Results may differ.")
    except (RuntimeError, TypeError) as lock_exc:
        # Best-effort; do not raise
        logger.debug(f"Fallback marking failed: {lock_exc}")


def get_fallback_status() -> dict:
    """Get status of fallback usage for current run — exposed for UI/API."""
    with _LEGACY_LOCK:
        return {
            "fallback_used": _LEGACY_FALLBACK_USED,
            "fallback_count": len(_LEGACY_FALLBACK_REASONS),
            "reasons": list(_LEGACY_FALLBACK_REASONS[-10:]),  # Last 10 reasons
        }


def reset_fallback_state() -> None:
    """Reset fallback trackers at the start of a new pipeline run."""
    try:
        with _LEGACY_LOCK:
            global _LEGACY_FALLBACK_USED, _LEGACY_FALLBACK_REASONS
            _LEGACY_FALLBACK_USED = False
            _LEGACY_FALLBACK_REASONS = []
    except (RuntimeError, NameError):
        pass
