"""Adaptive gate relaxation — tracks consecutive dry cycles and unblocks
trading when the system is stuck rejecting all candidates.

Deployed 2026-07-21 after 6 consecutive pipelines (Fri 07-17 #4/#5, Mon
07-20 #1/#2/#3, Tue 07-21 #1) all died on:
    "Confidence filter dropped X stocks (< High, regime=MODERATE_UP)"
    "No stocks pass confidence filter"

Root context: `confidence_regime_relax` (env TRADE_CONFIDENCE_REGIME_RELAX)
was disabled 2026-07-03 after PR was bought under buggy conditions (2
stacked bugs). Since then, bullish regimes (MODERATE_UP with VIX 11-12)
produce zero buys because the confidence filter tightened too hard.

This module: no manual flip. When N consecutive dry cycles hit due to
Confidence-blocked, auto-relax to Medium for the NEXT cycle. Reset the
streak on any buy. Alert user via Telegram on every state change.

State file: data/state/adaptive_gates.json (persists between runs).
Env kill-switch: TRADE_ADAPTIVE_GATES_ENABLED=0 disables the whole thing.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/state/adaptive_gates.json")

_DEFAULTS = {
    "confidence_dry_streak": 0,
    "confidence_last_updated": "",
    "confidence_relaxed_active": False,
    "confidence_relaxed_since": "",
    "confidence_last_regime": "",
}

# Regimes where Medium confidence is defensible (macro tailwind compensates
# for weaker per-stock confirmation). Mirror _MEDIUM_OK_REGIMES from policy.
_BULLISH_REGIMES = {"MODERATE_UP", "STRONG_UPTREND", "TREND_UP", "STRONG_UP", "UPTREND"}


def _load() -> dict:
    if not STATE_PATH.exists():
        return dict(_DEFAULTS)
    try:
        data = json.loads(STATE_PATH.read_text())
        return {**_DEFAULTS, **data}
    except Exception:
        return dict(_DEFAULTS)


def _save(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["confidence_last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)


def get_state() -> dict:
    """Public read of current adaptive state (for status/debug)."""
    return _load()


def get_adaptive_confidence_relaxed() -> bool:
    """True if the confidence gate should be relaxed to Medium right now.

    Called from policy.confidence_floor() at every candidate evaluation.
    Reads state — no side effects.
    """
    return _load().get("confidence_relaxed_active", False)


def record_pipeline_outcome(
    bought: int,
    confidence_dropped: bool,
    regime: str,
    threshold: int = 5,
) -> Optional[str]:
    """Called at the end of every auto-trade pipeline. Updates streak state.

    Args:
        bought: number of BUYs in this pipeline
        confidence_dropped: True iff the confidence filter rejected candidates
                            (i.e. we had candidates that would have passed
                            everything else if not for confidence)
        regime: the current market regime string
        threshold: dry-streak count at which auto-relax activates

    Returns:
        A message string if state changed (fed to Telegram), else None.
    """
    state = _load()
    old_relaxed = bool(state.get("confidence_relaxed_active", False))
    old_streak = int(state.get("confidence_dry_streak", 0))

    # Any buy → reset. This is the strongest signal that gates are OK.
    if bought > 0:
        if old_streak > 0 or old_relaxed:
            state["confidence_dry_streak"] = 0
            state["confidence_relaxed_active"] = False
            state["confidence_last_regime"] = regime
            _save(state)
            return (
                f"✅ Adaptive gate RESET — {bought} bought.\n"
                f"Dry streak was {old_streak}, relax was {'ON' if old_relaxed else 'OFF'}.\n"
                f"Confidence gate back to strict (High) for next cycle."
            )
        return None

    # 0 buys but NOT blocked by confidence — irrelevant for this counter.
    if not confidence_dropped:
        return None

    # 0 buys AND confidence blocked → increment streak.
    new_streak = old_streak + 1
    state["confidence_dry_streak"] = new_streak
    state["confidence_last_regime"] = regime

    is_bullish = regime.upper() in _BULLISH_REGIMES
    msg = None

    if new_streak >= threshold and is_bullish and not old_relaxed:
        # Activate relax.
        state["confidence_relaxed_active"] = True
        state["confidence_relaxed_since"] = datetime.now(timezone.utc).isoformat()
        msg = (
            f"🔓 Adaptive gate ACTIVATED\n"
            f"{new_streak} consecutive dry cycles due to Confidence < High.\n"
            f"Regime {regime} is bullish → auto-relaxing to Medium for next cycle.\n"
            f"Will reset on any successful buy. "
            f"Kill switch: TRADE_ADAPTIVE_GATES_ENABLED=0."
        )
    elif new_streak == threshold - 1 and is_bullish and not old_relaxed:
        # Pre-warning one cycle before activation.
        msg = (
            f"⏳ Adaptive gate PRE-WARN\n"
            f"{new_streak}/{threshold} dry cycles (Confidence blocked).\n"
            f"Regime {regime}. Next dry cycle → auto-relax to Medium."
        )
    elif old_relaxed and not is_bullish:
        # Regime turned non-bullish — deactivate relax as safety.
        state["confidence_relaxed_active"] = False
        msg = (
            f"🔒 Adaptive gate DEACTIVATED — regime shift\n"
            f"Regime {regime} is no longer bullish. Restoring strict Confidence "
            f"(High) even though streak is {new_streak}."
        )

    _save(state)
    return msg


def force_reset() -> None:
    """Manual reset — for CLI/debug. Wipes state to defaults."""
    _save(dict(_DEFAULTS))
