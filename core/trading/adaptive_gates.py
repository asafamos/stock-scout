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
    # Analyst PT veto (added 2026-07-21 second phase). Same pattern:
    # after N consecutive dry cycles blocked by analyst PT overvalued,
    # auto-flip to SOFT MODE (cap target instead of veto).
    "analyst_pt_dry_streak": 0,
    "analyst_pt_relaxed_active": False,
    "analyst_pt_relaxed_since": "",
    # RR gate (added 2026-07-23 — task #145). After N cycles blocked by
    # RR filter, relax min_rr_to_trade from 2.5 → 2.0. Observed 2026-07-23:
    # MODERATE_UP regime + VIX 12 → all candidates had RR 1.7-2.2, gate
    # blocked everything even with confidence relaxed.
    "rr_dry_streak": 0,
    "rr_relaxed_active": False,
    "rr_relaxed_since": "",
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


def get_adaptive_rr_relaxed() -> bool:
    """True if the RR gate should be relaxed to 2.0 (from 2.5) right now.

    Called from order_manager's RR filter step. Reads state — no side
    effects. Activated after N consecutive cycles blocked by RR filter.
    See task #145.
    """
    return _load().get("rr_relaxed_active", False)


def get_adaptive_analyst_pt_relaxed() -> bool:
    """True if the analyst PT veto should switch to SOFT MODE right now.

    Called from order_manager._cap_target_with_analysts. Reads state — no
    side effects.

    SOFT MODE: instead of vetoing (returning None), cap target to
    max(analyst_high, current * 1.06). Buys the stock with a more
    conservative target instead of rejecting outright.
    """
    return _load().get("analyst_pt_relaxed_active", False)


def record_pipeline_outcome(
    bought: int,
    confidence_dropped: bool,
    regime: str,
    threshold: int = 5,
    analyst_pt_dropped: bool = False,
    analyst_pt_threshold: int = 3,
    rr_dropped: bool = False,
    rr_threshold: int = 3,
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

    old_pt_relaxed = bool(state.get("analyst_pt_relaxed_active", False))
    old_pt_streak = int(state.get("analyst_pt_dry_streak", 0))

    old_rr_relaxed = bool(state.get("rr_relaxed_active", False))
    old_rr_streak = int(state.get("rr_dry_streak", 0))

    # Any buy → reset ALL streaks. This is the strongest signal gates are OK.
    if bought > 0:
        changes = []
        if old_streak > 0 or old_relaxed:
            state["confidence_dry_streak"] = 0
            state["confidence_relaxed_active"] = False
            state["confidence_last_regime"] = regime
            changes.append(f"Confidence streak was {old_streak}, relax was {'ON' if old_relaxed else 'OFF'}")
        if old_pt_streak > 0 or old_pt_relaxed:
            state["analyst_pt_dry_streak"] = 0
            state["analyst_pt_relaxed_active"] = False
            changes.append(f"Analyst PT streak was {old_pt_streak}, relax was {'ON' if old_pt_relaxed else 'OFF'}")
        if old_rr_streak > 0 or old_rr_relaxed:
            state["rr_dry_streak"] = 0
            state["rr_relaxed_active"] = False
            changes.append(f"RR streak was {old_rr_streak}, relax was {'ON' if old_rr_relaxed else 'OFF'}")
        if changes:
            _save(state)
            return (
                f"✅ Adaptive gates RESET — {bought} bought.\n"
                + "\n".join(changes)
                + "\nAll gates back to strict for next cycle."
            )
        return None

    # 0 buys and NOTHING gate-blocked → nothing to do (rare — maybe all skipped by
    # gap/slippage/other). Skip all trackers.
    if not confidence_dropped and not analyst_pt_dropped and not rr_dropped:
        return None

    is_bullish = regime.upper() in _BULLISH_REGIMES
    msg = None

    # Confidence streak tracking (only if confidence was the blocker)
    new_streak = old_streak
    if confidence_dropped:
        new_streak = old_streak + 1
        state["confidence_dry_streak"] = new_streak
        state["confidence_last_regime"] = regime

    if confidence_dropped:
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

    # ── Analyst PT dry-streak (independent of Confidence) ──
    # This runs regardless of what happened with Confidence — analyst PT
    # can block even when Confidence passed (e.g. today's Pipeline #2).
    pt_msg = None
    if analyst_pt_dropped:
        new_pt_streak = old_pt_streak + 1
        state["analyst_pt_dry_streak"] = new_pt_streak
        if new_pt_streak >= analyst_pt_threshold and not old_pt_relaxed:
            state["analyst_pt_relaxed_active"] = True
            state["analyst_pt_relaxed_since"] = datetime.now(timezone.utc).isoformat()
            pt_msg = (
                f"🔓 Adaptive gate ACTIVATED (Analyst PT)\n"
                f"{new_pt_streak} consecutive dry cycles blocked by Analyst PT veto.\n"
                f"Switching to SOFT MODE (cap target at analyst_high or +6% "
                f"instead of veto). Will reset on any successful buy. "
                f"Kill: TRADE_ADAPTIVE_GATES_ENABLED=0."
            )
        elif new_pt_streak == analyst_pt_threshold - 1 and not old_pt_relaxed:
            pt_msg = (
                f"⏳ Adaptive gate PRE-WARN (Analyst PT)\n"
                f"{new_pt_streak}/{analyst_pt_threshold} cycles blocked by Analyst PT.\n"
                f"Next dry cycle → auto-switch to SOFT MODE."
            )
    # ── RR dry-streak (independent of Confidence + PT) ──
    # Task #145 — 2026-07-23: RR gate was the actual blocker today after
    # Confidence was relaxed. Adaptive extension covers it symmetrically.
    rr_msg = None
    if rr_dropped:
        new_rr_streak = old_rr_streak + 1
        state["rr_dry_streak"] = new_rr_streak
        if new_rr_streak >= rr_threshold and not old_rr_relaxed:
            state["rr_relaxed_active"] = True
            state["rr_relaxed_since"] = datetime.now(timezone.utc).isoformat()
            rr_msg = (
                f"🔓 Adaptive gate ACTIVATED (RR)\n"
                f"{new_rr_streak} consecutive dry cycles blocked by RR filter.\n"
                f"Relaxing min_rr_to_trade from 2.5 → 2.0 for next cycle. "
                f"Will reset on any successful buy. "
                f"Kill: TRADE_ADAPTIVE_GATES_ENABLED=0."
            )
        elif new_rr_streak == rr_threshold - 1 and not old_rr_relaxed:
            rr_msg = (
                f"⏳ Adaptive gate PRE-WARN (RR)\n"
                f"{new_rr_streak}/{rr_threshold} cycles blocked by RR filter.\n"
                f"Next dry cycle → auto-relax RR floor 2.5 → 2.0."
            )

    # Persist state changes (streaks and relax flags) regardless of whether
    # a message was generated — this cycle happened and must be recorded.
    _save(state)

    # Combine messages (any combination could move in the same cycle).
    parts = [m for m in [msg, pt_msg, rr_msg] if m]
    combined = "\n\n".join(parts) if parts else None
    return combined


def force_reset() -> None:
    """Manual reset — for CLI/debug. Wipes state to defaults."""
    _save(dict(_DEFAULTS))
