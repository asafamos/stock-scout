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
    # ML gate (added 2026-08-05 — task #146). Same low-vol MODERATE_UP
    # pattern: today's top-10 candidates had ML prob 0.31-0.37, all below
    # 0.4 floor. Adaptive covers this like RR — after N=5 dry cycles blocked
    # by ML, relax floor 0.40 → 0.35. Threshold is HIGHER (5 vs 3 for RR)
    # because ML confidence is a stronger signal to respect. DEFAULT OFF —
    # requires TRADE_ADAPTIVE_ML_ENABLED=1 env to activate.
    "ml_dry_streak": 0,
    "ml_relaxed_active": False,
    "ml_relaxed_since": "",
    # Score gate (added 2026-08-28 — task #148). SIDEWAYS-regime pattern:
    # Aug 25-27 saw 3-day zero-buy streak because regime_score_floor pushed
    # effective floor to 75+ in SIDEWAYS. 42K sim shows SIDEWAYS score <73
    # = +3.90% mean (n=140) — a real signal we were blocking. Adaptive
    # covers this like RR/ML — after N=3 dry cycles blocked by score,
    # relax floor by 5pt (73 → 68) for next cycle. DEFAULT ON.
    "score_dry_streak": 0,
    "score_relaxed_active": False,
    "score_relaxed_since": "",
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


def get_adaptive_ml_relaxed() -> bool:
    """True if the ML gate should be relaxed to 0.35 (from 0.40) right now.

    Called from order_manager's ML filter, policy.evaluate_static_gates,
    and risk_manager.can_open_position (defense-in-depth trio). Reads
    state only — no side effects. Activated after N consecutive cycles
    blocked by ML filter. GATED BY TRADE_ADAPTIVE_ML_ENABLED env: if
    that's not set to 1, always returns False (recorder still tracks
    streaks for observability, but the relax flag stays inert). Task #146.
    """
    import os as _os
    if _os.getenv("TRADE_ADAPTIVE_ML_ENABLED", "0").strip() not in ("1", "true", "True", "yes", "YES"):
        return False
    return _load().get("ml_relaxed_active", False)


def get_adaptive_score_relaxed() -> bool:
    """True if the score gate should be relaxed by 5pt right now.

    Called from policy.regime_score_floor() at every gate evaluation.
    Reads state — no side effects. Activated after N consecutive cycles
    blocked by score filter. See task #148 (SIDEWAYS-regime zero-buy Aug 25-27).
    """
    return _load().get("score_relaxed_active", False)


def get_adaptive_score_relax_amount() -> float:
    """How many points to drop the score floor when relaxed. Default 5."""
    import os as _os_s
    try:
        return float(_os_s.getenv("ADAPTIVE_SCORE_RELAX_POINTS", "5"))
    except (ValueError, TypeError):
        return 5.0


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
    ml_dropped: bool = False,
    ml_threshold: int = 5,
    score_dropped: bool = False,
    score_threshold: int = 3,
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

    old_ml_relaxed = bool(state.get("ml_relaxed_active", False))
    old_ml_streak = int(state.get("ml_dry_streak", 0))

    old_score_relaxed = bool(state.get("score_relaxed_active", False))
    old_score_streak = int(state.get("score_dry_streak", 0))

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
        if old_ml_streak > 0 or old_ml_relaxed:
            state["ml_dry_streak"] = 0
            state["ml_relaxed_active"] = False
            changes.append(f"ML streak was {old_ml_streak}, relax was {'ON' if old_ml_relaxed else 'OFF'}")
        if old_score_streak > 0 or old_score_relaxed:
            state["score_dry_streak"] = 0
            state["score_relaxed_active"] = False
            changes.append(f"Score streak was {old_score_streak}, relax was {'ON' if old_score_relaxed else 'OFF'}")
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
    if not confidence_dropped and not analyst_pt_dropped and not rr_dropped and not ml_dropped and not score_dropped:
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

    # ── ML dry-streak (independent of other gates) ──
    # Task #146 — 2026-08-05: ML gate blocked all 10 top-scoring candidates
    # today (ML 0.31-0.37, floor 0.40). Adaptive extension covers it
    # symmetrically. IMPORTANT: activation is gated by env
    # TRADE_ADAPTIVE_ML_ENABLED=1 in the READER (get_adaptive_ml_relaxed).
    # Streak tracking runs unconditionally for observability; the actual
    # gate relaxation is opt-in.
    ml_msg = None
    if ml_dropped:
        new_ml_streak = old_ml_streak + 1
        state["ml_dry_streak"] = new_ml_streak
        if new_ml_streak >= ml_threshold and not old_ml_relaxed:
            state["ml_relaxed_active"] = True
            state["ml_relaxed_since"] = datetime.now(timezone.utc).isoformat()
            import os as _os_ml
            _ml_env_on = _os_ml.getenv("TRADE_ADAPTIVE_ML_ENABLED", "0").strip() in ("1", "true", "True", "yes", "YES")
            ml_msg = (
                f"🔓 Adaptive gate ACTIVATED (ML)\n"
                f"{new_ml_streak} consecutive dry cycles blocked by ML filter.\n"
                f"{'✅ Relaxing min_ml_prob from 0.40 → 0.35 for next cycle.' if _ml_env_on else '⚠️ OBSERVED ONLY — env TRADE_ADAPTIVE_ML_ENABLED=0 (opt-in). Set to 1 to activate relax.'}\n"
                f"Will reset on any successful buy. "
                f"Kill: TRADE_ADAPTIVE_GATES_ENABLED=0."
            )
        elif new_ml_streak == ml_threshold - 1 and not old_ml_relaxed:
            ml_msg = (
                f"⏳ Adaptive gate PRE-WARN (ML)\n"
                f"{new_ml_streak}/{ml_threshold} cycles blocked by ML filter.\n"
                f"Next dry cycle → ACTIVATE (relax only if TRADE_ADAPTIVE_ML_ENABLED=1)."
            )

    # ── Score dry-streak (independent of other gates) ──
    # Task #148 — 2026-08-28: after SIDEWAYS 3-day zero-buy streak. Score
    # filter blocked every candidate (regime_score_floor pushed effective
    # floor to 75). 42K sim: SIDEWAYS score <73 = +3.90% mean (n=140), so
    # we were blocking real signal. Adaptive relax drops the floor by
    # ADAPTIVE_SCORE_RELAX_POINTS (default 5) after N=3 dry cycles.
    # Applied in policy.regime_score_floor().
    score_msg = None
    if score_dropped:
        new_score_streak = old_score_streak + 1
        state["score_dry_streak"] = new_score_streak
        if new_score_streak >= score_threshold and not old_score_relaxed:
            state["score_relaxed_active"] = True
            state["score_relaxed_since"] = datetime.now(timezone.utc).isoformat()
            score_msg = (
                f"🔓 Adaptive gate ACTIVATED (Score)\n"
                f"{new_score_streak} consecutive dry cycles blocked by score filter.\n"
                f"Relaxing effective score floor by 5pt for next cycle "
                f"(e.g. 73 → 68). Will reset on any successful buy. "
                f"Kill: TRADE_ADAPTIVE_GATES_ENABLED=0."
            )
        elif new_score_streak == score_threshold - 1 and not old_score_relaxed:
            score_msg = (
                f"⏳ Adaptive gate PRE-WARN (Score)\n"
                f"{new_score_streak}/{score_threshold} cycles blocked by score filter.\n"
                f"Next dry cycle → auto-relax score floor by 5pt."
            )

    # Persist state changes (streaks and relax flags) regardless of whether
    # a message was generated — this cycle happened and must be recorded.
    _save(state)

    # Combine messages (any combination could move in the same cycle).
    parts = [m for m in [msg, pt_msg, rr_msg, ml_msg, score_msg] if m]
    combined = "\n\n".join(parts) if parts else None
    return combined


def force_reset() -> None:
    """Manual reset — for CLI/debug. Wipes state to defaults."""
    _save(dict(_DEFAULTS))
