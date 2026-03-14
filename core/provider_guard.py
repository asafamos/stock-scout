from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class ProviderDecision:
    ALLOW = "ALLOW"
    BLOCK_PRECHECK = "BLOCK_PRECHECK"  # Blocked by preflight/no key/auth
    BLOCK_COOLDOWN = "BLOCK_COOLDOWN"  # Temporarily open circuit due to runtime failures
    HALF_OPEN = "HALF_OPEN"  # Allowing one probe request to test recovery


class CircuitState:
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, skip requests
    HALF_OPEN = "HALF_OPEN"  # Try one request to test recovery


@dataclass
class ProviderState:
    disabled_permanent: bool = False
    cooldown_until_utc: Optional[datetime] = None
    last_error_code: Optional[int] = None
    last_reason: Optional[str] = None
    consecutive_failures: int = 0
    circuit_state: str = CircuitState.CLOSED
    last_success_utc: Optional[datetime] = None
    total_failures: int = 0
    total_successes: int = 0

    @property
    def failure_rate(self) -> float:
        total = self.total_failures + self.total_successes
        if total == 0:
            return 0.0
        return self.total_failures / total


# Configurable thresholds
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 3,       # consecutive failures before opening circuit
    "cooldown_base_seconds": 30,  # base cooldown for 5xx errors
    "cooldown_max_seconds": 300,  # max cooldown (5 minutes)
    "rate_limit_cooldown": 120,   # cooldown for 429 rate limits
    "half_open_after_seconds": 60,  # try recovery after this many seconds in OPEN state
    "half_open_success_to_close": 1,  # successes needed in HALF_OPEN to close circuit
}


class ProviderGuard:
    def __init__(self) -> None:
        self._states: Dict[str, ProviderState] = {}

    def _get(self, provider: str) -> ProviderState:
        key = str(provider).upper()
        st = self._states.get(key)
        if not st:
            st = ProviderState()
            self._states[key] = st
        return st

    def update_from_preflight(self, provider_status: Dict) -> None:
        """Ingest preflight status to mark permanent disables (no_key/auth_error/disabled)."""
        try:
            for key, meta in (provider_status or {}).items():
                st = self._get(key)
                if isinstance(meta, dict):
                    status = str(meta.get("status", "")).lower()
                    if status in ("no_key", "auth_error", "disabled"):
                        st.disabled_permanent = True
                        st.last_reason = status
                elif isinstance(meta, bool):
                    if not meta:
                        st.disabled_permanent = True
                        st.last_reason = "disabled"
        except Exception:
            pass

    def allow(self, provider: str, capability: str = "") -> Tuple[bool, str, str]:
        """Return (allowed, reason, decision_enum)."""
        st = self._get(provider)
        now = datetime.now(timezone.utc)

        # Permanent block from preflight/auth
        if st.disabled_permanent:
            return False, (st.last_reason or "disabled_permanent"), ProviderDecision.BLOCK_PRECHECK

        # Cooldown block — check for half-open transition
        if st.cooldown_until_utc and st.circuit_state == CircuitState.OPEN:
            if now >= st.cooldown_until_utc:
                # Cooldown expired — transition to HALF_OPEN, allow one probe
                st.circuit_state = CircuitState.HALF_OPEN
                st.cooldown_until_utc = None
                logger.info("Provider %s: circuit HALF_OPEN — allowing probe request", provider)
                return True, "half_open_probe", ProviderDecision.HALF_OPEN
            else:
                return False, "cooldown", ProviderDecision.BLOCK_COOLDOWN

        # HALF_OPEN: allow requests (one at a time effectively)
        if st.circuit_state == CircuitState.HALF_OPEN:
            return True, "half_open_probe", ProviderDecision.HALF_OPEN

        return True, "", ProviderDecision.ALLOW

    def record_success(self, provider: str) -> None:
        st = self._get(provider)
        st.total_successes += 1
        st.last_success_utc = datetime.now(timezone.utc)

        if st.circuit_state == CircuitState.HALF_OPEN:
            # Success in half-open → close circuit (recovered)
            st.circuit_state = CircuitState.CLOSED
            st.consecutive_failures = 0
            st.last_error_code = None
            st.last_reason = None
            st.cooldown_until_utc = None
            logger.info("Provider %s: circuit CLOSED — recovered", provider)
        else:
            st.consecutive_failures = 0
            st.last_error_code = None
            st.last_reason = None
            st.cooldown_until_utc = None

    def record_failure(
        self,
        provider: str,
        http_status: Optional[int] = None,
        reason: Optional[str] = None,
        capability: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        """Record a failure with flexible signature."""
        # Normalize args
        if status_code is not None:
            http_status = status_code
        st = self._get(provider)
        st.last_error_code = http_status
        st.last_reason = (reason or st.last_reason)
        st.consecutive_failures = max(0, st.consecutive_failures) + 1
        st.total_failures += 1
        now = datetime.now(timezone.utc)
        cfg = CIRCUIT_BREAKER_CONFIG

        # HALF_OPEN failure → revert to OPEN with longer cooldown
        if st.circuit_state == CircuitState.HALF_OPEN:
            st.circuit_state = CircuitState.OPEN
            backoff = min(cfg["cooldown_max_seconds"], cfg["cooldown_base_seconds"] * st.consecutive_failures)
            st.cooldown_until_utc = now + timedelta(seconds=backoff)
            logger.warning("Provider %s: HALF_OPEN probe failed — circuit re-OPEN for %ds", provider, backoff)
            return

        # 429 → rate limit: open circuit
        if http_status == 429:
            st.circuit_state = CircuitState.OPEN
            st.cooldown_until_utc = now + timedelta(seconds=cfg["rate_limit_cooldown"])
            logger.warning("Provider %s: rate limited — circuit OPEN for %ds", provider, cfg["rate_limit_cooldown"])
            return

        # 401/403 → auth: permanent disable
        if http_status in (401, 403):
            st.disabled_permanent = True
            logger.warning("Provider %s: auth failure (%d) — permanently disabled", provider, http_status)
            return

        # 5xx or generic failure → cooldown with backoff, open circuit after threshold
        if st.consecutive_failures >= cfg["failure_threshold"]:
            st.circuit_state = CircuitState.OPEN
            backoff = min(cfg["cooldown_max_seconds"], cfg["cooldown_base_seconds"] * st.consecutive_failures)
            st.cooldown_until_utc = now + timedelta(seconds=backoff)
            logger.warning(
                "Provider %s: %d consecutive failures — circuit OPEN for %ds",
                provider, st.consecutive_failures, backoff,
            )
        elif http_status and 500 <= http_status <= 599:
            backoff = min(cfg["cooldown_max_seconds"], cfg["cooldown_base_seconds"] * st.consecutive_failures)
            st.cooldown_until_utc = now + timedelta(seconds=backoff)

    def snapshot(self) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for k, st in self._states.items():
            out[k] = {
                "disabled_permanent": st.disabled_permanent,
                "circuit_state": st.circuit_state,
                "cooldown_until": st.cooldown_until_utc.isoformat() if st.cooldown_until_utc else None,
                "last_error_code": st.last_error_code,
                "last_reason": st.last_reason,
                "consecutive_failures": st.consecutive_failures,
                "total_failures": st.total_failures,
                "total_successes": st.total_successes,
                "failure_rate": round(st.failure_rate, 3),
                "last_success": st.last_success_utc.isoformat() if st.last_success_utc else None,
            }
        return out

    def reset(self) -> None:
        """Clear all provider states for testing purposes."""
        self._states.clear()


_GUARD_SINGLETON: Optional[ProviderGuard] = None


def get_provider_guard() -> ProviderGuard:
    global _GUARD_SINGLETON
    if _GUARD_SINGLETON is None:
        _GUARD_SINGLETON = ProviderGuard()
    return _GUARD_SINGLETON
