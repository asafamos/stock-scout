from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone


class ProviderDecision:
    ALLOW = "ALLOW"
    BLOCK_PRECHECK = "BLOCK_PRECHECK"  # Blocked by preflight/no key/auth
    BLOCK_COOLDOWN = "BLOCK_COOLDOWN"  # Temporarily open circuit due to runtime failures


@dataclass
class ProviderState:
    disabled_permanent: bool = False
    cooldown_until_utc: Optional[datetime] = None
    last_error_code: Optional[int] = None
    last_reason: Optional[str] = None
    consecutive_failures: int = 0


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

    def allow(self, provider: str, capability: str) -> Tuple[bool, str, str]:
        """Return (allowed, reason, decision_enum)."""
        st = self._get(provider)
        now = datetime.now(timezone.utc)
        # Permanent block from preflight/auth
        if st.disabled_permanent:
            return False, (st.last_reason or "disabled_permanent"), ProviderDecision.BLOCK_PRECHECK
        # Cooldown block
        if st.cooldown_until_utc and now < st.cooldown_until_utc:
            return False, "cooldown", ProviderDecision.BLOCK_COOLDOWN
        return True, "", ProviderDecision.ALLOW

    def record_success(self, provider: str) -> None:
        st = self._get(provider)
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
        """Record a failure with flexible signature.

        Supports both legacy calls: record_failure(provider, http_status, reason)
        and new calls: record_failure(provider, capability=..., status_code=..., reason=...).
        """
        # Normalize args
        if status_code is not None:
            http_status = status_code
        st = self._get(provider)
        st.last_error_code = http_status
        st.last_reason = (reason or st.last_reason)
        st.consecutive_failures = max(0, st.consecutive_failures) + 1
        now = datetime.now(timezone.utc)
        # 429 → rate limit: open circuit for 120s
        if http_status == 429:
            st.cooldown_until_utc = now + timedelta(seconds=120)
            return
        # 401/403 → auth: permanent disable until restart (preflight needed)
        if http_status in (401, 403):
            st.disabled_permanent = True
            return
        # 5xx → short cooldown with backoff
        if http_status and 500 <= http_status <= 599:
            base = 30
            backoff = min(240, base * st.consecutive_failures)
            st.cooldown_until_utc = now + timedelta(seconds=backoff)

    def snapshot(self) -> Dict[str, Dict[str, Optional[str]]]:
        out: Dict[str, Dict[str, Optional[str]]] = {}
        for k, st in self._states.items():
            out[k] = {
                "disabled_permanent": st.disabled_permanent,
                "cooldown_until": st.cooldown_until_utc.isoformat() if st.cooldown_until_utc else None,
                "last_error_code": st.last_error_code,
                "last_reason": st.last_reason,
                "consecutive_failures": st.consecutive_failures,
            }
        return out


_GUARD_SINGLETON: Optional[ProviderGuard] = None


def get_provider_guard() -> ProviderGuard:
    global _GUARD_SINGLETON
    if _GUARD_SINGLETON is None:
        _GUARD_SINGLETON = ProviderGuard()
    return _GUARD_SINGLETON
