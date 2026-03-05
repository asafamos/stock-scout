"""User identity helper for Stock Scout multi-user support.

On Streamlit Cloud with Google SSO configured, ``st.user``
returns the authenticated user's email.  Locally the helper falls back to
the ``STOCK_SCOUT_USER`` environment variable or ``"local"``.

The resolved identity is persisted to a small file
(``data/.user_identity``) so that transient SSO failures (timeouts,
network hiccups) during Streamlit reruns don't silently change the
active ``user_id`` — which would make portfolio data "disappear".

Usage::

    from core.auth import get_current_user

    user = get_current_user()
    user["email"]        # "alice@gmail.com"
    user["user_id"]      # "alice_at_gmail_com"  (safe for paths / DB keys)
    user["display_name"] # "alice"
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict

import streamlit as st

logger = logging.getLogger("stock_scout.auth")

_IDENTITY_FILE = Path(__file__).resolve().parent.parent / "data" / ".user_identity"


def _sanitize_for_path(email: str) -> str:
    """Convert an email address to a filesystem-safe identifier.

    >>> _sanitize_for_path("alice@gmail.com")
    'alice_at_gmail_com'
    """
    s = email.lower().strip()
    s = s.replace("@", "_at_")
    s = re.sub(r"[^a-z0-9_]", "_", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _save_identity(result: Dict[str, str]) -> None:
    """Persist identity to disk so it survives session_state resets."""
    try:
        _IDENTITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _IDENTITY_FILE.write_text(json.dumps(result))
    except Exception as exc:
        logger.debug("Could not save identity file: %s", exc)


def _load_identity() -> Dict[str, str] | None:
    """Load previously persisted identity from disk."""
    try:
        if _IDENTITY_FILE.exists():
            data = json.loads(_IDENTITY_FILE.read_text())
            if isinstance(data, dict) and data.get("user_id"):
                return data
    except Exception as exc:
        logger.debug("Could not read identity file: %s", exc)
    return None


def get_current_user() -> Dict[str, str]:
    """Return the current authenticated user as a dict.

    Returns:
        {
            "email":        raw email (or "local"),
            "user_id":      filesystem-safe identifier,
            "display_name": short human-readable name,
        }

    Resolution order:
      1. ``st.session_state`` cache (fast path, same rerun)
      2. Streamlit Cloud SSO (``st.user.email``)
      3. Persisted identity file (``data/.user_identity``)
      4. ``STOCK_SCOUT_USER`` env-var / ``"local"`` fallback

    When SSO succeeds, the identity is persisted to disk so that
    subsequent reruns where SSO temporarily fails will still use the
    correct user_id (and thus see the correct portfolio data).
    """
    if "_current_user" in st.session_state:
        return st.session_state["_current_user"]

    email: str | None = None

    # 1. Try Streamlit Cloud built-in auth (Google SSO)
    try:
        user_obj = st.user
        email = getattr(user_obj, "email", None)
        if email and isinstance(email, str) and "@" in email:
            logger.debug("SSO resolved email: %s", email)
        else:
            email = None
    except Exception:
        logger.debug("SSO lookup failed, will try fallbacks")

    # 2. Fallback: persisted identity file (survives session_state resets)
    if not email:
        saved = _load_identity()
        if saved and "@" in saved.get("email", ""):
            logger.info("Using persisted identity: %s (SSO unavailable this rerun)", saved["email"])
            st.session_state["_current_user"] = saved
            return saved

    # 3. Fallback: environment variable (local dev)
    if not email:
        email = os.getenv("STOCK_SCOUT_USER", "local")

    user_id = _sanitize_for_path(email)

    if "@" in email:
        display_name = email.split("@")[0]
    else:
        display_name = email

    result = {
        "email": email,
        "user_id": user_id,
        "display_name": display_name,
    }
    st.session_state["_current_user"] = result

    # Persist to disk when we got a real SSO email
    if "@" in email:
        _save_identity(result)

    return result


def is_authenticated() -> bool:
    """Return True if the user authenticated via SSO (not local fallback)."""
    user = get_current_user()
    return "@" in user["email"]
