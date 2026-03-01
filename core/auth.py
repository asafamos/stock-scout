"""User identity helper for Stock Scout multi-user support.

On Streamlit Cloud with Google SSO configured, ``st.experimental_user``
returns the authenticated user's email.  Locally the helper falls back to
the ``STOCK_SCOUT_USER`` environment variable or ``"local"``.

Usage::

    from core.auth import get_current_user

    user = get_current_user()
    user["email"]        # "alice@gmail.com"
    user["user_id"]      # "alice_at_gmail_com"  (safe for paths / DB keys)
    user["display_name"] # "alice"
"""

from __future__ import annotations

import os
import re
from typing import Dict

import streamlit as st


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


def get_current_user() -> Dict[str, str]:
    """Return the current authenticated user as a dict.

    Returns:
        {
            "email":        raw email (or "local"),
            "user_id":      filesystem-safe identifier,
            "display_name": short human-readable name,
        }

    The result is cached in ``st.session_state["_current_user"]`` so it is
    computed once per Streamlit session.
    """
    if "_current_user" in st.session_state:
        return st.session_state["_current_user"]

    email: str | None = None

    # 1. Try Streamlit Cloud built-in auth (Google SSO)
    try:
        user_obj = st.experimental_user
        email = getattr(user_obj, "email", None)
        if email and isinstance(email, str) and "@" in email:
            pass  # Valid email from SSO
        else:
            email = None
    except Exception:
        pass

    # 2. Fallback: environment variable (local dev)
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
    return result


def is_authenticated() -> bool:
    """Return True if the user authenticated via SSO (not local fallback)."""
    user = get_current_user()
    return "@" in user["email"]
