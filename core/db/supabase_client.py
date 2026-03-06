"""Supabase client wrapper for Stock Scout portfolio persistence.

Reads connection details from Streamlit secrets (preferred) or environment
variables.  Returns ``None`` when Supabase is not configured so callers can
fall back to the local DuckDB backend.

Required secrets / env vars::

    SUPABASE_URL  — e.g. https://xyzxyz.supabase.co
    SUPABASE_KEY  — anon (public) key from Supabase dashboard

"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("stock_scout.db.supabase")

_CLIENT_CACHE: Optional[object] = None


def _read_secret(key: str) -> Optional[str]:
    """Try Streamlit secrets first, then env vars."""
    # 1) Streamlit secrets (available on Streamlit Cloud and local with secrets.toml)
    try:
        import streamlit as st

        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass

    # 2) Environment variable
    return os.environ.get(key)


def get_supabase_client():
    """Return a ``supabase.Client`` if credentials are configured, else ``None``.

    The client is cached after the first successful creation.
    """
    global _CLIENT_CACHE
    if _CLIENT_CACHE is not None:
        return _CLIENT_CACHE

    url = _read_secret("SUPABASE_URL")
    key = _read_secret("SUPABASE_KEY")

    if not url or not key:
        logger.debug("Supabase not configured — falling back to DuckDB")
        return None

    try:
        from supabase import create_client

        client = create_client(url, key)
        _CLIENT_CACHE = client
        logger.info("Supabase client connected (%s)", url[:30])
        return client
    except ImportError:
        logger.warning("supabase-py not installed — pip install supabase")
        return None
    except Exception as exc:
        logger.error("Supabase connection failed: %s", exc)
        return None


def reset_supabase_client() -> None:
    """Clear cached client (useful for tests)."""
    global _CLIENT_CACHE
    _CLIENT_CACHE = None
