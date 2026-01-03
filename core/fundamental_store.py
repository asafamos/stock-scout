"""
Point-in-time fundamentals store (SQLite).

Purpose:
- LIVE mode: after a successful fundamentals fetch/aggregation, call
  save_fundamentals_snapshot() to persist a point-in-time (as_of_date) snapshot.
- BACKTEST mode: call load_fundamentals_as_of() to retrieve the latest snapshot
  on or before a given date, avoiding live API calls and reducing lookahead risk.

Storage:
- SQLite DB at data/fundamentals_store.sqlite (created if missing)
- Table fundamentals_snapshots(ticker, as_of_date, provider, payload_json)
- Unique index on (ticker, as_of_date, provider) to enable UPSERT semantics.

Notes:
- The payload is expected to be a JSON-serializable dict (e.g., the merged result
  from core.data_sources_v2.fetch_multi_source_data). Non-serializable values are
  stringified via json.dumps(..., default=str).
- To ensure uniqueness when provider is omitted, this module stores provider=None
  as an empty string "" when saving and will match the same convention when loading.
"""
from __future__ import annotations

import os
import json
import sqlite3
from pathlib import Path
from datetime import date
from typing import Mapping, Any, Optional, Dict


def get_fundamentals_db_path() -> str:
    """Return the path to the fundamentals SQLite DB (default: data/fundamentals_store.sqlite)."""
    return str(Path("data") / "fundamentals_store.sqlite")


def init_fundamentals_store() -> None:
    """Create the DB and table if they do not exist (idempotent)."""
    db_path = Path(get_fundamentals_db_path())
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fundamentals_snapshots (
                ticker TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                provider TEXT,
                payload_json TEXT NOT NULL
            );
            """
        )
        # Unique index to enable UPSERT on (ticker, as_of_date, provider)
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_fundamentals_snapshots_unique
            ON fundamentals_snapshots(ticker, as_of_date, provider);
            """
        )
        conn.commit()


def _normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider for storage: None -> empty string for uniqueness consistency."""
    return provider if provider is not None else ""


def save_fundamentals_snapshot(
    ticker: str,
    payload: Mapping[str, Any],
    as_of_date: date,
    provider: str | None = None,
) -> None:
    """
    Persist a point-in-time snapshot of fundamentals for (ticker, as_of_date, provider).

    - `payload` is the already-aggregated fundamentals dict (e.g., the merged result from data_sources_v2).
    - Implementation detail: JSON-encode the payload into `payload_json`.
    """
    init_fundamentals_store()  # Ensure DB exists
    db_path = get_fundamentals_db_path()

    payload_json = json.dumps(dict(payload), ensure_ascii=False, default=str)
    provider_norm = _normalize_provider(provider)
    as_of_str = as_of_date.strftime("%Y-%m-%d")

    # UPSERT by unique key (ticker, as_of_date, provider)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO fundamentals_snapshots (ticker, as_of_date, provider, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker, as_of_date, provider) DO UPDATE SET
                payload_json=excluded.payload_json
            ;
            """,
            (ticker.upper(), as_of_str, provider_norm, payload_json),
        )
        conn.commit()


def load_fundamentals_as_of(
    ticker: str,
    as_of_date: date,
    provider: str | None = None,
) -> Optional[Dict[str, Any]]:
    """
    Return the latest fundamentals snapshot for `ticker` whose `as_of_date` <= the given date.

    - If `provider` is not None, restrict to that provider.
    - If `provider` is None, use any provider (latest snapshot regardless of provider).
    - Return None if no snapshot exists.
    - Decode `payload_json` back into a dict before returning.
    """
    db_path = get_fundamentals_db_path()
    if not Path(db_path).exists():
        return None

    as_of_str = as_of_date.strftime("%Y-%m-%d")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if provider is not None:
            provider_norm = _normalize_provider(provider)
            cur = conn.execute(
                """
                SELECT payload_json FROM fundamentals_snapshots
                WHERE ticker = ? AND provider = ? AND as_of_date <= ?
                ORDER BY as_of_date DESC
                LIMIT 1;
                """,
                (ticker.upper(), provider_norm, as_of_str),
            )
        else:
            cur = conn.execute(
                """
                SELECT payload_json FROM fundamentals_snapshots
                WHERE ticker = ? AND as_of_date <= ?
                ORDER BY as_of_date DESC
                LIMIT 1;
                """,
                (ticker.upper(), as_of_str),
            )
        row = cur.fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row["payload_json"])  # type: ignore[index]
        except Exception:
            payload = {}
        return payload
