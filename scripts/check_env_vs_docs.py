"""Drift detector: verify .env.trading matches the frozen values documented
in CLAUDE.md. Fires a Telegram alert on divergence. Designed to run daily
via systemd timer.

Root cause it prevents: 2026-09-14 investigation found `TRADE_MIN_ML_PROB`,
`TRADE_MAX_ML_PROB`, `TRADE_MAX_RR`, `TRADE_RATCHET_T0_GAIN`, and
`TRADE_BLOCKED_SECTORS` had ALL silently drifted from the documented
freeze state — no commit trail, no memory writeup. Four of six post-freeze
losing buys passed only because of the drift. See
[[project_deep_investigation_sep14]].

Exit codes:
    0 = no drift
    2 = drift detected + alert sent (workflow-friendly warning)
    1 = error reading env / config

To update the expected values, edit `EXPECTED` below AND `CLAUDE.md`
together — they are meant to move in lockstep.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


# Frozen-state values documented in CLAUDE.md. Edit these ONLY together with
# a matching CLAUDE.md update AND a memory writeup — never silently.
EXPECTED = {
    "TRADE_MIN_SCORE":         "73.0",
    "TRADE_MAX_SCORE":         "85",
    "TRADE_MIN_FUNDAMENTAL_SCORE": "45",  # canonical: "min_fundamental_score" in config
    "TRADE_MIN_ML_PROB":       "0.40",
    "TRADE_MAX_ML_PROB":       "0.60",
    "TRADE_MIN_RR":            "2.5",
    "TRADE_MAX_RR":            "5.0",
    "TRADE_MIN_ATR_PCT":       "0.03",
    "TRADE_MAX_OPEN_POSITIONS":"3",
    "TRADE_MAX_DAILY_BUYS":    "3",
    "TRADE_MAX_POSITION_SIZE": "450",
    "TRADE_RATCHET_T0_GAIN":   "10.0",
    "TRADE_BREAK_EVEN_ENABLED":"0",
    "TRADE_DAY_N_KILL_ENABLED":"0",   # currently disabled per 2026-08-13 memory
    # Sectors: order-independent set compare
    "TRADE_BLOCKED_SECTORS":   {
        "Consumer Defensive", "Utilities", "Communication",
        "Materials", "Basic Materials", "Real Estate",
    },
}

# Values that are optional (present in some envs, absent in others). If
# present they must match; if absent they are OK.
OPTIONAL = {"TRADE_MIN_SCORE", "TRADE_MIN_FUNDAMENTAL_SCORE", "TRADE_MIN_RR", "TRADE_MIN_ATR_PCT"}


def _load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"')
    return env


def _norm_sectors(raw: str) -> set:
    return {s.strip() for s in raw.split(",") if s.strip()}


def main() -> int:
    root = Path("/home/stockscout/stock-scout-2") if Path("/home/stockscout").exists() \
           else Path(__file__).resolve().parent.parent
    env_path = root / ".env.trading"
    env = _load_env(env_path)

    if not env:
        print(f"ERROR: could not read {env_path}", file=sys.stderr)
        return 1

    drifts = []
    for key, expected in EXPECTED.items():
        actual = env.get(key)
        if actual is None:
            if key in OPTIONAL:
                continue
            drifts.append(f"{key}: MISSING (expected {expected!r})")
            continue
        # Set-compare for sector list
        if isinstance(expected, set):
            actual_set = _norm_sectors(actual)
            missing = expected - actual_set
            extra   = actual_set - expected
            if missing or extra:
                parts = []
                if missing: parts.append(f"missing {sorted(missing)}")
                if extra:   parts.append(f"extra {sorted(extra)}")
                drifts.append(f"{key}: {'; '.join(parts)}")
            continue
        # Scalar compare — normalize numeric formatting where possible
        if str(actual) == str(expected):
            continue
        try:
            if float(actual) == float(expected):
                continue
        except (TypeError, ValueError):
            pass
        drifts.append(f"{key}: {actual!r} != expected {expected!r}")

    if not drifts:
        print("✅ env matches CLAUDE.md frozen state — no drift")
        return 0

    print("🚨 ENV DRIFT DETECTED:")
    for d in drifts:
        print(f"  • {d}")

    # Telegram alert
    try:
        # Ensure we can import from repo root
        sys.path.insert(0, str(root))
        from core.trading.notifications import _send
        msg = (
            "🚨 <b>ENV DRIFT — .env.trading != CLAUDE.md</b>\n"
            f"Host: <code>{env_path}</code>\n\n"
            + "\n".join(f"• {d}" for d in drifts)
            + "\n\nSee [[deep-investigation-sep14]] for the pattern. "
            "Fix in .env.trading OR update CLAUDE.md+EXPECTED in this script "
            "AND write a memory entry."
        )
        _send(msg)
    except Exception as e:
        print(f"warning: telegram alert failed: {e}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
