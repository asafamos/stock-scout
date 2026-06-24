"""Diagnose env vars for non-ASCII characters (typically Hebrew copy-paste contamination).

Usage:
    python3 scripts/check_env_ascii.py
"""
import os

VARS = [
    "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_BUCKET",
    "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_ACCOUNT_ID", "R2_BUCKET",
]

print(f"{'var':35s} {'len':>5s}  status")
print("-" * 70)
for v in VARS:
    s = os.environ.get(v, "")
    bad_chars = [(i, c, hex(ord(c))) for i, c in enumerate(s) if ord(c) > 127]
    status = "CLEAN" if not bad_chars else f"CONTAMINATED ({len(bad_chars)} non-ASCII)"
    print(f"{v:35s} {len(s):>5d}  {status}")
    if bad_chars:
        for pos, ch, hx in bad_chars[:10]:
            print(f"    position {pos}: char={hx}  (showing surrounding 10 chars)")
            start = max(0, pos - 5)
            end = min(len(s), pos + 5)
            snippet = s[start:end].replace(ch, "[!!]")
            print(f"    snippet around: ...{snippet}...")
