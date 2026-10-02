"""One-off: delete the test rows from Supabase scan_history (user_id 'default').

Only rows whose universe_name is 'test'/'live_test' and universe_size <= 3 are
touched (assert below aborts otherwise). Their child scan_recommendations rows
are deleted first. Everything is backed up to data/backups/ before deletion.

Run from the repo root:  .venv/bin/python scripts/maintenance_clean_test_rows.py
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
c = create_client(os.environ["SUPABASE_URL"],
                  os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"])

parents = c.table("scan_history").select("*").eq("user_id", "default").execute().data
assert all(p["universe_name"] in ("test", "live_test") and (p["universe_size"] or 0) <= 3
           for p in parents), "unexpected non-test row under user_id 'default' — aborting"
ids = [p["scan_id"] for p in parents]

kids = []
for i in range(0, len(ids), 30):
    kids += c.table("scan_recommendations").select("*").in_("scan_id", ids[i:i + 30]).execute().data

backup = ROOT / "data" / "backups" / "supabase_test_rows_2026-10-02.json"
backup.parent.mkdir(parents=True, exist_ok=True)
backup.write_text(json.dumps({"scan_history": parents, "scan_recommendations": kids}, default=str))
print(f"backed up {len(parents)} scans + {len(kids)} recommendations -> {backup}")

for i in range(0, len(ids), 30):
    c.table("scan_recommendations").delete().in_("scan_id", ids[i:i + 30]).execute()
for i in range(0, len(ids), 30):
    c.table("scan_history").delete().in_("scan_id", ids[i:i + 30]).execute()

left = c.table("scan_history").select("*", count="exact").eq("user_id", "default").limit(1).execute().count
owner = c.table("scan_history").select("*", count="exact").eq("user_id", "stockscout_owner").limit(1).execute().count
print(f"'default' rows left: {left} (expect 0) | owner rows untouched: {owner}")
