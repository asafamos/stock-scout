"""Delete files from Supabase Storage AFTER verifying they're in R2.

Designed as the safety-first companion to migrate_supabase_to_r2.py.
Will refuse to delete a file unless an identical copy (same size) exists
in R2 first. This means even if the migration missed something, the
deletion won't take it down with it.

Usage
-----
1. dry-run (mandatory first step — shows what would be deleted, deletes
   nothing):
       python scripts/delete_supabase_after_r2_migration.py --dry-run

2. Live run — types DELETE <N> at the prompt to confirm:
       python scripts/delete_supabase_after_r2_migration.py

Safety
------
- Per-file HEAD check in R2 BEFORE deleting from Supabase. If R2 doesn't
  have the file at the expected size, that file is SKIPPED with a warning
  (we'd rather leave it in Supabase than have a window where it's nowhere).
- Interactive confirmation requires typing the exact phrase 'DELETE <N>'
  where N is the count of files about to be deleted. Prevents accidental
  Enter-key oopses.
- Reads its config from the SAME .env.facegreet-migration so there's no
  divergence in credentials or bucket names between migration and delete.

Reversibility
-------------
There is NO undo on Supabase Storage deletes. The whole point of running
the migration first is that R2 is the source of truth after this. To
"undo", you'd run migrate in reverse (upload R2 → Supabase) — write that
script if you ever need it; not provided here because the whole goal of
this exercise is to STOP using Supabase Storage.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("delete")


def env(name: str, required: bool = True) -> str:
    v = os.environ.get(name, "").strip()
    if required and not v:
        logger.error("Missing env var: %s", name)
        sys.exit(2)
    return v


def get_clients():
    """Return (sb_client, r2_client, supabase_bucket, r2_bucket).

    Reuses the JWT-signing logic from migrate_supabase_to_r2 so a self-
    signed service_role JWT is generated for new sb_secret_* keys.
    """
    # Reuse the migrate script's auth resolution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from migrate_supabase_to_r2 import (
        MigrationConfig,
        _resolve_supabase_auth_key,
        get_r2_client,
    )

    cfg = MigrationConfig(
        supabase_url=env("SUPABASE_URL"),
        supabase_key=env("SUPABASE_SERVICE_ROLE_KEY"),
        supabase_bucket=os.environ.get("SUPABASE_BUCKET", "").strip() or "facegreet-videos",
        r2_endpoint=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        r2_access_key=env("R2_ACCESS_KEY"),
        r2_secret_key=env("R2_SECRET_KEY"),
        r2_bucket=os.environ.get("R2_BUCKET", "").strip() or "facegreet-videos",
    )
    auth = _resolve_supabase_auth_key(cfg)

    from supabase import create_client
    sb = create_client(cfg.supabase_url, auth)

    r2 = get_r2_client(cfg)
    return sb, r2, cfg.supabase_bucket, cfg.r2_bucket


def list_supabase(sb, bucket: str) -> list[dict]:
    """Recursive listing, mirroring migrate script logic."""
    out = []
    queue = [""]
    seen = set()
    while queue:
        prefix = queue.pop()
        if prefix in seen:
            continue
        seen.add(prefix)
        try:
            items = sb.storage.from_(bucket).list(
                path=prefix or None,
                options={"limit": 10000, "offset": 0,
                         "sortBy": {"column": "name", "order": "asc"}},
            )
        except Exception as e:
            logger.error("List failed at prefix %r: %s", prefix, e)
            continue
        for it in items:
            name = it.get("name")
            if not name:
                continue
            full = f"{prefix}/{name}".lstrip("/") if prefix else name
            if it.get("metadata"):
                out.append({"key": full, "size": it["metadata"].get("size", 0)})
            else:
                queue.append(full)
        logger.info("  listed %r: %d files so far", prefix or "/", len(out))
    return out


def verify_in_r2(r2, bucket: str, key: str, expected_size: int) -> tuple[bool, str]:
    """Return (ok, reason). ok=True means safe to delete from Supabase."""
    try:
        head = r2.head_object(Bucket=bucket, Key=key)
        actual = head["ContentLength"]
        if expected_size and actual != expected_size:
            return False, f"size mismatch: sb={expected_size} r2={actual}"
        return True, "verified"
    except Exception as e:
        code = ""
        if hasattr(e, "response"):
            code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False, "not in R2"
        return False, f"r2 head error: {e}"


def delete_one(sb, sb_bucket: str, r2, r2_bucket: str, file: dict, dry_run: bool):
    key = file["key"]
    size = int(file.get("size") or 0)
    ok, reason = verify_in_r2(r2, r2_bucket, key, size)
    if not ok:
        return key, "skipped", reason
    if dry_run:
        return key, "would_delete", reason
    try:
        sb.storage.from_(sb_bucket).remove([key])
        return key, "deleted", reason
    except Exception as e:
        return key, "error", str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be deleted; delete nothing.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel HEAD+delete workers (default 8).")
    args = ap.parse_args()

    sb, r2, sb_bucket, r2_bucket = get_clients()
    logger.info("Source (Supabase): bucket=%s", sb_bucket)
    logger.info("Mirror (R2):       bucket=%s", r2_bucket)

    logger.info("Listing Supabase bucket...")
    files = list_supabase(sb, sb_bucket)
    total_bytes = sum(int(f.get("size") or 0) for f in files)
    logger.info("Found %d files, total %.1f MB", len(files), total_bytes / 1024 / 1024)
    if not files:
        logger.info("Nothing to delete. Exiting.")
        return

    if not args.dry_run:
        n = len(files)
        confirm_phrase = f"DELETE {n}"
        print()
        print("=" * 60)
        print(f"  About to delete {n} files ({total_bytes / 1024 / 1024:.1f} MB)")
        print(f"  from Supabase bucket '{sb_bucket}'.")
        print()
        print(f"  Each file will first be HEAD-checked in R2 bucket")
        print(f"  '{r2_bucket}' for a size match. Files NOT verified in R2")
        print(f"  will be SKIPPED (left in Supabase).")
        print()
        print(f"  To confirm, type exactly:  {confirm_phrase}")
        print(f"  Anything else cancels.")
        print("=" * 60)
        answer = input("> ").strip()
        if answer != confirm_phrase:
            logger.info("Confirmation phrase didn't match. Aborting.")
            sys.exit(0)
        logger.info("Confirmed. Starting deletion.")

    counters = {"deleted": 0, "would_delete": 0, "skipped": 0, "error": 0}
    bytes_freed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(delete_one, sb, sb_bucket, r2, r2_bucket, f, args.dry_run): f
            for f in files
        }
        for i, fut in enumerate(as_completed(futures), 1):
            f = futures[fut]
            try:
                key, status, reason = fut.result()
            except Exception as e:
                key, status, reason = f["key"], "error", str(e)
            counters[status] = counters.get(status, 0) + 1
            if status in ("deleted", "would_delete"):
                bytes_freed += int(f.get("size") or 0)
            if status == "skipped":
                logger.warning("[%d/%d] SKIP %s — %s", i, len(files), key, reason)
            elif status == "error":
                logger.error("[%d/%d] ERROR %s — %s", i, len(files), key, reason)
            elif i % 50 == 0 or i == len(files):
                logger.info("[%d/%d] %s %s", i, len(files), status, key)

    elapsed = time.time() - started
    logger.info("─" * 50)
    logger.info("DONE in %.1fs", elapsed)
    for k, v in counters.items():
        logger.info("  %s: %d", k, v)
    logger.info("  bytes freed: %.2f MB", bytes_freed / 1024 / 1024)


if __name__ == "__main__":
    main()
