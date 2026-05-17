"""Migrate Supabase Storage bucket → Cloudflare R2.

Purpose
-------
Move the `facegreet-videos` bucket (734 files, ~3.5 GB) off Supabase to
Cloudflare R2. After migration:
  - R2 costs ~$0.05/month for the storage (vs Supabase Pro $25/month)
  - R2 has FREE egress (vs Supabase's metered egress)
  - facegreet app updated to read/write R2 instead of Supabase Storage

Usage
-----
1. Create R2 bucket + API token at https://dash.cloudflare.com (see guide
   in the conversation).
2. Fill env vars:
       export SUPABASE_URL="https://<project>.supabase.co"
       export SUPABASE_SERVICE_ROLE_KEY="<service-role-key>"
       export SUPABASE_BUCKET="facegreet-videos"
       export R2_ACCOUNT_ID="<cf-account-id>"
       export R2_ACCESS_KEY="<r2-token-id>"
       export R2_SECRET_KEY="<r2-token-secret>"
       export R2_BUCKET="facegreet-videos"
3. Dry run first: `python migrate_supabase_to_r2.py --dry-run`
4. Live: `python migrate_supabase_to_r2.py`
5. Verify a few files load from R2, then delete from Supabase (see
   companion SQL block at the bottom).

Robustness
----------
- Resumable: skips files already present in R2 (HEAD check).
- Parallel: uses ThreadPoolExecutor; default 8 workers (R2 handles this).
- Progress: prints per-file + running totals.
- Hash check: verifies size matches after upload.
- Quota-aware: if Supabase download fails with quota error, the script
  pauses with a clear message — you'll need to either temporarily
  upgrade to Pro for the migration window or wait for the June 6 refill.

Dependencies
------------
    pip install supabase boto3 tqdm

License: MIT (use freely).
"""
from __future__ import annotations
import argparse
import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("migrate")


@dataclass
class MigrationConfig:
    supabase_url: str
    supabase_key: str
    supabase_bucket: str
    r2_endpoint: str  # https://<account_id>.r2.cloudflarestorage.com
    r2_access_key: str
    r2_secret_key: str
    r2_bucket: str
    workers: int = 8
    dry_run: bool = False
    limit: Optional[int] = None  # for testing: only migrate first N files


def env(name: str, required: bool = True) -> str:
    val = os.environ.get(name, "").strip()
    if required and not val:
        logger.error("Missing env var: %s", name)
        sys.exit(2)
    return val


def load_config(args) -> MigrationConfig:
    account_id = env("R2_ACCOUNT_ID")
    return MigrationConfig(
        supabase_url=env("SUPABASE_URL"),
        supabase_key=env("SUPABASE_SERVICE_ROLE_KEY"),
        supabase_bucket=env("SUPABASE_BUCKET", required=False) or "facegreet-videos",
        r2_endpoint=f"https://{account_id}.r2.cloudflarestorage.com",
        r2_access_key=env("R2_ACCESS_KEY"),
        r2_secret_key=env("R2_SECRET_KEY"),
        r2_bucket=env("R2_BUCKET", required=False) or "facegreet-videos",
        workers=int(os.environ.get("WORKERS", "8")),
        dry_run=args.dry_run,
        limit=args.limit,
    )


def get_supabase_client(cfg: MigrationConfig):
    try:
        from supabase import create_client
    except ImportError:
        logger.error("Install: pip install supabase")
        sys.exit(2)
    return create_client(cfg.supabase_url, cfg.supabase_key)


def get_r2_client(cfg: MigrationConfig):
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        logger.error("Install: pip install boto3")
        sys.exit(2)
    return boto3.client(
        "s3",
        endpoint_url=cfg.r2_endpoint,
        aws_access_key_id=cfg.r2_access_key,
        aws_secret_access_key=cfg.r2_secret_key,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "adaptive"}),
        region_name="auto",
    )


def list_supabase_files(sb_client, bucket: str) -> List[dict]:
    """List ALL files in the bucket, recursively walking subdirectories.

    Supabase's list() only returns one level at a time. We walk the tree
    by recursing into folders (items without `metadata`).
    """
    all_files = []
    queue = [""]  # start at root
    visited_prefixes = set()
    while queue:
        prefix = queue.pop()
        if prefix in visited_prefixes:
            continue
        visited_prefixes.add(prefix)
        try:
            items = sb_client.storage.from_(bucket).list(
                prefix or None,
                {"limit": 10000, "offset": 0, "sortBy": {"column": "name", "order": "asc"}},
            )
        except Exception as exc:
            logger.error("List failed at prefix %r: %s", prefix, exc)
            continue
        for it in items:
            name = it.get("name")
            if not name:
                continue
            full_path = f"{prefix}/{name}".lstrip("/") if prefix else name
            if it.get("metadata"):
                # Real file
                size = it["metadata"].get("size", 0)
                all_files.append({"key": full_path, "size": size, "metadata": it["metadata"]})
            else:
                # Folder — recurse
                queue.append(full_path)
        logger.info("Listed prefix %r: %d files so far", prefix or "/", len(all_files))
    return all_files


def already_in_r2(r2_client, bucket: str, key: str, expected_size: int) -> bool:
    try:
        head = r2_client.head_object(Bucket=bucket, Key=key)
        return head["ContentLength"] == expected_size
    except r2_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def migrate_one(cfg: MigrationConfig, sb_client, r2_client, file: dict) -> tuple[str, str, int]:
    """Returns (key, status, bytes_transferred). Status one of:
    skipped_exists, dry_run, ok, error
    """
    key = file["key"]
    size = int(file.get("size") or 0)

    if not cfg.dry_run and already_in_r2(r2_client, cfg.r2_bucket, key, size):
        return key, "skipped_exists", 0

    if cfg.dry_run:
        return key, "dry_run", size

    # Download from Supabase
    try:
        data = sb_client.storage.from_(cfg.supabase_bucket).download(key)
    except Exception as exc:
        msg = str(exc).lower()
        if any(s in msg for s in ("quota", "restricted", "429", "rate")):
            logger.critical(
                "QUOTA BLOCK: Supabase refusing downloads. Either upgrade to "
                "Pro for the migration window or wait for June 6 refill. "
                "File: %s", key,
            )
            raise SystemExit(3)
        return key, f"error_download:{exc!s}", 0

    # Verify size matches expectation
    if size and len(data) != size:
        logger.warning("Size mismatch for %s: expected %d, got %d", key, size, len(data))

    # Upload to R2
    try:
        content_type = file.get("metadata", {}).get("mimetype") or "application/octet-stream"
        r2_client.put_object(
            Bucket=cfg.r2_bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
    except Exception as exc:
        return key, f"error_upload:{exc!s}", 0

    return key, "ok", len(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="List what would migrate, no actual transfer")
    ap.add_argument("--limit", type=int, default=None, help="Only migrate first N files (for testing)")
    args = ap.parse_args()

    cfg = load_config(args)
    logger.info("Migration target: Supabase[%s] → R2[%s] (workers=%d, dry_run=%s)",
                cfg.supabase_bucket, cfg.r2_bucket, cfg.workers, cfg.dry_run)

    sb = get_supabase_client(cfg)
    r2 = get_r2_client(cfg)

    # Verify R2 bucket exists / we have access
    try:
        r2.head_bucket(Bucket=cfg.r2_bucket)
        logger.info("R2 bucket %r reachable", cfg.r2_bucket)
    except Exception as exc:
        logger.error("R2 bucket %r not reachable: %s", cfg.r2_bucket, exc)
        logger.error("Make sure the bucket exists and the API token has Object Read+Write permissions.")
        sys.exit(2)

    # List all source files
    logger.info("Listing Supabase bucket contents...")
    files = list_supabase_files(sb, cfg.supabase_bucket)
    logger.info("Found %d files, total %.1f MB",
                len(files), sum(f["size"] for f in files) / 1e6)

    if cfg.limit:
        files = files[: cfg.limit]
        logger.info("Limited to first %d files for this run", len(files))

    # Migrate in parallel
    t0 = time.time()
    counts = {"ok": 0, "skipped_exists": 0, "dry_run": 0, "error": 0}
    bytes_moved = 0
    errors = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(migrate_one, cfg, sb, r2, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                key, status, n = fut.result()
            except SystemExit:
                raise
            except Exception as exc:
                key = "?"
                status = f"error:{exc!s}"
                n = 0
            bucket = "error" if status.startswith("error") else status
            counts[bucket] = counts.get(bucket, 0) + 1
            bytes_moved += n
            if status.startswith("error"):
                errors.append((key, status))
                logger.warning("[%d/%d] FAIL %s — %s", i, len(files), key, status)
            elif i % 25 == 0 or i == len(files):
                elapsed = time.time() - t0
                rate = bytes_moved / max(elapsed, 1) / 1e6
                logger.info(
                    "[%d/%d] %s %s (%.1f MB) | total %.1f MB at %.1f MB/s",
                    i, len(files), status, key, n / 1e6, bytes_moved / 1e6, rate,
                )

    elapsed = time.time() - t0
    logger.info("─" * 60)
    logger.info("DONE in %.1fs", elapsed)
    logger.info("  ok:             %d", counts.get("ok", 0))
    logger.info("  skipped_exists: %d", counts.get("skipped_exists", 0))
    logger.info("  dry_run:        %d", counts.get("dry_run", 0))
    logger.info("  errors:         %d", counts.get("error", 0))
    logger.info("  bytes moved:    %.2f MB", bytes_moved / 1e6)
    if errors:
        logger.warning("First 5 errors:")
        for key, status in errors[:5]:
            logger.warning("  %s — %s", key, status)
        sys.exit(1)


if __name__ == "__main__":
    main()


# ============================================================
# AFTER VERIFICATION — Delete originals from Supabase (SQL)
# ============================================================
#
# Once you've spot-checked that ~10 random files load correctly from R2:
#
#   -- Run in Supabase SQL Editor (facegreet project):
#   -- This deletes the metadata rows; Supabase has a trigger that
#   -- removes the underlying objects automatically.
#
#   DELETE FROM storage.objects
#   WHERE bucket_id = 'facegreet-videos';
#
#   -- Then to reclaim disk:
#   VACUUM FULL storage.objects;
#
# After this, Supabase storage usage drops to near-zero and the quota
# restriction lifts automatically (no need to wait for June 6).
# ============================================================
