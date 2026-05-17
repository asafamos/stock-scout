#!/bin/bash
# ============================================================
# facegreet → R2 migration — June-6 runbook
# ============================================================
# Run on/after June 6, 2026 (when Supabase Free quota refills).
#
# Prerequisites (all ALREADY done as of May 17):
#   ✓ Cloudflare R2 bucket `facegreet-videos` created
#   ✓ R2 API token created with Object Read+Write
#   ✓ .env.facegreet-migration file exists with all credentials
#   ✓ supabase + boto3 + requests installed in .venv
#
# Usage:
#   Interactive:   bash scripts/run_facegreet_migration.sh
#   Auto (no prompt):  bash scripts/run_facegreet_migration.sh --yes
#
# What this does:
#   1. Loads credentials from .env.facegreet-migration
#   2. Validates all env vars are clean ASCII
#   3. Runs a dry-run on 5 files (sanity check)
#   4. PROMPTS for confirmation (unless --yes)
#   5. Runs full migration (734 files, ~3.5 GB, ~10-30 min)
#   6. Prints SQL to delete from Supabase after spot-check
# ============================================================

set -e  # exit on any error

# Parse args
AUTO_YES=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) AUTO_YES=1 ;;
    esac
done

cd "$(dirname "$0")/.."

ENV_FILE=".env.facegreet-migration"
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Missing $ENV_FILE"
    echo "   Copy template:  cp .env.facegreet-migration.template $ENV_FILE"
    echo "   Then edit the REPLACE_ME lines with real credentials."
    exit 1
fi

echo "── 1. Loading credentials ──"
set -a
source "$ENV_FILE"
set +a

echo ""
echo "── 2. Validating env vars (clean ASCII?) ──"
.venv/bin/python scripts/check_env_ascii.py || {
    echo "❌ Some env vars have non-ASCII contamination. Fix $ENV_FILE."
    exit 1
}

echo ""
echo "── 3. Dry-run on 5 files ──"
.venv/bin/python scripts/migrate_supabase_to_r2.py --dry-run --limit 5

echo ""
if [ "$AUTO_YES" -eq 1 ]; then
    echo "AUTO MODE (--yes): proceeding to live migration without prompt."
else
    echo "If the dry-run above looks healthy (no QUOTA BLOCK errors, 5 files listed),"
    echo "we're ready for the live migration of 734 files (~3.5 GB, ~10-30 minutes)."
    echo ""
    read -r -p "Proceed with LIVE migration? [yes/no]: " ans
    if [ "$ans" != "yes" ]; then
        echo "Aborted by user."
        exit 0
    fi
fi

echo ""
echo "── 4. Live migration ──"
.venv/bin/python scripts/migrate_supabase_to_r2.py

echo ""
echo "── 5. NEXT STEPS (manual) ──"
echo ""
echo "  a) Spot-check ~5 random files in R2 — open Cloudflare dashboard:"
echo "     https://dash.cloudflare.com/91006435fd2340fdf98a522be868b198/r2/default/buckets/facegreet-videos"
echo ""
echo "  b) Update facegreet app code to read/write R2 instead of Supabase Storage."
echo "     (see the patterns in conversation history — presigned URLs, custom domain.)"
echo ""
echo "  c) Once facegreet is fully on R2, delete originals from Supabase:"
echo "     Open: https://supabase.com/dashboard/project/ewkrpukgyqndmznituzg/sql/new"
echo "     Run:"
echo ""
echo "       DELETE FROM storage.objects WHERE bucket_id = 'facegreet-videos';"
echo "       VACUUM FULL storage.objects;"
echo ""
echo "  d) After delete, Supabase storage drops to ~0 → quota restriction lifts automatically (~30 min)."
echo ""
echo "  e) Final cleanup:"
echo "       rm $ENV_FILE          # remove local secrets"
echo "       # Revoke R2 token + Supabase service_role token if you don't need them"
