#!/bin/bash
# Safe code-only deploy from origin/main to VPS — never touches data/trades/*.
#
# Background (2026-04-30): a backup-then-restore pattern in earlier deploys
# accidentally rewrote data/trades/open_positions.json with a stale snapshot,
# losing TDW close + PL+ORCL opens. The tracker fell out of sync with IB
# until manually fixed. This script eliminates the risk by ONLY checking
# out specific code/script paths from origin — never touching data/trades.
#
# Usage:
#   bash scripts/safe_deploy.sh

set -euo pipefail

ROOT=/home/stockscout/stock-scout-2
cd "$ROOT"

echo "=== Safe deploy from origin/main ==="
git fetch origin main --quiet

# Code paths only — never data/trades/*
PATHS_TO_DEPLOY=(
    "core/"
    "scripts/"
    "deploy/"
    "stock_scout.py"
    ".github/workflows/"
)

for p in "${PATHS_TO_DEPLOY[@]}"; do
    if git cat-file -e "origin/main:$p" 2>/dev/null; then
        git checkout origin/main -- "$p" 2>&1 | tail -3 || true
    fi
done

# Code-only checkout for specific data files we DO want from git:
# - data/scans/* (overwrites local — these are git-managed scan outputs)
# - models/*.pkl (overwrites local — managed by ML training pipeline)
git checkout origin/main -- data/scans/latest_scan.parquet \
    data/scans/latest_scan.json \
    data/scans/latest_scan.meta.json 2>/dev/null || true

# DO NOT touch:
#   - data/trades/   (live tracker state)
#   - data/outcomes/ (ML feedback loop accumulates here)
#   - data/cache/    (insider signal cache, etc)

echo "Deploy complete. Verifying critical files NOT modified:"
for f in data/trades/open_positions.json data/trades/trade_log.json data/trades/portfolio_snapshot.json; do
    if [ -f "$f" ]; then
        echo "  $f: $(stat -c %y "$f")"
    fi
done

echo ""
echo "=== Done. To restart services, run: ==="
echo "  sudo systemctl restart stockscout-monitor.service"
