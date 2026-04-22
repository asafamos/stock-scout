#!/bin/bash
# Nightly backup: push trade_log + open_positions + scan outcomes to git.
#
# Why: data/trades/*.json is the single source of truth for our trade history.
# If the VPS dies, these files are gone → lifetime P&L and analytics lost.
# Git provides free off-site versioned backup.
#
# Timer: nightly 23:00 UTC (after market close, before midnight).

set -u
cd /home/stockscout/stock-scout-2 || exit 1

# Only commit if there are actual changes
CHANGED_FILES=()
for f in data/trades/trade_log.json \
         data/trades/open_positions.json \
         data/outcomes/pending_scans.jsonl \
         data/outcomes/scan_outcomes.jsonl; do
    if git diff --quiet HEAD -- "$f" 2>/dev/null; then
        continue  # no changes
    fi
    CHANGED_FILES+=("$f")
done

if [ ${#CHANGED_FILES[@]} -eq 0 ]; then
    echo "No state changes to back up"
    exit 0
fi

TODAY=$(date -u +%Y-%m-%d)
git add "${CHANGED_FILES[@]}"
git commit -m "backup: nightly state snapshot $TODAY

Files updated:
$(printf '  - %s\n' "${CHANGED_FILES[@]}")

Automated by scripts/backup_state.sh" --author="stockscout-backup <backup@stockscout.local>" \
    > /dev/null 2>&1

# Push to backup branch — keeps main clean of auto-commits
BACKUP_BRANCH="state-backup"
git push origin HEAD:$BACKUP_BRANCH --force > /dev/null 2>&1

# Reset local HEAD pointer back to main so future pulls work cleanly
git reset --soft HEAD~1 > /dev/null 2>&1
git reset HEAD "${CHANGED_FILES[@]}" > /dev/null 2>&1

echo "Backed up ${#CHANGED_FILES[@]} file(s) to state-backup branch"
