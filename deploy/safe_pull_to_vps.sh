#!/usr/bin/env bash
# Safe deploy script for the VPS.
#
# CRITICAL: a naive `git reset --hard origin/main` on the VPS would
# DELETE data/trades/*.json from the working tree. These files were
# git-tracked in earlier commits (≤577e9dc) but moved to .gitignore
# in commit 9bcc9d0 — git treats their absence in main as "deletion"
# and removes them from the working tree on hard-reset. The runtime
# tracker would then be wiped, and the monitor's next cycle would
# react as if all positions just closed (the very incident we already
# debugged once today, in the opposite direction).
#
# This script:
#   1. Backs up every runtime JSON in data/trades/* and data/state/*
#   2. Does the fetch + reset
#   3. Restores the runtime files on top of the new tree
#   4. Restarts only the services whose code actually changed
#   5. Verifies the monitor came back live before exiting
#
# Usage (from your Mac):
#   ssh root@87.99.142.12 'bash -s' < deploy/safe_pull_to_vps.sh
#
# Or copy-paste interactively after SSH'ing in.

set -euo pipefail

REPO=/home/stockscout/stock-scout-2
SCOUT_USER=stockscout
BACKUP_DIR=/tmp/stockscout-pre-deploy-$(date +%s)

cd "$REPO"

echo "━━━ Step 1: Back up runtime files (so reset --hard doesn't wipe them) ━━━"
# 2026-09-28 fix: expanded from *.json only to include non-JSON runtime artefacts
# that `git reset --hard` would otherwise silently discard:
#   - data/trades/*.jsonl (executions ledger — CRITICAL for P&L attribution)
#   - data/outcomes/*.jsonl (pending scans + resolved scan_outcomes — feeds ML)
#   - .env.trading edits made on VPS (drift-detector expects these)
#   - data/cache/*.json (insider signals, analyst PT cache)
#   - Anything else the VPS has locally modified
mkdir -p "$BACKUP_DIR/trades" "$BACKUP_DIR/outcomes" "$BACKUP_DIR/state" "$BACKUP_DIR/cache" "$BACKUP_DIR/etc"
for f in data/trades/*.json data/trades/*.jsonl; do
    [ -f "$f" ] && cp -p "$f" "$BACKUP_DIR/trades/"
done
for f in data/outcomes/*.jsonl; do
    [ -f "$f" ] && cp -p "$f" "$BACKUP_DIR/outcomes/"
done
for f in data/state/*; do
    [ -f "$f" ] && cp -p "$f" "$BACKUP_DIR/state/"
done
if [ -d data/cache ]; then
    for f in data/cache/*.json; do
        [ -f "$f" ] && cp -p "$f" "$BACKUP_DIR/cache/"
    done
fi
# .env.trading — critical config. Never in git, but VPS edits it.
if [ -f .env.trading ]; then
    cp -p .env.trading "$BACKUP_DIR/etc/.env.trading"
fi
# List every file the VPS has locally modified beyond origin/main — audit trail
# so we know exactly what's being preserved. We DO NOT backup git-tracked local
# edits (those are intentional divergence and reset will discard them).
sudo -u "$SCOUT_USER" git status --short > "$BACKUP_DIR/git-status-pre-reset.txt" 2>&1 || true
echo "  backup → $BACKUP_DIR"
ls -la "$BACKUP_DIR"/*/ 2>/dev/null | head -20

echo ""
echo "━━━ Step 2: Fetch + reset main ━━━"
sudo -u "$SCOUT_USER" git fetch origin main
BEFORE=$(sudo -u "$SCOUT_USER" git rev-parse HEAD)
sudo -u "$SCOUT_USER" git reset --hard origin/main
AFTER=$(sudo -u "$SCOUT_USER" git rev-parse HEAD)
echo "  HEAD: ${BEFORE:0:7} → ${AFTER:0:7}"

echo ""
echo "━━━ Step 3: Restore runtime files ━━━"
# 2026-09-28: restore same expanded set as Step 1 backup
mkdir -p data/trades data/outcomes data/state data/cache
for f in "$BACKUP_DIR/trades"/*.json "$BACKUP_DIR/trades"/*.jsonl; do
    [ -f "$f" ] && cp -p "$f" data/trades/
done
for f in "$BACKUP_DIR/outcomes"/*.jsonl; do
    [ -f "$f" ] && cp -p "$f" data/outcomes/
done
for f in "$BACKUP_DIR/state"/*; do
    [ -f "$f" ] && cp -p "$f" data/state/
done
for f in "$BACKUP_DIR/cache"/*.json; do
    [ -f "$f" ] && cp -p "$f" data/cache/
done
# Restore .env.trading only if the reset removed it (usually it's gitignored so
# survives, but be defensive)
if [ ! -f .env.trading ] && [ -f "$BACKUP_DIR/etc/.env.trading" ]; then
    cp -p "$BACKUP_DIR/etc/.env.trading" .env.trading
    echo "  .env.trading restored from backup (was missing after reset)"
fi
chown -R "$SCOUT_USER:$SCOUT_USER" data/trades data/outcomes data/state data/cache
echo "  restored runtime state from backup:"
ls -la data/trades/ data/outcomes/ data/state/ 2>/dev/null | head -20

echo ""
echo "━━━ Step 4: Restart services that picked up code changes ━━━"
# Order matters — broadcaster pushes state, monitor reads tracker,
# poller reads commands. Restart broadcaster last so the dashboard
# sees the post-restart state immediately.
for svc in stockscout-monitor stockscout-command-poller stockscout-statusbot; do
    systemctl restart "$svc" || echo "  ⚠ $svc restart failed"
    sleep 1
done
# Broadcaster is a oneshot triggered by timer — pull the timer
systemctl restart stockscout-state-broadcaster.timer || true
echo "  ✓ services restarted"

echo ""
echo "━━━ Step 5: Verify monitor came back live ━━━"
sleep 4
ACTIVE=$(systemctl is-active stockscout-monitor)
echo "  stockscout-monitor: $ACTIVE"
if [ "$ACTIVE" != "active" ]; then
    echo "  ⚠ monitor not active — investigating:"
    systemctl status stockscout-monitor --no-pager | head -10
    exit 1
fi

echo ""
echo "━━━ Step 6: First log lines to confirm new code is loaded ━━━"
journalctl -u stockscout-monitor --no-pager -n 12 | tail -12

echo ""
echo "━━━ Step 7: Verify tracker file restored correctly ━━━"
sudo -u "$SCOUT_USER" python3 -c "
import json
data = json.load(open('data/trades/open_positions.json'))
print(f'  tracker has {len(data)} open positions')
for p in data:
    print(f'    {p.get(\"ticker\")}: {p.get(\"quantity\")} @ \${p.get(\"entry_price\",0):.2f}')
"

echo ""
echo "✅ Deploy complete. Backup remains at: $BACKUP_DIR"
echo "   (delete after a day if everything looks healthy: rm -rf $BACKUP_DIR)"
