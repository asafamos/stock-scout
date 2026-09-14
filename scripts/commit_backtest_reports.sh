#!/usr/bin/env bash
# Commit + push reports/backtest_{latest,history}.jsonl to main. Called
# from stockscout-weekly-backtest.service after the replay + summary run.
# Race-safe: retries up to 3 times with pull --rebase before push.
set -euo pipefail

cd /home/stockscout/stock-scout-2

git config user.name  "stockscout-vps[bot]"
git config user.email "stockscout-vps@noreply.local"

git add reports/backtest_latest.json reports/backtest_history.jsonl 2>/dev/null || true

if git diff --cached --quiet; then
  echo "commit_backtest_reports: no changes to commit"
  exit 0
fi

git commit -m "Weekly replay backtest results $(date -u +%Y-%m-%d) [skip ci]"

for attempt in 1 2 3; do
  if git push; then
    echo "commit_backtest_reports: push OK on attempt $attempt"
    exit 0
  fi
  echo "commit_backtest_reports: push rejected (attempt $attempt), pulling with rebase and retrying"
  git pull --rebase origin main || true
done

echo "commit_backtest_reports: push failed after 3 attempts" >&2
exit 1
