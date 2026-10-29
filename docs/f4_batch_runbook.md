# F4 daily scoring batch — runbook for the scheduled (unattended) Claude session

Prereg: docs/research_prereg_F4_earnings_text.md. Run from `/Users/asafamos/StockScout/stock-scout-2`. You are a FRESH session with no memory: follow the steps literally.
**Never change the prompt, never look at prices/returns, never run `f4_resolve.py` for anything but counts, never score releases yourself — only fresh subagents score.**
The batch must be committed BEFORE 09:30 America/New_York (the entry open); the helper refuses to write after that.

## Steps
1. `git pull -q --rebase --autostash origin main`
2. `.venv/bin/python scripts/forward/f4_collect.py collect` (1-4 minutes; prints JSON with `entry_date` and counts). If it fails, retry ONCE; a second failure → alert (step 7) and STOP. If it says the events file already exists, the batch was already collected today → run `status`, report, STOP.
3. `.venv/bin/python scripts/forward/f4_collect.py nbatches --entry-date D` → N batches (0 is possible: then go straight to step 6 with no score files).
4. For k in 0..N-1 print the exact prompt: `.venv/bin/python scripts/forward/f4_collect.py prompt k --entry-date D`. The command also writes the batch's releases to a file named at the end of the prompt, so the prompt itself is short. Start N fresh `general-purpose` subagents IN PARALLEL (at most 8 at a time), each one with that printed text and nothing else (no extra context); they read that one file with the Read tool.
5. Save each reply verbatim to the session scratch directory as `s0.json … s{N-1}.json` (in batch order).
6. `.venv/bin/python scripts/forward/f4_collect.py finalize --entry-date D --scores s0.json s1.json … --commit` — validates ids and integer scores -5..5, writes `D_scores.json` with the prompt hash, commits and pushes. If validation fails for one batch, re-ask THAT batch once with the same prompt; a second failure → alert and STOP.
7. Telegram (always): `.venv/bin/python -c "from core.trading import notifications as n; n._send('<message>')"`. Success: `📰 F4 D: N events scored (X positive / Y negative)`. Failure: `⚠️ F4 D NOT scored: <reason>` — then tell the owner it can still be run by hand before 09:30 ET.
8. Verify `git status -sb` is in sync with origin and `data/forward_text/batches/D_scores.json` is on origin/main. Report one line.

## Schedule (local Israel time; Mac and Claude app must be on)
- Monday-Friday 15:20 (= 08:20 America/New_York in winter). Until the US clocks change on 2026-11-01 the equivalent is 14:20, so a one-off task covers Friday 2026-10-30.
- Safety net: GitHub workflow `f4_watchdog.yml` alerts on Telegram if the day's scores file is missing from origin/main.
