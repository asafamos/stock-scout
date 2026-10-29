# F2 weekly round — runbook for the scheduled (unattended) Claude session

Purpose: run the weekly round of the forward test F2 (docs/research_prereg_F2_llm_picker.md, amendment A6) so it no longer depends on a human opening a session.
Run from `/Users/asafamos/StockScout/stock-scout-2` (the research data `research_data/` exists only on this Mac, which is why this cannot run on GitHub/the VPS).
You are a FRESH session with no memory: follow these steps literally. **Never change a prompt, never look at any return/outcome, never run `f2_resolve.py`.**

## Steps
1. `git pull -q --rebase --autostash origin main`
2. `.venv/bin/python scripts/forward/f2_round.py plan` → JSON `{round_date, asof, deadline_utc, status}`.
   - `done` → print "F2 round already logged" and STOP (the Monday catch-up run ends here when Sunday's run worked).
   - `too_late` → the market has opened; a round can no longer count. Send the alert (step 7) and STOP. Do not run a late round.
   - `run` → continue with `round_date = D` from the JSON.
3. `.venv/bin/python scripts/forward/f2_round.py build --round-date D` (3-8 minutes; builds the pool, the numeric pack and the news pack with data as of the Friday close). If it fails, retry ONCE; if it fails again send the alert and STOP.
4. For each arm print the exact prompt: `.venv/bin/python scripts/forward/f2_round.py prompt A --round-date D` (also B, C). Start THREE fresh `general-purpose` subagents in parallel, one per arm, whose prompt is exactly that text and nothing else (no extra context). Arm C is the only one allowed to use the web.
5. Save each subagent's JSON reply verbatim to the session scratch directory as `a.json`, `b.json`, `c.json`.
6. `.venv/bin/python scripts/forward/f2_round.py finalize --round-date D --a a.json --b b.json --c c.json --commit`
   It validates (exactly 5 distinct tickers from the pool, confidence 1-5), refuses after the entry open, writes `round_D_picks.json` with hashes, commits and pushes. If it raises a validation error, re-ask THAT arm once with the same prompt; a second failure → alert and STOP.
7. Telegram (always, success or failure): `.venv/bin/python -c "from core.trading import notifications as n; n._send('<message>')"`. Success text: `🗓️ F2 round D logged — A:[..] B:[..] C:[..]`. Failure text: `⚠️ F2 round D NOT logged: <reason> — run it manually before the Monday open (tell Claude: "תריץ סבב F2")`.
8. Verify `git status -sb` shows the branch in sync with origin and that `round_D_picks.json` exists on origin/main. Report one line.

## Schedule (local Israel time; the Mac and the Claude app must be on)
- Sunday 14:00 — main run. Monday 10:00 — catch-up (does nothing if the Sunday run succeeded: step 2 returns `done`).
- Independent safety net: the GitHub workflow `f2_watchdog.yml` alerts on Telegram at Monday 11:30 UTC if no round for the latest Sunday is on origin/main, i.e. before the 13:30/14:30 UTC open.

## Why this design
Deterministic code (`scripts/forward/f2_round.py`, unit-tested in tests/test_f2_round.py) does everything that needs no judgement: dates, deadline, prompts, validation, hashing, commit. The only judgement-free-but-model-dependent part is the three blind pickers, which are fresh subagents exactly as in the manually run rounds.
