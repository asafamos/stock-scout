# PREREG F2R — F2 run retroactively on the period AFTER the model's knowledge cutoff (written 2026-10-04, BEFORE any retro pick was made)

Why: a forward test (F2) needs months. But the picker (Claude) has a training cutoff of ~June 2026, so July–September 2026 is "future" to it: it cannot recall what happened. Running F2 rounds on weekly dates in that window with point-in-time data
gives an early, CLEAN-ENOUGH answer today. It is weaker than F2 proper (few rounds) and carries disclosed contamination risks (below); F2 forward rounds continue regardless and remain the primary evidence.

## Design (identical to F2 except the dates)
- Rounds: Sundays **2026-07-05, 07-12, 07-19, 07-26, 08-02, 08-09, 08-16, 08-23, 08-30** (9 rounds). Entry = next trading day's Open; exit = Open 20 trading days later (all resolved by 2026-10-02). Rounds before 2026-07-05 are NOT used (too close to the cutoff).
- Pool per round: 100 names drawn with seed = round date (YYYYMMDD) from the tradable universe AS OF THAT DATE (Close ≥ $5, 20-day ADV$ ≥ $1M, marketCap ≥ $300M, using only data up to the round date, survivorship-free panel incl. names that delisted later).
- Two blind arms exactly as F2 A1: **A numbers** (price/vol/short-interest with 9-business-day lag/analyst net upgrades/insider/congress counts — all point-in-time by their own dates) and **B news-only** (≤ 4 headlines from the 7 days before the round, Polygon; no numbers). Same fixed prompts (hashes logged).
- Picker: a fresh Claude subagent per (round, arm), no shared context, answers JSON with exactly 5 tickers from the pool. The pack is delivered as a single file the subagent is told to read; it is told to read nothing else.
- Controls: pool equal-weight mean (primary), 10,000 random-5 draws per round, SPY (reported).
- **Outcome data is NOT looked at until all 18 (round, arm) pick files are written and committed.**

## Contamination mitigations and disclosed residual risks
- Tickers that appear in the project's own memory/CLAUDE.md (traded or discussed names: APH, VG, DELL, PACS, HNGE, CGAU, AAL, FRSH, FTNT, TEO, IVZ, ARCB, MRX, STUB, ARWR, QQQM, QQQ, IEF, SPY) are EXCLUDED from retro pools (subagents may see that context).
- Residual: (1) the model may know partial pre-cutoff context for a name (e.g. a pending catalyst) — that is legitimate "knowledge", but any leakage of post-cutoff events would inflate results; the model's real cutoff is not exactly known. (2) Weak power: 9 rounds → only large effects are detectable. (3) Pool built with the research panel's `profile` for sector/industry (current snapshot; names without a profile are dropped, a mild survivorship effect that is IDENTICAL for picks and controls).

## Decision rule (fixed now; this is an EARLY-SIGNAL test, not a go-live test)
Statistic per arm: mean over the 9 rounds of [mean 20-day return of the 5 picks − pool-mean return]. One-sided permutation test (random-5 draws, same rounds), α = 0.025 per arm (two arms).
- **"Early evidence of skill"** only if p < 0.025 AND mean excess ≥ +1.0% AND the effect is positive in ≥ 6 of 9 rounds. 
- **"No sign of skill"** if the mean excess is ≤ 0 or p ≥ 0.20.
- Anything in between: "inconclusive" (expected for 9 rounds). Neither result changes the F2 forward test or any live setting; a positive result only raises the priority of F2 and of a paper sleeve.
Also reported: hit-rate vs pool, confidence 5 vs 1, sector/size tilt of the picks vs the pool, and whether picks are just the highest-momentum names in the pool (rank of 12-1 momentum).
