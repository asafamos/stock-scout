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

---
## RESULT (run 2026-10-04; all 18 pick files were committed in git BEFORE any outcome was computed) — VERDICT: "NO SIGN OF SKILL" for both arms
9 rounds (entry Mon 2026-07-06 … 2026-08-31; exit +20 trading days, last exit 2026-09-29), 100 names per round, all prices available. Equal-weight pool mean averaged +0.24% per 20 days; SPY +1.15%.
| arm | mean 20d return of 5 picks | mean excess vs pool | SE | rounds positive | permutation p (one-sided) | hit-rate (picks > pool mean) | beta-adjusted excess |
|---|---|---|---|---|---|---|---|
| A numbers | −2.52% | **−2.75%** | 1.40% | **1 / 9** | 0.952 | 18 / 45 | −2.59% |
| B news-only | −0.94% | **−1.18%** | 2.10% | 4 / 9 | 0.753 | 20 / 45 | −1.73% |
Prereg rule: "No sign of skill" if mean excess ≤ 0 or p ≥ 0.20 → both arms. Arm A's picks were concentrated in names near 52-week highs (momentum continuation); arm B's picks were far larger companies (mean mcap $178B vs pool $31B) because only 10–20% of pool names had any news (Polygon returns ≈1,000 articles/week).
Diagnostics (not preregistered decisions): in the same pools a naive rule "top-5 by 12-month return minus 1-month return" averaged +4.27% excess (SE 2.55%, t ≈ 1.7, 7/9 rounds positive) while "top-5 nearest the 52-week high" averaged −0.95%.
So the LLM underperformed a trivial rule built from the very same data.
Limits: only 9 rounds (SE 1.4–2.1%), so effects smaller than roughly ±3–4% cannot be detected; the news arm is weak because free news coverage is thin; contamination would bias results UPWARD for the AI, so the negative result is the more informative direction.
Next: F2 forward rounds continue unchanged (primary evidence). At evaluation the naive-momentum baseline (computable from the logged pools) is reported alongside the two arms.
