# PREREG P1 — Does the post-fix Score / the eligible set select better 20-day outcomes? (written 2026-10-19, BEFORE any outcome exists)

Question (the owner's, 2026-10-19): "what do we think of the logic and quality of the recommendations after all the recent changes?" Answer today: the plumbing is cleaner
(five silent failures fixed, ML out of decisions, real RR in the Score, cap 99), but there is **no outcome data on the new Score yet**; the OLD score had a
rank-IC of −0.088 against the 20-day outcome on 1,375 resolved candidates (2026-08-25 … 10-06; overlapping, so not independent evidence, but no sign of skill).
This document fixes how the new Score will be judged so that the answer, when it arrives, cannot be argued after the fact.

**State at writing:** `scan_outcomes.jsonl` on the VPS holds 0 resolved records with scan_date ≥ 2026-10-08 (latest resolved scan_date: 2026-09-18). Records resolve ≈20 trading days after
their scan, so the first post-fix outcomes appear around 2026-11-05. The evaluator prints coverage only until the criteria below are met.

## Data (fixed)
- Source: `data/outcomes/scan_outcomes.jsonl` (VPS), records with `scan_date ≥ 2026-10-08` (first scans of the fully fixed pipeline: ML out of decisions, real RR in the Score, RS_63d populated, D/E as a ratio, cap 99), `resolved = true`.
- One observation per (ticker, scan_date) (earliest recorded). Outcome y = `final_return_pct` (20-trading-day return, hold to horizon, gross, resolver v2). **Excess = y − mean(y of all resolved records on the same scan_date)** (removes the market factor).
- Scans with RS_63d/Close NaN > 30% are blocked from the repository by the integrity check (commit 60f86aa8) and cannot enter; records without a valid entry price do not resolve.
- **Eligible flag** = the FROZEN static gates (`policy.evaluate_static_gates`, CONFIG as of 2026-10-19: Score 73–99, RR 2.5–5.0, ATR ≥ 0.03, Fundamental ≥ 45, Confidence High, blocked sectors/regimes, ML gate off) replayed on the committed scan parquet(s) of that scan_date (any scan that day). Gate inputs such as SignalQuality are not stored in `scan_outcomes`, hence the replay (`scripts/research/p1_eval_new_score.py`).

## Two primary tests (the only decisions)
- **T1 — Score signal:** for each scan_date with ≥ 30 resolved names, Spearman rank-IC between `score` and excess return; statistic = mean over dates. **Pass: mean IC ≥ +0.03 AND ticker-permutation p < 0.025.**
- **T2 — Eligible set:** per scan_date, mean excess of the eligible names (dates with ≥ 1 eligible); statistic = mean over dates. **Pass: mean ≥ +1.0 pp (≈ 2× the 0.5 pp round-trip cost) AND ticker-permutation p < 0.025 AND the moving-block-bootstrap (block = 10 dates) 2.5th percentile > 0.**
- Inference is by **ticker-path permutation** (re-pair each ticker's whole signal path with another ticker's outcome path, 2,000 draws): the same names recur every day and 20-day windows overlap, so a date-level bootstrap or a within-date permutation treats ~40 dates as ~40 independent looks. Calibrated in simulation (below).
- Multiple testing: two primary tests → α = 0.025 each.

## Verdict (fixed)
- Both pass → **DEMONSTRATED EDGE**. Exactly one passes → **PARTIAL (owner decision)**. Neither passes and mean IC ≤ 0.01 and eligible excess ≤ 0.5 pp → **NO DEMONSTRATED EDGE**. Otherwise **INCONCLUSIVE**.
- Reported but never decisions: IC of the Score after removing its rank-dependence on ATR_Pct ("does it add beyond volatility?" — the old ML was an ATR proxy), the ATR-rank IC as a naive baseline, the IC of Fundamental_Score and TechScore separately.

## Stopping rule (single look, blind until then)
- **No return statistic is shown until BOTH:** ≥ 25 resolved scan_dates with ≥ 30 names each AND ≥ 150 resolved eligible (ticker, scan_date) observations. Before that, `p1_eval_new_score.py` prints coverage counts only.
- Evaluate **once** when both hold (earliest plausible: December 2026). If INCONCLUSIVE, observation continues and **one FINAL look is allowed on/after 2027-03-31 with α = 0.0125 per test** (Bonferroni over the two looks). No other looks.

## Calibration (simulation, scratch script; 40 dates × 60 persistent names, AR ρ = 0.95 in both the signal and the outcome path)
- H0 (no link between Score and outcome): T1 passes 2.5 %, T2 0.8 %, either 3.3 % (N = 120) — i.e. at the nominal level. The first version of the test (date-level block bootstrap p-values) passed T1 **24 %** of the time under H0 and was rejected before use.
- Power: IC ≈ 0.39 / eligible excess ≈ 3.3 pp → ~100 %; IC ≈ 0.20 / 1.6 pp → T1 90 %, T2 60 %; IC ≈ 0.09 / 0.8 pp → T1 32 %, T2 20 %.
- **Consequence, stated plainly:** with ~40 scan dates this test can only see a LARGE effect (IC ≳ 0.2, excess ≳ 1.6 pp). A genuine but modest edge will most likely read INCONCLUSIVE, and "NO DEMONSTRATED EDGE" does not prove there is none — it means none large enough to matter after ~0.5 pp costs. These numbers assume the simulated noise level; real outcome noise is probably higher, so real power is probably lower.

## Disclosed limits
- `scan_outcomes` holds the scan's ≈100 best-ranked names per scan, not the whole universe: T1 measures ranking WITHIN that set. Selection into the set is not tested here.
- y is hold-to-20-days; the bot exits with the atr_wide trail and trade management. P1 measures selection, not the whole system's P&L (that is the ledger's job).
- One market regime so far (MODERATE_UP). A result in one regime is not a result in all of them.
- The Score formula is part of what is tested: **any change to the Score formula or to the eligibility gates (other than bug fixes that restore the documented behaviour) restarts this study** with a new preregistration. Until the evaluation, the frozen-gate rule (no-flipflop) stays.

## What happens after
Nothing automatic. DEMONSTRATED EDGE → keep, and a separate decision on sizing. NO DEMONSTRATED EDGE or INCONCLUSIVE after the final look → the owner decides between keeping the legacy channel as a small experiment, simplifying to CoreTrend only, or a passive allocation. The evaluator never changes a live setting.

## Addendum A1 (2026-10-29, before any outcome exists) — pre-committed decision rule for the legacy channel
Recorded at the owner's request ("yes", 2026-10-29) so that what to do with the legacy buy channel cannot be argued after the verdict. It changes no test, threshold, data selection or look schedule above; it only turns "What happens after" into a table. The owner may change this table BEFORE a verdict is produced, never after.

| P1 verdict | Legacy channel (Score/gates → small stock positions) |
|---|---|
| DEMONSTRATED EDGE | Keep running; sizing is a separate decision (do not scale on one study) |
| PARTIAL (exactly one test passes) | Owner decision |
| INCONCLUSIVE after the FINAL look (on/after 2027-03-31) | Simplify to CoreTrend only: stop new legacy buys, let open positions exit on their trails |
| NO DEMONSTRATED EDGE (either look) | Stop new legacy buys now, open positions exit on their trails |

Until a verdict: the legacy channel stays running and untouched (owner instruction: the channel is not to be stopped; gates stay frozen, no-flipflop). The live ledger is NOT the evidence for this decision — with 3 slots, a $150 minimum notional and ~$46 free cash it produces too few trades; P1 (candidate-level) and the shadow arms are.

