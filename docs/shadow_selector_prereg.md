# Shadow selector — pre-registration (written 2026-09-29, before any forward result exists)

## Why
Audit of 16.8k resolved scan candidates (85 scan dates, Mar–Sep 2026), per-date rank IC vs realised return:
`Fundamental_Score` +0.081 (t≈6.6), `RR` +0.066, `ML` +0.016, **`Score` +0.006**. The live Score gate
(73–85) therefore selects on a number that carries no ranking information, while fundamentals do.
That analysis is **exploratory**: the feature was chosen after looking at the data, windows overlap,
entry was the scan price and there was no cost model. It is a hypothesis, not evidence. Per the
no-flip-flop rule the live selector does not change on it. This document fixes, in advance, how the
hypothesis will be tested on data that does not exist yet.

## Rule under test — `S1_v1` (frozen; a change = a new rule id)
Each scan day: universe = scan rows with `Fundamental_Score` ≥ 45, sector not in the live
`blocked_sectors_list`, `Close` > 0. Rank by `Fundamental_Score` desc (ties: ticker asc). Pick the top 3.
(No Score / ML / RR / confidence gates — that is the point of the test.)

## Data (forward only)
`scripts/shadow_log.py` runs every trading day after the close (`stockscout-shadow.timer`,
17:30 New York) on the newest scan available in `origin/main` at that moment; one log per scan date (a second run for the same date is ignored). It stores the
**whole** scan (all rows), the S1 flag/rank, and `live_gate_pass` (today's static live gates).
Sampling does not depend on bot capacity, cash or positions.

## Measurement
Entry = open of the first session after the scan date. Exit = close of the 20th session counting the
entry session (20 trading days). Benchmark = SPY over the same window. Windows that do not have all 20
sessions stay unresolved. Round-trip cost 0.5% subtracted (report `--cost` to vary). Inference unit =
scan date; 95% CI by bootstrap over dates (`scripts/outcome_stats.py`).

## Decision rule
No verdict before **60 resolved scan dates** (≈ mid-January 2027). Then:
* **S1 passes** iff (a) mean net excess return vs SPY of S1 has CI lower bound > 0, **and**
  (b) S1 − ALL (equal-weight of the whole scan) has CI lower bound > 0.
  A pass means "propose a live change to the owner", never "apply it" — the owner decides.
* Otherwise S1 does **not** pass and the live selector stays as it is.
* `S1 − LIVE` is reported for information (LIVE is an approximation: all rows passing the static gates,
  before the ranker narrows to ≤3).
* No looking at partial results to adjust the rule, the horizon, the cost or N. Peeking is allowed;
  acting on it is not.

## Known limits
Same-day picks are correlated (handled by clustering on date); 20-session windows of consecutive dates
overlap (CI is optimistic — treat borderline passes as fails); a 3-month regime is one regime.

---
## Addendum A (2026-09-29 evening — still before any forward result; 1 scan logged)

**Offline evidence that changes the prior for S1.** Point-in-time test on 431 large/mid-cap tickers,
FMP quarterly statements lagged to `acceptedDate`+1, daily FMP market cap, 2019-06 → 2026-08, 92
non-overlapping 20-session windows:
* Quality / valuation / growth factors (ROE, margins, FCF yield, earnings yield, B/P, leverage,
  accruals) have rank IC ≈ 0 (|t| < 2.3, several with the "wrong" sign). Revenue growth is weakly positive (t≈1.8).
* The only stable signals were a volatility premium (`ATR_Pct` IC +0.05) and a small-size premium
  (`log_mcap` IC −0.05, t −4.4) — both risk premia, both inflated by survivorship (universe = today's large caps).
* Walk-forward (purged, expanding, test years 2021–2026): a 16-feature fundamental model IC ≈ 0.00; the
  5-feature technical model like the live one ≈ ATR alone (IC 0.03, t 1.3); "technical + size" is best
  (IC 0.036, t 2.1, top-decile excess +1.8%/20d, t 3.5) — and that is just the vol/size tilt.
* **Correction of the audit's earlier claim** that Fundamental_Score was "the only robust signal
  (t=6.6)": that t-stat treated 85 overlapping scan days as independent; they cover ≈ 9 independent
  20-session windows, and the point-in-time store shows the effect flipping sign between January and
  February 2026. Treat fundamentals-first (S1) as a low-prior hypothesis. It stays in the test because
  the live universe (≈2000 names incl. small caps) differs from the offline one.

**New exploratory rule `S2_v1`** (vol + small-size tilt, see `scripts/shadow_log.py`) is added to the
log from 2026-09-29 (the one scan already logged was re-logged with the flag). Because two rules are now
tested, each needs one-sided P(mean ≤ 0) < **0.025** (Bonferroni) for both "net excess vs SPY" and
"beats ALL" — the bootstrap CI text in the report is unchanged, the verdict thresholds are not.
S2 is inherently a higher-risk selector (small, volatile names); a pass means "propose", never "apply".
Everything else (entry/exit, cost, 60-date minimum, no peeking-driven changes) is unchanged.

---
## Addendum B (2026-09-29 night — before any forward result): exit policies and the CANARY

Offline exit study (1,780 random entries 2019-2026, net of 0.5% cost, paired vs LEGACY over 356 date
clusters): the legacy exit (trail 9% for 7 sessions then 5.5%, ≤20 sessions) was the worst tested policy
in every calendar year; every ATR-scaled alternative beat it with CI lower bound > 0 (CANARY = trail
clip(4×ATR%, 8, 20)% for ≤30 sessions: +1.45pp/trade, CI lo +0.95). Caveats: survivorship-biased
universe, bull-tilted sample, ratchet tiers and targets not modelled.

Forward test, added to the daily shadow job (`shadow_resolve.resolve_exits` → `shadow_exit_outcomes.jsonl`,
report section "Exit-policy comparison"): every logged scan row is run through the policies in
`core/trading/exit_sim.POLICIES` (LEGACY, HOLD20, CANARY, ATR4_60, ATR5_60, TRAIL15_60, HOLD60), entry =
next open, gross of cost (0.5% applied in the report), unit = scan date.
**Canary criterion:** after ≥ 60 resolved dates, CANARY − LEGACY on the ALL arm must have one-sided
P(mean ≤ 0) < 0.025; otherwise the canary is rolled back (`TRADE_EXIT_PROFILE=legacy`). The other policies
are descriptive (they show whether CANARY is a knife-edge choice) and carry no decision weight.

Live: the canary (`TRADE_EXIT_PROFILE=atr_wide`) applies to NEW buys only, with risk-based size
(a full initial-trail loss ≤ 4% of NetLiq — at ~$800 NetLiq that is roughly half the legacy size), no
ratchet/time-tighten/partial-profit, time exit 42 calendar days (earnings-aware cap still applies).
Live closes are far too few to validate (≈170 trades for a 1%/trade edge); they are monitored, not judged.

---
## Addendum C (2026-09-30 — before any forward result): rule S3_v1 = the live v2 sleeve

The owner approved a bounded live sleeve trading the volatility + small-size tilt. To keep "what trades"
identical to "what is measured", the tradable version of S2 is logged as **S3_v1** (`core/trading/v2_selector.py`,
shared by the live sleeve and the shadow logger): universe = Close ≥ 5, ATR_Pct > 0, market cap > 0, average
dollar volume ≥ $5M, sector not blocked; score = pct_rank(ATR_Pct) + pct_rank(−market cap); top 3/day (the live
sleeve trades the best available name, max 1 open position). PEAD was deliberately left OUT: in the offline
simulation the PEAD + vol/size composite did worse than vol/size alone (+1.12% vs +3.67% excess vs SPY).
Three rules are now tested (S1, S2, S3) → Bonferroni: each needs one-sided P(mean ≤ 0) < 0.05/3 ≈ 0.0167 for
both "net excess vs SPY" and "beats ALL". S2 and S3 are the same idea (S3 adds tradability filters); S3 is the
one that carries the live decision. Live protocol: signal = latest completed-session scan in origin/main, order at
09:31 ET next session (marketable limit from a real-time quote), gap/slippage guard ±6% vs the signal close,
atr_wide exit, risk cap 4% of NetLiq per trade.

**Sleeve self-stop (capital protection, not validation):** the sleeve switches itself off
(`data/state/v2_sleeve_disabled.json`, manual re-enable) if its own closed trades reach a cumulative realized
loss ≥ $60 (n ≥ 3) or a mean < −2%/trade after 10 closes. Go-live is gated on DRY_RUN verification and the
owner's explicit confirmation.
