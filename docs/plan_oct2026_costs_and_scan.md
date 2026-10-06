# Plan — costs, position sizing and scan honesty (prepared 2026-10-06; NOTHING below is deployed except where stated)

Status of code: P1 is implemented behind `TRADE_MIN_POSITION_NOTIONAL_USD` (default 0 = no behaviour change; tests in tests/test_notional_floor.py). It is in `main` but NOT on the VPS.
P2/P3 are specifications only. Everything that changes a live gate needs the owner's explicit go; the legacy gates stay frozen until then.

## 0. Honest framing
- Account ≈ $827; CoreTrend holds ≈ 75% (QQQM). The legacy channel deploys ≈ $154 (19%). Even a perfect legacy channel moves NetLiq by tens of dollars; the return of the account is dominated by QQQM.
- None of the items below creates alpha (Score/ML have no demonstrated edge). They remove avoidable costs and distortions, and stop the scan from reporting precision it does not have.

## 1. Facts (from the ledger, 2026-10-02…05)
IBKR commission = max($1.00 min, $0.005/share) capped at **1% of the trade value** (Fixed plan). Observed: HNGE $0.9775 on $97.75, PACS $0.4267 on $42.67, PATH $0.1305 on $13.03 (exactly 1%). Tiered activation is NOT yet proven (a $98+ fill after the switch would show $0.35).
| position | Fixed $/leg (%) | Tiered $/leg (%) | round trip Fixed / Tiered |
|---|---|---|---|
| $13 | 0.13 (1.00%) | 0.13 (1.00%) | 2.00% / 2.00% |
| $43 | 0.43 (1.00%) | 0.35 (0.81%) | 2.00% / 1.63% |
| $98 | 0.98 (1.00%) | 0.35 (0.36%) | 2.00% / 0.71% |
| $150 | 1.00 (0.67%) | 0.35 (0.23%) | 1.33% / 0.47% |
| $270 | 1.00 (0.37%) | 0.35 (0.13%) | 0.74% / 0.26% |
| $450 | 1.00 (0.22%) | 0.35 (0.08%) | 0.44% / 0.16% |
The three legacy positions pay ≈ $3.07 round trip on $154 deployed (**2.0%**); all four positions ≈ $5.07 = 0.61% of NetLiq. Absolute damage is small, relative drag on the legacy channel is large. With zero demonstrated edge, any sub-$150 buy is negative EV by ≈ 1.3–2% before spread.

## 2. P1 — cost-aware notional floor (implemented, OFF)
Rule: if `qty × price < TRADE_MIN_POSITION_NOTIONAL_USD`, skip the buy ("Position $13 below the cost-aware floor $150 (round-trip commission ≈ 2.0% of the position)"). Also stops the candidate loop early when deployable cash < floor.
Consequence you must accept: with CoreTrend holding 75%, the legacy channel can afford a position ≥ $150 only if cash ≥ $150+buffer.
| NetLiq | cash left after CoreTrend (≈25%) | positions affordable at floor $150 | at floor $270 |
|---|---|---|---|
| $827 | ≈ $207 | 1 | 0 |
| $1,500 | ≈ $375 | 2 | 1 |
| $2,500 | ≈ $625 | 3 (cap) | 2 |
So at today's size a floor of $150 means ONE legacy position at a time (fewer, larger, cheaper trades); $270 means effectively none until a deposit. Choice of floor = owner decision (0 = status quo).
Rollout when approved: (1) deploy core/trading/{config,policy,order_manager}.py via deploy/safe_pull_to_vps.sh; (2) set `TRADE_MIN_POSITION_NOTIONAL_USD=<value>` in VPS `.env.trading` and update `scripts/check_env_vs_docs.py` EXPECTED + CLAUDE.md in the same commit; (3) verify: no pause file, positions protected, a dry run logs the floor skip; (4) rollback = env 0.
Success criteria: no 1-share dust buys; commissions ≤ 0.75% round trip on new buys; no unprotected position; the dry-cycle alert shows "below the cost-aware floor" as the skip reason (not a bug).

## 3. P2 — replace the ML gate (0.40–0.60) by an explicit volatility rule, parity first (spec)
Why: ML ≈ ATR proxy (Spearman 0.84 with ATR_Pct on 2026-10-06's scan; walk-forward IC ≤ 0.04), trained on survivors, ATR defined differently in training (High−Low) and serving (true range).
Steps: (a) from the scan history (git versions of latest_scan.parquet, ~60k rows) fit the ATR_Pct band that reproduces ML∈[0.40,0.60] membership and report agreement; (b) shadow-log both decisions per candidate for ≥ 2 weeks; (c) switch only if agreement ≥ 90% and the disagreements are not systematically better/worse on outcomes; (d) keep the ML score in the data for research, drop it as a gate. No threshold is tuned to outcomes.
Risk: the gate currently excludes the very-low-ML tail (≈ low volatility). Parity guarantees no behaviour change at the switch; the benefit is transparency and removing a model that has to be retrained.

## 4. P3 — make the scan inputs correct (spec; each item gets a unit test, owner approves the batch)
1. ATR train/serve skew: make training use true range (or serving High−Low); retrain only if the ML score stays in use.
2. VIX: the scan stamped 9.785 on every row (SPY realised-vol proxy when providers fail). Use a real VIX series (verify FMP/Polygon endpoint) and FLAG when the proxy is used; regime thresholds and VIX_Min_RR read it.
3. Fundamentals units: provider median mixes percent/decimal (ROE, margins, D/E) → normalise per field before taking the median.
4. Partial intraday bar: scans stamped ~11:06 ET use an in-progress bar (VolSurge median 0.18) → compute Volume_Surge from the last COMPLETED session or time-scale it.
5. RS_63d key (`advanced_filters.py:660` reads `RS_63d`, producer returns `rs_63d`) and sector label split (blocklist uses exact strings: "Consumer Staples" vs "Consumer Defensive"): canonicalise labels.
6. Reliability_Score is saturated (2 unique values on 2026-10-06) and RR in Score is constant 2.0 in `compute_rr`: drop from the score or give them real definitions.
Effect on live gates: items 1,3,4,6 change which names pass; because Score has no demonstrated edge the expected P&L change is ≈ 0, but they break the "freeze" → batch deploy only with owner approval and a before/after candidate diff on one saved scan.

## 5. P4 — what actually moves outcomes at this size
Deposit toward $2k (lifts the sub-$2k IB restrictions, makes 2–3 sensibly sized legacy positions affordable, shrinks the fixed-cost drag); confirm Tiered with a fill ≥ $100; keep CoreTrend as the core; adopt forward-tested ideas (F1 congress/insider, F2 LLM picker) only if they pass their pre-registered rules.

## 6. Decisions needed from the owner
1. P1: floor value — 0 (status quo) / 150 (≈ one legacy position at a time) / 270.
2. P2: approve the shadow-run (read-only logging, no behaviour change).
3. P3: approve the batch (which items), after reading the before/after candidate diff.
4. Deposit plan (amount/date), and a ≥ $100 fill to verify Tiered.


---
## UPDATE 2026-10-06 (later) — results of P1 deployment, P2 analysis, and the VIX root cause
**P1 deployed** (owner approved): floor 150 live on the VPS; single shared `effective_min_position_usd` (loop early-stop, DryCycle alert, preflight); adaptive-gate streaks verified unaffected (they only advance when a filter empties the pool).
**P2 result — parity is NOT possible, and the real problem is deeper.** On 80,883 candidate rows from 461 scan snapshots (2025-12 … 2026-10) the best single ATR_Pct threshold reproduces the ML gate (0.40–0.60) only 76% of the time (percentile variant 72%), far below the 90% bar. Reason: the ML score's SCALE moves with every retrain — monthly median ML 0.54–0.59 (Feb–Mar), 0.35 (Apr), 0.40–0.45 (May–Jun), 0.34 (Aug–Oct) — so a fixed 0.40–0.60 window passes 62% of candidates in Feb, 47% in May and 0.7% in Aug. The gate's strictness is silently changing; it carries no stable meaning (and no information beyond ATR: Spearman 0.84; prior audit: gate lift ≈ 1.0).
Decision for the owner (NOT changed): (A) drop the ML gate and rely on the explicit ATR_Pct ≥ 0.03 gate (transparent, stable; pool becomes larger — the bot is capacity-limited anyway), (B) make it scale-stable (pass-rate fixed at the original ≈27%), or (C) keep + monitor. Interim: (C) is in place via the monitor below.
**VIX root cause fixed (commit below):** the scan VIX was a SPY realised-vol proxy (9.8 vs real 15.1) because (1) `get_index_series` stripped the caret (`FMP symbol=VIX` returns [], `^VIX` works), (2) it used FMP `/light` (no OHLC, rejected as "missing columns"), (3) it preferred Polygon, which our plan does not serve for VIX. Fixed all three; the scan now writes `VIX_Source`; regression test `tests/test_vix_source.py`; live check returns 15.06 from FMP. Effect on gates today: none (min-RR tier is 1.5 at VIX 9.8 and 15; it tightens at VIX ≥ 20 — which the proxy would have understated in stress). ML does not use VIX features.
**Monitoring added:** `scripts/scan_quality_report.py` + `.github/workflows/scan_quality.yml` (daily): ML-gate pass-rate (RED if <3% or >60%), VIX source/mismatch vs real VIX, saturated columns, stale scan; history in `data/scan_quality/history.jsonl`; Telegram only when RED flags change. Run on today's scan it already reports VIX_MISMATCH (9.78 vs 15.05) — the scan that produced it ran before the fix.

---
## UPDATE 2026-10-06 (night) — P2 option A turned out bigger than described; replay results (NOT applied)
Owner answered "א" (= option A, drop the ML gate). Before changing a live gate I audited every place the ML scale acts, and found it acts in TWO places, not one:
1. the explicit gate `min_ml_prob/max_ml_prob` (policy.evaluate_static_gates + order_manager pre-filter; risk_manager reuses policy);
2. INSIDE the Score: `ml_delta` (±5 pts; mean −3 to −4 pts since the model's median is 0.34 < 0.5; up to +3 when the scale is high), `ML_GATES_STRONG` multiplicative penalty/bonus and an AUC-tiered boost (tier chosen by AUC 0.5848, only 0.005 above the 0.58 cut).
So dropping only the explicit gate would leave the drifting ML scale shifting every Score by several points (the Score 73–85 window and the Score-ranked pool change with each retrain).
**Replay on 40 saved scans (Jul–Oct 2026) through the REAL static gates** (`scripts/research/ml_decoupling_replay.py`):
| config | mean eligible candidates / scan | scans with ≥ 1 eligible | distinct tickers ever eligible |
|---|---|---|---|
| C0 current (ML gate + ML in score) | 0.33 | 20% | 13 |
| A1 gate off only | 0.65 | 38% | 22 |
| A2 gate off + ML removed from the score | 0.78 | 40% | 30 (11 of them also eligible today) |
Reading: the current system finds an eligible candidate in 1 scan out of 5; dropping the explicit gate doubles it; fully decoupling adds a bit more and broadens the set (30 vs 13 names). It does NOT flood the bot (0.8 per scan; the bot is capacity-limited to ~1 slot anyway) and every other gate (Score window, Fundamental ≥ 45, RR, ATR floor, SignalQuality High, sector, regime) still applies.
Conclusion: option A is safe in magnitude, but it must be done as A2 (gate off AND score decoupled) to actually remove the dependence on the drifting ML scale. Implementation plan (not yet done): flags `TRADE_ML_GATE_ENABLED` (policy + order_manager pre-filter; adaptive-ML relax becomes inert) and `ML_IN_SCORE` (scoring_engine: ml_delta=0, no ML penalty/bonus), both default = current behaviour; tests with parity at defaults and replay at off; env + drift EXPECTED + CLAUDE.md in one commit; rollback = flags back.
Caveat: the "score without ML" replay assumes the later pipeline stage (W6) is additive-neutral; and the Score window 73–85 was calibrated WITH ML inside the score — with ML out, Score levels move up ≈ 3–4 pts on average, so the window then selects a (slightly) different, larger set. Score itself has no demonstrated edge, so no performance claim is made either way.


---
## UPDATE 2026-10-07 — P2-A2 IMPLEMENTED AND DEPLOYED (owner approved "כן תבצע א2 במלואו")
What changed (commit 6c508266, one commit incl. drift EXPECTED + CLAUDE.md + tests):
- **Trade side:** `TRADE_ML_GATE_ENABLED=0` on the VPS (code default stays 1 = legacy) → `policy.evaluate_static_gates` and the `order_manager` pre-filter skip the ML window (risk_manager and the dashboard reuse policy). Adaptive-ML relax becomes inert.
- **Score side (scans on GitHub Actions/Streamlit use the code default):** `ML_IN_DECISIONS=0` (default, `core/scoring_config.py`): no `ml_delta`/`ML_GATES` penalty-bonus in `compute_final_score_20d`, no `ml_delta` in `compute_overall_score`, no ML in `calculate_conviction_score`, no ML bypass of the min-score filter (runner), no ML weight in swing strength (runner + ticker_scoring) and no "High ML breakout probability" reason in `SignalQuality`.
- Kept: ML_20d_Prob computed and stored; the explicit ATR floor; the ranker's 20% ML rank weight (scale-invariant inside the pool); the scan-quality monitor now reports ML pass-rate as INFO.
- Tests: tests/test_ml_decoupling.py (defaults decoupled, flags restore legacy), 3 mechanism tests updated to run with the flag on; full suite 1083 passed.
- Verified on the VPS with the production env: ML 0.20/0.34/0.50/0.90 → no ML failures; ATR floor still rejects ATR 0.01; drift clean; not paused; positions protected.
Watch-list for the next scans (first scan with the new score runs at the next scheduled pipeline): (1) scan completes without errors, (2) Score levels shift up ≈ 3–4 pts on average (ML no longer subtracts), (3) eligible candidates per scan ≈ 0.8 (replay) vs 0.33 before, (4) scan-quality monitor shows VIX_Source=FMP and VIX_Value ≈ real VIX.
Rollback: `TRADE_ML_GATE_ENABLED=1` (VPS .env.trading) + `ML_IN_DECISIONS=1` (scan environment/default in scoring_config) + drift EXPECTED + CLAUDE.md in one commit.
