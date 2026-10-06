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
