# Research protocol — hunting for a REAL edge (fixed 2026-10-02, BEFORE looking at any new data)

Why this exists: in the last year every "edge" that looked good in-sample (ML score, vol/size tilt, cohort vetoes,
fundamental-first selector) evaporated out-of-sample or once survivorship was removed. This protocol is the gate any
new signal must pass before ONE live dollar depends on it. It is written first so it cannot be bent to fit results.

## 0. Hard rules
1. **Point-in-time or it doesn't count.** Every feature must carry a real as-of timestamp (announcement / filing /
   record date) and be used only after it. Restated snapshots (today's fundamentals, today's constituents) are rejected.
2. **Survivorship-free universe.** Include delisted names (research_data/univ_*.pkl: 5,129 PIT symbols incl. delisted).
   Delisting return = last price (or -100% if bankruptcy flag) — never dropped.
3. **Pre-registration.** Each hypothesis is written (signal, direction, horizon, universe, cost model) in
   `docs/research_prereg_*.md` BEFORE running. Number of variants tried is recorded and used in the correction below.
4. **Time split fixed in advance.** Development ≤ 2015-12-31; a LOCKED holdout 2016-2026 is evaluated ONCE per
   hypothesis. If it fails, the hypothesis is dead — no re-tuning on the holdout. (The 1971-2004 index data is a
   second, older holdout for index-level rules only.)
5. **Multiple-testing discipline.** t-stats use non-overlapping windows and cluster by date. With K hypotheses tried,
   the bar is |t| ≥ 3.0 (not 2.0) on the holdout AND the same sign on both halves of the holdout AND in ≥ 70% of
   calendar years. Bonferroni/BH-adjusted p is reported.
6. **Costs are part of the signal.** Net of Tiered commissions ($0.35 min, ~$0.0035/sh) + spread by market-cap band +
   our realistic slippage (≥0.3%), at OUR account size ($800 now; $2k/$5k scenarios). Gross-only results do not count.
7. **Tradability filter.** Universe limited to what the account can actually trade: price ≥ $5, ADV$ ≥ $1M,
   whole shares, ≤ 3 positions. A signal that only works in untradable micro-caps is recorded as "not tradable".
8. **Benchmark = SPY total return, same period, same costs.** Success = net excess return > 0 with CI lower bound > 0
   on the locked holdout, not "positive return".
9. **Capacity check:** at the signal's holding period and our position count, does a trade list actually exist most
   weeks (not 0 candidates)?

## 1. Pipeline
A. Data audit (what each of the 10 providers truly gives, PIT/history/survivorship) → `research_data/data_audit/`.
B. Hypothesis list ranked by (new information content) × (PIT quality) × (history depth) × (tradability).
C. For each hypothesis in order: prereg → dev-period test → (only if dev passes) one locked-holdout run → verdict.
D. Survivors → shadow-trade live (no money) for ≥ 60 trading days via the existing shadow selector.
E. Only then a small live sleeve with the existing safety rails (kill file, loss self-stop) — owner decides.

## 2. What is already dead (do not retest without a new data source)
Score/ML as ranker (IC≈0), price momentum/reversal in large caps, basic estimate revisions, basic insider buys, basic PIT
fundamental ratios, vol/size tilt (survivorship mirage), cohort vetoes (failed OOS). Small-cap near-52w-high and PEAD
exist (t≈4-5) but are not net-tradable at our size — they may be revisited ONLY with a cheaper execution path.

## 3. Reporting
Every result is reported with: n, dates, dev vs holdout, gross vs net, t (clustered), adjusted p, #variants tried,
and a plain verdict (ALIVE / DEAD / NOT TRADABLE). Negative results are logged too.
