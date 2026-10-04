# PREREG H6 — Classic 12-1 month cross-sectional momentum, long-only, monthly (written 2026-10-04, BEFORE any momentum run on the panel)

Origin disclosure: this hypothesis was suggested by an EXPLORATORY diagnostic in F2R (a naive "12m−1m return top-5" rule beat the LLM pickers in 9 post-cutoff rounds, +4.3%, t≈1.7). Those 9 rounds (Jul–Sep 2026) overlap the last ~3 months of the holdout below;
that overlap is small (3 of ~56 months) and is disclosed. Earlier audits found momentum ≈ 0 in a PIT top-500 (large caps), so the question here is U1 (cap ≥ $300M, incl. mid/small caps) — a genuinely open question.
Protocol: docs/research_protocol_oct2026.md; split as H1–H5 with amendment A1 (holdout always evaluated once).

## Definition (fixed now)
- Panel: research_data/univ_px.pkl / univ_mcap.pkl (5,356 symbols incl. delisted, daily 2018-06 … 2026-09-15).
- Decision date = last trading day of each month. Signal M = Close[t−21] / Close[t−252] − 1 (return from 12 months ago to 1 month ago; skips the most recent month). Needs ≥ 252 prior days → first decision 2019-06-28.
- Universe U1 at the decision close: Close ≥ $5, 20-day ADV$ ≥ $1M, marketCap ≥ $300M.
- **Portfolio P1 (the single pre-registered test, K = 1):** equal-weight the top DECILE of U1 by M; entry = Open of the next trading day; exit = Open 21 trading days later (delisted before exit → last available Open). Excess = P1 return − equal-weight mean return of all U1 names (same dates).
- Secondary, NOT a test: top 5% (closer to a 3-5 name account), bottom decile, and the top-minus-bottom spread.
- Split: development = decision months 2019-06 … 2021-12 (diagnosis only); LOCKED holdout = 2022-01 … the last month whose exit fits inside the data (decision 2026-07-31), evaluated ONCE.

## Success criteria (all on the holdout, else DEAD)
1. Mean monthly excess > 0 with t ≥ 3.0 (single test; months are non-overlapping; Newey-West lag 1 also reported).
2. Positive in both halves of the holdout and in ≥ 70% of calendar years 2022–2026.
3. Net of costs (spread 5/10/25 bp by cap band + $0.35/leg on $270 + 0.15% slippage, applied to the average pick) mean excess > 0 with 95% CI lower bound > 0.
4. Capacity: ≥ 3 names exist in the top decile every month (trivially true) — and the P1 return is not dominated by one month (report the median and the result excluding the best month).
Reporting: n months, mean, median, t, NW-t, by year, halves, net, hit-rate of months > 0, and the top-5% / bottom-decile / spread diagnostics.

---
## RESULT (run 2026-10-04; holdout run once) — VERDICT: DEAD (positive but tiny, not significant, ≈ 0 net of costs)
| test | period | months | mean excess/month | median | t | NW-t | net of costs (95% CI) |
|---|---|---|---|---|---|---|---|
| P1 top-decile 12-1 momentum | dev 2019-06…2021-12 | 31 | +0.60% | +0.94% | +0.69 | +0.64 | +0.03% [−1.67, +1.73] |
| P1 | **holdout 2022-01…2026-07** | 55 | **+0.46%** | +0.51% | **+0.85** | +0.88 | **−0.10% [−1.17, +0.98]** |
Holdout by year: 2022 +0.73%, 2023 +0.52%, 2024 +0.94%, 2025 +0.41%, 2026 −0.80% (4/5 positive); halves +0.82% / +0.09%; excluding the best month +0.29%; months > 0: 55%.
Secondary (not a test): top 5% +0.55%/month; bottom decile −0.88%; top-minus-bottom spread +1.35%/month (needs a short leg → not tradable in a cash account < $2k).
Bar not met (t ≥ 3.0 needed; net CI includes 0). Same pattern as H1: the information sits mainly on the SHORT side.
Lesson: the exploratory "+4.3%" of the naive momentum rule in the 9 post-cutoff F2R rounds does NOT generalise — over 86 months the same idea earns ≈ +0.5%/month gross and ≈ 0 net. Nine rounds were an outlier window; this is exactly why single short windows (including the LLM-vs-rule comparison in F2R) must not be over-read.
