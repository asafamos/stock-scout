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
