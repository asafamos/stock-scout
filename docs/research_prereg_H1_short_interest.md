# PREREG H1 — FINRA short interest as a cross-sectional predictor (written 2026-10-02, BEFORE any short-interest data was downloaded or analysed)

Protocol: docs/research_protocol_oct2026.md. Amendment forced by data: price panel + short interest only exist from 2018, so the
"development ≤ 2015" split is impossible. Replacement split, fixed now:
- **Development:** decision dates 2018-06-01 … 2021-12-31 (includes the 2020 crash and the 2021 retail squeeze).
- **LOCKED holdout:** decision dates 2022-01-01 … 2026-09-15 (includes the 2022 bear market). Evaluated ONCE, only if development passes.

## Hypothesis
Stocks with high short interest (as % of shares outstanding) underperform over the following 10 trading days; low short interest
outperforms. Hypothesised sign of IC(SIR, forward return): **negative**. (Literature: informed shorting; effect has shrunk in
large caps and may be strongest in small/mid caps.)

## Data and point-in-time rules
- Short interest: Polygon `/stocks/v1/short-interest`, fields `settlement_date, short_interest, avg_daily_volume, days_to_cover`.
- **Publication lag imposed: 9 business days after `settlement_date`** (FINRA publishes ~7–8 business days later; 9 is conservative).
  A value is usable only from that date. Decision date = first trading day on/after the lag; **entry = next trading day's Open**.
- Shares outstanding ≈ marketCap / Close on the settlement date (from the PIT panel `univ_mcap.pkl`, `univ_px.pkl`).
- Survivorship-free: panel includes delisted names (5,356 symbols). Delisted before exit date → exit at last available Open
  (no -100% assumption beyond what price shows; bankruptcies are flagged and counted separately).
- Ticker reuse hazard: restrict to symbols whose panel price/mcap history is continuous over the holding window.

## Signals (exactly two; no others will be tried)
- **S1 = SIR = short_interest / shares_outstanding** (primary).
- **S2 = days_to_cover** (secondary).
Direction pre-registered: higher → worse. No change-in-SI variant, no thresholds tuned.

## Universe (decision-date filters, computed from data available at that date)
- U1 "tradable": Close ≥ $5, 20-day ADV$ ≥ $1M, marketCap ≥ $300M.
- U2 "large": U1 and marketCap ≥ $2B.
Total tests K = 2 signals × 2 universes = 4. (Bonferroni-adjusted p reported with K=4.)

## Horizon and measurement
- Forward return: Open(t+1) → Open(t+11) (10 trading days), minus the equal-weight universe mean return on the same dates.
- Primary statistic: per-decision-date Spearman rank IC; t-stat from the time-series of ICs (decision dates are ~10–11 trading days
  apart, i.e. approximately non-overlapping; Newey-West with 2 lags also reported).
- Secondary: decile spread (D1 low SIR minus D10 high SIR) and **long-only** D1-vs-universe excess (what we can actually trade).
- Net of costs: round trip = spread (5 bp ≥$10B; 10 bp $2–10B; 25 bp $300M–2B) + Tiered commission $0.35/leg on a $270 position
  (≈0.26% round trip) + 0.15% slippage.

## Success criteria (ALL must hold on the locked holdout; otherwise DEAD)
1. |t(IC)| ≥ 3.0 with the hypothesised sign, for S1 in U1 **or** U2 (the better is NOT cherry-picked: both reported, BH-adjusted).
2. Same sign of mean IC in both halves of the holdout (2022-01…2024-05 and 2024-06…2026-09).
3. Same sign in ≥ 70% of calendar years 2022–2026.
4. Long-only D1-minus-universe excess, net of the costs above, > 0 with 95% CI lower bound > 0.
5. Candidate list non-empty on ≥ 80% of weeks at our size (≥3 names in D1 passing U1).

## Reporting
Development results first (IC, t, deciles). If development shows IC with the right sign and |t| ≥ 2, run the holdout ONCE and report
the verdict ALIVE / DEAD / NOT TRADABLE with all numbers. No re-tuning after seeing the holdout.
