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

---
## RESULT — development period (2018-06 … 2021-12), run 2026-10-02 — VERDICT: DEAD in development, holdout NOT run
84 decision dates, 182,818 rows, ~2,200 names/date (U1), ~1,350 (U2). Short interest rows 3.25M (210 settlement dates).
| signal | universe | mean IC | t | NW-t | D1−D10 spread / 10d |
|---|---|---|---|---|---|
| SIR | U1 | −0.019 | −1.41 | −1.37 | +0.003% |
| DTC | U1 | −0.003 | −0.40 | −0.49 | −0.196% |
| SIR | U2 | −0.013 | −0.98 | −0.98 | +0.020% |
| DTC | U2 | −0.003 | −0.33 | −0.36 | −0.208% |
Hypothesised sign (negative) holds weakly for SIR, but |t| < 2 in every test, the decile spread is ≈ 0 (D1 and D10 both ≈ −0.11% vs
the universe mean), and the sign flips in 2020 (+0.038). Prereg gate to touch the holdout: right sign AND |t| ≥ 2 in development → NOT met.
The locked holdout (2022-01 … 2026-09) was therefore never evaluated and stays unseen. Conclusion: no usable edge in FINRA short interest
at a 10-day horizon on a tradable universe in this window. (Caveat: 84 dates; a true IC of −0.02 would need ~250 dates to detect.)

---
## AMENDMENT A1 (2026-10-02, owner-authorised in chat: "yes" to running the recent holdout once)
After the development verdict above (DEAD in development), the owner asked that the recent regime (2022-01 … 2026-09) still be evaluated once,
because markets change and an effect could exist only recently. Documented exception: ONE holdout run, K rises from 4 to 8 for multiple-testing purposes
(bar stays |t| ≥ 3.0 on the holdout, plus all other criteria above). Result recorded below whatever it is; no re-run, no re-tuning.

---
## RESULT — LOCKED HOLDOUT (2022-01 … 2026-09), single run 2026-10-02 — VERDICT: ALIVE as a cross-sectional predictor, NOT TRADABLE long-only
112 decision dates, 258,858 rows, ~2,300 names/date (U1).
| signal | universe | mean IC | t | NW-t | D1(low SI) excess | D10(high SI) excess | D1−D10 / 10d |
|---|---|---|---|---|---|---|---|
| SIR | U1 | −0.039 | **−3.17** | −3.07 | +0.04% | **−0.67%** | +0.72% |
| DTC | U1 | −0.016 | −1.97 | −1.73 | −0.36% | −0.21% | −0.15% |
| SIR | U2 | −0.032 | −2.68 | −2.56 | +0.03% | −0.43% | +0.46% |
| DTC | U2 | −0.018 | −2.50 | −2.27 | −0.11% | −0.18% | +0.08% |
SIR/U1 clears the |t| ≥ 3.0 bar with the right sign (K=8 Bonferroni p ≈ 0.012); IC negative in every calendar year 2022-2026 (−0.049, −0.046, −0.050, −0.017, −0.030).
The development period had shown only IC −0.019 (t −1.4): the effect is concentrated in the recent regime (consistent with the owner's regime-change point).
BUT the information sits entirely on the SHORT side: the highest-SI decile underperforms by 0.67%/10d, while the lowest-SI decile earns ≈ 0 excess
(long-only D1 excess +0.04%, t +0.29) → prereg criterion 4 (long-only net excess > 0, CI lower bound > 0) FAILS. A cash account < $2k cannot short.
Usable form: an AVOID filter (exclude top-decile short interest from any long candidate list). Expected gain is small: it only changes outcomes when a
candidate would have come from the top decile.
