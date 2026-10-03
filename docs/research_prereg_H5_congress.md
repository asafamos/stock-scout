# PREREG H5 — Congressional stock purchases (Senate + House): post-disclosure drift (written 2026-10-03, BEFORE any congress-trade data was downloaded)

Protocol: docs/research_protocol_oct2026.md. Split as H1–H4 with amendment A1: development = disclosure dates 2018-06-01…2021-12-31 (diagnosis + data-hygiene checks only);
LOCKED holdout = 2022-01-01…2026-09-15, evaluated ONCE. Event definitions are checked on development data by sample inspection BEFORE the holdout run; any correction is an amendment made with the holdout unseen.

## Hypothesis
Members of Congress (or their spouses/dependents) may trade on better-than-public information; stocks they BUY outperform the market after the trade becomes public. Expected sign: positive.
Disclosure lag is long (median ≈ 28 days, p90 ≫ 100), so only the post-DISCLOSURE drift can be traded; much of any edge may already be gone — that is exactly what we test.

## Data and point-in-time rules
- FMP `/stable/senate-trades?symbol=X` and `/stable/house-trades?symbol=X`, per symbol, **paginated** (`limit=250&page=0,1,2…` until empty; a single call truncates busy tickers — verified for AAPL: 559 House rows over 3 pages).
- Fields used: `disclosureDate` (the only usable timestamp), `transactionDate`, `type`, `amount` (range string), `assetType`.
- Valid event row: assetType "Stock" (or empty), type contains "Purchase", amount lower bound ≥ $15,001 (parsed from the range), disclosureDate ≥ transactionDate and ≤ transactionDate + 400 days (drops typos).
- Event unit = (stock, disclosureDate) across both chambers; a stock's event is dropped if one fired for it in the previous 60 trading days.
- **Entry = Open of the NEXT trading day after `disclosureDate`.** Exit = Open h trading days later, h ∈ {20, 60}. Delisted before exit → last available Open.
- Excess = event return − mean h-day return (same dates) of all U1 names. Panel research_data/univ_px.pkl (5,356 symbols incl. delisted).

## Universe: U1 only (Close ≥ $5, 20-day ADV$ ≥ $1M, marketCap ≥ $300M, prior close). Tests: exactly 2 (h=20, h=60), K = 2.
Statistic: mean excess per event clustered by calendar month; t across months (+ Newey-West 3 lags). Median reported.

## Success criteria (all on the LOCKED holdout, else DEAD)
1. Mean excess > 0 and |t| ≥ 3.0 (K=2) for h=20 or h=60. 2. Positive in both halves and ≥ 70% of calendar years 2022-2026.
3. Net of costs (5/10/25 bp by cap band + $0.35/leg on $270 + 0.15% slippage) mean excess > 0, 95% CI lower bound > 0. 4. ≥ 4 qualifying events per month.
Secondary, NOT a test: split by chamber (Senate vs House) and by market-cap band, to show where any effect lives.
