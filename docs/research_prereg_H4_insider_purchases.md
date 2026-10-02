# PREREG H4 — Insider open-market purchase clusters / large purchases: 60-day drift (written 2026-10-03, BEFORE any insider data was downloaded)

Protocol: docs/research_protocol_oct2026.md. Split as H1–H3 with amendment A1 (recent holdout always evaluated once):
development = filing dates 2018-06-01…2021-12-31 (diagnosis and DATA-HYGIENE checks only — event counts, field sanity, never return-based rule changes);
LOCKED holdout = 2022-01-01…2026-09-15, evaluated ONCE. Lessons applied from H3: event definitions are verified on development data for correctness
(sample inspection) before the holdout run, and any correction is documented as an amendment with the holdout still unseen.

## Hypothesis
Open-market purchases by company insiders (own money, not awards/exercises) convey positive private information; stocks with a purchase CLUSTER or a LARGE purchase
outperform the market over the next 60 trading days (long horizon: costs ≈ 0.5% round trip are amortised). Expected sign: positive.

## Data and point-in-time rules
- FMP `/stable/insider-trading/search?symbol=X&transactionType=P-Purchase&limit=1000` (per symbol; delisted included; fields `filingDate`, `transactionDate`, `reportingCik`, `typeOfOwner`, `securitiesTransacted`, `price`, `formType`).
- Valid purchase row: transactionType P-Purchase, formType starts with "4", price ≥ $1, securitiesTransacted > 0, transactionDate within 90 days before filingDate (drops date typos), insider type contains "officer" or "director".
- value = securitiesTransacted × price; rows aggregated per (symbol, reportingCik, transactionDate).
- **E1 "cluster":** ≥ 2 DISTINCT insiders (reportingCik) each with aggregated purchase value ≥ $10,000 within a rolling 30-calendar-day window of transactionDates. Event date = the filingDate of the filing that completes the cluster.
- **E2 "large":** a single insider-day aggregated purchase value ≥ $250,000. Event date = its filingDate.
- Event unit = (stock, event date); a stock's event is dropped if the same definition fired for it in the previous 60 trading days.
- **Entry = Open of the NEXT trading day after the event filingDate. Exit = Open 60 trading days later.** Delisted before exit → last available Open.
- Excess = event return − mean 60-day return (same dates) of all U1 names. Panel research_data/univ_px.pkl (5,356 symbols incl. delisted).

## Universe: U1 only (Close ≥ $5, 20-day ADV$ ≥ $1M, marketCap ≥ $300M at the prior close). Tests: exactly 2 (E1, E2), horizon 60d. K = 2.
Statistic: mean excess per event, clustered by calendar month; t across months (+ Newey-West 3 lags). Median also reported.

## Success criteria (all on the LOCKED holdout, else DEAD)
1. Mean excess > 0 and |t| ≥ 3.0 (Bonferroni K=2) for E1 or E2.
2. Positive in both halves of the holdout and in ≥ 70% of calendar years 2022-2026.
3. Net of costs (5/10/25 bp spread by cap band + $0.35/leg on a $270 position + 0.15% slippage), mean excess > 0 with 95% CI lower bound > 0.
4. Capacity: ≥ 4 qualifying events per month on average.
Reported secondarily (NOT a test): split of mean excess by market-cap band ($0.3–2B vs ≥ $2B) to show where any effect lives.
