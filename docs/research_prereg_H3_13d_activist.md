# PREREG H3 — Initial Schedule 13D filings (activist / active-intent >5% holders): post-filing drift (written 2026-10-02, BEFORE any 13D/13G data was downloaded or analysed)

Protocol: docs/research_protocol_oct2026.md. Split as H1/H2 with amendment A1 (recent holdout always evaluated once):
development = filing dates 2018-06-01…2021-12-31 (diagnosis only); LOCKED holdout = 2022-01-01…2026-09-15, evaluated ONCE.
Lesson from H2: at our size a round trip costs ≈ 0.5%, so horizons are LONGER (20 and 60 trading days) where costs are amortised and per-trade effects can be larger.

## Hypothesis
After an initial 13D filing (a >5% holder declaring active intent) the stock outperforms the market over the next 20 and 60 trading days (the filing-day jump is
already gone; we ask whether a further drift remains). Expected sign: positive.

## Data and point-in-time rules
- FMP `/stable/acquisition-of-beneficial-ownership?symbol=X` (per symbol, all history; delisted included — SIVB verified): `filingDate`, `acceptedDate` (day only),
  `percentOfClass`, `nameOfReportingPerson`, `typeOfReportingPerson`, `url`. No form-type field.
- **Classification rule (fixed now):** url lower-cased. 13D initial = contains `13d` AND does NOT contain `13da`, `13d/a`, `_13d_a`, or `amend`. 13G/other = ignored. Rows with neither `13d` nor `13g` ignored.
- Event unit = (stock, filingDate); several reporting persons the same day = one event. An event is dropped if the same stock had a 13D-initial event in the previous 60 trading days.
- **Entry = Open of the NEXT trading day after `filingDate`.** Exit = Open h trading days after entry, h ∈ {20, 60}. Delisted before exit → last available Open.
- Excess return = event return − mean h-day return (same entry/exit dates) of all names passing U1 that day. Panel: research_data/univ_px.pkl (5,356 symbols incl. delisted).

## Universe: U1 only (Close ≥ $5, 20-day ADV$ ≥ $1M, marketCap ≥ $300M at the prior close). No other universe will be tested.
## Tests: exactly 2 (K = 2): h=20 and h=60. Statistic: mean excess per event, clustered by calendar month (events are sparse); t across months (+ Newey-West 3 lags).

## Success criteria (all on the LOCKED holdout, else DEAD)
1. Mean excess > 0 with |t| ≥ 3.0 (K=2 Bonferroni) for h=20 or h=60.
2. Positive in both halves of the holdout and in ≥ 70% of calendar years 2022-2026.
3. Net of costs (spread 5/10/25 bp by cap band + $0.35/leg on a $270 position + 0.15% slippage) mean excess > 0 with 95% CI lower bound > 0.
4. Capacity: ≥ 4 qualifying events per month on average (a 3-position account needs roughly one per week).
Also reported: events per year, by-year means, median (not only mean) to flag outlier-driven results.

---
## AMENDMENT A2 (2026-10-02) — classification rule corrected AFTER the development run exposed it as defective (holdout still UNSEEN)
What happened: the first development run (rule above) gave mean excess −1.90% (20d, t −1.98) and −3.25% (60d) — opposite to the hypothesis — with 47,606 "13D-initial" rows.
Inspecting URL samples (a data-hygiene check, not a return-based search) showed the substring rule is wrong: (1) `d884513dsc13g.htm` is a 13G but contains "13d" inside the accession-derived
number ("45**13d**sc"); (2) since Dec-2024 filings use `xslschedule_13d_x01/primary_doc.xml` for BOTH initial 13D and 13D/A, so the URL cannot separate them.
Corrected definition (fixed before looking at any holdout number; chosen for correctness, not for results):
- 13D-type row = url (lower-cased) matches regex `(?<!\d)13d` (a "13d" not preceded by a digit). 13G rows are never events.
- Event = the FIRST 13D-type row ever seen for the pair (symbol, nameOfReportingPerson normalised) in the full downloaded history (1996+); later rows by the same person for the same stock are treated as amendments. This also handles the XML era.
- Then unit/dedupe/entry/exit/universe/horizons/criteria exactly as above. K stays 2, but it is disclosed that the development period was inspected twice (old and new definition).
Residual limitation: old `.txt` filings (pre-2013) cannot be classified from the URL, so a stock's earlier 13D by the same person may be missed; those few later filings could be mislabeled "first". Reported, not fixed.
