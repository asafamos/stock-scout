# PREREG H2 — Analyst upgrades / downgrades (post-event drift) (written 2026-10-02, BEFORE any grades data was downloaded or analysed)

Protocol: docs/research_protocol_oct2026.md. Same split as H1 (data-limited, price panel from 2018-06):
- **Development:** event dates 2018-06-01 … 2021-12-31. **LOCKED holdout:** 2022-01-01 … 2026-09-15, evaluated ONCE, only if development passes.

## Hypothesis
After an analyst **upgrade** the stock outperforms the market over the next 10 trading days; after a **downgrade** it underperforms
(post-announcement drift). Expected signs: upgrade > 0, downgrade < 0. (Literature: drift is real but has shrunk; larger in small/mid caps.)

## Data and point-in-time rules
- FMP `/stable/grades?symbol=X`: one row per rating action with `date` (day-level only), `gradingCompany`, `previousGrade`, `newGrade`,
  `action` ∈ {upgrade, downgrade, maintain}. Includes delisted names (SIVB verified). Only `upgrade`/`downgrade` rows are events.
- Timing is unknown inside the day → **entry = Open of the NEXT trading day after the event date** (conservative: gives up the
  announcement-day reaction). Exit = Open 10 trading days after entry.
- Event unit = (stock, trading day): net = #upgrades − #downgrades that day; net>0 → Up event, net<0 → Down event, 0 → skipped.
  A stock's event is dropped if the same direction already occurred for that stock in the previous 10 trading days (no overlap).
- Panel: research_data/univ_px.pkl (5,356 symbols incl. delisted, 2018-06+). Delisted before exit → last available Open.
- Excess return = event return − mean 10-day return (same entry/exit dates) of ALL names passing U1 on that day.

## Universes (filters computed at the event's decision day, only from past data)
- U1: Close ≥ $5, 20-day ADV$ ≥ $1M, marketCap ≥ $300M. U2: U1 and marketCap ≥ $2B.

## Tests (exactly 4; K = 4): {Up, Down} × {U1, U2}. Single horizon (10 trading days). No other horizons, thresholds, or grade-level sub-splits will be tried.
Statistic: average excess return per event-DATE (cluster by date), t-stat across dates (+ Newey-West 2 lags). Events with < 3 per date pooled by month.

## Success criteria (all on the LOCKED holdout, else DEAD)
1. Right sign and |t| ≥ 3.0 for Up (long-only tradable form) in U1 or U2 (BH-adjusted over K=4).
2. Same sign in both halves of the holdout (2022-01…2024-05, 2024-06…2026-09) and in ≥ 70% of calendar years.
3. Net of costs (spread 5/10/25 bp by cap band + Tiered $0.35/leg on a $270 position ≈ 0.26% RT + 0.15% slippage) mean excess per Up event > 0 with 95% CI lower bound > 0.
4. Capacity: ≥ 3 qualifying Up events per week on average at our size filters.
Gate to open the holdout: development shows the right sign AND |t| ≥ 2 for that test.

---
## AMENDMENT A1 (2026-10-02, made BEFORE any H2 grades data was analysed; prompted by owner's point that markets change)
The "gate to open the holdout" sentence above is REPLACED: the LOCKED recent holdout (2022-01 … 2026-09) is evaluated ONCE for all 4 tests
regardless of what development shows (an effect that only exists in the recent regime must not be missed). Development (2018-06…2021-12)
is reported for stability/diagnosis only. The success criteria on the holdout are unchanged, and the number of tests stays K = 4.
Also reported: mean excess by calendar year, and by half-years, to show whether any effect is decaying. This amendment applies to H2 onward;
for H1 (already analysed in development under the old gate) a single holdout run is a documented exception that the owner may authorise (K for H1 rises to 8).

---
## RESULT (run 2026-10-02) — VERDICT: DEAD (does not clear the bar; effect small and decaying; negative after costs)
Data: 104,820 upgrade/downgrade rows, 3,988 symbols; after netting/dedupe 24,595 dev events and 24,720 holdout events. Per-event-date clustering.
| test | period | events | mean excess /10d | t | NW-t | net of costs (CI) |
|---|---|---|---|---|---|---|
| Up U1 | dev 2018-21 | 9,322 | +0.241% | +2.61 | +2.54 | −0.276% [−0.457, −0.095] |
| Up U1 | **holdout 2022-26** | 10,903 | **+0.189%** | **+1.99** | +1.89 | **−0.319% [−0.504, −0.134]** |
| Up U2 | holdout | 9,334 | +0.068% | +0.73 | +0.70 | −0.413% [−0.596, −0.230] |
| Down U1 | holdout | 11,532 | +0.061% (wrong sign) | +0.58 | +0.53 | short side −0.581% |
| Down U2 | holdout | 9,167 | +0.056% (wrong sign) | +0.51 | +0.50 | short side −0.539% |
Holdout Up U1 by year: 2022 +0.77%, 2023 −0.04%, 2024 +0.14%, 2025 +0.07%, 2026 −0.10% → the drift is concentrated in 2022 and has faded to ≈ 0; halves +0.31% / +0.06%.
Bar not met: |t| = 1.99 < 3.0; second-half effect ≈ 0; net-of-cost CI entirely below 0 (round-trip cost ≈ 0.5% at our size exceeds the ≈ 0.2% gross drift). Capacity is ample (~45 Up events/week) but irrelevant.
Downgrades show no drift (wrong sign, t ≈ 0.5). Conclusion: analyst rating changes carry no tradable post-event drift at a 10-day horizon at our size in 2018-2026.
