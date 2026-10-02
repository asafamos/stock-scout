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
