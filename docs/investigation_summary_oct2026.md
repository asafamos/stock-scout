# Investigation summary — what is real, what is dead, what is running (2026-10-02 … 2026-10-04)

Account: ~$817 NetLiq (IBKR, cash account, sub-$2k tier). Since 2026-04-12: roughly −16% vs SPY +12.5%.
Method for everything below: docs/research_protocol_oct2026.md (point-in-time data, survivorship-free universe of 5,356 symbols incl. delisted, pre-registered hypotheses,
development 2018-21 / locked recent holdout 2022-26 run once, |t| ≥ 3 bar, net of costs at our size).

## 1. Where the money went
- Costs ≈ half of the loss (≈ 0.5% round trip at our size: $1 min commission per leg on Fixed pricing, spread, slippage). Tiered pricing was switched on; first fill after the switch will confirm (expect $0.35–0.50 vs $1.00).
- Selection ≈ zero (−0.64pp ± 1.35 vs −0.71 random-entry baseline). Exit rules ≤ 0.4pp, not significant. (From the stage 7-10 audit; ledger-level attribution not run on the VPS ledger.)

## 2. The stock-picking pipeline has no demonstrated edge
- Score IC ≈ 0; ~63% of Score variance is one input (TechScore); RR constant 2.0 in `compute_rr`; Reliability saturated (stage 1-6 audits).
- ML = volatility proxy (Spearman 0.83 with ATR); trained on survivors (IC +0.048 on survivors vs +0.016, t 0.8, on a PIT top-500); train/serve ATR skew (High−Low vs true range); synthetic "VIX" when sources fail.
- 9 of 12 gates remove winners and losers about equally; the full gate stack passes ~1 in 250 rows (median 2 names/date) → top-3 selection is not distinguishable from random.
- The "402 real closes" behind every frozen gate are PAPER-tracked positions (all rows shares = 100), not fills.
- Verdict: not fixable by tuning. Bugs found are worth fixing for honesty (several fixed 2026-10-04) but would not create alpha.

## 3. New-information hunt (10 data sources audited; free tiers ≈ what everyone has)
| hypothesis | result | tradable for us? |
|---|---|---|
| H1 FINRA short interest (10d) | real signal in the recent regime (SIR IC −0.039, t −3.2, negative every year 2022-26); ALL on the short side (top decile −0.67%/10d, bottom decile ≈ 0) | No — cannot short in a cash account < $2k. Usable only as an avoid-filter. |
| H2 analyst upgrades/downgrades (10d) | +0.19% holdout (t 2.0), decaying to ≈ 0 after 2022; net of costs −0.32% | No |
| H3 initial 13D filings (20/60d) | negative point estimates (−1.2% / −1.4%), not significant | No |
| H4 insider purchase clusters / large (60d) | +0.64% (t 1.1), second half ≈ 0, net ≈ 0; exploratory small-cap band ≈ +1.6% in both periods (not a test) | Not confirmed |
| H5 congress purchases (20/60d) | +0.66% / +1.15% (t 2.3 / 2.0), positive in 5/5 years at 60d, mostly ≥ $2B names; below the |t| ≥ 3 bar, net CI includes 0; dev period ≈ 0 | Not confirmed — best candidate → prospective test |
Lesson: effects that exist are small and decaying and sit near our ≈ 0.5% round-trip cost. Premium data (options, transcripts, 13F, estimate vintages) is not on our plans.

## 4. What the evidence supports
- CoreTrend (QQQM while QQQ > 10-month SMA, else IEF): a RISK REDUCER, not a return enhancer. On Nasdaq/S&P 1971-2004 (out of sample) it cut max drawdown ~40% (−47% vs −75% Nasdaq) and won 2000-02, but CAGR was 1-3pts below buy&hold with cash at 0%.
  On QQQ 2008-2026 (total return): rule 13.5% CAGR / −22.6% DD vs QQQ buy&hold 17.5% / −44.8%. IEF vs T-bills as the risk-off asset is a wash → keep IEF.
- It does not "beat SPY" on the evidence; it trades ~4pts of CAGR for half the drawdown.

## 5. What is running now
- CoreTrend LIVE (2 QQQM + 30% TRAIL; next decision on the last trading day of October). Legacy channel still live under frozen gates (owner decision: do not pause). v2 sleeve paused.
- Prospective forward test F1 (docs/research_prereg_F1_forward_test.md): GitHub Action collects congress + insider purchases daily with first-seen timestamps; decision rules fixed in advance (congress n ≥ 300, small-cap insider clusters n ≥ 150). No trading impact.
- Fixes deployed 2026-10-04: ignore-list across monitor/healthcheck/reconcile/daily-loss breaker, /perf key, ledger ingest with empty tracker, rejected-TRAIL "recovered" message, no live buys outside the regular session.

## 6. Decisions that are the owner's
1. Deposit toward $2k (lifts the sub-$2k order restrictions; shrinks fixed-cost drag).
2. Whether to keep the legacy channel running (currently yes). Evidence says expected return ≈ −costs.
3. OpenAI credit ($10–20) to test LLM scoring of 8-K/10-K text (the only deep text corpus available); prior is modest.
4. IBKR: review account activity and rotate the password (API/VNC ports were exposed until 2026-09-29; no sign of intrusion).
5. Whether to adopt a small capped sleeve if F1 passes (not before ~Aug 2027).

## 7. Open technical items
- Daily-loss breaker measures lifetime unrealized, not today's change (conservative, not fixed).
- Vol-based position sizing never scales (ATR unit mismatch); fixing changes sizes = tuning, left alone.
- `resolve_forward.py` (computes F1 returns) not written yet; needed before the first F1 evaluation.
- Verify the first Tiered-priced commission.
