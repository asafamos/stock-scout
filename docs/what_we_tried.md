# What we already tried (one page, as of 2026-10-29) — read this BEFORE proposing a new idea

Sources: docs/research_prereg_*.md (each has its RESULT section), docs/research_protocol_oct2026.md, scripts/research/README.md,
research_data/beat_spy_search.csv, research_data/capital_scenarios.csv, research_data/small_cap_signals.log. Protocol: point-in-time universe incl.
delisted names, locked holdout evaluated once, |t| >= 3, net of costs at our account size.

## A. Stock-selection signals — none cleared the bar
| Idea | Result |
|---|---|
| Live Score / ML as ranker | Rank-IC ~0 (0.018 on 18.9k resolved candidates; old score 20d mean ~0%). ML = ATR/volatility proxy (IC vs ATR 0.84), removed from decisions 2026-10-07 |
| Cohort veto, vol/size tilt (S3 / V2 sleeve), fundamentals-first selector | Failed out-of-sample or was a survivorship mirage (S3 no better than random: -0.4% vs -0.3% CAGR, SPY +15.8%) |
| H1 short interest | DEAD (dev), recent regime also checked |
| H2 analyst grade changes | DEAD (small, decaying, negative after costs) |
| H3 13D activist stakes | DEAD (no positive drift) |
| H4 insider purchases / clusters | DEAD vs the bar (small, decaying, ~0 net) |
| H5 congress purchases | Closest: holdout t 2.0-2.3, net CI includes 0 -> not confirmed; F1 forward test collecting |
| H6 12-1 momentum (top decile, U1) | DEAD: +0.46%/mo gross, -0.10% net, t 0.85 |
| Small-cap near-52w-high, PEAD / earnings surprise | Rank-IC t 3-5 but top-decile net vs SPY negative in every band -> not tradable at our size |
| LLM stock pickers (F2R retro 18 files, F3 36 files) | "NO SIGN OF SKILL"; F2 forward test running weekly (verdict ~Apr 2027) |

## B. Exits
Tight trails / short horizons lose; only wide stops / long horizons beat the legacy exit (HOLD60 +1.57pp, ATR trail +1.04pp on random PIT entries). The
live atr_wide canary is only +0.28pp on the unbiased sample (CI includes 0). Break-even stop and Day-N kill: net negative / disabled.

## C. Allocation and index rules (beat_spy_search, 20 strategies, IS 2005-15 vs OOS 2016-26)
Only QQQ-based rules beat SPY out of sample. Diversified trend / momentum / factor rules did NOT beat SPY OOS: GEM dual momentum (9.5% vs 14.8%),
SPY 6/10/12-month trend, sector momentum top-3, 60/40, vol-target, RSP, IWM, QUAL, USMV (MTUM beat on CAGR only, not Sharpe). QQQ buy&hold beat QQQ 10m-trend
in CAGR in both halves (19.9% vs 15.2% OOS); the trend rule buys a smaller drawdown (-22.6% vs -32.6%), not more return. CoreTrend = that rule; edge over SPY small
(15.2% vs 14.8% OOS), moderate confidence, one mega-cap-growth bet. Also checked: long index history 1971-2004 (coretrend_longhistory.py) and the risk-off
asset (IEF vs BIL vs cash).

## D. Costs at our size (capital_scenarios.csv, K=5 positions, $821)
Active trading cost 13.2%/yr on Fixed, 9.4% Tiered, 3.8% Lite; random portfolios swing from -12.7% to +17.7% CAGR depending only on plan and luck. Legacy ledger: $50.6 commissions
in 5 months (~5% of capital). Mitigations live: $150 notional floor. Open: verify Tiered with a >=$100 fill.

## E. Process lessons
Gate flip-flops (MAX_SCORE 4x in a week), silent env drift (fixed with drift detector), 8-day stale scan, dead drawdown gate, adaptive "relax after N dry cycles" gates
that force trades when nothing qualifies. Rule since 2026-10-02: preregister -> dev -> one holdout run -> shadow 60d -> live sleeve. Frozen gates until P1.

## F. Still running (the only open evidence)
P1 (Score + eligible set, first outcomes ~11-05, earliest verdict Dec 2026), F2 (LLM picker, weekly, ~Apr 2027), shadow arms (~mid-Jan 2027), F1 (congress/insider, 2027).
