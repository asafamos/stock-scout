# StockScout — dated checkpoints & open questions (as of 2026-10-29)

One page for "what are we waiting for, until when, and what decides it". Each item names its source of truth. Update it when an item resolves; do not
add new tuning here (the gates are frozen — see CLAUDE.md "FREEZE" and docs/research_prereg_P1_new_score_eval.md).

## A. Evidence about recommendation quality (the open question)
| What | Status 2026-10-29 | Next date | Decided by |
|---|---|---|---|
| **P1 — post-fix Score & eligible set** (docs/research_prereg_P1_new_score_eval.md) | 0 resolved records with scan_date ≥ 2026-10-08 (a record resolves ≈ 20 trading days after its scan) | First outcomes ≈ **2026-11-05**. Evaluator is BLIND until ≥ 25 resolved dates and ≥ 150 eligible obs (earliest ≈ **Dec 2026**); one FINAL look on/after **2027-03-31** | `python scripts/research/p1_eval_new_score.py --outcomes <copy of VPS data/outcomes/scan_outcomes.jsonl>` |
| **F2 — LLM stock picker, forward** (docs/research_prereg_F2_llm_picker.md, A1–A6; AUTOMATED since 2026-10-29, runbook docs/f2_round_runbook.md) | Rounds logged: 10-04, 10-06 (off-schedule), 10-11, 10-18, 10-29 (off-schedule). **Sun 10-25 was MISSED** (no session ran it). | First maturities: round 10-04 ≈ **11-02**, 10-06 ≈ 11-04, 10-11 ≈ 11-09, 10-18 ≈ 11-16, 10-29 ≈ late Nov. Next round **Sun 11-01 14:00 Israel (scheduled task `f2-weekly-round`; Monday 10:00 catch-up; GH `f2_watchdog` alerts Mon 11:30 UTC if missing)**. Verdict only with ≥ 20 valid rounds (26 planned → ≈ Apr 2027) | `python scripts/forward/f2_resolve.py` (prints no verdict before 20 resolved rounds) |
| **F4 — LLM reads earnings-release TEXT, forward, event-level** (docs/research_prereg_F4_earnings_text.md, runbook docs/f4_batch_runbook.md; approved 2026-10-29) | First batch Fri 10-30 (one-off task 14:20 Israel), then weekdays 15:20 Israel (`f4-daily-batch`); GH `f4_watchdog` alerts if a day is missing | Earnings season ends ≈ mid-Nov; stopping rule ≥ 1,000 resolved events over ≥ 30 entry dates (resolve = entry + 20 trading days) → earliest verdict ≈ **mid-Dec 2026/Jan 2027**. Evaluator DONE 2026-10-29 (`scripts/forward/f4_eval.py`, blind until the rule is met, prereg A2) | `python scripts/forward/f4_eval.py` (refuses before the stopping rule) |
| **Shadow arms** (S1 fund-first, S2 vol+size, canary vs legacy exits; docs/shadow_selector_prereg.md) | Collecting daily (VPS timer stockscout-shadow) | ≥ 60 resolved scan dates ≈ **mid-Jan 2027** | `data/outcomes/shadow_report.txt` on the VPS — do NOT read before the date (blind) |
| **F1 — congress/insider forward test** (docs/research_prereg_F1_forward_test.md) | Collecting (GH Action forward_collect, 22:30 UTC weekdays) | Resolver not written yet; needed ≈ **mid-2027** (n ≥ 300 qualifying events may not be reached by 2027-12-31) | TODO: write `f1_resolve.py` before ≈ 2027-05 |
| **Real closes under the restored config** ("n ≥ 20 post-restore closes", window 2026-10-15 → 11-30) | n = 0 closes since 2026-09-14 (HNGE/PACS/PATH still open). With 3 slots, a $150 minimum notional and CoreTrend holding ≈ 75 %, ≈ 1 legacy position at a time → n = 20 would take years. | **This criterion will not be met in the window.** P1 (hundreds of resolved candidate observations) is the realistic substitute | — |

## B. Operations & trading (dated)
| What | Date | Note |
|---|---|---|
| **CoreTrend month-end decision** | Fri **2026-10-30** (alert timer 17:50 ET); execution Mon **2026-11-02** 09:45 ET, retry 11:45 ET | Signal as of 09-30: RISK_ON (QQQ 739.77 vs SMA10m 670.76, +10.3 %). What-if on 10-28 data (QQQ 725.61): SMA10m would be 681.89, gap +6.4 %; QQQ must CLOSE 10-30 below **681.89** (≈ 6.0 % under the 10-28 close) to flip to RISK_OFF (SELL QQQM via TRAIL-modify, IEF buy on a later run). |
| **HNGE earnings** | **2026-11-02** | Trail tightened 15.1 % → 3.0 % by the earnings-warn mechanism (position ≈ +17 %); PACS earnings 2026-11-09 (will tighten the same way ≈ 5 days before). |
| **Overnight-scan degradation — root cause** | tonight ≈ **23:53 UTC** (run that straddles UTC midnight) | The integrity guard blocks the bad scan (it did on 10-28/29). Read the `[HISTORY]` / `[HISTORY2]` lines in that run's log: they were silent on 10-28, so downloads "worked" by count; the loss happens later. Hypothesis "yfinance next-day bar" was tested and rejected. |
| **IB weekly 2FA** | recurs every Monday ≈ 04:00 UTC (also 10-14, 10-17) | API is blind until the push is approved (10-19: 04:00 → 13:48 UTC). IB-side TRAILs keep protecting. Owner decision pending: a fixed-time reminder. |

## B2. Legacy channel — pre-committed rule (P1 addendum A1, 2026-10-29)
Keep it running untouched until a P1 verdict. DEMONSTRATED EDGE → keep (sizing separate); PARTIAL → owner decides; INCONCLUSIVE after the final look (≥ 2027-03-31) → CoreTrend only (no new legacy buys, open positions exit on trails); NO DEMONSTRATED EDGE → stop new legacy buys, positions exit on trails. Table is in docs/research_prereg_P1_new_score_eval.md "Addendum A1".
Cash mechanics (checked in code 2026-10-29): CoreTrend `plan()` tops up QQQM daily (09:45 ET) whenever `min(100% NetLiq − held value, cash − $25)` ≥ one QQQM share (≈ $293), i.e. free cash ≳ $318. Legacy buys need a free slot AND `qty × price ≥ $150`. So HNGE's proceeds (≈ $116 → cash ≈ $160) do NOT flow to CoreTrend; they wait for a legacy candidate ≥ $150. Even with HNGE+PACS+PATH all closed, cash ≈ $225 < $318. A deposit of ≳ $275 goes to CoreTrend at the next 09:45 run (the legacy channel has no free slot to compete).

## C. Owner decisions / actions still open
- Deposit toward **$2k** (unlocks standalone sells, LMT targets, pyramiding, Day-N Kill).
- **IB pricing plan — CONFIRMED FIXED (ledger check 2026-10-29)**: all 56 fills match the Fixed model ($1.00 minimum per order, 1 % cap, ≈ $0.005–0.01 regulatory fee on sells); total $50.60. A Tiered-base model ($0.35 min, $0.0035/share) would have cost ≈ $18.9 (saving ≈ $31.7 in 5 months, before pass-through fees). Owner action: switch the pricing structure in the IB Account Management (Tiered); then verify with a ≥ $100 fill (expect ≈ $0.35 + small fees). Lite ($0 commission) is probably not usable with API trading — not verified.
- Review IBKR account activity and rotate the password (security exposure of 2026-09-29).
- `Communication` vs `Communication Services` blocking policy (code deliberately does not unify them).
- V2 sleeve (paused 2026-09-30): stays paused unless the owner decides otherwise.
- Slippage: post-2026-09-29 legacy buys filled +1.2 % / +1.5 % / +0.2 % vs the same-day scan price (n = 3, includes intraday drift) — too few to act on; lifetime figure in the Telegram status (+1.54 %, 48 buys) mixes eras. Re-check at n ≥ 10 post-fix buys.

## D. Housekeeping done recently (for context)
Score cap 85 → 99 (scale shift), ML out of decisions, real RR in the Score, RS_63d (2 root causes), D/E ratio, sector synonyms, VIX source, notional floor, scan-price guard (integrity check + monitor), CoreTrend retry, healthcheck false-alarm fix, pre-pipeline cash warning removed, 19 old failing tests → suite green (1129 passed).
