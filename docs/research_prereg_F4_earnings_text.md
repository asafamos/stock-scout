# PREREG F4 — Does an LLM reading the TEXT of earnings releases predict drift that numbers alone do not? (APPROVED by the owner 2026-10-29, committed before the first event is collected; every event it uses occurs AFTER this commit)

Why this and not another numeric signal: H1–H6 and the small-cap/PEAD scan (docs/what_we_tried.md) show that public numeric signals are priced or untradable at our size. The one information channel not yet tested is
unstructured text read at scale: guidance wording, tone, risk language, one-off items in an earnings release. A language model can do that for hundreds of releases per season; a numeric rule cannot.
Why event-level and forward-only: F2 gives ~15 picks per week (verdict ~Apr 2027). One earnings season yields >1,000 qualifying releases, so the question can be answered in weeks. A backtest is NOT possible: the
model has seen how past releases turned out (look-ahead leakage), so only text scored BEFORE its outcome exists counts.

## Event definition (fixed)
- Source: SEC 8-K filings with Item 2.02 and an EX-99.1 press release. Timestamp = SEC acceptance datetime (point-in-time). The collector stores `accepted_utc` and `first_seen_utc`; an event counts only if it was scored and committed to git BEFORE its entry open.
- Universe U_F4 at the prior close: US common stock, Close ≥ $5, 20-day ADV$ ≥ $1M, marketCap $300M–$10B. (Mega/large caps are priced fast; the hypothesis is slow pricing in the smaller band.)
- Dedupe: one event per (ticker, quarter).

## Timing / tradability (fixed; this is what a retail account can really do)
- Scoring batch each trading day at 08:20 America/New_York: all qualifying filings accepted after the previous batch cut-off up to 08:20 ET. Entry = **Open of that day's session (09:30 ET)**; filings accepted 08:20–16:00 ET are scored in the next batch and enter the following open (a delay of up to ~24h is the realistic cost of a non-HFT process, and it is part of the test).
- Exit = Open 20 trading days after entry (secondary horizon: 5 days). Return gross; costs reported separately (5/10/25 bp spread by cap band + $0.35 or $1.00/leg on $270 + 0.15% slippage).
- Excess = event return − SPY return over the same open-to-open dates. Information-content diagnostic: event return − equal-weight mean of all scored events entering the same open.

## The scorer (fixed)
- A fresh subagent per batch of ≤ 10 releases, no tools, no web. Input per release: the EX-99.1 text (first 12,000 characters) with the company name and ticker masked, plus consensus EPS and revenue from the earnings calendar (FMP `earnings-calendar`) if present. NO price, volume or reaction data.
- Output: integer score −5…+5 (expected 20-day relative performance), one line of reason. Prompt and its sha256 are fixed in `scripts/forward/f4_prompt.txt` before the first batch; releases are scored in random order; scored files are committed with hashes before the entry open.
- Reference baseline (not an arm, same events): numeric surprise = (EPS actual − estimate) / |estimate|, rank-based.

## Primary test (ONE hypothesis, K = 1)
Rank-IC (Spearman) between the LLM score and the 20-day excess return vs SPY, pooled over events, inference by permutation within entry-week (events in the same week share market conditions), one-sided, alpha = 0.05.
**Success needs ALL of:**
1. IC ≥ 0.05 and one-sided p < 0.05;
2. Mean excess of events with score ≥ +3 minus mean of all scored events ≥ +1.0 pp at 20 days;
3. Net of the cost model above, mean excess vs SPY of events with score ≥ +3 > 0 with the one-sided 95% lower bound > 0 (week-clustered);
4. Positive in both halves of the sample (by entry date) and the result is not dominated by one week (report without the best week).
Otherwise **DEAD** (no re-tuning, no new prompt on the same events).

## Stopping rule (single look, blind until then)
No return statistic is computed until ≥ **1,000** resolved events across ≥ 30 distinct entry dates. Before that only counts are shown. Evaluate once. Minimum detectable effect at n = 1,000 is IC ≈ 0.07 (a smaller true effect will usually read inconclusive; a DEAD verdict means "none large enough to matter after costs", not "none").
Secondary, reported but never decisions: 5-day horizon; partial IC of the score after controlling for numeric EPS/revenue surprise and the day-1 gap (does text add beyond numbers?); by cap band; by sector; score-bucket monotonicity.

## What would change the live system
Nothing automatically. A pass is a recommendation to the owner for a small, capped, kill-switched sleeve (cost plan Tiered strongly preferred). A fail closes the idea. The live gates stay frozen (no-flipflop).

## Operations (to be built after approval)
- Collector `scripts/forward/f4_collect.py` (deterministic: Finnhub 8-K list → SEC item/text fetch → universe filter → `data/forward_text/events.jsonl`), scorer session via the scheduled-task pattern of docs/f2_round_runbook.md (08:20 ET, Mon–Fri) with a GitHub watchdog, resolver `f4_resolve.py` (prints counts only until the stopping rule is met).
- sec.gov requires a declared contact in the User-Agent: the owner approved using his regular address on 2026-10-29. It lives ONLY in the untracked `.env` as `F4_SEC_CONTACT` (never committed).

## Disclosed limits
Masking names does not remove all priors (the model may recognise a company from its products). Consensus data may be missing for many small caps. One earnings season = one market regime and clustered events. Token cost is real (≈ 1.5–3M input tokens per 1,000 events).

## Addendum A1 (2026-10-29, before the first event) — novelty check against our own history
Asked by the owner: make sure this was not already done. Checked docs/, scripts/, core/ and memory:
- **Numeric earnings surprise** (EPS/revenue vs consensus, small-cap scan `small_cap_signals`): TESTED — rank-IC 0.04–0.05 (t≈4.2–4.3) but top-decile net of costs vs SPY ≤ 0 in every cell. It is therefore used here only as the reference baseline on the SAME events, and the score is given the consensus numbers as context.
- **LLM + headlines** (F2R arm B, F3 arm D, retro rounds): TESTED — "no sign of skill". Headlines are thin and are not the release text.
- **LLM reading of the earnings-release / 8-K text itself**: NEVER tested. Recorded on 2026-10-02 as "10-K/8-K text (Lazy Prices) — heavy, low prior" and as an open idea ("OpenAI credit to test LLM scoring of 8-K/10-K text"; docs/investigation_summary_oct2026.md). Earnings-call transcripts are NOT available on our data plans (FMP returns 402).
- Prior stays modest, as noted then. What is new is the design: forward-only, event-level (>1,000 events per season instead of ~15 picks per week), masked names, and a numeric-surprise baseline on the same events.

