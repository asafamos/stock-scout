# PREREG F1 — Prospective (forward) test of two candidate effects (written 2026-10-03; every event it uses occurs AFTER this date)

Why: H5 (congress purchases) came close but below the bar on historical data (holdout t≈2.0–2.3, net CI includes 0) and H4 (insider clusters) showed an exploratory small-cap pattern that cannot be
confirmed on data already seen. The only clean confirmation is data that did not exist when the hypotheses were formed. No money is involved (paper tracking only).

## Collection (point-in-time by construction)
Daily GitHub Action (`.github/workflows/forward_collect.yml`, 22:30 UTC weekdays) runs `scripts/forward/collect_forward.py` and appends to `data/forward/*.jsonl` with `first_seen_utc`.
An event counts only if `first_seen_utc` is before its entry open (guards against collector outages / back-filled rows). Collection never touches trading code or the VPS.

## Hypothesis F1a — congress purchases (exactly the H5 definition)
Valid row: type contains "Purchase", assetType Stock, amount lower bound ≥ $15,001, disclosureDate ≥ transactionDate and ≤ +400 days. Event = (stock, disclosureDate) across Senate+House; drop if the same stock fired in the previous 60 trading days.
U1 at entry (Close ≥ $5, ADV$20 ≥ $1M, marketCap ≥ $300M at the prior close). Entry = Open of the next trading day after `disclosureDate`; exit = Open 60 trading days later (single horizon; the 20d variant is NOT tested).
Excess = event return − SPY return over the same dates (SPY, not the U1 mean, because no universe panel exists going forward; an equal-weight U1 benchmark is reported as a diagnostic only).
**Decision rule (fixed now):** evaluate ONCE, when the number of resolved qualifying events reaches **n ≥ 300** (≈ 10 months at ~30/month). One-sided test (H1: mean excess > 0), clustered by calendar month, alpha = 0.05, single pre-registered hypothesis (no multiplicity),
AND net-of-cost mean excess (5/10/25 bp by cap band + $0.35/leg on $270 + 0.15% slippage) > 0 with one-sided 95% lower bound > 0. Both required, else DEAD. If n < 300 by 2027-12-31, report as inconclusive.

## Hypothesis F1b — small-cap insider purchase clusters (the H4 E1 definition restricted to the band where H4 looked promising)
Valid rows and cluster rule exactly as H4 (≥ 2 distinct officers/directors, each ≥ $10,000, transactionDates within 30 days; event date = filing date completing the cluster; 60-trading-day dedupe).
Universe: U1 and marketCap in [$300M, $2B) at the prior close. Entry/exit as F1a (60d), excess vs SPY. **Decision rule:** evaluate ONCE at **n ≥ 150** resolved events, same test and net-of-cost requirement as F1a.
Note: the insider purchase store starts empty on 2026-10-03 (+ the latest feed's backlog); clusters require the 30-day lookback to fill, so the first F1b events appear after ≈ 1 month.

## What would change the live system
Nothing automatically. A pass is a recommendation to the owner (who decides) for a small, capped, kill-switched sleeve; a fail closes the idea.
