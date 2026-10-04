# PREREG F2 — Prospective test: can an LLM pick stocks that outperform? (written 2026-10-04, BEFORE the first picks)

Question (the owner's): "People use AI to find stocks before they rise — does it work?" Backtests of an LLM are contaminated (it has read the past), so this is a FORWARD, paper-only test.
Every pick is logged and committed to git BEFORE the entry open, so the commit time proves it was not made with hindsight.

## Design
- **Weekly round**, every Sunday for 26 consecutive weeks starting 2026-10-04 (rounds missed for operational reasons are logged; ≥ 20 valid rounds are needed for the verdict).
- **Pool:** 100 US stocks drawn by a seeded random sample (seed = YYYYMMDD of the round) from the tradable universe U1 (Close ≥ $5, 20-day ADV$ ≥ $1M, marketCap ≥ $300M, data as of Friday's close). Pool membership is logged. The pool is the fair control: the LLM must beat the pool, not the market.
- **Information given to the picker (point-in-time, same for every round):** per stock — last close, 1/3/6/12-month returns, distance from the 52-week high, 20-day realised volatility, ADV$, market cap, sector/industry, beta, short interest % of shares (latest FINRA settlement, 9-day publication lag), analyst net upgrades in the last 60 days, and whether insiders / Congress bought in the last 30 days. No news text, no web access.
- **Picker:** a fresh Claude instance with NO session history, a fixed prompt (stored in `scripts/forward/f2_prompt.txt`, hash logged), instructed to answer with JSON only: exactly 5 tickers from the pool (+ confidence 1–5 and a one-line reason each). Prompt text is not changed during the test.
- **Entry / exit:** Open of the next trading day (Monday) after the round; exit = Open 20 trading days later. Equal weight, no stops.

## Controls (all computed from the logged pool)
1. **Pool mean** (equal-weight all 100) over the same dates — primary benchmark.
2. **Random-5 permutation:** 10,000 random draws of 5 names from the same pool per round (same dates) — gives the null distribution of "any 5 names".
3. SPY (reported only).
4. (Reported only) the legacy bot's score-ranked top-5 within the same pool when computable.

## Decision rule (fixed now; evaluated ONCE after the 26th round has resolved, ≈ mid-2027; no interim decisions)
Primary statistic: mean over rounds of [mean 20-day return of the 5 picks − pool-mean return]. Inference: permutation test against control 2 (one-sided, α = 0.05) with week-block resampling (rounds overlap 4×, so Newey-West / block bootstrap with 4-round blocks).
**The LLM "works for us" only if BOTH:** (a) one-sided permutation p < 0.05, AND (b) mean 20-day excess vs the pool ≥ +1.0% (≈ covers our ~0.5% round-trip cost with margin) with a one-sided 95% lower bound > 0.
Otherwise the verdict is "no demonstrated edge". Also reported: hit rate vs pool, calibration of the 1–5 confidence (does confidence 5 beat confidence 1?), and picks' size/sector tilt vs the pool (to show whether any result is just beta/size).

## What would change the live system
Nothing automatically. A pass is a recommendation to the owner for a small, capped, kill-switched sleeve; a fail closes the idea. Paper only until then.

---
## AMENDMENT A1 (2026-10-04, BEFORE any pick was made) — second arm: news-only ("information that comes before the numbers")
Owner's point: people who find stocks "before they rise" use information that precedes technical/fundamental data (news, events), not price statistics. So each round has TWO independent blind pickers on the SAME 100-name pool:
- **Arm A (numbers):** the data pack above (price/volume/short-interest/analyst/insider/congress statistics).
- **Arm B (news-only):** for each pool name only ticker, sector/industry and up to 4 most recent headlines from the last 7 days (Polygon `/v2/reference/news`, title + publisher + date); NO prices, returns, volatility, valuation or any numeric data. Names without news show "no news in the last 7 days".
Same prompt text (arm B's prompt replaces "point-in-time data" with "point-in-time news headlines"); separate fresh Claude instances with no shared context; both log picks + pack hashes before the entry open.
Multiple testing: two arms → each arm is judged at one-sided α = 0.025 (Bonferroni) on the permutation test; the "+1.0% net-useful" margin and every other rule are unchanged per arm. Also reported: arm A vs arm B head-to-head and the overlap of their picks.
