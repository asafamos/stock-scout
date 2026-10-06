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

---
## AMENDMENT A2 (2026-10-06, BEFORE the first round that uses it, 2026-10-11) — arm C: an agent WITH web access (what people actually do)
Retro rounds (F2R/F3) could not test an agent that browses the web, because search returns today's facts, i.e. it would see the future of any past round. Going forward it can be tested cleanly. Starting with the round of **2026-10-11**, each round adds:
- **Arm C (agentic research):** a fresh Claude subagent receives the same 100-name pool (the arm-A pack, as the starting list) and is ALLOWED to use web search / page fetching to research whichever names it wants, with a budget of ≈ 25 tool calls, using only information published up to the round date. It answers with the same JSON (5 tickers from the pool, confidence 1–5, one-line reason) plus up to 5 source URLs. Prompt text stored in `scripts/forward/f2_prompt_agent.txt` (hash logged). Picks + sources are committed before Monday's open like arms A/B.
- Entry/exit, pool, controls and the decision rule are unchanged. Three arms (A, B, C) now share the pools → each arm is judged at one-sided permutation α = 0.0125; arm C's clock starts at 2026-10-11 (26 rounds → ≈ 2027-04-04); A and B continue from 2026-10-04.
- Reported at evaluation: whether C beats A/B head-to-head, whether C's picks differ from A's, and the naive 12-1 momentum baseline (computable from the logged pools).

---
## AMENDMENT A3 (2026-10-06, BEFORE any pick of this round) — off-schedule round "2026-10-06" and an earlier start for arm C
Owner asked to run a round on Tuesday 2026-10-06 (market already open). To keep the rule "picks are committed before the entry open", this extra round uses data as of **Monday 2026-10-05 close**, picks are made and committed on 2026-10-06, and the entry is the **Open of Wednesday 2026-10-07** (explicit `entry_open_date` in the picks file; exit = Open 20 trading days later). Pool seed = 20261006.
Arm C (web agent) is also run in this round, so arm C's clock starts at this round rather than 2026-10-11 (A2 is amended accordingly; nothing of arm C had been run). Extra off-schedule rounds count as additional observations; the evaluation uses all valid rounds with block resampling for overlap (the "≥ 20 valid rounds" minimum is unchanged).
