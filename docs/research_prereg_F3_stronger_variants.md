# PREREG F3 — two stronger variants of the LLM picker, on the same post-cutoff rounds (written 2026-10-04, BEFORE any F3 pick)

Context: F2R found "no sign of skill" for arm A (numbers) and arm B (news-only) on 9 post-cutoff rounds. The owner asked to test (1) better data and (2) a different horizon. Same 9 rounds, same 100-name pools, same
blind-picker protocol (fresh subagent per (round, variant), reads one file, JSON only) as F2R. Outcome data is NOT looked at until all 18 new pick files are committed.
- **Arm D — richer data:** numbers (the A pack) AND the last-7-days headlines per name (the B pack) in ONE file. 20-trading-day horizon (as F2R).
- **Arm H — shorter horizon:** the A pack, but the question is "outperform the pool average over the next 5 trading days" (exit = Open 5 trading days after the entry Open; same entry as F2R).
Not testable retroactively (and therefore NOT done here): horizons longer than 20 days (not enough post-cutoff history), and an agent with web access (it would see post-period facts). Both stay forward-only ideas.
Multiple testing: the same pools now carry four arms (A, B, D, H) → each arm judged at one-sided permutation α = 0.0125. "Early evidence of skill" only if p < 0.0125 AND mean excess ≥ +1.0% (D, 20d) / ≥ +0.5% (H, 5d) AND positive in ≥ 6 of 9 rounds. "No sign of skill" if mean excess ≤ 0 or p ≥ 0.20. Else inconclusive.
Controls: pool mean over the same horizon (primary), 10,000 random-5 draws, SPY. The arms are NOT independent (same pools/dates); results are reported together with A and B. Contamination caveats and mitigations as F2R. These are early-signal tests only; F2 forward rounds remain the primary evidence.

---
## RESULT (run 2026-10-04; all 36 pick files — A, B, D, H × 9 rounds — were committed BEFORE any outcome was computed) — VERDICT: "NO SIGN OF SKILL" for both new arms
Same 9 rounds, same pools (equal-weight pool mean averaged +0.24% per 20d, +0.70% per 5d). One-sided permutation vs 10,000 random-5 draws.
| arm | horizon | mean excess vs pool | SE | rounds positive | perm p | hit-rate | picks vs pool return |
|---|---|---|---|---|---|---|---|
| A numbers (F2R) | 20d | −2.75% | 1.40% | 1/9 | 0.952 | 18/45 | −2.52% / +0.24% |
| B news-only (F2R) | 20d | −1.18% | 2.10% | 4/9 | 0.756 | 20/45 | −0.94% / +0.24% |
| **D numbers + news** | 20d | **−2.23%** | 1.22% | **1/9** | **0.913** | 16/45 | −1.99% / +0.24% |
| **H 5-day horizon** | 5d | **−0.73%** | 0.38% | **3/9** | **0.767** | 17/45 | −0.03% / +0.70% |
Rule: "no sign of skill" if mean excess ≤ 0 or p ≥ 0.20 → arms D and H (and A, B). Adding the news to the numbers barely changed the picks (33/45 identical to arm A) and not the outcome; the 5-day picks overlapped arm A in 28/45.
All four variants underperformed the pool, mainly because the picker repeatedly chose names near 52-week highs / strong 3–6-month momentum, which did poorly over Jul–Sep 2026. The arms share pools and a common picking style → they are NOT four independent confirmations; only 9 rounds each (SE 0.4–2.1%).
Not testable retroactively and still open: horizons > 20 days; a tool-using agent with live web/real-time data (forward only); paid/higher-quality data.
