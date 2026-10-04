# PREREG F3 — two stronger variants of the LLM picker, on the same post-cutoff rounds (written 2026-10-04, BEFORE any F3 pick)

Context: F2R found "no sign of skill" for arm A (numbers) and arm B (news-only) on 9 post-cutoff rounds. The owner asked to test (1) better data and (2) a different horizon. Same 9 rounds, same 100-name pools, same
blind-picker protocol (fresh subagent per (round, variant), reads one file, JSON only) as F2R. Outcome data is NOT looked at until all 18 new pick files are committed.
- **Arm D — richer data:** numbers (the A pack) AND the last-7-days headlines per name (the B pack) in ONE file. 20-trading-day horizon (as F2R).
- **Arm H — shorter horizon:** the A pack, but the question is "outperform the pool average over the next 5 trading days" (exit = Open 5 trading days after the entry Open; same entry as F2R).
Not testable retroactively (and therefore NOT done here): horizons longer than 20 days (not enough post-cutoff history), and an agent with web access (it would see post-period facts). Both stay forward-only ideas.
Multiple testing: the same pools now carry four arms (A, B, D, H) → each arm judged at one-sided permutation α = 0.0125. "Early evidence of skill" only if p < 0.0125 AND mean excess ≥ +1.0% (D, 20d) / ≥ +0.5% (H, 5d) AND positive in ≥ 6 of 9 rounds. "No sign of skill" if mean excess ≤ 0 or p ≥ 0.20. Else inconclusive.
Controls: pool mean over the same horizon (primary), 10,000 random-5 draws, SPY. The arms are NOT independent (same pools/dates); results are reported together with A and B. Contamination caveats and mitigations as F2R. These are early-signal tests only; F2 forward rounds remain the primary evidence.
