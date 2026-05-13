"""Deep forensic analysis on all closed trades.

The cohort analysis (2026-05-13) revealed surface patterns but mis-attributed
cause. This script digs deeper:

  1. Adopted vs system-bought: who's actually winning?
  2. Per-trade predictor table: score, RR, stop-distance, sector, regime
  3. What's the actual correlation between score/RR and outcome?
  4. What did our LOSERS have in common?
  5. What did our WINNERS have in common?
  6. Rejected candidates analysis (the ones we SKIPPED) — did they win?

Output: written digest + one-shot insights.

Usage: python -m scripts.forensic_analysis
"""
from __future__ import annotations

import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
TRADE_LOG = ROOT / "data" / "trades" / "trade_log.json"


def _load_log() -> list:
    return json.loads(TRADE_LOG.read_text())


def _pair_trades(log: list) -> list:
    """Return list of (open_event, close_event, pnl, hold_days, adopted_flag)."""
    opens = defaultdict(list)
    pairs = []
    for e in sorted(log, key=lambda e: str(e.get("timestamp", ""))):
        a = e.get("action")
        t = e.get("ticker")
        if a == "OPEN":
            opens[t].append(e)
        elif a == "CLOSE":
            o_list = opens.get(t, [])
            pnl = float(e.get("pnl", 0) or 0)
            if o_list:
                o = o_list.pop(0)
                adopted = False
            else:
                # No OPEN — adopted from IB
                o = {"ticker": t, "timestamp": e.get("timestamp", "")}
                adopted = True
            try:
                ts_o = datetime.fromisoformat(str(o.get("timestamp", ""))[:19])
                ts_c = datetime.fromisoformat(str(e.get("timestamp", ""))[:19])
                hd = (ts_c - ts_o).days
            except Exception:
                hd = 0
            pairs.append((o, e, pnl, hd, adopted))
    return pairs


def main():
    log = _load_log()
    pairs = _pair_trades(log)

    if not pairs:
        print("No trades to analyze.")
        return

    print("=" * 70)
    print("  FORENSIC ANALYSIS — every angle of our 11 trades")
    print("=" * 70)

    # ── 1. ADOPTED vs SYSTEM-BOUGHT ─────────────
    print("\n1. ADOPTED vs SYSTEM-BOUGHT")
    print("-" * 70)
    adopted = [p for p in pairs if p[4]]
    bought = [p for p in pairs if not p[4]]
    for label, group in [("ADOPTED (manual buys)", adopted), ("SYSTEM-BOUGHT", bought)]:
        if not group:
            continue
        n = len(group)
        wins = sum(1 for _, _, pnl, _, _ in group if pnl > 0)
        total = sum(pnl for _, _, pnl, _, _ in group)
        gross_w = sum(pnl for _, _, pnl, _, _ in group if pnl > 0)
        gross_l = abs(sum(pnl for _, _, pnl, _, _ in group if pnl < 0))
        pf = gross_w / gross_l if gross_l > 0 else 999.0
        print(f"  {label}:")
        print(f"    n={n}  WR={wins/n*100:.0f}%  P&L=${total:+.2f}  PF={pf:.2f}")
        for o, c, pnl, hd, _ in group:
            t = c.get("ticker")
            r = str(c.get("reason", ""))[:25]
            print(f"      {t:6s}  ${pnl:+7.2f}  hold={hd:2d}d  {r}")

    # ── 2. SCORE COHORT — does higher score = better outcome? ───
    print("\n2. SCORE → OUTCOME (system-bought only — adopted have no score)")
    print("-" * 70)
    print(f"  {'Ticker':<7} {'Score':>6} {'RR':>6} {'StopDist':>9} {'Hold':>5} {'P&L':>9} {'Reason'}")
    print("  " + "-" * 65)
    score_to_pnl = []
    for o, c, pnl, hd, ad in bought:
        if ad:
            continue
        s = float(o.get("score", 0) or 0)
        e = float(o.get("price", 0) or 0)
        sl = float(o.get("stop_loss", 0) or 0)
        tp = float(o.get("target_price", 0) or 0)
        stop_dist = (e - sl) / e * 100 if e > 0 else 0
        rr = (tp - e) / (e - sl) if (e - sl) > 0 else 0
        score_to_pnl.append((s, pnl))
        print(f"  {c.get('ticker'):<7} {s:>6.1f} {rr:>6.2f} {stop_dist:>8.1f}% {hd:>4}d ${pnl:>+7.2f}  {str(c.get('reason',''))[:22]}")

    # Simple correlation: high score → high P&L?
    if len(score_to_pnl) >= 4:
        score_avg = mean(s for s, _ in score_to_pnl)
        winners_avg_score = mean(s for s, p in score_to_pnl if p > 0) if any(p > 0 for _, p in score_to_pnl) else 0
        losers_avg_score = mean(s for s, p in score_to_pnl if p < 0) if any(p < 0 for _, p in score_to_pnl) else 0
        print(f"\n  Avg score (all):     {score_avg:.1f}")
        print(f"  Avg score winners:   {winners_avg_score:.1f}")
        print(f"  Avg score losers:    {losers_avg_score:.1f}")
        print(f"  Δ:                   {winners_avg_score - losers_avg_score:+.1f} pts")
        if abs(winners_avg_score - losers_avg_score) > 2:
            print("  ⚠ Score MIGHT separate winners — investigate raising min_score")
        else:
            print("  ⚪ Score barely separates winners — selection issue is elsewhere")

    # ── 3. RR DISTRIBUTION ──────────────────────
    print("\n3. R:R RATIO → OUTCOME")
    print("-" * 70)
    rr_winners = []
    rr_losers = []
    for o, c, pnl, hd, ad in bought:
        if ad:
            continue
        e = float(o.get("price", 0) or 0)
        sl = float(o.get("stop_loss", 0) or 0)
        tp = float(o.get("target_price", 0) or 0)
        if e <= 0 or e <= sl:
            continue
        rr = (tp - e) / (e - sl)
        if pnl > 0:
            rr_winners.append(rr)
        else:
            rr_losers.append(rr)
    if rr_winners and rr_losers:
        print(f"  Winners avg R:R:     {mean(rr_winners):.2f}")
        print(f"  Losers  avg R:R:     {mean(rr_losers):.2f}")
        if mean(rr_winners) > mean(rr_losers):
            print("  → Higher R:R correlates with wins (intuitive)")
        else:
            print("  ⚠ Losers had HIGHER R:R — RR alone isn't a selector!")

    # ── 4. HOLD DAYS DISTRIBUTION ─────────────
    print("\n4. HOLD DAYS — only non-adopted")
    print("-" * 70)
    hold_winners = [hd for _, _, pnl, hd, ad in bought if pnl > 0 and not ad]
    hold_losers = [hd for _, _, pnl, hd, ad in bought if pnl < 0 and not ad]
    if hold_winners:
        print(f"  Winners hold (avg):  {mean(hold_winners):.1f}d  range {min(hold_winners)}-{max(hold_winners)}d")
    if hold_losers:
        print(f"  Losers  hold (avg):  {mean(hold_losers):.1f}d  range {min(hold_losers)}-{max(hold_losers)}d")

    # ── 5. STOP-LOSS DISTANCE ANALYSIS ──────
    print("\n5. INITIAL STOP DISTANCE")
    print("-" * 70)
    stop_winners = []
    stop_losers = []
    for o, c, pnl, hd, ad in bought:
        if ad:
            continue
        e = float(o.get("price", 0) or 0)
        sl = float(o.get("stop_loss", 0) or 0)
        if e > 0 and sl > 0:
            stop_dist = (e - sl) / e * 100
            if pnl > 0:
                stop_winners.append(stop_dist)
            else:
                stop_losers.append(stop_dist)
    if stop_winners and stop_losers:
        print(f"  Winners initial stop: {mean(stop_winners):.1f}% (avg)")
        print(f"  Losers  initial stop: {mean(stop_losers):.1f}% (avg)")
        if mean(stop_losers) > mean(stop_winners) + 0.5:
            print("  ⚠ Losers had WIDER stops — wider stop ≠ safer here")
        elif mean(stop_winners) > mean(stop_losers) + 0.5:
            print("  → Winners had wider stops — let trades breathe?")

    # ── 6. CLOSE REASON DEEP DIVE ────────────
    print("\n6. CLOSE REASON × P&L (full lifetime)")
    print("-" * 70)
    reasons = defaultdict(lambda: {"pnl": 0.0, "n": 0, "wins": 0})
    for o, c, pnl, hd, ad in pairs:
        r = str(c.get("reason", "unknown")).lower()
        if "trail_fired_in_profit" in r:
            bucket = "trail (was in profit)"
        elif "trail" in r and "oil_crash" in r:
            bucket = "trail (oil crash)"
        elif "trail" in r and "day_trade" in r:
            bucket = "trail (DAY TRADE VIOLATION)"
        elif "trail" in r and "open" in r:
            bucket = "trail (gap-down at open)"
        elif "trail" in r:
            bucket = "trail (other)"
        elif "target" in r or "stop_or_target" in r:
            bucket = "target hit"
        elif "market sell" in r:
            bucket = "manual market sell"
        else:
            bucket = r[:25]
        reasons[bucket]["pnl"] += pnl
        reasons[bucket]["n"] += 1
        if pnl > 0:
            reasons[bucket]["wins"] += 1
    for bucket, data in sorted(reasons.items(), key=lambda x: -x[1]["pnl"]):
        wr = data["wins"] / data["n"] * 100 if data["n"] > 0 else 0
        print(f"  {bucket:<32}  n={data['n']:>2}  WR={wr:>3.0f}%  P&L ${data['pnl']:+7.2f}")

    # ── 7. PER-TICKER TICKER POST-MORTEM ─────
    print("\n7. EACH TRADE'S STORY")
    print("-" * 70)
    for o, c, pnl, hd, ad in pairs:
        t = c.get("ticker")
        e = float(o.get("price", 0) or 0)
        sl = float(o.get("stop_loss", 0) or 0)
        tp = float(o.get("target_price", 0) or 0)
        s = o.get("score", "?")
        r = str(c.get("reason", ""))
        ad_flag = " [ADOPTED]" if ad else ""
        if e > 0 and sl > 0 and tp > 0:
            stop_dist = (e - sl) / e * 100
            rr = (tp - e) / (e - sl) if (e - sl) > 0 else 0
            print(f"  {t:6s}{ad_flag}  entry=${e:.2f}  stop=${sl:.2f} (-{stop_dist:.1f}%)  target=${tp:.2f}  RR={rr:.2f}  score={s}")
        else:
            print(f"  {t:6s}{ad_flag}  (no scan data)")
        print(f"          → ${pnl:+.2f}  hold={hd}d  reason: {r[:40]}")

    # ── 8. KEY INSIGHTS ────────────────────────
    print("\n8. INSIGHTS")
    print("-" * 70)
    insights = []
    # 1. Adopted vs system
    if adopted and bought:
        ad_wr = sum(1 for _, _, p, _, _ in adopted if p > 0) / len(adopted)
        sb_wr = sum(1 for _, _, p, _, _ in bought if p > 0) / len(bought)
        if ad_wr - sb_wr > 0.20:
            insights.append(
                f"⚠ Adopted trades WIN-RATE {ad_wr*100:.0f}% vs system {sb_wr*100:.0f}%. "
                f"Manual instincts outperform — investigate WHAT system is missing."
            )
    # 2. Score separation
    if winners_avg_score and losers_avg_score and abs(winners_avg_score - losers_avg_score) < 2:
        insights.append(
            f"⚠ Score nearly identical for winners ({winners_avg_score:.1f}) and "
            f"losers ({losers_avg_score:.1f}). Min-score increase won't help — "
            f"selection model is missing some other signal."
        )
    # 3. RR
    if rr_winners and rr_losers and mean(rr_losers) > mean(rr_winners):
        insights.append(
            f"⚠ Losers had HIGHER R:R than winners — system may be picking "
            f"'lottery tickets' (high target, low probability). Consider biasing "
            f"toward LOWER R:R (closer, more achievable targets)."
        )
    # 4. Hold time
    if hold_winners and hold_losers:
        if mean(hold_winners) > mean(hold_losers) * 2:
            insights.append(
                f"→ Winners hold {mean(hold_winners):.1f}d, losers {mean(hold_losers):.1f}d. "
                f"Patience pays — but losers EXIT early, which is correct behavior."
            )
    # 5. Specific stocks
    losers_tickers = [c.get("ticker") for o, c, pnl, _, _ in bought if pnl < 0]
    insights.append(
        f"📊 System-bought losers: {losers_tickers}"
    )
    winners_tickers = [c.get("ticker") for o, c, pnl, _, _ in bought if pnl > 0]
    insights.append(
        f"📊 System-bought winners: {winners_tickers}"
    )

    for i in insights:
        print(f"  {i}")


if __name__ == "__main__":
    main()
