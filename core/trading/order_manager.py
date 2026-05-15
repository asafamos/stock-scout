"""Order orchestration — the brain of auto-trading.

Reads scan recommendations, runs them through risk checks,
and executes buy + stop + target orders via IBKR.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.trading.config import CONFIG, TradingConfig
from core.trading.ibkr_client import IBKRClient, TradeResult
from core.trading.position_tracker import PositionTracker
from core.trading.risk_manager import RiskManager
from core.trading import notifications as notify

logger = logging.getLogger(__name__)


# Small process-local cache so we don't hit yfinance per _execute_single.
# Keyed by ticker; value is (analyst_mean, analyst_high, n_analysts, fetched_at_ts).
_ANALYST_CACHE: dict = {}
_ANALYST_CACHE_TTL = 6 * 3600  # 6 hours — analyst PTs change slowly

# SPY 5-day return cache — fetched once per scan run to enable
# momentum-vs-market filter. Critical insight from 31-day forensic
# comparison (2026-05-14): StockScout underperformed SPY by -1.41pp
# largely because our ranking favored high-R:R Energy/value names while
# SPY's gains were driven by mega-cap tech. A candidate that LAGS SPY
# in the recent 5 days is statistically unlikely to outperform over a
# 20-day swing — the rising tide isn't lifting their boat.
_SPY_5D_CACHE: dict = {"return": None, "fetched_at": 0.0}
_SPY_CACHE_TTL = 6 * 3600  # 6 hours — daily-bar precision is fine


def _fetch_spy_5d_return() -> float:
    """SPY's 5-day return %, cached. Returns 0.0 on any failure (filter
    becomes a no-op rather than rejecting all candidates)."""
    import time as _time
    if _SPY_5D_CACHE["return"] is not None:
        if _time.time() - _SPY_5D_CACHE["fetched_at"] < _SPY_CACHE_TTL:
            return _SPY_5D_CACHE["return"]

    def _do_fetch():
        import yfinance as yf
        hist = yf.Ticker("SPY").history(period="10d", interval="1d")
        if hist is None or len(hist) < 6:
            return 0.0
        closes = hist["Close"].dropna()
        if len(closes) < 6:
            return 0.0
        # 5-day return = (latest_close / close_5_days_ago - 1) * 100
        return (closes.iloc[-1] / closes.iloc[-6] - 1.0) * 100.0

    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TO
        with ThreadPoolExecutor(max_workers=1) as ex:
            r = ex.submit(_do_fetch).result(timeout=10.0)
        _SPY_5D_CACHE["return"] = r
        _SPY_5D_CACHE["fetched_at"] = _time.time()
        return r
    except Exception as e:
        logger.warning("SPY 5d return fetch failed: %s — filter disabled this run", e)
        return 0.0

# Hard wall-clock timeout for yfinance .info calls. Real-world failure
# 2026-05-05: a manual evaluator run blocked for 11+ min on per-candidate
# `yf.Ticker(...).info` because yfinance has NO native timeout — its
# urllib session uses Python's default (= forever). With 17 candidates,
# even a 60s soft hang per call burns 17 min before the trade window
# even closes. We wrap the call in a thread with .result(timeout=N) so
# a slow yfinance just returns the (0,0,0) "no data" path instead of
# stalling the whole pipeline.
_YFINANCE_INFO_TIMEOUT_SEC = 6.0


def _fetch_analyst_target(ticker: str) -> tuple:
    """Fetch (mean_pt, high_pt, n_analysts) from yfinance. Returns (0,0,0) on
    any failure or timeout. The timeout (default 6s) prevents the trade
    evaluator from stalling indefinitely when yfinance is rate-limiting or
    just slow (their CDN occasionally takes 30-60s for .info responses)."""
    import time as _time
    cached = _ANALYST_CACHE.get(ticker)
    if cached and (_time.time() - cached[3] < _ANALYST_CACHE_TTL):
        return cached[:3]

    def _do_fetch():
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return (
            float(info.get("targetMeanPrice", 0) or 0),
            float(info.get("targetHighPrice", 0) or 0),
            int(info.get("numberOfAnalystOpinions", 0) or 0),
        )

    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TO
        with ThreadPoolExecutor(max_workers=1) as ex:
            mean_pt, high_pt, n = ex.submit(_do_fetch).result(
                timeout=_YFINANCE_INFO_TIMEOUT_SEC
            )
        _ANALYST_CACHE[ticker] = (mean_pt, high_pt, n, _time.time())
        return mean_pt, high_pt, n
    except _TO:
        logger.warning(
            "Analyst fetch TIMED OUT (%.0fs) for %s — proceeding without consensus",
            _YFINANCE_INFO_TIMEOUT_SEC, ticker,
        )
        # Cache empty result for 5min so we don't retry every call site
        _ANALYST_CACHE[ticker] = (0.0, 0.0, 0, _time.time() - _ANALYST_CACHE_TTL + 300)
        return 0.0, 0.0, 0
    except Exception as e:
        logger.debug("Analyst fetch failed for %s: %s", ticker, e)
        return 0.0, 0.0, 0


def _cap_target_with_analysts(ticker: str, current_price: float,
                               scan_target: float) -> float:
    """Cap scan target by Wall Street consensus.

    Returns:
        - None if the stock is rated overvalued (analyst_mean < current_price)
          and there are enough analysts (≥3) covering it to trust that signal.
        - Otherwise, midpoint of (scan_target, analyst_mean), floored at scan_target
          × 0.95 so we don't over-reduce. If no analyst data, returns scan_target.
    """
    if scan_target <= 0 or current_price <= 0:
        return scan_target
    mean_pt, high_pt, n = _fetch_analyst_target(ticker)
    # No coverage or thin coverage → trust scan-target
    if mean_pt <= 0 or n < 3:
        return scan_target
    # Overvalued per consensus (with meaningful coverage) → skip entirely
    if mean_pt < current_price:
        logger.info(
            "Analyst veto for %s: mean PT $%.2f < current $%.2f (n=%d)",
            ticker, mean_pt, current_price, n,
        )
        return None
    # Cap target at midpoint between scan's target and analyst mean.
    # Lean slightly toward scan (0.55) so we don't over-trim — scan has
    # technical momentum data analysts lack.
    blended = scan_target * 0.55 + mean_pt * 0.45
    # Floor: never reduce below 95% of original scan target
    capped = max(blended, scan_target * 0.95)
    return round(capped, 2)


class OrderManager:
    """Orchestrates scan → risk check → order execution → tracking."""

    def __init__(self, config: Optional[TradingConfig] = None):
        self.cfg = config or CONFIG
        self.client = IBKRClient(self.cfg)
        self.tracker = PositionTracker(self.cfg)
        self.risk = RiskManager(self.client, self.tracker, self.cfg)

    def execute_recommendations(
        self,
        scan_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict]:
        """Main entry: read scan, filter, execute trades.

        Returns list of trade result dicts for logging/display.
        """
        logger.info("=" * 60)
        logger.info("AUTO-TRADE: Starting execution")
        logger.info(self.cfg.summary())
        logger.info("=" * 60)

        # 1. Load scan results
        if scan_df is None:
            scan_df = self._load_scan_results()
        if scan_df is None or scan_df.empty:
            logger.warning("No scan results available — aborting")
            return []

        # 2. Connect to IBKR FIRST so the candidate filter can use real IB
        # position data for dedup. Previously connect() ran AFTER filter,
        # which meant `client.get_positions()` inside _filter_candidates
        # raised + was swallowed → ibkr_held was always empty. The tracker
        # was the only de-dup source — and a stale tracker (e.g. KNX
        # phantom) would let us over-buy. (Audit finding #6.)
        if not self.client.connect():
            logger.error("Failed to connect to IBKR — aborting")
            return []

        # 3. Filter candidates (now with live IB data available)
        candidates = self._filter_candidates(scan_df)
        if candidates.empty:
            logger.info("No candidates passed filters")
            return []

        logger.info("Candidates after filtering: %d", len(candidates))

        # Audit M5+H6 (2026-05-01): pre-warm analyst-PT cache for top-N
        # candidates BEFORE entering the trade loop. Each
        # `_cap_target_with_analysts` call inside `_execute_single` would
        # otherwise hit yfinance during the time-sensitive trade window.
        # If yfinance is rate-limiting at scan-end (fairly common), trades
        # would skip with no analyst data — silently bypassing the
        # analyst-veto gate. Pre-fetching outside the trade loop also
        # lets us short-circuit when yfinance is fully down (we know
        # before placing any orders that the analyst gate is degraded).
        try:
            n_to_prefetch = min(len(candidates), int(self.cfg.max_daily_buys * 2))
            top_tickers = candidates.head(n_to_prefetch).get("Ticker", pd.Series([])).tolist()
            for _t in top_tickers:
                _fetch_analyst_target(_t)  # populates module-level cache
            # Also pre-warm SPY for the momentum-vs-SPY filter — same
            # network round-trip pattern, so do it now rather than during
            # the time-sensitive trade window.
            _spy = _fetch_spy_5d_return()
            logger.info(
                "Pre-warmed: %d analyst-PT entries + SPY 5d=%+.2f%%",
                n_to_prefetch, _spy,
            )
        except Exception as _pw_err:
            logger.debug("Pre-warm skipped: %s", _pw_err)

        results = []
        try:
            # 4. Execute trades
            for _, row in candidates.iterrows():
                result = self._execute_single(row)
                results.append(result)

                # Stop if daily limit reached
                if self.tracker.daily_buy_count() >= self.cfg.max_daily_buys:
                    logger.info("Daily buy limit reached — stopping")
                    break
        finally:
            self.client.disconnect()

        # 5. Summary + notifications
        bought = [r for r in results if r.get("status") == "success"]
        skipped = [r for r in results if r.get("status") == "skipped"]
        failed = [r for r in results if r.get("status") == "error"]
        logger.info(
            "AUTO-TRADE complete: %d bought, %d skipped, %d failed",
            len(bought), len(skipped), len(failed),
        )

        # Telegram notification
        notify.notify_scan_complete(
            total=len(scan_df) if scan_df is not None else 0,
            candidates=len(candidates),
            bought=len(bought),
        )
        for r in failed:
            notify.notify_error("Trade", f"{r.get('ticker')}: {r.get('error')}")

        return results

    def check_exits(self) -> List[Dict]:
        """Check open positions for target-date exits."""
        expired = self.tracker.check_target_date_exits()
        if not expired:
            logger.info("No positions at target date")
            return []

        results = []
        if not self.client.connect():
            logger.error("Cannot connect to IBKR for exit check")
            return []

        try:
            for ticker in expired:
                pos = self.tracker.get_position(ticker)
                if not pos:
                    continue
                logger.info("Target date reached for %s — closing", ticker)
                trade = self.client.buy_market(ticker, 0)  # placeholder
                # In practice: sell at market
                # For now, just log
                results.append({
                    "ticker": ticker,
                    "action": "TARGET_DATE_EXIT",
                    "status": "pending_manual" if self.cfg.dry_run else "executed",
                })
        finally:
            self.client.disconnect()

        return results

    def resubmit_protections(self, only_ticker: Optional[str] = None) -> List[Dict]:
        """Re-submit protective orders (trailing stop + limit sell) for open positions.

        Use when orders expired because they were placed as DAY instead of GTC,
        or via command_bus to manually re-protect a specific ticker.

        only_ticker: if provided, only resubmits for that one position.
        """
        positions = self.tracker.get_open_positions()
        if only_ticker:
            positions = [p for p in positions if p["ticker"].upper() == only_ticker.upper()]
        if not positions:
            logger.info("No open positions — nothing to resubmit")
            return []

        if not self.client.connect():
            logger.error("Failed to connect to IBKR — cannot resubmit")
            return []

        results = []
        try:
            for pos in positions:
                ticker = pos["ticker"]
                qty = pos["quantity"]
                entry_price = pos["entry_price"]
                stop_loss = pos.get("stop_loss", 0)
                target_price = pos.get("target_price", 0)

                # Calculate trailing stop % from stored stop loss
                if stop_loss > 0 and entry_price > 0:
                    trail_pct = round((entry_price - stop_loss) / entry_price * 100, 1)
                    trail_pct = max(3.0, min(trail_pct, 8.0))
                else:
                    trail_pct = pos.get("trailing_stop_pct", self.cfg.trailing_stop_pct)

                logger.info(
                    "RESUBMIT: %s x%d | Trail: %.1f%% | Target: $%.2f",
                    ticker, qty, trail_pct, target_price,
                )

                result = self.client.resubmit_protective_orders(
                    ticker=ticker,
                    qty=qty,
                    trail_pct=trail_pct,
                    target_price=target_price,
                )

                # Update position with new order IDs
                new_order_ids = {
                    "buy": pos.get("order_ids", {}).get("buy", 0),
                    "trailing_stop": result["trailing_stop"].order_id,
                    "limit_sell": result["limit_sell"].order_id,
                    "oca_group": result.get("oca_group", ""),
                }
                self._update_position_order_ids(ticker, new_order_ids)

                status = "success" if result["trailing_stop"].status != "Error" else "error"
                results.append({
                    "ticker": ticker,
                    "status": status,
                    "trail_pct": trail_pct,
                    "target_price": target_price,
                    "order_ids": new_order_ids,
                })

                # Notify
                notify.notify_buy(
                    ticker, qty, entry_price, stop_loss, target_price,
                    pos.get("score", 0),
                    trail_pct=trail_pct,
                    rr=0,
                    target_date=pos.get("target_date", ""),
                    prefix="🔄 RESUBMIT",
                )
        finally:
            self.client.disconnect()

        ok = sum(1 for r in results if r["status"] == "success")
        fail = sum(1 for r in results if r["status"] == "error")
        logger.info("RESUBMIT complete: %d ok, %d failed", ok, fail)
        return results

    def _update_position_order_ids(self, ticker: str, new_order_ids: dict):
        """Update order IDs for an existing position."""
        positions = self.tracker.get_open_positions()
        for p in positions:
            if p["ticker"] == ticker:
                p["order_ids"] = new_order_ids
                break
        self.tracker._save_positions(positions)

    def emergency_close_all(self) -> bool:
        """Kill switch: cancel all open orders."""
        logger.warning("EMERGENCY CLOSE ALL triggered")
        if not self.client.connect():
            logger.error("Cannot connect for emergency close")
            return False
        try:
            return self.client.cancel_all_orders()
        finally:
            self.client.disconnect()

    # ── Internals ─────────────────────────────────────────────

    def _load_scan_results(self) -> Optional[pd.DataFrame]:
        """Load the most recent scan — auto (GH Actions) or manual (Streamlit).

        Compares modification times of:
          - data/scans/latest_scan_live.json/.parquet  (Streamlit / manual)
          - data/scans/latest_scan.parquet/.json       (GH Actions / auto)
        and loads whichever is newer.
        """
        scans_dir = Path("data/scans")
        candidates = [
            scans_dir / "latest_scan_live.json",
            scans_dir / "latest_scan_live.parquet",
            scans_dir / "latest_scan.parquet",
        ]
        # Find the most recently modified scan file
        best_path = None
        best_mtime = 0.0
        for p in candidates:
            if p.exists() and p.stat().st_mtime > best_mtime:
                best_mtime = p.stat().st_mtime
                best_path = p

        if best_path is None:
            logger.error("No scan results found in %s", scans_dir)
            return None

        # Staleness check — refuse to trade on scan data older than 4 hours.
        # Entry/stop/target from an old scan may be hundreds of bps off and
        # protective orders sized to stale volatility are unsafe.
        import time as _time
        age_sec = _time.time() - best_mtime
        MAX_SCAN_AGE_HOURS = 4

        # FALLBACK (2026-05-05 incident): if the file mtime says >4h but
        # git knows about a recent commit on this path, trust the git
        # commit time. `git checkout` sometimes preserves the blob's
        # original mtime when content hashes match, which made fresh
        # scans look 66h old to the previous staleness check.
        if age_sec > MAX_SCAN_AGE_HOURS * 3600:
            try:
                import subprocess as _sp
                # %ct = committer date (UNIX timestamp). Most recent commit
                # touching this exact path on the local repo.
                rel = str(best_path).replace(str(scans_dir.parent.parent) + "/", "")
                result = _sp.run(
                    ["git", "log", "-1", "--format=%ct", "--", rel],
                    capture_output=True, text=True, timeout=5,
                    cwd=scans_dir.parent.parent,
                )
                if result.returncode == 0 and result.stdout.strip():
                    commit_ts = int(result.stdout.strip())
                    git_age_sec = _time.time() - commit_ts
                    if git_age_sec < MAX_SCAN_AGE_HOURS * 3600:
                        logger.warning(
                            "File mtime says %.1fh old but git commit was "
                            "%.1fh ago — trusting git (likely git-checkout "
                            "preserved blob mtime).",
                            age_sec / 3600, git_age_sec / 3600,
                        )
                        age_sec = git_age_sec
            except Exception as _e:
                logger.debug("git commit-time fallback failed: %s", _e)

        if age_sec > MAX_SCAN_AGE_HOURS * 3600:
            hours = age_sec / 3600
            logger.error(
                "Scan file %s is %.1fh old (max %dh) — refusing to trade on stale data",
                best_path.name, hours, MAX_SCAN_AGE_HOURS,
            )
            return None

        logger.info("Loading scan from: %s (modified %s, age %.1fm)",
                     best_path.name,
                     pd.Timestamp.fromtimestamp(best_mtime).strftime("%Y-%m-%d %H:%M"),
                     age_sec / 60)
        try:
            if best_path.suffix == ".parquet":
                return pd.read_parquet(best_path)
            return pd.read_json(best_path)
        except Exception as e:
            logger.error("Failed to load scan results from %s: %s", best_path, e)
            return None

    def _filter_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply portfolio-informed smart filters for auto-trading.

        Filters based on virtual portfolio analysis (98 closed trades):
        - Score sweet spot: Q3-Q4 perform best, Q5 underperforms
        - ML Prob >= 0.4: below this, expected return is negative
        - Blocked sectors: Consumer Defensive = -4.47% avg, 20% win rate
        """
        # Normalize column names
        score_col = self._find_col(df, ["FinalScore_20d", "Score", "final_score"])
        rr_col = self._find_col(df, ["RewardRisk", "RR_Ratio", "RR", "rr"])
        ticker_col = self._find_col(df, ["Ticker", "ticker", "Symbol"])
        ml_col = self._find_col(df, ["ML_20d_Prob", "ml_prob", "ML_Prob"])
        sector_col = self._find_col(df, ["Sector", "sector"])
        regime_col = self._find_col(df, ["Market_Regime", "market_regime", "Regime"])
        # SignalQuality is the richer signal (High/Medium/Speculative based on signal count);
        # Confidence_Level is always "medium" for most scans, less useful
        confidence_col = self._find_col(df, ["SignalQuality", "Signal_Quality", "Confidence_Level", "Confidence"])
        reliability_col = self._find_col(df, ["Reliability_Score", "Reliability", "reliability"])
        entry_col = self._find_col(df, ["Entry_Price", "entry_price"])
        stop_col = self._find_col(df, ["Stop_Loss", "stop_loss"])

        # ── R:R NORMALIZATION (revised 2026-05-14) ──
        # Forensic analysis on 8 system-bought closes showed losers had
        # AVG R:R 4.12 vs winners 2.50 — lottery-ticket pattern. Root cause:
        # scan emits artificially tight stop_loss values (ELVN: stop 1.6%
        # below entry → R:R inflates to 12.28). The 25% ranking weight on
        # R:R then promotes these "lottery" candidates over realistic ones.
        #
        # Fix: compute and inject an `effective_rr` column. Uses raw target
        # but FLOORS stop distance at 3% before computing R:R. Removes the
        # numerator inflation without rejecting the trade outright — the
        # min_rr_to_trade=2.0 floor still gates. CF (winner) had a scan
        # data corruption (stop > entry); for those rows we fall back to
        # raw RR if available (so CF isn't double-penalized by our fix).
        if entry_col and stop_col and entry_col in df.columns and stop_col in df.columns:
            e_vals = pd.to_numeric(df[entry_col], errors="coerce")
            s_vals = pd.to_numeric(df[stop_col], errors="coerce")
            t_vals_col = self._find_col(df, ["Target_Price", "target_price"])
            if t_vals_col:
                t_vals = pd.to_numeric(df[t_vals_col], errors="coerce")
                # Effective stop pct floored at 3%
                raw_stop_pct = (e_vals - s_vals) / e_vals * 100
                eff_stop_pct = raw_stop_pct.clip(lower=3.0)  # floor 3%
                eff_stop = e_vals * (1 - eff_stop_pct / 100)
                # R:R using floored stop
                eff_rr = (t_vals - e_vals) / (e_vals - eff_stop)
                df = df.copy()
                df["_effective_rr"] = eff_rr

        if not score_col or not ticker_col:
            logger.error("Missing required columns (Score/Ticker) in %s",
                         list(df.columns)[:15])
            return pd.DataFrame()

        result = df.copy()
        initial_count = len(result)

        # ── Market Regime Gate (CRITICAL) ───────────────────────
        # Don't trade at all if market is in PANIC/CORRECTION
        if regime_col and regime_col in result.columns:
            # Check the regime — it should be consistent across all rows (market-wide)
            regimes = result[regime_col].dropna().astype(str).str.upper().unique()
            blocked = [r for r in regimes if r in self.cfg.blocked_regimes_list]
            if blocked:
                logger.warning(
                    "MARKET REGIME BLOCK: regime=%s is blocked — no trades today",
                    blocked[0],
                )
                from core.trading import notifications as notify
                notify.notify_error(
                    "Market Regime",
                    f"Market regime is {blocked[0]} — auto-trade BLOCKED. No buys today."
                )
                return pd.DataFrame()

        # Score band filter — regime-aware via policy.regime_score_floor
        # (same helper risk_manager + dashboard preview use). Single source
        # of truth — when CONFIG knobs change, all three paths update together.
        cur_regime = ""
        if regime_col and regime_col in result.columns:
            _r = result[regime_col].dropna().astype(str).str.upper()
            if not _r.empty:
                cur_regime = _r.iloc[0]
        from core.trading.policy import regime_score_floor
        _min_score = regime_score_floor(cur_regime, self.cfg)
        scores = pd.to_numeric(result[score_col], errors="coerce")
        result = result[
            (scores >= _min_score) &
            (scores <= self.cfg.max_score_to_trade)
        ]
        if result.empty:
            logger.info(
                "No stocks pass score filter (%.0f-%.0f, regime=%s)",
                _min_score, self.cfg.max_score_to_trade, cur_regime or "default",
            )
            return result

        # ML probability filter
        if ml_col and ml_col in result.columns:
            ml_vals = pd.to_numeric(result[ml_col], errors="coerce")
            before = len(result)
            result = result[ml_vals >= self.cfg.min_ml_prob]
            dropped = before - len(result)
            if dropped:
                logger.info("ML filter dropped %d stocks (ML < %.2f)",
                            dropped, self.cfg.min_ml_prob)
            if result.empty:
                logger.info("No stocks pass ML filter (>= %.2f)", self.cfg.min_ml_prob)
                return result

        # Sector blocklist filter
        blocked = self.cfg.blocked_sectors_list
        if blocked and sector_col and sector_col in result.columns:
            before = len(result)
            result = result[~result[sector_col].isin(blocked)]
            dropped = before - len(result)
            if dropped:
                logger.info("Sector filter dropped %d stocks (blocked: %s)",
                            dropped, blocked)

        # ── MOMENTUM-vs-SPY FILTER (added 2026-05-14 from SPY comparison) ──
        # 31-day forensic showed StockScout underperformed SPY by -1.41pp
        # despite a healthy PF 2.42. Root cause: our ranking favored
        # high-R:R Energy/value names while SPY's gains were dominated
        # by mega-cap tech (which has low R:R but persistent momentum).
        #
        # CAVEAT: our scan deliberately selects mean-reversion / pullback
        # setups, so scan-avg Return_5d ≈ 0% (below SPY). A strict filter
        # (e.g., SPY - 1pp) rejected 100% of candidates in initial test.
        # Solution: very loose default threshold (5pp lag) — only blocks
        # CATASTROPHIC relative weakness. The 5% ranking boost (below)
        # provides the real bias toward market-leaders.
        # Toggle: TRADE_MOMENTUM_VS_SPY_ENABLED (default true).
        # Threshold: TRADE_MOMENTUM_VS_SPY_MAX_LAG (default 5.0pp).
        # SPY fetch fails open — filter becomes a no-op.
        import os as _os
        mom_enabled = _os.getenv("TRADE_MOMENTUM_VS_SPY_ENABLED", "true").lower() in ("1", "true", "yes")
        mom_lag_max = float(_os.getenv("TRADE_MOMENTUM_VS_SPY_MAX_LAG", "5.0"))
        ret5d_col = self._find_col(result, ["Return_5d", "Returns_5d", "5d_Return"])
        if mom_enabled and ret5d_col and ret5d_col in result.columns:
            spy_5d = _fetch_spy_5d_return()
            if spy_5d != 0.0:  # successful fetch (0.0 means failure)
                # Allow within `mom_lag_max` pp under SPY (small underperformance OK)
                # but reject anything more
                threshold = spy_5d - mom_lag_max
                ret_vals = pd.to_numeric(result[ret5d_col], errors="coerce")
                # NaN rows pass (don't penalize missing data)
                mom_ok = ret_vals.isna() | (ret_vals >= threshold)
                before = len(result)
                result = result[mom_ok]
                dropped = before - len(result)
                if dropped:
                    logger.info(
                        "Momentum-vs-SPY filter: SPY 5d=%+.2f%%, threshold=%+.2f%%, "
                        "dropped %d candidate(s) lagging market",
                        spy_5d, threshold, dropped,
                    )
                if result.empty:
                    logger.info(
                        "No candidates pass momentum-vs-SPY filter (SPY 5d=%+.2f%%)",
                        spy_5d,
                    )
                    return result

        # RR filter — uses _effective_rr if available (computed above with
        # stop_distance floor 3%), otherwise raw rr_col. Forensic analysis
        # 2026-05-14 showed scan-tight stops produce R:R 10+ that
        # over-promotes lottery-ticket candidates.
        if "_effective_rr" in result.columns:
            rr_vals = pd.to_numeric(result["_effective_rr"], errors="coerce")
            # Fall back to raw rr_col when effective is NaN (e.g., CF
            # had stop > entry, effective math goes weird)
            if rr_col and rr_col in result.columns:
                raw_rr = pd.to_numeric(result[rr_col], errors="coerce")
                rr_vals = rr_vals.fillna(raw_rr)
            result = result[rr_vals >= self.cfg.min_rr_to_trade]
            if result.empty:
                logger.info(
                    "No stocks pass effective-RR filter (>= %.1f)",
                    self.cfg.min_rr_to_trade,
                )
                return result
        elif rr_col and rr_col in result.columns:
            rr_vals = pd.to_numeric(result[rr_col], errors="coerce")
            result = result[rr_vals >= self.cfg.min_rr_to_trade]
            if result.empty:
                logger.info("No stocks pass RR filter (>= %.1f)", self.cfg.min_rr_to_trade)
                return result

        # Confidence filter — regime-aware. In TREND_UP / MODERATE_UP markets
        # Medium-quality setups are tradable (the macro tailwind compensates
        # for thinner per-stock confirmation). In SIDEWAYS we keep the High
        # bar; DISTRIBUTION is even stricter (cfg default already "High",
        # but we don't lower it). PANIC/CORRECTION are blocked entirely
        # upstream by blocked_regimes.
        if confidence_col and confidence_col in result.columns:
            # Confidence floor — regime-aware via policy.confidence_floor
            # (single source of truth — preview and production agree).
            from core.trading.policy import confidence_floor as _cf
            from core.trading.policy import _CONF_MAP as conf_map
            cur_regime = ""
            if regime_col and regime_col in result.columns:
                _r = result[regime_col].dropna().astype(str).str.upper()
                if not _r.empty:
                    cur_regime = _r.iloc[0]
            min_conf_val = _cf(cur_regime, self.cfg)
            conf_vals = result[confidence_col].astype(str).str.upper().map(
                lambda x: conf_map.get(x, 0)
            )
            before = len(result)
            result = result[conf_vals >= min_conf_val]
            dropped = before - len(result)
            if dropped:
                _floor_label = {3: "High", 2: "Medium", 1: "Low"}.get(min_conf_val, "?")
                logger.info(
                    "Confidence filter dropped %d stocks (< %s, regime=%s)",
                    dropped, _floor_label, cur_regime or "default",
                )
            if result.empty:
                logger.info(
                    "No stocks pass confidence filter (regime=%s)",
                    cur_regime or "default",
                )
                return result

        # Reliability filter — only trade stocks with reliable data
        if reliability_col and reliability_col in result.columns:
            rel_vals = pd.to_numeric(result[reliability_col], errors="coerce")
            before = len(result)
            result = result[rel_vals >= self.cfg.min_reliability]
            dropped = before - len(result)
            if dropped:
                logger.info("Reliability filter dropped %d stocks (< %.0f)",
                            dropped, self.cfg.min_reliability)
            if result.empty:
                logger.info("No stocks pass reliability filter (>= %.0f)", self.cfg.min_reliability)
                return result

        # Remove already-held tickers (check both tracker AND live IBKR positions)
        ibkr_held = set()
        try:
            for p in self.client.get_positions():
                if p.quantity > 0:
                    ibkr_held.add(p.ticker)
        except Exception:
            pass  # If not connected yet, rely on tracker only
        result = result[
            ~result[ticker_col].apply(
                lambda t: self.tracker.is_holding(t) or t in ibkr_held
            )
        ]

        logger.info("Smart filter: %d → %d candidates", initial_count, len(result))

        # ── Combined rank — SINGLE-PASS WEIGHTED SUM ──
        # Audit H5 (2026-05-01): the previous version applied multipliers
        # SEQUENTIALLY:
        #     rank = base(45/25/20) → rank * 0.9 + sector * 0.1
        #            → rank * 0.9 + insider * 0.1
        # That produced an effective weighting of:
        #     score   = 0.45 * 0.9 * 0.9 = 0.3645  (advertised: 0.45)
        #     RR      = 0.25 * 0.9 * 0.9 = 0.2025  (advertised: 0.25)
        #     ML      = 0.20 * 0.9 * 0.9 = 0.1620  (advertised: 0.20)
        #     sector  = 0.10 * 0.9       = 0.0900  (advertised: 0.10)
        #     insider = 0.10             = 0.1000  (advertised: 0.10)
        #     ─────────────────────────────────────
        #     total                    = 0.9190 ≠ 1.0
        # The headline weights AND the total were both wrong.
        #
        # Now: collect every available signal as a (weight, normalized series)
        # pair, then compute the weighted sum in ONE pass at the end.
        # Weights are renormalized to sum to exactly 1.0 across whatever
        # signals are present this run (so missing data degrades gracefully
        # without silently re-distributing weight).
        def _norm(series):
            s = pd.to_numeric(series, errors="coerce")
            rng = s.max() - s.min()
            if pd.isna(rng) or rng < 1e-9:
                return s * 0.0 + 0.5  # all-equal → neutral 0.5
            return (s - s.min()) / rng

        # Documented-intent weights (must sum to 1.0 when all signals present).
        WEIGHT_SCORE = 0.45
        WEIGHT_RR = 0.25
        WEIGHT_ML = 0.20
        WEIGHT_SECTOR = 0.05
        WEIGHT_INSIDER = 0.05

        signals = [(WEIGHT_SCORE, _norm(result[score_col]))]

        # RANKING: prefer effective_rr (stop-distance floored at 3%) over
        # raw rr_col. Then ALSO cap at 5 — even if a ticker has wide stop
        # but absurdly high target (target $200 from $30 entry = 5x+ raw R:R
        # but no stop inflation), prevent one extreme from compressing
        # everyone else into the bottom half. Filter (min_rr_to_trade=2)
        # is unaffected — it uses the same normalized value.
        rr_for_ranking_src = None
        if "_effective_rr" in result.columns:
            rr_for_ranking_src = result["_effective_rr"]
        elif rr_col and rr_col in result.columns:
            rr_for_ranking_src = result[rr_col]
        if rr_for_ranking_src is not None:
            rr_for_ranking = pd.to_numeric(rr_for_ranking_src, errors="coerce").clip(upper=5.0)
            signals.append((WEIGHT_RR, _norm(rr_for_ranking)))

        # MOMENTUM-vs-SPY ranking BOOST (added 2026-05-14).
        # In addition to the hard filter above, give a small ranking weight
        # to candidates that BEAT SPY in the recent 5d. Helps prefer
        # market-leaders within the surviving pool. Weight kept modest
        # (5%) so it doesn't dominate fundamental signals. Disabled if
        # the SPY fetch failed (filter would also be disabled).
        if (mom_enabled and ret5d_col and ret5d_col in result.columns
                and _SPY_5D_CACHE.get("return") is not None):
            spy_5d_val = _SPY_5D_CACHE["return"]
            # Per-candidate "momentum surplus" vs SPY (+ = outperform)
            surplus = pd.to_numeric(result[ret5d_col], errors="coerce") - spy_5d_val
            # Cap so one super-strong outlier doesn't compress the rest
            surplus_capped = surplus.clip(-10.0, 10.0)
            WEIGHT_MOM = 0.05
            signals.append((WEIGHT_MOM, _norm(surplus_capped)))

        if ml_col and ml_col in result.columns:
            signals.append((WEIGHT_ML, _norm(result[ml_col])))

        # Sector momentum signal (positive ranking input, not just block).
        if sector_col and sector_col in result.columns:
            try:
                sector_boost = self._compute_sector_momentum_boost(result[sector_col])
                signals.append((WEIGHT_SECTOR, sector_boost))
            except Exception as _e:
                logger.debug("sector momentum boost skipped: %s", _e)

        # ── LIVE-WR SECTOR AWARENESS (Audit H7, 2026-05-01) ──
        # Adjust rank by historical performance OF THE SAME SECTOR in
        # this strategy's live trade log. Idea: if we've lost 5/5 in
        # Energy lately, rank Energy candidates LOWER even when their
        # forward-looking signals (score, RR, ML) are strong — the live
        # data is telling us this strategy isn't capturing those wins.
        #
        # Implementation: compute per-sector P&L over the last
        # `live_wr_window` closed trades, normalize to [0..1] across
        # all sectors present, give it a small (default 5%) weight in
        # the rank. New sectors with no trade history get 0.5 (neutral).
        # Disabled when there are <5 closed trades total (premature).
        if sector_col and sector_col in result.columns:
            try:
                LIVE_WR_WINDOW = 30
                LIVE_WR_WEIGHT = 0.05
                trade_log = self.tracker.get_trade_log()
                closes = [
                    t for t in trade_log
                    if t.get("action") == "CLOSE"
                    and t.get("pnl") is not None
                ]
                if len(closes) >= 5:
                    recent_closes = closes[-LIVE_WR_WINDOW:]
                    sector_pnls: Dict[str, list] = {}
                    for c in recent_closes:
                        sec_c = str(c.get("sector", "") or "")
                        if not sec_c:
                            # Look up from open_positions history if tracker
                            # didn't persist sector on the CLOSE row.
                            continue
                        sector_pnls.setdefault(sec_c, []).append(
                            float(c.get("pnl") or 0)
                        )
                    # Average pnl per sector → normalize to [0..1]
                    if sector_pnls:
                        avg_by_sector = {s: sum(v) / len(v) for s, v in sector_pnls.items()}
                        vals = list(avg_by_sector.values())
                        rng = max(vals) - min(vals)
                        if rng > 1e-9:
                            sector_live_wr = {
                                s: (v - min(vals)) / rng
                                for s, v in avg_by_sector.items()
                            }
                        else:
                            sector_live_wr = {s: 0.5 for s in avg_by_sector}
                        # Map per-row; unknown sectors → 0.5 (neutral)
                        live_wr_series = result[sector_col].astype(str).map(
                            lambda s: sector_live_wr.get(s, 0.5)
                        ).astype(float)
                        signals.append((LIVE_WR_WEIGHT, live_wr_series))
                        logger.info(
                            "LIVE-WR sector boost: %s",
                            ", ".join(
                                f"{s}={sector_live_wr.get(s, 0.5):.2f}"
                                for s in avg_by_sector
                            ),
                        )
            except Exception as _e:
                logger.debug("live-WR sector ranking skipped: %s", _e)

        # Insider buying signal (Form 4 from SEC EDGAR). Only computed
        # for the top-20 candidates by base rank to cap API spend; the
        # rest get 0, which is the unbiased neutral for an additive signal.
        if self.cfg.insider_signal_enabled:
            try:
                # Compute a provisional rank (score-only) to pick the top 20
                provisional_rank = _norm(result[score_col])
                top_idx = provisional_rank.sort_values(ascending=False).head(20).index
                from core.data.insider_signal import insider_score as _ins_score
                insider_vals = []
                for idx in result.index:
                    if idx in top_idx:
                        ticker = str(result.loc[idx, ticker_col])
                        insider_vals.append(_ins_score(ticker))
                    else:
                        insider_vals.append(0.0)
                ins_series = pd.Series(insider_vals, index=result.index)
                signals.append((WEIGHT_INSIDER, ins_series))
                if (ins_series > 0).any():
                    boosted = result[ins_series > 0][ticker_col].tolist()
                    logger.info("INSIDER BOOST: %s have insider buying", boosted)
            except Exception as _e:
                logger.debug("insider signal skipped: %s", _e)

        # Renormalize weights so they sum to 1.0 across whatever signals
        # are actually present (graceful degradation: a yfinance outage
        # disabling sector momentum won't silently shift weight to insider).
        total_weight = sum(w for w, _ in signals)
        if total_weight <= 0:
            rank = _norm(result[score_col])  # fallback
        else:
            rank = sum((w / total_weight) * series for w, series in signals)

        # Log the effective weights for observability — invisible weight
        # drift bugs (like the previous compounded multipliers) will show
        # up immediately in the logs after this commit.
        try:
            effective = ", ".join(
                f"{name}={w/total_weight:.2f}"
                for name, (w, _) in zip(
                    ["score", "rr", "ml", "sector", "insider"][:len(signals)],
                    signals,
                )
            )
            logger.info("RANK WEIGHTS (effective): %s (total signals: %d)",
                        effective, len(signals))
        except Exception:
            pass

        result["_rank_score"] = rank
        result = result.sort_values("_rank_score", ascending=False)
        result = result.drop(columns=["_rank_score"])

        return result

    def _compute_sector_momentum_boost(self, sector_series) -> "pd.Series":
        """Map each row's sector to a 30-day-momentum-based [0,1] score.
        Strong sectors (XLE +5%) get ~1.0, weak sectors (XLE -5%) get ~0.0.

        Cached per-call: only fetches each unique ETF once per filter run.
        Falls back to neutral 0.5 on errors.
        """
        import pandas as pd
        sector_to_etf = {
            "Technology": "XLK", "Communication Services": "XLC",
            "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
            "Financial Services": "XLF", "Health Care": "XLV",
            "Healthcare": "XLV", "Industrials": "XLI",
            "Real Estate": "XLRE", "Utilities": "XLU",
            "Materials": "XLB", "Basic Materials": "XLB",
            "Energy": "XLE",
        }
        # Cache momentum per unique ETF
        unique_sectors = set(s for s in sector_series.dropna().unique() if s)
        etf_momentum = {}
        try:
            import yfinance as yf
            for sec in unique_sectors:
                etf = sector_to_etf.get(sec)
                if etf is None:
                    etf_momentum[sec] = 0.5  # unknown sector
                    continue
                try:
                    # Same timeout pattern as _fetch_analyst_target — yfinance
                    # .history() can also hang on slow CDN responses.
                    def _fetch_etf_hist(_etf=etf):
                        return yf.Ticker(_etf).history(period="35d", interval="1d")
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _ETO
                    try:
                        with ThreadPoolExecutor(max_workers=1) as _ex:
                            hist = _ex.submit(_fetch_etf_hist).result(timeout=8.0)
                    except _ETO:
                        logger.warning(
                            "Sector ETF %s history fetch TIMED OUT — using neutral 0.5",
                            etf,
                        )
                        etf_momentum[sec] = 0.5
                        continue
                    if hist is None or len(hist) < 25:
                        etf_momentum[sec] = 0.5
                        continue
                    closes = hist["Close"].dropna()
                    if len(closes) < 25:
                        etf_momentum[sec] = 0.5
                        continue
                    mom_30d = (closes.iloc[-1] / closes.iloc[-25] - 1) * 100
                    # Map -10%..+10% to 0..1 via clipping; neutral at 0% → 0.5
                    boost = max(0.0, min(1.0, (mom_30d + 10) / 20))
                    etf_momentum[sec] = boost
                except Exception:
                    etf_momentum[sec] = 0.5
        except Exception:
            return pd.Series([0.5] * len(sector_series), index=sector_series.index)
        return sector_series.map(lambda s: etf_momentum.get(s, 0.5)).astype(float)

    def _execute_single(self, row: pd.Series) -> Dict:
        """Execute a single recommendation: buy + trailing stop + limit sell."""
        ticker = row.get("Ticker", row.get("ticker", ""))
        score = float(row.get("FinalScore_20d", row.get("Score", 0)))
        rr = float(row.get("RewardRisk", row.get("RR", 0)))
        entry = float(row.get("Entry_Price", row.get("entry_price", 0)))
        target = float(row.get("Target_Price", row.get("target_price", 0)))
        stop = float(row.get("Stop_Loss", row.get("stop_loss", 0)))
        # Compute target exit date from HoldingDays or Target_Date
        _raw_target_date = row.get("Target_Date", row.get("target_date", ""))
        _holding_days = row.get("HoldingDays", row.get("holding_days", 0))
        if _raw_target_date and str(_raw_target_date).strip():
            target_date = str(_raw_target_date)
        elif _holding_days and int(_holding_days) > 0:
            from datetime import datetime, timedelta
            target_date = (datetime.utcnow() + timedelta(days=int(_holding_days))).strftime("%Y-%m-%d")
        else:
            target_date = ""

        # EARNINGS-AWARE TARGET DATE (added 2026-05-05).
        # Real-world failure today: ELVN bought with target_date 2026-05-20
        # but earnings 2026-05-13 — earnings block-gate (5 days) wasn't
        # triggered since 8 > 5, but we'd still hold THROUGH earnings,
        # exposing the position to overnight gap risk on a $543 trade
        # (a -10% gap = $54 loss, ~10% of the position). Solution: cap
        # target_date at earnings_date - 1 day if earnings falls within
        # the planned hold horizon. The position auto-exits day-before
        # earnings; if we want to ride through, operator can /resubmit
        # with explicit override.
        if target_date:
            try:
                from datetime import datetime as _dt, date as _date, timedelta as _td
                _t = _dt.strptime(target_date, "%Y-%m-%d").date()
                # Try yfinance for next earnings date — wrapped in same
                # ThreadPoolExecutor timeout pattern as analyst PT fetch
                # so a slow/hung yfinance doesn't block the trade.
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as _ETO
                def _fetch_earn(_t=ticker):
                    import yfinance as yf
                    cal = yf.Ticker(_t).calendar or {}
                    eds = cal.get("Earnings Date") if isinstance(cal, dict) else None
                    if eds and isinstance(eds, list) and eds:
                        return eds[0]
                    return None
                try:
                    with ThreadPoolExecutor(max_workers=1) as _ex:
                        ed = _ex.submit(_fetch_earn).result(timeout=5.0)
                except _ETO:
                    ed = None
                except Exception:
                    ed = None
                if ed and isinstance(ed, _date) and ed <= _t:
                    safe_t = ed - _td(days=1)
                    today = _date.today()
                    # Only apply if the safer date is still in the future
                    # (i.e. earnings is more than 1 day away). Otherwise
                    # we'd set target_date in the past — let the buy
                    # itself be re-evaluated by the earnings_block gate.
                    if safe_t > today:
                        original_td = target_date
                        target_date = safe_t.strftime("%Y-%m-%d")
                        logger.info(
                            "Earnings-aware target_date for %s: %s → %s "
                            "(earnings on %s, exit 1 day prior)",
                            ticker, original_td, target_date, ed.isoformat(),
                        )
            except Exception as _ee:
                logger.debug("Earnings-date check failed for %s (non-fatal): %s",
                             ticker, _ee)

        # Use current price as entry estimate if Entry_Price not available
        price = entry if entry > 0 else float(row.get("Close", row.get("close", 0)))
        scan_price = price  # remember scan-time price for gap_guard/slippage compare

        # ── LIVE PRICE REFRESH ────────────────────────────────────────────
        # Scan completed 60-90 minutes ago; prices have moved. Professional
        # algo systems (AQR, Renaissance, Quantopian's `data.current()`)
        # separate signal generation (slow) from execution-time price
        # discovery (fast). We do the same: fetch the latest quote from
        # IB and use it as our entry price, then re-derive stop/target
        # PROPORTIONALLY so R:R stays as the scan intended.
        # Falls back to scan price on any quote failure.
        try:
            live_price = self.client.get_live_price(ticker)
        except Exception as _live_err:
            logger.debug("live price fetch raised for %s: %s", ticker, _live_err)
            live_price = None
        if live_price and live_price > 0 and scan_price > 0:
            move_pct = (live_price - scan_price) / scan_price * 100

            # SLIPPAGE HARD-REJECT — refuse to trade against a price that
            # has moved >5% from scan time. Beyond that we're chasing a
            # different stock than the one the scan analyzed (halt+reopen,
            # gap-up news, etc). Proportional rescaling of stop/target
            # masks the problem; backtests showed runaway losses on these
            # entries. (Audit 2026-04-30 finding #3.)
            if abs(move_pct) > 5.0:
                logger.warning(
                    "SLIPPAGE REJECT %s: scan $%.2f → live $%.2f (%+.2f%%, > 5%% threshold)",
                    ticker, scan_price, live_price, move_pct,
                )
                return {
                    "ticker": ticker, "status": "skipped",
                    "reason": (
                        f"Live price ${live_price:.2f} moved {move_pct:+.1f}% "
                        f"from scan ${scan_price:.2f} (> 5% reject threshold)"
                    ),
                }

            # Re-derive stop & target using the scan's intended R:R/stop ratios
            # so the trade preserves its risk profile around the live price.
            if stop > 0:
                stop_pct = (scan_price - stop) / scan_price  # positive number
                stop = round(live_price * (1 - stop_pct), 2)
            if target > 0:
                tgt_pct = (target - scan_price) / scan_price  # positive number
                target = round(live_price * (1 + tgt_pct), 2)
            logger.info(
                "LIVE REFRESH %s: scan $%.2f → live $%.2f (%+.2f%%) — "
                "stop adjusted to $%.2f, target to $%.2f",
                ticker, scan_price, live_price, move_pct, stop, target,
            )
            price = live_price  # downstream sizing/gating uses the live price

        # Analyst target cap — compare scan's target to Wall Street consensus.
        # If analyst mean < current price, the stock is rated overvalued —
        # refuse to trade.  Otherwise, cap our target at midpoint between
        # scan-target and analyst-mean so we don't target beyond consensus.
        adjusted_target = _cap_target_with_analysts(ticker, price, target)
        if adjusted_target is None:
            return {"ticker": ticker, "status": "skipped",
                    "reason": f"Analyst mean PT below current price (overvalued)"}
        if adjusted_target < target:
            logger.info(
                "Target capped by analyst consensus for %s: "
                "scan=$%.2f → adjusted=$%.2f",
                ticker, target, adjusted_target,
            )
            target = adjusted_target

        # Extract extra context for risk checks
        sector = str(row.get("Sector", row.get("sector", "")))
        atr_pct = float(row.get("ATR_Pct", row.get("atr_pct", 0)) or 0)

        # Gap protection — skip if stock gapped unfavorably vs scan entry.
        # Tightened 2026-04-22 from ±3% to ±2%: commodity names (TDW, oil)
        # routinely gap 2-3% on macro news; previous 3% threshold missed
        # the TDW oil-crash entry that stopped out 3 days later.
        scan_close = float(row.get("Close", row.get("close", 0)) or 0)
        if scan_close > 0 and price > 0:
            gap_pct = (price - scan_close) / scan_close * 100
            if gap_pct > 2.0:
                return {"ticker": ticker, "status": "skipped",
                        "reason": f"Gap up {gap_pct:+.1f}% vs scan (entry risk too high)"}
            if gap_pct < -2.0:
                return {"ticker": ticker, "status": "skipped",
                        "reason": f"Gap down {gap_pct:+.1f}% vs scan (possible news event)"}

        # News catalyst gate — refuse to enter when the stock has had a
        # large 24h move on heavy volume. The technicals look great
        # (just rallied!) but the catalyst is already priced in — we'd
        # be the bagholder. Crude proxy: 24h price move > 8% AND volume
        # > 2× average. (Recommendation #5 from 2026-04-30 audit.)
        try:
            move_24h_pct = abs(float(row.get("Price_Change_1d_pct",
                                            row.get("Pct_Change_1d", 0)) or 0))
            vol_ratio = float(row.get("VolumeSurge",
                                       row.get("Volume_Surge", 1)) or 1)
            if move_24h_pct >= 8.0 and vol_ratio >= 2.0:
                logger.info(
                    "NEWS CATALYST SKIP %s: 24h move %.1f%% on %.1fx volume "
                    "— catalyst priced in",
                    ticker, move_24h_pct, vol_ratio,
                )
                return {"ticker": ticker, "status": "skipped",
                        "reason": (
                            f"News catalyst already priced in: "
                            f"24h move {move_24h_pct:.1f}% on {vol_ratio:.1f}x volume"
                        )}
        except Exception:
            pass

        # Risk check — now validates target/stop sanity too.
        # Pass market_regime so the score floor adjusts to the regime
        # (matches scoring_config.REGIME_MIN_SCORE + 5 buffer) instead of
        # using the static 73 that blocked all SIDEWAYS-day trades.
        _row_regime = str(row.get("Market_Regime", "") or "").upper()
        _row_ml = float(row.get("ML_20d_Prob", row.get("ml_prob", 0)) or 0)
        _row_sq = str(row.get("SignalQuality", row.get("Confidence_Level", "")))
        _row_rel = float(row.get("Reliability_Score", row.get("Reliability", 100)) or 100)
        allowed, reason = self.risk.can_open_position(
            ticker, price, score, rr, sector=sector, atr_pct=atr_pct,
            stop_loss=stop, target_price=target, market_regime=_row_regime,
            ml_prob=_row_ml,
            signal_quality=_row_sq, reliability_score=_row_rel,
        )
        if not allowed:
            logger.info("SKIP %s: %s", ticker, reason)
            return {"ticker": ticker, "status": "skipped", "reason": reason}

        # Calculate quantity — cash + volatility + ML + throttle aware sizing.
        # Conviction (score+RR), ML probability, and the rolling-window
        # performance throttle all flow through here.
        cash = self.client.get_cash_balance()
        available_cash = max(0, cash - self.cfg.cash_reserve)
        qty = self.risk.calculate_qty(
            price, cash_available=available_cash, atr_pct=atr_pct,
            score=score, rr=rr, ml_prob=_row_ml,
            throttle_mult=getattr(self.risk, "_last_throttle_mult", 1.0),
        )
        if qty <= 0:
            return {"ticker": ticker, "status": "skipped",
                    "reason": f"Can't afford {ticker} @ ${price:.2f} (cash=${cash:.0f})"}

        # Reduce position size in cautious regimes (DISTRIBUTION)
        regime = str(row.get("Market_Regime", "")).upper()
        if regime in self.cfg.reduce_regimes_list:
            qty = max(1, qty // 2)
            logger.info("REGIME CAUTION: %s — reducing %s qty to %d", regime, ticker, qty)

        logger.info("EXECUTING: BUY %d x %s @ ~$%.2f (score=%.1f, RR=%.2f)",
                     qty, ticker, price, score, rr)

        # Calculate trailing stop % — volatility-adaptive AND regime-adaptive.
        # Blend 3 sources, then scale the result by a regime multiplier so
        # trends get wider trails (let winners run) while choppy / declining
        # regimes get tighter trails (cut faster). (Recommendation #4 from
        # 2026-04-30 audit.)
        #
        # Sources:
        #   1. Scan's stop loss → (price - stop) / price
        #   2. ATR-based (1.5 × ATR%) → dynamic per-stock volatility match
        #   3. Fallback: config default
        # Regime multiplier:
        #   TREND_UP / MODERATE_UP: 1.20 (wider — let winners run)
        #   SIDEWAYS / NEUTRAL:     1.00 (baseline)
        #   DISTRIBUTION:           0.85 (tighter — sellers active)
        #   CORRECTION / PANIC:     0.70 (much tighter — preserve capital)
        _trail_candidates = []
        if stop > 0 and price > 0:
            _trail_candidates.append(round((price - stop) / price * 100, 1))
        if atr_pct > 0:
            # 1.5x ATR gives stop that's wide enough to avoid noise, tight enough to protect
            _trail_candidates.append(round(atr_pct * 1.5, 1))
        if _trail_candidates:
            base_trail_pct = sum(_trail_candidates) / len(_trail_candidates)
        else:
            base_trail_pct = self.cfg.trailing_stop_pct

        # Regime adjustment
        # 2026-05-15 PARITY FIX: STRONG_UPTREND and UPTREND were missing
        # from the bullish list, defaulting to 1.0 instead of 1.20. ILMN
        # (bought today in STRONG_UPTREND) got a tighter trail (4.0%
        # instead of 4.8%) — exactly the trail-too-tight whipsaw setup.
        # policy.py already had these regimes in its bullish list at
        # line 537 — this brings order_manager into parity.
        regime_mult = 1.0
        _row_regime = str(row.get("Market_Regime", "") or "").upper()
        if _row_regime in ("TREND_UP", "MODERATE_UP", "BULLISH",
                           "STRONG_UPTREND", "UPTREND"):
            regime_mult = 1.20
        elif _row_regime in ("DISTRIBUTION",):
            regime_mult = 0.85
        elif _row_regime in ("CORRECTION", "BEARISH", "PANIC"):
            regime_mult = 0.70
        # SIDEWAYS / NEUTRAL / unknown → 1.0 baseline

        trail_pct = base_trail_pct * regime_mult
        # Floor/cap for safety. Floor is configurable via
        # TRADE_MIN_INITIAL_TRAIL_PCT (default 4.0) — protects against
        # day-1 noise stopouts on a 20-day swing thesis. The ratchet in
        # monitor_positions can still tighten BELOW this floor later
        # (tier 1=4%, tier 2=3%, tier 3=2%) once peak gains earn that
        # protection. Cap stays at 9% — even in raging bull, $300
        # positions don't need 12% trails ($36 of paper loss).
        initial_floor = float(getattr(self.cfg, "min_initial_trail_pct", 4.0))
        # ATR-based floor (added 2026-05-15) — prevents trail tighter than
        # the stock's normal daily volatility. Without this, ILMN (ATR 4.49%)
        # could get a 4.0% trail = guaranteed whipsaw on a single 1σ day.
        atr_floor_mult = float(getattr(self.cfg, "initial_trail_atr_floor_mult", 0.0))
        atr_floor = atr_pct * atr_floor_mult if (atr_pct > 0 and atr_floor_mult > 0) else 0
        effective_floor = max(initial_floor, atr_floor)
        trail_pct = max(effective_floor, min(trail_pct, 9.0))
        logger.info(
            "  Trail %.1f%% (base %.1f%% × regime %.2f, ATR %.1f%%, scan stop %.1f%%, "
            "atr_floor %.1f%%, regime=%s)",
            trail_pct, base_trail_pct, regime_mult,
            atr_pct if atr_pct > 0 else 0,
            (price - stop) / price * 100 if stop > 0 else 0,
            atr_floor,
            _row_regime or "default",
        )

        # Execute as OCA bracket: buy + trailing stop + limit sell (linked)
        bracket = self.client.buy_with_bracket(
            ticker=ticker,
            qty=qty,
            trail_pct=trail_pct,
            target_price=target,
        )

        buy_result = bracket["buy"]
        if buy_result.status in ("Error",):
            # Buy didn't fill (status="Error" from bracket_order's hard-reject
            # path means: timeout, rejection, or zero-quantity fill). Skip
            # add_position so we don't create a phantom OPEN entry that
            # later gets reconciled with a fake P&L.
            logger.warning(
                "Buy unfilled for %s — skipping tracker write: %s",
                ticker, buy_result.error,
            )
            try:
                notify.notify_error(
                    "Buy unfilled",
                    f"⚠️ {ticker} ${price:.2f} buy did not fill: "
                    f"{buy_result.error}. No position recorded; no protective "
                    f"orders placed (correctly). Likely cause: cash<$2k rule, "
                    f"insufficient buying power, or IB account restriction."
                )
            except Exception:
                pass
            return {"ticker": ticker, "status": "error",
                    "error": buy_result.error}

        # Use the ACTUAL filled quantity (in case of partial fill — bracket_order
        # already truncated the protective qty to match).
        actual_qty = buy_result.quantity or qty
        if actual_qty != qty:
            logger.warning(
                "%s partial fill: ordered %d, filled %d — recording %d in tracker",
                ticker, qty, actual_qty, actual_qty,
            )
        qty = actual_qty

        filled_price = buy_result.filled_price or price
        if filled_price <= 0:
            # Defensive — shouldn't happen given the bracket_order hard-reject,
            # but if it slips through, refusing to record is safer than recording
            # a $0 cost basis that breaks all P&L math downstream.
            logger.error(
                "%s: filled_price <= 0 (%.2f) despite Filled status — refusing add_position",
                ticker, filled_price,
            )
            return {"ticker": ticker, "status": "error",
                    "error": "zero filled_price"}

        # Slippage — actual vs scan-expected price. A consistent positive
        # slippage across many trades means we're getting worse fills than
        # the scan suggested; useful signal for tuning order type (MKT vs LMT)
        # or pre-trade gap protection.
        if filled_price > 0 and price > 0:
            slip_abs = filled_price - price
            slip_pct = slip_abs / price * 100
            if abs(slip_pct) >= 0.10:  # log only meaningful slippage (≥10bps)
                direction = "worse" if slip_abs > 0 else "better"
                logger.info(
                    "SLIPPAGE %s: expected $%.2f, filled $%.2f "
                    "(${%+.3f}, %+.2f%%, %s than expected)",
                    ticker, price, filled_price, slip_abs, slip_pct, direction,
                )
                # Notify on significant slippage (>50bps) — warns that market
                # conditions aren't matching the scan.
                if abs(slip_pct) >= 0.50:
                    try:
                        notify._send(
                            f"📉 <b>SLIPPAGE ALERT {ticker}</b>\n"
                            f"Expected: ${price:.2f}\n"
                            f"Filled: ${filled_price:.2f} "
                            f"({slip_pct:+.2f}%, {direction})"
                        )
                    except Exception:
                        pass

        # Validate protective orders — if rejected (margin etc), retry after fill
        trail_ok = bracket["trailing_stop"].status not in ("Error", "Cancelled", "Inactive")
        limit_ok = bracket["limit_sell"].status not in ("Error", "Cancelled", "Inactive")

        if not trail_ok or not limit_ok:
            logger.warning("Protective orders rejected for %s (trail=%s, limit=%s) — "
                           "will retry via monitor after fill",
                           ticker, bracket["trailing_stop"].status,
                           bracket["limit_sell"].status)
            notify.notify_error("Protection",
                f"⚠️ {ticker}: Protective orders REJECTED after buy! "
                f"Trail: {bracket['trailing_stop'].status}, "
                f"Limit: {bracket['limit_sell'].status}. "
                f"Monitor will auto-resubmit after fill.")

        order_ids = {
            "buy": buy_result.order_id,
            "trailing_stop": bracket["trailing_stop"].order_id,
            "limit_sell": bracket["limit_sell"].order_id,
            "oca_group": bracket.get("oca_group", ""),
        }

        # Step 4: Track position FIRST — before any user-facing notification.
        # Audit finding: previously notify_buy was sent BEFORE add_position.
        # If the tracker write failed, the user got "✅ BUY filled @ $X" in
        # Telegram with NO matching audit row — easy to miss the followup
        # error message and end up with a tracker/IB mismatch.
        # New ordering:
        #   1. add_position (tracker write)
        #   2. notify_buy (only on success — the message is ALWAYS backed
        #      by an audit row)
        #   3. on tracker failure: send a CRITICAL error notification
        #      instead of the routine BUY notification.
        # The position is still protected in IBKR by OCA either way; this
        # change only affects the Telegram message ordering/accuracy.
        ml_prob = float(row.get("ML_20d_Prob", row.get("ml_prob", 0)) or 0)
        _tracker_ok = False
        try:
            self.tracker.add_position(
                ticker=ticker,
                quantity=qty,
                entry_price=filled_price,
                stop_loss=stop,
                target_price=target,
                target_date=target_date if target_date else None,
                trailing_stop_pct=trail_pct,
                score=score,
                order_ids=order_ids,
                scan_price=scan_price,
            )
            _tracker_ok = True
        except Exception as _tracker_err:
            logger.error(
                "CRITICAL: position tracker add failed for %s after buy filled: %s",
                ticker, _tracker_err,
            )
            try:
                notify.notify_error(
                    "TRACKER WRITE FAILED",
                    f"🚨 {ticker} BOUGHT {qty}@${filled_price:.2f} "
                    f"(OCA {order_ids.get('oca_group','?')}) but position "
                    f"could NOT be written to tracker: {_tracker_err}\n\n"
                    f"Monitor is BLIND to this position. "
                    f"Manually add to data/trades/open_positions.json ASAP."
                )
            except Exception:
                pass
            # Don't re-raise — the position is protected in IBKR by OCA,
            # and the user has been alerted. Continue returning success
            # so the caller sees the buy went through.

        # Step 5: Send the BUY notification — only after the audit row is
        # safely on disk. If the tracker write failed above, the user
        # already got a more important CRITICAL alert; don't pile on a
        # second routine notification that suggests everything's fine.
        if _tracker_ok:
            notify.notify_buy(
                ticker, qty, filled_price, stop, target, score,
                trail_pct=trail_pct, rr=rr, target_date=target_date,
            )
        # Store extra data (sector, ml_prob) for monitor's exit logic
        try:
            _all = self.tracker.get_open_positions()
            for _p in _all:
                if _p["ticker"] == ticker:
                    _p["sector"] = sector
                    _p["entry_ml_prob"] = ml_prob
                    _p["entry_atr_pct"] = atr_pct
                    break
            self.tracker._save_positions(_all)
        except Exception:
            pass

        return {
            "ticker": ticker,
            "status": "success",
            "quantity": qty,
            "entry_price": filled_price,
            "target_price": target,
            "stop_loss": stop,
            "trailing_stop_pct": trail_pct,
            "order_ids": order_ids,
        }

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column name."""
        for c in candidates:
            if c in df.columns:
                return c
        return None
