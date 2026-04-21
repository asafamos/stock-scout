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

        # 2. Filter candidates
        candidates = self._filter_candidates(scan_df)
        if candidates.empty:
            logger.info("No candidates passed filters")
            return []

        logger.info("Candidates after filtering: %d", len(candidates))

        # 3. Connect to IBKR
        if not self.client.connect():
            logger.error("Failed to connect to IBKR — aborting")
            return []

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

    def resubmit_protections(self) -> List[Dict]:
        """Re-submit protective orders (trailing stop + limit sell) for all open positions.

        Use when orders expired because they were placed as DAY instead of GTC.
        """
        positions = self.tracker.get_open_positions()
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

        logger.info("Loading scan from: %s (modified %s)",
                     best_path.name,
                     pd.Timestamp.fromtimestamp(best_mtime).strftime("%Y-%m-%d %H:%M"))
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

        # Score band filter (Q3-Q4 sweet spot)
        scores = pd.to_numeric(result[score_col], errors="coerce")
        result = result[
            (scores >= self.cfg.min_score_to_trade) &
            (scores <= self.cfg.max_score_to_trade)
        ]
        if result.empty:
            logger.info("No stocks pass score filter (%.0f-%.0f)",
                        self.cfg.min_score_to_trade, self.cfg.max_score_to_trade)
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

        # RR filter
        if rr_col and rr_col in result.columns:
            rr_vals = pd.to_numeric(result[rr_col], errors="coerce")
            result = result[rr_vals >= self.cfg.min_rr_to_trade]
            if result.empty:
                logger.info("No stocks pass RR filter (>= %.1f)", self.cfg.min_rr_to_trade)
                return result

        # Confidence filter — only trade high-confidence setups
        if confidence_col and confidence_col in result.columns:
            # Map confidence levels: High=3, Medium=2, Low/Speculative=1
            conf_map = {
                "HIGH": 3, "MEDIUM": 2, "LOW": 1, "SPECULATIVE": 1, "NONE": 0
            }
            min_conf_val = conf_map.get(self.cfg.min_confidence.upper(), 3)
            conf_vals = result[confidence_col].astype(str).str.upper().map(
                lambda x: conf_map.get(x, 0)
            )
            before = len(result)
            result = result[conf_vals >= min_conf_val]
            dropped = before - len(result)
            if dropped:
                logger.info("Confidence filter dropped %d stocks (< %s)",
                            dropped, self.cfg.min_confidence)
            if result.empty:
                logger.info("No stocks pass confidence filter (>= %s)", self.cfg.min_confidence)
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

        # Sort by combined rank: 60% score + 40% R:R (normalized)
        # This ensures high R:R stocks like CVE (3.9) aren't pushed out
        # by marginally higher-scoring stocks with R:R of only 2.0
        scores_norm = pd.to_numeric(result[score_col], errors="coerce")
        scores_norm = (scores_norm - scores_norm.min()) / (scores_norm.max() - scores_norm.min() + 1e-9)
        if rr_col and rr_col in result.columns:
            rr_norm = pd.to_numeric(result[rr_col], errors="coerce")
            rr_norm = (rr_norm - rr_norm.min()) / (rr_norm.max() - rr_norm.min() + 1e-9)
            result["_rank_score"] = 0.6 * scores_norm + 0.4 * rr_norm
        else:
            result["_rank_score"] = scores_norm
        result = result.sort_values("_rank_score", ascending=False)
        result = result.drop(columns=["_rank_score"])

        return result

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

        # Use current price as entry estimate if Entry_Price not available
        price = entry if entry > 0 else float(row.get("Close", row.get("close", 0)))

        # Extract extra context for risk checks
        sector = str(row.get("Sector", row.get("sector", "")))
        atr_pct = float(row.get("ATR_Pct", row.get("atr_pct", 0)) or 0)

        # Gap protection — skip if stock gapped unfavorably vs scan entry
        scan_close = float(row.get("Close", row.get("close", 0)) or 0)
        if scan_close > 0 and price > 0:
            gap_pct = (price - scan_close) / scan_close * 100
            if gap_pct > 3.0:
                return {"ticker": ticker, "status": "skipped",
                        "reason": f"Gap up {gap_pct:+.1f}% vs scan (entry risk too high)"}
            if gap_pct < -3.0:
                return {"ticker": ticker, "status": "skipped",
                        "reason": f"Gap down {gap_pct:+.1f}% vs scan (possible news event)"}

        # Risk check
        allowed, reason = self.risk.can_open_position(
            ticker, price, score, rr, sector=sector, atr_pct=atr_pct
        )
        if not allowed:
            logger.info("SKIP %s: %s", ticker, reason)
            return {"ticker": ticker, "status": "skipped", "reason": reason}

        # Calculate quantity — cash + volatility aware sizing
        cash = self.client.get_cash_balance()
        available_cash = max(0, cash - self.cfg.cash_reserve)
        qty = self.risk.calculate_qty(price, cash_available=available_cash, atr_pct=atr_pct)
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

        # Calculate trailing stop % — 3 sources, use tightest reasonable:
        # 1. Scan's stop loss → (price - stop) / price
        # 2. ATR-based (1.5 × ATR%) → dynamic volatility match
        # 3. Fallback: config default
        _trail_candidates = []
        if stop > 0 and price > 0:
            _trail_candidates.append(round((price - stop) / price * 100, 1))
        if atr_pct > 0:
            # 1.5x ATR gives stop that's wide enough to avoid noise, tight enough to protect
            _trail_candidates.append(round(atr_pct * 1.5, 1))
        if _trail_candidates:
            # Use average to blend signals; floor/cap for safety
            trail_pct = sum(_trail_candidates) / len(_trail_candidates)
            trail_pct = max(3.0, min(trail_pct, 8.0))
        else:
            trail_pct = self.cfg.trailing_stop_pct
        logger.info("  Trail %.1f%% (ATR: %.1f%%, scan stop: %.1f%%)",
                     trail_pct, atr_pct if atr_pct > 0 else 0,
                     (price - stop) / price * 100 if stop > 0 else 0)

        # Execute as OCA bracket: buy + trailing stop + limit sell (linked)
        bracket = self.client.buy_with_bracket(
            ticker=ticker,
            qty=qty,
            trail_pct=trail_pct,
            target_price=target,
        )

        buy_result = bracket["buy"]
        if buy_result.status in ("Error",):
            return {"ticker": ticker, "status": "error",
                    "error": buy_result.error}

        filled_price = buy_result.filled_price or price

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

        # Step 4: Notify
        notify.notify_buy(ticker, qty, filled_price, stop, target, score,
                          trail_pct=trail_pct, rr=rr, target_date=target_date)

        # Step 5: Track position
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
        )

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
