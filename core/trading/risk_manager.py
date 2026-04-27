"""Pre-trade risk checks — gate every order through here."""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Optional, Tuple

from core.trading.config import CONFIG
from core.trading.ibkr_client import IBKRClient
from core.trading.position_tracker import PositionTracker

logger = logging.getLogger(__name__)


class RiskManager:
    """Validates every trade against position limits and account state."""

    def __init__(self, client: IBKRClient, tracker: PositionTracker,
                 config=None):
        self.client = client
        self.tracker = tracker
        self.cfg = config or CONFIG

    def check_daily_loss_breaker(self) -> Tuple[bool, str]:
        """Return (allowed, reason). Blocks new buys if today's P&L < -max_daily_loss_pct.

        Uses a short-lived cache to smooth out stale portfolio() reads, and
        when the threshold is breached, re-checks once (IB prices can lag
        by 1-2 min — a single stale snapshot shouldn't halt trading).
        """
        def _compute_pct_and_parts():
            net = self.client.get_net_liquidation()
            if net <= 0:
                return None, 0.0, 0.0, 0.0
            today = date.today().isoformat()
            realized = 0.0
            for t in self.tracker.get_trade_log():
                if t.get("action") == "CLOSE" and str(t.get("timestamp", "")).startswith(today):
                    realized += float(t.get("pnl", 0) or 0)
            unrealized = 0.0
            try:
                for p in self.client._ib.portfolio():
                    if p.position != 0:
                        unrealized += float(p.unrealizedPNL or 0)
            except Exception:
                pass
            total_today = realized + unrealized
            pct = (total_today / net) * 100
            return pct, realized, unrealized, net

        try:
            pct, realized, unrealized, net = _compute_pct_and_parts()
            if pct is None:
                return True, ""

            if pct <= -self.cfg.max_daily_loss_pct:
                # Confirm before blocking — IB prices can lag; don't halt on
                # a single stale reading. Wait 3s and recompute.
                import time as _time
                _time.sleep(3)
                pct2, realized, unrealized, _ = _compute_pct_and_parts()
                if pct2 is not None and pct2 <= -self.cfg.max_daily_loss_pct:
                    return False, (
                        f"Daily loss breaker: P&L {pct2:+.2f}% <= -{self.cfg.max_daily_loss_pct}% "
                        f"(realized ${realized:.0f} + unrealized ${unrealized:.0f}, "
                        f"confirmed on recheck)"
                    )
                logger.info(
                    "Daily loss breaker recovered on recheck (first=%+.2f%%, second=%+.2f%%)",
                    pct, pct2 or 0.0,
                )
            return True, ""
        except Exception as e:
            logger.warning("Daily loss breaker check failed: %s", e)
            return True, ""  # Fail open (don't block trades on error)

    def check_sector_concentration(self, new_sector: str) -> Tuple[bool, str]:
        """Block if we'd exceed max_sector_positions in same sector."""
        if not new_sector:
            return True, ""
        # Count existing positions in same sector
        same_sector = 0
        for p in self.tracker.get_open_positions():
            if str(p.get("sector", "")).strip().lower() == new_sector.strip().lower():
                same_sector += 1
        if same_sector >= self.cfg.max_sector_positions:
            return False, (
                f"Sector concentration: already {same_sector} positions in {new_sector} "
                f"(max {self.cfg.max_sector_positions})"
            )
        return True, ""

    def check_portfolio_correlation(self, new_ticker: str) -> Tuple[bool, str, float]:
        """Block if adding new_ticker would push mean pairwise correlation
        above the configured threshold.

        Uses 60-day daily returns. If we hold <2 positions, always allows
        (single pair not meaningful). If data can't be fetched, allows
        (fail-open — don't block on infrastructure failures).

        Returns (allowed, reason, projected_mean_corr).
        """
        if not getattr(self.cfg, "correlation_check_enabled", True):
            return True, "", 0.0

        open_positions = self.tracker.get_open_positions()
        if len(open_positions) < 1:
            return True, "", 0.0  # portfolio empty → no correlation issue

        # Build ticker universe: existing + proposed
        existing_tickers = [p["ticker"] for p in open_positions]
        if new_ticker in existing_tickers:
            return True, "", 0.0  # already held — filtered elsewhere
        universe = existing_tickers + [new_ticker]
        if len(universe) < 2:
            return True, "", 0.0

        try:
            import yfinance as yf
            import numpy as np
            data = yf.download(
                universe, period="90d", interval="1d",
                progress=False, auto_adjust=True, group_by="ticker",
            )
            # Build returns dataframe
            closes = {}
            for t in universe:
                try:
                    col = data[t]["Close"] if t in data.columns.get_level_values(0) else data["Close"]
                    closes[t] = col.dropna()
                except Exception:
                    continue
            if len(closes) < 2:
                return True, "", 0.0  # not enough data — fail open
            # Align on dates
            import pandas as pd
            df = pd.DataFrame(closes).dropna()
            if len(df) < 30:
                return True, "", 0.0  # need ≥30 days for meaningful corr
            returns = df.pct_change().dropna()
            corr_matrix = returns.corr()
            # Mean of upper triangle (excluding diagonal)
            mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
            pairs = corr_matrix.where(mask).stack()
            if pairs.empty:
                return True, "", 0.0
            mean_corr = float(pairs.mean())

            thresh = float(self.cfg.max_portfolio_correlation)
            if mean_corr > thresh:
                # Identify the most-correlated existing peer for a helpful message
                try:
                    new_corrs = corr_matrix[new_ticker].drop(new_ticker).sort_values(ascending=False)
                    top_peer = new_corrs.index[0]
                    top_val = float(new_corrs.iloc[0])
                    detail = f" (highest: {top_peer} ρ={top_val:+.2f})"
                except Exception:
                    detail = ""
                return False, (
                    f"Portfolio correlation too high: "
                    f"mean ρ={mean_corr:+.2f} > {thresh:.2f}{detail}"
                ), mean_corr
            logger.info(
                "Correlation check OK: adding %s → mean ρ=%+.2f (max %.2f)",
                new_ticker, mean_corr, thresh,
            )
            return True, "", mean_corr
        except Exception as e:
            logger.warning("Correlation check failed for %s: %s", new_ticker, e)
            return True, "", 0.0  # fail-open

    def check_sector_momentum(self, sector: str) -> Tuple[bool, str, float]:
        """Check sector ETF momentum — block new buys into weak sectors.

        Uses the SPDR sector ETFs (XLE, XLF, XLK, etc.) as proxies. If a
        sector is down >5% over the past 30 trading days OR down >2% today,
        we refuse new entries in that sector. This directly addresses the
        TDW oil-crash scenario: XLE would have been down heavily and the
        entry would have been blocked.

        Returns (allowed, reason, sector_mom_pct).
        """
        if not sector:
            return True, "", 0.0

        # Sector name → SPDR ETF mapping (Yahoo tickers). Covers Yahoo
        # Finance's standard sector names (what our scan uses).
        etf_map = {
            "technology":            "XLK",
            "healthcare":            "XLV",
            "financial services":    "XLF",
            "communication services": "XLC",
            "consumer cyclical":     "XLY",
            "consumer defensive":    "XLP",
            "industrials":           "XLI",
            "energy":                "XLE",
            "utilities":             "XLU",
            "real estate":           "XLRE",
            "basic materials":       "XLB",
        }
        etf = etf_map.get(sector.strip().lower())
        if not etf:
            return True, "", 0.0  # unknown sector → no gate

        try:
            import yfinance as yf
            hist = yf.Ticker(etf).history(period="45d", interval="1d")
            if hist is None or len(hist) < 25:
                return True, "", 0.0  # not enough data to judge
            closes = hist["Close"].dropna()
            if len(closes) < 25:
                return True, "", 0.0
            ret_30d = (closes.iloc[-1] / closes.iloc[-25] - 1) * 100
            ret_1d = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100 if len(closes) >= 2 else 0
            if ret_30d < -5.0:
                return False, (
                    f"Sector momentum weak: {sector} ({etf}) "
                    f"down {ret_30d:+.1f}% over 30d (< -5%)"
                ), ret_30d
            if ret_1d < -2.0:
                return False, (
                    f"Sector momentum weak: {sector} ({etf}) "
                    f"down {ret_1d:+.1f}% today (< -2%)"
                ), ret_30d
            logger.info(
                "Sector %s momentum OK: 30d=%+.1f%%, 1d=%+.1f%%",
                sector, ret_30d, ret_1d,
            )
            return True, "", ret_30d
        except Exception as e:
            logger.warning("Sector momentum check failed for %s: %s", sector, e)
            return True, "", 0.0  # fail-open on data failure

    def can_open_position(
        self,
        ticker: str,
        price: float,
        score: float = 0.0,
        rr: float = 0.0,
        sector: str = "",
        atr_pct: float = 0.0,
        stop_loss: float = 0.0,
        target_price: float = 0.0,
        market_regime: str = "",
    ) -> Tuple[bool, str]:
        """Return (allowed, reason). Reason is empty string if allowed.

        market_regime: when provided, the minimum-score gate becomes
        regime-aware (matches scoring_config.REGIME_MIN_SCORE + small
        buffer). Without this, a static threshold of 73 was blocking
        ALL SIDEWAYS-day trades because the scan's regime-adjusted
        scores top out around 70 in those markets.
        """

        # 0. Trade levels sanity — refuse buys with missing/absurd stops or targets.
        # Protects against scan rows where stop_loss/target_price are missing,
        # zero, NaN, or on the wrong side of entry (would leave the position
        # unbounded on upside or with a stop that can never trigger).
        # NaN guards are essential: pandas `to_numeric(..., errors="coerce")`
        # can silently produce NaN values. `NaN > 0` is False and `NaN <= 0`
        # is also False, so without explicit isnan() checks, a NaN would
        # slip through every comparison and IBKR would reject the order
        # with a cryptic "invalid auxPrice" error after the buy already filled.
        try:
            if math.isnan(price) or math.isnan(stop_loss) or math.isnan(target_price):
                return False, (
                    f"NaN in trade levels "
                    f"(price={price}, stop={stop_loss}, target={target_price})"
                )
        except (TypeError, ValueError):
            return False, "Non-numeric trade levels"
        if price <= 0:
            return False, f"Invalid price (${price:.2f})"
        MIN_TARGET_MULT = 1.02   # target ≥ entry × 1.02 (at least +2%)
        MAX_STOP_MULT   = 0.995  # stop  ≤ entry × 0.995 (at least -0.5%)
        if target_price <= 0 or target_price < price * MIN_TARGET_MULT:
            return False, (
                f"Invalid target (${target_price:.2f}) — must be "
                f">= ${price * MIN_TARGET_MULT:.2f} (entry+2%)"
            )
        if stop_loss <= 0 or stop_loss >= price * MAX_STOP_MULT:
            return False, (
                f"Invalid stop (${stop_loss:.2f}) — must be "
                f"<= ${price * MAX_STOP_MULT:.2f} (entry-0.5%)"
            )

        # 1. Daily loss circuit breaker
        allowed, reason = self.check_daily_loss_breaker()
        if not allowed:
            return False, reason

        # 0a. Day-trade prevention (cash account cannot re-buy same-day sell)
        try:
            today = date.today().isoformat()
            for t in self.tracker.get_trade_log():
                if (t.get("ticker") == ticker
                        and t.get("action") in ("CLOSE", "PARTIAL")
                        and str(t.get("timestamp", "")).startswith(today)):
                    return False, (
                        f"Day-trade block: {ticker} was sold today "
                        f"(cash account can't re-buy same day)"
                    )
        except Exception:
            pass

        # 0b. Sector concentration check
        if sector:
            allowed, reason = self.check_sector_concentration(sector)
            if not allowed:
                return False, reason

        # 0c. Sector momentum gate — skip weak sectors (would have blocked TDW
        # during the oil crash; XLE was down >5% that week).
        if sector:
            allowed, reason, _sector_mom = self.check_sector_momentum(sector)
            if not allowed:
                return False, reason

        # 0d. Portfolio correlation check — don't over-concentrate risk.
        # Complements sector limits: catches correlation that crosses sector
        # boundaries (e.g. tech+semi, energy+materials, finance+REIT pairs).
        allowed, reason, _corr = self.check_portfolio_correlation(ticker)
        if not allowed:
            return False, reason

        # 1. Already holding
        if self.tracker.is_holding(ticker):
            return False, f"Already holding {ticker}"

        # 2. Max open positions
        if self.tracker.open_count >= self.cfg.max_open_positions:
            return False, (
                f"Max open positions reached "
                f"({self.tracker.open_count}/{self.cfg.max_open_positions})"
            )

        # 3. Daily buy limit
        daily = self.tracker.daily_buy_count()
        if daily >= self.cfg.max_daily_buys:
            return False, (
                f"Daily buy limit reached ({daily}/{self.cfg.max_daily_buys})"
            )

        # 4. Calculate qty using DYNAMIC cash-aware sizing
        cash = self.client.get_cash_balance()
        available_cash = max(0, cash - self.cfg.cash_reserve)
        qty_est = self.calculate_qty(price, cash_available=available_cash)

        if qty_est == 0:
            return False, (
                f"Can't afford any shares "
                f"(cash=${cash:,.0f}, price=${price:.2f}, reserve=${self.cfg.cash_reserve:.0f})"
            )

        actual_cost = qty_est * price

        # 5. Portfolio exposure
        new_exposure = self.tracker.total_exposure + actual_cost
        if new_exposure > self.cfg.max_portfolio_exposure:
            return False, (
                f"Would exceed max exposure "
                f"(${new_exposure:,.0f} > ${self.cfg.max_portfolio_exposure:,.0f})"
            )

        # 6. Cash sanity check
        if cash < actual_cost:
            return False, (
                f"Insufficient cash (${cash:,.0f} < ${actual_cost:,.0f})"
            )

        # 6. Score filter — regime-aware. The scan's REGIME_MIN_SCORE table
        # adjusts the floor based on regime (TREND_UP=55, MODERATE_UP=60,
        # SIDEWAYS=70, DISTRIBUTION=75, etc). Apply the same shape here +
        # small +5 buffer so trades demand higher conviction than scan
        # inclusion. Falls back to static cfg.min_score_to_trade when
        # regime is missing/unknown.
        _trade_min = float(self.cfg.min_score_to_trade)
        if market_regime:
            try:
                from core.scoring_config import REGIME_MIN_SCORE
                _scan_min = float(REGIME_MIN_SCORE.get(
                    market_regime.upper(), _trade_min - 5.0
                ))
                # Trade buffer over scan inclusion
                _trade_min = _scan_min + 5.0
            except Exception:
                pass
        if score < _trade_min:
            return False, (
                f"Score too low ({score:.1f} < {_trade_min:.1f} "
                f"for regime={market_regime or 'default'})"
            )

        # 7. R:R filter
        if rr < self.cfg.min_rr_to_trade:
            return False, f"R:R too low ({rr:.2f} < {self.cfg.min_rr_to_trade})"

        # 8. Market hours
        if not self.client.is_market_open() and not self.cfg.dry_run:
            return False, "Market is closed"

        return True, ""

    # Conviction tiers for dynamic sizing — see calculate_qty docstring.
    # Key insight: fixed $300 for everything is wasteful. High-conviction
    # setups (high score + clean R:R) deserve more capital; marginal
    # setups get less or get skipped entirely.
    _CONVICTION_TIERS = (
        # (min_score, min_rr, size_multiplier)
        (82.0, 2.5, 1.35),  # HIGH:   +35% of base  (~$405 on $300 base)
        (78.0, 2.2, 1.15),  # HIGH-2: +15%          (~$345)
        (75.0, 2.0, 1.00),  # MED:    base size    (~$300)
        (73.0, 1.8, 0.70),  # LOW:    -30%         (~$210)
        # below this range, trade is skipped by existing filters anyway
    )

    def _conviction_multiplier(self, score: float, rr: float) -> float:
        """Return size multiplier ∈ {0.70, 1.00, 1.15, 1.35} based on tier."""
        for min_score, min_rr, mult in self._CONVICTION_TIERS:
            if score >= min_score and rr >= min_rr:
                return mult
        # Below lowest tier → smallest size (filter handles rejection, but
        # if it slips through we use the most conservative sizing)
        return 0.70

    def calculate_qty(self, price: float, cash_available: float = None,
                      atr_pct: float = 0.0, score: float = 0.0,
                      rr: float = 0.0) -> int:
        """Calculate number of shares to buy — conviction × volatility × cash sizing.

        Sizing logic (applied in order):
        1. **Conviction tier**: score + R:R determine size multiplier
           - Score ≥82, RR ≥2.5 → 1.35× base (high conviction)
           - Score ≥78, RR ≥2.2 → 1.15×
           - Score ≥75, RR ≥2.0 → 1.00× (base)
           - Score ≥73, RR ≥1.8 → 0.70× (marginal)
        2. **Volatility scaling** (if atr_pct given):
           - 2% ATR baseline → 1.0×; scales inversely with ATR
           - Formula: vol_factor = clamp(2.0 / max(atr_pct, 1.0), 0.5, 1.0)
        3. target_spend = base × conviction_mult × vol_factor
        4. qty = floor(target_spend / price), clamped to cash_available
        5. Fallback: 1 share if we can afford it

        Examples (base $300):
        - score=85, rr=2.8, ATR 2% → 1.35 × 1.0 × $300 = $405
        - score=76, rr=2.1, ATR 3% → 1.0 × 0.67 × $300 = $201
        - score=74, rr=1.9, ATR 2% → 0.7 × 1.0 × $300 = $210
        """
        if price <= 0:
            return 0

        base_spend = self.cfg.max_position_size
        if cash_available is not None:
            base_spend = min(base_spend, cash_available)

        # Conviction multiplier — HIGH signal gets more capital
        conviction_mult = self._conviction_multiplier(score, rr) if (score or rr) else 1.0

        # Volatility factor: high-vol stocks get smaller positions
        vol_factor = 1.0
        if atr_pct and atr_pct > 0:
            vol_factor = max(0.5, min(2.0 / max(atr_pct, 1.0), 1.0))

        target_spend = base_spend * conviction_mult * vol_factor
        # Never allow a single position to exceed 1.5× base (cap on upside)
        target_spend = min(target_spend, self.cfg.max_position_size * 1.5)
        # And never exceed available cash
        if cash_available is not None:
            target_spend = min(target_spend, cash_available)

        logger.info(
            "Sizing: score=%.1f rr=%.2f atr=%.1f%% → "
            "conviction=%.2fx vol=%.2fx spend=$%.0f (base=$%.0f)",
            score, rr, atr_pct, conviction_mult, vol_factor,
            target_spend, base_spend,
        )

        # target_spend already has vol_factor + conviction applied above
        qty = math.floor(target_spend / price)

        # Fallback: allow 1 share if we can afford it
        if qty == 0:
            max_affordable = (
                cash_available if cash_available is not None
                else self.cfg.max_position_size * 2
            )
            if price <= max_affordable:
                qty = 1

        return max(qty, 0)

    def get_portfolio_summary(self) -> dict:
        return {
            "cash": self.client.get_cash_balance(),
            "net_liquidation": self.client.get_net_liquidation(),
            "open_positions": self.tracker.open_count,
            "total_exposure": self.tracker.total_exposure,
            "daily_buys_today": self.tracker.daily_buy_count(),
            "remaining_capacity": (
                self.cfg.max_open_positions - self.tracker.open_count
            ),
        }
