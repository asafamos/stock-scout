"""
Modern stock card renderer — HTML-based cards with score bars and dark mode.
Used by stock_scout.py to render recommendation cards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from ui.card_helpers import (
    to_float,
    normalize_prob,
    get_ml_prob_from_row,
    risk_class,
    headline_story,
    fmt_num,
)


def _score_tier(score: float) -> str:
    """Return 'high', 'medium', or 'low' based on score value."""
    if score >= 65:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def _ml_dot_class(p: Optional[float]) -> str:
    """Return 'high', 'mid', or 'low' CSS class for ML probability."""
    if p is None or not np.isfinite(p):
        return "low"
    norm = p if p <= 1 else p / 100.0
    if norm > 0.60:
        return "high"
    if norm >= 0.40:
        return "mid"
    return "low"


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _safe_pct(v, scale: float = 1.0) -> float:
    """Convert value to a 0-100 percentage, safe for bar widths."""
    fv = to_float(v)
    if not np.isfinite(fv):
        return 0.0
    return _clamp(fv * scale)


def render_stock_card(row: pd.Series, rank: int, score_label: str = "FinalScore_20d") -> str:
    """
    Build a modern HTML card for one stock recommendation.

    Returns an HTML string to be rendered via st.html() or st.markdown(unsafe_allow_html=True).
    """
    ticker = row.get("Ticker", "N/A")
    company = row.get("shortName", row.get("Company", row.get("Name", "")))
    sector = row.get("Sector", row.get("sector", "Unknown"))
    if sector in (None, "", "Unknown"):
        sector = ""

    # Score
    final_score = to_float(row.get("FinalScore_20d", row.get("Score", np.nan)))
    score_str = f"{final_score:.0f}" if np.isfinite(final_score) else "—"
    tier = _score_tier(final_score) if np.isfinite(final_score) else "low"

    # Risk class
    rc = risk_class(row)
    rc_lower = rc.lower() if rc else "spec"
    card_class = "core" if "core" in rc_lower else "spec"
    badge_label = "CORE" if card_class == "core" else "SPEC"

    # ML
    ml_prob = get_ml_prob_from_row(row)
    ml_norm = normalize_prob(ml_prob)
    ml_pct_str = f"{ml_norm * 100:.0f}%" if ml_norm is not None else "—"
    ml_dot = _ml_dot_class(ml_prob)

    # Key metrics
    rr = to_float(row.get("RR", row.get("RR_Ratio", row.get("RewardRisk", np.nan))))
    rr_str = f"{rr:.2f}x" if np.isfinite(rr) else "—"

    rsi = to_float(row.get("RSI", np.nan))
    rsi_str = f"{rsi:.0f}" if np.isfinite(rsi) else "—"

    rel = to_float(row.get("ReliabilityScore", row.get("Reliability_Score", row.get("Reliability_v2", np.nan))))
    rel_str = f"{rel:.0f}" if np.isfinite(rel) else "—"

    # Entry / Target / Upside
    entry_price = to_float(row.get("Price_Yahoo", row.get("Unit_Price", row.get("Entry_Price", np.nan))))
    target_price = to_float(row.get("Target_Price", np.nan))
    if np.isfinite(entry_price) and np.isfinite(target_price) and entry_price > 0:
        upside = ((target_price - entry_price) / entry_price) * 100
        upside_str = f"+{upside:.1f}%"
    else:
        upside = np.nan
        upside_str = "—"
    entry_str = f"${entry_price:.2f}" if np.isfinite(entry_price) else "—"
    target_str = f"${target_price:.2f}" if np.isfinite(target_price) else "—"

    # Score breakdown bars (0-100 range for bar widths)
    tech_score_raw = to_float(row.get("TechScore_20d", row.get("Technical_S", row.get("tech_score", np.nan))))
    fund_score_raw = to_float(row.get("FundamentalScore", row.get("Fundamental_S", row.get("fund_score", np.nan))))
    rr_score_raw = to_float(row.get("rr_score_v2", row.get("rr_score", np.nan)))
    tech_bar = _safe_pct(tech_score_raw)
    fund_bar = _safe_pct(fund_score_raw)
    ml_bar = _safe_pct(ml_norm, scale=100) if ml_norm is not None else 0
    rr_bar = _safe_pct(rr_score_raw)

    # Storyline
    story = headline_story(row)

    # Detail items
    atr_pct = to_float(row.get("ATR_Price", row.get("ATR_Pct", np.nan)))
    de = to_float(row.get("DE_f", row.get("debt_to_equity", row.get("Debt_to_Equity", row.get("Leverage", np.nan)))))
    fund_src = to_float(row.get("Fundamental_Sources_Count", row.get("fund_sources_used_v2", row.get("sources_used_count", np.nan))))
    price_src = to_float(row.get("Price_Sources_Count", row.get("price_sources_used_v2", row.get("price_sources", np.nan))))
    price_std = to_float(row.get("Price_STD", row.get("price_std", np.nan)))
    quality = to_float(row.get("Quality_Score_F", np.nan))
    growth = to_float(row.get("Growth_Score_F", np.nan))
    valuation = to_float(row.get("Valuation_Score_F", np.nan))
    fund_cov = to_float(row.get("Fund_Coverage_Pct", np.nan))

    html = f"""
    <div class="ss-card {card_class}">
      <div class="ss-card-body">
        <!-- Header -->
        <div class="ss-card-header">
          <div class="ss-card-left">
            <div class="ss-ticker-row">
              <span class="ss-ticker">{ticker}</span>
              <span class="ss-risk-badge {card_class}">{badge_label}</span>
              {"<span class='ss-sector-badge'>" + sector + "</span>" if sector else ""}
            </div>
            {"<div class='ss-company'>" + company + "</div>" if company else ""}
          </div>
          <div class="ss-card-right">
            <div class="ss-score-circle {tier}">
              <span class="ss-score-value">{score_str}</span>
              <span class="ss-score-label">SCORE</span>
            </div>
          </div>
        </div>

        <!-- Metrics strip -->
        <div class="ss-metrics-grid">
          <div class="ss-metric">
            <span class="ss-metric-value">{entry_str}</span>
            <span class="ss-metric-label">Entry</span>
          </div>
          <div class="ss-metric">
            <span class="ss-metric-value">{target_str}</span>
            <span class="ss-metric-label">Target</span>
          </div>
          <div class="ss-metric">
            <span class="ss-metric-value">{upside_str}</span>
            <span class="ss-metric-label">Upside</span>
          </div>
          <div class="ss-metric">
            <span class="ss-metric-value">{rr_str}</span>
            <span class="ss-metric-label">R / R</span>
          </div>
        </div>

        <!-- Score breakdown bars -->
        <div class="ss-breakdown">
          <div class="ss-bar-row">
            <span class="ss-bar-label">Technical</span>
            <div class="ss-bar-track"><div class="ss-bar-fill tech" style="width:{tech_bar:.0f}%"></div></div>
            <span class="ss-bar-value">{fmt_num(tech_score_raw, '.0f')}</span>
          </div>
          <div class="ss-bar-row">
            <span class="ss-bar-label">Fundament</span>
            <div class="ss-bar-track"><div class="ss-bar-fill fund" style="width:{fund_bar:.0f}%"></div></div>
            <span class="ss-bar-value">{fmt_num(fund_score_raw, '.0f')}</span>
          </div>
          <div class="ss-bar-row">
            <span class="ss-bar-label">ML 20d</span>
            <div class="ss-bar-track"><div class="ss-bar-fill ml" style="width:{ml_bar:.0f}%"></div></div>
            <span class="ss-bar-value">{ml_pct_str}</span>
          </div>
          <div class="ss-bar-row">
            <span class="ss-bar-label">R/R Score</span>
            <div class="ss-bar-track"><div class="ss-bar-fill rr" style="width:{rr_bar:.0f}%"></div></div>
            <span class="ss-bar-value">{fmt_num(rr_score_raw, '.0f')}</span>
          </div>
        </div>

        <!-- Quick stats row (no duplication — each metric shown once) -->
        <div class="ss-ml-row">
          <span class="ss-ml-dot {ml_dot}"></span>
          <span class="ss-ml-text">ML: <strong>{ml_pct_str}</strong> &nbsp;|&nbsp; RSI: <strong>{rsi_str}</strong> &nbsp;|&nbsp; Reliability: <strong>{rel_str}</strong> &nbsp;|&nbsp; Fund Coverage: <strong>{fmt_num(fund_cov, '.0f')}%</strong></span>
        </div>

        <!-- Storyline -->
        {"<div class='ss-storyline'>" + story + "</div>" if story else ""}

        <!-- Expandable details (deeper metrics not shown above) -->
        <details class="ss-details">
          <summary>View full breakdown</summary>
          <div class="ss-detail-grid">
            <div class="ss-detail-item">
              <span class="ss-detail-label">ATR/Price</span>
              <span class="ss-detail-value">{fmt_num(atr_pct, '.3f')}</span>
            </div>
            <div class="ss-detail-item">
              <span class="ss-detail-label">D/E Leverage</span>
              <span class="ss-detail-value">{fmt_num(de, '.2f')}</span>
            </div>
            <div class="ss-detail-item">
              <span class="ss-detail-label">Quality</span>
              <span class="ss-detail-value">{fmt_num(quality, '.0f')}</span>
            </div>
            <div class="ss-detail-item">
              <span class="ss-detail-label">Growth</span>
              <span class="ss-detail-value">{fmt_num(growth, '.0f')}</span>
            </div>
            <div class="ss-detail-item">
              <span class="ss-detail-label">Valuation</span>
              <span class="ss-detail-value">{fmt_num(valuation, '.0f')}</span>
            </div>
            <div class="ss-detail-item">
              <span class="ss-detail-label">Fund Sources</span>
              <span class="ss-detail-value">{fmt_num(fund_src, '.0f')}</span>
            </div>
            <div class="ss-detail-item">
              <span class="ss-detail-label">Price Sources</span>
              <span class="ss-detail-value">{fmt_num(price_src, '.0f')}</span>
            </div>
            <div class="ss-detail-item">
              <span class="ss-detail-label">Price STD</span>
              <span class="ss-detail-value">{fmt_num(price_std, '.2f')}</span>
            </div>
          </div>
        </details>
      </div>
    </div>
    """
    return html


def render_kpi_strip(total: int, core_count: int, spec_count: int,
                     avg_score: Optional[float] = None,
                     regime: str = "neutral") -> str:
    """Render a KPI summary strip as HTML."""
    avg_str = f"{avg_score:.1f}" if avg_score is not None and np.isfinite(avg_score) else "—"
    regime_display = {
        "bullish": "Bullish",
        "neutral": "Neutral",
        "bearish": "Bearish",
    }.get(regime, regime.title())

    return f"""
    <div class="ss-kpi-strip">
      <div class="ss-kpi">
        <div class="ss-kpi-value">{total}</div>
        <div class="ss-kpi-label">Total Stocks</div>
      </div>
      <div class="ss-kpi">
        <div class="ss-kpi-value">{core_count}</div>
        <div class="ss-kpi-label">Core</div>
      </div>
      <div class="ss-kpi">
        <div class="ss-kpi-value">{spec_count}</div>
        <div class="ss-kpi-label">Speculative</div>
      </div>
      <div class="ss-kpi">
        <div class="ss-kpi-value">{avg_str}</div>
        <div class="ss-kpi-label">Avg Score</div>
      </div>
      <div class="ss-kpi">
        <div class="ss-kpi-value">{regime_display}</div>
        <div class="ss-kpi-label">Market Regime</div>
      </div>
    </div>
    """


def render_section_header(title: str, count: int, section_type: str = "core") -> str:
    """Render a section header for Core or Speculative stocks."""
    icon = "&#x1F6E1;" if section_type == "core" else "&#x1F680;"  # shield or rocket
    return f"""
    <div class="ss-section-header">
      <div class="ss-icon {section_type}">{icon}</div>
      <h2>{title}</h2>
      <span class="ss-count">{count} stocks</span>
    </div>
    """


def render_ml_legend() -> str:
    """Render ML badge legend."""
    return """
    <div class="ss-legend">
      <span style="font-weight:600;">ML Legend:</span>
      <span class="ss-legend-item"><span class="ss-legend-dot" style="background:#22c55e;"></span> &gt;60%</span>
      <span class="ss-legend-item"><span class="ss-legend-dot" style="background:#eab308;"></span> 40-60%</span>
      <span class="ss-legend-item"><span class="ss-legend-dot" style="background:#ef4444;"></span> &lt;40%</span>
    </div>
    """
