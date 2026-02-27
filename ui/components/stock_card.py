"""
Modern stock card renderer — HTML-based cards with score bars and dark mode.
Used by stock_scout.py to render recommendation cards.
"""
from __future__ import annotations

import html as html_mod
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


def _esc(text) -> str:
    """HTML-escape a string to prevent broken rendering."""
    if text is None:
        return ""
    return html_mod.escape(str(text))


def render_stock_card(row: pd.Series, rank: int, score_label: str = "FinalScore_20d") -> str:
    """
    Build a modern HTML card for one stock recommendation.

    Returns an HTML string to be rendered via st.markdown(unsafe_allow_html=True).
    NOTE: HTML must start at column 0 (no leading spaces) to avoid Streamlit's
    Markdown parser treating it as a code block.
    """
    ticker = _esc(row.get("Ticker", "N/A"))
    company = _esc(row.get("shortName", row.get("Company", row.get("Name", ""))))
    sector = _esc(row.get("Sector", row.get("sector", "Unknown")))
    if sector in ("", "Unknown"):
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
    entry_price = to_float(row.get("Price_Yahoo", row.get("Unit_Price", row.get("Entry_Price", row.get("Price", row.get("Close", np.nan))))))
    target_price = to_float(row.get("Target_Price", np.nan))
    if np.isfinite(entry_price) and np.isfinite(target_price) and entry_price > 0:
        upside = ((target_price - entry_price) / entry_price) * 100
        upside_str = f"+{upside:.1f}%"
    else:
        upside = np.nan
        upside_str = "—"
    entry_str = f"${entry_price:.2f}" if np.isfinite(entry_price) else "—"
    target_str = f"${target_price:.2f}" if np.isfinite(target_price) else "—"

    # Target date — per-stock based on ATR/volatility holding period
    target_date_str = ""
    holding_days_val = to_float(row.get("Holding_Days", np.nan))
    holding_days = int(holding_days_val) if np.isfinite(holding_days_val) else 20

    # First check if pipeline provided a target date
    _target_date_col = row.get("Target_Date", None)
    if _target_date_col is not None and str(_target_date_col) not in ("", "nan", "None", "NaT"):
        try:
            _td = pd.Timestamp(_target_date_col)
            if not pd.isna(_td):
                target_date_str = _td.strftime("%b %d")
        except Exception:
            pass
    # Fallback: compute from As_Of_Date + per-stock holding days
    if not target_date_str:
        _as_of = row.get("As_Of_Date", row.get("timestamp", None))
        if _as_of is not None:
            try:
                import datetime as _dt
                if isinstance(_as_of, (int, float)) and _as_of > 1e9:
                    _scan_dt = pd.Timestamp(_dt.datetime.fromtimestamp(_as_of))
                else:
                    _scan_dt = pd.Timestamp(_as_of)
                _target_dt = _scan_dt + pd.offsets.BDay(holding_days)
                target_date_str = _target_dt.strftime("%b %d")
            except Exception:
                pass

    # Score breakdown bars (0-100 range for bar widths)
    tech_score_raw = to_float(row.get("TechScore_20d", row.get("Technical_S", row.get("tech_score", np.nan))))
    fund_score_raw = to_float(row.get("FundamentalScore", row.get("Fundamental_S", row.get("Fundamental_Score", row.get("fund_score", np.nan)))))

    # RR score: use rr_score_v2 if available, otherwise derive from RR ratio
    rr_score_raw = to_float(row.get("rr_score_v2", row.get("rr_score", np.nan)))
    if not np.isfinite(rr_score_raw):
        # Convert RR ratio to 0-100 score: RR=1→30, RR=2→60, RR=3→80, RR=4+→95
        _rr_ratio = to_float(row.get("RR", row.get("RR_Ratio", row.get("RewardRisk", np.nan))))
        if np.isfinite(_rr_ratio) and _rr_ratio > 0:
            rr_score_raw = min(95.0, max(10.0, _rr_ratio * 25.0 + 5.0))
    tech_bar = _safe_pct(tech_score_raw)
    fund_bar = _safe_pct(fund_score_raw)
    ml_bar = _safe_pct(ml_norm, scale=100) if ml_norm is not None else 0
    rr_bar = _safe_pct(rr_score_raw)

    # Storyline
    story = headline_story(row)

    # Detail items
    atr_pct = to_float(row.get("ATR_Price", row.get("ATR_Pct", row.get("ADR_Pct", np.nan))))
    de = to_float(row.get("DE_f", row.get("debt_to_equity", row.get("Debt_to_Equity", row.get("Debt_Equity", row.get("debt_equity", row.get("Leverage", np.nan)))))))
    fund_src = to_float(row.get("Fundamental_Sources_Count", row.get("fund_sources_used_v2", row.get("sources_used_count", np.nan))))
    price_src = to_float(row.get("Price_Sources_Count", row.get("price_sources_used_v2", row.get("Sources_Used", np.nan))))
    price_std = to_float(row.get("Price_STD", row.get("price_std", row.get("Historical_StdDev", np.nan))))
    quality = to_float(row.get("Quality_Score_F", row.get("Quality", np.nan)))
    growth = to_float(row.get("Growth_Score_F", np.nan))
    valuation = to_float(row.get("Valuation_Score_F", row.get("Valuation", np.nan)))
    pe = to_float(row.get("PE", row.get("pe", row.get("PE_Ratio", row.get("PE_f", np.nan)))))
    roe = to_float(row.get("ROE", row.get("roe", row.get("ROE_f", row.get("Quality", np.nan)))))
    beta = to_float(row.get("Beta", row.get("beta", np.nan)))
    market_cap = to_float(row.get("Market_Cap", row.get("market_cap", np.nan)))
    fund_cov = to_float(row.get("Fund_Coverage_Pct", row.get("Fundamental_Coverage_Pct", np.nan)))

    # Build sector badge
    sector_badge = f'<span class="ss-sector-badge">{sector}</span>' if sector else ""
    company_div = f'<div class="ss-company">{company}</div>' if company else ""
    story_div = f'<div class="ss-storyline">{_esc(story)}</div>' if story else ""

    # All HTML at column 0 — no leading whitespace — to prevent Markdown code-block
    return (
        f'<div class="ss-card {card_class}">'
        f'<div class="ss-card-body">'
        f'<div class="ss-card-header">'
        f'<div class="ss-card-left">'
        f'<div class="ss-ticker-row">'
        f'<span class="ss-ticker">{ticker}</span>'
        f'<span class="ss-risk-badge {card_class}">{badge_label}</span>'
        f'{sector_badge}'
        f'</div>'
        f'{company_div}'
        f'</div>'
        f'<div class="ss-card-right">'
        f'<div class="ss-score-circle {tier}">'
        f'<span class="ss-score-value">{score_str}</span>'
        f'<span class="ss-score-label">SCORE</span>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div class="ss-metrics-grid">'
        f'<div class="ss-metric"><span class="ss-metric-value">{rr_str}</span><span class="ss-metric-label">R / R</span></div>'
        f'<div class="ss-metric"><span class="ss-metric-value">{upside_str}</span><span class="ss-metric-label">Upside</span></div>'
        f'<div class="ss-metric"><span class="ss-metric-value">{target_str}</span><span class="ss-metric-label">Target</span></div>'
        f'<div class="ss-metric"><span class="ss-metric-value">{entry_str}</span><span class="ss-metric-label">Entry</span></div>'
        f'<div class="ss-metric"><span class="ss-metric-value">{target_date_str if target_date_str else "—"}</span><span class="ss-metric-label">Target ({holding_days}d)</span></div>'
        f'</div>'
        f'<div class="ss-breakdown">'
        f'<div class="ss-bar-row"><span class="ss-bar-label">Technical</span><div class="ss-bar-track"><div class="ss-bar-fill tech" style="width:{tech_bar:.0f}%"></div></div><span class="ss-bar-value">{fmt_num(tech_score_raw, ".0f")}</span></div>'
        f'<div class="ss-bar-row"><span class="ss-bar-label">Fundament</span><div class="ss-bar-track"><div class="ss-bar-fill fund" style="width:{fund_bar:.0f}%"></div></div><span class="ss-bar-value">{fmt_num(fund_score_raw, ".0f")}</span></div>'
        f'<div class="ss-bar-row"><span class="ss-bar-label">ML 20d</span><div class="ss-bar-track"><div class="ss-bar-fill ml" style="width:{ml_bar:.0f}%"></div></div><span class="ss-bar-value">{ml_pct_str}</span></div>'
        f'<div class="ss-bar-row"><span class="ss-bar-label">R/R Score</span><div class="ss-bar-track"><div class="ss-bar-fill rr" style="width:{rr_bar:.0f}%"></div></div><span class="ss-bar-value">{fmt_num(rr_score_raw, ".0f")}</span></div>'
        f'</div>'
        f'<div class="ss-ml-row">'
        f'<span class="ss-ml-dot {ml_dot}"></span>'
        f'<span class="ss-ml-text">ML: <strong>{ml_pct_str}</strong> &nbsp;|&nbsp; RSI: <strong>{rsi_str}</strong> &nbsp;|&nbsp; Reliability: <strong>{rel_str}</strong> &nbsp;|&nbsp; Fund Coverage: <strong>{fmt_num(fund_cov, ".0f")}%</strong></span>'
        f'</div>'
        f'{story_div}'
        f'<details class="ss-details"><summary>View full breakdown</summary>'
        f'<div class="ss-detail-grid">'
        f'<div class="ss-detail-item"><span class="ss-detail-label">P/E</span><span class="ss-detail-value">{fmt_num(pe, ".1f")}</span></div>'
        f'<div class="ss-detail-item"><span class="ss-detail-label">ROE</span><span class="ss-detail-value">{fmt_num(roe, ".1f")}%</span></div>'
        f'<div class="ss-detail-item"><span class="ss-detail-label">Beta</span><span class="ss-detail-value">{fmt_num(beta, ".2f")}</span></div>'
        f'<div class="ss-detail-item"><span class="ss-detail-label">D/E</span><span class="ss-detail-value">{fmt_num(de, ".2f")}</span></div>'
        f'<div class="ss-detail-item"><span class="ss-detail-label">Quality</span><span class="ss-detail-value">{fmt_num(quality, ".0f")}</span></div>'
        f'<div class="ss-detail-item"><span class="ss-detail-label">Growth</span><span class="ss-detail-value">{fmt_num(growth, ".0f")}</span></div>'
        f'<div class="ss-detail-item"><span class="ss-detail-label">Valuation</span><span class="ss-detail-value">{fmt_num(valuation, ".0f")}</span></div>'
        f'<div class="ss-detail-item"><span class="ss-detail-label">ATR/Price</span><span class="ss-detail-value">{fmt_num(atr_pct, ".3f")}</span></div>'
        f'<div class="ss-detail-item"><span class="ss-detail-label">Fund Sources</span><span class="ss-detail-value">{fmt_num(fund_src, ".0f")}</span></div>'
        f'<div class="ss-detail-item"><span class="ss-detail-label">Fund Coverage</span><span class="ss-detail-value">{fmt_num(fund_cov, ".0f")}%</span></div>'
        f'</div>'
        f'</details>'
        f'</div>'
        f'</div>'
    )


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

    return (
        f'<div class="ss-kpi-strip">'
        f'<div class="ss-kpi"><div class="ss-kpi-value">{regime_display}</div><div class="ss-kpi-label">Market Regime</div></div>'
        f'<div class="ss-kpi"><div class="ss-kpi-value">{avg_str}</div><div class="ss-kpi-label">Avg Score</div></div>'
        f'<div class="ss-kpi"><div class="ss-kpi-value">{spec_count}</div><div class="ss-kpi-label">Speculative</div></div>'
        f'<div class="ss-kpi"><div class="ss-kpi-value">{core_count}</div><div class="ss-kpi-label">Core</div></div>'
        f'<div class="ss-kpi"><div class="ss-kpi-value">{total}</div><div class="ss-kpi-label">Total Stocks</div></div>'
        f'</div>'
    )


def render_section_header(title: str, count: int, section_type: str = "core") -> str:
    """Render a section header for Core or Speculative stocks."""
    icon = "&#x1F6E1;" if section_type == "core" else "&#x1F680;"
    return (
        f'<div class="ss-section-header">'
        f'<div class="ss-icon {section_type}">{icon}</div>'
        f'<h2>{_esc(title)}</h2>'
        f'<span class="ss-count">{count} stocks</span>'
        f'</div>'
    )


def render_ml_legend() -> str:
    """Render ML badge legend."""
    return (
        '<div class="ss-legend">'
        '<span style="font-weight:600;">ML Legend:</span>'
        '<span class="ss-legend-item"><span class="ss-legend-dot" style="background:#22c55e;"></span> &gt;60%</span>'
        '<span class="ss-legend-item"><span class="ss-legend-dot" style="background:#eab308;"></span> 40-60%</span>'
        '<span class="ss-legend-item"><span class="ss-legend-dot" style="background:#ef4444;"></span> &lt;40%</span>'
        '</div>'
    )
