"""
UI rendering utilities for Stock Scout (extracted from stock_scout.py)
"""
import streamlit as st
import pandas as pd
import numpy as np
import html as html_escape

def _safe_str(v, default: str = "N/A") -> str:
    if v is None:
        return default
    try:
        return str(v)
    except Exception:
        return default

def _num(v) -> float:
    try:
        if v is None:
            return np.nan
        return float(v)
    except Exception:
        return np.nan

def _is_finite(v) -> bool:
    try:
        return np.isfinite(v)
    except Exception:
        return False

def build_clean_card(row: pd.Series, speculative: bool = False) -> str:
    esc = html_escape.escape
    ticker = esc(_safe_str(row.get("Ticker", "N/A")))
    overall_rank = row.get("Overall_Rank", "N/A")
    from core.scoring_config import get_canonical_score
    overall_score = get_canonical_score(row)
    score_20d = row.get("FinalScore_20d", None)
    target_price = _num(row.get("Target_Price", np.nan))
    entry_price = _num(
        row.get(
            "Entry_Price",
            row.get(
                "Price_Yahoo",
                row.get(
                    "Unit_Price",
                    row.get("Close", np.nan)
                )
            )
        )
    )
    target_date = _safe_str(row.get("Target_Date", "N/A"))
    target_source = _safe_str(row.get("Target_Source", "N/A"))
    rr_ratio = _num(row.get("RR_Ratio", row.get("rr", np.nan)))
    rr_score = _num(row.get("rr_score_v2", np.nan))
    rr_band = _safe_str(row.get("rr_band", ""), "")
    risk_meter = _num(row.get("risk_meter_v2", np.nan))
    risk_band_label = _safe_str(row.get("risk_band", row.get("Risk_Level", "N/A")))
    reliability_pct = _num(row.get("reliability_pct", row.get("Reliability_v2", np.nan)))
    reliability_band_label = _safe_str(row.get("reliability_band", "N/A"))
    if (not reliability_band_label) or reliability_band_label == "N/A":
        if np.isfinite(reliability_pct):
            if reliability_pct >= 75:
                reliability_band_label = "High"
            elif reliability_pct >= 40:
                reliability_band_label = "Medium"
            else:
                reliability_band_label = "Low"
    ml_prob = _num(row.get("ML_20d_Prob", row.get("ML_Probability", np.nan)))
    def ml_conf_band(p: float) -> str:
        if not np.isfinite(p):
            return "N/A"
        if p < 0.60:
            return "Low"
        if p < 0.75:
            return "Medium"
        return "High"
    ml_conf_band_label = _safe_str(row.get("ml_conf_band", ml_conf_band(ml_prob)))
    quality_level = _safe_str(row.get("Quality_Level", row.get("quality_level", "N/A")))
    quality_score = _num(row.get("Quality_Score_Numeric", np.nan))
    conv_base = _num(row.get("conviction_v2_base", np.nan))
    fund_score = _num(row.get("Fundamental_S", np.nan))
    tech_score = _num(row.get("Technical_S", np.nan))
    price_sources = _safe_str(row.get("Price_Sources_Line", row.get("price_sources_line", "")), "")
    fund_sources = _safe_str(row.get("Fund_Sources_Line", row.get("fund_sources_line", "")), "")
    sources_line = ""
    if price_sources or fund_sources:
        sources_line = f"Data sources: Prices - {price_sources or 'N/A'}; Fundamentals - {fund_sources or 'N/A'}"
    def fmt_money(v):
        return f"${v:.2f}" if _is_finite(v) else "N/A"
    def fmt_pct(v):
        return f"{v:.1f}%" if _is_finite(v) else "N/A"
    def fmt_score(v):
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return "0"
            v = len(v) if all(isinstance(x, str) for x in v) else v[0]
        try:
            fv = _num(v)
            return f"{float(fv):.0f}" if _is_finite(fv) else "N/A"
        except (TypeError, ValueError):
            return _safe_str(v, "N/A")
    entry_fmt = fmt_money(entry_price)
    target_fmt = fmt_money(target_price)
    if _is_finite(entry_price) and _is_finite(target_price) and entry_price > 0:
        potential_gain_pct = ((target_price - entry_price) / entry_price) * 100
        potential_fmt = f"+{potential_gain_pct:.1f}%"
    else:
        potential_fmt = "N/A"
    target_badge = ""
    if target_source == "AI":
        target_badge = '<span class="badge ai">AI</span>'
    elif target_source == "Technical":
        target_badge = '<span class="badge tech">Tech</span>'
    rr_ratio_fmt = f"{rr_ratio:.2f}" if _is_finite(rr_ratio) else "N/A"
    overall_score_fmt = fmt_score(overall_score)
    score_20d_fmt = fmt_score(score_20d) if score_20d is not None else "N/A"
    if _is_finite(quality_score):
        quality_score_fmt = f"{quality_score:.1f}" if abs(quality_score - round(quality_score)) > 0.05 else f"{int(round(quality_score))}"
    else:
        quality_score_fmt = "N/A"
    fund_reliability = _num(row.get("Fundamental_Reliability_v2", row.get("Fundamental_Reliability", np.nan)))
    price_reliability = _num(row.get("Price_Reliability_v2", row.get("Price_Reliability", np.nan)))
    fund_rel_fmt = f"{fund_reliability:.1f}" if _is_finite(fund_reliability) else "N/A"
    price_rel_fmt = f"{price_reliability:.1f}" if _is_finite(price_reliability) else "N/A"
    risk_fmt = f"{fmt_score(risk_meter)} ({_safe_str(risk_band_label,'N/A')})"
    reliability_fmt = f"{_safe_str(reliability_band_label,'N/A')} (F:{fund_rel_fmt}% / P:{price_rel_fmt}%)"
    ml_fmt = f"{_safe_str(ml_conf_band_label,'N/A')} (p={ml_prob*100:.1f}%)" if _is_finite(ml_prob) else "N/A (no model data)"
    type_badge = "SPEC" if speculative else "CORE"
    if bool(row.get("Fallback_Display", False)):
        type_badge += " (FB)"
    warning = ""
    if (_is_finite(rr_ratio) and rr_ratio < 1.5) or (_is_finite(risk_meter) and risk_meter > 70):
        warning = " ⚠️"
    rr_5_20 = _num(row.get("RangeRatio_5_20", row.get("range_ratio_5_20", np.nan)))
    tightness = _num(row.get("Tightness_Ratio", row.get("tightness_ratio", np.nan)))
    eps_g = _num(row.get("eps_g_yoy", row.get("EPS YoY", np.nan)))
    is_coiled = (np.isfinite(rr_5_20) and rr_5_20 < 0.7) or (np.isfinite(tightness) and tightness < 0.6)
    is_growth = np.isfinite(eps_g) and eps_g > 0.25
    coil_badge = "<span class='badge coil'>🎯 COILED</span>" if is_coiled else ""
    growth_badge = "<span class='badge growth'>🚀 GROWTH BOOST</span>" if is_growth else ""
    bullets = []
    bullet_html = ""
    if warning:
        if _is_finite(fund_score):
            if fund_score >= 60:
                bullets.append(f"Fundamentals solid ({fmt_score(fund_score)})")
            elif fund_score < 50:
                bullets.append(f"Weak fundamentals ({fmt_score(fund_score)})")
        if _is_finite(tech_score) and tech_score >= 65:
            bullets.append("Technical momentum")
        if _is_finite(rr_ratio) and rr_ratio >= 1.5:
            bullets.append(f"RR {rr_ratio_fmt}x")
        if reliability_band_label:
            bullets.append(f"Reliability {reliability_band_label}")
        if ml_conf_band_label in ("High", "Medium"):
            bullets.append(f"ML {ml_conf_band_label}")
        if overall_rank not in (None, "N/A"):
            bullets.append(f"Rank #{overall_rank}")
        if potential_fmt not in ("N/A", None):
            bullets.append(f"Upside {potential_fmt}")
        if bullets:
            items = "".join(f"<li>{html_escape.escape(b)}</li>" for b in bullets[:6])
            bullet_html = f"<ul class='signal-bullets'>{items}</ul>"
    card_html = f"""
<div class='clean-card { 'speculative' if speculative else 'core' }'>
    <div class='card-header'>
        <div class='ticker-line'><span class='ticker-badge ltr'>{ticker}</span><span class='type-badge'>{type_badge}</span><span class='rank-badge ltr'>#{overall_rank}</span> {coil_badge} {growth_badge}</div>
        <h2 class='overall-score'>{overall_score_fmt}<span class='score-label ltr'>/100</span>{warning}</h2>
    </div>
    <div class='entry-target-line'>Entry <b class='ltr'>{entry_fmt}</b> -> Target <b class='ltr'>{target_fmt}</b> {target_badge} <span class='potential ltr'>{potential_fmt}</span></div>
    {bullet_html}
    
    <div class='top-grid'>
        <div class='field'><span class='label'>R/R</span><span class='value tabular ltr'>{rr_ratio_fmt} <span class='band ltr'>{rr_band}</span></span></div>
        <div class='field'><span class='label'>Risk</span><span class='value tabular ltr'>{risk_fmt}</span></div>
        <div class='field'><span class='label'>Reliability</span><span class='value tabular ltr'>{reliability_fmt}</span></div>
        <div class='field'><span class='label'>ML</span><span class='value tabular ltr'>{ml_fmt}</span></div>
        <div class='field'><span class='label'>Quality</span><span class='value tabular ltr'>{quality_level} ({quality_score_fmt})</span></div>
        <div class='field'><span class='label'>Fundamental Score</span><span class='value tabular ltr'>{fmt_score(fund_score)}</span></div>
    </div>
    <details class='more-info'>
        <summary>More Details</summary>
        <div class='detail-grid'>
            <div class='field'><span class='label'>Target Date</span><span class='value ltr'>{target_date}</span></div>
            <div class='field'><span class='label'>ML Probability</span><span class='value ltr'>{fmt_pct(ml_prob * 100) if np.isfinite(ml_prob) else 'N/A'}</span></div>
            <div class='field'><span class='label'>Base Conviction</span><span class='value ltr'>{fmt_score(conv_base)}</span></div>
            <div class='field'><span class='label'>Fund Sources</span><span class='value ltr'>{fmt_score(row.get('fund_sources_used_v2', row.get('Fundamental_Sources_Count', 0)))}</span></div>
            <div class='field'><span class='label'>Price Sources</span><span class='value ltr'>{fmt_score(row.get('price_sources_used_v2', row.get('Price_Sources_Count', 0)))}</span></div>
            <div class='field'><span class='label'>Price Std Dev</span><span class='value ltr'>{fmt_money(row.get('Price_STD_v2', np.nan))}</span></div>
        </div>
    </details>
</div>
"""
    return card_html

def render_data_sources_overview(provider_status: dict, provider_usage: dict, results: pd.DataFrame) -> None:
    synonyms = {
        "Alpha Vantage": "Alpha",
        "Nasdaq": "NasdaqDL",
        "Yahoo": "Yahoo",
    }
    table_rows = []
    for provider_name, status_info in provider_status.items():
        ok = bool(status_info.get("ok", False))
        status_icon = "🟢" if ok else "🔴"
        status_text = "פעיל" if ok else "תקלה / חסום"
        usage_key = provider_name if provider_name in provider_usage else synonyms.get(provider_name, provider_name)
        usage_info = provider_usage.get(usage_key, {})
        used_price = bool(usage_info.get("used_price"))
        used_fund = bool(usage_info.get("used_fundamentals"))
        used_ml = bool(usage_info.get("used_ml"))
        implemented = bool(usage_info.get("implemented", True))
        if not implemented:
            status_icon = "⚪"
            status_text = "לא רלוונטי בריצה זו"
        if used_price or used_fund or used_ml:
            row = {
                "ספק": provider_name,
                "סטטוס": status_icon + " " + status_text,
                "מחיר": "✅" if used_price else "",
                "פונדמנטלי": "✅" if used_fund else "",
                "ML": "✅" if used_ml else "",
            }
            table_rows.append(row)
    if not table_rows:
        st.info("לא נמצאו ספקים פעילים בריצה זו.")
        return
    styled = pd.DataFrame(table_rows)
    st.markdown("### 🔌 מקורות נתונים")
    st.dataframe(styled, width='stretch', hide_index=True)
    st.caption(f"סה\"כ ספקים פעילים: {sum(1 for r in table_rows if '🟢' in r['סטטוס'])} / {len(provider_status)}")
