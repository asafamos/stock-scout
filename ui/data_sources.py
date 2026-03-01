"""Data sources overview — modern card-based design matching recommendation cards."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_data_sources_overview(
    provider_status: dict,
    provider_usage: dict,
    results: pd.DataFrame,
) -> None:
    """Render a modern, card-based data sources overview.

    Uses the same design language as stock recommendation cards —
    compact cards with status bars and icons.
    """
    synonyms = {
        "Alpha Vantage": "Alpha",
        "Nasdaq": "NasdaqDL",
        "Yahoo": "Yahoo",
    }

    providers = []
    active_count = 0
    used_count = 0

    for provider_name, status_info in provider_status.items():
        ok = bool(status_info.get("ok", False))

        usage_key = (
            provider_name
            if provider_name in provider_usage
            else synonyms.get(provider_name, provider_name)
        )
        usage_info = provider_usage.get(usage_key, {})

        used_price = bool(usage_info.get("used_price"))
        used_fund = bool(usage_info.get("used_fundamentals"))
        used_ml = bool(usage_info.get("used_ml"))
        implemented = bool(usage_info.get("implemented", True))

        is_used = used_price or used_fund or used_ml

        roles = []
        if used_price:
            roles.append("Price")
        if used_fund:
            roles.append("Fund")
        if used_ml:
            roles.append("ML")

        if ok:
            active_count += 1
        if is_used:
            used_count += 1

        providers.append({
            "name": provider_name,
            "ok": ok,
            "used": is_used,
            "implemented": implemented,
            "roles": roles,
        })

    if not providers:
        return

    total = len(providers)

    # Build provider pills HTML
    pills_html = ""
    for p in providers:
        if not p["implemented"]:
            dot = "var(--ss-text-muted, #94a3b8)"
            border = "var(--ss-border, #e2e8f0)"
            opacity = "0.5"
        elif p["used"]:
            dot = "var(--ss-green, #22c55e)"
            border = "var(--ss-green, #22c55e)"
            opacity = "1"
        elif p["ok"]:
            dot = "var(--ss-yellow, #eab308)"
            border = "var(--ss-border, #e2e8f0)"
            opacity = "0.85"
        else:
            dot = "var(--ss-red, #ef4444)"
            border = "var(--ss-border, #e2e8f0)"
            opacity = "0.7"

        role_text = f'<span style="font-size:0.6rem;color:var(--ss-text-muted,#94a3b8);margin-left:4px;">{", ".join(p["roles"])}</span>' if p["roles"] else ""

        pills_html += f"""
        <div style="
            display:inline-flex; align-items:center; gap:6px;
            padding: 5px 12px; border-radius: 999px;
            border: 1px solid {border};
            background: var(--ss-bg-card, #fff);
            font-size: 0.78rem; font-weight: 600;
            color: var(--ss-text-primary, #0f172a);
            opacity: {opacity};
            direction: ltr;
        ">
            <span style="width:8px;height:8px;border-radius:50%;background:{dot};flex-shrink:0;"></span>
            {p['name']}{role_text}
        </div>
        """

    # Summary bar (like score bar)
    used_pct = int((used_count / total) * 100) if total > 0 else 0
    bar_color = "linear-gradient(90deg, #10b981, #34d399)" if used_pct >= 50 else "linear-gradient(90deg, #f59e0b, #fbbf24)"

    html = f"""
    <div style="
        background: var(--ss-bg-card, #fff);
        border: 1px solid var(--ss-border, #e2e8f0);
        border-radius: var(--ss-radius-md, 12px);
        padding: 16px 20px;
        margin: 12px 0;
        box-shadow: var(--ss-shadow-sm, 0 1px 3px rgba(0,0,0,0.06));
        direction: ltr;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-size:0.88rem; font-weight:700; color:var(--ss-text-primary, #0f172a);">
                🔌 Data Sources
            </span>
            <span style="font-size:0.78rem; font-weight:600; color:var(--ss-text-muted, #94a3b8);">
                {used_count} used · {active_count} active · {total} total
            </span>
        </div>
        <div style="
            height: 6px; background: var(--ss-bar-bg, #e2e8f0);
            border-radius: 3px; overflow: hidden; margin-bottom: 12px;
        ">
            <div style="width:{used_pct}%; height:100%; background:{bar_color}; border-radius:3px;"></div>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:6px;">
            {pills_html}
        </div>
    </div>
    """

    with st.expander("🔌 Data Sources", expanded=False):
        st.markdown(html, unsafe_allow_html=True)
