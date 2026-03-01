"""Data sources overview — clean styled table matching 2026 design system."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_data_sources_overview(
    provider_status: dict,
    provider_usage: dict,
    results: pd.DataFrame,
) -> None:
    """Render a styled table showing all data providers, their key status,
    connectivity, and whether data was actually fetched from them.

    Matches the Stock Scout 2026 design language.
    """
    synonyms = {
        "Alpha Vantage": "Alpha",
        "Nasdaq": "NasdaqDL",
        "Yahoo": "Yahoo",
    }

    rows_html = ""
    active_count = 0
    used_count = 0
    total = 0

    for provider_name, status_info in provider_status.items():
        total += 1
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

        is_used = used_price or used_fund or used_ml

        if ok:
            active_count += 1
        if is_used:
            used_count += 1

        # Icons
        key_icon = "✅" if ok else "❌"
        connected_icon = "🟢" if ok else "⚫"

        # Data roles fetched
        roles = []
        if used_price:
            roles.append("Price")
        if used_fund:
            roles.append("Fund")
        if used_ml:
            roles.append("ML")
        data_str = ", ".join(roles) if roles else "—"

        # Row style: dim if not active
        opacity = "1" if ok else "0.55"
        data_color = "var(--ss-green, #22c55e)" if is_used else "var(--ss-text-muted, #94a3b8)"

        rows_html += f"""<tr style="opacity:{opacity};">
            <td style="padding:8px 12px; font-weight:600; font-size:0.82rem; color:var(--ss-text-primary, #0f172a);">{provider_name}</td>
            <td style="padding:8px 12px; text-align:center;">{key_icon}</td>
            <td style="padding:8px 12px; text-align:center;">{connected_icon}</td>
            <td style="padding:8px 12px; font-size:0.78rem; color:{data_color}; font-weight:500;">{data_str}</td>
        </tr>"""

    if total == 0:
        return

    # Header row style
    th_style = (
        "padding:8px 12px; font-size:0.7rem; font-weight:700; "
        "color:var(--ss-text-muted, #94a3b8); text-transform:uppercase; "
        "letter-spacing:0.04em; border-bottom:2px solid var(--ss-border, #e2e8f0);"
    )

    table_html = f"""
<div style="
    background: var(--ss-bg-card, #fff);
    border: 1px solid var(--ss-border, #e2e8f0);
    border-radius: var(--ss-radius-md, 12px);
    padding: 16px 20px;
    margin: 12px 0;
    box-shadow: var(--ss-shadow-sm, 0 1px 3px rgba(0,0,0,0.06));
    direction: ltr;
">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <span style="font-size:0.88rem; font-weight:700; color:var(--ss-text-primary, #0f172a);">
            🔌 Data Sources
        </span>
        <span style="font-size:0.78rem; font-weight:600; color:var(--ss-text-muted, #94a3b8);">
            {used_count} used · {active_count} active · {total} total
        </span>
    </div>
    <table style="
        width:100%; border-collapse:collapse;
        font-family:var(--ss-font, system-ui);
    ">
        <thead>
            <tr>
                <th style="{th_style} text-align:left;">Provider</th>
                <th style="{th_style} text-align:center;">API Key</th>
                <th style="{th_style} text-align:center;">Connected</th>
                <th style="{th_style} text-align:left;">Data Fetched</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</div>"""

    with st.expander("🔌 Data Sources", expanded=False):
        st.markdown(table_html, unsafe_allow_html=True)
