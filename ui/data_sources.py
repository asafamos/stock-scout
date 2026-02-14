"""Data sources overview — canonical provider-status table (Hebrew/RTL)."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_data_sources_overview(
    provider_status: dict,
    provider_usage: dict,
    results: pd.DataFrame,
) -> None:
    """Render a dynamic, compact data sources table with Hebrew labels."""
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

        if not implemented:
            status_icon = "⚪"
            status_text = "לא רלוונטי בריצה זו"

        if used_price or used_fund or used_ml:
            used_icon = "🟢"
            used_text = "בשימוש"
        else:
            used_icon = "⚪"
            used_text = "לא בשימוש"

        details_parts = []
        if used_price:
            details_parts.append("מחיר")
        if used_fund:
            details_parts.append("פונדמנטלי")
        if used_ml:
            details_parts.append("ML")

        usage_detail = " | ".join(details_parts) if details_parts else "—"

        table_rows.append(
            {
                "ספק": provider_name,
                "סטטוס": f"{status_icon} {status_text}",
                "שימוש": f"{used_icon} {used_text}",
                "פרטים": usage_detail,
            }
        )

    if not table_rows:
        return

    df_sources = pd.DataFrame(table_rows)
    df_sources["ספק"] = df_sources["ספק"].astype(str)

    styled = (
        df_sources.style.set_properties(
            subset=["ספק"],
            **{
                "direction": "ltr",
                "text-align": "left",
                "font-size": "14px",
                "white-space": "nowrap",
            },
        )
        .set_properties(
            subset=["סטטוס", "שימוש", "פרטים"],
            **{"text-align": "center", "font-size": "14px"},
        )
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("font-size", "15px")]}]
        )
    )

    st.markdown("### 🔌 מקורות נתונים")
    st.dataframe(styled, width="stretch", hide_index=True)

    used_count = 0
    for provider_name in provider_status:
        usage_key = (
            provider_name
            if provider_name in provider_usage
            else synonyms.get(provider_name, provider_name)
        )
        info = provider_usage.get(usage_key, {})
        if info.get("used_price") or info.get("used_fundamentals") or info.get("used_ml"):
            used_count += 1
    st.caption(f'סה"כ ספקים פעילים: {used_count} / {len(provider_status)}')
