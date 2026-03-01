"""
Stock Scout 2026 Design System
Modern, responsive CSS with dark mode support and proper contrast.
"""


def get_design_css() -> str:
    """Return the complete design system CSS with dark mode support."""
    return """
<style>
/* ================================================================
   STOCK SCOUT 2026 — DESIGN SYSTEM
   Supports: light mode, dark mode (auto via prefers-color-scheme),
   RTL (Hebrew), responsive breakpoints.
   ================================================================ */

/* ---------- CSS Variables (Light Mode) ---------- */
:root {
    --ss-bg-primary: #f8fafc;
    --ss-bg-card: #ffffff;
    --ss-bg-card-hover: #f1f5f9;
    --ss-bg-surface: #f1f5f9;
    --ss-bg-badge: #eff6ff;
    --ss-border: #e2e8f0;
    --ss-border-hover: #cbd5e1;
    --ss-text-primary: #0f172a;
    --ss-text-secondary: #475569;
    --ss-text-muted: #94a3b8;
    --ss-text-on-accent: #ffffff;
    --ss-accent: #3b82f6;
    --ss-accent-hover: #2563eb;
    --ss-core-accent: #10b981;
    --ss-core-bg: #ecfdf5;
    --ss-core-text: #065f46;
    --ss-spec-accent: #f59e0b;
    --ss-spec-bg: #fffbeb;
    --ss-spec-text: #92400e;
    --ss-green: #22c55e;
    --ss-green-bg: #f0fdf4;
    --ss-yellow: #eab308;
    --ss-yellow-bg: #fefce8;
    --ss-red: #ef4444;
    --ss-red-bg: #fef2f2;
    --ss-shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
    --ss-shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    --ss-shadow-lg: 0 8px 24px rgba(0,0,0,0.10);
    --ss-radius-sm: 8px;
    --ss-radius-md: 12px;
    --ss-radius-lg: 16px;
    --ss-radius-full: 999px;
    --ss-font: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --ss-mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
    --ss-score-ring-bg: #e2e8f0;
    --ss-bar-bg: #e2e8f0;
    --ss-btn-text: #ffffff;
}

/* ---------- Dark Mode Variables ---------- */
@media (prefers-color-scheme: dark) {
    :root {
        --ss-bg-primary: #0f172a;
        --ss-bg-card: #1e293b;
        --ss-bg-card-hover: #334155;
        --ss-bg-surface: #1e293b;
        --ss-bg-badge: #1e3a5f;
        --ss-border: #334155;
        --ss-border-hover: #475569;
        --ss-text-primary: #f1f5f9;
        --ss-text-secondary: #cbd5e1;
        --ss-text-muted: #64748b;
        --ss-text-on-accent: #ffffff;
        --ss-accent: #60a5fa;
        --ss-accent-hover: #93bbfd;
        --ss-core-accent: #34d399;
        --ss-core-bg: #064e3b;
        --ss-core-text: #a7f3d0;
        --ss-spec-accent: #fbbf24;
        --ss-spec-bg: #78350f;
        --ss-spec-text: #fde68a;
        --ss-green: #4ade80;
        --ss-green-bg: #14532d;
        --ss-yellow: #facc15;
        --ss-yellow-bg: #713f12;
        --ss-red: #f87171;
        --ss-red-bg: #7f1d1d;
        --ss-shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
        --ss-shadow-md: 0 4px 12px rgba(0,0,0,0.4);
        --ss-shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
        --ss-score-ring-bg: #334155;
        --ss-bar-bg: #334155;
        --ss-btn-text: #ffffff;
    }
}

/* Also handle Streamlit's dark theme class + common dark selectors */
[data-theme="dark"],
.stApp[data-theme="dark"],
[data-testid="stAppViewContainer"][style*="background-color: rgb(14, 17, 23)"],
html[data-theme="dark"] {
    --ss-bg-primary: #0f172a;
    --ss-bg-card: #1e293b;
    --ss-bg-card-hover: #334155;
    --ss-bg-surface: #1e293b;
    --ss-bg-badge: #1e3a5f;
    --ss-border: #334155;
    --ss-border-hover: #475569;
    --ss-text-primary: #f1f5f9;
    --ss-text-secondary: #cbd5e1;
    --ss-text-muted: #64748b;
    --ss-accent: #60a5fa;
    --ss-core-accent: #34d399;
    --ss-core-bg: #064e3b;
    --ss-core-text: #a7f3d0;
    --ss-spec-accent: #fbbf24;
    --ss-spec-bg: #78350f;
    --ss-spec-text: #fde68a;
    --ss-green: #4ade80;
    --ss-green-bg: #14532d;
    --ss-yellow: #facc15;
    --ss-yellow-bg: #713f12;
    --ss-red: #f87171;
    --ss-red-bg: #7f1d1d;
    --ss-shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
    --ss-shadow-md: 0 4px 12px rgba(0,0,0,0.4);
    --ss-shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
    --ss-score-ring-bg: #334155;
    --ss-bar-bg: #334155;
}

/* ---------- Global RTL + Base ---------- */
body, .stApp, .main, .block-container {
    direction: rtl;
    text-align: right;
    font-family: var(--ss-font);
}

h1, h2, h3, h4, h5, h6 {
    text-align: right;
    color: var(--ss-text-primary);
}

/* Force LTR for tickers, numbers, inputs */
span.ltr, .ltr, .ss-ticker, .ss-number, .ss-score-value,
input, textarea, code, pre,
.stTextInput, .stNumberInput, .stSlider, .stSelectbox {
    direction: ltr !important;
    unicode-bidi: embed;
    text-align: left !important;
}

/* ---------- Page Header ---------- */
.ss-page-header {
    padding: 0 0 16px 0;
    margin-bottom: 8px;
    text-align: center;
    direction: ltr;
}

.ss-page-header h1 {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--ss-text-primary);
    margin: 0 0 4px 0;
    text-align: center !important;
}

.ss-page-header .ss-subtitle {
    font-size: 0.875rem;
    color: var(--ss-text-muted);
    margin: 0;
    text-align: center;
}

/* ---------- Section Headers ---------- */
.ss-section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--ss-border);
    direction: ltr;
}

.ss-section-header .ss-icon {
    width: 36px;
    height: 36px;
    border-radius: var(--ss-radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}

.ss-section-header .ss-icon.core {
    background: var(--ss-core-bg);
}

.ss-section-header .ss-icon.spec {
    background: var(--ss-spec-bg);
}

.ss-section-header h2 {
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
    color: var(--ss-text-primary);
}

.ss-section-header .ss-count {
    font-size: 0.8rem;
    color: var(--ss-text-muted);
    background: var(--ss-bg-surface);
    padding: 2px 10px;
    border-radius: var(--ss-radius-full);
    font-weight: 600;
}

/* ---------- KPI Strip ---------- */
.ss-kpi-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 16px 0;
}

.ss-kpi {
    background: var(--ss-bg-card);
    border: 1px solid var(--ss-border);
    border-radius: var(--ss-radius-md);
    padding: 16px;
    text-align: center;
    box-shadow: var(--ss-shadow-sm);
}

.ss-kpi .ss-kpi-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--ss-text-primary);
    font-variant-numeric: tabular-nums;
    font-family: var(--ss-mono);
}

.ss-kpi .ss-kpi-label {
    font-size: 0.75rem;
    color: var(--ss-text-muted);
    margin-top: 2px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ---------- Stock Card ---------- */
.ss-card {
    background: var(--ss-bg-card);
    border: 1px solid var(--ss-border);
    border-radius: var(--ss-radius-lg);
    padding: 0;
    margin: 12px 0;
    box-shadow: var(--ss-shadow-sm);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
    overflow: hidden;
}

.ss-card:hover {
    box-shadow: var(--ss-shadow-md);
    border-color: var(--ss-border-hover);
}

/* Card accent stripe */
.ss-card.core {
    border-top: 3px solid var(--ss-core-accent);
}

.ss-card.spec {
    border-top: 3px solid var(--ss-spec-accent);
}

/* Card inner layout — force LTR since all card content is English/numbers */
.ss-card-body {
    padding: 16px 20px;
    direction: ltr;
    text-align: left;
}

/* Row 1: Header (ticker + score) */
.ss-card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
}

.ss-card-left {
    flex: 1;
    min-width: 0;
}

.ss-card-right {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.ss-ticker-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.ss-ticker {
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--ss-text-primary);
    letter-spacing: 0.02em;
    font-family: var(--ss-mono);
    direction: ltr;
}

.ss-company {
    font-size: 0.82rem;
    color: var(--ss-text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 200px;
}

.ss-sector-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: var(--ss-radius-full);
    font-size: 0.7rem;
    font-weight: 600;
    background: var(--ss-bg-surface);
    color: var(--ss-text-secondary);
    letter-spacing: 0.02em;
}

.ss-risk-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: var(--ss-radius-full);
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.ss-risk-badge.core {
    background: var(--ss-core-bg);
    color: var(--ss-core-text);
}

.ss-risk-badge.spec {
    background: var(--ss-spec-bg);
    color: var(--ss-spec-text);
}

/* Score circle */
.ss-score-circle {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    position: relative;
    flex-shrink: 0;
}

.ss-score-circle .ss-score-value {
    font-size: 1.4rem;
    font-weight: 800;
    font-family: var(--ss-mono);
    line-height: 1;
    direction: ltr;
}

.ss-score-circle .ss-score-label {
    font-size: 0.55rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.8;
    line-height: 1;
    margin-top: 2px;
}

.ss-score-circle.high {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #166534;
}

.ss-score-circle.medium {
    background: linear-gradient(135deg, #fef9c3, #fde68a);
    color: #854d0e;
}

.ss-score-circle.low {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    color: #991b1b;
}

/* Dark mode score circle overrides */
@media (prefers-color-scheme: dark) {
    .ss-score-circle.high {
        background: linear-gradient(135deg, #064e3b, #065f46);
        color: #a7f3d0;
    }
    .ss-score-circle.medium {
        background: linear-gradient(135deg, #713f12, #854d0e);
        color: #fde68a;
    }
    .ss-score-circle.low {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: #fecaca;
    }
}

[data-theme="dark"] .ss-score-circle.high,
html[data-theme="dark"] .ss-score-circle.high {
    background: linear-gradient(135deg, #064e3b, #065f46);
    color: #a7f3d0;
}
[data-theme="dark"] .ss-score-circle.medium,
html[data-theme="dark"] .ss-score-circle.medium {
    background: linear-gradient(135deg, #713f12, #854d0e);
    color: #fde68a;
}
[data-theme="dark"] .ss-score-circle.low,
html[data-theme="dark"] .ss-score-circle.low {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    color: #fecaca;
}

/* Row 2: Metrics grid */
.ss-metrics-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 12px;
}

.ss-metric {
    background: var(--ss-bg-surface);
    border-radius: var(--ss-radius-sm);
    padding: 10px 12px;
    text-align: center;
}

.ss-metric .ss-metric-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--ss-text-primary);
    font-family: var(--ss-mono);
    font-variant-numeric: tabular-nums;
    direction: ltr;
    display: block;
}

.ss-metric .ss-metric-label {
    font-size: 0.65rem;
    color: var(--ss-text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
    display: block;
}

/* Row 3: Score breakdown bars */
.ss-breakdown {
    margin-bottom: 12px;
}

.ss-bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}

.ss-bar-label {
    width: 72px;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--ss-text-secondary);
    text-align: left;
    flex-shrink: 0;
}

.ss-bar-track {
    flex: 1;
    height: 8px;
    background: var(--ss-bar-bg);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.ss-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}

.ss-bar-fill.tech { background: linear-gradient(90deg, #6366f1, #818cf8); }
.ss-bar-fill.fund { background: linear-gradient(90deg, #10b981, #34d399); }
.ss-bar-fill.ml { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.ss-bar-fill.rr { background: linear-gradient(90deg, #ec4899, #f472b6); }

.ss-bar-value {
    width: 36px;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--ss-text-primary);
    text-align: left;
    font-family: var(--ss-mono);
    flex-shrink: 0;
    direction: ltr;
}

/* Row 4: Headline story */
.ss-storyline {
    font-size: 0.82rem;
    color: var(--ss-text-secondary);
    padding: 8px 0;
    border-top: 1px solid var(--ss-border);
    line-height: 1.5;
}

/* Row 5: ML probability inline */
.ss-ml-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--ss-bg-surface);
    border-radius: var(--ss-radius-sm);
    margin-bottom: 8px;
}

.ss-ml-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

.ss-ml-dot.high { background: var(--ss-green); }
.ss-ml-dot.mid { background: var(--ss-yellow); }
.ss-ml-dot.low { background: var(--ss-red); }

.ss-ml-text {
    font-size: 0.78rem;
    color: var(--ss-text-secondary);
}

.ss-ml-text strong {
    color: var(--ss-text-primary);
    font-family: var(--ss-mono);
}

/* ---------- Detail expandable area ---------- */
.ss-details {
    border-top: 1px solid var(--ss-border);
    padding-top: 12px;
    margin-top: 4px;
}

.ss-details summary {
    cursor: pointer;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--ss-accent);
    outline: none;
    padding: 4px 0;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 6px;
}

.ss-details summary::-webkit-details-marker { display: none; }

.ss-details summary::before {
    content: "\\25B6";
    font-size: 0.6rem;
    transition: transform 0.2s;
    display: inline-block;
}

.ss-details[open] summary::before {
    transform: rotate(90deg);
}

.ss-detail-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    margin-top: 10px;
}

.ss-detail-item {
    padding: 8px;
    background: var(--ss-bg-surface);
    border-radius: var(--ss-radius-sm);
}

.ss-detail-item .ss-detail-label {
    font-size: 0.65rem;
    color: var(--ss-text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    display: block;
}

.ss-detail-item .ss-detail-value {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--ss-text-primary);
    font-family: var(--ss-mono);
    display: block;
    margin-top: 2px;
    direction: ltr;
}

/* ---------- Summary Banner ---------- */
.ss-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: var(--ss-bg-surface);
    border: 1px solid var(--ss-border);
    border-radius: var(--ss-radius-md);
    font-size: 0.78rem;
    color: var(--ss-text-secondary);
    margin: 8px 0;
    flex-wrap: wrap;
    direction: ltr;
}

.ss-banner-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.ss-banner-dot.live { background: var(--ss-green); }
.ss-banner-dot.cached { background: var(--ss-yellow); }

/* ---------- ML Legend ---------- */
.ss-legend {
    display: flex;
    gap: 16px;
    align-items: center;
    font-size: 0.72rem;
    color: var(--ss-text-muted);
    padding: 6px 0;
    flex-wrap: wrap;
}

.ss-legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
}

.ss-legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ==========================================================
   STREAMLIT OVERRIDES — 2026 MODERN UI
   ========================================================== */

/* ---------- Expanders ---------- */
.stExpander {
    border-color: var(--ss-border) !important;
    border-radius: var(--ss-radius-md) !important;
    background: var(--ss-bg-card) !important;
    box-shadow: var(--ss-shadow-sm) !important;
}

.stExpander summary {
    font-weight: 600 !important;
    color: var(--ss-text-primary) !important;
}

/* ---------- Metrics ---------- */
div[data-testid="stMetricValue"] {
    font-family: var(--ss-mono);
    font-variant-numeric: tabular-nums;
    font-weight: 800 !important;
}

div[data-testid="stMetricLabel"] {
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.04em;
    color: var(--ss-text-muted) !important;
}

/* ---------- SIDEBAR — Glass-style panel ---------- */
section[data-testid="stSidebar"] {
    background: var(--ss-bg-card) !important;
    border-right: 1px solid var(--ss-border) !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    color: var(--ss-text-primary) !important;
    text-transform: uppercase;
    margin-bottom: 0.5rem !important;
}

section[data-testid="stSidebar"] hr {
    border: none !important;
    height: 1px !important;
    background: var(--ss-border) !important;
    margin: 1rem 0 !important;
}

/* Sidebar metric cards */
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: var(--ss-bg-surface) !important;
    border-radius: var(--ss-radius-sm) !important;
    padding: 10px 14px !important;
    border: 1px solid var(--ss-border) !important;
    margin-bottom: 6px !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
}

/* Sidebar selectbox */
section[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--ss-text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

section[data-testid="stSidebar"] .stSelectbox > div > div {
    border-radius: var(--ss-radius-sm) !important;
    border-color: var(--ss-border) !important;
    font-size: 0.82rem !important;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    border-radius: var(--ss-radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--ss-border) !important;
    background: var(--ss-bg-surface) !important;
    color: var(--ss-text-primary) !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--ss-accent) !important;
    color: var(--ss-btn-text) !important;
    border-color: var(--ss-accent) !important;
    box-shadow: var(--ss-shadow-sm) !important;
}

section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    font-size: 0.72rem !important;
    color: var(--ss-text-muted) !important;
}

/* ---------- PAGE HEADER — Branded Top Bar ---------- */
.ss-page-header {
    padding: 0 0 20px 0;
    margin-bottom: 12px;
    border-bottom: 2px solid var(--ss-border);
    position: relative;
    text-align: center;
    direction: ltr;
}

.ss-page-header::before {
    content: "";
    position: absolute;
    bottom: -2px;
    left: 50%;
    transform: translateX(-50%);
    width: 120px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--ss-accent), transparent);
    border-radius: 1px;
}

.ss-page-header h1 {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    color: var(--ss-text-primary) !important;
    margin: 0 0 6px 0 !important;
    line-height: 1.2;
    text-align: center !important;
}

.ss-page-header .ss-subtitle {
    font-size: 0.82rem;
    text-align: center;
    color: var(--ss-text-muted);
    margin: 0;
    font-weight: 500;
}

/* ---------- PROGRESS BAR — Hide native (we use custom HTML bar) ---------- */
div[data-testid="stProgress"] {
    display: none !important;
}

/* ---------- TABLES / DATAFRAMES — Modern ---------- */
div[data-testid="stDataFrame"] {
    border-radius: var(--ss-radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--ss-border) !important;
    box-shadow: var(--ss-shadow-sm) !important;
}

div[data-testid="stDataFrame"] table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
}

div[data-testid="stDataFrame"] th {
    background: var(--ss-bg-surface) !important;
    color: var(--ss-text-secondary) !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 10px 14px !important;
    border-bottom: 2px solid var(--ss-border) !important;
    position: sticky;
    top: 0;
    z-index: 1;
}

div[data-testid="stDataFrame"] td {
    padding: 8px 14px !important;
    font-size: 0.82rem !important;
    font-family: var(--ss-mono) !important;
    font-variant-numeric: tabular-nums;
    border-bottom: 1px solid var(--ss-border) !important;
    color: var(--ss-text-primary) !important;
}

div[data-testid="stDataFrame"] tr:hover td {
    background: var(--ss-bg-card-hover) !important;
}

/* ---------- BUTTONS — Polished ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: var(--ss-radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
    padding: 8px 20px !important;
    letter-spacing: 0.01em;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--ss-shadow-md) !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    color: var(--ss-btn-text) !important;
    background: var(--ss-accent) !important;
    border-color: var(--ss-accent) !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: var(--ss-accent-hover) !important;
    border-color: var(--ss-accent-hover) !important;
}

.stButton > button p,
.stButton > button div,
.stButton > button span,
.stDownloadButton > button p,
.stDownloadButton > button div,
.stDownloadButton > button span {
    color: inherit !important;
}

/* ---------- DOWNLOAD BUTTONS — Styled ---------- */
.stDownloadButton > button {
    background: var(--ss-bg-card) !important;
    border: 1px solid var(--ss-border) !important;
    color: var(--ss-text-primary) !important;
}

.stDownloadButton > button:hover {
    border-color: var(--ss-accent) !important;
    color: var(--ss-accent) !important;
}

/* ---------- SELECT BOX / NUMBER INPUT — Refined ---------- */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    border-radius: var(--ss-radius-sm) !important;
    border-color: var(--ss-border) !important;
    font-size: 0.85rem !important;
    font-family: var(--ss-font) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--ss-accent) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* Labels above inputs */
.stSelectbox label,
.stMultiSelect label,
.stNumberInput label,
.stTextInput label {
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: var(--ss-text-secondary) !important;
}

/* ---------- TABS — Modern pill style ---------- */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    border-radius: var(--ss-radius-full) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 6px 18px !important;
    margin: 0 4px !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    background: var(--ss-accent) !important;
    color: var(--ss-btn-text) !important;
}

div[data-testid="stTabs"] button[aria-selected="false"] {
    color: var(--ss-text-secondary) !important;
}

div[data-testid="stTabs"] button[aria-selected="false"]:hover {
    background: var(--ss-bg-surface) !important;
    color: var(--ss-text-primary) !important;
}

/* Tab underline removal */
div[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--ss-border) !important;
    gap: 4px !important;
    padding-bottom: 8px !important;
}

/* ---------- SUCCESS / WARNING / ERROR — Polished alerts ---------- */
div[data-testid="stAlert"] {
    border-radius: var(--ss-radius-md) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-left-width: 4px !important;
}

/* ---------- SECTION DIVIDERS (st.markdown("---")) ---------- */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--ss-border) !important;
    margin: 1.5rem 0 !important;
}

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
    .ss-metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .ss-detail-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .ss-card-header {
        flex-direction: column;
        gap: 8px;
    }
    .ss-card-right {
        align-self: flex-end;
    }
    .ss-kpi-strip {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 480px) {
    .ss-metrics-grid {
        grid-template-columns: 1fr 1fr;
    }
    .ss-detail-grid {
        grid-template-columns: 1fr;
    }
    .ss-card-body {
        padding: 12px 14px;
    }
}

/* ---------- Sidebar collapsed fix ---------- */
/* When sidebar is collapsed, ensure main content stays left-aligned */
[data-testid="stSidebar"][aria-expanded="false"] ~ section[data-testid="stMain"] {
    margin-left: 0 !important;
}

/* Collapsed sidebar: hide content, prevent it from floating to center */
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}

/* Ensure collapsed control (expand arrow) stays at left edge */
[data-testid="collapsedControl"] {
    position: fixed !important;
    left: 0 !important;
    z-index: 999 !important;
    direction: ltr !important;
}

/* ---------- Sidebar collapse button RTL fix ---------- */
/* Force collapse/expand buttons to LTR so arrow direction is correct */
button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] button,
[data-testid="stSidebar"] button[kind="headerNoPadding"] {
    direction: ltr !important;
}

/* Flip the chevron icon to correct orientation in RTL context */
button[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] button svg,
[data-testid="stSidebar"] button[kind="headerNoPadding"] svg {
    transform: scaleX(-1);
}

/* Sidebar styling for collapsed/expanded states */
[data-testid="stSidebar"] {
    direction: rtl;
}

[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] code {
    direction: ltr !important;
}

/* ---------- Streamlit status/expander LTR ---------- */
div[data-testid="stStatus"],
div[data-testid="stExpander"] {
    direction: ltr;
}

/* Caption/detail text under progress bar — LTR with proper dot alignment */
.stCaption, [data-testid="stCaptionContainer"] {
    direction: ltr;
    text-align: left;
}

/* ---------- Scrollbar styling ---------- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--ss-text-muted);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--ss-text-secondary); }

/* ---------- EMPTY STATE ---------- */
.ss-empty-state {
    text-align: center;
    padding: 64px 24px;
    color: var(--ss-text-muted);
}

.ss-empty-state .ss-empty-icon {
    font-size: 3.5rem;
    margin-bottom: 16px;
    opacity: 0.6;
}

.ss-empty-state h3 {
    color: var(--ss-text-secondary) !important;
    text-align: center !important;
    font-size: 1.1rem !important;
    font-weight: 600;
    margin-bottom: 8px;
}

.ss-empty-state p {
    text-align: center !important;
    font-size: 0.85rem;
    max-width: 400px;
    margin: 0 auto;
}

/* ---------- PORTFOLIO BUTTONS ---------- */
.ss-portfolio-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    width: 100%;
    padding: 8px 16px;
    border-radius: var(--ss-radius-sm);
    font-size: 0.78rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: var(--ss-font);
    direction: ltr;
    text-align: center;
}

.ss-portfolio-btn.add {
    background: transparent;
    color: var(--ss-accent);
    border: 2px dashed var(--ss-accent);
}

.ss-portfolio-btn.add:hover {
    background: var(--ss-bg-badge);
    border-color: var(--ss-accent-hover);
    color: var(--ss-accent-hover);
}

.ss-portfolio-btn.in-portfolio {
    background: var(--ss-green-bg);
    color: var(--ss-green);
    border: 2px solid var(--ss-green);
    cursor: default;
    opacity: 0.85;
}

/* ---------- PORTFOLIO SECTION ---------- */
.ss-portfolio-card {
    background: var(--ss-bg-card);
    border: 1px solid var(--ss-border);
    border-radius: var(--ss-radius-md);
    padding: 14px 18px;
    margin: 8px 0;
    box-shadow: var(--ss-shadow-sm);
    direction: ltr;
    text-align: left;
}

.ss-portfolio-card:hover {
    box-shadow: var(--ss-shadow-md);
    border-color: var(--ss-border-hover);
}

.ss-portfolio-card .pf-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.ss-portfolio-card .pf-ticker {
    font-size: 1.1rem;
    font-weight: 800;
    font-family: var(--ss-mono);
    color: var(--ss-text-primary);
}

.ss-portfolio-card .pf-return {
    font-size: 1rem;
    font-weight: 800;
    font-family: var(--ss-mono);
}

.ss-portfolio-card .pf-return.positive { color: var(--ss-green); }
.ss-portfolio-card .pf-return.negative { color: var(--ss-red); }
.ss-portfolio-card .pf-return.neutral  { color: var(--ss-text-muted); }

.ss-portfolio-card .pf-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    font-size: 0.72rem;
}

.ss-portfolio-card .pf-metric-label {
    color: var(--ss-text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.62rem;
}

.ss-portfolio-card .pf-metric-value {
    color: var(--ss-text-primary);
    font-weight: 700;
    font-family: var(--ss-mono);
    font-size: 0.78rem;
}

/* Portfolio sidebar summary */
.ss-portfolio-summary {
    background: var(--ss-bg-card);
    border: 1px solid var(--ss-border);
    border-radius: var(--ss-radius-md);
    padding: 12px 14px;
    margin: 8px 0;
    direction: ltr;
    text-align: left;
}

.ss-portfolio-summary .pf-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    font-size: 0.78rem;
}

.ss-portfolio-summary .pf-stat-label {
    color: var(--ss-text-muted);
    font-weight: 600;
}

.ss-portfolio-summary .pf-stat-value {
    color: var(--ss-text-primary);
    font-weight: 700;
    font-family: var(--ss-mono);
}

</style>
"""
