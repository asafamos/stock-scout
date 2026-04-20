"""Stock Scout 2026 — Modern Design Layer.

Overlays the existing design system with 2026-era polish:
- Softer shadows, smoother gradients, glass morphism
- Better typography (variable fonts, optical sizing)
- Micro-interactions (hover states, focus rings)
- Modern data density (more info, less chrome)
- Enhanced dark mode contrast
"""


def get_design_2026_css() -> str:
    return """
<style>
/* STOCKSCOUT 2026 v3 - cache bust 20260421 */
/* ================================================================
   STOCK SCOUT 2026 — MODERN UI LAYER (forced overlay)
   All rules use !important to guarantee override of design_system.py
   ================================================================ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ---------- Refined Color Tokens ---------- */
:root {
    --ss-bg-primary: #fafbfc;
    --ss-bg-card: rgba(255, 255, 255, 0.8);
    --ss-bg-card-hover: #ffffff;
    --ss-bg-surface: #f1f5f9;
    --ss-border: rgba(226, 232, 240, 0.7);
    --ss-border-subtle: rgba(226, 232, 240, 0.4);
    --ss-text-primary: #0b1220;
    --ss-text-secondary: #475569;
    --ss-text-muted: #94a3b8;

    /* Modern shadows — soft, layered */
    --ss-shadow-sm: 0 1px 2px 0 rgba(15, 23, 42, 0.04);
    --ss-shadow-md: 0 4px 6px -1px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.04);
    --ss-shadow-lg: 0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.05);
    --ss-shadow-xl: 0 20px 25px -5px rgba(15, 23, 42, 0.10), 0 8px 10px -6px rgba(15, 23, 42, 0.06);
    --ss-shadow-glow: 0 0 0 1px rgba(59, 130, 246, 0.1), 0 4px 12px rgba(59, 130, 246, 0.15);

    /* Subtle gradients */
    --ss-grad-accent: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    --ss-grad-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
    --ss-grad-warning: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    --ss-grad-danger: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    --ss-grad-bg: linear-gradient(180deg, #fafbfc 0%, #f1f5f9 100%);

    /* Motion */
    --ss-ease: cubic-bezier(0.4, 0, 0.2, 1);
    --ss-ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
    --ss-dur-fast: 150ms;
    --ss-dur-med: 250ms;
    --ss-dur-slow: 400ms;
}

@media (prefers-color-scheme: dark) {
    :root {
        --ss-bg-primary: #0b1220;
        --ss-bg-card: rgba(17, 24, 39, 0.7);
        --ss-bg-card-hover: rgba(31, 41, 55, 0.9);
        --ss-bg-surface: #111827;
        --ss-border: rgba(55, 65, 81, 0.5);
        --ss-border-subtle: rgba(55, 65, 81, 0.3);
        --ss-text-primary: #f1f5f9;
        --ss-text-secondary: #cbd5e1;
        --ss-text-muted: #64748b;

        --ss-shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.2);
        --ss-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.25), 0 2px 4px -2px rgba(0, 0, 0, 0.15);
        --ss-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.30), 0 4px 6px -4px rgba(0, 0, 0, 0.20);
        --ss-shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.35), 0 8px 10px -6px rgba(0, 0, 0, 0.25);
        --ss-grad-bg: linear-gradient(180deg, #0b1220 0%, #111827 100%);
    }
}

/* ---------- Global Typography ---------- */
html, body, [class*="css"], .stApp, .main, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-feature-settings: 'cv11', 'ss01', 'ss03', 'cv04', 'cv02' !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    text-rendering: optimizeLegibility !important;
}

h1, h2, h3, h4, h5, h6 {
    letter-spacing: -0.02em !important;
    font-weight: 700 !important;
    color: var(--ss-text-primary) !important;
}

h1 { font-size: 2rem !important; line-height: 1.2 !important; }
h2 { font-size: 1.5rem !important; line-height: 1.3 !important; }
h3 { font-size: 1.25rem !important; line-height: 1.4 !important; }

/* ---------- Streamlit App Container ---------- */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--ss-grad-bg) !important;
}

/* ---------- Main Content Area ---------- */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px;
}

/* ---------- Section Headers ---------- */
.ss-section-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 0;
    margin: 20px 0 16px 0;
    border-bottom: 1px solid var(--ss-border);
}

.ss-section-header h2 {
    margin: 0;
    font-size: 1.375rem;
    font-weight: 700;
    letter-spacing: -0.015em;
}

.ss-section-header .ss-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: var(--ss-shadow-sm);
}

/* ---------- Cards (all variants) ---------- */
.ss-card,
.ss-recommendation-card,
.ss-card-wrapper,
[data-testid="stVerticalBlockBorderWrapper"],
div[style*="border"][style*="radius"]:not([class*="ss-portfolio"]):not(.ss-section-header) {
    background: var(--ss-bg-card) !important;
    backdrop-filter: blur(16px) saturate(1.2) !important;
    -webkit-backdrop-filter: blur(16px) saturate(1.2) !important;
    border: 1px solid var(--ss-border) !important;
    border-radius: 16px !important;
    box-shadow: var(--ss-shadow-md) !important;
    transition: transform var(--ss-dur-med) var(--ss-ease),
                box-shadow var(--ss-dur-med) var(--ss-ease),
                border-color var(--ss-dur-fast) var(--ss-ease) !important;
}

.ss-card:hover,
.ss-recommendation-card:hover {
    box-shadow: var(--ss-shadow-lg) !important;
    border-color: rgba(59, 130, 246, 0.4) !important;
    transform: translateY(-2px) !important;
}

/* Score circle — make it pop */
.ss-score-circle, [class*="score-circle"] {
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
}

/* ---------- Portfolio Summary (Sidebar) ---------- */
.ss-portfolio-summary {
    background: var(--ss-bg-card);
    backdrop-filter: blur(10px);
    border: 1px solid var(--ss-border);
    border-radius: 12px;
    padding: 14px;
    margin: 8px 0;
    box-shadow: var(--ss-shadow-sm);
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: -0.005em !important;
    transition: all var(--ss-dur-fast) var(--ss-ease) !important;
    box-shadow: var(--ss-shadow-sm) !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--ss-shadow-md) !important;
}

.stButton > button[kind="primary"] {
    background: var(--ss-grad-accent) !important;
    border: none !important;
    color: white !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: var(--ss-shadow-glow) !important;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: var(--ss-bg-card);
    backdrop-filter: blur(8px);
    border: 1px solid var(--ss-border);
    border-radius: 12px;
    padding: 14px;
    transition: all var(--ss-dur-med) var(--ss-ease);
}

[data-testid="stMetric"]:hover {
    box-shadow: var(--ss-shadow-md);
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: var(--ss-text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    font-feature-settings: 'tnum';
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--ss-bg-surface);
    border-right: 1px solid var(--ss-border);
}

[data-testid="stSidebar"] > div > div {
    padding: 1.25rem 1rem;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--ss-bg-surface);
    padding: 4px;
    border-radius: 10px;
    border: 1px solid var(--ss-border);
}

.stTabs [data-baseweb="tab"] {
    padding: 8px 16px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all var(--ss-dur-fast) var(--ss-ease);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--ss-bg-card) !important;
    box-shadow: var(--ss-shadow-sm);
    color: var(--ss-text-primary) !important;
}

/* ---------- Expanders ---------- */
[data-testid="stExpander"] {
    background: var(--ss-bg-card);
    border: 1px solid var(--ss-border) !important;
    border-radius: 12px !important;
    box-shadow: var(--ss-shadow-sm);
}

[data-testid="stExpander"] summary {
    padding: 12px 14px !important;
    font-weight: 600 !important;
}

/* ---------- Inputs ---------- */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1px solid var(--ss-border) !important;
    transition: border-color var(--ss-dur-fast) var(--ss-ease), box-shadow var(--ss-dur-fast) var(--ss-ease) !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
}

/* ---------- Dataframes ---------- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--ss-border);
    box-shadow: var(--ss-shadow-sm);
}

/* ---------- Progress Bars ---------- */
.stProgress > div > div > div {
    background: var(--ss-grad-accent) !important;
    border-radius: 999px !important;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    backdrop-filter: blur(8px);
}

/* ---------- Info-specific polish ---------- */
.ss-bg-badge {
    background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(99,102,241,0.08)) !important;
}

/* ---------- Subtle animations ---------- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.ss-section-header, .ss-card, [data-testid="stMetric"] {
    animation: fadeIn var(--ss-dur-med) var(--ss-ease) both;
}

/* ---------- Scrollbars ---------- */
*::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
*::-webkit-scrollbar-track {
    background: transparent;
}
*::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.3);
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: padding-box;
}
*::-webkit-scrollbar-thumb:hover {
    background: rgba(148, 163, 184, 0.5);
    background-clip: padding-box;
}

/* ---------- Hide Streamlit branding/footer ---------- */
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

/* ---------- Responsive Improvements ---------- */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    h1 { font-size: 1.5rem; }
    h2 { font-size: 1.25rem; }
}

/* ---------- Top decoration: subtle ambient glow ---------- */
body::before {
    content: '';
    position: fixed;
    top: -10%;
    left: 50%;
    transform: translateX(-50%);
    width: 60%;
    height: 300px;
    background: radial-gradient(ellipse, rgba(59, 130, 246, 0.08) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

@media (prefers-color-scheme: dark) {
    body::before {
        background: radial-gradient(ellipse, rgba(99, 102, 241, 0.12) 0%, transparent 70%);
    }
}
</style>
"""
