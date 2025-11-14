"""
Modern Design System for Stock Scout
Tesla/Apple-inspired minimal futuristic aesthetic
"""

# Color Palette - Modern & Minimal
COLORS = {
    # Base colors
    "background": "#FAFAFA",
    "surface": "#FFFFFF",
    "surface_elevated": "#FFFFFF",
    
    # Text
    "text_primary": "#071028",
    "text_secondary": "#55607a",
    "text_tertiary": "#94a3b8",
    
    # Brand/Accent (modern navy + teal)
    "primary": "#0B1226",
    "primary_hover": "#0f2138",
    "accent": "#06B6D4",
    "accent_hover": "#0891b2",
    
    # Status colors
    "success": "#10B981",
    "success_bg": "#ECFDF5",
    "warning": "#F59E0B",
    "warning_bg": "#FEF3C7",
    "danger": "#EF4444",
    "danger_bg": "#FEE2E2",
    "info": "#3B82F6",
    "info_bg": "#EFF6FF",
    
    # Core/Speculative
    "core": "#10B981",
    "core_bg": "#F0FDF4",
    "core_border": "#86EFAC",
    "speculative": "#F59E0B",
    "speculative_bg": "#FEF3C7",
    "speculative_border": "#FBBF24",
    
    # Borders & Dividers
    "border": "#E5E7EB",
    "border_light": "#F3F4F6",
    "divider": "#F3F4F6",
}

# Typography System
FONTS = {
    "primary": "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', 'Segoe UI', system-ui, sans-serif",
    "mono": "'SF Mono', 'JetBrains Mono', 'Fira Code', Consolas, monospace",
}

# Shadows - Subtle elevation
SHADOWS = {
    "xs": "0 1px 2px 0 rgba(0, 0, 0, 0.03)",
    "sm": "0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.03)",
    "md": "0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -2px rgba(0, 0, 0, 0.04)",
    "lg": "0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -4px rgba(0, 0, 0, 0.05)",
    "xl": "0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.06)",
}

# Spacing System (rem-based)
SPACING = {
    "xs": "0.25rem",   # 4px
    "sm": "0.5rem",    # 8px
    "md": "1rem",      # 16px
    "lg": "1.5rem",    # 24px
    "xl": "2rem",      # 32px
    "2xl": "3rem",     # 48px
    "3xl": "4rem",     # 64px
}

# Border Radius
RADIUS = {
    "sm": "0.375rem",  # 6px
    "md": "0.5rem",    # 8px
    "lg": "0.75rem",   # 12px
    "xl": "1rem",      # 16px
    "full": "9999px",
}

# Transitions
TRANSITIONS = {
    "fast": "150ms cubic-bezier(0.4, 0, 0.2, 1)",
    "base": "200ms cubic-bezier(0.4, 0, 0.2, 1)",
    "slow": "300ms cubic-bezier(0.4, 0, 0.2, 1)",
    "smooth": "400ms cubic-bezier(0.4, 0, 0.1, 1)",
}

def get_modern_css():
    """Generate complete modern CSS with design system"""
    return f"""
<style>
/* ==================== GLOBAL RESET & BASE ==================== */
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

:root {{
    --color-bg: {COLORS['background']};
    --color-surface: {COLORS['surface']};
    --color-text: {COLORS['text_primary']};
    --color-text-secondary: {COLORS['text_secondary']};
    --color-primary: {COLORS['primary']};
    --color-accent: {COLORS['accent']};
    --font-primary: {FONTS['primary']};
    --font-mono: {FONTS['mono']};
    --shadow-sm: {SHADOWS['sm']};
    --shadow-md: {SHADOWS['md']};
    --shadow-lg: {SHADOWS['lg']};
    --radius-md: {RADIUS['md']};
    --radius-lg: {RADIUS['lg']};
    --transition: {TRANSITIONS['base']};
}}

/* Override Streamlit defaults */
.stApp {{
    background-color: var(--color-bg);
}}

section.main > div {{
    padding: 2rem 1rem;
    max-width: 1400px;
    margin: 0 auto;
}}

/* ==================== TYPOGRAPHY ==================== */
.stApp, .stMarkdown, p, div {{
    font-family: var(--font-primary);
    color: var(--color-text);
    line-height: 1.6;
}}

h1, h2, h3, h4 {{
    font-family: var(--font-primary);
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--color-text);
}}

h1 {{ font-size: clamp(2rem, 5vw, 3rem); margin-bottom: 1rem; }}
h2 {{ font-size: clamp(1.5rem, 3.5vw, 2rem); margin-bottom: 0.75rem; }}
h3 {{ font-size: clamp(1.25rem, 2.5vw, 1.5rem); margin-bottom: 0.5rem; }}

/* ==================== MODERN CARD SYSTEM ==================== */
.modern-card {{
    background: var(--color-surface);
    border: 1px solid {COLORS['border']};
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-sm);
    transition: all var(--transition);
    direction: ltr;
    text-align: left;
}}

.modern-card:hover {{
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}}

.card-core {{
    background: linear-gradient(135deg, {COLORS['core_bg']} 0%, {COLORS['surface']} 100%);
    border-left: 4px solid {COLORS['core']};
}}

.card-speculative {{
    background: linear-gradient(135deg, {COLORS['speculative_bg']} 0%, {COLORS['surface']} 100%);
    border-left: 4px solid {COLORS['speculative']};
}}

/* ==================== BADGE SYSTEM ==================== */
.badge {{
    display: inline-flex;
    align-items: center;
    padding: 0.375rem 0.875rem;
    border-radius: {RADIUS['full']};
    font-size: 0.875rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
    transition: all {TRANSITIONS['fast']};
}}

.badge-primary {{
    background: {COLORS['primary']};
    color: white;
}}

.badge-success {{
    background: {COLORS['success_bg']};
    color: {COLORS['success']};
    border: 1px solid {COLORS['success']};
}}

.badge-warning {{
    background: {COLORS['warning_bg']};
    color: {COLORS['warning']};
    border: 1px solid {COLORS['warning']};
}}

.badge-danger {{
    background: {COLORS['danger_bg']};
    color: {COLORS['danger']};
    border: 1px solid {COLORS['danger']};
}}

.badge-info {{
    background: {COLORS['info_bg']};
    color: {COLORS['info']};
    border: 1px solid {COLORS['info']};
}}

/* ==================== GRID SYSTEM ==================== */
.data-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr));
    gap: 1rem;
    margin-top: 1rem;
}}

.data-item {{
    padding: 0.75rem;
    background: {COLORS['surface_elevated']};
    border-radius: var(--radius-md);
    border: 1px solid {COLORS['border_light']};
    transition: background {TRANSITIONS['fast']};
}}

.data-item:hover {{
    background: {COLORS['background']};
}}

.data-label {{
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}}

.data-value {{
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--color-text);
    font-family: var(--font-mono);
}}

/* ==================== RESPONSIVE TABLES ==================== */
.modern-table-container {{
    overflow-x: auto;
    border-radius: var(--radius-lg);
    background: var(--color-surface);
    box-shadow: var(--shadow-sm);
    margin: 1rem 0;
}}

.modern-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
}}

.modern-table th {{
    background: {COLORS['background']};
    padding: 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 2px solid {COLORS['border']};
}}

.modern-table td {{
    padding: 1rem;
    text-align: left;
    border-bottom: 1px solid {COLORS['border_light']};
    transition: background {TRANSITIONS['fast']};
}}

.modern-table tr:hover td {{
    background: {COLORS['background']};
}}

.modern-table tr:last-child td {{
    border-bottom: none;
}}

/* ==================== BUTTONS & INTERACTIONS ==================== */
.stButton > button {{
    background: {COLORS['primary']} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 1.5rem !important;
    font-family: var(--font-primary) !important;
    font-weight: 600 !important;
    font-size: 0.9375rem !important;
    cursor: pointer !important;
    transition: all var(--transition) !important;
    box-shadow: var(--shadow-sm) !important;
}}

.stButton > button:hover {{
    background: {COLORS['primary_hover']} !important;
    color: #FFFFFF !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}}

/* ==================== METRICS & KPIs ==================== */
[data-testid="stMetricValue"] {{
    font-family: var(--font-mono);
    font-size: 2rem !important;
    font-weight: 700;
    color: var(--color-text);
}}

[data-testid="stMetricLabel"] {{
    font-size: 0.875rem !important;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ==================== SIDEBAR MODERN STYLE ==================== */
[data-testid="stSidebar"] {{
    background: var(--color-surface);
    border-right: 1px solid {COLORS['border']};
    padding: 1.5rem 1rem;
}}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > * {{
    font-size: 0.9375rem;
}}

/* ==================== MOBILE RESPONSIVE ==================== */
@media (max-width: 768px) {{
    section.main > div {{
        padding: 1rem 0.75rem;
    }}
    
    .modern-card {{
        padding: 1rem;
        margin: 0.75rem 0;
    }}
    
    .data-grid {{
        grid-template-columns: 1fr;
        gap: 0.75rem;
    }}
    
    h1 {{ font-size: 1.75rem; }}
    h2 {{ font-size: 1.5rem; }}
    h3 {{ font-size: 1.25rem; }}
    
    .modern-table th,
    .modern-table td {{
        padding: 0.75rem 0.5rem;
        font-size: 0.875rem;
    }}
}}

/* ==================== UTILITY CLASSES ==================== */
.text-center {{ text-align: center; }}
.text-muted {{ color: var(--color-text-secondary); }}
.text-small {{ font-size: 0.875rem; }}
.text-mono {{ font-family: var(--font-mono); }}
.fw-600 {{ font-weight: 600; }}
.fw-700 {{ font-weight: 700; }}
.mb-1 {{ margin-bottom: 0.5rem; }}
.mb-2 {{ margin-bottom: 1rem; }}
.mt-1 {{ margin-top: 0.5rem; }}
.mt-2 {{ margin-top: 1rem; }}
.flex {{ display: flex; }}
.flex-wrap {{ flex-wrap: wrap; }}
.gap-1 {{ gap: 0.5rem; }}
.gap-2 {{ gap: 1rem; }}
.align-center {{ align-items: center; }}
.justify-between {{ justify-content: space-between; }}

/* Hide Streamlit branding */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Progress bar styling */
.stProgress > div > div > div {{
    background-color: var(--color-accent);
    border-radius: {RADIUS['full']};
}}

/* Spinner styling */
.stSpinner > div {{
    border-top-color: var(--color-accent) !important;
}}
</style>
"""

# Export for use in stock_scout.py
__all__ = ['get_modern_css', 'COLORS', 'FONTS', 'SHADOWS', 'SPACING', 'RADIUS', 'TRANSITIONS']
