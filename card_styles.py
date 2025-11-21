"""
Embedded CSS for recommendation cards (used in st_html iframes)
Since st_html creates isolated iframes, we need to inject CSS directly into each card
"""

def get_card_css():
    """Returns CSS to be embedded in each card HTML"""
    return """
<style>
* { margin:0; padding:0; box-sizing:border-box; }
.card-wrapper { margin:0 !important; padding:0 !important; }
.clean-card { border:1px solid #e2e8f0; border-radius:12px; padding:0.85rem 0.95rem; background:#fff; margin:0 0 2px 0 !important; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.clean-card.core { border-left:4px solid #10B981; }
.clean-card.speculative { border-left:4px solid #F59E0B; }
.card-header { margin-bottom:1rem; display:flex; justify-content:space-between; align-items:flex-start; }
.ticker-line { display:flex; gap:0.5rem; align-items:center; }
.ticker-badge { font-size:1.1rem; font-weight:700; color:#0f172a; font-family:'SF Mono', 'Roboto Mono', monospace; }
.type-badge { background:#e2e8f0; color:#475569; padding:3px 8px; border-radius:6px; font-size:0.65rem; font-weight:600; text-transform:uppercase; }
.field { display:flex; flex-direction:row; justify-content:space-between; align-items:baseline; padding:6px 8px; background:#f8fafc; border-radius:6px; border:1px solid #f1f5f9; }
.field .label { flex:1; color:#64748b; font-size:0.55rem; font-weight:600; text-transform:uppercase; letter-spacing:0.4px; text-align:left; }
.field .value { flex:1; font-family:'SF Mono','Roboto Mono', monospace; font-size:0.8rem; font-weight:600; color:#0f172a; text-align:right; font-variant-numeric:tabular-nums; }
.value.tabular { font-variant-numeric:tabular-nums; }
.value .band { color:#64748b; font-size:0.75rem; font-weight:400; }
.detail-grid { display:grid; grid-template-columns:repeat(2, 1fr); gap:4px; margin-top:4px; padding-top:4px; }
.detail-grid .field { background:#fafbfc; border:1px solid #f1f5f9; padding:8px 10px; }
.detail-grid .field .label { font-size:0.65rem; }
.detail-grid .field .value { font-size:0.8rem; }
.badge.ai { background:#10B981; color:#fff; padding:3px 6px; border-radius:4px; font-size:0.65rem; font-weight:600; margin-left:4px; }
.badge.tech { background:#6366f1; color:#fff; padding:3px 6px; border-radius:4px; font-size:0.65rem; font-weight:600; margin-left:4px; }
</style>
"""
