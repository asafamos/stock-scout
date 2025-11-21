"""
Embedded CSS for recommendation cards (used in st_html iframes)
Since st_html creates isolated iframes, we need to inject CSS directly into each card
"""

def get_card_css():
    """Returns CSS to be embedded in each card HTML"""
    return """
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: 'Inter', system-ui, sans-serif; font-size:14px; line-height:1.5; color:#1e293b; }
.clean-card { border:1px solid #e2e8f0; border-radius:12px; padding:1.25rem; background:#fff; margin-bottom:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.clean-card.core { border-left:4px solid #10B981; }
.clean-card.speculative { border-left:4px solid #F59E0B; }
.card-header { margin-bottom:1rem; display:flex; justify-content:space-between; align-items:flex-start; }
.ticker-line { display:flex; gap:0.5rem; align-items:center; }
.ticker-badge { font-size:1.1rem; font-weight:700; color:#0f172a; font-family:'SF Mono', 'Roboto Mono', monospace; }
.type-badge { background:#e2e8f0; color:#475569; padding:3px 8px; border-radius:6px; font-size:0.65rem; font-weight:600; text-transform:uppercase; }
.overall-score { font-size:2rem; font-weight:700; color:#0f172a; font-variant-numeric:tabular-nums; margin:0; text-align:right; }
.score-label { font-size:1rem; font-weight:400; color:#64748b; margin-left:4px; }
.top-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; margin-bottom:0.75rem; }
.field { display:flex; flex-direction:column; gap:4px; padding:10px 12px; background:#f8fafc; border-radius:8px; border:1px solid #f1f5f9; }
.field .label { color:#64748b; font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
.field .value { font-family:'SF Mono', 'Roboto Mono', monospace; font-size:0.9rem; font-weight:500; color:#0f172a; }
.value.tabular { font-variant-numeric:tabular-nums; }
.value .band { color:#64748b; font-size:0.75rem; font-weight:400; }
details.more-info { margin-top:0.75rem; }
details summary { cursor:pointer; font-weight:600; color:#475569; font-size:0.8rem; padding:6px 0; border-top:1px solid #e2e8f0; user-select:none; }
details summary:hover { color:#0f172a; }
.detail-grid { display:grid; grid-template-columns:repeat(2, 1fr); gap:8px; margin-top:8px; padding-top:8px; }
.detail-grid .field { background:#fafbfc; border:1px solid #f1f5f9; padding:8px 10px; }
.detail-grid .field .label { font-size:0.65rem; }
.detail-grid .field .value { font-size:0.8rem; }
.badge.ai { background:#10B981; color:#fff; padding:3px 6px; border-radius:4px; font-size:0.65rem; font-weight:600; margin-left:4px; }
.badge.tech { background:#6366f1; color:#fff; padding:3px 6px; border-radius:4px; font-size:0.65rem; font-weight:600; margin-left:4px; }
</style>
"""
