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
.clean-card { border:1px solid #e2e8f0; border-radius:12px; padding:0.75rem 0.85rem; background:#fff; margin:0 0 6px 0 !important; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
.clean-card.core { border-left:4px solid #10B981; }
.clean-card.speculative { border-left:4px solid #F59E0B; }
.card-header { margin-bottom:0.75rem; display:flex; justify-content:space-between; align-items:flex-start; }
.ticker-line { display:flex; gap:0.5rem; align-items:center; }
.ticker-badge { font-size:1.05rem; font-weight:700; color:#0f172a; font-family:'SF Mono', 'Roboto Mono', monospace; }
.type-badge { background:#e2e8f0; color:#475569; padding:3px 8px; border-radius:6px; font-size:0.65rem; font-weight:600; text-transform:uppercase; }
.top-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:4px; margin-bottom:6px; }
.field { display:flex; flex-direction:row; justify-content:space-between; align-items:baseline; padding:5px 7px; background:#f8fafc; border-radius:6px; border:1px solid #f1f5f9; }
.field .label { flex:1; color:#64748b; font-size:0.55rem; font-weight:600; text-transform:uppercase; letter-spacing:0.4px; text-align:left; }
.field .value { flex:1; font-family:'SF Mono','Roboto Mono', monospace; font-size:0.8rem; font-weight:600; color:#0f172a; text-align:right; font-variant-numeric:tabular-nums; }
.value.tabular { font-variant-numeric:tabular-nums; }
.value .band { color:#64748b; font-size:0.75rem; font-weight:400; }
.more-info { margin-top:6px; }
.more-info summary { cursor:pointer; color:#475569; font-size:0.7rem; font-weight:600; padding:4px 0; user-select:none; }
.more-info summary:hover { color:#0f172a; }
.detail-grid { display:grid; grid-template-columns:repeat(2, 1fr); gap:4px; margin-top:6px; padding-top:6px; }
.detail-grid .field { background:#fafbfc; border:1px solid #f1f5f9; padding:6px 8px; }
.detail-grid .field .label { font-size:0.6rem; }
.detail-grid .field .value { font-size:0.75rem; }
.overall-score { font-size:2rem; font-weight:800; color:#0f172a; line-height:1; }
.score-label { font-size:0.9rem; color:#64748b; font-weight:400; }
.badge.ai { background:#10B981; color:#fff; padding:3px 6px; border-radius:4px; font-size:0.65rem; font-weight:600; margin-left:4px; }
.badge.tech { background:#6366f1; color:#fff; padding:3px 6px; border-radius:4px; font-size:0.65rem; font-weight:600; margin-left:4px; }
</style>
"""
