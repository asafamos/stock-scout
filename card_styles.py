"""
Embedded CSS for recommendation cards (used in st_html iframes)
Since st_html creates isolated iframes, we need to inject CSS directly into each card
"""

def get_card_css():
    """Returns CSS to be embedded in each card HTML"""
    return """
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: #071028;
    background: transparent;
}

.modern-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.03);
    direction: ltr;
    text-align: left;
}

.card-core {
    background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 100%);
    border-left: 4px solid #10B981;
}

.card-speculative {
    background: linear-gradient(135deg, #FEF3C7 0%, #FFFFFF 100%);
    border-left: 4px solid #F59E0B;
}

.flex-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
}

.modern-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.375rem 0.875rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
    background: #0B1226;
    color: white;
}

.badge-success {
    background: #ECFDF5;
    color: #10B981;
    border: 1px solid #10B981;
}

.badge-warning {
    background: #FEF3C7;
    color: #F59E0B;
    border: 1px solid #F59E0B;
}

.badge-primary {
    background: #0B1226;
    color: white;
}

.badge-quality-high {
    background: #ECFDF5;
    color: #10B981;
    border: 1px solid #10B981;
}

.badge-quality-medium {
    background: #FEF3C7;
    color: #F59E0B;
    border: 1px solid #F59E0B;
}

.badge-quality-low {
    background: #FEE2E2;
    color: #EF4444;
    border: 1px solid #EF4444;
}

.badge-danger {
    background: #FEE2E2;
    color: #EF4444;
    border: 1px solid #EF4444;
}

.badge-partial {
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #F59E0B;
    font-size: 0.75rem;
    padding: 0.25rem 0.625rem;
}

.badge-missing {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #EF4444;
    font-size: 0.75rem;
    padding: 0.25rem 0.625rem;
}

.warning-box {
    background: #FEF3C7;
    border: 1px solid #F59E0B;
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    color: #92400E;
    font-size: 0.875rem;
}

.modern-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
}

.item {
    padding: 0.75rem;
    background: #F9FAFB;
    border-radius: 8px;
    border: 1px solid #F3F4F6;
    font-size: 0.875rem;
    direction: ltr;
    text-align: left;
}

.item b {
    color: #55607a;
    font-weight: 600;
    display: block;
    margin-bottom: 0.25rem;
    text-align: left;
}

.item .value {
    direction: ltr;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

.section-divider {
    grid-column: 1 / -1;
    font-weight: 700;
    font-size: 1rem;
    color: #071028;
    padding: 0.5rem 0;
    border-bottom: 2px solid #E5E7EB;
    margin-top: 0.5rem;
}

@media (max-width: 768px) {
    .modern-card {
        padding: 1rem;
    }
    .modern-grid {
        grid-template-columns: 1fr;
        gap: 0.5rem;
    }
    .item {
        padding: 0.5rem;
        font-size: 0.8125rem;
    }
}
</style>
"""
