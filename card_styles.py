
"""
Embedded CSS for recommendation cards.
Used by the main Streamlit app when rendering build_clean_card().
"""

def get_card_css() -> str:
    """Return the full CSS block for recommendation cards."""
    return """
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* Container for each card as rendered by Streamlit */
.recommend-card {
    width: 100%;
    margin: 8px 0;
}

/* Main card */
.clean-card {
    display: block;
    position: relative;
    width: 100%;
    max-width: 100%;
    min-width: 0;
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
    padding: 14px 16px 12px 16px;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 sans-serif;
    color: #0f172a;
}

/* Core vs Spec border accent */
.clean-card.core {
    border-left: 4px solid #10b981;
}

.clean-card.speculative {
    border-left: 4px solid #f97316;
}

/* Header: ticker + score */
.card-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 4px;
}

.ticker-line {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.95rem;
    font-weight: 600;
    color: #0f172a;
}

.ticker-badge {
    padding: 2px 8px;
    border-radius: 999px;
    background: #eff6ff;
    color: #1d4ed8;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.rank-badge {
    padding: 2px 6px;
    border-radius: 999px;
    background: #f3f4f6;
    color: #4b5563;
    font-size: 0.75rem;
    font-weight: 600;
}

.overall-score {
    font-size: 1.6rem;
    font-weight: 700;
    color: #111827;
    display: flex;
    align-items: baseline;
    gap: 4px;
}

.overall-score .score-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #6b7280;
}

/* Top line: entry/target/potential */
.entry-target-line {
    font-size: 0.82rem;
    color: #374151;
    margin-bottom: 6px;
}

.entry-target-line b {
    font-weight: 600;
}

.potential {
    margin-left: 6px;
    font-weight: 600;
}

/* Small type badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.badge.core {
    background: #dcfce7;
    color: #15803d;
}

.badge.spec {
    background: #fef3c7;
    color: #b45309;
}

.badge.ai {
    background: #10b981;
    color: #ffffff;
}

.badge.tech {
    background: #6366f1;
    color: #ffffff;
}

/* 2-row grid with 3 items per row on desktop */
.top-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 4px 10px;
    margin: 4px 0 6px 0;
}

.field .label {
    display: block;
    font-size: 0.7rem;
    color: #6b7280;
    margin-bottom: 1px;
}

.field .value {
    font-size: 0.82rem;
    font-weight: 500;
    color: #111827;
}

.field .band {
    margin-left: 4px;
    font-size: 0.72rem;
    color: #6b7280;
}

/* Details section */
.more-info {
    margin-top: 4px;
    font-size: 0.78rem;
}

.more-info > summary {
    cursor: pointer;
    font-weight: 500;
    color: #4b5563;
    outline: none;
}

.more-info[open] > summary {
    color: #1d4ed8;
}

.detail-grid {
    margin-top: 4px;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 3px 10px;
}

.detail-grid .field .label {
    font-size: 0.7rem;
    color: #6b7280;
}

.detail-grid .field .value {
    font-size: 0.78rem;
    color: #111827;
}

/* Small helper text */
.helper-text {
    margin-top: 4px;
    font-size: 0.7rem;
    color: #6b7280;
}

/* Make numbers align nicely */
.tabular {
    font-variant-numeric: tabular-nums;
}

/* Signal bullets list */
.signal-bullets {
    list-style: none;
    margin: 4px 0;
    padding-left: 0;
}

.signal-bullets li {
    padding: 2px 0;
    font-size: 0.78rem;
    color: #374151;
    line-height: 1.3;
}

.signal-bullets li::before {
    content: "• ";
    color: #10b981;
    font-weight: 600;
    margin-right: 4px;
}

/* ML 20d line */
.ml20d-line {
    font-size: 0.75rem;
    color: #6b7280;
    margin: 2px 0;
}

/* ML gating badge */
.ml-gating-badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    background: #dcfce7;
    color: #15803d;
    font-size: 0.7rem;
    font-weight: 600;
    margin: 2px 0;
}

/* Type badge styling */
.type-badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    margin-left: 4px;
}

.type-badge.core {
    background: #dcfce7;
    color: #15803d;
}

.type-badge.spec {
    background: #fef3c7;
    color: #b45309;
}

/* Responsive tweaks */
@media (max-width: 920px) {
    .clean-card {
        padding: 12px 12px 10px 12px;
    }
}

@media (max-width: 640px) {
    .card-header {
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
    }
    .top-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .detail-grid {
        grid-template-columns: repeat(1, minmax(0, 1fr));
    }
}
</style>
"""
