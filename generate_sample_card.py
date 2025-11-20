"""
Sample card generator for validation
"""
import pandas as pd
import numpy as np
from stock_scout import build_clean_card
from card_styles import get_card_css

# Create a sample row
sample_row = pd.Series({
    'Ticker': 'AAPL',
    'Overall_Rank': 1,
    'overall_score': 85.5,
    'conviction_v2_final': 85.5,
    'Target_Price': 225.0,
    'Entry_Price': 195.0,
    'Target_Date': '2025-04-15',
    'Target_Source': 'AI',
    'rr': 2.8,
    'rr_score_v2': 82.0,
    'rr_band': 'Good',
    'risk_meter_v2': 35.2,
    'risk_label_v2': 'Low',
    'Reliability_Score': 92.3,
    'reliability_score_v2': 92.3,
    'Price_Reliability': 95.0,
    'Fundamental_Reliability': 89.5,
    'ml_status': 'High (88%)',
    'ML_Confidence': 'High (88%)',
    'conviction_v2_base': 83.0,
})

# Generate card HTML
card_html = get_card_css() + build_clean_card(sample_row, speculative=False)

# Write to file
with open('/workspaces/stock-scout-2/sample_card.html', 'w') as f:
    f.write(card_html)

print("✅ Sample card generated: sample_card.html")
print(f"Overall Score: {sample_row['overall_score']:.1f}/100")
print(f"RR: {sample_row['rr']:.2f} (Score: {sample_row['rr_score_v2']:.0f}, Band: {sample_row['rr_band']})")
print(f"Risk Meter: {sample_row['risk_meter_v2']:.0f} ({sample_row['risk_label_v2']})")
