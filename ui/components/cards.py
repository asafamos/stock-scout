import numpy as np
from core.scoring_config import get_canonical_score

def build_clean_card(row, speculative=False):
    """
    Build a clean HTML card for a stock recommendation.
    Header: Ticker, Badge, Overall Score only
    Top 6 fields: Target, RR, Risk, Reliability, ML, Quality
    Rest in <details> collapsible section
    No emojis except ⚠️ for warnings
    Tabular numbers, consistent formatting
    """
    # ...existing code...
    # (Paste the full function body from stock_scout.py lines 210-570 here)
    # ...existing code...
    return card_html
