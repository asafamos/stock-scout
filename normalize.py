from __future__ import annotations
import pandas as pd
from typing import Dict
from fundamentals import zscore_by_group, _cap_bucket


def normalize_tech(df: pd.DataFrame, sectors: pd.Series, market_caps: pd.Series) -> pd.DataFrame:
    """
    Sector- and market-cap-aware z-score normalization of technical features.
    Returns DataFrame with suffix `_z` for each input metric handled.
    """
    caps = market_caps.apply(_cap_bucket)
    sec = sectors.fillna("Unknown")
    out = pd.DataFrame(index=df.index)

    mapping = {
        "rsi": "rsi_z",
        "mom_1m": "mom_1m_z",
        "mom_3m": "mom_3m_z",
        "mom_6m": "mom_6m_z",
        "near_52w_high": "near_52w_high_z",
    }

    for src, dst in mapping.items():
        if src in df.columns:
            out[dst] = zscore_by_group(df[src], sec, caps)
        else:
            out[dst] = pd.Series(0.0, index=df.index)
    return out