import pandas as pd
import numpy as np
from scoring import allocate_budget

def test_allocator_respects_budget_and_caps():
    idx = ["A", "B", "C", "D"]
    prices = pd.Series([10.0, 20.0, 5.0, 50.0], index=idx)
    scores = pd.Series([1.0, 0.8, 0.5, 0.2], index=idx)
    atr_pct = pd.Series([0.02, 0.03, 0.05, 0.02], index=idx)
    sector = pd.Series(["S1", "S1", "S2", "S2"], index=idx)
    dvol = pd.Series([2_000_000, 2_000_000, 2_000_000, 2_000_000], index=idx)
    budget = 100_000.0
    alloc = allocate_budget(scores, prices, atr_pct, sector, dvol, budget, max_pos_pct=0.5, max_sector_pct=0.7)
    total_cost = alloc["cost"].sum()
    assert total_cost <= budget + 1e-6
    merged = alloc.join(pd.DataFrame({"sector": sector}))
    per_sector = merged.groupby("sector")["cost"].sum()
    assert per_sector.max() <= budget * 0.7 + 1e-6