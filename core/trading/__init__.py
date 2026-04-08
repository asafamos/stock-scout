"""StockScout Auto-Trading Module — IBKR Integration.

Connects StockScout scan recommendations to Interactive Brokers
for automated order execution with risk management.

Safety defaults:
    DRY_RUN  = True   (log orders, don't send)
    PAPER_MODE = True  (paper trading port)
"""
