"""Full-pipeline backtest engine for Stock Scout.

Validates the complete scoring pipeline (technical + fundamental + ML +
pattern + risk) against historical data using walk-forward simulation.
"""

from __future__ import annotations
