"""
Lightweight performance regression test for the pipeline.

This test verifies that basic pipeline operations (DataFrame manipulation,
simple calculations) remain fast to catch accidental performance regressions.

Run with: pytest tests/test_pipeline_timing_smoke.py -v
"""
import time
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineTimingSmoke:
    """Smoke tests for pipeline performance (non-strict thresholds)."""
    
    def test_dataframe_operations_fast(self):
        """Test that common DataFrame operations remain fast."""
        # Create a small results DataFrame
        df = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "Close": [150.0, 370.0, 140.0, 165.0, 250.0],
            "Volume": [50000000, 40000000, 35000000, 30000000, 25000000],
        })
        
        # Common operations
        t0 = time.perf_counter()
        
        # Filter top by volume
        top_k = df.nlargest(3, "Volume")
        
        # Add new columns
        df["Rank"] = df["Close"].rank()
        df["Log_Volume"] = np.log10(df["Volume"])
        
        # Iterate and compute
        for idx, row in top_k.iterrows():
            _ = row["Close"] * row["Volume"]
        
        elapsed = time.perf_counter() - t0
        
        # Should be very fast (< 50ms)
        assert elapsed < 0.05, f"DataFrame ops too slow: {elapsed:.3f}s"
        
        # Verify output
        assert len(top_k) == 3
        assert "Rank" in df.columns
    
    def test_core_pipeline_scoring_loop(self):
        """Test core scoring loop remains fast."""
        # Build a small results DataFrame like in the main pipeline
        results = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "Close": [150.0, 370.0, 140.0, 165.0],
            "Quality_Score_F": [75.0, 68.0, 72.0, 70.0],
            "Growth_Score_F": [65.0, 72.0, 68.0, 70.0],
            "Valuation_Score_F": [80.0, 70.0, 75.0, 68.0],
            "Stability_Score_F": [70.0, 75.0, 72.0, 71.0],
            "Leverage_Score_F": [60.0, 65.0, 62.0, 64.0],
            "PE": [25.0, 30.0, 28.0, 27.0],
            "PS": [5.0, 9.0, 7.0, 8.0],
            "Beta": [1.2, 0.8, 1.0, 1.1],
            "Fund_Coverage_Pct": [0.8, 0.7, 0.75, 0.72],
            "Fund_Sources_Count": [3, 2, 3, 2],
        })
        
        t0 = time.perf_counter()
        
        # Simulate scoring pipeline (like calculate_reliability_v2 loop)
        for idx, row in results.iterrows():
            # Compute fundamentals score
            q_score = row.get("Quality_Score_F", 50)
            g_score = row.get("Growth_Score_F", 50)
            v_score = row.get("Valuation_Score_F", 50)
            s_score = row.get("Stability_Score_F", 50)
            l_score = row.get("Leverage_Score_F", 50)
            
            fund_score = (q_score * 0.30 + g_score * 0.25 + v_score * 0.25 +
                         s_score * 0.10 + l_score * 0.10)
            
            # Compute reliability factors
            coverage = row.get("Fund_Coverage_Pct", 0.0)
            sources = int(row.get("Fund_Sources_Count", 0))
            reliability_factor = coverage * (0.5 + 0.5 * min(sources / 4.0, 1.0))
        
        elapsed = time.perf_counter() - t0
        
        # Should be < 10ms for 4 tickers
        assert elapsed < 0.01, f"Scoring loop too slow: {elapsed:.3f}s for 4 tickers"
    
    def test_multi_iteration_performance(self):
        """Test that multiple passes over DataFrame remain performant."""
        # Large-ish DataFrame to test iteration performance
        n = 50
        df = pd.DataFrame({
            "Ticker": [f"TK{i}" for i in range(n)],
            "Close": np.random.uniform(10, 500, n),
            "Volume": np.random.uniform(1000000, 100000000, n),
        })
        
        t0 = time.perf_counter()
        
        # Pass 1: Initialize columns
        df["Score_1"] = 0.0
        df["Score_2"] = 0.0
        df["Score_3"] = 0.0
        
        # Pass 2: Fill columns with loop iterations (like reliability computation)
        for idx, row in df.iterrows():
            df.at[idx, "Score_1"] = row["Close"] * 0.5
            df.at[idx, "Score_2"] = row["Volume"] / 1000000.0
            df.at[idx, "Score_3"] = (row["Close"] + row["Volume"] / 1000000.0) / 2.0
        
        # Pass 3: Compute derived columns
        df["Final_Score"] = (df["Score_1"] + df["Score_2"] + df["Score_3"]) / 3.0
        
        elapsed = time.perf_counter() - t0
        
        # Should be < 100ms for 50 tickers through 3 passes
        assert elapsed < 0.1, f"Multi-iteration too slow: {elapsed:.3f}s for 50 tickers"
        
        # Verify
        assert len(df) == 50
        assert df["Final_Score"].notna().all()
    
    def test_historical_stddev_computation_limited_subset(self):
        """Test that computing historical StdDev only for top-K is fast."""
        # Simulate data_map (historical data per ticker)
        data_map = {}
        for i in range(100):
            ticker = f"TK{i}"
            # Mock historical prices (30 days)
            data_map[ticker] = pd.DataFrame({
                "Close": np.random.uniform(80, 120, 30)
            })
        
        # Build results DataFrame
        results = pd.DataFrame({
            "Ticker": [f"TK{i}" for i in range(100)],
            "Close": np.random.uniform(10, 500, 100),
        })
        results["Historical_StdDev"] = np.nan
        
        t0 = time.perf_counter()
        
        # Only compute for top-10 by volume (like price verification step)
        top_k_indices = list(range(10))
        
        for i in top_k_indices:
            ticker = results.loc[i, "Ticker"]
            if ticker in data_map:
                hist = data_map[ticker]
                if len(hist) >= 5:
                    recent = hist["Close"].tail(min(30, len(hist)))
                    if len(recent) >= 5:
                        results.at[i, "Historical_StdDev"] = float(recent.std())
        
        elapsed = time.perf_counter() - t0
        
        # Should be < 10ms for 10 tickers' historical analysis
        assert elapsed < 0.01, f"Historical StdDev computation too slow: {elapsed:.3f}s for 10 tickers"
        
        # Verify
        assert results["Historical_StdDev"].notna().sum() >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
