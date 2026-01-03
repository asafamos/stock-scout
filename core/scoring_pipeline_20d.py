import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Iterable
from core.unified_logic import build_technical_indicators, compute_tech_score_20d_v2
from core.ml_20d_inference import (
    ML_20D_AVAILABLE,
    compute_ml_20d_probabilities_raw,
    apply_live_v3_adjustments,
    PREFERRED_SCORING_MODE_20D,
)


def _canonical_ml_prob(df: pd.DataFrame) -> pd.Series:
    """Pick the best available ML probability column and expose it as ML_20d_Prob."""
    if "ML_20d_Prob_live_v3" in df.columns:
        base = df["ML_20d_Prob_live_v3"].astype(float)
    elif "ML_20d_Prob" in df.columns:
        base = df["ML_20d_Prob"].astype(float)
    elif "ML_20d_Prob_raw" in df.columns:
        base = df["ML_20d_Prob_raw"].astype(float)
    else:
        base = pd.Series(0.5, index=df.index, dtype=float)
    base = base.clip(0.0, 1.0)
    return base


def _canonical_tech(df: pd.DataFrame) -> pd.Series:
    """Pick the best available technical score base column."""
    if "TechScore_20d_v2" in df.columns:
        base = df["TechScore_20d_v2"].astype(float)
    elif "TechScore_20d" in df.columns:
        base = df["TechScore_20d"].astype(float)
    elif "TechScore_20d_v2_raw" in df.columns:
        base = df["TechScore_20d_v2_raw"].astype(float) * 100.0
    else:
        base = pd.Series(50.0, index=df.index, dtype=float)
    return base.fillna(50.0)


def compute_final_scores_20d(df: pd.DataFrame, include_ml: bool = True) -> pd.DataFrame:
    """
    Compute canonical ML_20d_Prob, TechScore_20d (ranked 0-100), and FinalScore_20d (50/50 rank blend).
    - tech rank: percent-rank of technical base
    - ml rank: percent-rank of ML prob (or neutral 0.5 when ML disabled)
    - FinalScore_20d = 0.5*tech_rank + 0.5*ml_rank, scaled to 0-100
    """
    out = df.copy()

    # Canonical ML probability
    ml_prob = _canonical_ml_prob(out) if include_ml else pd.Series(0.5, index=out.index, dtype=float)
    out["ML_20d_Prob"] = ml_prob

    # Technical base and rank
    tech_base = _canonical_tech(out)
    tech_rank = tech_base.rank(pct=True, method="average")
    out["TechScore_20d"] = tech_rank * 100.0

    # ML rank (handle NaNs)
    ml_rank = ml_prob.fillna(0.5).rank(pct=True, method="average") if include_ml else pd.Series(0.5, index=out.index, dtype=float)

    # Final score 50/50 blend
    out["FinalScore_20d"] = (0.5 * tech_rank + 0.5 * ml_rank) * 100.0
    
    # Create alias for backward compatibility with UI
    out["FinalScore"] = out["FinalScore_20d"]

    return out


def apply_20d_sorting(df: pd.DataFrame, policy: Optional[str] = None) -> pd.DataFrame:
    """Sort the scored universe according to the policy with specified tie-breakers."""
    mode = (policy or PREFERRED_SCORING_MODE_20D or "hybrid").lower()
    cols = df.columns
    if mode == "ml_only":
        key1 = "ML_20d_Prob" if "ML_20d_Prob" in cols else cols[0]
        return df.sort_values([key1, "FinalScore_20d", "Ticker"], ascending=[False, False, True]).reset_index(drop=True)
    if mode in {"rank_blend", "hybrid", "hybrid_overlay"}:
        key_ml = "ML_20d_Prob" if "ML_20d_Prob" in cols else cols[0]
        key_fs = "FinalScore_20d" if "FinalScore_20d" in cols else key_ml
        return df.sort_values([key_fs, key_ml, "Ticker"], ascending=[False, False, True]).reset_index(drop=True)
    if mode == "tech_only":
        key_tech = "TechScore_20d" if "TechScore_20d" in cols else cols[0]
        key_fs = "FinalScore_20d" if "FinalScore_20d" in cols else key_tech
        return df.sort_values([key_tech, key_fs, "Ticker"], ascending=[False, False, True]).reset_index(drop=True)
    # default fallback
    return df.sort_values(["FinalScore_20d", "Ticker"], ascending=[False, True]).reset_index(drop=True)


def score_universe_20d(
    as_of_date,
    horizon_days: int,
    universe: Iterable[str],
    price_data: Dict[str, pd.DataFrame],
    include_ml: bool = True,
    logger: Any = None,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build features, compute ML prob (live_v3 if available), compute Tech rank + FinalScore_20d (50/50 rank blend),
    and return a scored DataFrame (unsorted). Filters are expected to be applied by the caller when needed.
    """
    rows = []
    log = logger.info if logger else (lambda *_: None)
    warn = logger.warning if logger else (lambda *_: None)

    for tkr in universe:
        df = price_data.get(tkr)
        if df is None or df.empty:
            continue
        df = df.sort_index()
        idx_pos = df.index.searchsorted(as_of_date, side="right") - 1
        if idx_pos < 0:
            continue
        df_hist = df.iloc[: idx_pos + 1]
        if len(df_hist) < 60:
            continue

        indicators = build_technical_indicators(df_hist)
        # Enrich with multi-period returns for ML feature alignment
        try:
            from core.ml_features_v3 import compute_multi_period_returns, compute_breakout_features
            enriched = compute_multi_period_returns(df_hist)
            enriched2 = compute_breakout_features(df_hist)
            # Attach latest returns into the indicator row so ML can consume
            row = indicators.iloc[-1].copy()
            for c in ["Return_5d","Return_10d","Return_20d","Return_60d","Return_120d"]:
                if c in enriched.columns:
                    row[c] = float(enriched[c].iloc[-1])
            for c in [
                "BB_Width","KC_Width","Squeeze_On_Flag",
                "Vol_Contraction_Ratio","Volume_Relative_20d",
                "MA_Slope_S","MA_Slope_L"
            ]:
                if c in enriched2.columns:
                    val = enriched2[c].iloc[-1]
                    row[c] = float(val) if pd.notna(val) else np.nan
        except Exception:
            row = indicators.iloc[-1]

        # Base row
        rec = {
            "Ticker": tkr,
            "As_Of_Date": df_hist.index[-1],
            "Price_As_Of_Date": float(df_hist["Close"].iloc[-1]),
            "RSI": float(row.get("RSI", np.nan)),
            "ATR_Pct": float(row.get("ATR_Pct", np.nan)),
            "RR": float(row.get("RR", np.nan)),
            "MomCons": float(row.get("MomCons", np.nan)),
            "VolSurge": float(row.get("VolSurge", np.nan)),
        }

        # Big winner features
        from core.unified_logic import compute_big_winner_signal_20d

        bw = compute_big_winner_signal_20d(row)
        rec["BigWinnerScore_20d"] = float(bw.get("BigWinnerScore_20d", 0.0))
        rec["BigWinnerFlag_20d"] = int(bw.get("BigWinnerFlag_20d", 0))

        # Technical score (raw)
        try:
            tech_raw = compute_tech_score_20d_v2(row)  # 0-1
        except Exception:
            tech_raw = 0.5
        rec["TechScore_20d_v2_raw"] = float(tech_raw)

        # ML probability
        if include_ml and ML_20D_AVAILABLE:
            try:
                ml_prob_raw = compute_ml_20d_probabilities_raw(row)
            except Exception:
                ml_prob_raw = np.nan
        else:
            ml_prob_raw = np.nan
        rec["ML_20d_Prob_raw"] = float(ml_prob_raw) if np.isfinite(ml_prob_raw) else np.nan

        # Horizon returns (if future data available)
        try:
            idx_fwd = idx_pos + horizon_days
            if idx_fwd < len(df):
                price_0 = float(df["Close"].iloc[idx_pos])
                price_h = float(df["Close"].iloc[idx_fwd])
                rec[f"Return_{horizon_days}d"] = float(price_h / price_0 - 1.0)
            else:
                rec[f"Return_{horizon_days}d"] = np.nan
        except Exception:
            rec[f"Return_{horizon_days}d"] = np.nan

        # Relative strength vs benchmark (SPY) if provided
        if benchmark_df is not None and "date" in benchmark_df.columns:
            try:
                as_of = pd.to_datetime(rec["As_Of_Date"]) if "As_Of_Date" in rec else None
                if as_of is not None:
                    spy_row = benchmark_df[benchmark_df["date"] == as_of]
                    if not spy_row.empty:
                        rec["SPY_20d_ret"] = float(spy_row.iloc[0].get("SPY_20d_ret", np.nan))
                        # If model expects relative features, ensure they exist
                        if pd.notna(rec.get("Return_20d", np.nan)) and pd.notna(rec.get("SPY_20d_ret", np.nan)):
                            rec["RS_SPY_20d"] = float(rec.get("Return_20d") - rec.get("SPY_20d_ret"))
            except Exception:
                pass

        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # ATR percentile for live_v3
    if "ATR_Pct" in out.columns:
        out["ATR_Pct_percentile"] = out["ATR_Pct"].rank(pct=True, method="average")
    else:
        out["ATR_Pct_percentile"] = 0.5

    # Apply live_v3 adjustments if ML present
    if include_ml and ML_20D_AVAILABLE and "ML_20d_Prob_raw" in out.columns:
        out["ML_20d_Prob_live_v3"] = apply_live_v3_adjustments(out, prob_col="ML_20d_Prob_raw")
        out["ML_20d_Prob"] = out["ML_20d_Prob_live_v3"]
    else:
        out["ML_20d_Prob"] = out["ML_20d_Prob_raw"].fillna(0.5) if "ML_20d_Prob_raw" in out.columns else 0.5

    # Compute FinalScore_20d (rank blend)
    out = compute_final_scores_20d(out, include_ml=include_ml and ML_20D_AVAILABLE)

    return out
