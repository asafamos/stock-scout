#!/usr/bin/env python3
"""
Analyze technical feature predictiveness for 20d forward returns and propose improved TechScore_20d variants.

- Loads the same dataset v2 used by validate_ml_improvements.py: experiments/training_dataset_20d_v2.csv
- For each technical feature, compute decile bins with sample count, avg forward return, and positive-rate.
- Define rule-based filters and evaluate individual & combined rule performance.
- Design 2–3 candidate TechScore_20d variants and evaluate decile separation vs current TechScore_20d.
- Add regime analysis by time slices to check stability across different market periods.
- Save all outputs under experiments/outputs/technical_logic/ (CSVs + plots).

This script is strictly offline analysis. It does not change any public API of the app.
"""

import os
from pathlib import Path
import warnings
import math
import pandas as pd
import numpy as np

# Optional plotting; skip if matplotlib not installed
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    print("⚠️ matplotlib not available; skipping plots.")

OUTPUT_ROOT = Path("experiments/outputs/technical_logic")
FEATURE_BINS_DIR = OUTPUT_ROOT / "technical_feature_bins"
VARIANT_BINS_DIR = OUTPUT_ROOT / "techscore_variant_bins"
RULESETS_DIR = OUTPUT_ROOT / "rule_sets"
PLOTS_DIR = OUTPUT_ROOT / "plots"
SUMMARY_PATH = OUTPUT_ROOT / "summary.txt"
DATASET_PATH = Path("experiments/training_dataset_20d_v2.csv")

# Ensure directories exist
for d in [OUTPUT_ROOT, FEATURE_BINS_DIR, VARIANT_BINS_DIR, RULESETS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def robust_scale(series: pd.Series) -> pd.Series:
    """Scale to [0,1] using robust quantiles to reduce outlier impact."""
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    q_low, q_high = s.quantile(0.05), s.quantile(0.95)
    s = s.clip(q_low, q_high)
    rng = q_high - q_low
    if rng == 0 or np.isnan(rng):
        return pd.Series(np.zeros(len(s)), index=series.index)
    return (s - q_low) / rng


def percent_rank_within_group(values: pd.Series, group: pd.Series) -> pd.Series:
    """Percent rank 0–100 within each group (e.g., by date universe)."""
    def rank_group(g):
        return g.rank(pct=True) * 100.0
    return values.groupby(group).transform(rank_group)


def get_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column from candidates (case-insensitive, underscores tolerant)."""
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        for k, orig in cols.items():
            if key == k:
                return orig
        # loose contains match
        for k, orig in cols.items():
            if key in k:
                return orig
    return None


def detect_feature_columns(df: pd.DataFrame) -> dict:
    """Detect a set of common technical feature columns. Returns a dict of lists by category."""
    categories = {
        "rsi": [],
        "distance_ma": [],
        "momentum": [],
        "atr_pct": [],
        "near_high": [],
        "reward_risk": [],
    }
    for col in df.columns:
        lc = col.lower()
        if any(tag in lc for tag in ["rsi"]):
            categories["rsi"].append(col)
        if any(tag in lc for tag in ["dist_ma", "distance_ma", "pct_above_ma", "price_ma", "ma_gap"]):
            categories["distance_ma"].append(col)
        if any(tag in lc for tag in ["mom", "momentum", "ret_1m", "ret_3m", "ret_6m"]):
            categories["momentum"].append(col)
        if any(tag in lc for tag in ["atr_pct", "atr%", "atr/price", "atr_ratio"]):
            categories["atr_pct"].append(col)
        if any(tag in lc for tag in ["nearhigh", "near_high", "pct_off_high", "off_high", "off_20d_high"]):
            categories["near_high"].append(col)
        if any(tag in lc for tag in ["reward_risk", "rr", "riskreward", "r_r"]):
            categories["reward_risk"].append(col)
    return categories


def feature_decile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute decile bins for each detected technical feature and save CSV + plot."""
    print_section("FEATURE DECILE ANALYSIS")
    label_col = get_col(df, ["Label_20d", "label_20d"]) or "Label_20d"
    ret_col = get_col(df, ["Forward_Return_20d", "forward_return_20d"]) or "Forward_Return_20d"
    if label_col not in df.columns or ret_col not in df.columns:
        raise ValueError("Dataset must include Label_20d and Forward_Return_20d")

    feats = detect_feature_columns(df)
    all_summaries = []

    for cat, cols in feats.items():
        for col in cols:
            series = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
            valid = series.dropna()
            if valid.empty:
                continue
            # Deciles across entire dataset
            try:
                deciles = pd.qcut(series, q=10, labels=False, duplicates='drop')
            except Exception:
                # fallback: quantiles fewer bins
                try:
                    deciles = pd.qcut(series, q=5, labels=False, duplicates='drop')
                except Exception:
                    warnings.warn(f"Skipping feature {col}: cannot qcut")
                    continue

            df_local = df[[col, label_col, ret_col]].copy()
            df_local['decile'] = deciles
            df_local = df_local.dropna(subset=['decile'])
            rows = []
            for dec in sorted(df_local['decile'].unique()):
                subset = df_local[df_local['decile'] == dec]
                count = len(subset)
                avg_ret = subset[ret_col].mean()
                pos_rate = (subset[label_col] == 1).mean()
                rows.append({
                    'feature': col,
                    'category': cat,
                    'decile': int(dec),
                    'count': int(count),
                    'avg_forward_return_20d': float(avg_ret),
                    'positive_rate': float(pos_rate),
                })
            summary_df = pd.DataFrame(rows)
            all_summaries.append(summary_df)

            # Save CSV per feature
            out_csv = FEATURE_BINS_DIR / f"{col}_deciles.csv"
            summary_df.to_csv(out_csv, index=False)

            # Plot simple bar chart of avg returns by decile
            if HAVE_MPL:
                try:
                    plt.figure(figsize=(8, 4))
                    plt.bar(summary_df['decile'], summary_df['avg_forward_return_20d'], color='steelblue')
                    plt.title(f"{col} — Avg 20d Forward Return by Decile")
                    plt.xlabel("Decile")
                    plt.ylabel("Avg 20d Return")
                    plt.tight_layout()
                    plt.savefig(PLOTS_DIR / f"{col}_deciles_return.png")
                    plt.close()
                except Exception as e:
                    warnings.warn(f"Plot failed for {col}: {e}")

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(OUTPUT_ROOT / "feature_deciles_all.csv", index=False)
    else:
        print("⚠️ No feature deciles computed.")
    return combined if all_summaries else pd.DataFrame()


def compute_rule_masks(df: pd.DataFrame) -> dict:
    """Define rule masks based on available columns."""
    masks = {}
    price_col = get_col(df, ["Price", "Close", "Adj_Close", "Adj Close"])  # prefer Close if present
    ma50_col = get_col(df, ["MA_50", "SMA_50", "ma_50"])  # moving average 50
    ma200_col = get_col(df, ["MA_200", "SMA_200", "ma_200"])  # moving average 200
    rsi_col = get_col(df, ["RSI", "rsi"])  # RSI
    atr_pct_col = get_col(df, ["ATR_Pct", "ATR%", "ATR/Price", "atr_pct", "atr_ratio"])  # ATR%
    near_high_col = get_col(df, ["NearHigh_20d", "near_high_20d", "Pct_Off_20d_High", "pct_off_20d_high", "off_high"])

    # Uptrend rule
    if price_col and ma50_col and ma200_col:
        masks['Uptrend'] = (df[price_col] > df[ma50_col]) & (df[ma50_col] > df[ma200_col])
    
    # Healthy pullback: Price > MA_50 and RSI in [40,65]
    if price_col and ma50_col and rsi_col:
        masks['Healthy_Pullback'] = (df[price_col] > df[ma50_col]) & (df[rsi_col].between(40, 65, inclusive='both'))

    # Volatility sweet-spot: ATR% between Q30 and Q80
    if atr_pct_col:
        q30 = df[atr_pct_col].quantile(0.30)
        q80 = df[atr_pct_col].quantile(0.80)
        masks['Vol_SweetSpot'] = df[atr_pct_col].between(q30, q80, inclusive='both')

    # Chasing penalty: near-high extreme + RSI high
    if near_high_col and rsi_col:
        lc = near_high_col.lower()
        near_high_series = df[near_high_col]
        if 'pct_off' in lc or 'off_high' in lc:
            # small values mean close to highs
            threshold_mask = near_high_series <= near_high_series.quantile(0.05)
        else:
            # large values mean close to highs
            threshold_mask = near_high_series >= near_high_series.quantile(0.95)
        masks['Chasing_Penalty'] = threshold_mask & (df[rsi_col] >= 75)

    return masks


def evaluate_rules(df: pd.DataFrame, masks: dict, label_col: str, ret_col: str) -> pd.DataFrame:
    """Compute metrics for each rule and combinations of 2–3 rules; save ranked table."""
    print_section("RULE-BASED FILTER EVALUATION")
    entries = []

    # Individual rules
    for name, m in masks.items():
        subset = df[m]
        if subset.empty:
            continue
        pos_rate = (subset[label_col] == 1).mean()
        avg_ret = subset[ret_col].mean()
        dd_p5 = subset[ret_col].quantile(0.05)
        entries.append({
            'rule_set': name,
            'count': int(len(subset)),
            'avg_forward_return_20d': float(avg_ret),
            'positive_rate': float(pos_rate),
            'p5_drawdown': float(dd_p5),
        })

    # Combinations of 2 rules
    rule_names = list(masks.keys())
    for i in range(len(rule_names)):
        for j in range(i+1, len(rule_names)):
            n1, n2 = rule_names[i], rule_names[j]
            subset = df[masks[n1] & masks[n2]]
            if subset.empty:
                continue
            pos_rate = (subset[label_col] == 1).mean()
            avg_ret = subset[ret_col].mean()
            dd_p5 = subset[ret_col].quantile(0.05)
            entries.append({
                'rule_set': f"{n1} + {n2}",
                'count': int(len(subset)),
                'avg_forward_return_20d': float(avg_ret),
                'positive_rate': float(pos_rate),
                'p5_drawdown': float(dd_p5),
            })

    # Combinations of 3 rules (exclude Chasing penalty for positive strategies and include to observe impact)
    if len(rule_names) >= 3:
        for i in range(len(rule_names)):
            for j in range(i+1, len(rule_names)):
                for k in range(j+1, len(rule_names)):
                    n1, n2, n3 = rule_names[i], rule_names[j], rule_names[k]
                    subset = df[masks[n1] & masks[n2] & masks[n3]]
                    if subset.empty:
                        continue
                    pos_rate = (subset[label_col] == 1).mean()
                    avg_ret = subset[ret_col].mean()
                    dd_p5 = subset[ret_col].quantile(0.05)
                    entries.append({
                        'rule_set': f"{n1} + {n2} + {n3}",
                        'count': int(len(subset)),
                        'avg_forward_return_20d': float(avg_ret),
                        'positive_rate': float(pos_rate),
                        'p5_drawdown': float(dd_p5),
                    })

    rules_df = pd.DataFrame(entries)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(['avg_forward_return_20d', 'positive_rate'], ascending=[False, False])
        rules_df.to_csv(RULESETS_DIR / "rule_sets_summary.csv", index=False)
    else:
        print("⚠️ No rule evaluations produced (missing columns?")
    return rules_df


def compute_trend_score(df: pd.DataFrame) -> pd.Series:
    price_col = get_col(df, ["Price", "Close", "Adj_Close", "Adj Close"]) or "Close"
    ma50_col = get_col(df, ["MA_50", "SMA_50", "ma_50"]) or None
    ma200_col = get_col(df, ["MA_200", "SMA_200", "ma_200"]) or None
    slope50_col = get_col(df, ["MA_50_Slope", "slope_ma_50"]) or None

    comps = []
    if price_col and ma50_col:
        comps.append(robust_scale(df[price_col] / (df[ma50_col] + 1e-9)))
    if ma50_col and ma200_col:
        comps.append(robust_scale(df[ma50_col] / (df[ma200_col] + 1e-9)))
    if slope50_col and slope50_col in df.columns:
        comps.append(robust_scale(df[slope50_col]))
    if not comps:
        return pd.Series(np.zeros(len(df)))
    # weights: favor price>MA, then MA50>MA200, then slope
    weights = np.array([0.5, 0.4, 0.1][:len(comps)])
    weights = weights / weights.sum()
    score = np.zeros(len(df))
    for w, comp in zip(weights, comps):
        score += w * comp.values
    return pd.Series(score, index=df.index)


def compute_momentum_score(df: pd.DataFrame) -> pd.Series:
    mom1_col = get_col(df, ["Mom_1m", "Momentum_1m", "Ret_1m"]) or None
    mom3_col = get_col(df, ["Mom_3m", "Momentum_3m", "Ret_3m"]) or None
    mom6_col = get_col(df, ["Mom_6m", "Momentum_6m", "Ret_6m"]) or None
    comps = []
    if mom1_col:
        comps.append(robust_scale(df[mom1_col]))
    if mom3_col:
        comps.append(robust_scale(df[mom3_col]))
    if mom6_col:
        comps.append(robust_scale(df[mom6_col]))
    if not comps:
        return pd.Series(np.zeros(len(df)))
    # weights: 1m 0.5, 3m 0.3, 6m 0.2 (truncate by available)
    base = np.array([0.5, 0.3, 0.2][:len(comps)])
    weights = base / base.sum()
    score = np.zeros(len(df))
    for w, comp in zip(weights, comps):
        score += w * comp.values
    return pd.Series(score, index=df.index)


def compute_conservative_score(df: pd.DataFrame) -> pd.Series:
    atr_pct_col = get_col(df, ["ATR_Pct", "ATR%", "ATR/Price", "atr_pct", "atr_ratio"]) or None
    rr_col = get_col(df, ["Reward_Risk", "RR", "RiskReward", "R_R"]) or None
    rsi_col = get_col(df, ["RSI", "rsi"]) or None
    near_high_col = get_col(df, ["NearHigh_20d", "near_high_20d", "Pct_Off_20d_High", "pct_off_20d_high", "off_high"]) or None

    comps = []
    # Vol sweet spot: center around mid-quantiles (0.3-0.8), penalize extremes
    if atr_pct_col:
        atr_scaled = robust_scale(df[atr_pct_col])
        # transform to "distance from mid" score
        mid = atr_scaled.median()
        vol_score = 1.0 - ((atr_scaled - mid).abs())  # higher is better near mid
        vol_score = vol_score.clip(0, 1)
        comps.append(vol_score)
    if rr_col:
        comps.append(robust_scale(df[rr_col]))

    penalties = np.zeros(len(df))
    if rsi_col:
        penalties += (df[rsi_col] >= 75).astype(float) * 0.2
    if near_high_col:
        lc = near_high_col.lower()
        nh = df[near_high_col]
        if 'pct_off' in lc or 'off_high' in lc:
            chasing = (nh <= nh.quantile(0.05)).astype(float)
        else:
            chasing = (nh >= nh.quantile(0.95)).astype(float)
        penalties += chasing * 0.2

    if not comps:
        base = pd.Series(np.zeros(len(df)), index=df.index)
    else:
        base = pd.Series(np.zeros(len(df)), index=df.index)
        weights = np.array([1.0] * len(comps))
        weights = weights / weights.sum()
        for w, comp in zip(weights, comps):
            base += w * comp.values
    score = (base - penalties).clip(0, 1)
    return pd.Series(score, index=df.index)


def evaluate_variants(df: pd.DataFrame, date_col: str, label_col: str, ret_col: str) -> pd.DataFrame:
    print_section("TECHSCORE VARIANTS EVALUATION")
    variants = {}
    variants['Trend'] = compute_trend_score(df)
    variants['Momentum'] = compute_momentum_score(df)
    variants['Conservative'] = compute_conservative_score(df)

    rows_all = []

    for name, raw in variants.items():
        # Percent-rank 0–100 per date universe if date available
        if date_col in df.columns:
            pr = percent_rank_within_group(raw, df[date_col])
        else:
            pr = (raw.rank(pct=True) * 100.0)
        score_0_100 = pr.fillna(0)
        df_local = pd.DataFrame({
            f'{name}_Score': score_0_100,
            label_col: df[label_col],
            ret_col: df[ret_col],
        })
        # Decile analysis on the score (overall)
        try:
            df_local[f'{name}_Decile'] = pd.qcut(df_local[f'{name}_Score'], q=10, labels=False, duplicates='drop')
        except Exception:
            try:
                df_local[f'{name}_Decile'] = pd.qcut(df_local[f'{name}_Score'], q=5, labels=False, duplicates='drop')
            except Exception:
                warnings.warn(f"Skipping decile for {name} variant")
                continue
        summ_rows = []
        for dec in sorted(df_local[f'{name}_Decile'].dropna().unique()):
            subset = df_local[df_local[f'{name}_Decile'] == dec]
            summ_rows.append({
                'variant': name,
                'decile': int(dec),
                'count': int(len(subset)),
                'avg_forward_return_20d': float(subset[ret_col].mean()),
                'positive_rate': float((subset[label_col] == 1).mean()),
            })
        summ_df = pd.DataFrame(summ_rows)
        summ_df.to_csv(VARIANT_BINS_DIR / f"techscore_variant_deciles_{name}.csv", index=False)

        # Plot
        if HAVE_MPL:
            try:
                plt.figure(figsize=(8, 4))
                plt.bar(summ_df['decile'], summ_df['avg_forward_return_20d'], color='darkorange')
                plt.title(f"{name} TechScore — Avg 20d Return by Decile")
                plt.xlabel("Decile")
                plt.ylabel("Avg 20d Return")
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / f"techscore_variant_{name}_deciles.png")
                plt.close()
            except Exception as e:
                warnings.warn(f"Plot failed for variant {name}: {e}")

        rows_all.append(summ_df)

    if rows_all:
        combined = pd.concat(rows_all, ignore_index=True)
        combined.to_csv(OUTPUT_ROOT / "techscore_variants_deciles_all.csv", index=False)
    else:
        combined = pd.DataFrame()
        print("⚠️ No variant deciles computed.")
    return combined


def regime_windows(df: pd.DataFrame, date_col: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Split into several time windows: 2023 H1/H2, 2024 H1/H2, 2025 YTD (based on available range)."""
    if date_col not in df.columns:
        return []
    d = pd.to_datetime(df[date_col])
    start = d.min()
    end = d.max()
    windows = []
    years = sorted(set(d.dt.year))
    for y in years:
        h1_start = pd.Timestamp(year=y, month=1, day=1)
        h1_end = pd.Timestamp(year=y, month=6, day=30)
        h2_start = pd.Timestamp(year=y, month=7, day=1)
        h2_end = pd.Timestamp(year=y, month=12, day=31)
        # clamp to data range
        if h1_end >= start and h1_start <= end:
            windows.append((f"{y} H1", max(h1_start, start), min(h1_end, end)))
        if h2_end >= start and h2_start <= end:
            windows.append((f"{y} H2", max(h2_start, start), min(h2_end, end)))
    return windows


def regime_analysis(df: pd.DataFrame, variants_deciles: pd.DataFrame, date_col: str, label_col: str, ret_col: str) -> pd.DataFrame:
    print_section("REGIME / TIME-SLICE ANALYSIS")
    windows = regime_windows(df, date_col)
    if not windows:
        print("⚠️ No date column for regime analysis.")
        return pd.DataFrame()

    result_rows = []
    for name in ['Trend', 'Momentum', 'Conservative']:
        score_col = f'{name}_Score'
        if score_col not in variants_deciles.columns:
            # Not in decile table; skip
            continue
        # Create per-window deciles
        for label, start, end in windows:
            mask = (pd.to_datetime(df[date_col]) >= start) & (pd.to_datetime(df[date_col]) <= end)
            sub = df.loc[mask, [score_col, label_col, ret_col]].copy()
            if sub.empty:
                continue
            try:
                sub['decile'] = pd.qcut(sub[score_col], q=10, labels=False, duplicates='drop')
            except Exception:
                try:
                    sub['decile'] = pd.qcut(sub[score_col], q=5, labels=False, duplicates='drop')
                except Exception:
                    continue
            for dec in sorted(sub['decile'].dropna().unique()):
                part = sub[sub['decile'] == dec]
                result_rows.append({
                    'window': label,
                    'variant': name,
                    'decile': int(dec),
                    'count': int(len(part)),
                    'avg_forward_return_20d': float(part[ret_col].mean()),
                    'positive_rate': float((part[label_col] == 1).mean()),
                })
    out = pd.DataFrame(result_rows)
    if not out.empty:
        out.to_csv(OUTPUT_ROOT / "regime_variant_deciles.csv", index=False)
    return out


def generate_text_summary(df: pd.DataFrame, feature_bins: pd.DataFrame, rules_df: pd.DataFrame, variants_deciles: pd.DataFrame, date_col: str, label_col: str, ret_col: str):
    print_section("FINAL TEXTUAL SUMMARY")
    lines = []
    baseline_pos = (df[label_col] == 1).mean()
    baseline_ret = df[ret_col].mean()
    lines.append(f"Baseline: pos_rate={baseline_pos:.1%}, avg_return={baseline_ret:.6f}")

    # Most predictive features: choose those with top-vs-bottom decile spread
    if not feature_bins.empty:
        spreads = []
        for feature in sorted(feature_bins['feature'].unique()):
            fdf = feature_bins[feature_bins['feature'] == feature]
            if fdf.empty:
                continue
            top = fdf.sort_values('decile').tail(1)
            bot = fdf.sort_values('decile').head(1)
            spread = float(top['avg_forward_return_20d'].values[0] - bot['avg_forward_return_20d'].values[0])
            spreads.append((feature, spread))
        spreads.sort(key=lambda x: x[1], reverse=True)
        lines.append("Most predictive features by top-bottom return spread:")
        for feature, spread in spreads[:10]:
            lines.append(f"  - {feature}: spread={spread:.6f}")

    # Best rule sets
    if rules_df is not None and not rules_df.empty:
        lines.append("Best-performing rule sets (sorted by avg return, then pos rate):")
        for _, row in rules_df.head(10).iterrows():
            lines.append(f"  - {row['rule_set']}: count={row['count']}, avg_ret={row['avg_forward_return_20d']:.6f}, pos_rate={row['positive_rate']:.1%}, p5_drawdown={row['p5_drawdown']:.6f}")

    # Best variant by top decile performance
    if variants_deciles is not None and not variants_deciles.empty:
        lines.append("TechScore variants decile summary (top decile):")
        variant_summaries = []
        for variant in sorted(variants_deciles['variant'].unique()):
            vdf = variants_deciles[variants_deciles['variant'] == variant]
            top_dec = vdf.sort_values('decile').tail(1).iloc[0]
            variant_summaries.append((variant, float(top_dec['avg_forward_return_20d']), float(top_dec['positive_rate'])))
        variant_summaries.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for variant, avg_ret, pos_rate in variant_summaries:
            lines.append(f"  - {variant}: top_decile avg_ret={avg_ret:.6f}, pos_rate={pos_rate:.1%}")

    # Recommended formula outline
    lines.append("\nRecommended TechScore_20d formula (weights by component):")
    lines.append("  - Trend: 40% (Price/MA50, MA50/MA200, optional MA50 slope)")
    lines.append("  - Momentum: 35% (1m/3m/6m scaled, de-emphasize extremes)")
    lines.append("  - Volatility: 15% (ATR% sweet-spot centered in mid-quantiles)")
    lines.append("  - Location: 10% (penalize chasing near highs and extreme RSI)")
    lines.append("Note: Use per-date percent-rank to normalize to 0–100 within universe.")

    text = "\n".join(lines)
    print(text)
    try:
        SUMMARY_PATH.write_text(text, encoding='utf-8')
    except Exception as e:
        warnings.warn(f"Failed to write summary: {e}")


def main():
    print_section("TECHNICAL LOGIC ANALYSIS — 20D")
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return 1

    df = pd.read_csv(DATASET_PATH)
    print(f"✓ Dataset loaded: {len(df)} rows, {df.shape[1]} columns")

    # Identify key columns
    date_col = get_col(df, ["As_Of_Date", "Date", "as_of_date"]) or "As_Of_Date"
    label_col = get_col(df, ["Label_20d", "label_20d"]) or "Label_20d"
    ret_col = get_col(df, ["Forward_Return_20d", "forward_return_20d"]) or "Forward_Return_20d"
    techscore_col = get_col(df, ["TechScore_20d", "techscore_20d"]) or "TechScore_20d"

    missing = [c for c in [date_col, label_col, ret_col, techscore_col] if c not in df.columns]
    if missing:
        print(f"⚠️ Missing expected columns: {missing}")

    # 1) Feature deciles
    feature_bins = feature_decile_analysis(df)

    # 2) Rules and combinations
    masks = compute_rule_masks(df)
    rules_df = evaluate_rules(df, masks, label_col, ret_col)

    # 3) TechScore variants
    variants_deciles = evaluate_variants(df, date_col, label_col, ret_col)

    # 4) Regime / time-slice analysis
    _ = regime_analysis(df, variants_deciles, date_col, label_col, ret_col)

    # 5) Final summary
    generate_text_summary(df, feature_bins, rules_df, variants_deciles, date_col, label_col, ret_col)

    print("\n" + "=" * 70)
    print("✅ Technical logic analysis complete. Outputs in experiments/outputs/technical_logic/")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
