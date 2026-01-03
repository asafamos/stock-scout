import os
import sys
import glob
import pandas as pd


def _find_latest_file(paths):
    candidates = []
    exts = (".parquet", ".csv")
    patterns = ("scan_*", "snapshot_*", "latest_scan*", "latest_scan_live*")
    for base in paths:
        if not os.path.isdir(base):
            continue
        for pat in patterns:
            for ext in exts:
                candidates.extend(glob.glob(os.path.join(base, f"{pat}{ext}")))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_latest_df():
    try:
        from core.scan_io import load_latest_scan  # type: ignore
        df, meta = load_latest_scan()
        return df, "scan_io", meta
    except Exception:
        pass
    paths = ["data/scans", "."]
    latest = _find_latest_file(paths)
    if latest is None:
        raise RuntimeError("No scan/snapshot files found in data/scans or workspace root")
    if latest.endswith(".parquet"):
        df = pd.read_parquet(latest)
    else:
        df = pd.read_csv(latest)
    return df, "fallback", {"path": latest}

def _load_specific(path: str):
    if os.path.exists(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    return None


def _range_issues(series, lo, hi):
    bad = series.isna() | (series < lo) | (series > hi)
    return int(bad.sum())


def validate(df: pd.DataFrame) -> dict:
    issues = {
        "critical": [],
        "warnings": [],
        "summary": {}
    }

    cols = set(df.columns)

    if "FinalScore_20d" in cols:
        issues["summary"]["FinalScore_20d_nonnull"] = int(df["FinalScore_20d"].notna().sum())
        bad_range = _range_issues(df["FinalScore_20d"], 0.0, 100.0)
        if bad_range:
            issues["critical"].append(f"FinalScore_20d out-of-range/null count: {bad_range}")
        if "Score" in cols:
            mism = df[(df["Score"].round(6) != df["FinalScore_20d"].round(6))]
            if not mism.empty:
                issues["critical"].append(f"Score != FinalScore_20d rows: {len(mism)}")
    else:
        issues["critical"].append("Missing FinalScore_20d column")

    if "ConvictionScore" in cols:
        bad_conv = _range_issues(df["ConvictionScore"], 0.0, 100.0)
        if bad_conv:
            issues["warnings"].append(f"ConvictionScore out-of-range/null count: {bad_conv}")
    else:
        issues["warnings"].append("Missing ConvictionScore column")

    ml_col = "ML_20d_Prob" if "ML_20d_Prob" in cols else None
    if ml_col:
        s = df[ml_col].dropna()
        gt1 = int((s > 1.0).sum())
        lt0 = int((s < 0.0).sum())
        if lt0:
            issues["warnings"].append(f"ML_20d_Prob < 0 count: {lt0}")
        if gt1:
            issues["warnings"].append(f"ML_20d_Prob > 1 count: {gt1} (likely percent scale)")
    else:
        issues["warnings"].append("Missing ML_20d_Prob column")

    rel_col = None
    for c in ("Reliability_v2", "reliability_v2", "reliability_pct"):
        if c in cols:
            rel_col = c
            break
    if rel_col:
        bad_rel = _range_issues(df[rel_col].astype(float), 0.0, 100.0)
        if bad_rel:
            issues["warnings"].append(f"Reliability ({rel_col}) out-of-range/null count: {bad_rel}")
        if "reliability_band" in cols:
            missing_band = int(df["reliability_band"].isna().sum())
            if missing_band:
                issues["warnings"].append(f"Missing reliability_band rows: {missing_band}")
        else:
            issues["warnings"].append("Missing reliability_band column")
    else:
        issues["warnings"].append("Missing reliability column (v2/pct)")

    if "risk_gate_status_v2" in cols and "risk_gate_penalty_v2" in cols:
        bad_pen = _range_issues(df["risk_gate_penalty_v2"].astype(float), 0.0, 1.0)
        if bad_pen:
            issues["critical"].append(f"risk_gate_penalty_v2 out-of-range/null count: {bad_pen}")
        if "buy_amount_v2" in cols:
            blocked = df[df["risk_gate_status_v2"] == "blocked"]
            bad_block = int((blocked["buy_amount_v2"].astype(float) > 0.0).sum())
            if bad_block:
                issues["warnings"].append(f"Blocked rows with positive buy_amount_v2: {bad_block}")
            neg_buy = int((df["buy_amount_v2"].astype(float) < 0.0).sum())
            if neg_buy:
                issues["critical"].append(f"Negative buy_amount_v2 rows: {neg_buy}")
        else:
            issues["warnings"].append("Missing buy_amount_v2 column")
    else:
        issues["warnings"].append("Missing risk_gate_status_v2 or risk_gate_penalty_v2 columns")

    if "FinalScore" in cols:
        mism_final = None
        if "FinalScore_20d" in cols:
            mism_final = df[(df["FinalScore"].round(6) != df["FinalScore_20d"].round(6))]
        issues["warnings"].append(
            f"Legacy FinalScore present; mismatch with FinalScore_20d: {0 if mism_final is None else len(mism_final)}"
        )

    return issues


def main():
    # Prefer comparing auto vs live snapshots if available
    scans_dir = os.path.join("data", "scans")
    auto_path = os.path.join(scans_dir, "latest_scan.parquet")
    live_path = os.path.join(scans_dir, "latest_scan_live.parquet")

    auto_df = _load_specific(auto_path)
    live_df = _load_specific(live_path)

    printed_any = False
    exit_code = 0
    if auto_df is not None:
        issues = validate(auto_df)
        print("=== Auto Scan (latest_scan.parquet) ===")
        print(f"Rows: {len(auto_df)}")
        print("Critical:")
        for msg in issues["critical"]:
            print(f"- {msg}")
        print("Warnings:")
        for msg in issues["warnings"]:
            print(f"- {msg}")
        printed_any = True
        if issues["critical"]:
            exit_code = 1

    if live_df is not None:
        issues = validate(live_df)
        print("=== Live Scan (latest_scan_live.parquet) ===")
        print(f"Rows: {len(live_df)}")
        print("Critical:")
        for msg in issues["critical"]:
            print(f"- {msg}")
        print("Warnings:")
        for msg in issues["warnings"]:
            print(f"- {msg}")
        printed_any = True
        if issues["critical"]:
            exit_code = 1

    if not printed_any:
        # Fallback to generic latest
        try:
            df, source, meta = _load_latest_df()
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(2)
        issues = validate(df)
        print("=== Score Consistency Report (fallback latest) ===")
        print(f"Rows: {len(df)}")
        print("Critical:")
        for msg in issues["critical"]:
            print(f"- {msg}")
        print("Warnings:")
        for msg in issues["warnings"]:
            print(f"- {msg}")
        if issues["critical"]:
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
