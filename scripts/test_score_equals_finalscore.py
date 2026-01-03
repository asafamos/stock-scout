import sys
import pandas as pd

from scripts.validate_score_consistency import _load_latest_df


def main():
    df, source, meta = _load_latest_df()
    if "FinalScore_20d" not in df.columns or "Score" not in df.columns:
        print("ERROR: Missing FinalScore_20d or Score columns")
        sys.exit(2)
    mism = df[(pd.to_numeric(df["Score"], errors="coerce").round(6) != pd.to_numeric(df["FinalScore_20d"], errors="coerce").round(6))]
    if mism.empty:
        print("OK: Score equals FinalScore_20d for all rows")
        sys.exit(0)
    else:
        print(f"FAIL: {len(mism)} rows where Score != FinalScore_20d")
        # Show top 5 mismatches
        print(mism[["Ticker","Score","FinalScore_20d"]].head().to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
