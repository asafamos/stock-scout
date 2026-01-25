import pandas as pd
from core import pipeline_runner as pr
out = pr.run_scan_smoke()
res = out.get("result", {})
df = res.get("results_df", pd.DataFrame())
print("Rows:", len(df))
print("Cols:", list(df.columns))
if len(df) > 0:
    row = df.head(1).to_dict(orient="records")[0]
    print("SignalReasons:", row.get("SignalReasons"))
    print("SignalQuality:", row.get("SignalQuality"))
    print("Pattern_Score:", row.get("Pattern_Score"))
