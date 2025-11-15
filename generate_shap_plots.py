"""Generate SHAP summary plots for XGBoost model interpretation."""

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Load model and data
with open('model_xgboost_5d.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['feature_names']

# Load backtest signals
df = pd.read_csv('backtest_signals_20251115_094015.csv')
df.columns = df.columns.str.strip()

# Build feature matrix
from train_recommender import build_dataset
ds = build_dataset(df, horizon=5, target='pos')
X = ds.drop(columns=['y', 'Ticker', 'Date'])

# Compute SHAP values
print("Computing SHAP values for all samples...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Save full SHAP values
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df.to_csv('shap_values_full.csv', index=False)
print(f"✓ Saved SHAP values for {len(X)} samples")

# Summary plot (feature importance by impact)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
plt.title("SHAP Feature Impact on 5-Day Return Prediction", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
print("✓ Saved shap_summary.png")

# Bar plot (mean absolute SHAP values)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
plt.title("Mean SHAP Value (Feature Importance)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
print("✓ Saved shap_importance.png")

print("\n📊 SHAP Analysis Complete")
print("Files: shap_values_full.csv, shap_summary.png, shap_importance.png")
