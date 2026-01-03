from scripts.train_rolling_ml_20d import train_and_save_bundle

if __name__ == "__main__":
    try:
        path, bundle = train_and_save_bundle()
        print("✓ Trained and saved:", path)
        print("Metrics:", bundle.get("metrics", {}))
    except Exception as e:
        print("✗ Training skipped/failed:", e)
