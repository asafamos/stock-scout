"""
Hyperparameter Tuning for Stock Scout ML using Optuna.

Optimizes model hyperparameters using Precision@20 as the objective.
Uses TimeSeriesSplit cross-validation to prevent data leakage.

Usage:
    python scripts/tune_hyperparams.py --trials 50 --timeout 1800
    python scripts/tune_hyperparams.py --trials 100 --timeout 3600 --universe 1000
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Suppress warnings during optimization
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import from training script
from scripts.train_rolling_ml_20d import (
    fetch_polygon_history,
    get_universe_tickers,
    calculate_features,
    calculate_market_regime,
    fetch_sector_etf_data,
    precision_at_k,
    EnsembleClassifier,
    POLYGON_KEY,
)

# Directories
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def _check_api_key():
    """Verify POLYGON_API_KEY is set."""
    if not POLYGON_KEY:
        print("❌ ERROR: POLYGON_API_KEY environment variable is required.")
        print("   export POLYGON_API_KEY=your_api_key_here")
        sys.exit(1)


# Feature list (must match training script)
FEATURES = [
    'RSI', 'ATR_Pct', 'Return_20d', 'Return_10d', 'Return_5d',
    'VCP_Ratio', 'Tightness_Ratio', 'Dist_From_52w_High', 'MA_Alignment',
    'Volume_Surge', 'Up_Down_Volume_Ratio',
    'Momentum_Consistency', 'RS_vs_SPY_20d',
    'Market_Regime', 'Market_Volatility', 'Market_Trend', 'High_Volatility',
    'Sector_RS', 'Sector_Momentum', 'Sector_Rank',
    'Volume_Ratio_20d', 'Volume_Trend', 'Up_Volume_Ratio',
    'Volume_Price_Confirm', 'Relative_Volume_Rank',
    'Distance_From_52w_Low', 'Consolidation_Tightness', 'Days_Since_52w_High',
    'Price_vs_SMA50', 'Price_vs_SMA200', 'SMA50_vs_SMA200', 'MA_Slope_20d',
    'Distance_To_Resistance', 'Support_Strength',
]


def load_training_data(universe_limit: int = 500, lookback_days: int = 730) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare training data.
    
    Args:
        universe_limit: Max stocks to include
        lookback_days: Days of historical data
    
    Returns:
        (X, y) tuple of features and labels
    """
    print("📥 Loading training data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Fetch SPY benchmark
    print("   Fetching SPY benchmark...")
    spy_df = fetch_polygon_history("SPY", start_str, end_str)
    spy_returns = None
    market_regime_df = None
    
    if spy_df is not None and len(spy_df) > 50:
        spy_returns = spy_df['Close'].pct_change(20)
        market_regime_df = calculate_market_regime(spy_df)
        print(f"   ✅ SPY data loaded ({len(spy_df)} days)")
    
    # Fetch sector ETF data
    print("   Fetching sector ETF data...")
    sector_etf_returns = fetch_sector_etf_data(start_str, end_str)
    
    # Get universe
    tickers = get_universe_tickers(universe_limit)
    print(f"   Universe: {len(tickers)} tickers")
    
    # Fetch stock data
    all_data = []
    print(f"   Fetching stock data (threads=15)...")
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_ticker = {
            executor.submit(fetch_polygon_history, t, start_str, end_str): t 
            for t in tickers
        }
        completed = 0
        for future in as_completed(future_to_ticker):
            completed += 1
            if completed % 100 == 0:
                print(f"      ... processed {completed}/{len(tickers)}")
            
            t = future_to_ticker[future]
            df = future.result()
            if df is not None and len(df) > 50:
                df = calculate_features(df, spy_returns, market_regime_df, sector_etf_returns, t)
                df['Ticker'] = t
                all_data.append(df)
    
    if not all_data:
        raise RuntimeError("No data downloaded!")
    
    full_df = pd.concat(all_data)
    print(f"   Total rows: {len(full_df):,}")
    
    # Create labels (top 20% performers)
    WINNER_PERCENTILE = 80
    threshold = full_df['Forward_Return_20d'].quantile(WINNER_PERCENTILE / 100.0)
    full_df['Label'] = (full_df['Forward_Return_20d'] >= threshold).astype(int)
    
    print(f"   Winner threshold: {threshold*100:.1f}% return")
    print(f"   Class balance: {full_df['Label'].mean()*100:.1f}% winners")
    
    # Sort by date for time series split
    full_df = full_df.sort_index()
    
    X = full_df[FEATURES].fillna(0)
    y = full_df['Label']
    
    return X, y


def create_objective(X: pd.DataFrame, y: pd.Series, n_cv_splits: int = 3):
    """
    Create Optuna objective function for hyperparameter optimization.
    
    Args:
        X: Feature matrix
        y: Labels
        n_cv_splits: Number of CV splits
    
    Returns:
        Objective function for Optuna
    """
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: maximize average Precision@20 across CV folds."""
        
        # ==========================================
        # HistGradientBoostingClassifier parameters
        # ==========================================
        histgb_params = {
            'max_iter': trial.suggest_int('histgb_max_iter', 100, 500),
            'learning_rate': trial.suggest_float('histgb_lr', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('histgb_max_depth', 3, 8),
            'min_samples_leaf': trial.suggest_int('histgb_min_samples_leaf', 10, 50),
            'l2_regularization': trial.suggest_float('histgb_l2', 0.01, 1.0, log=True),
            'max_bins': trial.suggest_categorical('histgb_max_bins', [64, 128, 255]),
        }
        
        # ==========================================
        # RandomForestClassifier parameters
        # ==========================================
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 400),
            'max_depth': trial.suggest_int('rf_max_depth', 4, 10),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 10, 50),
            'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', 0.3, 0.5]),
        }
        
        # ==========================================
        # LogisticRegression parameters
        # ==========================================
        lr_penalty = trial.suggest_categorical('lr_penalty', ['l1', 'l2'])
        lr_params = {
            'C': trial.suggest_float('lr_C', 0.01, 10.0, log=True),
            'penalty': lr_penalty,
            'solver': 'saga',  # Required for l1
        }
        
        # ==========================================
        # Ensemble weights
        # ==========================================
        w1 = trial.suggest_float('weight_histgb', 0.2, 0.6)
        w2 = trial.suggest_float('weight_rf', 0.2, 0.5)
        w3 = 1.0 - w1 - w2
        
        # Ensure minimum weight for LogisticRegression
        if w3 < 0.1:
            return 0.0
        
        weights = [w1, w2, w3]
        
        # ==========================================
        # Time-Series Cross-Validation
        # ==========================================
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        cv_scores = []
        cv_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Skip folds with insufficient positive samples
            if y_train.sum() < 10 or y_val.sum() < 5:
                continue
            
            try:
                # ---- Train HistGradientBoosting ----
                model1 = HistGradientBoostingClassifier(
                    **histgb_params,
                    class_weight='balanced',
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=42,
                )
                model1.fit(X_train, y_train)
                
                # ---- Train RandomForest ----
                model2 = RandomForestClassifier(
                    **rf_params,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                )
                model2.fit(X_train, y_train)
                
                # ---- Train LogisticRegression (with scaling) ----
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model3 = LogisticRegression(
                    **lr_params,
                    class_weight='balanced',
                    max_iter=2000,
                    random_state=42,
                )
                model3.fit(X_train_scaled, y_train)
                model3._needs_scaling = True
                
                # ---- Create Ensemble ----
                ensemble = EnsembleClassifier(
                    models=[model1, model2, model3],
                    weights=weights,
                    scaler=scaler,
                )
                
                # ---- Evaluate ----
                y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
                
                # Primary metric: Precision@20
                p20 = precision_at_k(y_val, y_pred_proba, k=20)
                cv_scores.append(p20)
                
                # Secondary metric: AUC
                try:
                    auc = roc_auc_score(y_val, y_pred_proba)
                    cv_aucs.append(auc)
                except:
                    pass
                
                # Report intermediate value for pruning
                trial.report(np.mean(cv_scores), fold)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                # Training failed for this fold
                continue
        
        if not cv_scores:
            return 0.0
        
        # Store additional metrics as user attributes
        trial.set_user_attr('cv_auc', np.mean(cv_aucs) if cv_aucs else 0.0)
        trial.set_user_attr('cv_std', np.std(cv_scores))
        trial.set_user_attr('n_folds', len(cv_scores))
        
        return np.mean(cv_scores)
    
    return objective


def run_tuning(
    n_trials: int = 100, 
    timeout: int = 3600,
    universe_limit: int = 500,
    n_cv_splits: int = 3,
    verbose: bool = True
) -> optuna.Study:
    """
    Run hyperparameter tuning.
    
    Args:
        n_trials: Number of Optuna trials
        timeout: Max tuning time in seconds
        universe_limit: Max stocks in universe
        n_cv_splits: Number of CV splits
        verbose: Print progress
    
    Returns:
        Optuna study object
    """
    _check_api_key()
    
    # Load data
    X, y = load_training_data(universe_limit=universe_limit)
    
    print(f"\n🔍 Starting Optuna Optimization")
    print(f"   Trials: {n_trials}")
    print(f"   Timeout: {timeout}s ({timeout/60:.0f} min)")
    print(f"   CV Splits: {n_cv_splits}")
    print(f"   Features: {len(FEATURES)}")
    print(f"   Samples: {len(X):,}")
    
    # Create study with pruning
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name='stock_scout_ml_tuning',
    )
    
    # Create objective
    objective = create_objective(X, y, n_cv_splits=n_cv_splits)
    
    # Progress callback
    best_value = [0.0]
    def progress_callback(study: optuna.Study, trial: optuna.FrozenTrial):
        if trial.value is not None and trial.value > best_value[0]:
            best_value[0] = trial.value
            if verbose:
                print(f"   🏆 Trial {trial.number}: P@20={trial.value:.4f} (new best!)")
        elif verbose and trial.number % 10 == 0:
            print(f"   Trial {trial.number}: P@20={trial.value:.4f if trial.value else 'pruned'}")
    
    # Run optimization
    print("\n   Starting trials...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        callbacks=[progress_callback],
    )
    
    # Report results
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\n   📊 Study Statistics:")
    print(f"      Total trials: {len(study.trials)}")
    print(f"      Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"      Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"      Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    print(f"\n   🎯 Best Trial:")
    print(f"      Trial number: {study.best_trial.number}")
    print(f"      Precision@20: {study.best_value:.4f}")
    
    if 'cv_auc' in study.best_trial.user_attrs:
        print(f"      CV AUC: {study.best_trial.user_attrs['cv_auc']:.4f}")
    if 'cv_std' in study.best_trial.user_attrs:
        print(f"      CV Std: {study.best_trial.user_attrs['cv_std']:.4f}")
    
    print(f"\n   📝 Best Hyperparameters:")
    
    # Group parameters by model
    histgb_params = {}
    rf_params = {}
    lr_params = {}
    ensemble_params = {}
    
    for key, value in study.best_params.items():
        if key.startswith('histgb_'):
            histgb_params[key.replace('histgb_', '')] = value
        elif key.startswith('rf_'):
            rf_params[key.replace('rf_', '')] = value
        elif key.startswith('lr_'):
            lr_params[key.replace('lr_', '')] = value
        elif key.startswith('weight_'):
            ensemble_params[key] = value
    
    print(f"\n      HistGradientBoostingClassifier:")
    for k, v in histgb_params.items():
        print(f"         {k}: {v}")
    
    print(f"\n      RandomForestClassifier:")
    for k, v in rf_params.items():
        print(f"         {k}: {v}")
    
    print(f"\n      LogisticRegression:")
    for k, v in lr_params.items():
        print(f"         {k}: {v}")
    
    print(f"\n      Ensemble Weights:")
    for k, v in ensemble_params.items():
        print(f"         {k}: {v:.3f}")
    w3 = 1.0 - ensemble_params.get('weight_histgb', 0) - ensemble_params.get('weight_rf', 0)
    print(f"         weight_lr (computed): {w3:.3f}")
    
    # Save results
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'best_params': study.best_params,
        'n_trials_total': len(study.trials),
        'n_trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_cv_splits': n_cv_splits,
            'universe_limit': universe_limit,
            'n_features': len(FEATURES),
            'n_samples': len(X),
        }
    }
    
    # Save to JSON
    output_path = MODELS_DIR / f"tuning_results_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n   📄 Results saved to: {output_path}")
    
    # Also save as latest
    latest_path = MODELS_DIR / "tuning_results_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save top 10 trials
    top_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True
    )[:10]
    
    top_trials_data = []
    for t in top_trials:
        top_trials_data.append({
            'trial': t.number,
            'value': t.value,
            'params': t.params,
            'user_attrs': t.user_attrs,
        })
    
    top_trials_path = REPORTS_DIR / f"tuning_top_trials_{timestamp}.json"
    with open(top_trials_path, 'w') as f:
        json.dump(top_trials_data, f, indent=2)
    print(f"   📄 Top 10 trials saved to: {top_trials_path}")
    
    return study


def plot_optimization_history(study: optuna.Study, output_dir: str = "reports"):
    """
    Plot Optuna optimization history and parameter importance.
    
    Args:
        study: Optuna study object
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )
        
        print("\n📊 Generating visualizations...")
        
        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_html(str(output_path / 'optuna_history.html'))
        print(f"   ✅ Optimization history: {output_path / 'optuna_history.html'}")
        
        # Parameter importance
        try:
            fig2 = plot_param_importances(study)
            fig2.write_html(str(output_path / 'optuna_importance.html'))
            print(f"   ✅ Parameter importance: {output_path / 'optuna_importance.html'}")
        except Exception as e:
            print(f"   ⚠️  Parameter importance skipped: {e}")
        
        # Parallel coordinate plot
        try:
            fig3 = plot_parallel_coordinate(study)
            fig3.write_html(str(output_path / 'optuna_parallel.html'))
            print(f"   ✅ Parallel coordinates: {output_path / 'optuna_parallel.html'}")
        except Exception as e:
            print(f"   ⚠️  Parallel coordinates skipped: {e}")
        
        # Slice plot for top parameters
        try:
            # Get top 5 most important parameters
            importance = optuna.importance.get_param_importances(study)
            top_params = list(importance.keys())[:5]
            
            fig4 = plot_slice(study, params=top_params)
            fig4.write_html(str(output_path / 'optuna_slice.html'))
            print(f"   ✅ Slice plot: {output_path / 'optuna_slice.html'}")
        except Exception as e:
            print(f"   ⚠️  Slice plot skipped: {e}")
        
    except ImportError:
        print("   ⚠️  Visualizations require: pip install plotly kaleido")
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")


def generate_training_config(study: optuna.Study, output_path: str = "models/optimized_config.py"):
    """
    Generate a Python config file with optimized hyperparameters.
    
    Args:
        study: Optuna study object
        output_path: Path to save config file
    """
    params = study.best_params
    
    # Extract parameters
    config = f'''"""
Optimized Hyperparameters for Stock Scout ML.

Generated by: scripts/tune_hyperparams.py
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Best Precision@20: {study.best_value:.4f}
"""

# HistGradientBoostingClassifier parameters
HISTGB_PARAMS = {{
    'max_iter': {params.get('histgb_max_iter', 300)},
    'learning_rate': {params.get('histgb_lr', 0.05)},
    'max_depth': {params.get('histgb_max_depth', 4)},
    'min_samples_leaf': {params.get('histgb_min_samples_leaf', 20)},
    'l2_regularization': {params.get('histgb_l2', 0.1)},
    'max_bins': {params.get('histgb_max_bins', 255)},
    'class_weight': 'balanced',
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 10,
    'random_state': 42,
}}

# RandomForestClassifier parameters
RF_PARAMS = {{
    'n_estimators': {params.get('rf_n_estimators', 200)},
    'max_depth': {params.get('rf_max_depth', 6)},
    'min_samples_leaf': {params.get('rf_min_samples_leaf', 20)},
    'max_features': {repr(params.get('rf_max_features', 'sqrt'))},
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
}}

# LogisticRegression parameters
LR_PARAMS = {{
    'C': {params.get('lr_C', 0.1)},
    'penalty': {repr(params.get('lr_penalty', 'l2'))},
    'solver': 'saga',
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': 42,
}}

# Ensemble weights
ENSEMBLE_WEIGHTS = [
    {params.get('weight_histgb', 0.45):.4f},  # HistGradientBoosting
    {params.get('weight_rf', 0.35):.4f},  # RandomForest
    {1.0 - params.get('weight_histgb', 0.45) - params.get('weight_rf', 0.35):.4f},  # LogisticRegression
]
'''
    
    with open(output_path, 'w') as f:
        f.write(config)
    
    print(f"\n   📄 Training config saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Tuning for Stock Scout ML',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--trials', type=int, default=50, 
                       help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=1800, 
                       help='Max tuning time in seconds')
    parser.add_argument('--universe', type=int, default=500,
                       help='Max stocks in universe')
    parser.add_argument('--cv-splits', type=int, default=3,
                       help='Number of CV splits')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--generate-config', action='store_true',
                       help='Generate optimized config file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔧 Stock Scout ML Hyperparameter Tuning")
    print("=" * 60)
    
    # Run tuning
    study = run_tuning(
        n_trials=args.trials,
        timeout=args.timeout,
        universe_limit=args.universe,
        n_cv_splits=args.cv_splits,
        verbose=True,
    )
    
    # Generate visualizations
    if not args.no_plot:
        plot_optimization_history(study)
    
    # Generate config file
    if args.generate_config:
        generate_training_config(study)
    
    print("\n" + "=" * 60)
    print("✅ Hyperparameter Tuning Complete!")
    print("=" * 60)
    
    return study


if __name__ == "__main__":
    main()
