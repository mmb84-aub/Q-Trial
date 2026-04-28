"""Feature selection benchmarking framework."""

import logging
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
from .interface import select_features
from .utils import handle_mixed_types, mean_pairwise_correlation

logger = logging.getLogger(__name__)


def evaluate_selection(
    df,
    outcome_column,
    selected_features,
    task_type="classification",
    test_size=0.2,
    random_state=42,
):
    """
    Evaluate predictive performance of a feature selection result.
    
    Args:
        df: Input DataFrame
        outcome_column: Name of target column
        selected_features: List of selected feature names (including outcome)
        task_type: "classification" or "regression"
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        Dict with performance metrics
    """
    try:
        # Prepare data
        feature_cols = [f for f in selected_features if f != outcome_column]
        
        if not feature_cols:
            logger.warning("No features to evaluate")
            return {"error": "No features selected"}
        
        X = df[feature_cols].copy()
        y = df[outcome_column].copy()
        
        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))
        y = y.fillna(y.mean())
        
        # Encode categoricals
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]
        
        X = X.values.astype(float)
        
        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data
        if task_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Train and evaluate
        if task_type == "classification":
            model = LogisticRegression(max_iter=1000, random_state=random_state)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            return {
                "n_features": X.shape[1],
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "auc": float(roc_auc_score(y_test, y_proba)),
            }
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            return {
                "n_features": X.shape[1],
                "r2": float(r2_score(y_test, y_pred)),
                "mse": float(mean_squared_error(y_test, y_pred)),
            }
    
    except Exception as e:
        logger.error(f"Performance evaluation failed: {e}")
        return {"error": str(e)}


def evaluate_stability(
    df,
    outcome_column,
    method,
    n_bootstrap=10,
    random_state=42,
    **kwargs,
):
    """
    Evaluate selection stability via bootstrap sampling.
    
    Runs feature selection on multiple bootstrap samples and measures
    how consistently each feature is selected.
    
    Args:
        df: Input DataFrame
        outcome_column: Name of target column
        method: Feature selection method
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
        **kwargs: Additional arguments to select_features
    
    Returns:
        Dict with stability metrics:
        - selection_frequency: Dict mapping feature -> selection frequency [0, 1]
        - mean_frequency: Average selection frequency
        - std_frequency: Std dev of selection frequencies
    """
    np.random.seed(random_state)
    
    feature_selection_counts = {}
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        df_boot = df.sample(n=len(df), replace=True, random_state=random_state + i)
        
        try:
            result = select_features(df_boot, outcome_column, method=method, **kwargs)
            selected = result.get("selected_features", [])
            
            for feat in selected:
                feature_selection_counts[feat] = feature_selection_counts.get(feat, 0) + 1
        
        except Exception as e:
            logger.warning(f"Bootstrap iteration {i} failed: {e}")
            continue
    
    # Compute selection frequencies
    selection_frequency = {feat: count / n_bootstrap for feat, count in feature_selection_counts.items()}
    
    frequencies = list(selection_frequency.values())
    
    return {
        "selection_frequency": selection_frequency,
        "mean_frequency": float(np.mean(frequencies)) if frequencies else 0.0,
        "std_frequency": float(np.std(frequencies)) if frequencies else 0.0,
        "n_bootstrap": n_bootstrap,
    }


def benchmark_all_methods(
    df,
    outcome_column,
    methods=None,
    task_type="classification",
    n_bootstrap=10,
    verbose=True,
):
    """
    Comprehensive benchmarking of all feature selection methods.
    
    Compares:
    1. Number of selected features
    2. Redundancy (mean pairwise correlation)
    3. Predictive performance (accuracy/AUC or R²/MSE)
    4. Selection stability (bootstrap consistency)
    5. Runtime
    
    Args:
        df: Input DataFrame
        outcome_column: Name of target column
        methods: List of method names (default: all)
        task_type: "classification" or "regression"
        n_bootstrap: Bootstrap samples for stability evaluation
        verbose: Print results
    
    Returns:
        Dict with results for each method:
        {
            "method_name": {
                "selected_features": [...],
                "n_features": int,
                "redundancy": float,
                "performance": {...},
                "stability": {...},
                "runtime_sec": float,
            }
        }
    """
    if methods is None:
        methods = ["mrmr", "lasso", "elastic_net", "univariate"]
    
    results = {}
    
    for method in methods:
        if verbose:
            logger.info(f"Benchmarking {method}...")
        
        try:
            # Run feature selection
            import time
            t0 = time.time()
            selection_result = select_features(
                df,
                outcome_column,
                method=method,
            )
            runtime = time.time() - t0
            
            selected_features = selection_result.get("selected_features", [])
            
            # Measure redundancy (on numeric features only)
            numeric_cols = [f for f in selected_features if f != outcome_column]
            try:
                prepared_df, _, _, _ = handle_mixed_types(df[numeric_cols + [outcome_column]])
                X_selected = prepared_df[numeric_cols].values.astype(float)
                redundancy = mean_pairwise_correlation(X_selected) if X_selected.shape[1] > 1 else 0.0
            except:
                redundancy = 0.0
            
            # Evaluate predictive performance
            perf = evaluate_selection(df, outcome_column, selected_features, task_type)
            
            # Evaluate stability
            stab = evaluate_stability(df, outcome_column, method, n_bootstrap)
            
            results[method] = {
                "selected_features": selected_features,
                "n_features": len(selected_features),
                "redundancy": float(redundancy),
                "performance": perf,
                "stability": stab,
                "runtime_sec": float(runtime),
            }
            
            if verbose:
                logger.info(
                    f"  ✓ {len(selected_features)} features, "
                    f"redundancy={redundancy:.3f}, runtime={runtime:.2f}s"
                )
        
        except Exception as e:
            logger.error(f"Benchmarking {method} failed: {e}")
            results[method] = {"error": str(e)}
    
    return results


def print_benchmark_summary(results, task_type="classification"):
    """
    Print human-readable summary of benchmark results.
    
    Args:
        results: Output from benchmark_all_methods
        task_type: "classification" or "regression"
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION BENCHMARK SUMMARY")
    print("="*80)
    
    # Table header
    if task_type == "classification":
        perf_metrics = "Accuracy    AUC"
    else:
        perf_metrics = "R²          MSE"
    
    print(f"{'Method':<15} {'Features':<10} {'Redundancy':<12} {perf_metrics:<20} {'Runtime':<10}")
    print("-"*80)
    
    # Table rows
    for method, result in results.items():
        if "error" in result:
            print(f"{method:<15} ERROR: {result['error'][:40]}")
        else:
            n_feat = result["n_features"]
            redundancy = result["redundancy"]
            runtime = result["runtime_sec"]
            perf = result["performance"]
            
            if task_type == "classification":
                acc = perf.get("accuracy", np.nan)
                auc = perf.get("auc", np.nan)
                perf_str = f"{acc:.3f}       {auc:.3f}"
            else:
                r2 = perf.get("r2", np.nan)
                mse = perf.get("mse", np.nan)
                perf_str = f"{r2:.3f}       {mse:.3f}"
            
            print(f"{method:<15} {n_feat:<10} {redundancy:<12.3f} {perf_str:<20} {runtime:<10.2f}s")
    
    print("="*80 + "\n")


def export_benchmark_results(results, output_path):
    """Export benchmark results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Benchmark results exported to {output_path}")
