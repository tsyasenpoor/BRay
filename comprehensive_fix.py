#!/usr/bin/env python3
"""
Comprehensive fix for the Bayesian optimization issues
Addresses data sparsity, model complexity, and evaluation methodology
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pickle

# Add the VariationalInference directory to path
sys.path.append('VariationalInference')

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
except ImportError:
    print("scikit-optimize is not installed. Please install it with 'pip install scikit-optimize'.")
    sys.exit(1)

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

def preprocess_data():
    """Preprocess the EMTAB data to address sparsity and improve performance."""
    
    print("=== DATA PREPROCESSING ===")
    
    # Load data
    adata = prepare_and_load_emtab()
    Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
    X = adata.X
    x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
    var_names = list(adata.var_names)
    sample_ids = adata.obs.index.tolist()
    
    # Convert to dense if sparse
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    print(f"Original data shape: {X.shape}")
    print(f"Original sparsity: {np.sum(X == 0) / X.size:.2%}")
    
    # 1. Remove genes with zero variance
    variance_selector = VarianceThreshold(threshold=0.01)
    X_var_filtered = variance_selector.fit_transform(X)
    selected_genes = variance_selector.get_support()
    var_names_filtered = [var_names[i] for i in range(len(var_names)) if selected_genes[i]]
    
    print(f"After variance filtering: {X_var_filtered.shape}")
    print(f"Removed {X.shape[1] - X_var_filtered.shape[1]} low-variance genes")
    
    # 2. Remove genes with too many zeros (>90% zeros)
    zero_ratio = np.sum(X_var_filtered == 0, axis=0) / X_var_filtered.shape[0]
    non_zero_genes = zero_ratio < 0.9
    X_sparsity_filtered = X_var_filtered[:, non_zero_genes]
    var_names_sparsity_filtered = [var_names_filtered[i] for i in range(len(var_names_filtered)) if non_zero_genes[i]]
    
    print(f"After sparsity filtering: {X_sparsity_filtered.shape}")
    print(f"Removed {X_var_filtered.shape[1] - X_sparsity_filtered.shape[1]} high-sparsity genes")
    print(f"New sparsity: {np.sum(X_sparsity_filtered == 0) / X_sparsity_filtered.size:.2%}")
    
    # 3. Normalize auxiliary variables
    scaler = StandardScaler()
    x_aux_scaled = scaler.fit_transform(x_aux)
    
    print(f"Auxiliary variables normalized")
    
    # 4. Create stratified labels for better splitting
    stratify_labels = []
    for i in range(len(Y)):
        if Y[i, 0] == 1 and Y[i, 1] == 1:
            stratify_labels.append(3)  # Both positive
        elif Y[i, 0] == 1:
            stratify_labels.append(1)  # CD positive
        elif Y[i, 1] == 1:
            stratify_labels.append(2)  # UC positive
        else:
            stratify_labels.append(0)  # Both negative
    
    stratify_labels = np.array(stratify_labels)
    
    return {
        'X': X_sparsity_filtered,
        'Y': Y,
        'x_aux': x_aux_scaled,
        'var_names': var_names_sparsity_filtered,
        'sample_ids': sample_ids,
        'stratify_labels': stratify_labels
    }

def create_improved_search_space():
    """Create an improved hyperparameter search space."""
    
    # Expanded and more reasonable search space
    space = [
        Real(0.001, 5.0, prior='log-uniform', name='alpha_eta'),
        Real(0.001, 5.0, prior='log-uniform', name='lambda_eta'),
        Real(0.01, 1.0, prior='log-uniform', name='alpha_beta'),
        Real(0.001, 5.0, prior='log-uniform', name='alpha_xi'),
        Real(0.001, 5.0, prior='log-uniform', name='lambda_xi'),
        Real(0.01, 1.0, prior='log-uniform', name='alpha_theta'),
        Real(0.1, 5.0, prior='log-uniform', name='sigma2_v'),
        Real(0.1, 5.0, prior='log-uniform', name='sigma2_gamma'),
        Integer(20, 200, name='d'),  # More reasonable d range
    ]
    
    return space

def evaluate_with_cross_validation(X, Y, x_aux, var_names, sample_ids, stratify_labels, hyperparams, n_folds=3, seed=42):
    """Evaluate model using cross-validation for more robust assessment."""
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_labels)):
        try:
            # Create train/val split for this fold
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            x_aux_train, x_aux_val = x_aux[train_idx], x_aux[val_idx]
            sample_ids_train = [sample_ids[i] for i in train_idx]
            sample_ids_val = [sample_ids[i] for i in val_idx]
            
            # Run model evaluation
            results = run_model_and_evaluate(
                x_data=X,
                x_aux=x_aux,
                y_data=Y,
                var_names=var_names,
                hyperparams=hyperparams,
                seed=seed + fold,
                test_size=0.0,  # No test set in CV
                val_size=0.0,   # No val set in CV
                max_iters=50,   # Reduced for speed
                return_probs=True,
                sample_ids=sample_ids,
                mask=None,
                scores=None,
                return_params=False,
                verbose=False,
                train_indices=train_idx,
                val_indices=val_idx
            )
            
            # Get validation F1 score
            val_f1 = results['val_metrics']['f1']
            cv_scores.append(val_f1)
            
        except Exception as e:
            print(f"Fold {fold} failed: {e}")
            cv_scores.append(0.0)
    
    return np.mean(cv_scores), cv_scores

def run_improved_optimization():
    """Run improved Bayesian optimization."""
    
    print("=== IMPROVED BAYESIAN OPTIMIZATION ===")
    
    # Preprocess data
    data = preprocess_data()
    X, Y, x_aux, var_names, sample_ids, stratify_labels = (
        data['X'], data['Y'], data['x_aux'], data['var_names'], 
        data['sample_ids'], data['stratify_labels']
    )
    
    # Create improved search space
    space = create_improved_search_space()
    
    # Output directory
    output_dir = "improved_bayesopt_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Record of all tried hyperparameters and their val F1
    results_log = []
    best_f1 = -1
    best_params = None
    
    @use_named_args(space)
    def objective(**params):
        global best_f1, best_params
        
        hp = dict(params)
        hp['d'] = int(round(hp['d']))
        
        # Evaluate with cross-validation
        avg_f1, cv_scores = evaluate_with_cross_validation(
            X, Y, x_aux, var_names, sample_ids, stratify_labels, hp, n_folds=3, seed=42
        )
        
        # Log results
        log_entry = dict(hp)
        log_entry['val_f1'] = avg_f1
        log_entry['cv_scores'] = cv_scores
        results_log.append(log_entry)
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_params = dict(hp)
            # Save best so far
            with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=2)
        
        # Save log
        pd.DataFrame(results_log).to_csv(os.path.join(output_dir, 'bayesopt_log.csv'), index=False)
        print(f"Tried: {hp}, avg_f1={avg_f1:.4f}, best_f1={best_f1:.4f}, cv_scores={[f'{s:.3f}' for s in cv_scores]}")
        
        return -avg_f1  # skopt minimizes
    
    # Run Bayesian optimization with more iterations
    print("Starting optimization with 50 iterations...")
    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=50,  # Increased iterations
        random_state=42,
        verbose=True,
    )
    
    print(f"\nOptimization completed!")
    print(f"Best hyperparameters found: {best_params}")
    print(f"Best validation F1: {best_f1:.4f}")
    
    # Save final results
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    return best_params, best_f1

def run_baseline_comparison():
    """Run baseline models for comparison."""
    
    print("\n=== BASELINE MODEL COMPARISON ===")
    
    # Load preprocessed data
    data = preprocess_data()
    X, Y, x_aux, var_names, sample_ids, stratify_labels = (
        data['X'], data['Y'], data['x_aux'], data['var_names'], 
        data['sample_ids'], data['stratify_labels']
    )
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    # Test logistic regression
    print("Testing Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_scores = cross_val_score(lr, X, Y[:, 0], cv=3, scoring='f1')
    print(f"Logistic Regression F1: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
    
    # Test random forest
    print("Testing Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X, Y[:, 0], cv=3, scoring='f1')
    print(f"Random Forest F1: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    
    return {
        'logistic_regression': lr_scores.mean(),
        'random_forest': rf_scores.mean()
    }

def main():
    """Run the comprehensive fix."""
    
    print("=== COMPREHENSIVE FIX FOR BAYESIAN OPTIMIZATION ===")
    
    # Run baseline comparison first
    baseline_scores = run_baseline_comparison()
    
    # Run improved optimization
    best_params, best_f1 = run_improved_optimization()
    
    print("\n=== FINAL RESULTS ===")
    print(f"Baseline Logistic Regression F1: {baseline_scores['logistic_regression']:.4f}")
    print(f"Baseline Random Forest F1: {baseline_scores['random_forest']:.4f}")
    print(f"Improved VI Model F1: {best_f1:.4f}")
    
    if best_f1 > max(baseline_scores.values()):
        print("✅ VI model outperforms baselines!")
    else:
        print("⚠️  VI model needs further improvement")
    
    print("\n=== NEXT STEPS ===")
    print("1. If VI model still underperforms, consider:")
    print("   - Using pathway information for feature engineering")
    print("   - Implementing ensemble methods")
    print("   - Trying different model architectures")
    print("2. If VI model performs well, scale up to full dataset")
    print("3. Consider using the best hyperparameters for production")

if __name__ == "__main__":
    main() 