#!/usr/bin/env python3
"""
IMPROVED Bayesian Optimization for EMTAB VI Model Hyperparameters
Addresses class imbalance, expands search space, and uses better evaluation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
except ImportError:
    print("scikit-optimize is not installed. Please install it with 'pip install scikit-optimize'.")
    sys.exit(1)

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

# EXPANDED search space with better ranges
space = [
    Real(0.001, 20.0, prior='log-uniform', name='alpha_eta'),
    Real(0.001, 20.0, prior='log-uniform', name='lambda_eta'),
    Real(0.0001, 2.0, prior='log-uniform', name='alpha_beta'),
    Real(0.001, 20.0, prior='log-uniform', name='alpha_xi'),
    Real(0.001, 20.0, prior='log-uniform', name='lambda_xi'),
    Real(0.0001, 2.0, prior='log-uniform', name='alpha_theta'),
    Real(0.001, 20.0, prior='log-uniform', name='sigma2_v'),
    Real(0.001, 20.0, prior='log-uniform', name='sigma2_gamma'),
    Integer(10, 500, name='d'),  # Reduced max d for stability
]

# Load data with stratification
SEED = 42
adata = prepare_and_load_emtab()
Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
X = adata.X
x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
var_names = list(adata.var_names)
sample_ids = adata.obs.index.tolist()

# Create stratified labels for splitting
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
    
    # Use cross-validation for more robust evaluation
    n_folds = 3
    cv_scores = []
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_labels)):
        try:
            # Run model with current fold
            results = run_model_and_evaluate(
                x_data=X,
                x_aux=x_aux,
                y_data=Y,
                var_names=var_names,
                hyperparams=hp,
                seed=SEED + fold,
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
            
            # Use balanced F1 score
            val_f1 = results['val_metrics']['f1']
            cv_scores.append(val_f1)
            
        except Exception as e:
            print(f"Fold {fold} failed: {e}")
            cv_scores.append(0.0)
    
    # Average CV score
    avg_f1 = np.mean(cv_scores)
    
    # Log
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
    print(f"Tried: {hp}, avg_f1={avg_f1:.4f}, best_f1={best_f1:.4f}, cv_scores={cv_scores}")
    
    return -avg_f1  # skopt minimizes

# Run Bayesian optimization with more iterations
res = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=100,  # Increased from 20
    random_state=SEED,
    verbose=True,
)

print(f"Best hyperparameters found: {best_params}")
print(f"Best validation F1: {best_f1}")

# Save final results
with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
    json.dump(best_params, f, indent=2)

print("Improved optimization completed!")
