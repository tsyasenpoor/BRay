"""
Bayesian Optimization for EMTAB VI Model Hyperparameters using scikit-optimize (skopt)

Dependencies:
- scikit-optimize (pip install scikit-optimize)
- numpy, pandas, scikit-learn, etc.

This script optimizes: alpha_eta, lambda_eta, alpha_beta, alpha_xi, lambda_xi, alpha_theta, sigma2_v, sigma2_gamma, d
Objective: maximize validation F1
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
except ImportError:
    print("scikit-optimize is not installed. Please install it with 'pip install scikit-optimize'.")
    sys.exit(1)

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

# Search space definition (log-uniform for most, integer for d)
space = [
    Real(0.01, 10.0, prior='log-uniform', name='alpha_eta'),
    Real(0.01, 10.0, prior='log-uniform', name='lambda_eta'),
    Real(0.001, 1.0, prior='log-uniform', name='alpha_beta'),
    Real(0.01, 10.0, prior='log-uniform', name='alpha_xi'),
    Real(0.01, 10.0, prior='log-uniform', name='lambda_xi'),
    Real(0.001, 1.0, prior='log-uniform', name='alpha_theta'),
    Real(0.01, 10.0, prior='log-uniform', name='sigma2_v'),
    Real(0.01, 10.0, prior='log-uniform', name='sigma2_gamma'),
    Integer(50, 2000, name='d'),
]
param_names = [dim.name for dim in space]

# EMTAB data loading (fixed seed for reproducibility)
SEED = 42
adata = prepare_and_load_emtab()
Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
X = adata.X
x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
var_names = list(adata.var_names)
sample_ids = adata.obs.index.tolist()

test_size = 0.15
val_size = 0.15

# Output directory
output_dir = "skopt_bayesopt_results"
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
    # Run model
    results = run_model_and_evaluate(
        x_data=X,
        x_aux=x_aux,
        y_data=Y,
        var_names=var_names,
        hyperparams=hp,
        seed=SEED,
        test_size=test_size,
        val_size=val_size,
        max_iters=100,
        return_probs=True,
        sample_ids=sample_ids,
        mask=None,
        scores=None,
        return_params=False,
        verbose=False,
    )
    val_f1 = results['val_metrics']['f1']
    # Log
    log_entry = dict(hp)
    log_entry['val_f1'] = val_f1
    results_log.append(log_entry)
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_params = dict(hp)
        # Save best so far
        with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=2)
    # Save log
    pd.DataFrame(results_log).to_csv(os.path.join(output_dir, 'bayesopt_log.csv'), index=False)
    print(f"Tried: {hp}, val_f1={val_f1:.4f}, best_f1={best_f1:.4f}")
    return -val_f1  # skopt minimizes

# Run Bayesian optimization
res = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=50,
    random_state=SEED,
    verbose=True,
)

print(f"Best hyperparameters found: {best_params}")
print(f"Best validation F1: {best_f1}")
with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
    json.dump(best_params, f, indent=2)
