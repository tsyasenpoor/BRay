#!/usr/bin/env python3
"""
Calibration analysis for variational inference model vs logistic regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import pickle
import os

# Add the VariationalInference directory to path
import sys
sys.path.append('VariationalInference')

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

def load_and_preprocess_data():
    """Load and preprocess EMTAB data."""
    
    print("Loading EMTAB data...")
    adata = prepare_and_load_emtab()
    
    Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
    X = adata.X
    x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
    var_names = list(adata.var_names)
    sample_ids = adata.obs.index.tolist()
    
    # Convert to dense if sparse
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Use only first label for simplicity
    Y_single = Y[:, 0]  # Crohn's disease
    
    return X, Y_single, x_aux, var_names, sample_ids

def run_vi_model(X, Y, x_aux, var_names, sample_ids, hyperparams):
    """Run variational inference model and get probabilities."""
    
    print("Running VI model...")
    
    # Reshape Y for the model
    Y_reshaped = Y.reshape(-1, 1)
    
    results = run_model_and_evaluate(
        x_data=X,
        x_aux=x_aux,
        y_data=Y_reshaped,
        var_names=var_names,
        hyperparams=hyperparams,
        seed=42,
        test_size=0.2,
        val_size=0.2,
        max_iters=50,
        return_probs=True,
        sample_ids=sample_ids,
        mask=None,
        scores=None,
        return_params=False,
        verbose=False,
    )
    
    # Get validation probabilities
    val_probs = np.array(results['val_probabilities'])
    val_labels = np.array(results['val_labels'])
    
    return val_probs[:, 0], val_labels[:, 0]  # First class only

def run_logistic_regression(X, Y, x_aux):
    """Run logistic regression and get probabilities."""
    
    print("Running logistic regression...")
    
    # Combine gene expression and auxiliary features
    X_combined = np.hstack([X, x_aux])
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_combined, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    # Fit logistic regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, Y_train)
    
    # Get probabilities
    lr_probs = lr.predict_proba(X_test)[:, 1]
    
    return lr_probs, Y_test

def analyze_calibration(probs, labels, model_name):
    """Analyze calibration of a model."""
    
    print(f"\n=== Calibration Analysis for {model_name} ===")
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, probs, n_bins=10
    )
    
    # Calculate Brier score (lower is better)
    brier_score = brier_score_loss(labels, probs)
    
    # Calculate reliability diagram
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Mean predicted probability: {np.mean(probs):.4f}")
    print(f"Actual positive rate: {np.mean(labels):.4f}")
    
    # Check for systematic bias
    bias = np.mean(probs) - np.mean(labels)
    print(f"Systematic bias: {bias:.4f}")
    
    if abs(bias) > 0.1:
        print("⚠️  LARGE SYSTEMATIC BIAS DETECTED!")
    elif abs(bias) > 0.05:
        print("⚠️  Moderate systematic bias detected")
    else:
        print("✅ Good calibration (low bias)")
    
    return {
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value,
        'brier_score': brier_score,
        'bias': bias,
        'probs': probs,
        'labels': labels
    }

def plot_calibration_curves(vi_results, lr_results):
    """Plot calibration curves for comparison."""
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Calibration curves
    plt.subplot(1, 2, 1)
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # VI model calibration
    plt.plot(vi_results['mean_predicted_value'], 
             vi_results['fraction_of_positives'], 
             'o-', label=f'VI Model (Brier: {vi_results["brier_score"]:.4f})')
    
    # Logistic regression calibration
    plt.plot(lr_results['mean_predicted_value'], 
             lr_results['fraction_of_positives'], 
             's-', label=f'Logistic Regression (Brier: {lr_results["brier_score"]:.4f})')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Probability distributions
    plt.subplot(1, 2, 2)
    
    plt.hist(vi_results['probs'], bins=20, alpha=0.7, label='VI Model', density=True)
    plt.hist(lr_results['probs'], bins=20, alpha=0.7, label='Logistic Regression', density=True)
    plt.axvline(np.mean(vi_results['labels']), color='red', linestyle='--', 
                label=f'True positive rate: {np.mean(vi_results["labels"]):.3f}')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def suggest_calibration_fixes(vi_results, lr_results):
    """Suggest fixes for calibration issues."""
    
    print("\n=== CALIBRATION FIX SUGGESTIONS ===")
    
    vi_brier = vi_results['brier_score']
    lr_brier = lr_results['brier_score']
    vi_bias = vi_results['bias']
    
    print(f"VI Model Brier Score: {vi_brier:.4f}")
    print(f"Logistic Regression Brier Score: {lr_brier:.4f}")
    
    if vi_brier > lr_brier * 1.2:
        print("⚠️  VI model has significantly worse calibration than logistic regression")
        
        print("\n🔧 SUGGESTED FIXES:")
        
        if abs(vi_bias) > 0.05:
            print("1. **Temperature Scaling**: Add a temperature parameter to calibrate probabilities")
            print("   P_calibrated = sigmoid(logit(P) / temperature)")
        
        print("2. **Platt Scaling**: Use logistic regression to recalibrate VI probabilities")
        print("   P_calibrated = sigmoid(a * logit(P) + b)")
        
        print("3. **Isotonic Regression**: Use non-parametric calibration")
        
        print("4. **Model Architecture Changes**:")
        print("   - Add uncertainty quantification to theta inference")
        print("   - Use ensemble methods (multiple VI runs)")
        print("   - Implement proper posterior sampling")
        
        print("5. **Loss Function Changes**:")
        print("   - Add calibration loss to ELBO")
        print("   - Use proper scoring rules (Brier score, log loss)")
        
    else:
        print("✅ VI model calibration is comparable to logistic regression")

def main():
    """Run calibration analysis."""
    
    print("=== CALIBRATION ANALYSIS ===")
    
    # Load data
    X, Y, x_aux, var_names, sample_ids = load_and_preprocess_data()
    
    # Use best hyperparameters from previous optimization
    best_hyperparams = {
        "alpha_eta": 0.01,
        "lambda_eta": 10.0,
        "alpha_beta": 0.001,
        "alpha_xi": 0.01,
        "lambda_xi": 0.01,
        "alpha_theta": 1.0,
        "sigma2_v": 10.0,
        "sigma2_gamma": 10.0,
        "d": 50
    }
    
    # Run VI model
    try:
        vi_probs, vi_labels = run_vi_model(X, Y, x_aux, var_names, sample_ids, best_hyperparams)
        vi_results = analyze_calibration(vi_probs, vi_labels, "VI Model")
    except Exception as e:
        print(f"VI model failed: {e}")
        return
    
    # Run logistic regression
    lr_probs, lr_labels = run_logistic_regression(X, Y, x_aux)
    lr_results = analyze_calibration(lr_probs, lr_labels, "Logistic Regression")
    
    # Plot results
    plot_calibration_curves(vi_results, lr_results)
    
    # Suggest fixes
    suggest_calibration_fixes(vi_results, lr_results)
    
    print("\n=== SUMMARY ===")
    print("Calibration analysis completed!")
    print("Check 'calibration_analysis.png' for visual results")
    print("This analysis helps understand why your VI model might be performing poorly")

if __name__ == "__main__":
    main() 