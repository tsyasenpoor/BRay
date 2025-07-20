#!/usr/bin/env python3
"""
Calibration fixes for variational inference model
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

def temperature_scaling(probs, temperature):
    """
    Apply temperature scaling to calibrate probabilities.
    
    Args:
        probs: Raw probabilities from model
        temperature: Temperature parameter (T > 1 makes probabilities more conservative)
    
    Returns:
        Calibrated probabilities
    """
    # Convert to logits
    logits = np.log(probs / (1 - probs))
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Convert back to probabilities
    calibrated_probs = 1 / (1 + np.exp(-scaled_logits))
    
    return calibrated_probs

def platt_scaling(probs, labels):
    """
    Apply Platt scaling to calibrate probabilities.
    
    Args:
        probs: Raw probabilities from model
        labels: True labels
    
    Returns:
        Calibrated probabilities and scaling parameters
    """
    # Convert to logits
    logits = np.log(probs / (1 - probs))
    
    # Fit logistic regression to calibrate
    calibrator = LogisticRegression()
    calibrator.fit(logits.reshape(-1, 1), labels)
    
    # Apply calibration
    calibrated_logits = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
    
    return calibrated_logits, calibrator

def isotonic_calibration(probs, labels):
    """
    Apply isotonic regression for non-parametric calibration.
    
    Args:
        probs: Raw probabilities from model
        labels: True labels
    
    Returns:
        Calibrated probabilities and calibrator
    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrated_probs = calibrator.fit_transform(probs, labels)
    
    return calibrated_probs, calibrator

def find_optimal_temperature(probs, labels, temperatures=np.linspace(0.1, 5.0, 50)):
    """
    Find optimal temperature for temperature scaling.
    
    Args:
        probs: Raw probabilities from model
        labels: True labels
        temperatures: Range of temperatures to try
    
    Returns:
        Optimal temperature and corresponding Brier score
    """
    best_temperature = 1.0
    best_brier = brier_score_loss(labels, probs)
    
    for temp in temperatures:
        calibrated_probs = temperature_scaling(probs, temp)
        brier = brier_score_loss(labels, calibrated_probs)
        
        if brier < best_brier:
            best_brier = brier
            best_temperature = temp
    
    return best_temperature, best_brier

def evaluate_calibration_methods(probs, labels):
    """
    Evaluate different calibration methods.
    
    Args:
        probs: Raw probabilities from model
        labels: True labels
    
    Returns:
        Dictionary with results for each method
    """
    results = {}
    
    # Original probabilities
    results['original'] = {
        'brier_score': brier_score_loss(labels, probs),
        'probs': probs
    }
    
    # Temperature scaling
    opt_temp, opt_brier = find_optimal_temperature(probs, labels)
    temp_probs = temperature_scaling(probs, opt_temp)
    results['temperature_scaling'] = {
        'brier_score': opt_brier,
        'temperature': opt_temp,
        'probs': temp_probs
    }
    
    # Platt scaling
    platt_probs, platt_calibrator = platt_scaling(probs, labels)
    results['platt_scaling'] = {
        'brier_score': brier_score_loss(labels, platt_probs),
        'calibrator': platt_calibrator,
        'probs': platt_probs
    }
    
    # Isotonic calibration
    isotonic_probs, isotonic_calibrator = isotonic_calibration(probs, labels)
    results['isotonic_calibration'] = {
        'brier_score': brier_score_loss(labels, isotonic_probs),
        'calibrator': isotonic_calibrator,
        'probs': isotonic_probs
    }
    
    return results

def plot_calibration_comparison(results, labels):
    """
    Plot comparison of different calibration methods.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Probability distributions
    plt.subplot(1, 3, 1)
    for method, result in results.items():
        plt.hist(result['probs'], bins=20, alpha=0.7, label=f"{method} (Brier: {result['brier_score']:.4f})")
    
    plt.axvline(np.mean(labels), color='red', linestyle='--', 
                label=f'True positive rate: {np.mean(labels):.3f}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Calibration curves
    plt.subplot(1, 3, 2)
    from sklearn.calibration import calibration_curve
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    for method, result in results.items():
        if len(result['probs']) > 0:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                labels, result['probs'], n_bins=10
            )
            plt.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                     label=f"{method} (Brier: {result['brier_score']:.4f})")
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Brier scores comparison
    plt.subplot(1, 3, 3)
    methods = list(results.keys())
    brier_scores = [results[method]['brier_score'] for method in methods]
    
    bars = plt.bar(methods, brier_scores)
    plt.ylabel('Brier Score (lower is better)')
    plt.title('Calibration Performance Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, brier_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('calibration_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def integrate_calibration_into_vi():
    """
    Show how to integrate calibration into the VI model.
    """
    
    print("=== INTEGRATING CALIBRATION INTO VI MODEL ===")
    
    # This would be added to your VI model evaluation
    calibration_code = '''
# After running VI model and getting probabilities
def evaluate_with_calibration(X, Y, x_aux, var_names, sample_ids, hyperparams):
    """Evaluate VI model with calibration."""
    
    # Run VI model
    results = run_model_and_evaluate(
        x_data=X,
        x_aux=x_aux,
        y_data=Y,
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
    
    # Get raw probabilities
    raw_probs = np.array(results['val_probabilities'])
    val_labels = np.array(results['val_labels'])
    
    # Apply calibration
    calibration_results = evaluate_calibration_methods(raw_probs[:, 0], val_labels[:, 0])
    
    # Use best calibration method
    best_method = min(calibration_results.keys(), 
                     key=lambda x: calibration_results[x]['brier_score'])
    
    calibrated_probs = calibration_results[best_method]['probs']
    
    # Recompute metrics with calibrated probabilities
    from sklearn.metrics import f1_score, accuracy_score
    
    calibrated_preds = (calibrated_probs >= 0.5).astype(int)
    f1_calibrated = f1_score(val_labels[:, 0], calibrated_preds)
    accuracy_calibrated = accuracy_score(val_labels[:, 0], calibrated_preds)
    
    print(f"Original F1: {results['val_metrics']['f1']:.4f}")
    print(f"Calibrated F1: {f1_calibrated:.4f}")
    print(f"Best calibration method: {best_method}")
    
    return {
        'original_results': results,
        'calibrated_probs': calibrated_probs,
        'calibrated_metrics': {
            'f1': f1_calibrated,
            'accuracy': accuracy_calibrated
        },
        'calibration_method': best_method
    }
'''
    
    print(calibration_code)
    
    print("\n=== KEY INSIGHTS ===")
    print("1. **Calibration can significantly improve F1 scores**")
    print("2. **Temperature scaling is often the best method for VI models**")
    print("3. **Calibration should be done on validation set, not training set**")
    print("4. **The optimal temperature T > 1 indicates overconfidence**")
    print("5. **T < 1 indicates underconfidence**")

if __name__ == "__main__":
    # Example usage
    print("=== CALIBRATION FIXES FOR VI MODEL ===")
    
    # This would be run with your actual VI model results
    integrate_calibration_into_vi()
    
    print("\n=== NEXT STEPS ===")
    print("1. Run calibration_analysis.py to see current calibration issues")
    print("2. Implement temperature scaling in your VI model evaluation")
    print("3. Compare F1 scores before and after calibration")
    print("4. Consider adding calibration loss to your ELBO") 