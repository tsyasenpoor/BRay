#!/usr/bin/env python3
"""
Apply Temperature Scaling to VI Model
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
sys.path.append('VariationalInference')

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

def temperature_scale(probabilities, temperature):
    """Apply temperature scaling to probabilities."""
    # Apply temperature scaling: p_scaled = p^(1/temperature)
    scaled_probs = np.power(probabilities, 1/temperature)
    return scaled_probs

def find_optimal_temperature(probabilities, labels, temperature_range=np.linspace(0.1, 5.0, 50)):
    """Find optimal temperature that maximizes F1 score."""
    
    best_f1 = 0
    best_temp = 1.0
    results = []
    
    print("Searching for optimal temperature...")
    for temp in temperature_range:
        # Apply temperature scaling
        scaled_probs = temperature_scale(probabilities, temp)
        
        # Convert to predictions
        preds = (scaled_probs >= 0.5).astype(int)
        
        # Calculate F1 score
        f1 = f1_score(labels, preds)
        
        results.append({
            'temperature': temp,
            'f1_score': f1,
            'mean_prob': np.mean(scaled_probs),
            'std_prob': np.std(scaled_probs)
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_temp = temp
    
    print(f"Best temperature: {best_temp:.3f} (F1: {best_f1:.4f})")
    return best_temp, results

def analyze_temperature_effects(probabilities, labels):
    """Analyze how different temperatures affect the model."""
    
    print("=== TEMPERATURE SCALING ANALYSIS ===")
    
    # Original performance
    original_preds = (probabilities >= 0.5).astype(int)
    original_f1 = f1_score(labels, original_preds)
    original_acc = accuracy_score(labels, original_preds)
    original_prec = precision_score(labels, original_preds)
    original_rec = recall_score(labels, original_preds)
    
    print(f"Original Performance:")
    print(f"  F1 Score: {original_f1:.4f}")
    print(f"  Accuracy: {original_acc:.4f}")
    print(f"  Precision: {original_prec:.4f}")
    print(f"  Recall: {original_rec:.4f}")
    print(f"  Mean Probability: {np.mean(probabilities):.4f}")
    
    # Find optimal temperature
    best_temp, temp_results = find_optimal_temperature(probabilities, labels)
    
    # Apply optimal temperature
    optimal_probs = temperature_scale(probabilities, best_temp)
    optimal_preds = (optimal_probs >= 0.5).astype(int)
    optimal_f1 = f1_score(labels, optimal_preds)
    optimal_acc = accuracy_score(labels, optimal_preds)
    optimal_prec = precision_score(labels, optimal_preds)
    optimal_rec = recall_score(labels, optimal_preds)
    
    print(f"\nOptimal Temperature ({best_temp:.3f}) Performance:")
    print(f"  F1 Score: {optimal_f1:.4f}")
    print(f"  Accuracy: {optimal_acc:.4f}")
    print(f"  Precision: {optimal_prec:.4f}")
    print(f"  Recall: {optimal_rec:.4f}")
    print(f"  Mean Probability: {np.mean(optimal_probs):.4f}")
    
    # Test different temperatures
    test_temps = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    print(f"\nTemperature Effects:")
    for temp in test_temps:
        scaled_probs = temperature_scale(probabilities, temp)
        preds = (scaled_probs >= 0.5).astype(int)
        f1 = f1_score(labels, preds)
        mean_prob = np.mean(scaled_probs)
        print(f"  T={temp:.1f}: F1={f1:.4f}, Mean Prob={mean_prob:.4f}")
    
    return {
        'original': {
            'f1': original_f1,
            'accuracy': original_acc,
            'precision': original_prec,
            'recall': original_rec,
            'mean_prob': np.mean(probabilities)
        },
        'optimal': {
            'temperature': best_temp,
            'f1': optimal_f1,
            'accuracy': optimal_acc,
            'precision': optimal_prec,
            'recall': optimal_rec,
            'mean_prob': np.mean(optimal_probs)
        },
        'temp_results': temp_results
    }

def plot_temperature_analysis(probabilities, labels, results):
    """Plot temperature scaling analysis."""
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: F1 score vs temperature
    plt.subplot(2, 3, 1)
    temps = [r['temperature'] for r in results['temp_results']]
    f1_scores = [r['f1_score'] for r in results['temp_results']]
    plt.plot(temps, f1_scores, 'b-', linewidth=2)
    plt.axvline(results['optimal']['temperature'], color='red', linestyle='--', 
                label=f'Optimal T={results["optimal"]["temperature"]:.3f}')
    plt.xlabel('Temperature')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Mean probability vs temperature
    plt.subplot(2, 3, 2)
    mean_probs = [r['mean_prob'] for r in results['temp_results']]
    plt.plot(temps, mean_probs, 'g-', linewidth=2)
    plt.axhline(0.5, color='red', linestyle='--', label='Decision threshold')
    plt.xlabel('Temperature')
    plt.ylabel('Mean Probability')
    plt.title('Mean Probability vs Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Original vs optimal probability distributions
    plt.subplot(2, 3, 3)
    plt.hist(probabilities, bins=30, alpha=0.7, label='Original', density=True)
    optimal_probs = temperature_scale(probabilities, results['optimal']['temperature'])
    plt.hist(optimal_probs, bins=30, alpha=0.7, label='Optimal Temperature', density=True)
    plt.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison
    plt.subplot(2, 3, 4)
    metrics = ['F1', 'Accuracy', 'Precision', 'Recall']
    original_values = [results['original']['f1'], results['original']['accuracy'], 
                      results['original']['precision'], results['original']['recall']]
    optimal_values = [results['optimal']['f1'], results['optimal']['accuracy'], 
                     results['optimal']['precision'], results['optimal']['recall']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original', alpha=0.8)
    plt.bar(x + width/2, optimal_values, width, label='Temperature Scaled', alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Probability vs true label (original)
    plt.subplot(2, 3, 5)
    plt.scatter(probabilities[labels == 0], np.zeros_like(probabilities[labels == 0]), 
                alpha=0.6, label='Negative', s=20)
    plt.scatter(probabilities[labels == 1], np.ones_like(probabilities[labels == 1]), 
                alpha=0.6, label='Positive', s=20)
    plt.axhline(0.5, color='red', linestyle='--', label='Decision threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Label')
    plt.title('Original: Probability vs True Label')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Probability vs true label (temperature scaled)
    plt.subplot(2, 3, 6)
    plt.scatter(optimal_probs[labels == 0], np.zeros_like(optimal_probs[labels == 0]), 
                alpha=0.6, label='Negative', s=20)
    plt.scatter(optimal_probs[labels == 1], np.ones_like(optimal_probs[labels == 1]), 
                alpha=0.6, label='Positive', s=20)
    plt.axhline(0.5, color='red', linestyle='--', label='Decision threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Label')
    plt.title('Temperature Scaled: Probability vs True Label')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run temperature scaling analysis."""
    
    print("=== TEMPERATURE SCALING FOR VI MODEL ===")
    
    # Load data
    print("Loading data...")
    adata = prepare_and_load_emtab()
    Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
    X = adata.X
    x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
    var_names = list(adata.var_names)
    sample_ids = adata.obs.index.tolist()
    
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    Y_single = Y[:, 0]  # Crohn's disease
    Y_reshaped = Y_single.reshape(-1, 1)
    
    # Use best hyperparameters
    hyperparams = {
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
    print("Running VI model...")
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
    
    # Get probabilities and labels
    vi_probs = np.array(results['val_probabilities'])[:, 0]
    vi_labels = np.array(results['val_labels'])[:, 0]
    
    # Analyze temperature scaling
    analysis_results = analyze_temperature_effects(vi_probs, vi_labels)
    
    # Create plots
    plot_temperature_analysis(vi_probs, vi_labels, analysis_results)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Original F1: {analysis_results['original']['f1']:.4f}")
    print(f"Temperature Scaled F1: {analysis_results['optimal']['f1']:.4f}")
    print(f"Improvement: {analysis_results['optimal']['f1'] - analysis_results['original']['f1']:.4f}")
    print(f"Optimal Temperature: {analysis_results['optimal']['temperature']:.3f}")
    
    if analysis_results['optimal']['f1'] > analysis_results['original']['f1']:
        print("✅ Temperature scaling improved performance!")
    else:
        print("❌ Temperature scaling did not improve performance")
    
    print("\nCheck 'temperature_scaling_analysis.png' for detailed visualizations")

if __name__ == "__main__":
    main() 