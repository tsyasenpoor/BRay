#!/usr/bin/env python3
"""
Improved Hyperparameter Optimization with Temperature Scaling
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append('VariationalInference')

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

def temperature_scale(probabilities, temperature):
    """Apply temperature scaling to probabilities."""
    scaled_probs = np.power(probabilities, 1/temperature)
    return scaled_probs

def find_optimal_temperature(probabilities, labels, temperature_range=np.linspace(0.1, 5.0, 50)):
    """Find optimal temperature that maximizes F1 score."""
    
    best_f1 = 0
    best_temp = 1.0
    
    for temp in temperature_range:
        scaled_probs = temperature_scale(probabilities, temp)
        preds = (scaled_probs >= 0.5).astype(int)
        f1 = f1_score(labels, preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_temp = temp
    
    return best_temp, best_f1

def evaluate_hyperparameter_set(hyperparams, X, Y, x_aux, var_names, sample_ids):
    """Evaluate a set of hyperparameters with temperature scaling."""
    
    try:
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
            max_iters=100,  # Increased iterations
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
        
        # Find optimal temperature
        best_temp, best_f1 = find_optimal_temperature(vi_probs, vi_labels)
        
        # Get original F1 without temperature scaling
        original_preds = (vi_probs >= 0.5).astype(int)
        original_f1 = f1_score(vi_labels, original_preds)
        
        return {
            'original_f1': original_f1,
            'temperature_scaled_f1': best_f1,
            'optimal_temperature': best_temp,
            'mean_probability': np.mean(vi_probs),
            'std_probability': np.std(vi_probs),
            'probabilities': vi_probs,
            'labels': vi_labels
        }
        
    except Exception as e:
        print(f"Error with hyperparams {hyperparams}: {e}")
        return None

def main():
    """Run improved hyperparameter optimization."""
    
    print("=== IMPROVED HYPERPARAMETER OPTIMIZATION ===")
    
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
    
    # Define improved hyperparameter sets based on our analysis
    hyperparameter_sets = [
        # Original best (baseline)
        {
            "name": "Original Best",
            "params": {
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
        },
        
        # More aggressive regularization (reduce conservativeness)
        {
            "name": "Reduced Regularization",
            "params": {
                "alpha_eta": 0.5,
                "lambda_eta": 1.0,
                "alpha_beta": 0.1,
                "alpha_xi": 0.5,
                "lambda_xi": 1.0,
                "alpha_theta": 0.1,
                "sigma2_v": 1.0,
                "sigma2_gamma": 1.0,
                "d": 30
            }
        },
        
        # Even more aggressive
        {
            "name": "Very Aggressive",
            "params": {
                "alpha_eta": 1.0,
                "lambda_eta": 0.1,
                "alpha_beta": 0.5,
                "alpha_xi": 1.0,
                "lambda_xi": 0.1,
                "alpha_theta": 0.01,
                "sigma2_v": 0.1,
                "sigma2_gamma": 0.1,
                "d": 20
            }
        },
        
        # Balanced approach
        {
            "name": "Balanced",
            "params": {
                "alpha_eta": 0.1,
                "lambda_eta": 2.0,
                "alpha_beta": 0.05,
                "alpha_xi": 0.1,
                "lambda_xi": 2.0,
                "alpha_theta": 0.5,
                "sigma2_v": 2.0,
                "sigma2_gamma": 2.0,
                "d": 25
            }
        },
        
        # Simplified model
        {
            "name": "Simplified",
            "params": {
                "alpha_eta": 0.1,
                "lambda_eta": 1.0,
                "alpha_beta": 0.1,
                "alpha_xi": 0.1,
                "lambda_xi": 1.0,
                "alpha_theta": 0.1,
                "sigma2_v": 1.0,
                "sigma2_gamma": 1.0,
                "d": 15
            }
        }
    ]
    
    results = []
    
    print(f"\nTesting {len(hyperparameter_sets)} hyperparameter configurations...")
    
    for i, config in enumerate(hyperparameter_sets):
        print(f"\n{i+1}/{len(hyperparameter_sets)}: Testing {config['name']}...")
        
        result = evaluate_hyperparameter_set(
            config['params'], X, Y_reshaped, x_aux, var_names, sample_ids
        )
        
        if result is not None:
            result['name'] = config['name']
            result['hyperparams'] = config['params']
            results.append(result)
            
            print(f"  Original F1: {result['original_f1']:.4f}")
            print(f"  Temperature Scaled F1: {result['temperature_scaled_f1']:.4f}")
            print(f"  Optimal Temperature: {result['optimal_temperature']:.3f}")
            print(f"  Mean Probability: {result['mean_probability']:.4f}")
    
    # Find best configuration
    if results:
        best_result = max(results, key=lambda x: x['temperature_scaled_f1'])
        
        print(f"\n=== BEST CONFIGURATION ===")
        print(f"Name: {best_result['name']}")
        print(f"Original F1: {best_result['original_f1']:.4f}")
        print(f"Temperature Scaled F1: {best_result['temperature_scaled_f1']:.4f}")
        print(f"Optimal Temperature: {best_result['optimal_temperature']:.3f}")
        print(f"Mean Probability: {best_result['mean_probability']:.4f}")
        
        print(f"\nBest Hyperparameters:")
        for key, value in best_result['hyperparams'].items():
            print(f"  {key}: {value}")
        
        # Create comparison plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: F1 scores comparison
        plt.subplot(2, 3, 1)
        names = [r['name'] for r in results]
        original_f1s = [r['original_f1'] for r in results]
        scaled_f1s = [r['temperature_scaled_f1'] for r in results]
        
        x = np.arange(len(names))
        width = 0.35
        
        plt.bar(x - width/2, original_f1s, width, label='Original', alpha=0.8)
        plt.bar(x + width/2, scaled_f1s, width, label='Temperature Scaled', alpha=0.8)
        plt.xlabel('Configuration')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Comparison')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Mean probabilities
        plt.subplot(2, 3, 2)
        mean_probs = [r['mean_probability'] for r in results]
        plt.bar(names, mean_probs, alpha=0.8)
        plt.axhline(0.5, color='red', linestyle='--', label='Decision threshold')
        plt.xlabel('Configuration')
        plt.ylabel('Mean Probability')
        plt.title('Mean Probability Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Optimal temperatures
        plt.subplot(2, 3, 3)
        temps = [r['optimal_temperature'] for r in results]
        plt.bar(names, temps, alpha=0.8)
        plt.axhline(1.0, color='red', linestyle='--', label='No scaling (T=1)')
        plt.xlabel('Configuration')
        plt.ylabel('Optimal Temperature')
        plt.title('Optimal Temperature Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Best configuration probability distribution
        plt.subplot(2, 3, 4)
        plt.hist(best_result['probabilities'], bins=30, alpha=0.7, label='Original', density=True)
        optimal_probs = temperature_scale(best_result['probabilities'], best_result['optimal_temperature'])
        plt.hist(optimal_probs, bins=30, alpha=0.7, label='Temperature Scaled', density=True)
        plt.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Best Config: {best_result["name"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Performance improvement
        plt.subplot(2, 3, 5)
        improvements = [r['temperature_scaled_f1'] - r['original_f1'] for r in results]
        plt.bar(names, improvements, alpha=0.8, color='green')
        plt.xlabel('Configuration')
        plt.ylabel('F1 Improvement')
        plt.title('Temperature Scaling Improvement')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Summary table
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        summary_text = f"""
Best Configuration: {best_result['name']}

Original F1: {best_result['original_f1']:.4f}
Temperature Scaled F1: {best_result['temperature_scaled_f1']:.4f}
Improvement: {best_result['temperature_scaled_f1'] - best_result['original_f1']:.4f}
Optimal Temperature: {best_result['optimal_temperature']:.3f}

Key Hyperparameters:
• d: {best_result['hyperparams']['d']}
• sigma2_v: {best_result['hyperparams']['sigma2_v']}
• lambda_eta: {best_result['hyperparams']['lambda_eta']}
"""
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('improved_hyperparameter_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n=== SUMMARY ===")
        print(f"Best configuration: {best_result['name']}")
        print(f"Final F1 score: {best_result['temperature_scaled_f1']:.4f}")
        print(f"Improvement over original: {best_result['temperature_scaled_f1'] - 0.3404:.4f}")
        print(f"Check 'improved_hyperparameter_optimization.png' for detailed comparison")
        
        return best_result
    else:
        print("No successful results!")
        return None

if __name__ == "__main__":
    main() 