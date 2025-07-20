#!/usr/bin/env python3
"""
Test if the inference fix improves performance
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import sys
sys.path.append('VariationalInference')

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

def test_inference_fix():
    """Test if the fixed inference improves performance."""
    
    print("=== TESTING INFERENCE FIX ===")
    
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
    
    # Test the best configuration from your results
    best_config = {
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
    
    print(f"\nTesting best configuration (F1=0.5051)...")
    print(f"  d={best_config['d']}, sigma2_v={best_config['sigma2_v']}")
    
    # Run multiple times to check stability
    results = []
    for seed in [42, 123, 456]:
        print(f"\n  Testing with seed {seed}...")
        
        try:
            result = run_model_and_evaluate(
                x_data=X,
                x_aux=x_aux,
                y_data=Y_reshaped,
                var_names=var_names,
                hyperparams=best_config,
                seed=seed,
                test_size=0.2,
                val_size=0.2,
                max_iters=100,
                return_probs=True,
                sample_ids=sample_ids,
                mask=None,
                scores=None,
                return_params=False,
                verbose=False,
            )
            
            # Get metrics
            val_f1 = result['val_metrics']['f1']
            test_f1 = result['test_metrics']['f1']
            val_acc = result['val_metrics']['accuracy']
            test_acc = result['test_metrics']['accuracy']
            
            results.append({
                'seed': seed,
                'val_f1': val_f1,
                'test_f1': test_f1,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'val_probs': result['val_probabilities'],
                'test_probs': result['test_probabilities']
            })
            
            print(f"    Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")
            print(f"    Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            
        except Exception as e:
            print(f"    Error with seed {seed}: {e}")
    
    if results:
        # Analyze results
        val_f1s = [r['val_f1'] for r in results]
        test_f1s = [r['test_f1'] for r in results]
        val_accs = [r['val_acc'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Validation F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
        print(f"Test F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
        print(f"Validation Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
        print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        
        # Check if we're getting close to the 0.5051 from optimization
        mean_val_f1 = np.mean(val_f1s)
        mean_test_f1 = np.mean(test_f1s)
        
        print(f"\n=== COMPARISON WITH OPTIMIZATION ===")
        print(f"Optimization F1: 0.5051")
        print(f"Current Val F1: {mean_val_f1:.4f}")
        print(f"Current Test F1: {mean_test_f1:.4f}")
        
        if abs(mean_val_f1 - 0.5051) < 0.05:
            print("✅ Val F1 is close to optimization result")
        else:
            print("⚠️  Val F1 differs significantly from optimization")
        
        if abs(mean_test_f1 - mean_val_f1) < 0.1:
            print("✅ Test and Val F1 are consistent (no overfitting)")
        else:
            print("⚠️  Large gap between Test and Val F1 (potential overfitting)")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot 1: F1 scores across seeds
        plt.subplot(2, 3, 1)
        seeds = [r['seed'] for r in results]
        x = np.arange(len(seeds))
        width = 0.35
        
        plt.bar(x - width/2, val_f1s, width, label='Validation', alpha=0.8)
        plt.bar(x + width/2, test_f1s, width, label='Test', alpha=0.8)
        plt.axhline(0.5051, color='red', linestyle='--', label='Optimization F1')
        plt.xlabel('Random Seed')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Stability')
        plt.xticks(x, seeds)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy scores
        plt.subplot(2, 3, 2)
        plt.bar(x - width/2, val_accs, width, label='Validation', alpha=0.8)
        plt.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
        plt.xlabel('Random Seed')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Stability')
        plt.xticks(x, seeds)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Probability distributions
        plt.subplot(2, 3, 3)
        all_val_probs = np.concatenate([r['val_probs'] for r in results])
        all_test_probs = np.concatenate([r['test_probs'] for r in results])
        
        plt.hist(all_val_probs, bins=30, alpha=0.7, label='Validation', density=True)
        plt.hist(all_test_probs, bins=30, alpha=0.7, label='Test', density=True)
        plt.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Probability Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Performance comparison
        plt.subplot(2, 3, 4)
        metrics = ['F1', 'Accuracy']
        val_means = [np.mean(val_f1s), np.mean(val_accs)]
        test_means = [np.mean(test_f1s), np.mean(test_accs)]
        
        x = np.arange(len(metrics))
        plt.bar(x - width/2, val_means, width, label='Validation', alpha=0.8)
        plt.bar(x + width/2, test_means, width, label='Test', alpha=0.8)
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Performance Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Stability analysis
        plt.subplot(2, 3, 5)
        val_stability = np.std(val_f1s)
        test_stability = np.std(test_f1s)
        
        plt.bar(['Validation', 'Test'], [val_stability, test_stability], alpha=0.8)
        plt.ylabel('F1 Standard Deviation')
        plt.title('Stability Analysis')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        summary_text = f"""
Inference Fix Test Results

Best Config (d=50):
• Val F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}
• Test F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}
• Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}
• Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}

Comparison:
• Optimization F1: 0.5051
• Current Val F1: {np.mean(val_f1s):.4f}
• Gap: {abs(np.mean(val_f1s) - 0.5051):.4f}

Inference Fix:
✅ Added logistic terms
✅ Added zeta updates
✅ Consistent with training
"""
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('inference_fix_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    else:
        print("No successful results!")
        return None

if __name__ == "__main__":
    test_inference_fix() 