#!/usr/bin/env python3
"""
Debug why we can't replicate the optimization results
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import sys
sys.path.append('VariationalInference')

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

def debug_optimization_replication():
    """Debug why we can't replicate the optimization results."""
    
    print("=== DEBUGGING OPTIMIZATION REPLICATION ===")
    
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
    
    # Best configuration from optimization
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
    
    print(f"\nTesting exact optimization configuration...")
    print(f"  d={best_config['d']}, sigma2_v={best_config['sigma2_v']}")
    
    # Test multiple seeds to see if we can find one that matches
    results = []
    seeds_to_test = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
    
    for seed in seeds_to_test:
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
            
            # Check if we're close to the optimization result
            if abs(val_f1 - 0.5051) < 0.02:
                print(f"    🎯 CLOSE MATCH! Val F1 {val_f1:.4f} is close to 0.5051")
            
        except Exception as e:
            print(f"    Error with seed {seed}: {e}")
    
    if results:
        # Analyze results
        val_f1s = [r['val_f1'] for r in results]
        test_f1s = [r['test_f1'] for r in results]
        val_accs = [r['val_acc'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        
        print(f"\n=== COMPREHENSIVE RESULTS ===")
        print(f"Validation F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
        print(f"Test F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
        print(f"Validation Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
        print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        
        # Find best and worst results
        best_val_idx = np.argmax(val_f1s)
        worst_val_idx = np.argmin(val_f1s)
        
        print(f"\n=== BEST/WORST RESULTS ===")
        print(f"Best Val F1: {val_f1s[best_val_idx]:.4f} (seed {results[best_val_idx]['seed']})")
        print(f"Worst Val F1: {val_f1s[worst_val_idx]:.4f} (seed {results[worst_val_idx]['seed']})")
        print(f"Range: {np.max(val_f1s) - np.min(val_f1s):.4f}")
        
        # Check if any result is close to optimization
        close_matches = [i for i, f1 in enumerate(val_f1s) if abs(f1 - 0.5051) < 0.05]
        if close_matches:
            print(f"✅ Found {len(close_matches)} results close to optimization F1=0.5051")
            for idx in close_matches:
                print(f"  Seed {results[idx]['seed']}: Val F1 = {val_f1s[idx]:.4f}")
        else:
            print("❌ No results close to optimization F1=0.5051")
        
        # Analyze probability distributions
        print(f"\n=== PROBABILITY ANALYSIS ===")
        all_val_probs = np.concatenate([r['val_probs'] for r in results])
        all_test_probs = np.concatenate([r['test_probs'] for r in results])
        
        print(f"Val probs - Mean: {np.mean(all_val_probs):.4f}, Std: {np.std(all_val_probs):.4f}")
        print(f"Test probs - Mean: {np.mean(all_test_probs):.4f}, Std: {np.std(all_test_probs):.4f}")
        
        # Check for extreme probabilities
        extreme_val = np.sum((all_val_probs < 0.1) | (all_val_probs > 0.9)) / len(all_val_probs)
        extreme_test = np.sum((all_test_probs < 0.1) | (all_test_probs > 0.9)) / len(all_test_probs)
        print(f"Extreme val probs: {extreme_val:.4f}")
        print(f"Extreme test probs: {extreme_test:.4f}")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
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
        plt.title('F1 Score Across Seeds')
        plt.xticks(x, seeds, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: F1 vs Optimization target
        plt.subplot(2, 3, 2)
        plt.scatter(seeds, val_f1s, alpha=0.7, label='Validation')
        plt.scatter(seeds, test_f1s, alpha=0.7, label='Test')
        plt.axhline(0.5051, color='red', linestyle='--', label='Optimization F1')
        plt.xlabel('Random Seed')
        plt.ylabel('F1 Score')
        plt.title('F1 vs Optimization Target')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Probability distributions
        plt.subplot(2, 3, 3)
        plt.hist(all_val_probs, bins=30, alpha=0.7, label='Validation', density=True)
        plt.hist(all_test_probs, bins=30, alpha=0.7, label='Test', density=True)
        plt.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Probability Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Performance stability
        plt.subplot(2, 3, 4)
        val_stability = np.std(val_f1s)
        test_stability = np.std(test_f1s)
        
        plt.bar(['Validation', 'Test'], [val_stability, test_stability], alpha=0.8)
        plt.ylabel('F1 Standard Deviation')
        plt.title('Performance Stability')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Best vs Worst comparison
        plt.subplot(2, 3, 5)
        best_result = results[best_val_idx]
        worst_result = results[worst_val_idx]
        
        comparison_data = [
            best_result['val_f1'], best_result['test_f1'],
            worst_result['val_f1'], worst_result['test_f1']
        ]
        labels = ['Best Val', 'Best Test', 'Worst Val', 'Worst Test']
        
        plt.bar(labels, comparison_data, alpha=0.8)
        plt.axhline(0.5051, color='red', linestyle='--', label='Optimization F1')
        plt.ylabel('F1 Score')
        plt.title('Best vs Worst Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        summary_text = f"""
Optimization Replication Debug

Target F1: 0.5051
Current Results:
• Val F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}
• Test F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}
• Best Val F1: {np.max(val_f1s):.4f}
• Worst Val F1: {np.min(val_f1s):.4f}

Analysis:
• Range: {np.max(val_f1s) - np.min(val_f1s):.4f}
• Close matches: {len(close_matches)}
• Mean gap: {abs(np.mean(val_f1s) - 0.5051):.4f}

Possible Issues:
• Different data splits
• Different random seeds
• Different evaluation protocol
• Model instability
"""
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                 fontsize=8, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('optimization_replication_debug.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    else:
        print("No successful results!")
        return None

if __name__ == "__main__":
    debug_optimization_replication() 