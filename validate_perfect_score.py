#!/usr/bin/env python3
"""
Validate the perfect F1 score from hyperparameter optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
sys.path.append('VariationalInference')

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

def validate_perfect_score():
    """Validate if the perfect F1 score is legitimate."""
    
    print("=== VALIDATING PERFECT F1 SCORE ===")
    
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
    
    # Test the configuration that achieved F1=1.0000
    perfect_config = {
        "alpha_eta": 8.90620438616169,
        "lambda_eta": 0.25135566617708294,
        "alpha_beta": 0.38003292140452,
        "alpha_xi": 1.0988100318524612,
        "lambda_xi": 0.22464551680532605,
        "alpha_theta": 0.0010959604536925846,
        "sigma2_v": 6.708188643346291,
        "sigma2_gamma": 0.4896262051737685,
        "d": 802
    }
    
    print(f"\nTesting perfect configuration with d={perfect_config['d']}...")
    
    # Run multiple times with different seeds to check stability
    results = []
    for seed in [42, 123, 456, 789, 999]:
        print(f"  Testing with seed {seed}...")
        
        try:
            result = run_model_and_evaluate(
                x_data=X,
                x_aux=x_aux,
                y_data=Y_reshaped,
                var_names=var_names,
                hyperparams=perfect_config,
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
            
            # Calculate metrics
            val_probs = np.array(result['val_probabilities'])[:, 0]
            val_labels = np.array(result['val_labels'])[:, 0]
            val_preds = (val_probs >= 0.5).astype(int)
            
            f1 = f1_score(val_labels, val_preds)
            acc = accuracy_score(val_labels, val_preds)
            prec = precision_score(val_labels, val_preds, zero_division=0)
            rec = recall_score(val_labels, val_preds, zero_division=0)
            
            results.append({
                'seed': seed,
                'f1': f1,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'probabilities': val_probs,
                'labels': val_labels,
                'predictions': val_preds
            })
            
            print(f"    F1: {f1:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
            
        except Exception as e:
            print(f"    Error with seed {seed}: {e}")
    
    if results:
        # Analyze stability
        f1_scores = [r['f1'] for r in results]
        acc_scores = [r['accuracy'] for r in results]
        
        print(f"\n=== STABILITY ANALYSIS ===")
        print(f"F1 scores: {f1_scores}")
        print(f"F1 mean: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"Accuracy scores: {acc_scores}")
        print(f"Accuracy mean: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
        
        # Check for overfitting indicators
        print(f"\n=== OVERFITTING ANALYSIS ===")
        
        # Check class distribution in validation set
        all_labels = np.concatenate([r['labels'] for r in results])
        class_counts = np.bincount(all_labels.astype(int))
        print(f"Validation set class distribution: {class_counts}")
        
        if len(class_counts) > 1:
            minority_ratio = min(class_counts) / sum(class_counts)
            print(f"Minority class ratio: {minority_ratio:.4f}")
            
            if minority_ratio < 0.1:
                print("⚠️  WARNING: Very imbalanced validation set - perfect F1 might be misleading")
        
        # Check probability distributions
        all_probs = np.concatenate([r['probabilities'] for r in results])
        print(f"Probability statistics:")
        print(f"  Mean: {np.mean(all_probs):.4f}")
        print(f"  Std: {np.std(all_probs):.4f}")
        print(f"  Min: {np.min(all_probs):.4f}")
        print(f"  Max: {np.max(all_probs):.4f}")
        
        # Check if probabilities are too extreme
        extreme_probs = np.sum((all_probs < 0.1) | (all_probs > 0.9))
        extreme_ratio = extreme_probs / len(all_probs)
        print(f"Extreme probabilities (<0.1 or >0.9): {extreme_ratio:.4f}")
        
        if extreme_ratio > 0.8:
            print("⚠️  WARNING: Most probabilities are extreme - potential overfitting")
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Plot 1: F1 scores across seeds
        plt.subplot(1, 3, 1)
        seeds = [r['seed'] for r in results]
        plt.bar(seeds, f1_scores, alpha=0.7)
        plt.axhline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(f1_scores):.4f}')
        plt.xlabel('Random Seed')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Stability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Probability distribution
        plt.subplot(1, 3, 2)
        plt.hist(all_probs, bins=30, alpha=0.7, density=True)
        plt.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Confusion matrix (average)
        plt.subplot(1, 3, 3)
        all_preds = np.concatenate([r['predictions'] for r in results])
        cm = confusion_matrix(all_labels, all_preds)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Average Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig('perfect_score_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    else:
        print("No successful results!")
        return None

if __name__ == "__main__":
    validate_perfect_score() 