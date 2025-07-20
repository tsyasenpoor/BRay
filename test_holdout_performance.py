#!/usr/bin/env python3
"""
Test the best hyperparameter configuration on holdout data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import sys
sys.path.append('VariationalInference')

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

def test_holdout_performance():
    """Test the best configuration on holdout data."""
    
    print("=== HOLDOUT PERFORMANCE TEST ===")
    
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
    
    # Best configurations to test
    configurations = {
        "Perfect F1 Config": {
            "alpha_eta": 8.90620438616169,
            "lambda_eta": 0.25135566617708294,
            "alpha_beta": 0.38003292140452,
            "alpha_xi": 1.0988100318524612,
            "lambda_xi": 0.22464551680532605,
            "alpha_theta": 0.0010959604536925846,
            "sigma2_v": 6.708188643346291,
            "sigma2_gamma": 0.4896262051737685,
            "d": 802
        },
        "Best Regular Config": {
            "alpha_eta": 0.01,
            "lambda_eta": 10.0,
            "alpha_beta": 0.001,
            "alpha_xi": 0.01,
            "lambda_xi": 0.01,
            "alpha_theta": 1.0,
            "sigma2_v": 10.0,
            "sigma2_gamma": 10.0,
            "d": 50
        },
        "Balanced Config": {
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
    }
    
    results = {}
    
    for config_name, config in configurations.items():
        print(f"\nTesting {config_name}...")
        print(f"  d={config['d']}, sigma2_v={config['sigma2_v']:.3f}")
        
        try:
            # Use a larger test size to get more reliable estimates
            result = run_model_and_evaluate(
                x_data=X,
                x_aux=x_aux,
                y_data=Y_reshaped,
                var_names=var_names,
                hyperparams=config,
                seed=42,
                test_size=0.3,  # Larger test set
                val_size=0.2,
                max_iters=100,
                return_probs=True,
                sample_ids=sample_ids,
                mask=None,
                scores=None,
                return_params=False,
                verbose=False,
            )
            
            # Get test set results
            test_probs = np.array(result['test_probabilities'])[:, 0]
            test_labels = np.array(result['test_labels'])[:, 0]
            test_preds = (test_probs >= 0.5).astype(int)
            
            # Calculate metrics
            f1 = f1_score(test_labels, test_preds)
            acc = accuracy_score(test_labels, test_preds)
            prec = precision_score(test_labels, test_preds, zero_division=0)
            rec = recall_score(test_labels, test_preds, zero_division=0)
            
            # Calculate AUC if possible
            try:
                auc = roc_auc_score(test_labels, test_probs)
            except:
                auc = np.nan
            
            # Get validation metrics for comparison
            val_probs = np.array(result['val_probabilities'])[:, 0]
            val_labels = np.array(result['val_labels'])[:, 0]
            val_preds = (val_probs >= 0.5).astype(int)
            val_f1 = f1_score(val_labels, val_preds)
            val_acc = accuracy_score(val_labels, val_preds)
            
            results[config_name] = {
                'test_f1': f1,
                'test_accuracy': acc,
                'test_precision': prec,
                'test_recall': rec,
                'test_auc': auc,
                'val_f1': val_f1,
                'val_accuracy': val_acc,
                'test_probabilities': test_probs,
                'test_labels': test_labels,
                'test_predictions': test_preds,
                'val_probabilities': val_probs,
                'val_labels': val_labels,
                'val_predictions': val_preds
            }
            
            print(f"  Test F1: {f1:.4f}, Test Accuracy: {acc:.4f}")
            print(f"  Val F1: {val_f1:.4f}, Val Accuracy: {val_acc:.4f}")
            print(f"  Test Precision: {prec:.4f}, Test Recall: {rec:.4f}")
            if not np.isnan(auc):
                print(f"  Test AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[config_name] = None
    
    # Analyze results
    print(f"\n=== HOLDOUT PERFORMANCE SUMMARY ===")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        # Create comparison table
        print(f"{'Configuration':<20} {'Test F1':<10} {'Test Acc':<10} {'Val F1':<10} {'Val Acc':<10} {'AUC':<10}")
        print("-" * 80)
        
        for config_name, result in valid_results.items():
            auc_str = f"{result['test_auc']:.4f}" if not np.isnan(result['test_auc']) else "N/A"
            print(f"{config_name:<20} {result['test_f1']:<10.4f} {result['test_accuracy']:<10.4f} "
                  f"{result['val_f1']:<10.4f} {result['val_accuracy']:<10.4f} {auc_str:<10}")
        
        # Check for overfitting (validation vs test performance)
        print(f"\n=== OVERFITTING ANALYSIS ===")
        for config_name, result in valid_results.items():
            val_test_diff_f1 = result['val_f1'] - result['test_f1']
            val_test_diff_acc = result['val_accuracy'] - result['test_accuracy']
            
            print(f"{config_name}:")
            print(f"  F1 drop (val→test): {val_test_diff_f1:.4f}")
            print(f"  Accuracy drop (val→test): {val_test_diff_acc:.4f}")
            
            if val_test_diff_f1 > 0.1:
                print(f"  ⚠️  WARNING: Large F1 drop suggests overfitting")
            if val_test_diff_acc > 0.1:
                print(f"  ⚠️  WARNING: Large accuracy drop suggests overfitting")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Performance comparison
        plt.subplot(2, 3, 1)
        config_names = list(valid_results.keys())
        test_f1s = [valid_results[name]['test_f1'] for name in config_names]
        val_f1s = [valid_results[name]['val_f1'] for name in config_names]
        
        x = np.arange(len(config_names))
        width = 0.35
        
        plt.bar(x - width/2, val_f1s, width, label='Validation', alpha=0.8)
        plt.bar(x + width/2, test_f1s, width, label='Test', alpha=0.8)
        plt.xlabel('Configuration')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Comparison')
        plt.xticks(x, config_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy comparison
        plt.subplot(2, 3, 2)
        test_accs = [valid_results[name]['test_accuracy'] for name in config_names]
        val_accs = [valid_results[name]['val_accuracy'] for name in config_names]
        
        plt.bar(x - width/2, val_accs, width, label='Validation', alpha=0.8)
        plt.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
        plt.xlabel('Configuration')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison')
        plt.xticks(x, config_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Precision vs Recall
        plt.subplot(2, 3, 3)
        test_precs = [valid_results[name]['test_precision'] for name in config_names]
        test_recs = [valid_results[name]['test_recall'] for name in config_names]
        
        plt.scatter(test_precs, test_recs, s=100, alpha=0.7)
        for i, name in enumerate(config_names):
            plt.annotate(name, (test_precs[i], test_recs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision vs Recall (Test Set)')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Probability distributions
        plt.subplot(2, 3, 4)
        for config_name, result in valid_results.items():
            plt.hist(result['test_probabilities'], bins=20, alpha=0.6, 
                    label=config_name, density=True)
        plt.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Test Set Probability Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: ROC curves
        plt.subplot(2, 3, 5)
        from sklearn.metrics import roc_curve
        for config_name, result in valid_results.items():
            if not np.isnan(result['test_auc']):
                fpr, tpr, _ = roc_curve(result['test_labels'], result['test_probabilities'])
                plt.plot(fpr, tpr, label=f"{config_name} (AUC={result['test_auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (Test Set)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Confusion matrices
        plt.subplot(2, 3, 6)
        best_config = max(valid_results.keys(), key=lambda x: valid_results[x]['test_f1'])
        best_result = valid_results[best_config]
        
        cm = confusion_matrix(best_result['test_labels'], best_result['test_predictions'])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix\n{best_config} (Test)')
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
        plt.savefig('holdout_performance_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find best configuration
        best_config = max(valid_results.keys(), key=lambda x: valid_results[x]['test_f1'])
        best_result = valid_results[best_config]
        
        print(f"\n=== BEST CONFIGURATION ===")
        print(f"Configuration: {best_config}")
        print(f"Test F1: {best_result['test_f1']:.4f}")
        print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
        print(f"Test Precision: {best_result['test_precision']:.4f}")
        print(f"Test Recall: {best_result['test_recall']:.4f}")
        if not np.isnan(best_result['test_auc']):
            print(f"Test AUC: {best_result['test_auc']:.4f}")
        
        return valid_results
    else:
        print("No successful results!")
        return None

if __name__ == "__main__":
    test_holdout_performance() 