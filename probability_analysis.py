#!/usr/bin/env python3
"""
Probability Distribution Analysis for VI Model
Analyze if the model is being too conservative (predictions near 0.5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
    """Run VI model and get detailed results."""
    
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
        return_params=True,  # Get parameters for analysis
        verbose=False,
    )
    
    return results

def run_logistic_regression(X, Y, x_aux):
    """Run logistic regression for comparison."""
    
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

def analyze_probability_distributions(vi_probs, lr_probs, vi_labels, lr_labels):
    """Analyze probability distributions to understand model behavior."""
    
    print("\n=== PROBABILITY DISTRIBUTION ANALYSIS ===")
    
    # VI Model Analysis
    print("\nVI Model Statistics:")
    print(f"  Mean probability: {np.mean(vi_probs):.4f}")
    print(f"  Std probability: {np.std(vi_probs):.4f}")
    print(f"  Min probability: {np.min(vi_probs):.4f}")
    print(f"  Max probability: {np.max(vi_probs):.4f}")
    print(f"  Median probability: {np.median(vi_probs):.4f}")
    
    # Check for conservative predictions
    near_05 = np.sum((vi_probs >= 0.4) & (vi_probs <= 0.6))
    near_05_pct = near_05 / len(vi_probs) * 100
    print(f"  Predictions near 0.5 (0.4-0.6): {near_05}/{len(vi_probs)} ({near_05_pct:.1f}%)")
    
    very_low = np.sum(vi_probs < 0.2)
    very_high = np.sum(vi_probs > 0.8)
    print(f"  Very low predictions (<0.2): {very_low}/{len(vi_probs)} ({very_low/len(vi_probs)*100:.1f}%)")
    print(f"  Very high predictions (>0.8): {very_high}/{len(vi_probs)} ({very_high/len(vi_probs)*100:.1f}%)")
    
    # Logistic Regression Analysis
    print("\nLogistic Regression Statistics:")
    print(f"  Mean probability: {np.mean(lr_probs):.4f}")
    print(f"  Std probability: {np.std(lr_probs):.4f}")
    print(f"  Min probability: {np.min(lr_probs):.4f}")
    print(f"  Max probability: {np.max(lr_probs):.4f}")
    print(f"  Median probability: {np.median(lr_probs):.4f}")
    
    near_05_lr = np.sum((lr_probs >= 0.4) & (lr_probs <= 0.6))
    near_05_pct_lr = near_05_lr / len(lr_probs) * 100
    print(f"  Predictions near 0.5 (0.4-0.6): {near_05_lr}/{len(lr_probs)} ({near_05_pct_lr:.1f}%)")
    
    very_low_lr = np.sum(lr_probs < 0.2)
    very_high_lr = np.sum(lr_probs > 0.8)
    print(f"  Very low predictions (<0.2): {very_low_lr}/{len(lr_probs)} ({very_low_lr/len(lr_probs)*100:.1f}%)")
    print(f"  Very high predictions (>0.8): {very_high_lr}/{len(lr_probs)} ({very_high_lr/len(lr_probs)*100:.1f}%)")
    
    return {
        'vi_stats': {
            'mean': np.mean(vi_probs),
            'std': np.std(vi_probs),
            'near_05_pct': near_05_pct,
            'very_low_pct': very_low/len(vi_probs)*100,
            'very_high_pct': very_high/len(vi_probs)*100
        },
        'lr_stats': {
            'mean': np.mean(lr_probs),
            'std': np.std(lr_probs),
            'near_05_pct': near_05_pct_lr,
            'very_low_pct': very_low_lr/len(lr_probs)*100,
            'very_high_pct': very_high_lr/len(lr_probs)*100
        }
    }

def analyze_discrimination_ability(vi_probs, lr_probs, vi_labels, lr_labels):
    """Analyze the model's ability to discriminate between classes."""
    
    print("\n=== DISCRIMINATION ANALYSIS ===")
    
    # VI Model discrimination
    vi_preds = (vi_probs >= 0.5).astype(int)
    vi_f1 = f1_score(vi_labels, vi_preds)
    vi_acc = accuracy_score(vi_labels, vi_preds)
    vi_prec = precision_score(vi_labels, vi_preds)
    vi_rec = recall_score(vi_labels, vi_preds)
    
    print(f"VI Model Performance:")
    print(f"  F1 Score: {vi_f1:.4f}")
    print(f"  Accuracy: {vi_acc:.4f}")
    print(f"  Precision: {vi_prec:.4f}")
    print(f"  Recall: {vi_rec:.4f}")
    
    # Logistic Regression discrimination
    lr_preds = (lr_probs >= 0.5).astype(int)
    lr_f1 = f1_score(lr_labels, lr_preds)
    lr_acc = accuracy_score(lr_labels, lr_preds)
    lr_prec = precision_score(lr_labels, lr_preds)
    lr_rec = recall_score(lr_labels, lr_preds)
    
    print(f"\nLogistic Regression Performance:")
    print(f"  F1 Score: {lr_f1:.4f}")
    print(f"  Accuracy: {lr_acc:.4f}")
    print(f"  Precision: {lr_prec:.4f}")
    print(f"  Recall: {lr_rec:.4f}")
    
    # Compare discrimination
    print(f"\nComparison:")
    print(f"  VI vs LR F1: {vi_f1:.4f} vs {lr_f1:.4f}")
    print(f"  VI vs LR Accuracy: {vi_acc:.4f} vs {lr_acc:.4f}")
    
    return {
        'vi_metrics': {'f1': vi_f1, 'accuracy': vi_acc, 'precision': vi_prec, 'recall': vi_rec},
        'lr_metrics': {'f1': lr_f1, 'accuracy': lr_acc, 'precision': lr_prec, 'recall': lr_rec}
    }

def plot_probability_analysis(vi_probs, lr_probs, vi_labels, lr_labels, stats):
    """Create comprehensive plots for probability analysis."""
    
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Probability distributions
    plt.subplot(2, 4, 1)
    plt.hist(vi_probs, bins=30, alpha=0.7, label='VI Model', density=True)
    plt.hist(lr_probs, bins=30, alpha=0.7, label='Logistic Regression', density=True)
    plt.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Zoom in around 0.5
    plt.subplot(2, 4, 2)
    plt.hist(vi_probs, bins=50, alpha=0.7, label='VI Model', density=True)
    plt.hist(lr_probs, bins=50, alpha=0.7, label='Logistic Regression', density=True)
    plt.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
    plt.xlim(0.3, 0.7)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Zoom: Predictions Near 0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Probability vs true label (VI)
    plt.subplot(2, 4, 3)
    plt.scatter(vi_probs[vi_labels == 0], np.zeros_like(vi_probs[vi_labels == 0]), 
                alpha=0.6, label='Negative', s=20)
    plt.scatter(vi_probs[vi_labels == 1], np.ones_like(vi_probs[vi_labels == 1]), 
                alpha=0.6, label='Positive', s=20)
    plt.axhline(0.5, color='red', linestyle='--', label='Decision threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Label')
    plt.title('VI Model: Probability vs True Label')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Probability vs true label (LR)
    plt.subplot(2, 4, 4)
    plt.scatter(lr_probs[lr_labels == 0], np.zeros_like(lr_probs[lr_labels == 0]), 
                alpha=0.6, label='Negative', s=20)
    plt.scatter(lr_probs[lr_labels == 1], np.ones_like(lr_probs[lr_labels == 1]), 
                alpha=0.6, label='Positive', s=20)
    plt.axhline(0.5, color='red', linestyle='--', label='Decision threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Label')
    plt.title('Logistic Regression: Probability vs True Label')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Performance comparison
    plt.subplot(2, 4, 5)
    metrics = ['F1', 'Accuracy', 'Precision', 'Recall']
    vi_values = [stats['vi_metrics']['f1'], stats['vi_metrics']['accuracy'], 
                 stats['vi_metrics']['precision'], stats['vi_metrics']['recall']]
    lr_values = [stats['lr_metrics']['f1'], stats['lr_metrics']['accuracy'], 
                 stats['lr_metrics']['precision'], stats['lr_metrics']['recall']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, vi_values, width, label='VI Model', alpha=0.8)
    plt.bar(x + width/2, lr_values, width, label='Logistic Regression', alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Conservativeness comparison
    plt.subplot(2, 4, 6)
    conservativeness_metrics = ['Near 0.5 (%)', 'Very Low (%)', 'Very High (%)']
    vi_cons = [stats['vi_stats']['near_05_pct'], stats['vi_stats']['very_low_pct'], 
               stats['vi_stats']['very_high_pct']]
    lr_cons = [stats['lr_stats']['near_05_pct'], stats['lr_stats']['very_low_pct'], 
               stats['lr_stats']['very_high_pct']]
    
    x = np.arange(len(conservativeness_metrics))
    plt.bar(x - width/2, vi_cons, width, label='VI Model', alpha=0.8)
    plt.bar(x + width/2, lr_cons, width, label='Logistic Regression', alpha=0.8)
    plt.xlabel('Conservativeness Metrics')
    plt.ylabel('Percentage')
    plt.title('Conservativeness Comparison')
    plt.xticks(x, conservativeness_metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: ROC curves
    plt.subplot(2, 4, 7)
    from sklearn.metrics import roc_curve, auc
    
    # VI ROC
    fpr_vi, tpr_vi, _ = roc_curve(vi_labels, vi_probs)
    auc_vi = auc(fpr_vi, tpr_vi)
    plt.plot(fpr_vi, tpr_vi, label=f'VI Model (AUC: {auc_vi:.3f})')
    
    # LR ROC
    fpr_lr, tpr_lr, _ = roc_curve(lr_labels, lr_probs)
    auc_lr = auc(fpr_lr, tpr_lr)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC: {auc_lr:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Summary statistics
    plt.subplot(2, 4, 8)
    plt.axis('off')
    
    summary_text = f"""
VI Model Summary:
• Mean Probability: {stats['vi_stats']['mean']:.3f}
• Std Probability: {stats['vi_stats']['std']:.3f}
• Near 0.5: {stats['vi_stats']['near_05_pct']:.1f}%
• F1 Score: {stats['vi_metrics']['f1']:.3f}
• AUC: {auc_vi:.3f}

Logistic Regression Summary:
• Mean Probability: {stats['lr_stats']['mean']:.3f}
• Std Probability: {stats['lr_stats']['std']:.3f}
• Near 0.5: {stats['lr_stats']['near_05_pct']:.1f}%
• F1 Score: {stats['lr_metrics']['f1']:.3f}
• AUC: {auc_lr:.3f}
"""
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('probability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def diagnose_issues(stats):
    """Diagnose the main issues with the VI model."""
    
    print("\n=== DIAGNOSIS ===")
    
    vi_near_05 = stats['vi_stats']['near_05_pct']
    lr_near_05 = stats['lr_stats']['near_05_pct']
    vi_f1 = stats['vi_metrics']['f1']
    lr_f1 = stats['lr_metrics']['f1']
    
    print(f"VI Model near 0.5: {vi_near_05:.1f}%")
    print(f"Logistic Regression near 0.5: {lr_near_05:.1f}%")
    print(f"VI F1: {vi_f1:.4f}")
    print(f"LR F1: {lr_f1:.4f}")
    
    if vi_near_05 > 50:
        print("\n🚨 MAJOR ISSUE: VI model is extremely conservative!")
        print("   - More than 50% of predictions are near 0.5")
        print("   - This suggests the model is 'hedging its bets'")
        print("   - Possible causes:")
        print("     • Hyperparameters too conservative")
        print("     • Model complexity too high")
        print("     • Insufficient training iterations")
        print("     • Poor initialization")
    
    elif vi_near_05 > 30:
        print("\n⚠️  MODERATE ISSUE: VI model is somewhat conservative")
        print("   - 30-50% of predictions are near 0.5")
        print("   - This suggests the model lacks confidence")
    
    else:
        print("\n✅ VI model is not overly conservative")
    
    if vi_f1 < lr_f1 * 0.8:
        print("\n🚨 PERFORMANCE ISSUE: VI model significantly underperforms logistic regression")
        print("   - This suggests the complexity is hurting rather than helping")
        print("   - Consider:")
        print("     • Reducing model complexity (lower d)")
        print("     • Using simpler hyperparameters")
        print("     • Adding regularization")
    
    elif vi_f1 < lr_f1:
        print("\n⚠️  VI model slightly underperforms logistic regression")
        print("   - The added complexity isn't providing benefits")
    
    else:
        print("\n✅ VI model performs competitively with logistic regression")

def suggest_fixes(stats):
    """Suggest specific fixes based on the analysis."""
    
    print("\n=== SUGGESTED FIXES ===")
    
    vi_near_05 = stats['vi_stats']['near_05_pct']
    vi_f1 = stats['vi_metrics']['f1']
    lr_f1 = stats['lr_metrics']['f1']
    
    if vi_near_05 > 40:
        print("🔧 For Conservative Model:")
        print("1. **Reduce regularization**: Lower sigma2_v and sigma2_gamma")
        print("2. **Increase model confidence**: Lower lambda_eta and lambda_xi")
        print("3. **Use more aggressive hyperparameters**:")
        print("   - alpha_eta: 0.1 → 0.5")
        print("   - lambda_eta: 10.0 → 1.0")
        print("   - sigma2_v: 10.0 → 1.0")
        print("4. **Increase training iterations**: max_iters: 50 → 200")
        print("5. **Reduce latent dimensions**: d: 50 → 20")
    
    if vi_f1 < lr_f1:
        print("\n🔧 For Poor Performance:")
        print("1. **Simplify the model**: Reduce d to 10-20")
        print("2. **Use pathway information**: Initialize with biological knowledge")
        print("3. **Try different hyperparameter ranges**:")
        print("   - alpha_beta: 0.001 → 0.1")
        print("   - alpha_theta: 1.0 → 0.1")
        print("4. **Add feature selection**: Use only top genes")
        print("5. **Consider ensemble methods**: Combine multiple VI runs")
    
    print("\n🔧 General Improvements:")
    print("1. **Feature engineering**: Use pathway information")
    print("2. **Data preprocessing**: Normalize features better")
    print("3. **Cross-validation**: Use k-fold instead of single split")
    print("4. **Early stopping**: Stop when ELBO plateaus")
    print("5. **Better initialization**: Use data-driven initialization")

def main():
    """Run the complete probability analysis."""
    
    print("=== PROBABILITY DISTRIBUTION ANALYSIS ===")
    
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
        vi_results = run_vi_model(X, Y, x_aux, var_names, sample_ids, best_hyperparams)
        vi_probs = np.array(vi_results['val_probabilities'])[:, 0]
        vi_labels = np.array(vi_results['val_labels'])[:, 0]
    except Exception as e:
        print(f"VI model failed: {e}")
        return
    
    # Run logistic regression
    lr_probs, lr_labels = run_logistic_regression(X, Y, x_aux)
    
    # Analyze distributions
    dist_stats = analyze_probability_distributions(vi_probs, lr_probs, vi_labels, lr_labels)
    
    # Analyze discrimination
    perf_stats = analyze_discrimination_ability(vi_probs, lr_probs, vi_labels, lr_labels)
    
    # Combine stats
    all_stats = {**dist_stats, **perf_stats}
    
    # Create plots
    plot_probability_analysis(vi_probs, lr_probs, vi_labels, lr_labels, all_stats)
    
    # Diagnose issues
    diagnose_issues(all_stats)
    
    # Suggest fixes
    suggest_fixes(all_stats)
    
    print("\n=== SUMMARY ===")
    print("Probability analysis completed!")
    print("Check 'probability_analysis.png' for detailed visualizations")
    print("This analysis reveals whether the issue is conservativeness or discrimination")

if __name__ == "__main__":
    main() 