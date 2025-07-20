#!/usr/bin/env python3
"""
Quick probability distribution check
"""

import numpy as np
import sys
sys.path.append('VariationalInference')

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

def quick_analysis():
    """Quick analysis of probability distributions."""
    
    print("=== QUICK PROBABILITY ANALYSIS ===")
    
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
    
    # Get probabilities
    vi_probs = np.array(results['val_probabilities'])[:, 0]
    vi_labels = np.array(results['val_labels'])[:, 0]
    
    print(f"\nVI Model Probability Analysis:")
    print(f"  Mean probability: {np.mean(vi_probs):.4f}")
    print(f"  Std probability: {np.std(vi_probs):.4f}")
    print(f"  Min probability: {np.min(vi_probs):.4f}")
    print(f"  Max probability: {np.max(vi_probs):.4f}")
    print(f"  Median probability: {np.median(vi_probs):.4f}")
    
    # Check conservativeness
    near_05 = np.sum((vi_probs >= 0.4) & (vi_probs <= 0.6))
    near_05_pct = near_05 / len(vi_probs) * 100
    print(f"  Predictions near 0.5 (0.4-0.6): {near_05}/{len(vi_probs)} ({near_05_pct:.1f}%)")
    
    very_low = np.sum(vi_probs < 0.2)
    very_high = np.sum(vi_probs > 0.8)
    print(f"  Very low predictions (<0.2): {very_low}/{len(vi_probs)} ({very_low/len(vi_probs)*100:.1f}%)")
    print(f"  Very high predictions (>0.8): {very_high}/{len(vi_probs)} ({very_high/len(vi_probs)*100:.1f}%)")
    
    # Performance metrics
    from sklearn.metrics import f1_score, accuracy_score
    vi_preds = (vi_probs >= 0.5).astype(int)
    vi_f1 = f1_score(vi_labels, vi_preds)
    vi_acc = accuracy_score(vi_labels, vi_preds)
    
    print(f"\nVI Model Performance:")
    print(f"  F1 Score: {vi_f1:.4f}")
    print(f"  Accuracy: {vi_acc:.4f}")
    
    # Diagnose the issue
    print(f"\n=== DIAGNOSIS ===")
    
    if near_05_pct > 50:
        print("🚨 MAJOR ISSUE: Model is extremely conservative!")
        print("   - More than 50% of predictions are near 0.5")
        print("   - This explains the poor F1 score")
        print("   - The model is essentially 'hedging its bets'")
    elif near_05_pct > 30:
        print("⚠️  MODERATE ISSUE: Model is somewhat conservative")
        print("   - 30-50% of predictions are near 0.5")
        print("   - This suggests the model lacks confidence")
    else:
        print("✅ Model is not overly conservative")
    
    if vi_f1 < 0.6:
        print("🚨 PERFORMANCE ISSUE: F1 score is poor")
        print("   - This suggests discrimination problems")
    else:
        print("✅ Performance is reasonable")
    
    # Suggest fixes
    print(f"\n=== SUGGESTED FIXES ===")
    
    if near_05_pct > 40:
        print("🔧 For Conservative Model:")
        print("1. Reduce regularization: sigma2_v and sigma2_gamma from 10.0 to 1.0")
        print("2. Increase model confidence: lambda_eta and lambda_xi from 10.0 to 1.0")
        print("3. Use more aggressive hyperparameters:")
        print("   - alpha_eta: 0.01 → 0.5")
        print("   - alpha_beta: 0.001 → 0.1")
        print("4. Reduce latent dimensions: d: 50 → 20")
        print("5. Increase training iterations: max_iters: 50 → 200")
    
    if vi_f1 < 0.6:
        print("\n🔧 For Poor Performance:")
        print("1. Simplify the model: Reduce d to 10-20")
        print("2. Use pathway information for initialization")
        print("3. Try different hyperparameter ranges")
        print("4. Add feature selection")
        print("5. Consider ensemble methods")
    
    return {
        'mean_prob': np.mean(vi_probs),
        'std_prob': np.std(vi_probs),
        'near_05_pct': near_05_pct,
        'f1_score': vi_f1,
        'accuracy': vi_acc
    }

if __name__ == "__main__":
    results = quick_analysis()
    print(f"\n=== SUMMARY ===")
    print(f"Mean probability: {results['mean_prob']:.3f}")
    print(f"Near 0.5: {results['near_05_pct']:.1f}%")
    print(f"F1 Score: {results['f1_score']:.3f}")
    print("This quick analysis reveals the main issue with your VI model.") 