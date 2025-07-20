#!/usr/bin/env python3
"""
Analysis script to understand why Bayesian optimization is performing poorly
and suggest improvements for the EMTAB dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Removed since not available
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pickle
import os

def analyze_bayesopt_results():
    """Analyze the Bayesian optimization results and identify issues."""
    
    print("=== BAYESIAN OPTIMIZATION RESULTS ANALYSIS ===")
    
    # Load the results
    results_file = "skopt_bayesopt_results/bayesopt_log.csv"
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found!")
        return
    
    results_df = pd.read_csv(results_file)
    print(f"Loaded {len(results_df)} optimization trials")
    
    # Basic statistics
    print(f"\nF1 Score Statistics:")
    print(f"  Mean: {results_df['val_f1'].mean():.4f}")
    print(f"  Std:  {results_df['val_f1'].std():.4f}")
    print(f"  Min:  {results_df['val_f1'].min():.4f}")
    print(f"  Max:  {results_df['val_f1'].max():.4f}")
    print(f"  Median: {results_df['val_f1'].median():.4f}")
    
    # Best result
    best_idx = results_df['val_f1'].idxmax()
    best_result = results_df.loc[best_idx]
    print(f"\nBest Result (F1 = {best_result['val_f1']:.4f}):")
    for col in results_df.columns:
        if col != 'val_f1':
            print(f"  {col}: {best_result[col]}")
    
    # Analyze parameter ranges
    print(f"\nParameter Analysis:")
    param_cols = [col for col in results_df.columns if col not in ['val_f1']]
    for col in param_cols:
        print(f"  {col}: {results_df[col].min():.4f} - {results_df[col].max():.4f}")
    
    # Check for patterns in high-performing configurations
    high_perf_threshold = results_df['val_f1'].quantile(0.8)
    high_perf_results = results_df[results_df['val_f1'] >= high_perf_threshold]
    
    print(f"\nHigh-performing configurations (top 20%, F1 >= {high_perf_threshold:.4f}):")
    print(f"  Count: {len(high_perf_results)}")
    if len(high_perf_results) > 0:
        for col in param_cols:
            mean_val = high_perf_results[col].mean()
            print(f"  {col}: {mean_val:.4f}")
    
    return results_df

def analyze_dataset_characteristics():
    """Analyze the EMTAB dataset characteristics."""
    
    print("\n=== DATASET CHARACTERISTICS ANALYSIS ===")
    
    try:
        # Try to load the EMTAB dataset
        data_path = "/labs/Aguiar/SSPA_BRAY/dataset/EMTAB11349/preprocessed/emtab_ensembl_converted.pkl"
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                adata = pickle.load(f)
            
            print(f"Dataset shape: {adata.shape}")
            
            # Analyze class distribution
            if "Crohn's disease" in adata.obs.columns:
                cd_positive = np.sum(adata.obs["Crohn's disease"])
                uc_positive = np.sum(adata.obs["ulcerative colitis"])
                total_samples = len(adata.obs)
                
                print(f"\nClass Distribution:")
                print(f"  Total samples: {total_samples}")
                print(f"  Crohn's disease positive: {cd_positive} ({cd_positive/total_samples:.2%})")
                print(f"  Ulcerative colitis positive: {uc_positive} ({uc_positive/total_samples:.2%})")
                print(f"  Both negative: {total_samples - cd_positive - uc_positive} ({(total_samples - cd_positive - uc_positive)/total_samples:.2%})")
                
                # Check for class imbalance
                cd_ratio = cd_positive / total_samples
                uc_ratio = uc_positive / total_samples
                
                if cd_ratio < 0.1 or cd_ratio > 0.9:
                    print(f"  ⚠️  SEVERE CLASS IMBALANCE in Crohn's disease: {cd_ratio:.2%}")
                elif cd_ratio < 0.2 or cd_ratio > 0.8:
                    print(f"  ⚠️  MODERATE CLASS IMBALANCE in Crohn's disease: {cd_ratio:.2%}")
                else:
                    print(f"  ✅ REASONABLE CLASS BALANCE in Crohn's disease: {cd_ratio:.2%}")
                
                if uc_ratio < 0.1 or uc_ratio > 0.9:
                    print(f"  ⚠️  SEVERE CLASS IMBALANCE in Ulcerative colitis: {uc_ratio:.2%}")
                elif uc_ratio < 0.2 or uc_ratio > 0.8:
                    print(f"  ⚠️  MODERATE CLASS IMBALANCE in Ulcerative colitis: {uc_ratio:.2%}")
                else:
                    print(f"  ✅ REASONABLE CLASS BALANCE in Ulcerative colitis: {uc_ratio:.2%}")
                
                # Analyze auxiliary variables
                if "age" in adata.obs.columns:
                    age_mean = adata.obs["age"].mean()
                    age_std = adata.obs["age"].std()
                    print(f"\nAuxiliary Variables:")
                    print(f"  Age: {age_mean:.1f} ± {age_std:.1f}")
                
                if "sex_female" in adata.obs.columns:
                    female_ratio = adata.obs["sex_female"].mean()
                    print(f"  Female ratio: {female_ratio:.2%}")
                
                # Analyze gene expression data
                X = adata.X
                if hasattr(X, 'toarray'):
                    X = X.toarray()
                
                print(f"\nGene Expression Statistics:")
                print(f"  Mean expression: {np.mean(X):.4f}")
                print(f"  Std expression: {np.std(X):.4f}")
                print(f"  Min expression: {np.min(X):.4f}")
                print(f"  Max expression: {np.max(X):.4f}")
                print(f"  Sparsity: {np.sum(X == 0) / X.size:.2%}")
                
                return {
                    'total_samples': total_samples,
                    'cd_positive': cd_positive,
                    'uc_positive': uc_positive,
                    'cd_ratio': cd_ratio,
                    'uc_ratio': uc_ratio,
                    'sparsity': np.sum(X == 0) / X.size
                }
            else:
                print("❌ Label columns not found in dataset")
                return None
        else:
            print(f"❌ Dataset file not found: {data_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def suggest_improvements(results_df, dataset_info):
    """Suggest improvements based on analysis."""
    
    print("\n=== IMPROVEMENT SUGGESTIONS ===")
    
    # 1. Class imbalance issues
    if dataset_info and (dataset_info['cd_ratio'] < 0.2 or dataset_info['cd_ratio'] > 0.8 or 
                         dataset_info['uc_ratio'] < 0.2 or dataset_info['uc_ratio'] > 0.8):
        print("🔧 CLASS IMBALANCE SOLUTIONS:")
        print("  1. Use class weights in the model")
        print("  2. Implement stratified sampling for train/val/test splits")
        print("  3. Use balanced accuracy or F1-score as optimization metric")
        print("  4. Consider data augmentation techniques")
        print("  5. Use SMOTE or similar oversampling methods")
    
    # 2. Model architecture issues
    print("\n🔧 MODEL ARCHITECTURE IMPROVEMENTS:")
    print("  1. Increase model complexity (more latent factors)")
    print("  2. Add dropout or regularization to prevent overfitting")
    print("  3. Use ensemble methods (combine multiple models)")
    print("  4. Implement early stopping based on validation performance")
    print("  5. Add batch normalization or layer normalization")
    
    # 3. Hyperparameter optimization issues
    print("\n🔧 HYPERPARAMETER OPTIMIZATION IMPROVEMENTS:")
    print("  1. Expand search space for hyperparameters")
    print("  2. Use different optimization algorithms (TPE, BOHB)")
    print("  3. Implement multi-objective optimization (F1 + stability)")
    print("  4. Use cross-validation instead of single train/val split")
    print("  5. Add more optimization iterations (current: 20)")
    
    # 4. Data preprocessing improvements
    print("\n🔧 DATA PREPROCESSING IMPROVEMENTS:")
    print("  1. Feature selection (remove low-variance genes)")
    print("  2. Dimensionality reduction (PCA, UMAP)")
    print("  3. Normalize auxiliary variables")
    print("  4. Handle missing values properly")
    print("  5. Use gene pathway information for feature engineering")
    
    # 5. Evaluation improvements
    print("\n🔧 EVALUATION IMPROVEMENTS:")
    print("  1. Use multiple random seeds for robustness")
    print("  2. Implement k-fold cross-validation")
    print("  3. Use multiple metrics (AUC, precision, recall)")
    print("  4. Add confidence intervals to results")
    print("  5. Compare against baseline models (logistic regression, random forest)")
    
    # 6. Specific recommendations based on results
    if results_df is not None:
        print("\n🔧 SPECIFIC RECOMMENDATIONS:")
        
        # Check if d (latent factors) is too small
        d_values = results_df['d'].values
        if np.max(d_values) < 100:
            print("  - Increase maximum d value (current max: {})".format(np.max(d_values)))
        
        # Check if regularization is too strong
        sigma2_values = results_df[['sigma2_v', 'sigma2_gamma']].values
        if np.min(sigma2_values) > 5:
            print("  - Reduce minimum sigma2 values (current min: {})".format(np.min(sigma2_values)))
        
        # Check if alpha values are too extreme
        alpha_cols = [col for col in results_df.columns if 'alpha_' in col]
        for col in alpha_cols:
            if results_df[col].min() < 0.01 or results_df[col].max() > 5:
                print(f"  - Adjust {col} range (current: {results_df[col].min():.3f} - {results_df[col].max():.3f})")

def create_improved_optimization_script():
    """Create an improved optimization script."""
    
    print("\n=== CREATING IMPROVED OPTIMIZATION SCRIPT ===")
    
    script_content = '''#!/usr/bin/env python3
"""
IMPROVED Bayesian Optimization for EMTAB VI Model Hyperparameters
Addresses class imbalance, expands search space, and uses better evaluation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
except ImportError:
    print("scikit-optimize is not installed. Please install it with 'pip install scikit-optimize'.")
    sys.exit(1)

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate

# EXPANDED search space with better ranges
space = [
    Real(0.001, 20.0, prior='log-uniform', name='alpha_eta'),
    Real(0.001, 20.0, prior='log-uniform', name='lambda_eta'),
    Real(0.0001, 2.0, prior='log-uniform', name='alpha_beta'),
    Real(0.001, 20.0, prior='log-uniform', name='alpha_xi'),
    Real(0.001, 20.0, prior='log-uniform', name='lambda_xi'),
    Real(0.0001, 2.0, prior='log-uniform', name='alpha_theta'),
    Real(0.001, 20.0, prior='log-uniform', name='sigma2_v'),
    Real(0.001, 20.0, prior='log-uniform', name='sigma2_gamma'),
    Integer(10, 500, name='d'),  # Reduced max d for stability
]

# Load data with stratification
SEED = 42
adata = prepare_and_load_emtab()
Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
X = adata.X
x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
var_names = list(adata.var_names)
sample_ids = adata.obs.index.tolist()

# Create stratified labels for splitting
stratify_labels = []
for i in range(len(Y)):
    if Y[i, 0] == 1 and Y[i, 1] == 1:
        stratify_labels.append(3)  # Both positive
    elif Y[i, 0] == 1:
        stratify_labels.append(1)  # CD positive
    elif Y[i, 1] == 1:
        stratify_labels.append(2)  # UC positive
    else:
        stratify_labels.append(0)  # Both negative

stratify_labels = np.array(stratify_labels)

# Output directory
output_dir = "improved_bayesopt_results"
os.makedirs(output_dir, exist_ok=True)

# Record of all tried hyperparameters and their val F1
results_log = []
best_f1 = -1
best_params = None

@use_named_args(space)
def objective(**params):
    global best_f1, best_params
    
    hp = dict(params)
    hp['d'] = int(round(hp['d']))
    
    # Use cross-validation for more robust evaluation
    n_folds = 3
    cv_scores = []
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_labels)):
        try:
            # Run model with current fold
            results = run_model_and_evaluate(
                x_data=X,
                x_aux=x_aux,
                y_data=Y,
                var_names=var_names,
                hyperparams=hp,
                seed=SEED + fold,
                test_size=0.0,  # No test set in CV
                val_size=0.0,   # No val set in CV
                max_iters=50,   # Reduced for speed
                return_probs=True,
                sample_ids=sample_ids,
                mask=None,
                scores=None,
                return_params=False,
                verbose=False,
                train_indices=train_idx,
                val_indices=val_idx
            )
            
            # Use balanced F1 score
            val_f1 = results['val_metrics']['f1']
            cv_scores.append(val_f1)
            
        except Exception as e:
            print(f"Fold {fold} failed: {e}")
            cv_scores.append(0.0)
    
    # Average CV score
    avg_f1 = np.mean(cv_scores)
    
    # Log
    log_entry = dict(hp)
    log_entry['val_f1'] = avg_f1
    log_entry['cv_scores'] = cv_scores
    results_log.append(log_entry)
    
    if avg_f1 > best_f1:
        best_f1 = avg_f1
        best_params = dict(hp)
        # Save best so far
        with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=2)
    
    # Save log
    pd.DataFrame(results_log).to_csv(os.path.join(output_dir, 'bayesopt_log.csv'), index=False)
    print(f"Tried: {hp}, avg_f1={avg_f1:.4f}, best_f1={best_f1:.4f}, cv_scores={cv_scores}")
    
    return -avg_f1  # skopt minimizes

# Run Bayesian optimization with more iterations
res = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=100,  # Increased from 20
    random_state=SEED,
    verbose=True,
)

print(f"Best hyperparameters found: {best_params}")
print(f"Best validation F1: {best_f1}")

# Save final results
with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
    json.dump(best_params, f, indent=2)

print("Improved optimization completed!")
'''
    
    with open('improved_bayesopt.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created improved_bayesopt.py")

def main():
    """Run the complete analysis."""
    
    # Analyze Bayesian optimization results
    results_df = analyze_bayesopt_results()
    
    # Analyze dataset characteristics
    dataset_info = analyze_dataset_characteristics()
    
    # Suggest improvements
    suggest_improvements(results_df, dataset_info)
    
    # Create improved optimization script
    create_improved_optimization_script()
    
    print("\n=== SUMMARY ===")
    print("The main issues identified:")
    print("1. Class imbalance in the EMTAB dataset")
    print("2. Limited hyperparameter search space")
    print("3. Single train/val split evaluation")
    print("4. Insufficient optimization iterations")
    print("5. Potential model complexity issues")
    
    print("\nNext steps:")
    print("1. Run the improved optimization script: python improved_bayesopt.py")
    print("2. Implement class balancing techniques")
    print("3. Try different model architectures")
    print("4. Use cross-validation for more robust evaluation")

if __name__ == "__main__":
    main() 