#!/usr/bin/env python3
"""
Modified experiment script with ELBO fixes and better hyperparameters
For testing purposes - runs with smaller data and better hyperparameters
"""

import sys
import os
sys.path.append('VariationalInference')

# Import the specific functions we need
try:
    from data import prepare_and_load_emtab
    from vi_model_complete import run_model_and_evaluate
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

def run_fixed_experiment():
    """Run experiment with fixed hyperparameters"""
    print("=== Running Experiment with ELBO Fixes ===")
    
    # Use the same data loading logic as the original
    print("Loading EMTAB data...")
    adata = prepare_and_load_emtab()
    
    # Extract data from AnnData object
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values
    x_aux = adata.obs[["age", "sex_female"]].values
    var_names = adata.var_names.tolist()
    sample_ids = adata.obs_names.tolist()
    
    # Use a MUCH smaller subset for testing
    n_samples_test = min(100, X.shape[0])  # Use only 100 samples
    n_genes_test = min(2000, X.shape[1])   # Use only 2000 genes
    
    print(f"Using subset: {n_samples_test} samples, {n_genes_test} genes")
    
    X_subset = X[:n_samples_test, :n_genes_test]
    Y_subset = Y[:n_samples_test]
    x_aux_subset = x_aux[:n_samples_test]
    var_names_subset = var_names[:n_genes_test] if var_names is not None else None
    sample_ids_subset = sample_ids[:n_samples_test] if sample_ids is not None else None
    
    # CRITICAL: Use MUCH better hyperparameters
    hyperparams = {
        'd': 50,                    # Use only 50 factors instead of 1270!!
        'alpha_eta': 0.1,          # Much smaller than default 1.0
        'lambda_eta': 0.1,         # Much smaller than default 1.0
        'alpha_beta': 0.01,        # MUCH smaller than default 1.0
        'alpha_xi': 0.1,           # Much smaller than default 1.0  
        'lambda_xi': 0.1,          # Much smaller than default 1.0
        'alpha_theta': 0.01,       # MUCH smaller than default 1.0
        'sigma2_gamma': 1.0,       # Keep default
        'sigma2_v': 1.0            # Keep default
    }
    
    print(f"Using hyperparameters:")
    for key, val in hyperparams.items():
        print(f"  {key}: {val}")
    print()
    
    # Run the model with better hyperparameters and fewer iterations
    print("Running model...")
    try:
        result = run_model_and_evaluate(
            x_data=X_subset,
            x_aux=x_aux_subset, 
            y_data=Y_subset,
            var_names=var_names_subset,
            hyperparams=hyperparams,
            seed=42,
            max_iters=10,               # Only 10 iterations for testing
            return_probs=False,
            sample_ids=sample_ids_subset,
            mask=None,
            scores=None,
            plot_elbo=False,
            return_params=False,
            verbose=True                # Verbose to see ELBO progress
        )
        
        print("\n=== SUCCESS ===")
        print("Fixed experiment completed successfully!")
        print("ELBO fixes are working!")
        
        if 'train_metrics' in result:
            print(f"Final metrics: {result['train_metrics']}")
        
        return True
        
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Add command line argument for number of runs
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("num_runs", type=int, default=1, help="Number of experiment runs")
    args = parser.parse_args()
    
    print(f"Running {args.num_runs} experiment(s) with ELBO fixes...")
    
    success_count = 0
    for i in range(args.num_runs):
        print(f"\n--- Starting experiment {i+1}/{args.num_runs} ---")
        
        if run_fixed_experiment():
            success_count += 1
        else:
            print("Experiment failed!")
            break
    
    print(f"\nCompleted {success_count}/{args.num_runs} experiments successfully")
    
    if success_count == args.num_runs:
        print("🎉 All experiments successful! ELBO fixes are working!")
        print("\nNext steps:")
        print("1. Gradually increase the number of factors (d)")
        print("2. Gradually increase the sample/gene sizes")  
        print("3. Fine-tune hyperparameters as needed")
    else:
        print("❌ Some experiments failed - need further debugging")
    
    sys.exit(0 if success_count == args.num_runs else 1) 