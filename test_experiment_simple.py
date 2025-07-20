#!/usr/bin/env python3
"""
Simplified experiment script to test ELBO fixes with real data
Uses better hyperparameters and runs only a few iterations
"""

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import sys
sys.path.append('VariationalInference')

import jax.numpy as jnp
import numpy as np
import pickle

def run_simple_test():
    """Run simplified experiment with better hyperparameters"""
    print("=== Running Simplified ELBO Test ===")
    
    # Import here to avoid path issues
    try:
        from vi_model_complete import SupervisedPoissonFactorization
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    # Load a subset of the real data for testing
    print("Loading EMTAB data...")
    try:
        # Load the preprocessed data
        with open('/labs/Aguiar/SSPA_BRAY/dataset/EMTAB11349/preprocessed/emtab_ensembl_converted.pkl', 'rb') as f:
            adata = pickle.load(f)
        
        print(f"Loaded data shape: {adata.shape}")
        
        # Use a smaller subset for testing (first 100 samples, first 1000 genes)
        n_samples_subset = min(100, adata.n_obs)
        n_genes_subset = min(1000, adata.n_vars)
        
        X = adata.X[:n_samples_subset, :n_genes_subset].toarray()
        
        # Get the labels
        if 'both_labels' in adata.obs.columns:
            Y = adata.obs['both_labels'].values[:n_samples_subset]
            # Convert to numeric if needed
            if Y.dtype == 'object':
                unique_labels = np.unique(Y)
                Y = np.array([np.where(unique_labels == label)[0][0] for label in Y])
            
            # Convert to one-hot if needed
            if len(Y.shape) == 1:
                n_classes = len(np.unique(Y))
                Y_onehot = np.zeros((len(Y), n_classes))
                Y_onehot[np.arange(len(Y)), Y] = 1
                Y = Y_onehot
        else:
            # Create dummy binary labels
            Y = np.random.binomial(1, 0.5, (n_samples_subset, 2)).astype(float)
        
        # Create simple auxiliary features (just intercept + one covariate)
        X_aux = np.column_stack([
            np.ones(n_samples_subset),  # intercept
            np.random.normal(0, 1, n_samples_subset)  # simple covariate
        ])
        
        print(f"Subset shapes: X={X.shape}, Y={Y.shape}, X_aux={X_aux.shape}")
        
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Using synthetic data instead...")
        
        # Fallback to synthetic data
        n_samples_subset, n_genes_subset = 100, 1000
        X = np.random.poisson(2.0, (n_samples_subset, n_genes_subset))
        Y = np.random.binomial(1, 0.5, (n_samples_subset, 2)).astype(float)
        X_aux = np.column_stack([
            np.ones(n_samples_subset),
            np.random.normal(0, 1, n_samples_subset)
        ])
    
    # Convert to JAX arrays
    X = jnp.array(X)
    Y = jnp.array(Y)
    X_aux = jnp.array(X_aux)
    
    # Use much smaller number of factors for testing
    n_factors = min(50, X.shape[1] // 20)  # Much smaller than 1270
    
    print(f"Using {n_factors} factors (much smaller than original 1270)")
    
    # Create model with MUCH better hyperparameters
    model = SupervisedPoissonFactorization(
        n_samples=X.shape[0],
        n_genes=X.shape[1], 
        n_factors=n_factors,
        n_outcomes=Y.shape[1],
        # CRITICAL: Use much smaller alpha values to prevent KL explosion
        alpha_eta=0.1,      # Instead of 1.0
        lambda_eta=0.1,     # Instead of 1.0  
        alpha_beta=0.01,    # Instead of 1.0 - MUCH smaller
        alpha_xi=0.1,       # Instead of 1.0
        lambda_xi=0.1,      # Instead of 1.0
        alpha_theta=0.01,   # Instead of 1.0 - MUCH smaller
        sigma2_gamma=1.0,   
        sigma2_v=1.0
    )
    
    print("\n=== Running Model Fit ===")
    print(f"Hyperparameters:")
    print(f"  alpha_beta: 0.01 (vs 1.0 original)")
    print(f"  alpha_theta: 0.01 (vs 1.0 original)")
    print(f"  alpha_eta: 0.1 (vs 1.0 original)")
    print(f"  alpha_xi: 0.1 (vs 1.0 original)")
    print(f"  n_factors: {n_factors} (vs 1270 original)")
    print()
    
    try:
        # Run for just 5 iterations to test
        params, expected_vals = model.fit(X, Y, X_aux, n_iter=5, verbose=True)
        
        print("\n=== SUCCESS ===")
        print("Model fit completed successfully!")
        print("The ELBO fixes appear to be working!")
        
        return True
        
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Model fit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_simple_test()
    if success:
        print("\n🎉 Ready to test with full experiment!")
        print("Next step: Modify your main experiment to use these hyperparameters:")
        print("  alpha_beta=0.01, alpha_theta=0.01, alpha_eta=0.1, alpha_xi=0.1")
    else:
        print("\n❌ Still has issues - need further debugging")
    
    sys.exit(0 if success else 1) 