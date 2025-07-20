#!/usr/bin/env python3
"""
Debug script to check the actual shapes of tau2_gamma and other parameters.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from VariationalInference.vi_model_complete import SupervisedPoissonFactorization

def debug_shapes():
    """Debug the shapes of parameters to understand the einsum issue."""
    print("=== Debugging Parameter Shapes ===")
    
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    
    # Create small synthetic data
    n_samples, n_genes, n_factors, n_outcomes = 10, 20, 5, 2
    X = random.poisson(key, 5.0, shape=(n_samples, n_genes))
    Y = random.bernoulli(key, 0.5, shape=(n_samples, n_outcomes)).astype(jnp.float32)
    X_aux = random.normal(key, shape=(n_samples, 2))
    
    print(f"Data shapes:")
    print(f"  X: {X.shape}")
    print(f"  Y: {Y.shape}")
    print(f"  X_aux: {X_aux.shape}")
    
    # Create model
    model = SupervisedPoissonFactorization(
        n_samples=n_samples,
        n_genes=n_genes, 
        n_factors=n_factors,
        n_outcomes=n_outcomes,
        alpha_beta=0.01,
        alpha_theta=0.01,
        alpha_eta=0.1,
        alpha_xi=0.1,
        key=key
    )
    
    print(f"\nModel parameters:")
    print(f"  n_samples: {model.n}")
    print(f"  n_genes: {model.p}")
    print(f"  n_factors: {model.K}")
    print(f"  n_outcomes: {model.kappa}")
    
    # Initialize parameters
    params = model.initialize_parameters(X, Y, X_aux)
    
    print(f"\nParameter shapes:")
    for key, value in params.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)} (scalar)")
    
    # Check specific shapes
    print(f"\nDetailed shape analysis:")
    print(f"  tau2_gamma shape: {params['tau2_gamma'].shape}")
    print(f"  tau2_gamma type: {type(params['tau2_gamma'])}")
    print(f"  tau2_gamma dtype: {params['tau2_gamma'].dtype}")
    print(f"  tau2_gamma min/max: {params['tau2_gamma'].min():.4f}, {params['tau2_gamma'].max():.4f}")
    
    # Try the einsum operation
    print(f"\nTesting einsum operation:")
    try:
        result = jnp.einsum('id,kd->ik', X_aux**2, params['tau2_gamma'])
        print(f"  Einsum successful! Result shape: {result.shape}")
    except Exception as e:
        print(f"  Einsum failed: {e}")
        print(f"  X_aux**2 shape: {X_aux**2.shape}")
        print(f"  tau2_gamma shape: {params['tau2_gamma'].shape}")
        
        # Try alternative approaches
        print(f"\nTrying alternative approaches:")
        try:
            # Method 1: Direct multiplication
            result1 = (X_aux**2) @ params['tau2_gamma'].T
            print(f"  Method 1 (X_aux**2 @ tau2_gamma.T): {result1.shape}")
        except Exception as e1:
            print(f"  Method 1 failed: {e1}")
        
        try:
            # Method 2: Manual loop
            result2 = jnp.zeros((X_aux.shape[0], params['tau2_gamma'].shape[0]))
            for i in range(X_aux.shape[0]):
                for k in range(params['tau2_gamma'].shape[0]):
                    result2 = result2.at[i, k].set(jnp.sum(X_aux[i]**2 * params['tau2_gamma'][k]))
            print(f"  Method 2 (manual loop): {result2.shape}")
        except Exception as e2:
            print(f"  Method 2 failed: {e2}")

if __name__ == "__main__":
    debug_shapes() 