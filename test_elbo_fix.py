#!/usr/bin/env python3

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import sys
sys.path.append('VariationalInference')

import jax.numpy as jnp
import jax.random as random
from vi_model_complete import SupervisedPoissonFactorization

def test_elbo_fix():
    """Test if the ELBO fixes work with small synthetic data"""
    print("Testing ELBO fixes with synthetic data...")
    
    # Create small synthetic dataset
    key = random.PRNGKey(42)
    n_samples, n_genes, n_factors, n_outcomes = 50, 100, 20, 2
    
    # Generate synthetic data
    X = random.poisson(key, 2.0, (n_samples, n_genes))
    Y = random.bernoulli(random.split(key)[0], 0.5, (n_samples, n_outcomes)).astype(jnp.float32)
    X_aux = random.normal(random.split(key)[1], (n_samples, 2))
    
    print(f"Data shapes: X={X.shape}, Y={Y.shape}, X_aux={X_aux.shape}")
    
    # Create model with reasonable hyperparameters
    model = SupervisedPoissonFactorization(
        n_samples=n_samples, 
        n_genes=n_genes, 
        n_factors=n_factors, 
        n_outcomes=n_outcomes,
        alpha_eta=1.0,      # Smaller values for stability
        lambda_eta=1.0,
        alpha_beta=0.1,     # Smaller alpha to reduce KL explosion
        alpha_xi=1.0,
        lambda_xi=1.0, 
        alpha_theta=0.1,    # Smaller alpha to reduce KL explosion
        sigma2_gamma=1.0,
        sigma2_v=1.0
    )
    
    print("Running model fit for 3 iterations...")
    try:
        params, expected_vals = model.fit(X, Y, X_aux, n_iter=3, verbose=True)
        print("SUCCESS: Model fit completed without errors!")
        return True
    except Exception as e:
        print(f"ERROR: Model fit failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_elbo_fix()
    sys.exit(0 if success else 1) 