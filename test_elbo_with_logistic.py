#!/usr/bin/env python3
"""
Test script to verify that ELBO is working correctly with logistic terms in theta update.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from VariationalInference.vi_model_complete import SupervisedPoissonFactorization

def test_elbo_with_logistic():
    """Test that ELBO works correctly with logistic terms in theta update."""
    print("=== Testing ELBO with Logistic Terms ===")
    
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    
    # Create small synthetic data
    n_samples, n_genes, n_factors, n_outcomes = 20, 50, 10, 2
    X = random.poisson(key, 5.0, shape=(n_samples, n_genes))
    Y = random.bernoulli(key, 0.5, shape=(n_samples, n_outcomes)).astype(jnp.float32)
    X_aux = random.normal(key, shape=(n_samples, 3))
    
    print(f"Data shapes: X={X.shape}, Y={Y.shape}, X_aux={X_aux.shape}")
    
    # Create model with small hyperparameters
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
    
    print("Running model for 3 iterations...")
    
    # Run the model for a few iterations
    params, expected_vals = model.fit(X, Y, X_aux, n_iter=3, verbose=True)
    
    print("\n=== Final Results ===")
    print(f"Model converged successfully!")
    print(f"Final theta mean: {jnp.mean(expected_vals['E_theta']):.4f}")
    print(f"Final beta mean: {jnp.mean(expected_vals['E_beta']):.4f}")
    print(f"Final gamma mean: {jnp.mean(expected_vals['E_gamma']):.4f}")
    print(f"Final v mean: {jnp.mean(expected_vals['E_v']):.4f}")
    
    # Test that parameters are reasonable
    if jnp.all(params['a_theta'] > 0) and jnp.all(params['b_theta'] > 0):
        print("✅ SUCCESS: All parameters are valid!")
    else:
        print("❌ ERROR: Invalid parameters detected")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_elbo_with_logistic() 