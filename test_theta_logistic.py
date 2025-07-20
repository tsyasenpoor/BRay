#!/usr/bin/env python3
"""
Test script to verify that logistic terms are working correctly in theta update.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from VariationalInference.vi_model_complete import SupervisedPoissonFactorization

def test_theta_logistic_update():
    """Test that theta update includes logistic regression terms."""
    print("=== Testing Theta Update with Logistic Terms ===")
    
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    
    # Create small synthetic data
    n_samples, n_genes, n_factors, n_outcomes = 10, 20, 5, 2
    X = random.poisson(key, 5.0, shape=(n_samples, n_genes))  # Fixed: removed duplicate shape
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
    
    # Initialize parameters
    params = model.initialize_parameters(X, Y, X_aux)
    expected_vals = model.expected_values(params)
    z = model.update_z_latent(X, expected_vals['E_theta'], expected_vals['E_beta'])
    
    print("\n--- Before theta update ---")
    print(f"theta shape: {expected_vals['E_theta'].shape}")
    print(f"theta mean: {jnp.mean(expected_vals['E_theta']):.4f}")
    print(f"theta std: {jnp.std(expected_vals['E_theta']):.4f}")
    
    # Test theta update with logistic terms
    theta_update = model.update_theta(params, expected_vals, z, Y, X_aux)
    
    print("\n--- After theta update ---")
    print(f"a_theta shape: {theta_update['a_theta'].shape}")
    print(f"b_theta shape: {theta_update['b_theta'].shape}")
    print(f"a_theta mean: {jnp.mean(theta_update['a_theta']):.4f}")
    print(f"b_theta mean: {jnp.mean(theta_update['b_theta']):.4f}")
    
    # Check that b_theta includes both Poisson and logistic terms
    # Poisson terms should be: E[xi] + sum(E[beta])
    poisson_terms = jnp.expand_dims(expected_vals['E_xi'], 1) + \
                   jnp.sum(expected_vals['E_beta'], axis=0)
    
    print(f"\n--- Term Analysis ---")
    print(f"Poisson terms mean: {jnp.mean(poisson_terms):.4f}")
    print(f"Total b_theta mean: {jnp.mean(theta_update['b_theta']):.4f}")
    print(f"Difference (logistic terms): {jnp.mean(theta_update['b_theta'] - poisson_terms):.4f}")
    
    # Verify that logistic terms are non-zero
    logistic_terms = theta_update['b_theta'] - poisson_terms
    if jnp.any(jnp.abs(logistic_terms) > 1e-10):
        print("✅ SUCCESS: Logistic terms are present in theta update!")
        print(f"Logistic terms range: {jnp.min(logistic_terms):.6f} to {jnp.max(logistic_terms):.6f}")
    else:
        print("❌ WARNING: Logistic terms appear to be zero or very small")
    
    # Test that the update produces valid Gamma parameters
    if jnp.all(theta_update['a_theta'] > 0) and jnp.all(theta_update['b_theta'] > 0):
        print("✅ SUCCESS: Valid Gamma parameters (a_theta > 0, b_theta > 0)")
    else:
        print("❌ ERROR: Invalid Gamma parameters detected")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_theta_logistic_update() 