#!/usr/bin/env python3
"""
Test script to verify that the Poisson likelihood fix works and ELBO is reasonable.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from VariationalInference.vi_model_complete import SupervisedPoissonFactorization

def test_elbo_poisson_fix():
    """Test that ELBO is reasonable after Poisson likelihood fix."""
    print("=== Testing ELBO Poisson Fix ===")
    
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    
    # Create small synthetic data
    n_samples, n_genes, n_factors, n_outcomes = 10, 20, 5, 2
    X = random.poisson(key, 5.0, shape=(n_samples, n_genes))
    Y = random.bernoulli(key, 0.5, shape=(n_samples, n_outcomes)).astype(jnp.float32)
    X_aux = random.normal(key, shape=(n_samples, 2))
    
    print(f"Data shapes: X={X.shape}, Y={Y.shape}, X_aux={X_aux.shape}")
    print(f"X range: {X.min()} to {X.max()}, mean: {X.mean():.2f}")
    print(f"Total counts in X: {X.sum()}")
    
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

    hyperparams = {
        "alpha_eta": 0.1, "lambda_eta": 3.0,
        "alpha_beta": 0.01,
        "alpha_xi": 0.1, "lambda_xi": 3.0,
        "alpha_theta": 0.01,
        "sigma2_v": 1.0, "sigma2_gamma": 1.0
    }
    
    # Initialize parameters
    params = model.initialize_parameters(X, Y, X_aux)
    
    # Initialize latent variables
    expected_vals = model.expected_values(params)
    z = model.update_z_latent(X, expected_vals['E_theta'], expected_vals['E_beta'])
    
    print(f"\nLatent z shape: {z.shape}")
    print(f"z range: {z.min():.2f} to {z.max():.2f}, mean: {z.mean():.2f}")
    print(f"Total z: {z.sum():.2f}")
    print(f"Sum of z should equal sum of X: {abs(z.sum() - X.sum()):.2e}")
    
    # Check expected values
    E_theta = expected_vals['E_theta']
    E_beta = expected_vals['E_beta']
    print(f"\nE_theta range: {E_theta.min():.2f} to {E_theta.max():.2f}")
    print(f"E_beta range: {E_beta.min():.2f} to {E_beta.max():.2f}")
    
    # Compute ELBO with components
    elbo_result = model.compute_elbo(X, Y, X_aux, params, z, return_components=True, debug_print=True)
    
    if isinstance(elbo_result, dict):
        print("\n=== ELBO Components ===")
        for key, value in elbo_result.items():
            print(f"{key:12}: {value:12.2f}")
        total_elbo = elbo_result['elbo']
    else:
        print(f"\nELBO: {elbo_result:.2f}")
        total_elbo = elbo_result
    
    print(f"\nTotal ELBO: {total_elbo:.2f}")
    
    # Check if ELBO is reasonable (should be negative and not too large)
    if total_elbo > 0:
        print("❌ ERROR: ELBO is positive! This is wrong!")
        return False
    elif total_elbo < -1e6:
        print("❌ WARNING: ELBO is extremely negative! This might indicate an issue.")
        return False
    else:
        print("✅ SUCCESS: ELBO is reasonable (negative and not too large)")
        return True

if __name__ == "__main__":
    success = test_elbo_poisson_fix()
    if success:
        print("\n🎉 Poisson likelihood fix is working!")
    else:
        print("\n💥 Poisson likelihood fix failed!") 