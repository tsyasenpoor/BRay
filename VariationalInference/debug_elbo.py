#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import json
from datetime import datetime
import argparse
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from vi_model_complete import (initialize_q_params, run_variational_inference,
                               evaluate_model, extract_top_genes, compute_elbo)

def clear_memory():
    gc.collect()
    if hasattr(jax, 'clear_caches'):
        jax.clear_caches()
    elif hasattr(jax.lib, 'xla_bridge'):
        jax.lib.xla_bridge.get_backend().clear_cache()

def debug_elbo_calculation(x_data, y_data, x_aux, hyperparams, mask=None, max_iters=3, seed=42):
    """
    Run a small number of VI iterations and closely monitor the ELBO calculation
    to identify numerical issues.
    """
    print(f"Debug ELBO calculation with d={hyperparams['d']}")
    print(f"Data shapes: x_data={x_data.shape}, y_data={y_data.shape}, x_aux={x_aux.shape}")
    
    # Use a smaller subset for debugging
    if x_data.shape[0] > 500:
        print("Using smaller subset of data for debugging")
        x_data_subset, _, y_data_subset, _, x_aux_subset, _ = train_test_split(
            x_data, y_data, x_aux, test_size=0.9, random_state=seed
        )
        x_data = x_data_subset
        y_data = y_data_subset
        x_aux = x_aux_subset
        print(f"Subset shapes: x_data={x_data.shape}, y_data={y_data.shape}, x_aux={x_aux.shape}")
    
    if mask is not None:
        print(f"Using mask with shape {mask.shape}")
        non_zero = np.count_nonzero(mask)
        total = mask.size
        print(f"Mask sparsity: {100 * (1 - non_zero/total):.2f}% ({non_zero} non-zeros of {total} total)")
    
    # Initialize parameters
    n, p = x_data.shape
    kappa = y_data.shape[1] if len(y_data.shape) > 1 else 1
    p_aux = x_aux.shape[1]
    d = hyperparams['d']
    
    print(f"Initializing q_params with n={n}, p={p}, kappa={kappa}, p_aux={p_aux}, d={d}")
    q_params = initialize_q_params(n, p, kappa, p_aux, d, seed=seed)
    
    elbo_history = []
    elbo_components = {
        'iter': [],
        'E_log_p_eta': [],
        'E_log_p_beta': [],
        'E_log_p_x': [],
        'E_log_p_xi': [],
        'E_log_p_theta': [],
        'E_log_p_y': [],
        'E_log_p_gamma': [],
        'E_log_p_upsilon': [],
        'E_log_q_eta': [],
        'E_log_q_beta': [],
        'E_log_q_xi': [],
        'E_log_q_theta': [],
        'E_log_q_gamma': [],
        'E_log_q_upsilon': [],
        'expected_log_joint': [],
        'expected_log_q': [],
        'elbo': []
    }
    
    # Run basic checks on the initialization
    print("\nChecking initial parameter values:")
    for param_name, param_val in q_params.items():
        if isinstance(param_val, jnp.ndarray):
            print(f"  {param_name}: shape={param_val.shape}, min={jnp.min(param_val):.6e}, max={jnp.max(param_val):.6e}")
            # Check for NaN or Inf values
            if jnp.any(jnp.isnan(param_val)):
                print(f"  WARNING: {param_name} contains NaN values!")
            if jnp.any(jnp.isinf(param_val)):
                print(f"  WARNING: {param_name} contains Inf values!")
            
    # Add custom monitoring for ELBO components
    def debug_compute_elbo(E_eta, E_beta, E_xi, E_theta, E_gamma, E_upsilon,
                      x_data, y_data, x_aux, hyperparams, q_params, iter_num):
        print(f"\n--- ELBO Calculation for Iteration {iter_num+1} ---")
        
        try:
            if len(y_data.shape) == 1:
                y_data = y_data.reshape(-1, 1)
            if len(x_aux.shape) == 1:
                x_aux = x_aux.reshape(-1, 1)

            c_prime = hyperparams['c_prime']
            d_prime = hyperparams['d_prime']
            c       = hyperparams['c']
            a_prime = hyperparams['a_prime']
            b_prime = hyperparams['b_prime']
            a       = hyperparams['a']
            tau     = hyperparams['tau']
            sigma   = hyperparams['sigma']

            E_eta_flat = E_eta.flatten()
            E_xi_flat  = E_xi.flatten()
            
            print(f"E_eta range: {jnp.min(E_eta_flat):.6e} to {jnp.max(E_eta_flat):.6e}")
            print(f"E_beta range: {jnp.min(E_beta):.6e} to {jnp.max(E_beta):.6e}")
            print(f"E_xi range: {jnp.min(E_xi_flat):.6e} to {jnp.max(E_xi_flat):.6e}")
            print(f"E_theta range: {jnp.min(E_theta):.6e} to {jnp.max(E_theta):.6e}")
            print(f"E_gamma range: {jnp.min(E_gamma):.6e} to {jnp.max(E_gamma):.6e}")
            print(f"E_upsilon range: {jnp.min(E_upsilon):.6e} to {jnp.max(E_upsilon):.6e}")
            
            # Prior distributions
            E_log_p_eta = jnp.sum(
                c_prime * jnp.log(d_prime)
                - jax.scipy.special.gammaln(c_prime)
                + (c_prime - 1) * jnp.log(E_eta_flat + 1e-10)
                - d_prime * E_eta_flat
            )
            
            E_eta_col = E_eta.reshape(-1, 1)
            if E_eta_col.shape[1] == 1 and E_beta.shape[1] > 1:
                E_eta_broadcast = jnp.tile(E_eta_col, (1, E_beta.shape[1]))
            else:
                E_eta_broadcast = E_eta_col
            E_log_p_beta = jnp.sum(
                c * jnp.log(E_eta_broadcast + 1e-10)
                - jax.scipy.special.gammaln(c)
                + (c - 1) * jnp.log(E_beta + 1e-10)
                - E_eta_broadcast * E_beta
            )
            
            rate = jnp.dot(E_theta, E_beta.T) + 1e-10
            E_log_p_x = jnp.sum(
                x_data * jnp.log(rate) - rate - jax.scipy.special.gammaln(x_data + 1)
            )
            
            E_log_p_xi = jnp.sum(
                a_prime * jnp.log(b_prime)
                - jax.scipy.special.gammaln(a_prime)
                + (a_prime - 1) * jnp.log(E_xi_flat + 1e-10)
                - b_prime * E_xi_flat
            )
            
            E_xi_col = E_xi.reshape(-1, 1)
            if E_xi_col.shape[1] == 1 and E_theta.shape[1] > 1:
                E_xi_broadcast = jnp.tile(E_xi_col, (1, E_theta.shape[1]))
            else:
                E_xi_broadcast = E_xi_col
            E_log_p_theta = jnp.sum(
                a * jnp.log(E_xi_broadcast + 1e-10)
                - jax.scipy.special.gammaln(a)
                + (a - 1) * jnp.log(E_theta + 1e-10)
                - E_xi_broadcast * E_theta
            )
            
            logits = (x_aux @ E_gamma.T) + jnp.einsum('nd,kd->nk', E_theta, E_upsilon)
            print(f"logits range: {jnp.min(logits):.6e} to {jnp.max(logits):.6e}")
            if jnp.any(jnp.isnan(logits)) or jnp.any(jnp.isinf(logits)):
                print("WARNING: logits contain NaN or Inf values!")
                
            probs = jax.nn.sigmoid(logits)
            print(f"probs range: {jnp.min(probs):.6e} to {jnp.max(probs):.6e}")
            
            # Add safety checks for extreme probabilities
            extreme_probs_near_0 = jnp.sum(probs < 1e-10)
            extreme_probs_near_1 = jnp.sum(probs > 1-1e-10)
            if extreme_probs_near_0 > 0 or extreme_probs_near_1 > 0:
                print(f"WARNING: Found {extreme_probs_near_0} probabilities near 0 and {extreme_probs_near_1} near 1")
                # Clip probabilities to avoid numerical issues in log
                probs = jnp.clip(probs, 1e-10, 1-1e-10)
                
            E_log_p_y = jnp.sum(
                y_data * jnp.log(probs + 1e-10)
                + (1 - y_data) * jnp.log(1 - probs + 1e-10)
            )
            
            E_log_p_gamma = -0.5 * jnp.sum(E_gamma**2) / sigma**2
            # Laplace prior on upsilon for sparsity
            E_log_p_upsilon = -jnp.sum(jnp.abs(E_upsilon)) / tau - E_upsilon.size * jnp.log(2 * tau)
            
            # Variational distributions
            alpha_eta = q_params['alpha_eta']
            omega_eta = q_params['omega_eta']
            E_log_q_eta = jnp.sum(
                alpha_eta * jnp.log(omega_eta+1e-10)
                - jax.scipy.special.gammaln(alpha_eta)
                + (alpha_eta - 1) * (jax.scipy.special.digamma(alpha_eta) - jnp.log(omega_eta+1e-10))
                - omega_eta * E_eta_flat
            )
            
            alpha_beta = q_params['alpha_beta']
            omega_beta = q_params['omega_beta']
            E_log_q_beta = jnp.sum(
                alpha_beta * jnp.log(omega_beta+1e-10)
                - jax.scipy.special.gammaln(alpha_beta)
                + (alpha_beta - 1) * (jax.scipy.special.digamma(alpha_beta) - jnp.log(omega_beta+1e-10))
                - omega_beta * E_beta
            )
            
            alpha_xi = q_params['alpha_xi']
            omega_xi = q_params['omega_xi']
            E_log_q_xi = jnp.sum(
                alpha_xi * jnp.log(omega_xi+1e-10)
                - jax.scipy.special.gammaln(alpha_xi)
                + (alpha_xi - 1) * (jax.scipy.special.digamma(alpha_xi) - jnp.log(omega_xi+1e-10))
                - omega_xi * E_xi_flat
            )
            
            alpha_theta = q_params['alpha_theta']
            omega_theta = q_params['omega_theta']
            E_log_q_theta = jnp.sum(
                alpha_theta * jnp.log(omega_theta+1e-10)
                - jax.scipy.special.gammaln(alpha_theta)
                + (alpha_theta - 1) * (jax.scipy.special.digamma(alpha_theta) - jnp.log(omega_theta+1e-10))
                - omega_theta * E_theta
            )
            
            kappa, p_aux = E_gamma.shape
            sigma_gamma_inv = q_params['sigma_gamma_inv']
            E_log_q_gamma = 0
            for kk in range(kappa):
                Sigma_gamma_k_inv = sigma_gamma_inv[kk] + jnp.eye(p_aux)*1e-8
                # Use more numerically stable approach for log det
                log_det = jnp.linalg.slogdet(jnp.linalg.inv(Sigma_gamma_k_inv))
                sign, logdet_val = log_det
                if sign < 0:
                    print(f"WARNING: Negative determinant for Sigma_gamma_inv[{kk}]!")
                log_det_term = -0.5 * logdet_val
                E_log_q_gamma += (log_det_term - 0.5 * p_aux*(1 + jnp.log(2*jnp.pi)))
            
            kappa2, d = E_upsilon.shape
            sigma_upsilon_inv = q_params['sigma_upsilon_inv']
            E_log_q_upsilon = 0
            for kk in range(kappa2):
                Sigma_ups_k_inv = sigma_upsilon_inv[kk] + jnp.eye(d)*1e-8
                # Use more numerically stable approach for log det
                log_det = jnp.linalg.slogdet(jnp.linalg.inv(Sigma_ups_k_inv))
                sign, logdet_val = log_det
                if sign < 0:
                    print(f"WARNING: Negative determinant for Sigma_upsilon_inv[{kk}]!")
                log_det_term = -0.5 * logdet_val
                E_log_q_upsilon += (log_det_term - 0.5 * d*(1 + jnp.log(2*jnp.pi)))
            
            expected_log_joint = (E_log_p_eta + E_log_p_beta + E_log_p_x + E_log_p_xi
                              + E_log_p_theta + E_log_p_y + E_log_p_gamma + E_log_p_upsilon)
            expected_log_q = (E_log_q_eta + E_log_q_beta + E_log_q_xi 
                          + E_log_q_theta + E_log_q_gamma + E_log_q_upsilon)
            elbo = expected_log_joint - expected_log_q
            
            # Print summary of components
            print(f"E_log_p_eta: {E_log_p_eta:.6e}")
            print(f"E_log_p_beta: {E_log_p_beta:.6e}")
            print(f"E_log_p_x: {E_log_p_x:.6e}")
            print(f"E_log_p_xi: {E_log_p_xi:.6e}")
            print(f"E_log_p_theta: {E_log_p_theta:.6e}")
            print(f"E_log_p_y: {E_log_p_y:.6e}")
            print(f"E_log_p_gamma: {E_log_p_gamma:.6e}")
            print(f"E_log_p_upsilon: {E_log_p_upsilon:.6e}")
            print(f"E_log_q_eta: {E_log_q_eta:.6e}")
            print(f"E_log_q_beta: {E_log_q_beta:.6e}")
            print(f"E_log_q_xi: {E_log_q_xi:.6e}")
            print(f"E_log_q_theta: {E_log_q_theta:.6e}")
            print(f"E_log_q_gamma: {E_log_q_gamma:.6e}")
            print(f"E_log_q_upsilon: {E_log_q_upsilon:.6e}")
            print(f"Expected log joint: {expected_log_joint:.6e}")
            print(f"Expected log q: {expected_log_q:.6e}")
            print(f"ELBO: {elbo:.6e}")
            
            # Record components
            elbo_components['iter'].append(iter_num)
            elbo_components['E_log_p_eta'].append(float(E_log_p_eta))
            elbo_components['E_log_p_beta'].append(float(E_log_p_beta))
            elbo_components['E_log_p_x'].append(float(E_log_p_x))
            elbo_components['E_log_p_xi'].append(float(E_log_p_xi))
            elbo_components['E_log_p_theta'].append(float(E_log_p_theta))
            elbo_components['E_log_p_y'].append(float(E_log_p_y))
            elbo_components['E_log_p_gamma'].append(float(E_log_p_gamma))
            elbo_components['E_log_p_upsilon'].append(float(E_log_p_upsilon))
            elbo_components['E_log_q_eta'].append(float(E_log_q_eta))
            elbo_components['E_log_q_beta'].append(float(E_log_q_beta))
            elbo_components['E_log_q_xi'].append(float(E_log_q_xi))
            elbo_components['E_log_q_theta'].append(float(E_log_q_theta))
            elbo_components['E_log_q_gamma'].append(float(E_log_q_gamma))
            elbo_components['E_log_q_upsilon'].append(float(E_log_q_upsilon))
            elbo_components['expected_log_joint'].append(float(expected_log_joint))
            elbo_components['expected_log_q'].append(float(expected_log_q))
            elbo_components['elbo'].append(float(elbo))
            
            return elbo
            
        except Exception as e:
            print(f"Exception in ELBO calculation: {e}")
            return float('nan')
    
    # Override the standard ELBO computation for debugging
    def debug_run_variation_inference():
        for iter_ix in range(max_iters):
            print(f"\n=== Starting VI iteration {iter_ix+1}/{max_iters} ===")
            
            # Update parameters
            q_params_new = run_variational_inference.__globals__['update_q_params'](
                q_params, x_data, y_data, x_aux, hyperparams, 
                label_scale=0.1, mask=mask
            )
            
            # Check updated parameters
            print("\nChecking updated parameter values:")
            for param_name, param_val in q_params_new.items():
                if isinstance(param_val, jnp.ndarray):
                    print(f"  {param_name}: min={jnp.min(param_val):.6e}, max={jnp.max(param_val):.6e}")
                    # Check for NaN or Inf values
                    if jnp.any(jnp.isnan(param_val)):
                        print(f"  WARNING: {param_name} contains NaN values!")
                        # Print the indices of NaN values
                        nan_indices = jnp.where(jnp.isnan(param_val))
                        print(f"  NaN indices: {nan_indices}")
                        # Print surrounding values if possible
                        for idx in zip(*nan_indices):
                            idx_array = np.array(idx)
                            print(f"  Value at {idx}: {param_val[idx]}")
                            # Try to get surrounding values
                            try:
                                for offset in [[-1], [1]]:
                                    adj_idx = tuple(idx_array + offset)
                                    if all(i >= 0 and i < d for i, d in zip(adj_idx, param_val.shape)):
                                        print(f"  Adjacent value at {adj_idx}: {param_val[adj_idx]}")
                            except:
                                pass
                            
                    if jnp.any(jnp.isinf(param_val)):
                        print(f"  WARNING: {param_name} contains Inf values!")
            
            # Update current parameters
            q_params.update(q_params_new)
            
            # Compute expectations
            E_eta = q_params['alpha_eta']/jnp.maximum(q_params['omega_eta'],1e-10)
            E_beta = q_params['alpha_beta']/jnp.maximum(q_params['omega_beta'],1e-10)
            if mask is not None:
                E_beta = E_beta * mask
            E_xi = q_params['alpha_xi']/jnp.maximum(q_params['omega_xi'],1e-10)
            E_theta = q_params['alpha_theta']/jnp.maximum(q_params['omega_theta'],1e-10)
            E_gamma = q_params['gamma']
            E_upsilon = q_params['upsilon']
            
            # Calculate ELBO
            elbo_val = debug_compute_elbo(
                E_eta, E_beta, E_xi, E_theta, E_gamma, E_upsilon,
                x_data, y_data, x_aux, hyperparams, q_params, iter_ix
            )
            
            elbo_history.append(float(elbo_val))
            print(f"Iteration {iter_ix+1} ELBO: {elbo_val:.6e}")
            
            # Check for NaN
            if jnp.isnan(elbo_val):
                print("WARNING: ELBO is NaN! Stopping early.")
                break
                
            # Clear memory
            clear_memory()
    
    # Run debugging VI
    debug_run_variation_inference()
    
    # Plot ELBO history and components
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(elbo_history, 'o-')
    plt.title('ELBO History')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.grid(True)
    
    # Plot ELBO components that contribute to bouncing
    plt.subplot(1, 2, 2)
    components_df = pd.DataFrame(elbo_components)
    if len(components_df) > 0:
        # Find components with largest variance
        numeric_cols = components_df.select_dtypes(include=[np.number]).columns
        # Calculate variance * scale factor to identify the most important components
        variances = components_df[numeric_cols].var() * components_df[numeric_cols].abs().mean()
        top_components = variances.nlargest(5).index.tolist()
        
        for component in top_components:
            if component != 'iter' and component in components_df.columns:
                plt.plot(components_df['iter'], components_df[component], 'o-', label=component)
        
        plt.title('Top Variable ELBO Components')
        plt.xlabel('Iteration')
        plt.ylabel('Component Value')
        plt.legend()
        plt.grid(True)
    
    # Save plot
    plot_filename = f"elbo_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_filename)
    print(f"Saved ELBO diagnostic plot to {plot_filename}")
    
    # Save detailed component data for analysis
    json_filename = f"elbo_components_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_filename, 'w') as f:
        # Filter out NaN values for JSON serialization
        clean_components = {k: [float(v) if not np.isnan(v) and not np.isinf(v) else "NaN" for v in vals] 
                           for k, vals in elbo_components.items() if k != 'iter'}
        clean_components['iter'] = elbo_components['iter']
        json.dump(clean_components, f, indent=2)
    print(f"Saved ELBO components to {json_filename}")
    
    return elbo_history, elbo_components

def main():
    parser = argparse.ArgumentParser(description="Debug ELBO calculation in variational inference model")
    parser.add_argument("--d", type=int, default=10, help="Dimension of latent space")
    parser.add_argument("--max_iter", type=int, default=5, help="Maximum iterations for variational inference")
    parser.add_argument("--data_file", type=str, help="Path to AnnData h5ad file", required=True)
    parser.add_argument("--use_mask", action="store_true", help="Use masking")
    args = parser.parse_args()
    
    print("Loading data...")
    import anndata as ad
    adata = ad.read_h5ad(args.data_file)
    
    print(f"Data loaded: {adata.shape[0]} observations, {adata.shape[1]} variables")
    
    # Use the cytokine label if available
    if 'cyto' in adata.obs.columns:
        y_data = adata.obs['cyto'].astype(float).values.reshape(-1, 1)
        print(f"Using 'cyto' as label, label counts: {pd.Series(y_data.flatten()).value_counts()}")
    else:
        # Use a synthetic label for testing
        print("No 'cyto' label found, creating synthetic label")
        y_data = np.random.binomial(1, 0.5, adata.shape[0]).reshape(-1, 1)
    
    # Convert data to dense format if needed
    X_data = adata.X
    if hasattr(X_data, 'toarray'):
        X_data = X_data.toarray()
    
    # Simple auxiliary data
    x_aux = np.ones((X_data.shape[0], 1))
    
    # Create mask if requested
    mask = None
    if args.use_mask:
        print("Creating random mask for debugging")
        p = X_data.shape[1]
        d = args.d
        mask = np.zeros((p, d))
        # Assign each gene to at least one program
        for i in range(p):
            mask[i, np.random.randint(0, d)] = 1
        print(f"Mask shape: {mask.shape}, non-zeros: {np.count_nonzero(mask)}")
    
    # Set hyperparameters
    hyperparams = {
        "c_prime": 2.0,  "d_prime": 3.0,
        "c":      0.6,
        "a_prime":2.0,   "b_prime": 3.0,
        "a":      0.6,
        "tau":    4.0,   "sigma":   4.0,
        "d":      args.d
    }
    
    # Run debugging
    elbo_history, elbo_components = debug_elbo_calculation(
        X_data, y_data, x_aux, hyperparams, mask=mask, max_iters=args.max_iter
    )
    
    print("\n=== ELBO History ===")
    for i, elbo in enumerate(elbo_history):
        print(f"Iteration {i+1}: {elbo:.6e}")
        
    if any(np.isnan(elbo) for elbo in elbo_history):
        print("\nWARNING: NaN values detected in ELBO history!")
        print("Possible causes:")
        print("1. Numerical instability in ELBO calculation")
        print("2. Very small or large values in parameter updates")
        print("3. Matrix inversion issues in covariance calculations")
        print("\nRecommended fixes:")
        print("1. Increase numerical stability offsets (currently using 1e-10)")
        print("2. Add parameter clipping to prevent extreme values")
        print("3. Use more stable methods for matrix operations")
    
    bouncing = False
    if len(elbo_history) >= 3:
        diffs = np.diff(elbo_history)
        if any(d < 0 for d in diffs):
            bouncing = True
            print("\nWARNING: ELBO is decreasing in some iterations!")
            print("Possible causes:")
            print("1. Learning rate too high - Newton method may be overshooting")
            print("2. Poor conditioning in the Hessian matrix for gamma/upsilon")
            print("3. Conflicts between optimization of different parameters")
            print("\nRecommended fixes:")
            print("1. Reduce learning rate in Newton method (currently 0.5)")
            print("2. Add stronger regularization to Hessian matrices")
            print("3. Use coordinate ascent with smaller steps")
    
    print("\nDebug analysis complete. See generated plots for more details.")

if __name__ == "__main__":
    main()