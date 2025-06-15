import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from scipy.special import expit
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, confusion_matrix, roc_auc_score, 
                            classification_report, roc_curve, auc)
from sklearn.model_selection import train_test_split
from functools import partial
import gc

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit, vmap, random

# Import memory tracking functions
from memory_tracking import get_memory_usage, log_memory, log_array_sizes, clear_memory

# Import necessary functions from vi_model_complete
from vi_model_complete import (neg_logpost_gamma, make_gamma_grad_and_hess, newton_step_gamma, newton_map_gamma,
                              neg_logpost_upsilon, make_upsilon_grad_and_hess, newton_step_upsilon, newton_map_upsilon, 
                              compute_elbo, fold_in_new_data, evaluate_model, extract_top_genes)

jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'cpu')

@jax.jit
def process_phi_batch(theta_batch, beta, x_batch):
    """Process a minibatch for phi computation."""
    numerator_phi_batch = jnp.einsum('bd,pd->bpd', theta_batch, beta)
    denom_phi_batch = jnp.sum(numerator_phi_batch, axis=2, keepdims=True) + 1e-10
    phi_batch = numerator_phi_batch / denom_phi_batch
    alpha_beta_batch_update = jnp.sum(x_batch[:, :, None] * phi_batch, axis=0)
    alpha_theta_batch_update = jnp.sum(x_batch[:, :, None] * phi_batch, axis=1)
    return alpha_beta_batch_update, alpha_theta_batch_update


def initialize_q_params(n, p, kappa, p_aux, d, seed=None, beta_init=None):
    """Initialize variational parameters."""
    log_memory(f"Before initialize_q_params (n={n}, p={p}, kappa={kappa}, p_aux={p_aux}, d={d})")
    
    if seed is None:
        random_int = jax.random.randint(jax.random.PRNGKey(0), (), 0, 2**30, dtype=jnp.int32)
        key = jax.random.PRNGKey(int(random_int))
    else:
        key = jax.random.PRNGKey(seed)

    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    
    alpha_eta = jnp.ones(p) + 0.01 * jax.random.normal(k1, (p,))
    
    # Initialize alpha_beta and omega_beta from beta_init if provided
    if beta_init is not None:
        print(f"Initializing alpha_beta from provided beta_init with shape {beta_init.shape}")
        omega_beta = jnp.ones((p, d))
        alpha_beta = beta_init * omega_beta
    else:
        alpha_beta = jnp.ones((p, d)) + 0.01 * jax.random.normal(k2, (p, d))
        omega_beta = jnp.ones((p, d))
    
    alpha_xi = jnp.ones(n) + 0.01 * jax.random.normal(k3, (n,))
    alpha_theta = jnp.ones((n, d)) + 0.01 * jax.random.normal(k4, (n, d))
    
    omega_eta = jnp.ones(p)
    omega_xi = jnp.ones(n)
    omega_theta = jnp.ones((n, d))

    gamma_init = jnp.array(jax.random.normal(k5, (kappa, p_aux))) * 0.01
    upsilon_init = jnp.array(jax.random.normal(k6, (kappa, d))) * 0.1

    sigma_gamma_inv_init = jnp.array([jnp.eye(p_aux) for _ in range(kappa)])
    sigma_upsilon_inv_init = jnp.array([jnp.eye(d) for _ in range(kappa)])
    
    q_params = {
        "alpha_eta": alpha_eta,
        "omega_eta": omega_eta,
        "alpha_beta": alpha_beta,
        "omega_beta": omega_beta,
        "alpha_xi": alpha_xi,
        "omega_xi": omega_xi,
        "alpha_theta": alpha_theta,
        "omega_theta": omega_theta,
        "gamma": gamma_init,
        "sigma_gamma_inv": sigma_gamma_inv_init,
        "upsilon": upsilon_init,
        "sigma_upsilon_inv": sigma_upsilon_inv_init,
        "iter_count": 0,
        "batch_schedule": None
    }
    
    return q_params


def update_svi_batch_params(q_params, x_batch, y_batch, x_aux_batch, hyperparams, 
                           full_data_size, minibatch_size, mask=None, label_scale=0.1):
    """Update variational parameters using a minibatch."""
    x_batch = jnp.array(x_batch)
    y_batch = jnp.array(y_batch)
    x_aux_batch = jnp.array(x_aux_batch)
    
    if len(y_batch.shape) == 1:
        y_batch = y_batch.reshape(-1, 1)
    
    if len(x_aux_batch.shape) == 1:
        x_aux_batch = x_aux_batch.reshape(-1, 1)

    batch_shape = x_batch.shape
    batch_size = batch_shape[0]
    
    # Early handling of mask to avoid redundant checks
    using_mask = mask is not None

    # Get current parameters
    alpha_eta_old = q_params['alpha_eta']
    omega_eta_old = q_params['omega_eta']
    alpha_beta_old = q_params['alpha_beta']
    omega_beta_old = q_params['omega_beta']
    alpha_xi_old = q_params['alpha_xi']
    omega_xi_old = q_params['omega_xi']
    alpha_theta_old = q_params['alpha_theta']
    omega_theta_old = q_params['omega_theta']
    gamma_old = q_params['gamma']
    upsilon_old = q_params['upsilon']
    sigma_gamma_inv_old = q_params['sigma_gamma_inv']
    sigma_upsilon_inv_old = q_params['sigma_upsilon_inv']
    
    # Get hyperparameters
    c_prime = hyperparams['c_prime']
    d_prime = hyperparams['d_prime']
    c = hyperparams['c']
    a_prime = hyperparams['a_prime']
    b_prime = hyperparams['b_prime']
    a = hyperparams['a']
    tau = hyperparams['tau']
    sigma = hyperparams['sigma']
    d = hyperparams['d']
    
    # Calculate scaling factor to convert local updates to global estimates
    scaling_factor = full_data_size / minibatch_size
    
    # Get the number of output classes
    kappa = y_batch.shape[1]
    
    # Compute current expectations for this batch
    E_theta_batch = alpha_theta_old[q_params['batch_indices']] / jnp.maximum(omega_theta_old[q_params['batch_indices']], 1e-10)
    E_beta_old = alpha_beta_old / jnp.maximum(omega_beta_old, 1e-10)
    
    # Apply mask to E_beta if using mask
    if using_mask:
        E_beta_old = E_beta_old * mask
    
    # Compute phi for this minibatch
    alpha_beta_batch_update = jnp.full_like(alpha_beta_old, c)
    alpha_theta_batch = jnp.full_like(E_theta_batch, a)
    
    # Process this batch
    beta_batch_update, theta_batch_update = process_phi_batch(
        E_theta_batch, E_beta_old, x_batch
    )
    
    # Scale the updates to represent the full dataset
    alpha_beta_batch_update += beta_batch_update * scaling_factor
    alpha_theta_batch += theta_batch_update
    
    # Apply mask to alpha_beta_batch_update if using mask
    if using_mask:
        alpha_beta_batch_update = alpha_beta_batch_update * mask
    
    # Add label signal for upsilon if necessary
    if kappa == 1:
        E_upsilon_1d = upsilon_old[0]
        label_signal = (y_batch.reshape(-1) - 0.5)[:, None] * E_upsilon_1d[None, :]
        alpha_theta_batch += label_scale * label_signal
    
    # Compute omega updates
    # For omega_beta, we approximate using the current batch for E_theta contribution
    E_eta = alpha_eta_old / jnp.maximum(omega_eta_old, 1e-10)
    
    # Calculate batch contribution to sumTheta
    sumTheta_batch = jnp.sum(E_theta_batch, axis=0) * scaling_factor 
    
    # Now calculate omega_beta update
    omega_beta_batch = E_eta[:, None] + sumTheta_batch[None, :]
    omega_beta_batch = jnp.clip(omega_beta_batch, 1e-6, 1e3)
    
    # For omega_theta, we only update the batch indices
    E_xi_batch = alpha_xi_old[q_params['batch_indices']] / jnp.maximum(omega_xi_old[q_params['batch_indices']], 1e-10)
    sumBeta = jnp.sum(E_beta_old, axis=0)
    
    omega_theta_batch = E_xi_batch[:, None] + sumBeta[None, :]
    omega_theta_batch = jnp.clip(omega_theta_batch, 1e-6, 1e3)
    
    # For alpha_eta and omega_eta (global parameters)
    alpha_eta_new = c_prime + c * d
    
    # Recalculate E_beta using the updated batch values
    E_beta_batch = alpha_beta_batch_update / jnp.maximum(omega_beta_batch, 1e-10)
    
    if using_mask:
        E_beta_batch = E_beta_batch * mask
    
    sumBetaEachGene = jnp.sum(E_beta_batch, axis=1)
    omega_eta_new = sumBetaEachGene + (c_prime / d_prime)
    omega_eta_new = jnp.clip(omega_eta_new, 1e-6, 1e3)
    
    # For alpha_xi and omega_xi (only update the batch indices)
    alpha_xi_batch = a_prime + a * d
    
    # Recalculate E_theta for the batch
    E_theta_new_batch = alpha_theta_batch / jnp.maximum(omega_theta_batch, 1e-10)
    
    sumThetaEachCell = jnp.sum(E_theta_new_batch, axis=1)
    omega_xi_batch = sumThetaEachCell + (a_prime / b_prime)
    omega_xi_batch = jnp.clip(omega_xi_batch, 1e-6, 1e3)
    
    # Update gamma and upsilon (global parameters)
    gamma_new = jnp.zeros_like(gamma_old)
    sigma_gamma_inv_new = jnp.zeros_like(sigma_gamma_inv_old)
    upsilon_new = jnp.zeros_like(upsilon_old)
    sigma_upsilon_inv_new = jnp.zeros_like(sigma_upsilon_inv_old)
    
    # Calculate E_theta for upsilon updates (using the local batch)
    E_theta_for_ups = alpha_theta_batch / jnp.maximum(omega_theta_batch, 1e-10)
    
    # Use damping factor to smooth updates
    damping_factor = 0.7
    
    for k in range(kappa):
        # Update gamma with scaled minibatch
        lr_gamma = 0.05
        gamma_k, H_gamma = newton_map_gamma(x_aux_batch, y_batch[:, k], sigma,
                                          gamma_init=gamma_old[k],
                                          learning_rate=lr_gamma)
        
        # Apply damping to smooth gamma updates
        gamma_k = gamma_old[k] * (1 - damping_factor) + gamma_k * damping_factor
        
        gamma_new = gamma_new.at[k, :].set(gamma_k)
        sigma_gamma_inv_new = sigma_gamma_inv_new.at[k].set(H_gamma)
        
        # Update upsilon - alternate between updating odd and even dimensions
        lr_upsilon = 0.05
        upsilon_k = upsilon_old[k].copy()
        
        # Only update even or odd dimensions based on iteration count
        iter_count = q_params.get('iter_count', 0)
        update_mask = jnp.ones_like(upsilon_k)
        
        if iter_count % 2 == 0:
            # Update even dimensions
            update_mask = update_mask.at[1::2].set(0)
        else:
            # Update odd dimensions
            update_mask = update_mask.at[0::2].set(0)
        
        # Get update for the selected dimensions
        upsilon_update, H_upsilon = newton_map_upsilon(E_theta_for_ups, y_batch[:, k], tau,
                                                     upsilon_init=upsilon_old[k],
                                                     learning_rate=lr_upsilon)
        
        # Apply update only to selected dimensions
        upsilon_k = upsilon_k * (1 - update_mask) + upsilon_update * update_mask
        
        # Apply damping to smooth upsilon updates
        upsilon_k = upsilon_old[k] * (1 - damping_factor) + upsilon_k * damping_factor
        
        # If using mask, ensure upsilon values are scaled appropriately for masked columns
        if using_mask:
            mask_present = jnp.sum(mask, axis=0) > 0
            if jnp.any(~mask_present):
                # Scale down upsilon values for columns with no mask entries
                upsilon_k = upsilon_k * jnp.where(mask_present, 1.0, 1e-5)
        
        upsilon_new = upsilon_new.at[k, :].set(upsilon_k)
        sigma_upsilon_inv_new = sigma_upsilon_inv_new.at[k].set(H_upsilon)
    
    # Learning rate for SVI (decreases over iterations)
    rho_t = 1 / ((q_params['iter_count'] + 10) ** 0.7)  # Robbins-Monro step-size
    
    # Update global parameters with SVI step
    q_params['alpha_eta'] = (1 - rho_t) * alpha_eta_old + rho_t * alpha_eta_new
    q_params['omega_eta'] = (1 - rho_t) * omega_eta_old + rho_t * omega_eta_new
    q_params['alpha_beta'] = (1 - rho_t) * alpha_beta_old + rho_t * alpha_beta_batch_update
    q_params['omega_beta'] = (1 - rho_t) * omega_beta_old + rho_t * omega_beta_batch
    q_params['gamma'] = gamma_new
    q_params['sigma_gamma_inv'] = sigma_gamma_inv_new
    q_params['upsilon'] = upsilon_new
    q_params['sigma_upsilon_inv'] = sigma_upsilon_inv_new
    
    # Update local parameters for this batch
    q_params['alpha_theta'] = q_params['alpha_theta'].at[q_params['batch_indices']].set(alpha_theta_batch)
    q_params['omega_theta'] = q_params['omega_theta'].at[q_params['batch_indices']].set(omega_theta_batch)
    q_params['alpha_xi'] = q_params['alpha_xi'].at[q_params['batch_indices']].set(alpha_xi_batch)
    q_params['omega_xi'] = q_params['omega_xi'].at[q_params['batch_indices']].set(omega_xi_batch)
    
    # Increment iteration counter
    q_params['iter_count'] += 1
    
    return q_params


def run_stochastic_variational_inference(x_data, y_data, x_aux, hyperparams,
                                         minibatch_size=100, epochs=10,
                                         q_params=None, tol=1e-6, verbose=True,
                                         label_scale=0.1, mask=None,
                                         patience=5, min_delta=1e-3, beta_init=None):
    """Run stochastic variational inference with minibatches."""
    log_memory("Starting run_stochastic_variational_inference")
    
    # Convert sparse matrices to dense arrays for JAX compatibility
    if hasattr(x_data, 'toarray'):
        print("Converting sparse x_data to dense array for JAX compatibility")
        x_data = x_data.toarray()
    
    # Ensure data is in JAX-compatible format
    x_data = jnp.array(x_data)
    y_data = jnp.array(y_data)
    x_aux = jnp.array(x_aux)
    
    print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}, x_aux shape: {x_aux.shape}")
    print(f"hyperparams['d']: {hyperparams['d']}")
    
    if mask is not None:
        print(f"Using mask with shape {mask.shape}")
        mask = jnp.array(mask)
    
    # Initialize parameters if not provided
    if q_params is None:
        n, p = x_data.shape
        if len(y_data.shape) == 1:
            kappa = 1
        else:
            kappa = y_data.shape[1]
        p_aux = x_aux.shape[1]
        d = hyperparams['d']
        print(f"Initializing q_params with n={n}, p={p}, kappa={kappa}, p_aux={p_aux}, d={d}")
        q_params = initialize_q_params(n, p, kappa, p_aux, d, beta_init=beta_init)
    
    # Define JIT-compiled functions for computing expectations
    @jax.jit
    def compute_expectations(alpha_eta, omega_eta, alpha_beta, omega_beta, 
                            alpha_xi, omega_xi, alpha_theta, omega_theta, mask=None):
        E_eta = alpha_eta / jnp.maximum(omega_eta, 1e-10)
        E_beta = alpha_beta / jnp.maximum(omega_beta, 1e-10)
        if mask is not None:
            E_beta = E_beta * mask
        E_xi = alpha_xi / jnp.maximum(omega_xi, 1e-10)
        E_theta = alpha_theta / jnp.maximum(omega_theta, 1e-10)
        return E_eta, E_beta, E_xi, E_theta
    
    # JIT-compile preparing ELBO inputs
    @jax.jit
    def prepare_elbo_inputs(E_eta, E_beta, E_xi, E_theta, E_gamma, E_upsilon):
        return E_eta, E_beta, E_xi, E_theta, E_gamma, E_upsilon
    
    elbo_history = []
    old_elbo = -1e15
    
    # Early stopping variables
    best_elbo = -1e15
    best_params = None
    patience_counter = 0
    best_iter = 0
    convergence_iter = None
    convergence_reason = None
    
    n_samples = x_data.shape[0]
    n_batches_per_epoch = int(np.ceil(n_samples / minibatch_size))
    total_iterations = epochs * n_batches_per_epoch
    
    if verbose:
        print(f"Starting SVI with {epochs} epochs, {n_batches_per_epoch} batches per epoch, patience={patience}")
        print(f"Total iterations: {total_iterations}, batch size: {minibatch_size}")
    
    for epoch in range(epochs):
        # Shuffle data indices at the beginning of each epoch
        random_key = jax.random.PRNGKey(epoch)
        indices = jax.random.permutation(random_key, n_samples)
        
        for batch_idx in range(n_batches_per_epoch):
            iter_ix = epoch * n_batches_per_epoch + batch_idx
            
            # Get the current batch indices
            start_idx = batch_idx * minibatch_size
            end_idx = min((batch_idx + 1) * minibatch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Store the indices in q_params for update_svi_batch_params
            q_params['batch_indices'] = batch_indices
            
            # Get batch data
            x_batch = x_data[batch_indices]
            y_batch = y_data[batch_indices] if len(y_data.shape) == 1 else y_data[batch_indices, :]
            x_aux_batch = x_aux[batch_indices]
            
            # Update parameters with this batch
            q_params = update_svi_batch_params(
                q_params, x_batch, y_batch, x_aux_batch, 
                hyperparams, n_samples, end_idx - start_idx, 
                mask=mask, label_scale=label_scale
            )
            
            # Calculate ELBO every few batches or at the end of an epoch
            if (batch_idx + 1) % min(10, n_batches_per_epoch) == 0 or batch_idx == n_batches_per_epoch - 1:
                # Compute expectations
                E_eta, E_beta, E_xi, E_theta = compute_expectations(
                    q_params['alpha_eta'], q_params['omega_eta'],
                    q_params['alpha_beta'], q_params['omega_beta'],
                    q_params['alpha_xi'], q_params['omega_xi'],
                    q_params['alpha_theta'], q_params['omega_theta'],
                    mask
                )
                
                E_gamma = q_params['gamma']
                E_upsilon = q_params['upsilon']
                
                # Prepare inputs for ELBO computation
                E_eta, E_beta, E_xi, E_theta, E_gamma, E_upsilon = prepare_elbo_inputs(
                    E_eta, E_beta, E_xi, E_theta, E_gamma, E_upsilon
                )
                
                # Compute ELBO
                elbo_val = compute_elbo(
                    E_eta, E_beta, E_xi, E_theta,
                    E_gamma, E_upsilon,
                    x_data, y_data, x_aux,
                    hyperparams, q_params, mask=mask
                )
                
                elbo_history.append(float(elbo_val))
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{n_batches_per_epoch}, "
                          f"ELBO = {elbo_val:.4f}")
                
                # Check for convergence
                if jnp.abs(elbo_val - old_elbo) < tol:
                    if verbose:
                        print(f"Converged. Change in ELBO ({jnp.abs(elbo_val - old_elbo):.6f}) "
                              f"is less than tolerance ({tol:.6f}).")
                    convergence_iter = iter_ix + 1
                    convergence_reason = f"ELBO change ({jnp.abs(elbo_val - old_elbo):.6e}) < tolerance ({tol:.6e})"
                    break
                
                # Early stopping logic
                improvement = elbo_val - best_elbo
                if elbo_val > best_elbo:
                    if improvement > min_delta:
                        best_elbo = elbo_val
                        best_iter = iter_ix + 1
                        # Deep copy the parameters
                        best_params = {k: v.copy() if isinstance(v, jnp.ndarray) else v 
                                      for k, v in q_params.items()}
                        patience_counter = 0
                        if verbose and iter_ix > 0:
                            print(f"  New best ELBO! Improvement: {improvement:.6f}")
                    else:
                        patience_counter += 1
                        if verbose:
                            print(f"  Small improvement: {improvement:.6f} < {min_delta:.6f}, "
                                  f"patience: {patience_counter}/{patience}")
                else:
                    patience_counter += 1
                    if verbose:
                        print(f"  No improvement, patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered after {iter_ix+1} iterations. "
                              f"No significant improvement in the last {patience} evaluations.")
                    if best_params is not None:
                        q_params = best_params
                        if verbose:
                            print(f"Restored best model with ELBO = {best_elbo:.4f} from iteration {best_iter}")
                    convergence_iter = iter_ix + 1
                    convergence_reason = f"Early stopping: no improvement > {min_delta:.6e} for {patience} evaluations"
                    break
                
                old_elbo = elbo_val
                
                # Try to clear memory periodically
                if (batch_idx + 1) % 10 == 0:
                    clear_memory()
        
        # Check if convergence happened during the epoch
        if convergence_iter is not None:
            break
    
    # If we completed all iterations without triggering early stopping
    if convergence_iter is None:
        convergence_iter = total_iterations
        convergence_reason = "Maximum iterations reached"
    
    # Print convergence summary
    print("\n" + "="*50)
    print(f"ELBO Convergence Summary:")
    print(f"  Total iterations: {convergence_iter}/{total_iterations}")
    print(f"  Best ELBO: {best_elbo:.6f} (at iteration {best_iter})")
    print(f"  Final ELBO: {elbo_history[-1]:.6f}")
    print(f"  Convergence reason: {convergence_reason}")
    print("="*50 + "\n")
    
    # Store convergence info in the model parameters
    q_params['convergence_info'] = {
        'total_iterations': convergence_iter,
        'max_iterations': total_iterations,
        'best_elbo': float(best_elbo),
        'best_iter': best_iter,
        'final_elbo': float(elbo_history[-1]),
        'convergence_reason': convergence_reason
    }
    
    # Remove batch indices from final parameters
    if 'batch_indices' in q_params:
        del q_params['batch_indices']
    
    return q_params, elbo_history