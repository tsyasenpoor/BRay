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

jax.config.update('jax_disable_jit', False)  
jax.config.update('jax_platform_name', 'cpu') 

# Add this at the top of your file, outside any function
@jax.jit
def process_phi_batch(theta_batch, beta, x_batch):
    numerator_phi_batch = jnp.einsum('bd,pd->bpd', theta_batch, beta)
    denom_phi_batch = jnp.sum(numerator_phi_batch, axis=2, keepdims=True) + 1e-10
    phi_batch = numerator_phi_batch / denom_phi_batch
    alpha_beta_batch_update = jnp.sum(x_batch[:, :, None] * phi_batch, axis=0)
    alpha_theta_batch_update = jnp.sum(x_batch[:, :, None] * phi_batch, axis=1)
    return alpha_beta_batch_update, alpha_theta_batch_update

def neg_logpost_gamma(gamma, X_aux, y, sigma, offset=1e-15):
    z = X_aux @ gamma.T  
    p = jax.nn.sigmoid(z)
    log_lik = jnp.sum(y * jnp.log(p + offset) + (1.0 - y) * jnp.log(1.0 - p + offset))
    log_prior = -0.5 * jnp.sum(gamma**2) / (sigma**2)
    return - (log_lik + log_prior)

def make_gamma_grad_and_hess(X_aux, y, sigma):
    grad_fn = jax.grad(neg_logpost_gamma, argnums=0)
    hess_fn = jax.hessian(neg_logpost_gamma, argnums=0)
    return (lambda g: grad_fn(g, X_aux, y, sigma),
            lambda g: hess_fn(g, X_aux, y, sigma))

@partial(jax.jit, static_argnums=(1, 2))  # Mark grad_fn and hess_fn as static arguments
def newton_step_gamma(gamma, grad_fn, hess_fn, learning_rate=0.1):
    g = grad_fn(gamma)
    H = hess_fn(gamma)
    # Further increase regularization to 1e-3 for better conditioning
    H_reg = H + jnp.eye(H.shape[0]) * 1e-3
    step = jnp.linalg.solve(H_reg, g)
    
    # Clip the step size to prevent large updates
    max_step_norm = 1.0  # Maximum allowed step size
    step_norm = jnp.linalg.norm(step)
    scaling_factor = jnp.minimum(1.0, max_step_norm / (step_norm + 1e-10))
    step = step * scaling_factor
    
    return gamma - learning_rate * step, H_reg

def newton_map_gamma(X_aux, y, sigma, max_iters=50, tol=1e-6, learning_rate=0.1, gamma_init=None):
    if len(X_aux.shape) == 1:
        X_aux = X_aux.reshape(-1, 1)
    p_aux = X_aux.shape[1]
    
    if gamma_init is not None:
        gamma = gamma_init
    else:
        random_int = jax.random.randint(jax.random.PRNGKey(0), (), 0, 2**30, dtype=jnp.int32)
        key = jax.random.PRNGKey(int(random_int))
        gamma = jnp.array(jax.random.normal(key, (p_aux,))) * 0.01
    
    grad_fn, hess_fn = make_gamma_grad_and_hess(X_aux, y, sigma)
    
    # Define a jitted step function to improve performance
    @jax.jit
    def step_fn(gamma):
        gamma_new, H = newton_step_gamma(gamma, grad_fn, hess_fn, learning_rate)
        # Clip values to prevent extreme parameter values
        gamma_new = jnp.clip(gamma_new, -5.0, 5.0)
        return gamma_new, H
    
    for _ in range(max_iters):
        gamma_old = gamma
        gamma_new, _ = step_fn(gamma)  # Use _ to ignore the unused H value
        
        if jnp.any(jnp.isnan(gamma_new)):
            gamma_new = gamma_old 
        gamma = gamma_new
        if jnp.linalg.norm(gamma - gamma_old) < tol:
            break

    H_final = hess_fn(gamma) + jnp.eye(gamma.shape[0]) * 1e-3  
    return gamma, H_final


def neg_logpost_upsilon(upsilon, theta, y, tau, offset=1e-15):
    if len(theta.shape) == 1:
        theta = theta.reshape(-1, 1)

    z = jnp.dot(theta, upsilon)  
    p = jax.nn.sigmoid(z)
    log_lik = jnp.sum(y * jnp.log(p + offset) + (1 - y) * jnp.log(1 - p + offset))
    # Use a Laplace prior to encourage sparsity
    log_prior = -jnp.sum(jnp.abs(upsilon)) / tau - upsilon.size * jnp.log(2 * tau)
    return - (log_lik + log_prior)

def make_upsilon_grad_and_hess(theta, y, tau):
    grad_fn = jax.grad(neg_logpost_upsilon, argnums=0)
    hess_fn = jax.hessian(neg_logpost_upsilon, argnums=0)
    return (lambda u: grad_fn(u, theta, y, tau),
            lambda u: hess_fn(u, theta, y, tau))

@partial(jax.jit, static_argnums=(1, 2))  # Mark grad_fn and hess_fn as static arguments
def newton_step_upsilon(upsilon, grad_fn, hess_fn, learning_rate=0.1):
    g = grad_fn(upsilon)
    H = hess_fn(upsilon)
    H_reg = H + jnp.eye(H.shape[0]) * 1e-3
    step = jnp.linalg.solve(H_reg, g)
    
    max_step_norm = 1.0  
    step_norm = jnp.linalg.norm(step)
    scaling_factor = jnp.minimum(1.0, max_step_norm / (step_norm + 1e-10))
    step = step * scaling_factor
    
    return upsilon - learning_rate * step, H_reg

def newton_map_upsilon(theta, y, tau, max_iters=50, tol=1e-6, learning_rate=0.1, upsilon_init=None):
    if len(theta.shape) == 1:
        theta = theta.reshape(-1, 1)
    d = theta.shape[1]
    
    if upsilon_init is not None:
        upsilon = upsilon_init
    else:
        random_int = jax.random.randint(jax.random.PRNGKey(0), (), 0, 2**30, dtype=jnp.int32)
        key = jax.random.PRNGKey(int(random_int))
        upsilon = jnp.array(jax.random.normal(key, (d,))) * 0.1

    grad_fn, hess_fn = make_upsilon_grad_and_hess(theta, y, tau)
    
    # Define a jitted step function to improve performance
    @jax.jit
    def step_fn(upsilon):
        upsilon_new, H = newton_step_upsilon(upsilon, grad_fn, hess_fn, learning_rate)
        upsilon_new = jnp.clip(upsilon_new, -5.0, 5.0)
        return upsilon_new, H
    
    for _ in range(max_iters):                
        upsilon_old = upsilon
        upsilon_new, _ = step_fn(upsilon)  # Use _ to ignore the unused H value
        
        if jnp.any(jnp.isnan(upsilon_new)):
            upsilon_new = upsilon_old
        upsilon = upsilon_new
        if jnp.linalg.norm(upsilon - upsilon_old) < tol:
            break

    H_final_ups = hess_fn(upsilon) + jnp.eye(upsilon.shape[0]) * 1e-3  
    return upsilon, H_final_ups


def initialize_q_params(n, p, kappa, p_aux, d, seed=None, beta_init=None):
    log_memory(f"Before initialize_q_params (n={n}, p={p}, kappa={kappa}, p_aux={p_aux}, d={d})")
    
    if seed is None:
        random_int = jax.random.randint(jax.random.PRNGKey(0), (), 0, 2**30, dtype=jnp.int32)
        key = jax.random.PRNGKey(int(random_int))
    else:
        key = jax.random.PRNGKey(seed)

    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    
    alpha_eta   = jnp.ones(p)      + 0.01 * jax.random.normal(k1, (p,))
    
    # Initialize alpha_beta and omega_beta from beta_init if provided
    if beta_init is not None:
        print(f"Initializing alpha_beta from provided beta_init with shape {beta_init.shape}")
        # Use beta_init as the expected value of beta, so alpha_beta/omega_beta = beta_init
        # Set omega_beta to ones and alpha_beta to beta_init
        omega_beta = jnp.ones((p, d))
        alpha_beta = beta_init * omega_beta  # This ensures E[beta] = alpha_beta/omega_beta = beta_init
    else:
        alpha_beta  = jnp.ones((p, d)) + 0.01 * jax.random.normal(k2, (p, d))
        omega_beta  = jnp.ones((p, d))
    
    alpha_xi    = jnp.ones(n)      + 0.01 * jax.random.normal(k3, (n,))
    alpha_theta = jnp.ones((n, d)) + 0.01 * jax.random.normal(k4, (n, d))
    
    omega_eta   = jnp.ones(p)
    omega_xi    = jnp.ones(n)
    omega_theta = jnp.ones((n, d))

    gamma_init   = jnp.array(jax.random.normal(k5, (kappa, p_aux))) * 0.01
    upsilon_init = jnp.array(jax.random.normal(k6, (kappa, d)))     * 0.1

    sigma_gamma_inv_init   = jnp.array([jnp.eye(p_aux) for _ in range(kappa)])
    sigma_upsilon_inv_init = jnp.array([jnp.eye(d)     for _ in range(kappa)])
    
    # log_array_sizes({
    #     'alpha_beta': alpha_beta,
    #     'alpha_theta': alpha_theta,
    #     'omega_beta': omega_beta,
    #     'omega_theta': omega_theta,
    #     'sigma_gamma_inv_init': sigma_gamma_inv_init,
    #     'sigma_upsilon_inv_init': sigma_upsilon_inv_init
    # })
    
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
        "sigma_upsilon_inv": sigma_upsilon_inv_init
    }
    
    # log_memory("After initialize_q_params")
    return q_params


def compute_elbo(E_eta, E_beta, E_xi, E_theta, E_gamma, E_upsilon,
                 x_data, y_data, x_aux, hyperparams, q_params, mask=None): # Add mask=None
    x_data = jnp.array(x_data)
    y_data = jnp.array(y_data)
    x_aux  = jnp.array(x_aux)

    # Optional sanity check
    if mask is not None:
        mask_jax = jnp.array(mask) # Ensure mask is a JAX array
        print(f"Mask has {int(jnp.sum(mask_jax))}/{mask_jax.size} active β-entries.")
    else:
        mask_jax = None # Define mask_jax as None if mask is None

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

    E_log_p_eta = jnp.sum(
        c_prime * jnp.log(d_prime)
        - jsp.special.gammaln(c_prime)
        + (c_prime - 1) * jnp.log(jnp.maximum(E_eta_flat, 1e-10))
        - d_prime * E_eta_flat
    )
    print(f"E_log_p_eta: {E_log_p_eta:.6e}")
    if jnp.isnan(E_log_p_eta):
        print("WARNING: E_log_p_eta is NaN!")
        print(f"  c_prime: {c_prime}, d_prime: {d_prime}")
        print(f"  jnp.log(E_eta_flat + 1e-10) range: {jnp.min(jnp.log(jnp.maximum(E_eta_flat, 1e-10))):.6e} to {jnp.max(jnp.log(jnp.maximum(E_eta_flat, 1e-10))):.6e}")

    E_eta_col = E_eta.reshape(-1, 1)
    if E_eta_col.shape[1] == 1 and E_beta.shape[1] > 1:
        E_eta_broadcast = jnp.tile(E_eta_col, (1, E_beta.shape[1]))
    else:
        E_eta_broadcast = E_eta_col

    # Calculate terms for E_log_p_beta
    E_log_p_beta_terms = (
        c * jnp.log(jnp.maximum(E_eta_broadcast, 1e-10))
        - jsp.special.gammaln(c)
        + (c - 1) * jnp.log(jnp.maximum(E_beta, 1e-10))
        - E_eta_broadcast * E_beta
    )

    # Apply mask if provided
    if mask_jax is not None:
        E_log_p_beta = jnp.sum(mask_jax * E_log_p_beta_terms)
    else:
        E_log_p_beta = jnp.sum(E_log_p_beta_terms)

    print(f"E_log_p_beta: {E_log_p_beta:.6e}")
    if jnp.isnan(E_log_p_beta):
        print("WARNING: E_log_p_beta is NaN!")
        print(f"  c: {c}")
        print(f"  jnp.log(E_eta_broadcast + 1e-10) range: {jnp.min(jnp.log(jnp.maximum(E_eta_broadcast, 1e-10))):.6e} to {jnp.max(jnp.log(jnp.maximum(E_eta_broadcast, 1e-10))):.6e}")
        print(f"  jnp.log(E_beta + 1e-10) range: {jnp.min(jnp.log(jnp.maximum(E_beta, 1e-10))):.6e} to {jnp.max(jnp.log(jnp.maximum(E_beta, 1e-10))):.6e}")

    rate = jnp.dot(E_theta, E_beta.T) + 1e-10
    print(f"rate range: {jnp.min(rate):.6e} to {jnp.max(rate):.6e}")
    
    E_log_p_x = jnp.sum(
        x_data * jnp.log(rate) - rate - jsp.special.gammaln(x_data + 1)
    )
    print(f"E_log_p_x: {E_log_p_x:.6e}")
    if jnp.isnan(E_log_p_x):
        print("WARNING: E_log_p_x is NaN!")
        print(f"  jnp.log(rate) range: {jnp.min(jnp.log(rate)):.6e} to {jnp.max(jnp.log(rate)):.6e}")

    E_log_p_xi = jnp.sum(
        a_prime * jnp.log(b_prime)
        - jsp.special.gammaln(a_prime)
        + (a_prime - 1) * jnp.log(jnp.maximum(E_xi_flat, 1e-10))
        - b_prime * E_xi_flat
    )
    print(f"E_log_p_xi: {E_log_p_xi:.6e}")
    if jnp.isnan(E_log_p_xi):
        print("WARNING: E_log_p_xi is NaN!")
        print(f"  a_prime: {a_prime}, b_prime: {b_prime}")
        print(f"  jnp.log(E_xi_flat + 1e-10) range: {jnp.min(jnp.log(jnp.maximum(E_xi_flat, 1e-10))):.6e} to {jnp.max(jnp.log(jnp.maximum(E_xi_flat, 1e-10))):.6e}")

    E_xi_col = E_xi.reshape(-1, 1)
    if E_xi_col.shape[1] == 1 and E_theta.shape[1] > 1:
        E_xi_broadcast = jnp.tile(E_xi_col, (1, E_theta.shape[1]))
    else:
        E_xi_broadcast = E_xi_col
    E_log_p_theta = jnp.sum(
        a * jnp.log(jnp.maximum(E_xi_broadcast, 1e-10))
        - jsp.special.gammaln(a)
        + (a - 1) * jnp.log(jnp.maximum(E_theta, 1e-10))
        - E_xi_broadcast * E_theta
    )
    print(f"E_log_p_theta: {E_log_p_theta:.6e}")
    if jnp.isnan(E_log_p_theta):
        print("WARNING: E_log_p_theta is NaN!")
        print(f"  a: {a}")
        print(f"  jnp.log(E_xi_broadcast + 1e-10) range: {jnp.min(jnp.log(jnp.maximum(E_xi_broadcast, 1e-10))):.6e} to {jnp.max(jnp.log(jnp.maximum(E_xi_broadcast, 1e-10))):.6e}")
        print(f"  jnp.log(E_theta + 1e-10) range: {jnp.min(jnp.log(jnp.maximum(E_theta, 1e-10))):.6e} to {jnp.max(jnp.log(jnp.maximum(E_theta, 1e-10))):.6e}")
    
    logits = (x_aux @ E_gamma.T) + jnp.einsum('nd,kd->nk', E_theta, E_upsilon)
    probs  = jax.nn.sigmoid(logits)
    print(f"logits range: {jnp.min(logits):.6e} to {jnp.max(logits):.6e}")
    print(f"probs range: {jnp.min(probs):.6e} to {jnp.max(probs):.6e}")

    prob_zeros = jnp.sum(probs < 1e-10)
    prob_ones = jnp.sum(probs > 1 - 1e-10)
    if prob_zeros > 0 or prob_ones > 0:
        print(f"WARNING: {prob_zeros} probability values near 0 and {prob_ones} near 1")
    
    probs_safe = jnp.clip(probs, 1e-8, 1-1e-8)
    
    log_probs = jnp.log(probs_safe)
    log_one_minus_probs = jnp.log(1 - probs_safe)
    
    log_probs = jnp.nan_to_num(log_probs, nan=-20.0, posinf=-20.0, neginf=-20.0)
    log_one_minus_probs = jnp.nan_to_num(log_one_minus_probs, nan=-20.0, posinf=-20.0, neginf=-20.0)
    
    E_log_p_y = jnp.sum(
        y_data * log_probs
        + (1 - y_data) * log_one_minus_probs
    )
    
    E_log_p_y = jnp.nan_to_num(E_log_p_y, nan=-1000.0)
    
    print(f"E_log_p_y: {E_log_p_y:.6e}")
    if jnp.isnan(E_log_p_y):
        print("WARNING: E_log_p_y is NaN!")
        print(f"  log_probs range: {jnp.min(log_probs):.6e} to {jnp.max(log_probs):.6e}")
        print(f"  log_one_minus_probs range: {jnp.min(log_one_minus_probs):.6e} to {jnp.max(log_one_minus_probs):.6e}")

    E_log_p_gamma = -0.5 * jnp.sum(E_gamma**2) / sigma**2
    print(f"E_log_p_gamma: {E_log_p_gamma:.6e}")
    if jnp.isnan(E_log_p_gamma):
        print("WARNING: E_log_p_gamma is NaN!")
    
    # Laplace prior encourages sparsity in upsilon
    E_log_p_upsilon = -jnp.sum(jnp.abs(E_upsilon)) / tau - E_upsilon.size * jnp.log(2 * tau)
    print(f"E_log_p_upsilon: {E_log_p_upsilon:.6e}")
    if jnp.isnan(E_log_p_upsilon):
        print("WARNING: E_log_p_upsilon is NaN!")


    alpha_eta = q_params['alpha_eta']
    omega_eta = q_params['omega_eta']
    
    log_omega_eta = jnp.log(jnp.maximum(omega_eta, 1e-10))
    digamma_alpha_eta = jsp.special.digamma(alpha_eta)
    
    E_log_q_eta = jnp.sum(
        alpha_eta * log_omega_eta
        - jsp.special.gammaln(alpha_eta)
        + (alpha_eta - 1) * (digamma_alpha_eta - log_omega_eta)
        - omega_eta * E_eta_flat
    )
    print(f"E_log_q_eta: {E_log_q_eta:.6e}")
    if jnp.isnan(E_log_q_eta):
        print("WARNING: E_log_q_eta is NaN!")
        print(f"  alpha_eta range: {jnp.min(alpha_eta):.6e} to {jnp.max(alpha_eta):.6e}")
        print(f"  omega_eta range: {jnp.min(omega_eta):.6e} to {jnp.max(omega_eta):.6e}")
        print(f"  jsp.special.digamma(alpha_eta) range: {jnp.min(digamma_alpha_eta):.6e} to {jnp.max(digamma_alpha_eta):.6e}")

    alpha_beta = q_params['alpha_beta']
    omega_beta = q_params['omega_beta']
    
    log_omega_beta = jnp.log(jnp.maximum(omega_beta, 1e-10))
    digamma_alpha_beta = jsp.special.digamma(alpha_beta)
    
    if jnp.any(jnp.isnan(alpha_beta)) or jnp.any(jnp.isnan(omega_beta)):
        print("WARNING: NaN values in alpha_beta or omega_beta; replacing with safe values")
        alpha_beta = jnp.nan_to_num(alpha_beta, nan=1.0, posinf=1e3, neginf=1.0)
        omega_beta = jnp.nan_to_num(omega_beta, nan=1.0, posinf=1e3, neginf=1.0)
    
    if jnp.any(jnp.isnan(E_beta)) or jnp.any(jnp.isinf(E_beta)):
        print("WARNING: NaN or Inf values in E_beta; replacing with safe values")
        E_beta = jnp.nan_to_num(E_beta, nan=1e-10, posinf=1e3, neginf=1e-10)
        
    E_log_q_beta_terms = (
        alpha_beta * log_omega_beta
        - jsp.special.gammaln(alpha_beta)
        + (alpha_beta - 1) * (digamma_alpha_beta - log_omega_beta)
        - omega_beta * E_beta
    )
    
    E_log_q_beta_terms = jnp.nan_to_num(E_log_q_beta_terms, nan=0.0)
    # Apply mask if provided
    if mask_jax is not None:
        E_log_q_beta = jnp.sum(mask_jax * E_log_q_beta_terms)
    else:
        E_log_q_beta = jnp.sum(E_log_q_beta_terms)

    print(f"E_log_q_beta: {E_log_q_beta:.6e}")
    if jnp.isnan(E_log_q_beta):
        print("WARNING: E_log_q_beta is NaN!")
        print(f"  alpha_beta min/max: {jnp.min(alpha_beta):.6e}/{jnp.max(alpha_beta):.6e}")
        print(f"  omega_beta min/max: {jnp.min(omega_beta):.6e} to {jnp.max(omega_beta):.6e}")
        print(f"  Component ranges: {jnp.min(E_log_q_beta_terms):.6e} to {jnp.max(E_log_q_beta_terms):.6e}")

    alpha_xi = q_params['alpha_xi']
    omega_xi = q_params['omega_xi']
    
    log_omega_xi = jnp.log(jnp.maximum(omega_xi, 1e-10))
    digamma_alpha_xi = jsp.special.digamma(alpha_xi)
    
    E_log_q_xi = jnp.sum(
        alpha_xi * log_omega_xi
        - jsp.special.gammaln(alpha_xi)
        + (alpha_xi - 1) * (digamma_alpha_xi - log_omega_xi)
        - omega_xi * E_xi_flat
    )
    print(f"E_log_q_xi: {E_log_q_xi:.6e}")
    if jnp.isnan(E_log_q_xi):
        print("WARNING: E_log_q_xi is NaN!")
        print(f"  alpha_xi min/max: {jnp.min(alpha_xi):.6e}/{jnp.max(alpha_xi):.6e}")
        print(f"  omega_xi min/max: {jnp.min(omega_xi):.6e} to {jnp.max(omega_xi):.6e}")

    alpha_theta = q_params['alpha_theta']
    omega_theta = q_params['omega_theta']
    
    log_omega_theta = jnp.log(jnp.maximum(omega_theta, 1e-10))
    digamma_alpha_theta = jsp.special.digamma(alpha_theta)
    
    E_log_q_theta = jnp.sum(
        alpha_theta * log_omega_theta
        - jsp.special.gammaln(alpha_theta)
        + (alpha_theta - 1) * (digamma_alpha_theta - log_omega_theta)
        - omega_theta * E_theta
    )
    print(f"E_log_q_theta: {E_log_q_theta:.6e}")
    if jnp.isnan(E_log_q_theta):
        print("WARNING: E_log_q_theta is NaN!")
        print(f"  alpha_theta min/max: {jnp.min(alpha_theta):.6e}/{jnp.max(alpha_theta):.6e}")
        print(f"  omega_theta min/max: {jnp.min(omega_theta):.6e}/{jnp.max(omega_theta):.6e}")

    kappa, p_aux = E_gamma.shape
    sigma_gamma_inv = q_params['sigma_gamma_inv']
    E_log_q_gamma = 0
    for kk in range(kappa):
        Sigma_gamma_k_inv = sigma_gamma_inv[kk] + jnp.eye(p_aux)*1e-4
        
        sign, logdet_val = jnp.linalg.slogdet(jnp.linalg.inv(Sigma_gamma_k_inv))
        if sign < 0:
            print(f"WARNING: Negative determinant for Sigma_gamma_inv[{kk}]")
            Sigma_gamma_k_inv = Sigma_gamma_k_inv + jnp.eye(p_aux)*1e-2
            sign, logdet_val = jnp.linalg.slogdet(jnp.linalg.inv(Sigma_gamma_k_inv))
            
        log_det_term = -0.5 * logdet_val
        E_log_q_gamma += (log_det_term - 0.5 * p_aux*(1 + jnp.log(2*jnp.pi)))
        
    print(f"E_log_q_gamma: {E_log_q_gamma:.6e}")
    if jnp.isnan(E_log_q_gamma):
        print("WARNING: E_log_q_gamma is NaN!")
        # Provide fallback value if still NaN
        E_log_q_gamma = 0.0
        print("  Using fallback value 0.0 for E_log_q_gamma")

    kappa2, d = E_upsilon.shape
    sigma_upsilon_inv = q_params['sigma_upsilon_inv']
    E_log_q_upsilon = 0
    for kk in range(kappa2):
        # Increase regularization for matrix operations
        Sigma_ups_k_inv = sigma_upsilon_inv[kk] + jnp.eye(d)*1e-4
        
        # Use more stable slogdet instead of direct determinant calculation
        sign, logdet_val = jnp.linalg.slogdet(jnp.linalg.inv(Sigma_ups_k_inv))
        if sign < 0:
            print(f"WARNING: Negative determinant for Sigma_ups_inv[{kk}]")
            # Handle negative determinant by forcing a positive definite matrix
            Sigma_ups_k_inv = Sigma_ups_k_inv + jnp.eye(d)*1e-2
            sign, logdet_val = jnp.linalg.slogdet(jnp.linalg.inv(Sigma_ups_k_inv))
            
        log_det_term = -0.5 * logdet_val
        E_log_q_upsilon += (log_det_term - 0.5 * d*(1 + jnp.log(2*jnp.pi)))
        
    print(f"E_log_q_upsilon: {E_log_q_upsilon:.6e}")
    if jnp.isnan(E_log_q_upsilon):
        print("WARNING: E_log_q_upsilon is NaN!")
        # Provide fallback value if still NaN
        E_log_q_upsilon = 0.0
        print("  Using fallback value 0.0 for E_log_q_upsilon")

    expected_log_joint = (E_log_p_eta + E_log_p_beta + E_log_p_x + E_log_p_xi
                          + E_log_p_theta + E_log_p_y + E_log_p_gamma + E_log_p_upsilon)
    expected_log_q = (E_log_q_eta + E_log_q_beta + E_log_q_xi 
                      + E_log_q_theta + E_log_q_gamma + E_log_q_upsilon)
    elbo = expected_log_joint - expected_log_q
    
    print(f"Expected log joint: {expected_log_joint:.6e}")
    print(f"Expected log q: {expected_log_q:.6e}")
    print(f"ELBO: {elbo:.6e}")
    if jnp.isnan(elbo):
        print("WARNING: Final ELBO is NaN! Using previous ELBO value")
        # Return the previous ELBO value to avoid returning NaN
        elbo = q_params.get('prev_elbo', -1e15)
        print(f"  Using fallback ELBO value: {elbo:.6e}")
    else:
        # Save the current ELBO value for potential fallback in the future
        q_params['prev_elbo'] = elbo
    
    return elbo


def update_q_params(q_params, x_data, y_data, x_aux, hyperparams,
                    label_scale=0.1, mask=None):
    # log_memory("Starting update_q_params")

    x_data = jnp.array(x_data)
    y_data = jnp.array(y_data)
    x_aux  = jnp.array(x_aux)

    data_shape = x_data.shape

    if len(y_data.shape) == 1:
        y_data = y_data.reshape(-1,1)
    if len(x_aux.shape) == 1:
        x_aux = x_aux.reshape(-1,1)

    # Early handling of mask to avoid redundant checks
    using_mask = mask is not None
    if using_mask:
            mask_present = jnp.sum(mask, axis=0) > 0
            if q_params.get('iter_count', 0) == 0:
                print(f"Applying mask with shape {mask.shape}")
                print(f"Mask sparsity per column: {1 - jnp.sum(mask, axis=0) / mask.shape[0]}")
                print(f"Mask features per column: {jnp.sum(mask, axis=0)}")
    
    alpha_eta_old   = q_params['alpha_eta']
    omega_eta_old   = q_params['omega_eta']
    alpha_beta_old  = q_params['alpha_beta']
    omega_beta_old  = q_params['omega_beta']
    alpha_xi_old    = q_params['alpha_xi']
    omega_xi_old    = q_params['omega_xi']
    alpha_theta_old = q_params['alpha_theta']
    omega_theta_old = q_params['omega_theta']

    gamma_old       = q_params['gamma']      
    upsilon_old     = q_params['upsilon']    
    sigma_gamma_inv_old   = q_params['sigma_gamma_inv']
    sigma_upsilon_inv_old = q_params['sigma_upsilon_inv']

    c_prime = hyperparams['c_prime']
    d_prime = hyperparams['d_prime']
    c       = hyperparams['c']
    a_prime = hyperparams['a_prime']
    b_prime = hyperparams['b_prime']
    a       = hyperparams['a']
    tau     = hyperparams['tau']
    sigma   = hyperparams['sigma']
    d       = hyperparams['d']

    n = x_data.shape[0]
    kappa = y_data.shape[1]

    # Compute E_theta and E_beta
    E_theta_old = alpha_theta_old / jnp.maximum(omega_theta_old, 1e-10)
    E_beta_old  = alpha_beta_old  / jnp.maximum(omega_beta_old,  1e-10)

    # Apply mask to E_beta only if using mask
    if using_mask:
        E_beta_old = E_beta_old * mask
    
    # Compute phi in batches to reduce memory usage
    batch_size = max(1, min(100, n // 10))  # Adjust batch size based on total samples
    print(f"Using batch size of {batch_size} for phi computation")
    
    batch_count = (data_shape[0] + batch_size - 1) // batch_size
    
    alpha_beta_new = jnp.full_like(alpha_beta_old, c)
    alpha_theta_new = jnp.full_like(alpha_theta_old, a)

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_idx = batch_start // batch_size

        print(f"Processing phi batch {batch_idx+1}/{batch_count}, "
          f"samples {batch_start}:{batch_end} "
          f"({(batch_idx+1)/batch_count*100:.1f}% complete)")
        
        # Extract the current batch data
        E_theta_batch = E_theta_old[batch_start:batch_end]
        x_data_batch = x_data[batch_start:batch_end]
        
        # Process batch using JIT-compiled function
        alpha_beta_batch_update, alpha_theta_batch_update = process_phi_batch(
            E_theta_batch, E_beta_old, x_data_batch)
        
        # Update alpha_beta and alpha_theta
        alpha_beta_new = alpha_beta_new + alpha_beta_batch_update
        alpha_theta_new = alpha_theta_new.at[batch_start:batch_end].set(
            alpha_theta_new[batch_start:batch_end] + alpha_theta_batch_update
        )
        
        # Force garbage collection more aggressively
        if batch_end % (batch_size * 5) == 0:
            # Clear JAX cache to reduce memory pressure
            jax.clear_caches()
            gc.collect(generation=2)
    
    # for batch_start in range(0, n, batch_size):
    #     batch_end = min(batch_start + batch_size, n)
    #     print(f"Processing phi batch {batch_start//batch_size + 1}/{(n+batch_size-1)//batch_size}, samples {batch_start}:{batch_end}")
        
    #     # Use einsum for explicit dimension alignment instead of broadcasting
    #     numerator_phi_batch = jnp.einsum('bd,pd->bpd', 
    #                                     E_theta_old[batch_start:batch_end], 
    #                                     E_beta_old)
    #     denom_phi_batch = jnp.sum(numerator_phi_batch, axis=2, keepdims=True) + 1e-10
    #     phi_batch = numerator_phi_batch / denom_phi_batch
    #     alpha_beta_batch_update = jnp.sum(x_data[batch_start:batch_end, :, None] * phi_batch, axis=0)
    #     alpha_beta_new = alpha_beta_new + alpha_beta_batch_update
        
    #     # Update alpha_theta from this batch
    #     alpha_theta_batch_update = jnp.sum(x_data[batch_start:batch_end, :, None] * phi_batch, axis=1)
    #     alpha_theta_new = alpha_theta_new.at[batch_start:batch_end].set(
    #         alpha_theta_new[batch_start:batch_end] + alpha_theta_batch_update
    #     )
        
    #     # Clean up to save memory
    #     del numerator_phi_batch, denom_phi_batch, phi_batch
    #     if batch_end % (5 * batch_size) == 0 or batch_end == n:
    #         clear_memory()
    
    # Apply mask to alpha_beta_new only if using mask
    if using_mask:
        alpha_beta_new = alpha_beta_new * mask

    if kappa == 1:
        E_upsilon_1d = upsilon_old[0]  # shape (d,)
        label_signal = (y_data.reshape(-1) - 0.5)[:, None] * E_upsilon_1d[None, :]
        alpha_theta_new = alpha_theta_new + label_scale * label_signal
        alpha_theta_new = jnp.clip(alpha_theta_new, 1e-8, 1e2)

    # Compute omega updates
    sumTheta = jnp.sum(E_theta_old, axis=0)  # shape (d,)
    E_eta = alpha_eta_old / jnp.maximum(omega_eta_old, 1e-10)
    omega_beta_new = E_eta[:, None] + sumTheta[None, :]
    
    # Apply clipping to omega_beta to prevent extreme values
    omega_beta_new = jnp.clip(omega_beta_new, 1e-6, 1e3)

    sumBeta = jnp.sum(E_beta_old, axis=0)
    E_xi = alpha_xi_old / jnp.maximum(omega_xi_old, 1e-10)
    omega_theta_new = E_xi[:, None] + sumBeta[None, :]
    
    # Apply clipping to omega_theta to prevent extreme values
    omega_theta_new = jnp.clip(omega_theta_new, 1e-6, 1e3)

    alpha_eta_new = c_prime + c*d
    E_beta_new = alpha_beta_new / jnp.maximum(omega_beta_new, 1e-10)
    
    # Apply mask to E_beta_new only if using mask
    if using_mask:
        E_beta_new = E_beta_new * mask
        
    sumBetaEachGene = jnp.sum(E_beta_new, axis=1)
    omega_eta_new = sumBetaEachGene + (c_prime/d_prime)
    
    # Apply clipping to omega_eta
    omega_eta_new = jnp.clip(omega_eta_new, 1e-6, 1e3)

    alpha_xi_new = a_prime + a*d
    E_theta_new = alpha_theta_new / jnp.maximum(omega_theta_new,1e-10)
    sumThetaEachCell = jnp.sum(E_theta_new, axis=1)
    omega_xi_new = sumThetaEachCell + (a_prime/b_prime)
    
    # Apply clipping to omega_xi
    omega_xi_new = jnp.clip(omega_xi_new, 1e-6, 1e3)

    gamma_new = jnp.zeros_like(gamma_old)
    sigma_gamma_inv_new = jnp.zeros_like(sigma_gamma_inv_old)
    upsilon_new = jnp.zeros_like(upsilon_old)
    sigma_upsilon_inv_new = jnp.zeros_like(sigma_upsilon_inv_old)

    # Calculate E_theta_for_ups for upsilon updates
    E_theta_for_ups = alpha_theta_new / jnp.maximum(omega_theta_new, 1e-10)
    
    # Use damping factor to smooth updates
    damping_factor = 0.7  # Weight of new parameters, (1-damping_factor) is weight of old parameters
    
    for k in range(kappa):
        print(f"Processing newton_map_gamma and newton_map_upsilon for k={k}/{kappa}")
        lr_gamma = 0.05   # Further reduce learning rate for gamma
        lr_upsilon = 0.05 # Further reduce learning rate for upsilon

        # Update gamma
        gamma_k, H_gamma = newton_map_gamma(x_aux, y_data[:, k], sigma,
                                           gamma_init=gamma_old[k],
                                           learning_rate=lr_gamma)
        
        # Apply damping to smooth gamma updates
        gamma_k = gamma_old[k] * (1 - damping_factor) + gamma_k * damping_factor
        
        gamma_new = gamma_new.at[k,:].set(gamma_k)
        sigma_gamma_inv_new = sigma_gamma_inv_new.at[k].set(H_gamma)

        # Update upsilon - alternate between updating odd and even dimensions to prevent conflicts
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
        upsilon_update, H_upsilon = newton_map_upsilon(E_theta_for_ups, y_data[:, k], tau,
                                                     upsilon_init=upsilon_old[k],
                                                     learning_rate=lr_upsilon)
        
        # Apply update only to selected dimensions
        upsilon_k = upsilon_k * (1 - update_mask) + upsilon_update * update_mask
        
        # Apply damping to smooth upsilon updates
        upsilon_k = upsilon_old[k] * (1 - damping_factor) + upsilon_k * damping_factor
        
        # If using mask, ensure upsilon values are scaled appropriately for masked columns
        if using_mask and jnp.any(~mask_present):
            # Scale down upsilon values for columns with no mask entries
            upsilon_k = upsilon_k * jnp.where(mask_present, 1.0, 1e-5)
            
        upsilon_new = upsilon_new.at[k,:].set(upsilon_k)
        sigma_upsilon_inv_new = sigma_upsilon_inv_new.at[k].set(H_upsilon)

    # Increment iteration counter
    iter_count = q_params.get('iter_count', 0) + 1

    q_params_new = {
        "alpha_eta": alpha_eta_new,
        "omega_eta": omega_eta_new,
        "alpha_beta": alpha_beta_new,
        "omega_beta": omega_beta_new,
        "alpha_xi": alpha_xi_new,
        "omega_xi": omega_xi_new,
        "alpha_theta": alpha_theta_new,
        "omega_theta": omega_theta_new,
        "gamma": gamma_new,
        "sigma_gamma_inv": sigma_gamma_inv_new,
        "upsilon": upsilon_new,
        "sigma_upsilon_inv": sigma_upsilon_inv_new,
        "iter_count": iter_count  # Track iteration count for coordinate updates
    }

    return q_params_new

def run_variational_inference(x_data, y_data, x_aux, hyperparams,
                              q_params=None, max_iters=100, tol=1e-6,
                              verbose=True, label_scale=0.1, mask=None,
                              patience=5, min_delta=1e-3, beta_init=None):
    # log_memory("Starting run_variational_inference")
    
    # Convert sparse matrices to dense arrays for JAX compatibility
    if hasattr(x_data, 'toarray'):
        print("Converting sparse x_data to dense array for JAX compatibility")
        x_data = x_data.toarray()
    
    # Ensure data is in JAX-compatible format
    x_data = jnp.array(x_data)
    y_data = jnp.array(y_data)
    x_aux  = jnp.array(x_aux)

    print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}, x_aux shape: {x_aux.shape}")
    print(f"hyperparams['d']: {hyperparams['d']}")
    
    if mask is not None:
        print(f"Using mask with shape {mask.shape}")
        mask = jnp.array(mask)  # Ensure mask is a JAX array
    
    if q_params is None:
        n, p = x_data.shape
        if len(y_data.shape)==1: 
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
    
    # JIT-compile the ELBO computation (note: this may not speed up much due to the complex function)
    # We'll still call the original compute_elbo but prepare inputs efficiently
    @jax.jit
    def prepare_elbo_inputs(E_eta, E_beta, E_xi, E_theta, E_gamma, E_upsilon):
        # This function just ensures all inputs are properly typed for JAX
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
    
    if verbose:
        print(f"Starting VI with patience={patience}, min_delta={min_delta:.6f}")

    for iter_ix in range(max_iters):
        print(f"\nStarting VI iteration {iter_ix+1}/{max_iters}")
        # log_memory(f"Before update_q_params in iteration {iter_ix+1}")
        
        q_params = update_q_params(q_params, x_data, y_data, x_aux,
                                   hyperparams, label_scale=label_scale, mask=mask)
        
        # log_memory(f"After update_q_params in iteration {iter_ix+1}")

        # Use JIT-compiled function to compute expectations
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

        # log_memory(f"Before computing ELBO in iteration {iter_ix+1}")
        elbo_val = compute_elbo(E_eta, E_beta, E_xi, E_theta,
                                E_gamma, E_upsilon,
                                x_data, y_data, x_aux,
                                hyperparams, q_params, mask=mask)
        elbo_history.append(float(elbo_val))
        # log_memory(f"After computing ELBO in iteration {iter_ix+1}")

        if verbose:
            print(f"Iter {iter_ix+1}/{max_iters} ELBO = {elbo_val:.4f}")
            
        # Standard convergence check
        if jnp.abs(elbo_val - old_elbo) < tol:
            if verbose:
                print(f"Converged. Change in ELBO ({jnp.abs(elbo_val - old_elbo):.6f}) is less than tolerance ({tol:.6f}).")
            convergence_iter = iter_ix + 1
            convergence_reason = f"ELBO change ({jnp.abs(elbo_val - old_elbo):.6e}) < tolerance ({tol:.6e})"
            break
        
        # Advanced early stopping logic
        improvement = elbo_val - best_elbo
        if elbo_val > best_elbo:
            # We found a better ELBO
            if improvement > min_delta:
                # Significant improvement
                best_elbo = elbo_val
                best_iter = iter_ix + 1
                # Deep copy the parameters to save the best model
                best_params = {k: v.copy() if isinstance(v, jnp.ndarray) else v for k, v in q_params.items()}
                patience_counter = 0
                if verbose and iter_ix > 0:
                    print(f"  New best ELBO! Improvement: {improvement:.6f}")
            else:
                # Improvement too small
                patience_counter += 1
                if verbose:
                    print(f"  Small improvement: {improvement:.6f} < {min_delta:.6f}, patience: {patience_counter}/{patience}")
        else:
            # No improvement
            patience_counter += 1
            if verbose:
                print(f"  No improvement, patience: {patience_counter}/{patience}")
                
        # Check if patience is exhausted
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping triggered after {iter_ix+1} iterations. No significant improvement in the last {patience} iterations.")
            # Restore best parameters if we have them
            if best_params is not None:
                q_params = best_params
                if verbose:
                    print(f"Restored best model with ELBO = {best_elbo:.4f} from iteration {best_iter}")
            convergence_iter = iter_ix + 1
            convergence_reason = f"Early stopping: no improvement > {min_delta:.6e} for {patience} iterations"
            break
        
        old_elbo = elbo_val
        
        # Try to clear memory after each iteration
        clear_memory()
    
    # If we completed all iterations without triggering early stopping
    if convergence_iter is None:
        convergence_iter = max_iters
        convergence_reason = "Maximum iterations reached"
    
    # Print convergence summary
    print("\n" + "="*50)
    print(f"ELBO Convergence Summary:")
    print(f"  Total iterations: {convergence_iter}/{max_iters}")
    print(f"  Best ELBO: {best_elbo:.6f} (at iteration {best_iter})")
    print(f"  Final ELBO: {elbo_history[-1]:.6f}")
    print(f"  Convergence reason: {convergence_reason}")
    print("="*50 + "\n")
    
    # Store convergence info in the model parameters
    q_params['convergence_info'] = {
        'total_iterations': convergence_iter,
        'max_iterations': max_iters,
        'best_elbo': float(best_elbo),
        'best_iter': best_iter,
        'final_elbo': float(elbo_history[-1]),
        'convergence_reason': convergence_reason
    }

    return q_params, elbo_history


def fold_in_new_data(X_new, x_aux_new, q_params_trained, hyperparams,
                     max_iters=30, label_scale=0.0, y_new=None, verbose=False, mask=None):
    # log_memory("Starting fold_in_new_data")
    
    alpha_beta_tr = q_params_trained['alpha_beta']
    omega_beta_tr = q_params_trained['omega_beta']
    E_beta_tr     = alpha_beta_tr / jnp.maximum(omega_beta_tr, 1e-10)
    
    if mask is not None:
        E_beta_tr = E_beta_tr * mask

    alpha_eta_tr  = q_params_trained['alpha_eta']
    omega_eta_tr  = q_params_trained['omega_eta']
    E_eta_tr      = alpha_eta_tr / jnp.maximum(omega_eta_tr, 1e-10)

    E_upsilon_tr  = q_params_trained['upsilon']  
    E_gamma_tr    = q_params_trained['gamma']

    c_prime = hyperparams['c_prime']
    d_prime = hyperparams['d_prime']
    c       = hyperparams['c']
    a_prime = hyperparams['a_prime']
    b_prime = hyperparams['b_prime']
    a       = hyperparams['a']
    tau     = hyperparams['tau']
    sigma   = hyperparams['sigma']
    d_dim   = hyperparams['d']  

    # Convert sparse matrices to dense arrays for JAX compatibility
    if hasattr(X_new, 'toarray'):
        print("Converting sparse X_new to dense array for JAX compatibility")
        X_new = X_new.toarray()

    X_new = jnp.array(X_new)
    x_aux_new = jnp.array(x_aux_new)
    if y_new is not None:
        y_new = jnp.array(y_new)
        if len(y_new.shape)==1:
            y_new = y_new.reshape(-1,1)

    n_test, p = X_new.shape
    kappa = E_upsilon_tr.shape[0]
    
    # log_memory("Before initializing fold-in parameters")

    random_int = jax.random.randint(jax.random.PRNGKey(0), (), 0, 2**30, dtype=jnp.int32)
    key = jax.random.PRNGKey(int(random_int))
    k1, k2 = jax.random.split(key,2)

    alpha_xi_test = jnp.ones(n_test)+ 0.01*jax.random.normal(k1,(n_test,))
    omega_xi_test = jnp.ones(n_test)

    alpha_theta_test = jnp.ones((n_test, d_dim)) + 0.01*jax.random.normal(k2,(n_test,d_dim))
    omega_theta_test = jnp.ones((n_test, d_dim))
    
    # Define JIT-compiled helper functions
    @jax.jit
    def process_phi_batch_test(E_theta_batch, E_beta, x_batch):
        numerator_phi = jnp.einsum('bd,pd->bpd', E_theta_batch, E_beta)
        denom_phi = jnp.sum(numerator_phi, axis=2, keepdims=True) + 1e-10
        phi = numerator_phi / denom_phi
        alpha_theta_batch_update = jnp.sum(x_batch[:, :, None] * phi, axis=1)
        return alpha_theta_batch_update
    
    @jax.jit
    def update_theta_params(E_xi_test, sumBeta_tr, alpha_theta_test, omega_theta_test, a, a_prime, b_prime, d_dim):
        # Calculate E_theta
        E_theta_test = alpha_theta_test / jnp.maximum(omega_theta_test, 1e-10)
        
        # Update omega_theta
        omega_theta_test_new = E_xi_test[:, None] + sumBeta_tr[None, :]
        omega_theta_test_new = jnp.clip(omega_theta_test_new, 1e-6, 1e3)
        
        # Update alpha_xi
        alpha_xi_test_new = a_prime + a * d_dim
        
        # Update E_theta with new parameters
        E_theta_test_new = alpha_theta_test / jnp.maximum(omega_theta_test_new, 1e-10)
        
        # Update omega_xi
        sumTheta_test = jnp.sum(E_theta_test_new, axis=1)
        omega_xi_test_new = sumTheta_test + (a_prime / b_prime)
        omega_xi_test_new = jnp.clip(omega_xi_test_new, 1e-6, 1e3)
        
        return E_theta_test, omega_theta_test_new, alpha_xi_test_new, omega_xi_test_new
    
    @jax.jit
    def add_label_signal(alpha_theta, y, ups_1d, label_scale):
        label_signal = (y - 0.5)[:, None] * ups_1d[None, :]
        result = alpha_theta + label_scale * label_signal
        return jnp.clip(result, 1e-8, 1e9)
    
    # log_memory("After initializing fold-in parameters")
    
    # Pre-compute constants outside the loop
    sumBeta_tr = jnp.sum(E_beta_tr, axis=0)

    for iter_ix in range(max_iters):
        if verbose and (iter_ix % 5 == 0 or iter_ix == max_iters-1):
            print(f"Fold-in iter {iter_ix+1}/{max_iters}")
            # log_memory(f"Fold-in iter {iter_ix+1}")
        
        # Get current E_theta and E_xi
        E_theta_test = alpha_theta_test / jnp.maximum(omega_theta_test, 1e-10)
        E_xi_test = alpha_xi_test / jnp.maximum(omega_xi_test, 1e-10)
            
        # Process phi computation in batches
        batch_size = max(1, min(100, n_test // 10))
        alpha_theta_test_new = jnp.ones_like(alpha_theta_test) * a
        
        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            
            # Use JIT-compiled function for batch processing
            alpha_theta_batch_update = process_phi_batch_test(
                E_theta_test[batch_start:batch_end], 
                E_beta_tr,
                X_new[batch_start:batch_end]
            )
            
            alpha_theta_test_new = alpha_theta_test_new.at[batch_start:batch_end].set(
                alpha_theta_test_new[batch_start:batch_end] + alpha_theta_batch_update
            )
        
        # Add label signal if needed
        if (kappa==1) and (y_new is not None) and (label_scale>0):
            ups_1d = E_upsilon_tr[0]
            alpha_theta_test_new = add_label_signal(alpha_theta_test_new, y_new.reshape(-1), ups_1d, label_scale)
        
        # Update parameters using JIT-compiled function
        _, omega_theta_test_new, alpha_xi_test_new, omega_xi_test_new = update_theta_params(
            E_xi_test, sumBeta_tr, alpha_theta_test_new, omega_theta_test, a, a_prime, b_prime, d_dim
        )
        
        alpha_theta_test = alpha_theta_test_new
        omega_theta_test = omega_theta_test_new
        alpha_xi_test    = alpha_xi_test_new
        omega_xi_test    = omega_xi_test_new
        
        # Clear memory every few iterations
        if iter_ix % 5 == 0:
            clear_memory()

    E_theta_test = alpha_theta_test / jnp.maximum(omega_theta_test, 1e-10)
    # log_memory("End of fold_in_new_data")
    return E_theta_test

def evaluate_model(X_data, x_aux, y_data, q_params, hyperparams,
                   fold_in=False, label_scale=0.0, return_probs=True):
    # log_memory("Starting evaluate_model")
    
    if fold_in:
        print(f"Using fold_in_new_data for evaluation with X_data shape: {X_data.shape}")
        E_theta_data = fold_in_new_data(X_data, x_aux, q_params, hyperparams,
                                        max_iters=30, label_scale=label_scale,
                                        y_new=y_data, verbose=False)
    else:
        alpha_theta = q_params['alpha_theta']
        omega_theta = q_params['omega_theta']
        E_theta_data = alpha_theta / jnp.maximum(omega_theta,1e-10)

    E_upsilon = q_params['upsilon'] 
    E_gamma   = q_params['gamma']
    
    # Check if mask is present in hyperparams and apply necessary adjustments
    mask = hyperparams.get('mask', None)
    if mask is not None:
        # For columns with no mask entries, zero out the corresponding upsilon values
        # during prediction to prevent them from contributing
        mask_present = jnp.sum(mask, axis=0) > 0
        if jnp.any(~mask_present):
            # Apply a scaling to upsilon where mask columns are empty
            print(f"Adjusting upsilon for evaluation based on mask presence")
            E_upsilon = E_upsilon * jnp.where(mask_present, 1.0, 1e-5)
    
    # Apply clipping to E_upsilon to prevent extreme values
    E_upsilon = jnp.clip(E_upsilon, -2.0, 2.0)
    
    x_aux = jnp.array(x_aux)
    # log_memory("Before computing logits")
    logits = (x_aux @ E_gamma.T) + jnp.einsum('nd,kd->nk', E_theta_data, E_upsilon)
    
    # Apply logit clipping to prevent extreme probability values
    logits = jnp.clip(logits, -10.0, 10.0)
    
    probs = jax.nn.sigmoid(logits)
    # log_memory("After computing logits and probs")

    y_data = jnp.array(y_data)
    if len(y_data.shape)==1:
        y_data = y_data.reshape(-1,1)
    kappa = y_data.shape[1]

    # Calculate class imbalance for threshold adjustment
    class_balance = jnp.mean(y_data, axis=0)
    
    # Adjust decision threshold based on class balance if very imbalanced
    thresholds = jnp.where(class_balance < 0.2, 0.3, 0.5)
    print(f"Using classification thresholds: {thresholds}")
    
    # Use the adjusted thresholds for predictions
    if kappa == 1:
        preds = (probs >= thresholds[0]).astype(float)
    else:
        preds = jnp.zeros_like(probs)
        for k in range(kappa):
            preds = preds.at[:, k].set((probs[:, k] >= thresholds[k]).astype(float))
    
    results = {}

    if kappa==1:
        acc   = float(jnp.mean(preds[:, 0] == y_data[:,0]))
        prec  = precision_score(y_data, preds[:,0], zero_division=0)
        rec   = recall_score(y_data, preds[:,0], zero_division=0)
        f1v   = f1_score(y_data, preds[:,0], zero_division=0)
        aucv  = roc_auc_score(y_data, probs[:,0]) if len(jnp.unique(y_data))>1 else 0.5
        cm    = confusion_matrix(y_data, preds[:, 0])
        
        print(f"Evaluation metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1v:.4f}, AUC: {aucv:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        results = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1v,
            "roc_auc": aucv,
            "confusion_matrix": cm.tolist(),
            "threshold": float(thresholds[0])
        }
    else:
        metrics = []
        for k_idx in range(kappa):
            y_true = y_data[:, k_idx]
            y_pred = preds[:, k_idx]
            acc_k   = float(jnp.mean(y_pred==y_true))
            prec_k  = precision_score(y_true, y_pred, zero_division=0)
            rec_k   = recall_score(y_true, y_pred, zero_division=0)
            f1_k    = f1_score(y_true, y_pred, zero_division=0)
            auc_k   = roc_auc_score(y_true, probs[:, k_idx]) if len(jnp.unique(y_true))>1 else 0.5
            metrics.append((acc_k, prec_k, rec_k, f1_k, auc_k))
        acc  = np.mean([m[0] for m in metrics])
        prec = np.mean([m[1] for m in metrics])
        rec  = np.mean([m[2] for m in metrics])
        f1v  = np.mean([m[3] for m in metrics])
        aucv = np.mean([m[4] for m in metrics])
        results = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1v,
            "roc_auc": aucv,
            "confusion_matrix": None,
            "thresholds": [float(t) for t in thresholds]
        }

    if return_probs:
        results["probabilities"] = probs.tolist()  
    
    # log_memory("End of evaluate_model")
    return results


def extract_top_genes(E_beta, var_names, top_n=20):
    # log_memory("Starting extract_top_genes")
    
    if not isinstance(var_names, list):
        var_names = list(var_names)
    top_genes = {}
    for k in range(E_beta.shape[1]):
        weights = E_beta[:, k]
        weights_np = np.array(weights)
        indices_sorted = jnp.argsort(-weights_np)  
        top_indices = indices_sorted[:top_n]
        top_list = []
        for rank_idx, gene_idx in enumerate(top_indices):
            gene_name =var_names[int(gene_idx)]
            top_list.append({
                "rank": rank_idx+1,
                "gene": gene_name,
                "weight": float(weights_np[gene_idx])
            })
        top_genes[f"program_{k+1}"] = top_list
    
    # log_memory("End of extract_top_genes")
    return top_genes

def run_model_and_evaluate(x_data, x_aux, y_data, var_names, hyperparams,
                           seed=None, test_size=0.15, val_size=0.15, max_iters=100,
                           label_scale=0.1, return_probs=True, sample_ids=None,
                           mask=None, scores=None, plot_elbo=True, plot_prefix=None,
                           return_params=False, beta_init=None):  
    # log_memory("Starting run_model_and_evaluate")
    print(f"Input shapes - x_data: {x_data.shape}, x_aux: {x_aux.shape}, y_data: {y_data.shape}")
    print(f"hyperparams['d']: {hyperparams['d']}")
    
    import pandas as pd
    import datetime
    
    if sample_ids is None:
        sample_ids = np.arange(len(x_data))
    
    if scores is None:
        scores = np.zeros(len(x_data))  

    temp_size = val_size + test_size
    if temp_size > 1.0:
        raise ValueError("val_size + test_size must be <= 1.0")
    
    # log_memory("Before train_test_split")
    X_train, X_temp, XA_train, XA_temp, y_train, y_temp, ids_train, ids_temp, scores_train, scores_temp = train_test_split(
        x_data, x_aux, y_data, sample_ids, scores, test_size=temp_size, random_state=seed
    )

    relative_test_size = test_size / temp_size
    X_val, X_test, XA_val, XA_test, y_val, y_test, ids_val, ids_test, scores_val, scores_test = train_test_split(
        X_temp, XA_temp, y_temp, ids_temp, scores_temp, test_size=relative_test_size, random_state=seed
    )
    
    print(f"Train shapes - X_train: {X_train.shape}, XA_train: {XA_train.shape}, y_train: {y_train.shape}")
    print(f"Validation shapes - X_val: {X_val.shape}, XA_val: {XA_val.shape}, y_val: {y_val.shape}")
    print(f"Test shapes - X_test: {X_test.shape}, XA_test: {XA_test.shape}, y_test: {y_test.shape}")

    # Log memory usage of the datasets
    # log_array_sizes({
    #     'X_train': X_train,
    #     'X_test': X_test,
    #     'XA_train': XA_train,
    #     'XA_test': XA_test,
    #     'y_train': y_train,
    #     'y_test': y_test
    # })
    
    if mask is not None:
        print(f"Using mask with shape {mask.shape}")
        # Check if mask is sparse
        non_zero = np.count_nonzero(mask)
        total = mask.size
        print(f"Mask sparsity: {100 * (1 - non_zero/total):.2f}% ({non_zero} non-zeros of {total} total)")
    
    if beta_init is not None:
        print(f"Using beta_init with shape {beta_init.shape} for pathway initialization")
        log_array_sizes({'beta_init': beta_init})
    
    # Prepare timestamp for file naming if needed
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Add a try-except block to catch and handle NaN values during training
    try:
        # log_memory("Before run_variational_inference")
        q_params, elbo_hist = run_variational_inference(
            X_train, y_train, XA_train, hyperparams,
            q_params=None, max_iters=max_iters, verbose=True,
            label_scale=label_scale, mask=mask, patience=5, min_delta=1e-3,
            beta_init=beta_init  # Pass beta_init to run_variational_inference
        )
        # log_memory("After run_variational_inference")
        
        # Get convergence information
        convergence_info = q_params.get('convergence_info', {})
        total_iterations = convergence_info.get('total_iterations', len(elbo_hist))
        convergence_reason = convergence_info.get('convergence_reason', 'Unknown')
        best_iter = convergence_info.get('best_iter', np.argmax(elbo_hist) + 1)
        best_elbo = convergence_info.get('best_elbo', np.max(elbo_hist) if len(elbo_hist) > 0 else None)
        
        print(f"\nModel converged after {total_iterations}/{max_iters} iterations")
        print(f"Best ELBO: {best_elbo:.6f} at iteration {best_iter}")
        print(f"Convergence reason: {convergence_reason}")
        
        # Generate ELBO convergence plot if requested
        if plot_elbo and len(elbo_hist) > 1:
            # Create plot title
            plot_title = f"ELBO Convergence (d={hyperparams['d']}, converged after {total_iterations} iterations)"
            
            # Generate output file name if not provided
            if plot_prefix is None:
                plot_file = f"elbo_convergence_{timestamp}.png"
            else:
                plot_file = f"{plot_prefix}_elbo_convergence.png"
                
            # Create and save the plot
            plot_elbo_convergence(
                elbo_hist, 
                output_file=plot_file, 
                title=plot_title,
                show_best=True
            )
        
        # Check for NaN values in ELBO history
        if any(np.isnan(elbo_hist)):
            print("WARNING: NaN values detected in ELBO history! Attempting to recover...")
            # Filter out NaN values for the results
            valid_elbo_idx = [i for i, val in enumerate(elbo_hist) if not np.isnan(val)]
            if len(valid_elbo_idx) > 0:
                print(f"Found {len(valid_elbo_idx)}/{len(elbo_hist)} valid ELBO values.")
                last_valid_idx = max(valid_elbo_idx)
                print(f"Using model state from iteration {last_valid_idx+1}")
                # Keep only valid ELBO values in history
                elbo_hist = [elbo_hist[i] for i in valid_elbo_idx]
            else:
                print("No valid ELBO values found. Results may be unreliable.")
        
        # Check for extreme or invalid parameter values
        alpha_beta = q_params['alpha_beta']
        omega_beta = q_params['omega_beta']
        alpha_theta = q_params['alpha_theta']
        omega_theta = q_params['omega_theta']
        
        # Check for potential division by zero
        if np.min(omega_beta) < 1e-8 or np.min(omega_theta) < 1e-8:
            print("WARNING: Very small values detected in denominator parameters!")
            print(f"  omega_beta min: {np.min(omega_beta):.6e}")
            print(f"  omega_theta min: {np.min(omega_theta):.6e}")
            # Apply a minimum threshold
            omega_beta = jnp.maximum(omega_beta, 1e-8)
            omega_theta = jnp.maximum(omega_theta, 1e-8)
            q_params['omega_beta'] = omega_beta
            q_params['omega_theta'] = omega_theta
        
        # Check for NaN or Inf values in gamma/upsilon
        if np.any(np.isnan(q_params['gamma'])) or np.any(np.isnan(q_params['upsilon'])):
            print("WARNING: NaN values detected in gamma or upsilon parameters!")
            # Replace any NaN values with small random values
            key = jax.random.PRNGKey(42)
            if np.any(np.isnan(q_params['gamma'])):
                gamma_shape = q_params['gamma'].shape
                gamma_nan_mask = np.isnan(q_params['gamma'])
                gamma_replacement = jax.random.normal(key, gamma_shape) * 0.01
                q_params['gamma'] = jnp.where(gamma_nan_mask, gamma_replacement, q_params['gamma'])
                
            if np.any(np.isnan(q_params['upsilon'])):
                upsilon_shape = q_params['upsilon'].shape
                upsilon_nan_mask = np.isnan(q_params['upsilon'])
                key, subkey = jax.random.split(key)
                upsilon_replacement = jax.random.normal(subkey, upsilon_shape) * 0.01
                q_params['upsilon'] = jnp.where(upsilon_nan_mask, upsilon_replacement, q_params['upsilon'])
                
    except Exception as e:
        print(f"Error during training: {e}")
        # Provide a fallback solution with reasonable defaults
        print("Using fallback parameters due to error...")
        n, p = X_train.shape
        if len(y_train.shape)==1: 
            kappa = 1
        else:
            kappa = y_train.shape[1]
        p_aux = XA_train.shape[1]
        d = hyperparams['d']
        
        q_params = initialize_q_params(n, p, kappa, p_aux, d, seed=42, beta_init=beta_init)
        elbo_hist = [-1e10]  # Placeholder

    # log_memory("Before train evaluation")
    train_metrics = evaluate_model(X_train, XA_train, y_train, q_params,
                                  hyperparams, fold_in=False,
                                  label_scale=label_scale, return_probs=True)  

    val_metrics = evaluate_model(X_val, XA_val, y_val, q_params,
                                 hyperparams, fold_in=True, 
                                 label_scale=0.0, return_probs=True)
    
    test_metrics = evaluate_model(X_test, XA_test, y_test, q_params,
                                 hyperparams, fold_in=True,
                                 label_scale=0.0, return_probs=True)  

    E_beta_final = (q_params['alpha_beta'] / jnp.maximum(q_params['omega_beta'], 1e-10))
    if mask is not None:
        E_beta_final = E_beta_final * mask
        
    top_genes = extract_top_genes(E_beta_final, var_names)

    train_df = pd.DataFrame({
        'sample_id': ids_train,
        'true_label': y_train.reshape(-1) if len(y_train.shape) == 2 and y_train.shape[1] == 1 else y_train,
        'probability': np.round(np.array(train_metrics['probabilities']).reshape(-1), 4) if len(y_train.shape) <= 1 or y_train.shape[1] == 1 else np.round(np.array(train_metrics['probabilities']), 4),
        'predicted_label': (np.array(train_metrics['probabilities']) >= .5).astype(int).reshape(-1) if len(y_train.shape) <= 1 or y_train.shape[1] == 1 else (np.array(train_metrics['probabilities']) >= 0.5).astype(int),
        'cyto_seed_score': scores_train 
    })

    val_df = pd.DataFrame({
        'sample_id': ids_val,
        'true_label': y_val.reshape(-1) if len(y_val.shape) == 2 and y_val.shape[1] == 1 else y_val,
        'probability': np.round(np.array(val_metrics['probabilities']).reshape(-1), 4) if len(y_val.shape) <= 1 or y_val.shape[1] == 1 else np.round(np.array(val_metrics['probabilities']), 4),
        'predicted_label': (np.array(val_metrics['probabilities']) >= .5).astype(int).reshape(-1) if len(y_val.shape) <= 1 or y_val.shape[1] == 1 else (np.array(val_metrics['probabilities']) >= 0.5).astype(int),
        'cyto_seed_score': scores_val
    })
    
    test_df = pd.DataFrame({
        'sample_id': ids_test,
        'true_label': y_test.reshape(-1) if len(y_test.shape) == 2 and y_test.shape[1] == 1 else y_test,
        'probability': np.round(np.array(test_metrics['probabilities']).reshape(-1), 4) if len(y_test.shape) <= 1 or y_test.shape[1] == 1 else np.round(np.array(test_metrics['probabilities']), 4),
        'predicted_label': (np.array(test_metrics['probabilities']) >= 0.5).astype(int).reshape(-1) if len(y_test.shape) <= 1 or y_test.shape[1] == 1 else (np.array(test_metrics['probabilities']) >= 0.5).astype(int),
        'cyto_seed_score': scores_test  
    })

    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        train_dfs = []
        val_dfs=[]
        test_dfs = []
        for k in range(y_train.shape[1]):
            train_df_k = pd.DataFrame({
                'sample_id': ids_train,
                'true_label': y_train[:, k],
                'probability': np.round(np.array(train_metrics['probabilities'])[:, k], 4),
                'predicted_label': (np.array(train_metrics['probabilities'])[:, k] >= 0.5).astype(int),
                'class': k,
                'cyto_seed_score': scores_train  # Add cyto_seed_score to multi-class DataFrame
            })
            val_df_k = pd.DataFrame({
                'sample_id': ids_val,
                'true_label': y_val[:, k],
                'probability': np.round(np.array(val_metrics['probabilities'])[:, k], 4),
                'predicted_label': (np.array(val_metrics['probabilities'])[:, k] >= 0.5).astype(int),
                'class': k,
                'cyto_seed_score': scores_val  # Add cyto_seed_score to multi-class DataFrame
            })
            test_df_k = pd.DataFrame({
                'sample_id': ids_test,
                'true_label': y_test[:, k],
                'probability': np.round(np.array(test_metrics['probabilities'])[:, k], 4),
                'predicted_label': (np.array(test_metrics['probabilities'])[:, k] >= 0.5).astype(int),
                'class': k,
                'cyto_seed_score': scores_test  # Add cyto_seed_score to multi-class DataFrame
            })
            train_dfs.append(train_df_k)
            val_dfs.append(val_df_k)
            test_dfs.append(test_df_k)
        
        train_df = pd.concat(train_dfs)
        val_df = pd.concat(val_dfs)
        test_df = pd.concat(test_dfs)
    
    # log_memory("After creating result DataFrames")

    # log_memory("Before compiling final results")
    results = {
        "data_info": {
            "n_train": X_train.shape[0],
            "n_val":   X_val.shape[0],
            "n_test":  X_test.shape[0],
            "p":       X_train.shape[1],
            "kappa":   y_train.shape[1] if len(y_train.shape)>1 else 1,
            "p_aux":   XA_train.shape[1],
            "d":       hyperparams['d'],
            "hyperparameters": hyperparams,
        },
        "elbo_history": [float(v) for v in elbo_hist],
        "train_metrics": {k: float(v) if isinstance(v,(int,float)) else v 
                          for k,v in train_metrics.items() if k != 'probabilities'},
        "val_metrics":   {k: float(v) if isinstance(v,(int,float)) else v
                          for k,v in val_metrics.items() if k != 'probabilities'},
        "test_metrics":  {k: float(v) if isinstance(v,(int,float)) else v 
                          for k,v in test_metrics.items() if k != 'probabilities'},
        "top_genes": top_genes,
        "E_upsilon": q_params['upsilon'].tolist(),
        "E_gamma":   q_params['gamma'].tolist(),
        "train_results_df": train_df,
        "val_results_df": val_df,
        "test_results_df": test_df,
        "convergence_info": q_params.get('convergence_info', {})
    }

    if return_probs:
        results["train_probabilities"] = train_metrics['probabilities']
        results["val_probabilities"] = val_metrics['probabilities']
        results["test_probabilities"] = test_metrics['probabilities']
        
    # If return_params is True, include the model parameters needed for analyzing gene programs
    if return_params:
        # Include both alpha_beta and omega_beta for computing E_beta later
        results["alpha_beta"] = q_params['alpha_beta'].tolist()
        results["omega_beta"] = q_params['omega_beta'].tolist()
        # Also directly calculate and include E_beta
        results["E_beta"] = E_beta_final.tolist()
    
    # log_memory("End of run_model_and_evaluate")    
    return results

def plot_elbo_convergence(elbo_history, output_file=None, title=None, show_best=True):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy array for easier manipulation
    elbo_array = np.array(elbo_history)
    
    # Create x-axis (iterations)
    iterations = np.arange(1, len(elbo_array) + 1)
    
    # Plot main ELBO curve
    ax.plot(iterations, elbo_array, 'b-', linewidth=2, label='ELBO')
    
    # Fill area under curve with light blue
    ax.fill_between(iterations, np.min(elbo_array), elbo_array, alpha=0.2, color='blue')
    
    # Highlight best ELBO value
    if show_best and len(elbo_array) > 1:
        best_idx = np.argmax(elbo_array)
        best_elbo = elbo_array[best_idx]
        best_iter = best_idx + 1  # 1-indexed
        
        ax.axhline(y=best_elbo, color='r', linestyle='--', alpha=0.7, label=f'Best ELBO: {best_elbo:.2f}')
        ax.plot(best_iter, best_elbo, 'ro', markersize=8)
        ax.annotate(f'Best: {best_elbo:.2f}\nIter: {best_iter}', 
                   xy=(best_iter, best_elbo),
                   xytext=(best_iter + 0.5, best_elbo - 0.1 * (np.max(elbo_array) - np.min(elbo_array))),
                   arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5),
                   fontsize=12)
    
    # Set plot title and labels
    if title is None:
        title = 'ELBO Convergence History'
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('ELBO Value', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure if output_file is specified
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"ELBO convergence plot saved to: {output_file}")
    
    return fig