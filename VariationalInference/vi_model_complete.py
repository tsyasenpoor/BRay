import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from scipy.special import expit, kv
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
    log_phi_unnormalized = jnp.einsum('bd,pd->bpd', jnp.log(jnp.maximum(theta_batch, 1e-10)), jnp.log(jnp.maximum(beta, 1e-10)))
    phi_batch = jax.nn.softmax(log_phi_unnormalized, axis=2)
    
    alpha_beta_batch_update = jnp.sum(x_batch[:, :, None] * phi_batch, axis=0)
    alpha_theta_batch_update = jnp.sum(x_batch[:, :, None] * phi_batch, axis=1)
    return alpha_beta_batch_update, alpha_theta_batch_update

@partial(jax.jit, static_argnums=(3,)) 
def update_regression_variational(mu_gamma_old, Sigma_gamma_old, mu_upsilon_old, Sigma_upsilon_old, 
                                  gig_a_ups, gig_b_ups_old,
                                  y_data, x_aux, hyperparams, E_theta, E_theta_theta_T):
    
    
    sigma_sq = hyperparams['sigma']**2
    p_aux = x_aux.shape[1]
    d = E_theta.shape[1]
    n, kappa = y_data.shape

    E_gamma = mu_gamma_old
    E_upsilon = mu_upsilon_old
    E_gamma_gamma_T = jnp.array([S + jnp.outer(m, m) for m, S in zip(E_gamma, Sigma_gamma_old)])
    E_upsilon_upsilon_T = jnp.array([S + jnp.outer(m, m) for m, S in zip(E_upsilon, Sigma_upsilon_old)])

    zeta_sq = jnp.zeros((n, kappa))
    for k in range(kappa):
        term1 = jnp.einsum('ii', E_theta_theta_T @ E_upsilon_upsilon_T[k])
        term2 = jnp.einsum('ij,ji->i', x_aux @ E_gamma_gamma_T[k], x_aux.T)
        term3 = 2 * (E_theta @ E_upsilon[k]) * (x_aux @ E_gamma[k])
        zeta_sq = zeta_sq.at[:, k].set(term1 + term2 + term3)
    
    zeta = jnp.sqrt(jnp.maximum(zeta_sq, 1e-10))
    lambda_zeta = jnp.tanh(zeta / 2) / jnp.maximum(4 * zeta, 1e-10)
    lambda_zeta = jnp.where(zeta < 1e-5, 1/8, lambda_zeta)

    E_upsilon_sq = jnp.array([jnp.diag(S) + m**2 for m, S in zip(mu_upsilon_old, Sigma_upsilon_old)])
    gig_b_ups_new = E_upsilon_sq
    
    E_inv_lambda_sq_ups = jnp.sqrt(gig_a_ups / jnp.maximum(gig_b_ups_new, 1e-10))

    mu_gamma_new = jnp.zeros_like(mu_gamma_old)
    Sigma_gamma_new = jnp.zeros_like(Sigma_gamma_old)
    mu_upsilon_new = jnp.zeros_like(mu_upsilon_old)
    Sigma_upsilon_new = jnp.zeros_like(Sigma_upsilon_old)

    for k in range(kappa):
        lambda_zeta_k = lambda_zeta[:, k]
        
        Sigma_gamma_inv_k = (1/sigma_sq) * jnp.eye(p_aux) + 2 * (x_aux.T * lambda_zeta_k) @ x_aux
        linear_term_gamma_k = (y_data[:, k] - 0.5) @ x_aux - 2 * ((lambda_zeta_k * (E_theta @ E_upsilon[k])) @ x_aux)
        
        Sigma_gamma_k = jnp.linalg.inv(Sigma_gamma_inv_k + 1e-4 * jnp.eye(p_aux))
        mu_gamma_k = Sigma_gamma_k @ linear_term_gamma_k
        
        mu_gamma_new = mu_gamma_new.at[k].set(mu_gamma_k)
        Sigma_gamma_new = Sigma_gamma_new.at[k].set(Sigma_gamma_k)

        prior_precision_diag = E_inv_lambda_sq_ups[k, :]
        Sigma_upsilon_inv_k = 2 * jnp.einsum('i,ij,ik->jk', lambda_zeta_k, E_theta, E_theta) + jnp.diag(prior_precision_diag)
        linear_term_upsilon_k = (y_data[:, k] - 0.5) @ E_theta - 2 * (lambda_zeta_k * (x_aux @ E_gamma[k])) @ E_theta

        Sigma_upsilon_k = jnp.linalg.inv(Sigma_upsilon_inv_k + 1e-4 * jnp.eye(d))
        mu_upsilon_k = Sigma_upsilon_k @ linear_term_upsilon_k
        
        mu_upsilon_new = mu_upsilon_new.at[k].set(mu_upsilon_k)
        Sigma_upsilon_new = Sigma_upsilon_new.at[k].set(Sigma_upsilon_k)
        
    return mu_gamma_new, Sigma_gamma_new, mu_upsilon_new, Sigma_upsilon_new, gig_b_ups_new


def initialize_q_params(n, p, kappa, p_aux, d, seed=None, beta_init=None):
    log_memory(f"Before initialize_q_params (n={n}, p={p}, kappa={kappa}, p_aux={p_aux}, d={d})")
    
    if seed is None:
        random_int = jax.random.randint(jax.random.PRNGKey(0), (), 0, 2**30, dtype=jnp.int32)
        key = jax.random.PRNGKey(int(random_int))
    else:
        key = jax.random.PRNGKey(seed)

    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 8)
    
    alpha_eta= jnp.ones(p)+ 0.01 * jax.random.normal(k1, (p,))
    
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

    mu_gamma = jnp.array(jax.random.normal(k5, (kappa, p_aux))) * 0.01
    Sigma_gamma = jnp.array([jnp.eye(p_aux) for _ in range(kappa)])

    mu_upsilon = jnp.array(jax.random.normal(k6, (kappa, d))) * 0.1
    Sigma_upsilon = jnp.array([jnp.eye(d) for _ in range(kappa)])

    tau = hyperparams['tau'] 
    gig_p_ups = 0.5

    gig_a_ups = jnp.ones((kappa, d)) / (2 * tau**2)
    gig_b_ups = jnp.ones((kappa, d))
    
    q_params = {
        "alpha_eta": alpha_eta,
        "omega_eta": omega_eta,
        "alpha_beta": alpha_beta,
        "omega_beta": omega_beta,
        "alpha_xi": alpha_xi,
        "omega_xi": omega_xi,
        "alpha_theta": alpha_theta,
        "omega_theta": omega_theta,
        "mu_gamma": mu_gamma,
        "Sigma_gamma": Sigma_gamma,
        "mu_upsilon": mu_upsilon,
        "Sigma_upsilon": Sigma_upsilon,
        "gig_p_ups": gig_p_ups,
        "gig_a_ups": gig_a_ups,
        "gig_b_ups": gig_b_ups,
    }
    
    return q_params


def update_q_params(q_params, x_data, y_data, x_aux, hyperparams, mask=None):
    x_data = jnp.array(x_data)
    y_data = jnp.array(y_data)
    if len(y_data.shape) == 1:
        y_data = y_data.reshape(-1, 1)
    x_aux = jnp.array(x_aux)
    
    using_mask = mask is not None
    
    c_prime, d_prime, c = hyperparams['c_prime'], hyperparams['d_prime'], hyperparams['c']
    a_prime, b_prime, a = hyperparams['a_prime'], hyperparams['b_prime'], hyperparams['a']
    d = hyperparams['d']

    alpha_eta_old, omega_eta_old = q_params['alpha_eta'], q_params['omega_eta']
    alpha_beta_old, omega_beta_old = q_params['alpha_beta'], q_params['omega_beta']
    alpha_xi_old, omega_xi_old = q_params['alpha_xi'], q_params['omega_xi']
    alpha_theta_old, omega_theta_old = q_params['alpha_theta'], q_params['omega_theta']
    mu_gamma_old, Sigma_gamma_old = q_params['mu_gamma'], q_params['Sigma_gamma']
    mu_upsilon_old, Sigma_upsilon_old = q_params['mu_upsilon'], q_params['Sigma_upsilon']
    gig_a_ups, gig_b_ups_old = q_params['gig_a_ups'], q_params['gig_b_ups']

    E_theta_old = alpha_theta_old / jnp.maximum(omega_theta_old, 1e-10)
    E_beta_old = alpha_beta_old / jnp.maximum(omega_beta_old, 1e-10)
    if using_mask:
        E_beta_old = E_beta_old * mask

    print("Updating regression parameters (gamma, upsilon)...")
    
    var_theta_old = alpha_theta_old / jnp.maximum(omega_theta_old**2, 1e-10)
    E_theta_theta_T = jnp.einsum('ni,nj->nij', E_theta_old, E_theta_old) + \
                      jnp.einsum('ni,ij->nij', var_theta_old, jnp.eye(d))

    (mu_gamma_new, Sigma_gamma_new, 
     mu_upsilon_new, Sigma_upsilon_new, 
     gig_b_ups_new) = update_regression_variational(
        mu_gamma_old, Sigma_gamma_old, mu_upsilon_old, Sigma_upsilon_old,
        gig_a_ups, gig_b_ups_old,
        y_data, x_aux, hyperparams, E_theta_old, E_theta_theta_T
    )

    print("Updating latent counts (phi) and alpha parameters...")
    n, p = x_data.shape
    alpha_beta_new = jnp.full_like(alpha_beta_old, c)
    
    
    batch_size = max(1, min(100, n // 10))
    alpha_theta_from_data = jnp.zeros_like(alpha_theta_old)
    
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        E_theta_batch = E_theta_old[batch_start:batch_end]
        x_data_batch = x_data[batch_start:batch_end]
        
        alpha_beta_batch_update, alpha_theta_batch_update = process_phi_batch(
            E_theta_batch, E_beta_old, x_data_batch
        )
        alpha_beta_new += alpha_beta_batch_update
        alpha_theta_from_data = alpha_theta_from_data.at[batch_start:batch_end].set(alpha_theta_batch_update)

    alpha_theta_new = a + alpha_theta_from_data
    
    if using_mask:
        alpha_beta_new = alpha_beta_new * mask

    print("Updating rate (omega) and hyperprior (eta, xi) parameters...")
    
    E_eta_old = alpha_eta_old / jnp.maximum(omega_eta_old, 1e-10)
    sum_E_theta_old = jnp.sum(E_theta_old, axis=0)
    omega_beta_new = E_eta_old[:, None] + sum_E_theta_old[None, :]
    omega_beta_new = jnp.clip(omega_beta_new, 1e-6, 1e3)

    E_xi_old = alpha_xi_old / jnp.maximum(omega_xi_old, 1e-10)
    sum_E_beta_old = jnp.sum(E_beta_old, axis=0) # Sum over genes
    omega_theta_new = E_xi_old[:, None] + sum_E_beta_old[None, :]
    omega_theta_new = jnp.clip(omega_theta_new, 1e-6, 1e3)
    
    E_beta_new = alpha_beta_new / jnp.maximum(omega_beta_new, 1e-10)
    if using_mask:
        E_beta_new = E_beta_new * mask
    alpha_eta_new = c_prime + c * jnp.sum(mask, axis=1) if using_mask else c_prime + c * d
    omega_eta_new = jnp.sum(E_beta_new, axis=1) + (c_prime / d_prime)
    omega_eta_new = jnp.clip(omega_eta_new, 1e-6, 1e3)

    E_theta_new = alpha_theta_new / jnp.maximum(omega_theta_new, 1e-10)
    alpha_xi_new = a_prime + a * d
    omega_xi_new = jnp.sum(E_theta_new, axis=1) + (a_prime / b_prime)
    omega_xi_new = jnp.clip(omega_xi_new, 1e-6, 1e3)
    
    q_params_new = {
        "alpha_eta": alpha_eta_new, "omega_eta": omega_eta_new,
        "alpha_beta": alpha_beta_new, "omega_beta": omega_beta_new,
        "alpha_xi": alpha_xi_new, "omega_xi": omega_xi_new,
        "alpha_theta": alpha_theta_new, "omega_theta": omega_theta_new,
        
        "mu_gamma": mu_gamma_new, "Sigma_gamma": Sigma_gamma_new,
        "mu_upsilon": mu_upsilon_new, "Sigma_upsilon": Sigma_upsilon_new,
        
        "gig_p_ups": q_params['gig_p_ups'],
        "gig_a_ups": gig_a_ups,
        "gig_b_ups": gig_b_ups_new,
    }

    return q_params_new

def compute_elbo(x_data, y_data, x_aux, hyperparams, q_params, mask=None):
    
    hp = hyperparams
    d = hp['d']
    kappa, p_aux = q_params['mu_gamma'].shape

    alpha_eta, omega_eta = q_params['alpha_eta'], q_params['omega_eta']
    alpha_beta, omega_beta = q_params['alpha_beta'], q_params['omega_beta']
    alpha_xi, omega_xi = q_params['alpha_xi'], q_params['omega_xi']
    alpha_theta, omega_theta = q_params['alpha_theta'], q_params['omega_theta']

    mu_gamma, Sigma_gamma = q_params['mu_gamma'], q_params['Sigma_gamma']
    mu_upsilon, Sigma_upsilon = q_params['mu_upsilon'], q_params['Sigma_upsilon']

    gig_p, gig_a, gig_b = q_params['gig_p_ups'], q_params['gig_a_ups'], q_params['gig_b_ups']

    E_eta = alpha_eta / jnp.maximum(omega_eta, 1e-10)
    E_log_eta = jsp.special.digamma(alpha_eta) - jnp.log(jnp.maximum(omega_eta, 1e-10))

    E_beta = alpha_beta / jnp.maximum(omega_beta, 1e-10)
    E_log_beta = jsp.special.digamma(alpha_beta) - jnp.log(jnp.maximum(omega_beta, 1e-10))
    if mask is not None:
        E_beta = E_beta * mask
        E_log_beta = E_log_beta * mask

    E_xi = alpha_xi / jnp.maximum(omega_xi, 1e-10)
    E_log_xi = jsp.special.digamma(alpha_xi) - jnp.log(jnp.maximum(omega_xi, 1e-10))

    E_theta = alpha_theta / jnp.maximum(omega_theta, 1e-10)
    E_log_theta = jsp.special.digamma(alpha_theta) - jnp.log(jnp.maximum(omega_theta, 1e-10))

    E_gamma, E_upsilon = mu_gamma, mu_upsilon
    E_gamma_sq = jnp.array([jnp.diag(S) + m**2 for m, S in zip(mu_gamma, Sigma_gamma)])
    E_upsilon_sq = jnp.array([jnp.diag(S) + m**2 for m, S in zip(mu_upsilon, Sigma_upsilon)])
    
   
    E_inv_lambda_sq = jnp.sqrt(gig_a / jnp.maximum(gig_b, 1e-10))
    E_lambda_sq = jnp.sqrt(gig_b / jnp.maximum(gig_a, 1e-10))


    
    elbo_p_eta = jnp.sum(hp['c_prime'] * jnp.log(hp['d_prime']) - jsp.special.gammaln(hp['c_prime']) + \
                       (hp['c_prime'] - 1) * E_log_eta - hp['d_prime'] * E_eta)

    elbo_p_beta = jnp.sum(hp['c'] * E_log_eta[:, None] - jsp.special.gammaln(hp['c']) + \
                        (hp['c'] - 1) * E_log_beta - E_eta[:, None] * E_beta)

    elbo_p_xi = jnp.sum(hp['a_prime'] * jnp.log(hp['b_prime']) - jsp.special.gammaln(hp['a_prime']) + \
                      (hp['a_prime'] - 1) * E_log_xi - hp['b_prime'] * E_xi)
                      
    elbo_p_theta = jnp.sum(hp['a'] * E_log_xi[:, None] - jsp.special.gammaln(hp['a']) + \
                         (hp['a'] - 1) * E_log_theta - E_xi[:, None] * E_theta)

    log_phi_unnormalized = E_log_theta[:, None, :] + E_log_beta[None, :, :]
    phi = jax.nn.softmax(log_phi_unnormalized, axis=2)
    E_z = x_data[..., None] * phi
    
    elbo_p_x = jnp.sum(E_z * (E_log_theta[:, None, :] + E_log_beta[None, :, :])) - jnp.sum(E_theta @ E_beta.T)

    elbo_p_gamma = -0.5 * jnp.sum(E_gamma_sq) / hp['sigma']**2

    elbo_p_upsilon = -0.5 * jnp.sum(E_upsilon_sq * E_inv_lambda_sq)
    
    elbo_p_lambda_sq = -0.5 * jnp.sum(E_lambda_sq) / hp['tau']**2

   
    mu_gamma_old, Sigma_gamma_old = q_params['mu_gamma'], q_params['Sigma_gamma']
    mu_upsilon_old, Sigma_upsilon_old = q_params['mu_upsilon'], q_params['Sigma_upsilon']
    E_gamma_gamma_T = jnp.array([S + jnp.outer(m, m) for m, S in zip(mu_gamma_old, Sigma_gamma_old)])
    E_upsilon_upsilon_T = jnp.array([S + jnp.outer(m, m) for m, S in zip(mu_upsilon_old, Sigma_upsilon_old)])
    E_theta_theta_T = jnp.einsum('ni,nj->nij', E_theta, E_theta) + jnp.einsum('ni,ij->nij', (alpha_theta / omega_theta**2), jnp.eye(d))
    n, kappa = y_data.shape

    zeta_sq = jnp.zeros((n, kappa))
    for k in range(kappa):
        term1 = jnp.einsum('ii', E_theta_theta_T @ E_upsilon_upsilon_T[k])
        term2 = jnp.einsum('ij,ji->i', x_aux @ E_gamma_gamma_T[k], x_aux.T)
        term3 = 2 * (E_theta @ E_upsilon[k]) * (x_aux @ E_gamma[k])
        zeta_sq = zeta_sq.at[:, k].set(term1 + term2 + term3)
    
    zeta = jnp.sqrt(jnp.maximum(zeta_sq, 1e-10))
    lambda_zeta = jnp.tanh(zeta / 2) / jnp.maximum(4 * zeta, 1e-10)
    lambda_zeta = jnp.where(zeta < 1e-5, 1/8.0, lambda_zeta)
    
    E_A = (E_theta @ E_upsilon.T) + (x_aux @ E_gamma.T)
    E_A_sq = zeta_sq
    
    elbo_p_y = jnp.sum((y_data - 0.5) * E_A - lambda_zeta * E_A_sq) # The bound itself, excluding const(zeta)
    
    expected_log_joint = (elbo_p_eta + elbo_p_beta + elbo_p_xi + elbo_p_theta +
                          elbo_p_x + elbo_p_gamma + elbo_p_upsilon + 
                          elbo_p_lambda_sq + elbo_p_y)

    
    H_eta = jnp.sum(alpha_eta - jnp.log(omega_eta) + jsp.special.gammaln(alpha_eta) + (1 - alpha_eta) * jsp.special.digamma(alpha_eta))
    H_beta = jnp.sum(alpha_beta - jnp.log(omega_beta) + jsp.special.gammaln(alpha_beta) + (1 - alpha_beta) * jsp.special.digamma(alpha_beta))
    H_xi = jnp.sum(alpha_xi - jnp.log(omega_xi) + jsp.special.gammaln(alpha_xi) + (1 - alpha_xi) * jsp.special.digamma(alpha_xi))
    H_theta = jnp.sum(alpha_theta - jnp.log(omega_theta) + jsp.special.gammaln(alpha_theta) + (1 - alpha_theta) * jsp.special.digamma(alpha_theta))
    
    H_gamma = 0.5 * jnp.sum([jnp.linalg.slogdet(2 * jnp.pi * jnp.e * S)[1] for S in Sigma_gamma])
    H_upsilon = 0.5 * jnp.sum([jnp.linalg.slogdet(2 * jnp.pi * jnp.e * S)[1] for S in Sigma_upsilon])

    
    sqrt_ab_np = np.array(jnp.sqrt(gig_a * gig_b))
    log_bessel_ratio = np.log(kv(gig_p + 1, sqrt_ab_np)) - np.log(kv(gig_p, sqrt_ab_np))
    H_lambda_sq = jnp.sum(jnp.log(2*gig_b/gig_a)/2 + jsp.special.gammaln(gig_p) - (gig_p-1)*log_bessel_ratio + sqrt_ab_np * (kv(gig_p+1, sqrt_ab_np)/kv(gig_p, sqrt_ab_np)))

    entropy = H_eta + H_beta + H_xi + H_theta + H_gamma + H_upsilon + H_lambda_sq
    
    elbo = expected_log_joint + entropy
    
    print("\n--- ELBO Breakdown ---")
    print(f"E[log p(eta)]:       {elbo_p_eta:.4f}")
    print(f"E[log p(beta)]:      {elbo_p_beta:.4f}")
    print(f"E[log p(xi)]:         {elbo_p_xi:.4f}")
    print(f"E[log p(theta)]:      {elbo_p_theta:.4f}")
    print(f"E[log p(x,z)]:         {elbo_p_x:.4f}")
    print(f"E[log p(gamma)]:      {elbo_p_gamma:.4f}")
    print(f"E[log p(upsilon|lam)]: {elbo_p_upsilon:.4f}")
    print(f"E[log p(lambda^2)]:  {elbo_p_lambda_sq:.4f}")
    print(f"E[log p(y)]:          {elbo_p_y:.4f}")
    print(f"H[q(eta)]:            {H_eta:.4f}")
    print(f"H[q(beta)]:           {H_beta:.4f}")
    print(f"H[q(xi)]:             {H_xi:.4f}")
    print(f"H[q(theta)]:          {H_theta:.4f}")
    print(f"H[q(gamma)]:          {H_gamma:.4f}")
    print(f"H[q(upsilon)]:        {H_upsilon:.4f}")
    print(f"H[q(lambda^2)]:      {H_lambda_sq:.4f}")
    print("----------------------")
    print(f"Total ELBO:           {elbo:.4f}")

    return elbo

def run_variational_inference(x_data, y_data, x_aux, hyperparams,
                              q_params=None, max_iters=100, tol=1e-6,
                              verbose=True, mask=None,
                              patience=5, min_delta=1e-3, beta_init=None):
    
    # --- SETUP (largely the same) ---
    if hasattr(x_data, 'toarray'):
        x_data = x_data.toarray()
    
    x_data = jnp.array(x_data)
    y_data = jnp.array(y_data)
    if len(y_data.shape) == 1:
        y_data = y_data.reshape(-1, 1)
    x_aux = jnp.array(x_aux)

    if mask is not None:
        mask = jnp.array(mask)
    
    if q_params is None:
        n, p = x_data.shape
        kappa = y_data.shape[1]
        p_aux = x_aux.shape[1]
        d = hyperparams['d']
        q_params = initialize_q_params(n, p, kappa, p_aux, d, hyperparams, beta_init=beta_init)

    elbo_history = []
    old_elbo = -jnp.inf
    best_elbo = -jnp.inf
    best_params = None
    patience_counter = 0
    convergence_iter = None
    convergence_reason = "Maximum iterations reached"

    if verbose:
        print(f"Starting VI with patience={patience}, min_delta={min_delta:.6f}")

    # --- MAIN LOOP ---
    for iter_ix in range(max_iters):
        print(f"\n--- VI Iteration {iter_ix + 1}/{max_iters} ---")
        
        # 1. Update all variational parameters
        q_params = update_q_params(q_params, x_data, y_data, x_aux, hyperparams, mask=mask)
        
        # 2. Compute the ELBO to monitor convergence
        elbo_val = compute_elbo(x_data, y_data, x_aux, hyperparams, q_params, mask=mask)
        
        # Handle potential NaN from ELBO calculation
        if jnp.isnan(elbo_val):
            print("WARNING: ELBO returned NaN. Halting training and reverting to last best parameters.")
            convergence_reason = "ELBO became NaN"
            break
            
        elbo_history.append(float(elbo_val))

        if verbose:
            print(f"Iter {iter_ix+1}/{max_iters} ELBO = {elbo_val:.4f}")
            
        # 3. Check for convergence and early stopping
        if abs(elbo_val - old_elbo) < tol:
            convergence_reason = f"ELBO change ({abs(elbo_val - old_elbo):.6e}) < tolerance ({tol:.6e})"
            break

        if elbo_val > best_elbo + min_delta:
            best_elbo = elbo_val
            best_params = {k: v.copy() for k, v in q_params.items()}
            patience_counter = 0
            if verbose: print(f"  New best ELBO found. Patience reset.")
        else:
            patience_counter += 1
            if verbose: print(f"  No significant improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            convergence_reason = f"Early stopping: no improvement > {min_delta:.6e} for {patience} iterations"
            break
        
        old_elbo = elbo_val
        clear_memory()

    # --- POST-LOOP ---
    if best_params is not None:
        q_params = best_params
        print(f"Restored best model with ELBO = {best_elbo:.4f}")
    
    convergence_iter = len(elbo_history)
    q_params['convergence_info'] = {
        'total_iterations': convergence_iter,
        'max_iterations': max_iters,
        'best_elbo': float(best_elbo),
        'final_elbo': float(elbo_history[-1]) if elbo_history else -jnp.inf,
        'convergence_reason': convergence_reason
    }

    print("\n" + "="*50)
    print(f"VI Convergence Summary:")
    print(f"  Total iterations: {convergence_iter}/{max_iters}")
    print(f"  Final ELBO: {q_params['convergence_info']['final_elbo']:.4f}")
    print(f"  Convergence reason: {convergence_reason}")
    print("="*50 + "\n")

    return q_params, elbo_history

def fold_in_new_data(X_new, x_aux_new, q_params_trained, hyperparams,
                     max_iters=30, y_new=None, verbose=False, mask=None):
    
    # --- 1. UNPACK TRAINED PARAMETERS ---
    alpha_beta_tr = q_params_trained['alpha_beta']
    omega_beta_tr = q_params_trained['omega_beta']
    E_beta_tr = alpha_beta_tr / jnp.maximum(omega_beta_tr, 1e-10)
    if mask is not None:
        E_beta_tr = E_beta_tr * mask

    # For folding in, we only need the posterior means of the regression coefficients
    E_upsilon_tr = q_params_trained['mu_upsilon']  # Use mu_upsilon
    E_gamma_tr = q_params_trained['mu_gamma']    # Use mu_gamma

    # Unpack hyperparameters
    a_prime, b_prime, a = hyperparams['a_prime'], hyperparams['b_prime'], hyperparams['a']
    d_dim = hyperparams['d']

    # --- 2. SETUP FOR NEW DATA ---
    if hasattr(X_new, 'toarray'):
        X_new = X_new.toarray()
    X_new = jnp.array(X_new)
    x_aux_new = jnp.array(x_aux_new)
    
    n_test, p = X_new.shape
    
    # Initialize variational parameters for the new data points' theta and xi
    key = jax.random.PRNGKey(datetime.now().microsecond)
    k1, k2 = jax.random.split(key, 2)
    alpha_xi_test = jnp.ones(n_test) + 0.01 * jax.random.normal(k1, (n_test,))
    omega_xi_test = jnp.ones(n_test)
    alpha_theta_test = jnp.ones((n_test, d_dim)) + 0.01 * jax.random.normal(k2, (n_test, d_dim))
    omega_theta_test = jnp.ones((n_test, d_dim))
    
    # --- 3. FOLD-IN ITERATIONS ---
    sumBeta_tr = jnp.sum(E_beta_tr, axis=0) # Pre-compute

    for iter_ix in range(max_iters):
        # Compute current expectations for theta and xi
        E_theta_test = alpha_theta_test / jnp.maximum(omega_theta_test, 1e-10)
        E_xi_test = alpha_xi_test / jnp.maximum(omega_xi_test, 1e-10)
            
        # Update alpha_theta from data likelihood (phi update)
        alpha_theta_test_new = a + jnp.sum(X_new[:, :, None] * jax.nn.softmax(E_theta_test[:, None, :] + E_beta_tr[None, :, :], axis=2), axis=1)

        # Update omega_theta
        omega_theta_test = E_xi_test[:, None] + sumBeta_tr[None, :]
        
        # Update alpha_xi and omega_xi
        alpha_xi_test = a_prime + a * d_dim
        omega_xi_test = jnp.sum(alpha_theta_test_new / jnp.maximum(omega_theta_test, 1e-10), axis=1) + (a_prime / b_prime)
        
        # Update alpha_theta for the next iteration
        alpha_theta_test = alpha_theta_test_new

    # Final expectation of theta for the new data
    E_theta_test = alpha_theta_test / jnp.maximum(omega_theta_test, 1e-10)
    return E_theta_test

def evaluate_model(X_data, x_aux, y_data, q_params, hyperparams,
                   fold_in=False, return_probs=True, mask=None):
    
    if fold_in:
        print(f"Folding in new data for evaluation...")
        # Note: The `fold_in_new_data` function should not require y_new for prediction
        E_theta_data = fold_in_new_data(X_data, x_aux, q_params, hyperparams,
                                        max_iters=30, y_new=None, verbose=False, mask=mask)
    else:
        alpha_theta = q_params['alpha_theta']
        omega_theta = q_params['omega_theta']
        E_theta_data = alpha_theta / jnp.maximum(omega_theta, 1e-10)

    # Use the posterior means for prediction
    E_upsilon = q_params['mu_upsilon']
    E_gamma   = q_params['mu_gamma']
    
    x_aux = jnp.array(x_aux)
    
    # --- COMPUTE LOGITS AND PROBABILITIES ---
    logits = (x_aux @ E_gamma.T) + jnp.einsum('nd,kd->nk', E_theta_data, E_upsilon)
    probs = jax.nn.sigmoid(logits)

    # --- PREPARE DATA FOR METRIC CALCULATION ---
    y_data = np.array(y_data) # Convert to numpy for sklearn compatibility
    if len(y_data.shape) == 1:
        y_data = y_data.reshape(-1, 1)
    kappa = y_data.shape[1]

    # --- SET DECISION THRESHOLD AND MAKE PREDICTIONS ---
    # Using a fixed 0.5 threshold is a robust default.
    # Dynamic thresholds can be added later if needed.
    threshold = 0.5
    preds = (np.array(probs) >= threshold).astype(int)
    
    # --- CALCULATE METRICS ---
    results = {}
    
    if kappa == 1:
        # Single-label classification
        y_true_flat = y_data.flatten()
        y_pred_flat = preds.flatten()
        probs_flat = np.array(probs).flatten()

        acc   = accuracy_score(y_true_flat, y_pred_flat)
        prec  = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        rec   = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        f1v   = f1_score(y_true_flat, y_pred_flat, zero_division=0)
        
        # ROC AUC requires at least one sample from each class
        try:
            aucv = roc_auc_score(y_true_flat, probs_flat)
        except ValueError:
            aucv = 0.5 # Default value if only one class is present
            
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        
        print(f"Evaluation Metrics (kappa=1):")
        print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1v:.4f}, AUC: {aucv:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        results = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1v,
            "roc_auc": aucv, "confusion_matrix": cm.tolist(), "threshold": threshold
        }
    else:
        # Multi-label classification (one-vs-rest metrics)
        metrics_per_class = []
        for k_idx in range(kappa):
            y_true_k = y_data[:, k_idx]
            y_pred_k = preds[:, k_idx]
            probs_k = np.array(probs)[:, k_idx]
            
            acc_k   = accuracy_score(y_true_k, y_pred_k)
            prec_k  = precision_score(y_true_k, y_pred_k, zero_division=0)
            rec_k   = recall_score(y_true_k, y_pred_k, zero_division=0)
            f1_k    = f1_score(y_true_k, y_pred_k, zero_division=0)
            try:
                auc_k = roc_auc_score(y_true_k, probs_k)
            except ValueError:
                auc_k = 0.5
            
            metrics_per_class.append({
                "class": k_idx, "accuracy": acc_k, "precision": prec_k,
                "recall": rec_k, "f1": f1_k, "roc_auc": auc_k
            })
            
        # Calculate macro-averaged metrics
        acc_macro  = np.mean([m['accuracy'] for m in metrics_per_class])
        prec_macro = np.mean([m['precision'] for m in metrics_per_class])
        rec_macro  = np.mean([m['recall'] for m in metrics_per_class])
        f1_macro   = np.mean([m['f1'] for m in metrics_per_class])
        auc_macro  = np.mean([m['roc_auc'] for m in metrics_per_class])
        
        print(f"Macro-Averaged Evaluation Metrics (kappa={kappa}):")
        print(f"  Accuracy: {acc_macro:.4f}, Precision: {prec_macro:.4f}, Recall: {rec_macro:.4f}, F1: {f1_macro:.4f}, AUC: {auc_macro:.4f}")
        
        results = {
            "accuracy": acc_macro, "precision": prec_macro, "recall": rec_macro,
            "f1": f1_macro, "roc_auc": auc_macro,
            "per_class_metrics": metrics_per_class, "threshold": threshold
        }
    
    if return_probs:
        results["probabilities"] = probs.tolist()
        
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

def plot_elbo_convergence(elbo_history, output_file=None, title=None, show_best=True):
    import matplotlib.pyplot as plt
    
    if not elbo_history:
        print("Cannot plot ELBO convergence: history is empty.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    elbo_array = np.array(elbo_history)
    iterations = np.arange(1, len(elbo_array) + 1)
    
    ax.plot(iterations, elbo_array, 'b-', linewidth=2, label='ELBO')
    ax.fill_between(iterations, elbo_array.min(), elbo_array, alpha=0.2, color='blue')
    
    if show_best and len(elbo_array) > 0:
        best_idx = np.argmax(elbo_array)
        best_elbo = elbo_array[best_idx]
        best_iter = best_idx + 1
        
        ax.axhline(y=best_elbo, color='r', linestyle='--', alpha=0.7, label=f'Best ELBO: {best_elbo:.2f}')
        ax.plot(best_iter, best_elbo, 'ro', markersize=8, zorder=5) # zorder to bring to front
        ax.annotate(f'Best: {best_elbo:.2f}\nIter: {best_iter}', 
                   xy=(best_iter, best_elbo),
                   xytext=(best_iter + 5, best_elbo), # Adjust text position
                   arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5),
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.7),
                   fontsize=10)
    
    ax.set_title(title if title else 'ELBO Convergence History', fontsize=14)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('ELBO Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"ELBO convergence plot saved to: {output_file}")
    
    # plt.show() # Uncomment if you want to see the plot interactively
    plt.close(fig) # Close the figure to free up memory
    
    return fig

def run_model_and_evaluate(x_data, x_aux, y_data, var_names, hyperparams,
                           seed=None, test_size=0.15, val_size=0.15, max_iters=100,
                           return_probs=True, sample_ids=None,
                           mask=None, scores=None, plot_elbo=True, plot_prefix=None,
                           return_params=False, beta_init=None):  
    
    # --- DATA SPLITTING (no changes here) ---
    print(f"Input shapes - x_data: {x_data.shape}, x_aux: {x_aux.shape}, y_data: {y_data.shape}")
    
    if sample_ids is None:
        sample_ids = np.arange(x_data.shape[0])
    if scores is None:
        scores = np.zeros(x_data.shape[0])  

    temp_size = val_size + test_size
    if temp_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")
    
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

    # --- MODEL TRAINING (updated call) ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        q_params, elbo_hist = run_variational_inference(
            X_train, y_train, XA_train, hyperparams,
            q_params=None, max_iters=max_iters, verbose=True,
            mask=mask, patience=5, min_delta=1e-3,
            beta_init=beta_init
        )
        
        convergence_info = q_params.get('convergence_info', {})
        
    except Exception as e:
        print(f"FATAL ERROR during training: {e}")
        # Create a dummy results structure to avoid crashing
        return { "error": str(e), "status": "failed" }

    # --- ELBO PLOTTING (updated to use convergence_info) ---
    if plot_elbo and len(elbo_hist) > 1:
        plot_title = f"ELBO Convergence (d={hyperparams['d']}, Converged Iter: {convergence_info.get('total_iterations', 'N/A')})"
        plot_file = f"{plot_prefix}_elbo_convergence.png" if plot_prefix else f"elbo_convergence_{timestamp}.png"
        plot_elbo_convergence(elbo_hist, output_file=plot_file, title=plot_title)
        
    # --- MODEL EVALUATION (updated calls) ---
    print("\n--- Evaluating on Train Set ---")
    train_metrics = evaluate_model(X_train, XA_train, y_train, q_params,
                                  hyperparams, fold_in=False,
                                  return_probs=True, mask=mask)  

    print("\n--- Evaluating on Validation Set ---")
    val_metrics = evaluate_model(X_val, XA_val, y_val, q_params,
                                 hyperparams, fold_in=True, 
                                 return_probs=True, mask=mask)
    
    print("\n--- Evaluating on Test Set ---")
    test_metrics = evaluate_model(X_test, XA_test, y_test, q_params,
                                 hyperparams, fold_in=True,
                                 return_probs=True, mask=mask)  

    # --- FINAL RESULTS COMPILATION ---
    # Extract final posterior mean of beta for gene program analysis
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

    results = {
        "data_info": {
            "n_train": X_train.shape[0], "n_val": X_val.shape[0], "n_test": X_test.shape[0],
            "p": X_train.shape[1], "kappa": y_train.shape[1] if len(y_train.shape)>1 else 1,
            "p_aux": XA_train.shape[1], "d": hyperparams['d'],
            "hyperparameters": hyperparams,
        },
        "elbo_history": elbo_hist,
        "convergence_info": convergence_info,
        "train_metrics": {k: v for k, v in train_metrics.items() if k != 'probabilities'},
        "val_metrics":   {k: v for k, v in val_metrics.items() if k != 'probabilities'},
        "test_metrics":  {k: v for k, v in test_metrics.items() if k != 'probabilities'},
        "top_genes": top_genes,
        "E_upsilon": q_params['mu_upsilon'].tolist(),
        "E_gamma":   q_params['mu_gamma'].tolist(),
        "train_results_df": train_df,
        "val_results_df": val_df,
        "test_results_df": test_df,
        "convergence_info": q_params.get('convergence_info', {})
    }
    
    if return_probs:
        results["train_probabilities"] = train_metrics['probabilities']
        results["val_probabilities"] = val_metrics['probabilities']
        results["test_probabilities"] = test_metrics['probabilities']
        
    if return_params:
        results["alpha_beta"] = q_params['alpha_beta'].tolist()
        results["omega_beta"] = q_params['omega_beta'].tolist()
        results["E_beta"] = E_beta_final.tolist()

    return results