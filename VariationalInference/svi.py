# Stochastic Variational Inference implementation
# for the SupervisedPoissonFactorization model using proper stochastic gradient ascent.

import numpy as np
import jax.numpy as jnp
import jax.random as random
from jax import grad, jit
import jax.scipy as jsp
import matplotlib.pyplot as plt
import os

from vi_model_complete import (
    SupervisedPoissonFactorization,
    logistic,
    lambda_jj,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score,
    roc_auc_score,
)


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray):
    """Compute basic classification metrics."""
    preds = (probs >= 0.5).astype(int)

    if y_true.ndim == 1 or y_true.shape[1] == 1:
        y_flat = y_true.reshape(-1)
        p_flat = probs.reshape(-1)
        pred_flat = preds.reshape(-1)

        metrics = {
            "accuracy": accuracy_score(y_flat, pred_flat),
            "precision": precision_score(y_flat, pred_flat, zero_division=0),
            "recall": recall_score(y_flat, pred_flat, zero_division=0),
            "f1": f1_score(y_flat, pred_flat, zero_division=0),
        }
        try:
            metrics["roc_auc"] = roc_auc_score(y_flat, p_flat)
        except ValueError:
            metrics["roc_auc"] = 0.5
    else:
        per_class = []
        for k in range(y_true.shape[1]):
            m = _compute_metrics(y_true[:, k], probs[:, k])
            per_class.append(m)
        metrics = {k: float(np.mean([m[k] for m in per_class])) for k in ["accuracy","precision","recall","f1","roc_auc"]}
        metrics["per_class_metrics"] = per_class  # type: ignore

    metrics["probabilities"] = probs.tolist()
    return metrics


def compute_natural_gradients(model, params, expected, X_b, Y_b, X_aux_b, batch_idx, scale):
    """
    Compute natural gradient updates for global parameters using proper SVI.
    This replaces the coordinate ascent updates with stochastic gradient ascent.
    """
    n_batch = X_b.shape[0]
    
    # =============== THETA UPDATE (LOCAL PARAMETERS) ===============
    # For local parameters (theta), we use standard coordinate ascent on the minibatch
    # without scaling because these are sample-specific parameters
    
    z_b = model.update_z_latent(X_b, expected["E_theta"][batch_idx], expected["E_beta"])
    
    # Standard theta update for the minibatch samples only
    a_theta_new = model.alpha_theta + jnp.sum(z_b, axis=1)
    
    # Poisson rate terms
    b_theta_poisson = jnp.expand_dims(expected["E_xi"][batch_idx], 1) + \
                     jnp.sum(expected["E_beta"], axis=0)
    
    # Logistic rate terms (regression part)
    b_theta_logistic = jnp.zeros_like(b_theta_poisson)
    theta_current = expected["E_theta"][batch_idx]
    lam = lambda_jj(params["zeta"][batch_idx])
    
    for k in range(model.kappa):
        term1 = -(Y_b[:, k:k+1] - 0.5) * expected["E_v"][k, :]
        aux_term = X_aux_b @ expected["E_gamma"][k]  # (n_batch,)
        term2 = -2.0 * jnp.expand_dims(lam[:, k], 1) * expected["E_v"][k, :] * \
                jnp.expand_dims(aux_term, 1)
        term3 = -2.0 * jnp.expand_dims(lam[:, k], 1) * \
                jnp.expand_dims(expected["E_v"][k] ** 2, 0) * theta_current
        b_theta_logistic += term1 + term2 + term3
    
    b_theta_new = b_theta_poisson + b_theta_logistic
    # === Diagnostics for b_theta_new ===
    b_theta_new_np = np.array(b_theta_new)
    print(f"[SVI] b_theta_new before clip: min={b_theta_new_np.min():.4g}, max={b_theta_new_np.max():.4g}, mean={b_theta_new_np.mean():.4g}, std={b_theta_new_np.std():.4g}")
    b_theta_new = jnp.maximum(b_theta_new, 1e-8)
    b_theta_new_np_clip = np.array(b_theta_new)
    print(f"[SVI] b_theta_new after clip: min={b_theta_new_np_clip.min():.4g}, max={b_theta_new_np_clip.max():.4g}, mean={b_theta_new_np_clip.mean():.4g}, std={b_theta_new_np_clip.std():.4g}")
    theta_update = {"a_theta": a_theta_new, "b_theta": b_theta_new}
    
    # =============== GLOBAL PARAMETER UPDATES (NATURAL GRADIENTS) ===============
    
    # Update z with new theta values for gradient computation
    E_theta_new = a_theta_new / b_theta_new
    z_b_updated = model.update_z_latent(X_b, E_theta_new, expected["E_beta"])

    
    # Compute lam_updated once for v and gamma updates
    theta_v_term = jnp.sum(jnp.expand_dims(E_theta_new, 1) * jnp.expand_dims(expected["E_v"], 0), axis=2)  # (n_batch, kappa)
    aux_gamma_term = X_aux_b @ expected["E_gamma"].T  # (n_batch, kappa)
    lam_updated = lambda_jj(jnp.abs(theta_v_term + aux_gamma_term) + 0.01)
    
    # ====== ETA UPDATE (Global) ======
    # Natural gradient: sufficient statistics from minibatch, scaled appropriately
    a_eta_grad = model.alpha_eta + scale * model.K * model.alpha_beta
    b_eta_grad = model.lambda_eta + scale * jnp.sum(expected["E_beta"], axis=1)
    eta_update = {"a_eta": jnp.full(model.p, a_eta_grad), "b_eta": b_eta_grad}
    
    # ====== XI UPDATE (Global aggregation of local statistics) ======  
    # This is a hybrid - xi depends on theta which are local, but xi parameters are global
    a_xi_grad = model.alpha_xi + scale * model.K * model.alpha_theta
    b_xi_grad = model.lambda_xi + scale * jnp.sum(E_theta_new, axis=1)
    xi_update = {"a_xi": jnp.full(n_batch, a_xi_grad), "b_xi": b_xi_grad}
    
    # ====== BETA UPDATE (Global) ======
    # Natural gradient for beta uses scaled sufficient statistics
    a_beta_grad = model.alpha_beta + scale * jnp.sum(z_b_updated, axis=0)
    b_beta_grad = jnp.expand_dims(b_eta_grad, 1) + \
                  scale * jnp.sum(E_theta_new, axis=0, keepdims=True)
    beta_update = {"a_beta": a_beta_grad, "b_beta": b_beta_grad}
    
    # ====== V UPDATE (Global) ======
    mu_v_new = jnp.zeros((model.kappa, model.K))
    tau2_v_new = jnp.zeros((model.kappa, model.K, model.K))
    
    I = jnp.eye(model.K) / model.sigma2_v
    
    for k in range(model.kappa):
        # Natural gradient precision matrix
        S = I + 2.0 * scale * (E_theta_new.T * lam_updated[:, k]) @ E_theta_new
        
        # Regularization for numerical stability
        S = S + jnp.eye(model.K) * 1e-6
        
        try:
            L = jnp.linalg.cholesky(S)
            Sigma_k = jsp.linalg.cho_solve((L, True), jnp.eye(model.K))
        except Exception as e:
            print(f"[SVI] Cholesky failed for v (k={k}): {e}")
            Sigma_k = jnp.eye(model.K) * 1e-2
        tau2_v_new = tau2_v_new.at[k].set(Sigma_k)
        
        # Natural gradient mean update
        rhs = ((Y_b[:, k] - 0.5) - 2.0 * lam_updated[:, k] * (X_aux_b @ expected["E_gamma"][k]))
        rhs = scale * rhs @ E_theta_new
        mu_k = Sigma_k @ rhs
        mu_v_new = mu_v_new.at[k].set(mu_k)
    
    v_update = {"mu_v": mu_v_new, "tau2_v": tau2_v_new}
    
    # ====== GAMMA UPDATE (Global) ======
    mu_gamma_new = jnp.zeros((model.kappa, X_aux_b.shape[1]))
    tau2_gamma_new = jnp.zeros((model.kappa, X_aux_b.shape[1], X_aux_b.shape[1]))
    
    I_gamma = jnp.eye(X_aux_b.shape[1]) / model.sigma2_gamma
    
    for k in range(model.kappa):
        # Natural gradient precision matrix
        S = I_gamma + 2.0 * scale * (X_aux_b.T * lam_updated[:, k]) @ X_aux_b
        S = S + jnp.eye(X_aux_b.shape[1]) * 1e-6
        
        try:
            L = jnp.linalg.cholesky(S)
            Sigma_k = jsp.linalg.cho_solve((L, True), jnp.eye(X_aux_b.shape[1]))
        except Exception as e:
            print(f"[SVI] Cholesky failed for gamma (k={k}): {e}")
            Sigma_k = jnp.eye(X_aux_b.shape[1]) * 1e-2
        tau2_gamma_new = tau2_gamma_new.at[k].set(Sigma_k)
        
        # Natural gradient mean update
        rhs = ((Y_b[:, k] - 0.5) - 2.0 * lam_updated[:, k] * (E_theta_new @ expected["E_v"][k]))
        rhs = scale * rhs @ X_aux_b
        mu_k = Sigma_k @ rhs
        mu_gamma_new = mu_gamma_new.at[k].set(mu_k)
    
    gamma_update = {"mu_gamma": mu_gamma_new, "tau2_gamma": tau2_gamma_new}
    
    # ====== ZETA UPDATE ======
    # Compute updated zeta using the new v and gamma parameters
    theta_v_new_term = jnp.sum(jnp.expand_dims(E_theta_new, 1) * jnp.expand_dims(mu_v_new, 0), axis=2)  # (n_batch, kappa)
    aux_gamma_new_term = X_aux_b @ mu_gamma_new.T  # (n_batch, kappa)
    zeta_new = jnp.abs(theta_v_new_term + aux_gamma_new_term) + 0.01
    zeta_update = {"zeta": zeta_new}
    
    return {
        "theta": theta_update,
        "eta": eta_update,
        "xi": xi_update, 
        "beta": beta_update,
        "v": v_update,
        "gamma": gamma_update,
        "zeta": zeta_update,
        "batch_idx": batch_idx
    }


def infer_theta_unsupervised(model, X_new, global_params, n_iter=20):
    """
    Infer theta for new samples using ONLY the gene expression data and 
    the learned Poisson factorization parameters (beta, eta).
    This is independent of the regression part (v, gamma) and labels.
    
    Note: For unseen data, we use CAVI-style coordinate ascent updates since:
    1. We're doing local inference for specific samples (not stochastic)
    2. We don't have labels, so no regression component to consider
    3. We want exact posterior inference for these samples given global params
    
    This differs from training where we use SGD updates for efficiency with minibatches.
    """
    n_new = X_new.shape[0]
    K = model.K
    
    # Initialize theta for new samples based on gene expression
    sample_totals = jnp.sum(X_new, axis=1)
    theta_init = jnp.outer(sample_totals / jnp.mean(sample_totals), jnp.ones(K)) * 0.1
    theta_init = jnp.maximum(theta_init, 0.01)
    
    # Convert to Gamma parameters
    theta_var = theta_init * 0.1  # Lower variance for more stable inference
    a_theta = (theta_init**2) / theta_var
    b_theta = theta_init / theta_var
    
    # Use global parameters 
    E_beta = global_params['a_beta'] / global_params['b_beta']
    E_eta = global_params['a_eta'] / global_params['b_eta']
    
    # xi for new samples - these are local but based on theta
    a_xi = jnp.full(n_new, model.alpha_xi + K * model.alpha_theta)
    b_xi = model.lambda_xi + jnp.sum(a_theta / b_theta, axis=1)
    
    for it in range(n_iter):
        # Current expectations
        E_theta = a_theta / b_theta
        E_xi = a_xi / b_xi
        
        # Update latent variables z using only Poisson factorization
        rates = jnp.expand_dims(E_theta, 1) * jnp.expand_dims(E_beta, 0)  # (n, p, K)
        total_rates = jnp.sum(rates, axis=2, keepdims=True)
        probs = rates / (total_rates + 1e-8)
        z = jnp.expand_dims(X_new, 2) * probs
        
        # Update xi
        a_xi_new = model.alpha_xi + K * model.alpha_theta  
        b_xi_new = model.lambda_xi + jnp.sum(E_theta, axis=1)
        b_xi_new = jnp.maximum(b_xi_new, 1e-8)
        
        a_xi = jnp.full(n_new, a_xi_new)
        b_xi = b_xi_new
        
        # Update theta using ONLY Poisson likelihood (no regression terms)
        a_theta_new = model.alpha_theta + jnp.sum(z, axis=1)
        b_theta_new = jnp.expand_dims(E_xi, 1) + jnp.sum(E_beta, axis=0)
        b_theta_new = jnp.maximum(b_theta_new, 1e-8)
        
        a_theta = a_theta_new
        b_theta = b_theta_new
    
    return a_theta / b_theta  # E[theta] for new samples


def fit_svi(model, X, Y, X_aux, n_iter=1000, batch_size=36, learning_rate=0.002, verbose=False, 
            track_elbo=False, elbo_freq=5, early_stopping=True, patience=50, min_delta=1e-4, beta_init=None):
    """
    Proper Stochastic Variational Inference using natural gradients.
    Separates local parameters (theta) from global parameters (beta, eta, v, gamma).
    
    Parameters:
    -----------
    track_elbo : bool
        Whether to track ELBO values during training
    elbo_freq : int
        Frequency (in iterations) to compute and store ELBO
    early_stopping : bool
        Whether to use early stopping based on ELBO convergence
    patience : int
        Number of iterations to wait after last improvement before stopping
    min_delta : float
        Minimum change in ELBO to qualify as an improvement
    beta_init : array, optional
        Initial values for beta parameters (for pathway initialization)
    """
    import time
    start_time = time.time()
    if verbose:
        print(f"Starting SVI training: {n_iter} iterations, batch_size={batch_size}, lr={learning_rate:.4f}")
    
    # === Data diagnostics ===
    print("[SVI] Data diagnostics:")
    print(f"  X shape: {X.shape}, min: {np.min(X):.4g}, max: {np.max(X):.4g}, mean: {np.mean(X):.4g}, std: {np.std(X):.4g}, zeros: {np.sum(X==0)}")
    print(f"  Y shape: {Y.shape}, unique: {np.unique(Y)}, min: {np.min(Y):.4g}, max: {np.max(Y):.4g}")
    print(f"  X_aux shape: {X_aux.shape}, min: {np.min(X_aux):.4g}, max: {np.max(X_aux):.4g}, mean: {np.mean(X_aux):.4g}, std: {np.std(X_aux):.4g}, zeros: {np.sum(X_aux==0)}")
    
    print("[SVI] Hyperparameters:")
    for k, v in model.__dict__.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
    
    n = X.shape[0]
    rng = np.random.default_rng(0)

    # Initialize parameters
    params = model.initialize_parameters(X, Y, X_aux, beta_init=beta_init)
    
    # ELBO tracking
    elbo_values = []
    elbo_iterations = []
    
    # Early stopping variables
    best_elbo = -np.inf
    no_improvement_count = 0
    stopped_early = False
    
    # Learning rate schedule
    base_lr = learning_rate
    
    elbo_values = []  # To store ELBO values for plotting
    
    for it in range(n_iter):
        # Sample minibatch
        batch_idx = rng.choice(n, size=min(batch_size, n), replace=False)
        X_b = X[batch_idx]
        Y_b = Y[batch_idx]
        X_aux_b = X_aux[batch_idx]
        
        # Compute current expectations
        expected = model.expected_values(params)
        # === Diagnostics for E_theta ===
        E_theta_np = np.array(expected["E_theta"])
        print(f"[SVI] iter {it+1} E_theta: min={E_theta_np.min():.4g}, max={E_theta_np.max():.4g}, mean={E_theta_np.mean():.4g}, std={E_theta_np.std():.4g}")
        
        # Scale factor for global parameter updates
        scale = n / X_b.shape[0]
        
        # Compute natural gradient updates
        try:
            updates = compute_natural_gradients(
                model, params, expected, X_b, Y_b, X_aux_b, batch_idx, scale
            )
            
            # Apply updates with learning rate decay
            lr = base_lr * (0.9 ** (it // 10))  # Decay every 10 iterations
            
            # Update global parameters with natural gradients
            # For natural parameters of exponential families, we can directly update
            params["a_eta"] = (1 - lr) * params["a_eta"] + lr * updates["eta"]["a_eta"]
            params["b_eta"] = (1 - lr) * params["b_eta"] + lr * updates["eta"]["b_eta"]
            
            params["a_beta"] = (1 - lr) * params["a_beta"] + lr * updates["beta"]["a_beta"]
            params["b_beta"] = (1 - lr) * params["b_beta"] + lr * updates["beta"]["b_beta"]
            
            params["mu_v"] = (1 - lr) * params["mu_v"] + lr * updates["v"]["mu_v"]
            # For covariance matrices, we need to be more careful to maintain positive definiteness
            # Using a simpler update that replaces rather than interpolates
            params["tau2_v"] = updates["v"]["tau2_v"]
            
            params["mu_gamma"] = (1 - lr) * params["mu_gamma"] + lr * updates["gamma"]["mu_gamma"]
            # Same for gamma covariance matrices
            params["tau2_gamma"] = updates["gamma"]["tau2_gamma"]
            
            # Update local parameters (theta, xi) directly for minibatch
            params["a_theta"] = params["a_theta"].at[batch_idx].set(updates["theta"]["a_theta"])
            params["b_theta"] = params["b_theta"].at[batch_idx].set(updates["theta"]["b_theta"])
            
            params["a_xi"] = params["a_xi"].at[batch_idx].set(updates["xi"]["a_xi"])
            params["b_xi"] = params["b_xi"].at[batch_idx].set(updates["xi"]["b_xi"])
            
            # Update zeta
            params["zeta"] = params["zeta"].at[batch_idx].set(updates["zeta"]["zeta"])
            
            # Stability checks
            max_theta = jnp.max(params["a_theta"] / params["b_theta"])
            if max_theta > 1e4:
                print(f"SVI iter {it+1}: Large theta detected ({max_theta:.2e}), applying regularization")
                params["b_theta"] = jnp.maximum(params["b_theta"], params["a_theta"] / 1e3)
            
            # Compute ELBO for current minibatch using SVI-specific function
            if track_elbo and (it % elbo_freq == 0):
                try:
                    elbo = compute_svi_elbo(
                        model, X_b, Y_b, X_aux_b, params, expected, 
                        batch_idx, scale, return_components=False, debug_print=False
                    )
                    elbo_values.append(float(elbo))
                    elbo_iterations.append(it + 1)
                    
                    # Early stopping check
                    if early_stopping:
                        if elbo > best_elbo + min_delta:
                            best_elbo = elbo
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                        
                        # Check if we should stop early
                        if no_improvement_count >= patience:
                            if verbose:
                                print(f"[SVI] Early stopping at iteration {it+1}: no improvement for {patience} checks")
                                print(f"[SVI] Best ELBO: {best_elbo:.4f}, Current ELBO: {elbo:.4f}")
                            stopped_early = True
                            break
                    
                except Exception as e:
                    if verbose:
                        print(f"SVI iter {it+1}: ELBO tracking failed: {e}")
                
                if verbose and (it % 5 == 0):
                    try:
                        elbo_components = compute_svi_elbo(
                            model, X_b, Y_b, X_aux_b, params, expected, 
                            batch_idx, scale, return_components=True, debug_print=False
                        )
                        improvement_status = "↑" if no_improvement_count == 0 else f"↓{no_improvement_count}"
                        print(f"SVI iter {it+1}, minibatch ELBO: {elbo_components['elbo']:.3f}, lr: {lr:.4f}, status: {improvement_status}")
                        for name, val in elbo_components.items():
                            if name != 'elbo':
                                print(f"  {name}: {float(val):.3g}")
                    except Exception as e:
                        print(f"SVI iter {it+1}: ELBO computation failed: {e}")
            elif early_stopping and not track_elbo:
                # If early stopping is requested but ELBO tracking is off, warn user
                if it == 0 and verbose:
                    print("[SVI] Warning: Early stopping requires ELBO tracking. Setting track_elbo=True.")
                track_elbo = True
        except Exception as e:
            print(f"SVI iter {it+1}: Update failed: {e}")
            # Continue with next iteration
            continue

    if verbose:
        end_time = time.time()
        if stopped_early:
            print(f"SVI training stopped early after {it+1} iterations in {end_time - start_time:.1f} seconds")
            print(f"Best ELBO: {best_elbo:.4f}")
        else:
            print(f"SVI training completed in {end_time - start_time:.1f} seconds")

    # Prepare results to return
    results = {
        'params': params,
        'expected': model.expected_values(params),
        'stopped_early': stopped_early,
        'final_iteration': it + 1,
        'best_elbo': best_elbo if early_stopping else None
    }
    
    if track_elbo:
        results['elbo_values'] = elbo_values
        results['elbo_iterations'] = elbo_iterations

    return results


def compute_svi_elbo(model, X_b, Y_b, X_aux_b, params, expected, batch_idx, scale, 
                    return_components=False, debug_print=False):
    """
    Compute SVI-specific ELBO using the stochastic gradient approach.
    This differs from CAVI ELBO as it uses scaled sufficient statistics for global parameters.
    
    Parameters:
    -----------
    model : SupervisedPoissonFactorization
        The model instance
    X_b : jnp.ndarray
        Minibatch gene expression data (n_batch, p)
    Y_b : jnp.ndarray
        Minibatch outcomes (n_batch, kappa)
    X_aux_b : jnp.ndarray
        Minibatch auxiliary data (n_batch, d)
    params : dict
        Current variational parameters
    expected : dict
        Current expected values 
    batch_idx : jnp.ndarray
        Indices of current batch
    scale : float
        Scaling factor n/n_batch for global parameters
    return_components : bool
        Whether to return individual ELBO components
    debug_print : bool
        Whether to print debug information
    """
    
    digamma = jsp.special.digamma
    gammaln = jsp.special.gammaln
    
    n_batch = X_b.shape[0]
    
    # Get batch-specific parameters
    a_theta_b = params['a_theta'][batch_idx]
    b_theta_b = params['b_theta'][batch_idx]
    a_xi_b = params['a_xi'][batch_idx]
    b_xi_b = params['b_xi'][batch_idx]
    zeta_b = params['zeta'][batch_idx]
    
    # Expected values for batch
    E_theta_b = a_theta_b / b_theta_b
    E_theta_sq_b = (a_theta_b * (a_theta_b + 1)) / b_theta_b**2
    E_xi_b = a_xi_b / b_xi_b
    
    # Global expected values
    E_beta = expected['E_beta']
    E_eta = expected['E_eta']
    
    # ========== POISSON LIKELIHOOD (scaled for SVI) ==========
    # Compute expected rate for minibatch: λ_ij = Σ_ℓ E[θ_iℓ] E[β_jℓ]
    expected_rate_b = jnp.sum(E_theta_b[:, None, :] * E_beta[None, :, :], axis=2)  # (n_batch, p)
    
    # Poisson log-likelihood for minibatch (scaled by n/n_batch)
    pois_term1 = scale * jnp.sum(X_b * jnp.log(expected_rate_b + 1e-8))
    pois_term2 = -scale * jnp.sum(expected_rate_b)
    pois_term3 = -scale * jnp.sum(gammaln(X_b + 1))
    pois_ll = pois_term1 + pois_term2 + pois_term3
    
    # ========== LOGISTIC REGRESSION LIKELIHOOD (scaled for SVI) ==========
    psi_b = E_theta_b @ params['mu_v'].T + X_aux_b @ params['mu_gamma'].T  # (n_batch, kappa)
    lam_b = lambda_jj(zeta_b)  # (n_batch, kappa)
    
    # Variance calculations for minibatch
    if params['tau2_v'].ndim == 3:
        tau2_v_diag = jnp.diagonal(params['tau2_v'], axis1=1, axis2=2)  # (kappa, K)
    else:
        tau2_v_diag = params['tau2_v']
        
    if params['tau2_gamma'].ndim == 3:
        tau2_gamma_diag = jnp.diagonal(params['tau2_gamma'], axis1=1, axis2=2)  # (kappa, d)
    else:
        tau2_gamma_diag = params['tau2_gamma']
    
    var_theta_v_b = jnp.einsum('ik,ck->ic', E_theta_sq_b, tau2_v_diag)
    var_x_aux_gamma_b = (X_aux_b**2) @ tau2_gamma_diag.T
    var_psi_b = var_theta_v_b + var_x_aux_gamma_b
    
    # Logistic likelihood terms (scaled)
    logit_term1 = scale * jnp.sum((Y_b - 0.5) * psi_b)
    logit_term2 = -scale * jnp.sum(lam_b * (psi_b**2 + var_psi_b))
    logit_term3 = scale * jnp.sum(lam_b * zeta_b**2 - jnp.log(1.0 + jnp.exp(-zeta_b)) - zeta_b / 2.0)
    logit_ll = logit_term1 + logit_term2 + logit_term3
    
    # ========== KL DIVERGENCES ==========
    def kl_gamma(a_q, b_q, a0, b0):
        """KL divergence between two Gamma distributions"""
        term1 = (a_q - a0) * digamma(a_q)
        term2 = gammaln(a0) - gammaln(a_q)
        term3 = a0 * (jnp.log(b_q) - jnp.log(b0))
        term4 = (a_q / b_q) * (b0 - b_q)
        return term1 + term2 + term3 + term4
    
    # KL for theta (only for minibatch samples)
    E_xi_broadcast = jnp.broadcast_to(E_xi_b[:, None], a_theta_b.shape)
    kl_theta = jnp.sum(kl_gamma(
        jnp.clip(a_theta_b, 1e-8, 1e6),
        jnp.clip(b_theta_b, 1e-8, 1e6),
        model.alpha_theta,
        jnp.clip(E_xi_broadcast, 1e-8, 1e6)
    ))
    
    # KL for xi (only for minibatch samples)
    kl_xi = jnp.sum(kl_gamma(
        jnp.clip(a_xi_b, 1e-8, 1e6),
        jnp.clip(b_xi_b, 1e-8, 1e6),
        model.alpha_xi, 
        model.lambda_xi
    ))
    
    # KL for global parameters (full contribution, not scaled)
    E_eta_broadcast = jnp.broadcast_to(E_eta[:, None], params['a_beta'].shape)
    kl_beta = jnp.sum(kl_gamma(
        jnp.clip(params['a_beta'], 1e-8, 1e6),
        jnp.clip(params['b_beta'], 1e-8, 1e6),
        model.alpha_beta,
        jnp.clip(E_eta_broadcast, 1e-8, 1e6)
    ))
    
    kl_eta = jnp.sum(kl_gamma(
        jnp.clip(params['a_eta'], 1e-8, 1e6),
        jnp.clip(params['b_eta'], 1e-8, 1e6),
        model.alpha_eta, 
        model.lambda_eta
    ))
    
    # KL for v (global)
    tau2_v_safe = jnp.clip(tau2_v_diag, 1e-8, 1e8)
    mu_v_safe = jnp.clip(params['mu_v'], -100.0, 100.0)
    kl_v = 0.5 * jnp.sum(
        (mu_v_safe**2 + tau2_v_safe) / model.sigma2_v - 
        jnp.log(tau2_v_safe) + jnp.log(model.sigma2_v) - 1
    )
    
    # KL for gamma (global)
    tau2_gamma_safe = jnp.clip(tau2_gamma_diag, 1e-8, 1e8)
    mu_gamma_safe = jnp.clip(params['mu_gamma'], -100.0, 100.0)
    kl_gamma_coef = 0.5 * jnp.sum(
        (mu_gamma_safe**2 + tau2_gamma_safe) / model.sigma2_gamma - 
        jnp.log(tau2_gamma_safe) + jnp.log(model.sigma2_gamma) - 1
    )
    
    kl_total = kl_theta + kl_xi + kl_beta + kl_eta + kl_v + kl_gamma_coef
    elbo = pois_ll + logit_ll - kl_total
    
    if debug_print:
        print(f"[SVI ELBO] Poisson LL: {pois_ll:.3f}, Logistic LL: {logit_ll:.3f}")
        print(f"[SVI ELBO] KL total: {kl_total:.3f}, ELBO: {elbo:.3f}")
        print(f"[SVI ELBO] Scale factor: {scale:.3f}, Batch size: {n_batch}")
    
    if return_components:
        components = {
            "pois_ll": float(pois_ll),
            "logit_ll": float(logit_ll),
            "kl_theta": float(kl_theta),
            "kl_xi": float(kl_xi),
            "kl_beta": float(kl_beta),
            "kl_eta": float(kl_eta),
            "kl_v": float(kl_v),
            "kl_gamma": float(kl_gamma_coef),
            "elbo": float(elbo),
            "scale": float(scale)
        }
        return components
    
    return float(elbo)


def run_model_and_evaluate(
    x_data,
    x_aux,
    y_data,
    var_names,
    hyperparams,
    seed=None,
    test_size=0.15,
    val_size=0.15,
    max_iters=1000,
    batch_size=36,
    learning_rate=0.002,
    return_probs=True,
    sample_ids=None,
    mask=None,
    scores=None,
    plot_elbo=False,
    plot_prefix=None,
    return_params=False,
    verbose=False,
    early_stopping=True,
    patience=50,
    min_delta=1e-4,
    beta_init=None,
):
    """Fit the model using SVI and evaluate on data splits.
    
    Additional Parameters:
    ---------------------
    early_stopping : bool
        Whether to use early stopping based on ELBO convergence
    patience : int
        Number of ELBO checks to wait after last improvement before stopping
    min_delta : float
        Minimum change in ELBO to qualify as an improvement
    beta_init : array, optional
        Initial values for beta parameters (for pathway initialization)
    """
    # If seed is None, use a valid random integer seed for sklearn
    if seed is None:
        seed = np.random.randint(0, 2**32)

    if y_data.ndim == 1:
        y_data = y_data.reshape(-1, 1)
    if x_aux.ndim == 1:
        x_aux = x_aux.reshape(-1, 1)

    n_samples, n_genes = x_data.shape
    kappa = y_data.shape[1]
    d = hyperparams.get("d", 1)

    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(indices, test_size=val_size + test_size, random_state=seed)
    val_rel = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_rel, random_state=seed)

    model = SupervisedPoissonFactorization(
        len(train_idx),
        n_genes,
        n_factors=d,
        n_outcomes=kappa,
        alpha_eta=hyperparams.get("alpha_eta", 1.0),
        lambda_eta=hyperparams.get("lambda_eta", 1.0),
        alpha_beta=hyperparams.get("alpha_beta", 1.0),
        alpha_xi=hyperparams.get("alpha_xi", 1.0),
        lambda_xi=hyperparams.get("lambda_xi", 1.0),
        alpha_theta=hyperparams.get("alpha_theta", 1.0),
        sigma2_gamma=hyperparams.get("sigma2_gamma", 1.0),
        sigma2_v=hyperparams.get("sigma2_v", 1.0),
        key=random.PRNGKey(seed),
    )

    svi_results = fit_svi(
        model,
        x_data[train_idx],
        y_data[train_idx],
        x_aux[train_idx],
        n_iter=max_iters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose,
        track_elbo=plot_elbo or early_stopping,  # Enable ELBO tracking if early stopping is used
        elbo_freq=max(1, max_iters // 20),  # Track ~20 points
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        beta_init=beta_init,
    )

    params = svi_results['params']
    expected = svi_results['expected']

    # Make predictions on training data
    all_probs_train = logistic(
        expected["E_theta"] @ params["mu_v"].T + x_aux[train_idx] @ params["mu_gamma"].T
    )

    # For validation and test sets, use unsupervised theta inference
    # This only uses gene expression data and learned Poisson factorization parameters
    E_theta_val = infer_theta_unsupervised(model, x_data[val_idx], params, n_iter=20)
    all_probs_val = logistic(
        E_theta_val @ params["mu_v"].T + x_aux[val_idx] @ params["mu_gamma"].T
    )
    
    E_theta_test = infer_theta_unsupervised(model, x_data[test_idx], params, n_iter=20)
    all_probs_test = logistic(
        E_theta_test @ params["mu_v"].T + x_aux[test_idx] @ params["mu_gamma"].T
    )

    train_metrics = _compute_metrics(y_data[train_idx], np.array(all_probs_train))
    val_metrics = _compute_metrics(y_data[val_idx], np.array(all_probs_val))
    test_metrics = _compute_metrics(y_data[test_idx], np.array(all_probs_test))

    results = {
        "train_metrics": {k: v for k, v in train_metrics.items() if k != "probabilities"},
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "probabilities"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "probabilities"},
        "hyperparameters": hyperparams,
        "training_info": {
            "stopped_early": svi_results.get('stopped_early', False),
            "final_iteration": svi_results.get('final_iteration', max_iters),
            "best_elbo": svi_results.get('best_elbo', None)
        }
    }

    if return_probs:
        results["train_probabilities"] = train_metrics["probabilities"]
        results["val_probabilities"] = val_metrics["probabilities"]
        results["test_probabilities"] = test_metrics["probabilities"]

    results["val_labels"] = y_data[val_idx].tolist()

    if return_params:
        for k, v in params.items():
            if isinstance(v, jnp.ndarray):
                results[k] = np.array(v).tolist()

    # Plot and save ELBO if requested
    if plot_elbo and 'elbo_values' in svi_results and len(svi_results['elbo_values']) > 0:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(svi_results['elbo_iterations'], svi_results['elbo_values'], 
                    'b-', linewidth=2, label="ELBO")
            
            # Mark early stopping point if applicable
            if svi_results.get('stopped_early', False):
                final_iter = svi_results.get('final_iteration', len(svi_results['elbo_values']))
                final_elbo_idx = next((i for i, iter_num in enumerate(svi_results['elbo_iterations']) 
                                     if iter_num >= final_iter), -1)
                if final_elbo_idx >= 0:
                    plt.axvline(x=svi_results['elbo_iterations'][final_elbo_idx], 
                              color='red', linestyle='--', alpha=0.7, 
                              label=f"Early stop (iter {final_iter})")
            
            plt.xlabel("Iteration")
            plt.ylabel("Evidence Lower Bound (ELBO)")
            plt.title("ELBO Convergence During SVI Training")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot to file if plot_prefix is provided
            if plot_prefix is not None:
                import os
                os.makedirs(plot_prefix, exist_ok=True)
                plot_path = os.path.join(plot_prefix, "svi_elbo_convergence.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"ELBO plot saved to {plot_path}")
                
                # Also save as PDF for publications
                pdf_path = os.path.join(plot_prefix, "svi_elbo_convergence.pdf")
                plt.savefig(pdf_path, bbox_inches='tight')
            
            # Store plot data in results
            results['elbo_plot_data'] = {
                'iterations': svi_results['elbo_iterations'],
                'values': svi_results['elbo_values']
            }
            
            plt.close()  # Close to prevent memory issues
            
        except ImportError:
            print("Warning: matplotlib not available, skipping ELBO plot")
        except Exception as e:
            print(f"Warning: Failed to create ELBO plot: {e}")

    return results