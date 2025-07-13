import jax.numpy as jnp
import jax.random as random
from jax import vmap, jit
from typing import Dict, Tuple
import jax
import jax.scipy as jsp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Helper functions
def lambda_jj(zeta):
    """Jaakola-Jordan lambda function"""
    return jnp.where(zeta == 0, 1/8, (1/(4*zeta)) * jnp.tanh(zeta/2))

def logistic(x):
    """Logistic function"""
    return 1 / (1 + jnp.exp(-x))

class SupervisedPoissonFactorization:
    def __init__(self, n_samples, n_genes, n_factors, n_outcomes, 
                 alpha_eta=1.0, lambda_eta=1.0, alpha_beta=1.0, 
                 alpha_xi=1.0, lambda_xi=1.0, alpha_theta=1.0,
                 sigma2_gamma=1.0, sigma2_v=1.0, key=None):
        
        self.n, self.p, self.K, self.kappa = n_samples, n_genes, n_factors, n_outcomes
        self.alpha_eta, self.lambda_eta = alpha_eta, lambda_eta
        self.alpha_beta, self.alpha_xi = alpha_beta, alpha_xi
        self.lambda_xi, self.alpha_theta = lambda_xi, alpha_theta
        self.sigma2_gamma, self.sigma2_v = sigma2_gamma, sigma2_v
        
        if key is None:
            key = random.PRNGKey(0)
        self.key = key
        
    def initialize_parameters(self, X, Y, X_aux):
        """Initialize variational parameters using informed guesses"""
        keys = random.split(self.key, 10)
        
        # Use data statistics for better initialization
        gene_means = jnp.mean(X, axis=0)  # Average expression per gene
        sample_totals = jnp.sum(X, axis=1)  # Total expression per sample
        
        # Initialize theta based on sample activity (total gene expression)
        # Higher expressing samples get higher theta values
        theta_init = jnp.outer(sample_totals / jnp.mean(sample_totals), 
                              jnp.ones(self.K)) + random.normal(keys[0], (self.n, self.K)) * 0.1
        theta_init = jnp.maximum(theta_init, 0.1)  # Ensure positive
        
        # Initialize beta based on gene expression levels
        # More expressed genes get higher beta values
        beta_init = jnp.outer(gene_means / jnp.mean(gene_means), 
                             jnp.ones(self.K)) + random.normal(keys[1], (self.p, self.K)) * 0.1
        beta_init = jnp.maximum(beta_init, 0.1)  # Ensure positive
        
        # Convert to Gamma parameters (method of moments)
        # For Gamma(a,b): mean = a/b, var = a/b^2
        # Use mean and add some variance
        theta_var = theta_init * 0.5  # Assume variance is half the mean
        a_theta = (theta_init**2) / theta_var
        b_theta = theta_init / theta_var
        
        beta_var = beta_init * 0.5
        a_beta = (beta_init**2) / beta_var
        b_beta = beta_init / beta_var
        
        # Initialize eta based on current beta estimates
        a_eta = jnp.full(self.p, self.alpha_eta + self.K * self.alpha_beta)
        b_eta = self.lambda_eta + jnp.mean(a_beta / b_beta, axis=1)
        
        # Initialize xi based on current theta estimates  
        a_xi = jnp.full(self.n, self.alpha_xi + self.K * self.alpha_theta)
        b_xi = self.lambda_xi + jnp.sum(a_theta / b_theta, axis=1)
        
        # Initialize gamma using simple linear regression (if possible)
        if Y.shape[0] > 1:
            # Simple linear regression Y ~ X_aux for initialization
            try:
                # Solve normal equations: gamma = (X_aux^T X_aux)^{-1} X_aux^T Y
                XTX = X_aux.T @ X_aux + jnp.eye(X_aux.shape[1]) * 1e-6  # Ridge for stability
                XTY = X_aux.T @ Y
                gamma_init = jnp.linalg.solve(XTX, XTY).T  # (kappa, x_aux_dim)
            except:
                gamma_init = random.normal(keys[2], (self.kappa, X_aux.shape[1])) * 0.1
        else:
            gamma_init = random.normal(keys[2], (self.kappa, X_aux.shape[1])) * 0.1
        
        mu_gamma = gamma_init
        tau2_gamma = jnp.ones((self.kappa, X_aux.shape[1])) * self.sigma2_gamma
        
        # Initialize v using correlation with outcomes if possible
        if Y.shape[0] > 1:
            # Use correlation between sample totals and outcomes
            sample_features = sample_totals[:, None]  # (n, 1)
            try:
                # Simple regression Y ~ sample_features for each factor
                corr_init = jnp.corrcoef(sample_features.flatten(), Y.flatten())[0, 1]
                if jnp.isnan(corr_init):
                    corr_init = 0.0
                v_init = jnp.ones((self.kappa, self.K)) * corr_init * 0.1
            except:
                v_init = random.normal(keys[3], (self.kappa, self.K)) * 0.1
        else:
            v_init = random.normal(keys[3], (self.kappa, self.K)) * 0.1
            
        mu_v = v_init
        tau2_v = jnp.ones((self.kappa, self.K)) * self.sigma2_v
        
        # Initialize zeta based on expected activations
        # Use a reasonable guess for the linear predictor magnitude.
        # Compute expected theta values for each factor
        theta_exp = a_theta / b_theta  # (n, K)

        # Expected linear predictor for each outcome
        # (n, κ) = (n, K) @ (κ, K)^T + (n, d) @ (κ, d)^T
        expected_linear = theta_exp @ mu_v.T + X_aux @ mu_gamma.T

        if expected_linear.ndim == 1:
            expected_linear = expected_linear[:, None]

        zeta = jnp.abs(expected_linear) + 0.1  # Small offset for stability
        
        return {
            'a_eta': a_eta, 'b_eta': b_eta,
            'a_xi': a_xi, 'b_xi': b_xi,
            'a_beta': a_beta, 'b_beta': b_beta,
            'a_theta': a_theta, 'b_theta': b_theta,
            'mu_gamma': mu_gamma, 'tau2_gamma': tau2_gamma,
            'mu_v': mu_v, 'tau2_v': tau2_v,
            'zeta': zeta
        }
    
    def expected_values(self, params):
        """Compute expected values from variational parameters"""
        E_eta = params['a_eta'] / params['b_eta']
        E_xi = params['a_xi'] / params['b_xi']
        E_beta = params['a_beta'] / params['b_beta']
        E_theta = params['a_theta'] / params['b_theta']
        E_gamma = params['mu_gamma']
        E_v = params['mu_v']
        
        # Second moments for theta (needed for regression)
        E_theta_sq = (params['a_theta'] / params['b_theta']**2) * (params['a_theta'] + 1)
        E_theta_theta_T = jnp.expand_dims(E_theta_sq, -1) * jnp.eye(self.K) + \
                         jnp.expand_dims(E_theta, -1) @ jnp.expand_dims(E_theta, -2)
        
        return {
            'E_eta': E_eta, 'E_xi': E_xi, 'E_beta': E_beta, 'E_theta': E_theta,
            'E_gamma': E_gamma, 'E_v': E_v, 'E_theta_theta_T': E_theta_theta_T
        }
    
    def update_z_latent(self, X, E_theta, E_beta):
        """Update latent variables z_ijl using multinomial probabilities"""
        # Compute rates for each factor
        rates = jnp.expand_dims(E_theta, 1) * jnp.expand_dims(E_beta, 0)  # (n, p, K)
        total_rates = jnp.sum(rates, axis=2, keepdims=True)  # (n, p, 1)
        
        # Multinomial probabilities
        probs = rates / (total_rates + 1e-8)  # (n, p, K)
        
        # Expected z_ijl = x_ij * prob_ijl
        z = jnp.expand_dims(X, 2) * probs  # (n, p, K)
        
        return z
    
    def update_eta(self, params, expected_vals):
        """Update eta parameters - equation (13)"""
        a_eta_new = self.alpha_eta + self.K * self.alpha_beta
        b_eta_new = self.lambda_eta + jnp.sum(expected_vals['E_beta'], axis=1)
        
        return {'a_eta': jnp.full(self.p, a_eta_new), 'b_eta': b_eta_new}
    
    def update_xi(self, params, expected_vals):
        """Update xi parameters - equation (14)"""
        a_xi_new = self.alpha_xi + self.K * self.alpha_theta
        b_xi_new = self.lambda_xi + jnp.sum(expected_vals['E_theta'], axis=1)
        
        return {'a_xi': jnp.full(self.n, a_xi_new), 'b_xi': b_xi_new}
    
    def update_beta(self, params, expected_vals, z):
        """Update beta parameters - equation (20)"""
        a_beta_new = self.alpha_beta + jnp.sum(z, axis=0)  # (p, K)
        b_beta_new = jnp.expand_dims(expected_vals['E_eta'], 1) + \
                     jnp.sum(expected_vals['E_theta'], axis=0)  # (p, K)
        
        return {'a_beta': a_beta_new, 'b_beta': b_beta_new}
    
    def update_v(self, params, expected_vals, Y, X_aux):
        """Update v parameters - equation (10)"""
        lambda_vals = lambda_jj(params['zeta'])  # (n, kappa)
        
        # Compute precision matrix (diagonal + low-rank update)
        prec_diag = 1/self.sigma2_v + 2 * jnp.sum(
            jnp.expand_dims(lambda_vals, -1) * 
            jnp.expand_dims(jnp.diagonal(expected_vals['E_theta_theta_T'], axis1=1, axis2=2), 1), 
            axis=0
        )  # (kappa, K)
        
        # Compute mean
        y_centered = Y - 0.5  # (n, kappa)
        reg_term = 2 * lambda_vals * jnp.sum(
            jnp.expand_dims(expected_vals['E_theta'], 1) * 
            jnp.expand_dims(X_aux @ expected_vals['E_gamma'].T, -1), 
            axis=2
        )  # (n, kappa)
        
        linear_term = jnp.sum(
            jnp.expand_dims(y_centered - reg_term, -1) * 
            jnp.expand_dims(expected_vals['E_theta'], 1), 
            axis=0
        )  # (kappa, K)
        
        # Update (assuming diagonal approximation for simplicity)
        tau2_v_new = 1 / prec_diag
        mu_v_new = tau2_v_new * linear_term
        
        return {'mu_v': mu_v_new, 'tau2_v': tau2_v_new}
    
    def update_gamma(self, params, expected_vals, Y, X_aux):
        """Update gamma parameters - equation (11)"""
        lambda_vals = lambda_jj(params['zeta'])  # (n, kappa)
        
        # Compute precision matrix
        prec_diag = 1/self.sigma2_gamma + 2 * jnp.einsum('ic,id->cd', lambda_vals, X_aux**2)
        
        # Compute mean
        y_centered = Y - 0.5  # (n, kappa)
        reg_term = 2 * lambda_vals * jnp.sum(
            jnp.expand_dims(expected_vals['E_theta'], 1) * 
            jnp.expand_dims(expected_vals['E_v'], 0), 
            axis=2
        )  # (n, kappa)
        
        linear_term = jnp.einsum('ic,id->cd', y_centered - reg_term, X_aux)
        
        # Update
        tau2_gamma_new = 1 / prec_diag
        mu_gamma_new = tau2_gamma_new * linear_term
        
        return {'mu_gamma': mu_gamma_new, 'tau2_gamma': tau2_gamma_new}
    
    def update_theta(self, params, expected_vals, z, Y, X_aux):
        """Update theta parameters - equation (27)"""
        # Shape parameter
        a_theta_new = self.alpha_theta + jnp.sum(z, axis=1)  # (n, K)
        
        # Rate parameter - using current theta for linearization
        E_theta_current = expected_vals['E_theta']
        
        # Poisson terms
        poisson_rate = jnp.expand_dims(expected_vals['E_xi'], 1) + \
                       jnp.sum(expected_vals['E_beta'], axis=0)  # (n, K)
        
        # Regression terms  
        lambda_vals = lambda_jj(params['zeta'])  # (n, kappa)
        y_centered = Y - 0.5  # (n, kappa)
        
        reg_rate = jnp.sum(
            jnp.expand_dims(lambda_vals, -1) * (
                -jnp.expand_dims(y_centered, -1) * jnp.expand_dims(expected_vals['E_v'], 0) -
                2 * jnp.expand_dims(expected_vals['E_v'], 0) * 
                jnp.expand_dims(X_aux @ expected_vals['E_gamma'].T, -1) +
                2 * jnp.expand_dims(expected_vals['E_v']**2, 0) * 
                jnp.expand_dims(E_theta_current, 1)
            ), 
            axis=1
        )  # (n, K)
        
        b_theta_new = poisson_rate + reg_rate
        
        return {'a_theta': a_theta_new, 'b_theta': b_theta_new}
    
    def update_zeta(self, params, expected_vals, Y, X_aux):
        """Update auxiliary parameters zeta for Jaakola-Jordan bound"""
        # Compute A = theta_i * v_k^T + x_aux_i * gamma_k^T
        A = jnp.sum(jnp.expand_dims(expected_vals['E_theta'], 1) * 
                   jnp.expand_dims(expected_vals['E_v'], 0), axis=2) + \
            X_aux @ expected_vals['E_gamma'].T  # (n, kappa)
        
        # Update zeta to tighten the bound
        zeta_new = jnp.abs(A)
        
        return {'zeta': zeta_new}
    
    def fit(self, X, Y, X_aux, n_iter=100, tol=1e-4):
        """Main fitting loop"""
        params = self.initialize_parameters(X, Y, X_aux)
        
        for iteration in range(n_iter):
            old_params = params.copy()
            
            # Compute expected values
            expected_vals = self.expected_values(params)
            
            # Update latent variables
            z = self.update_z_latent(X, expected_vals['E_theta'], expected_vals['E_beta'])
            
            # Update parameters
            params.update(self.update_eta(params, expected_vals))
            params.update(self.update_xi(params, expected_vals))
            params.update(self.update_beta(params, expected_vals, z))
            params.update(self.update_zeta(params, expected_vals, Y, X_aux))
            params.update(self.update_v(params, expected_vals, Y, X_aux))
            params.update(self.update_gamma(params, expected_vals, Y, X_aux))
            params.update(self.update_theta(params, expected_vals, z, Y, X_aux))
            
            # Check convergence (simple check on theta means)
            if iteration > 0:
                theta_diff = jnp.mean(jnp.abs(params['a_theta']/params['b_theta'] - 
                                             old_params['a_theta']/old_params['b_theta']))
                if theta_diff < tol:
                    print(f"Converged after {iteration+1} iterations")
                    break
        
        return params, expected_vals

    # ----------------- ELBO (unchanged, but with λ clip) ----------------
    def compute_elbo(self, X, Y, X_aux,
                    params: Dict[str, jnp.ndarray]) -> float:
        """Exact mean‑field ELBO (constants dropped)."""

        digamma   = jsp.special.digamma
        gammaln   = jsp.special.gammaln

        # ---------- Expectations ----------
        E_theta     = params['a_theta'] / params['b_theta']          # (n,K)
        E_theta_sq  = (params['a_theta'] * (params['a_theta'] + 1)) / params['b_theta']**2
        E_log_theta = digamma(params['a_theta']) - jnp.log(params['b_theta'])

        E_beta      = params['a_beta'] / params['b_beta']           # (p,K)
        E_log_beta  = digamma(params['a_beta'])  - jnp.log(params['b_beta'])

        # ---------- Poisson likelihood (latent‑count form) ----------
        pois_ll = (
            X[:, :, None] * (E_log_theta[:, None, :] + E_log_beta[None, :, :])
            - jnp.einsum('ik,jk->ij', E_theta, E_beta)
        ).sum()

        # ---------- Logistic JJ bound --------------------------------
        psi     = E_theta @ params['mu_v'].T + X_aux @ params['mu_gamma'].T  # (n,κ)
        zeta    = params['zeta']
        lam     = lambda_jj(zeta)                                            # (n,κ)

        # full variance of ψ
        var_psi = jnp.einsum('ik,ck->ic', E_theta_sq, params['tau2_v']) + \
                (X_aux ** 2) @ params['tau2_gamma'].T                    # (n,κ)

        logit_ll = jnp.sum(
            (Y - 0.5) * psi
            - lam * (psi**2 + var_psi)      # note the −ζ² term
            + lam * zeta**2
            - zeta / 2.0
        )

        # ---------- KL divergences (Gamma / Gaussian) ---------------
        def kl_gamma(a_q, b_q, a0, b0):
            """KL(q||p) for Gamma(shape, rate) random variables."""
            term1 = -a_q * jnp.log(b_q) + a0 * jnp.log(b0)
            term2 = gammaln(a0) - gammaln(a_q)
            term3 = (a_q - a0) * (digamma(a_q) - jnp.log(b_q))
            term4 = a0 * (b_q - b0) / b0
            return term1 + term2 + term3 + term4

        kl_theta = kl_gamma(params['a_theta'], params['b_theta'],
                            self.alpha_theta,
                            params['b_xi'][:, None]).sum()

        kl_beta  = kl_gamma(params['a_beta'], params['b_beta'],
                            self.alpha_beta,
                            params['b_eta'][:, None]).sum()

        kl_eta   = kl_gamma(params['a_eta'], params['b_eta'],
                            self.alpha_eta + self.K * self.alpha_beta,
                            self.lambda_eta).sum()

        kl_xi    = kl_gamma(params['a_xi'], params['b_xi'],
                            self.alpha_xi + self.K * self.alpha_theta,
                            self.lambda_xi).sum()

        kl_v = 0.5 * jnp.sum(
            params['mu_v']**2 + params['tau2_v'] - jnp.log(params['tau2_v']) - 1
        ) / self.sigma2_v

        kl_gamma_coef = 0.5 * jnp.sum(
            params['mu_gamma']**2 + params['tau2_gamma'] - jnp.log(params['tau2_gamma']) - 1
        ) / self.sigma2_gamma

        kl_total = kl_theta + kl_beta + kl_eta + kl_xi + kl_v + kl_gamma_coef

        return (pois_ll + logit_ll - kl_total).item()


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
        metrics["per_class_metrics"] = per_class

    metrics["probabilities"] = probs.tolist()
    return metrics


def run_model_and_evaluate(
    x_data,
    x_aux,
    y_data,
    var_names,
    hyperparams,
    seed=None,
    test_size=0.15,
    val_size=0.15,
    max_iters=100,
    return_probs=True,
    sample_ids=None,
    mask=None,
    scores=None,
    plot_elbo=False,
    plot_prefix=None,
    return_params=False,
):
    """Fit the model on all data and evaluate on splits."""

    if seed is None:
        seed = 0

    if y_data.ndim == 1:
        y_data = y_data.reshape(-1, 1)
    if x_aux.ndim == 1:
        x_aux = x_aux.reshape(-1, 1)

    n_samples, n_genes = x_data.shape
    kappa = y_data.shape[1]
    d = hyperparams.get("d", 1)

    model = SupervisedPoissonFactorization(
        n_samples,
        n_genes,
        d,
        kappa,
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

    params, expected = model.fit(x_data, y_data, x_aux, n_iter=max_iters)

    all_probs = logistic(
        expected["E_theta"] @ params["mu_v"].T + x_aux @ params["mu_gamma"].T
    )
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_size + test_size, random_state=seed
    )
    val_rel = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=val_rel, random_state=seed
    )

    train_metrics = _compute_metrics(y_data[train_idx], np.array(all_probs)[train_idx])
    val_metrics = _compute_metrics(y_data[val_idx], np.array(all_probs)[val_idx])
    test_metrics = _compute_metrics(y_data[test_idx], np.array(all_probs)[test_idx])

    results = {
        "train_metrics": {k: v for k, v in train_metrics.items() if k != "probabilities"},
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "probabilities"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "probabilities"},
        "hyperparameters": hyperparams,
    }

    if return_probs:
        results["train_probabilities"] = train_metrics["probabilities"]
        results["val_probabilities"] = val_metrics["probabilities"]
        results["test_probabilities"] = test_metrics["probabilities"]

    if return_params:
        for k, v in params.items():
            if isinstance(v, jnp.ndarray):
                results[k] = np.array(v).tolist()

    return results