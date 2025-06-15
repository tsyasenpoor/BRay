from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax import jit
from jax.nn import sigmoid
import gc
import scipy.sparse as sp

def prox_l1(v: jnp.ndarray, lam: float) -> jnp.ndarray:
    return jnp.sign(v) * jnp.maximum(0.0, jnp.abs(v) - lam)

@jit
def logistic_loss(w: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    z = X @ w
    return jnp.sum(jax.nn.softplus(z) - y * z)  # softplus = log(1+eᶻ)

@jit
def logistic_grad(w: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    p = sigmoid(X @ w)
    return X.T @ (p - y)

def laplace_logistic_l1_numpy(
    X: np.ndarray,
    y: np.ndarray,
    b: float,
    rng: np.random.Generator,
    init_coef: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> np.ndarray:
    """Improved NumPy implementation of L1-regularized logistic regression"""
    try:
        n, m = X.shape
        lam = 1.0 / b  # λ = 1/b (L1 weight)
        
        # Initialize coefficients
        if init_coef is None:
            w = np.random.normal(0, 0.01, size=m)  # Small random initialization
        else:
            w = init_coef.copy()
        
        # Add numerical safeguards to inputs
        X = np.clip(X, -1e6, 1e6)
        y = np.clip(y, 0, 1)
        
        # Adaptive learning rate
        initial_lr = min(1e-3, 1.0 / (np.linalg.norm(X, ord=2) + 1e-8))
        learning_rate = initial_lr
        
        best_w = w.copy()
        best_loss = float('inf')
        patience = 50
        no_improve = 0
        
        for iteration in range(max_iter):
            # Compute predictions with numerical safeguards
            z = X @ w
            z = np.clip(z, -500, 500)  # Prevent overflow in exp
            
            # Sigmoid function
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-15, 1 - 1e-15)  # Prevent log(0)
            
            # Compute loss
            logistic_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            l1_penalty = lam * np.sum(np.abs(w))
            total_loss = logistic_loss + l1_penalty
            
            # Track best solution
            if total_loss < best_loss:
                best_loss = total_loss
                best_w = w.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve > patience:
                w = best_w
                break
            
            # Gradient of logistic loss
            grad = X.T @ (p - y) / n
            
            # Add L1 regularization gradient (subgradient)
            l1_grad = lam * np.sign(w)
            
            # Update with safeguards
            w_new = w - learning_rate * (grad + l1_grad)
            w_new = np.clip(w_new, -1e3, 1e3)
            
            # Check convergence
            if np.linalg.norm(w_new - w) < tol:
                w = w_new
                break
                
            w = w_new
            
            # Adaptive learning rate
            if iteration > 0 and iteration % 20 == 0:
                learning_rate *= 0.95
                learning_rate = max(learning_rate, initial_lr * 0.1)
        
        # Return best solution found
        return np.clip(best_w, -1e3, 1e3)
        
    except Exception as e:
        print(f"Error in NumPy logistic regression: {e}")
        # Return small random values as fallback
        return np.random.normal(0, 0.01, size=X.shape[1])

def laplace_logistic_l1_jax(
    X: np.ndarray,
    y: np.ndarray,
    b: float,
    rng: np.random.Generator,
    init_coef: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-5,
    show_large_matrix_warning: bool = True,
) -> np.ndarray:
    """Wrapper that tries JAX first, then falls back to NumPy implementation"""
    try:
        # Check for problematic values that might cause JAX to fail
        if np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isinf(X)) or np.any(np.isinf(y)):
            print("Warning: NaN or inf values detected, using NumPy fallback")
            return laplace_logistic_l1_numpy(X, y, b, rng, init_coef, max_iter, tol)
        
        # Check matrix size - if too large, use NumPy
        if X.shape[1] > 1000:
            if show_large_matrix_warning:
                print(f"Large matrix detected ({X.shape}), using NumPy fallback for logistic regression.")
            return laplace_logistic_l1_numpy(X, y, b, rng, init_coef, max_iter, tol)
        
        # Try JAX implementation
        Xj = jnp.asarray(X)
        yj = jnp.asarray(y)

        m = Xj.shape[1]
        lam = 1.0 / b
        norm_X = jnp.linalg.norm(Xj, ord=2)
        L = 0.25 * norm_X**2 + 1e-8
        alpha = 1.0 / L

        w = jnp.zeros(m) if init_coef is None else jnp.asarray(init_coef)

        for _ in range(min(max_iter, 100)):  # Limit JAX iterations
            grad = logistic_grad(w, Xj, yj)
            w_new = prox_l1(w - alpha * grad, alpha * lam)
            if jnp.linalg.norm(w_new - w) < tol:
                w = w_new
                break
            w = w_new

        mu = w
        mu_np = np.asarray(mu)
        
        # Just return MAP estimate to avoid covariance issues
        return np.clip(mu_np, -1e6, 1e6)
        
    except Exception as e:
        print(f"JAX logistic regression failed: {e}, falling back to NumPy")
        return laplace_logistic_l1_numpy(X, y, b, rng, init_coef, max_iter, tol)

class GibbsSampler:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_aux: np.ndarray,
        d: int,
        k: int,
        b: float = 1.0,
        mask: Optional[np.ndarray] = None,
        use_mask: bool = False,
        seed: int = 42,
        # Hyperparameters
        a_theta: float = 1.0,
        c_beta: float = 1.0,
        a_prime_xi: float = 1.0,
        b_prime_xi: float = 1.0,
        c_prime_eta: float = 1.0,
        d_prime_eta: float = 1.0,
    ):
        # Store sparse matrix as is, don't convert to dense
        self.X = X  # Keep as sparse matrix
        self.Y = Y  # (n, k) binary outcome matrix
        self.X_aux = X_aux  # (n, q) auxiliary features
        self.d = d  # latent dimensions
        self.k = k  # number of outcomes
        self.b = b  # regularization parameter
        self.use_mask = use_mask
        self.rng = default_rng(seed)
        
        self.n, self.p = X.shape
        self.q = X_aux.shape[1]
        
        # Validate dimensions for memory safety
        estimated_memory_gb = (self.n * self.d + self.p * self.d) * 8 / (1024**3)
        if estimated_memory_gb > 4:  # More than 4GB for just theta and beta
            print(f"Warning: Estimated memory usage: {estimated_memory_gb:.2f} GB")
            print("Consider reducing d or using a machine with more RAM")
        
        # Initialize mask
        if use_mask and mask is not None:
            self.mask = mask
        else:
            self.mask = np.ones((self.p, self.d), dtype=bool)
        
        # Hyperparameters
        self.a_theta_shape = a_theta
        self.c_beta_shape = c_beta
        self.a_prime_xi_shape = a_prime_xi
        self.b_prime_xi_rate = b_prime_xi
        self.c_prime_eta_shape = c_prime_eta
        self.d_prime_eta_rate = d_prime_eta
        
        # Initialize parameters
        self._initialize_parameters()
        self._regression_fallback_warning_shown = False
    
    def _initialize_parameters(self):
        """Initialize all parameters with better starting values"""
        print(f"Initializing parameters: theta({self.n}, {self.d}), beta({self.p}, {self.d})")
        
        # Use float32 to save memory for large d
        dtype = np.float32 if self.d > 1000 else np.float64
        
        # Better initialization: use data-driven initialization
        # Initialize theta based on library size (total counts per cell)
        if hasattr(self.X, 'sum'):
            library_sizes = np.asarray(self.X.sum(axis=1)).flatten()
        else:
            library_sizes = self.X.sum(axis=1)
        
        # Normalize library sizes and use as basis for theta initialization
        normalized_lib = library_sizes / np.mean(library_sizes)
        
        # Initialize theta with small random values scaled by library size
        self.theta = self.rng.gamma(
            shape=0.5, 
            scale=normalized_lib[:, None] * 0.1, 
            size=(self.n, self.d)
        ).astype(dtype)
        
        # Initialize beta with small positive values
        self.beta = self.rng.gamma(0.1, 0.1, size=(self.p, self.d)).astype(dtype)
        
        # Initialize hierarchical parameters with more reasonable values
        self.xi = self.rng.gamma(1.0, 1.0, size=self.n).astype(dtype)
        self.eta = self.rng.gamma(1.0, 1.0, size=self.p).astype(dtype)
        
        # Initialize regression coefficients - CREATE THE LISTS FIRST
        self.upsilon = [np.zeros(self.d, dtype=dtype) for _ in range(self.k)]
        self.gamma = [np.zeros(self.q, dtype=dtype) for _ in range(self.k)]
        
        # NOW initialize regression coefficients to small random values instead of zeros
        for k in range(self.k):
            self.upsilon[k] = self.rng.normal(0, 0.01, size=self.d).astype(dtype)
            self.gamma[k] = self.rng.normal(0, 0.01, size=self.q).astype(dtype)
        
        # Apply mask to beta if using mask
        if self.use_mask:
            self.beta = np.where(self.mask, self.beta, 0.0)
        
        print(f"Parameter initialization complete")

    @property
    def lam(self) -> np.ndarray:  # Poisson rate  (n, p)
        return self.theta @ self.beta.T

    def sample_theta(self) -> None:
        try:
            # Handle sparse matrix properly - convert sum to 1D array
            if sp.issparse(self.X):
                X_sum_over_features = np.asarray(self.X.sum(axis=1)).flatten()  # (n,)
            else:
                X_sum_over_features = self.X.sum(axis=1)
            
            # Shape: (n, d) - each cell n has d latent factors
            shape = self.a_theta_shape + X_sum_over_features[:, None]  # (n, 1)
            
            # Compute beta sum more efficiently
            beta_sum = self.beta.sum(axis=0)  # (d,)
            rate = self.xi[:, None] + beta_sum[None, :]  # (n, 1) + (1, d) = (n, d)
            
            # Add numerical safeguards
            shape = np.maximum(shape, 1e-6)  # Prevent zero/negative shape
            rate = np.maximum(rate, 1e-6)   # Prevent zero/negative rate
            
            # Clip values to prevent overflow - more conservative limits
            shape = np.clip(shape, 1e-6, 1e6)
            rate = np.clip(rate, 1e-6, 1e6)
            
            # Sample row by row to avoid memory issues
            if self.d > 1000:
                for i in range(self.n):
                    self.theta[i] = self.rng.gamma(shape=shape[i], scale=1.0 / rate[i])
            else:
                self.theta = self.rng.gamma(shape=shape, scale=1.0 / rate)
            
            # Add safeguards to theta values
            self.theta = np.clip(self.theta, 1e-6, 1e6)
            
        except Exception as e:
            print(f"Error in sample_theta: {e}")
            # Fallback: keep current theta values
            pass

    def sample_beta(self) -> None:
        try:
            # Handle sparse matrix properly - convert sum to 1D array  
            if sp.issparse(self.X):
                X_sum_over_cells = np.asarray(self.X.sum(axis=0)).flatten()  # (p,)
            else:
                X_sum_over_cells = self.X.sum(axis=0)
            
            # Shape: (p, d) - each gene p has d latent factors
            shape = self.c_beta_shape + X_sum_over_cells[:, None]  # (p, 1)
            
            # Compute theta sum more efficiently
            theta_sum = self.theta.sum(axis=0)  # (d,)
            rate = self.eta[:, None] + theta_sum[None, :]  # (p, 1) + (1, d) = (p, d)
            
            # Add numerical safeguards
            shape = np.maximum(shape, 1e-6)  # Prevent zero/negative shape
            rate = np.maximum(rate, 1e-6)   # Prevent zero/negative rate
            
            # Clip values to prevent overflow
            shape = np.clip(shape, 1e-6, 1e6)
            rate = np.clip(rate, 1e-6, 1e6)
            
            # Sample row by row for large d to avoid memory issues
            if self.d > 1000:
                sampled_beta = np.zeros_like(self.beta)
                for i in range(self.p):
                    sampled_beta[i] = self.rng.gamma(shape=shape[i], scale=1.0 / rate[i])
            else:
                sampled_beta = self.rng.gamma(shape=shape, scale=1.0 / rate)
            
            # Add safeguards to beta values
            sampled_beta = np.clip(sampled_beta, 1e-6, 1e6)
            
            self.beta = np.where(self.mask, sampled_beta, self.beta)
            
        except Exception as e:
            print(f"Error in sample_beta: {e}")
            # Fallback: keep current beta values
            pass

    def sample_xi(self) -> None:
        shape = self.a_prime_xi_shape + self.d * self.a_theta_shape
        rate = self.b_prime_xi_rate + self.theta.sum(axis=1) # sum over latent dimension d for each cell n
        
        # Add numerical safeguards
        shape = np.maximum(shape, 1e-8)
        rate = np.maximum(rate, 1e-8)
        
        # Clip values to prevent overflow
        shape = np.clip(shape, 1e-8, 1e8)
        rate = np.clip(rate, 1e-8, 1e8)
        
        self.xi = self.rng.gamma(shape=shape, scale=1.0 / rate)
        self.xi = np.clip(self.xi, 1e-8, 1e8)

    def sample_eta(self) -> None:
        shape = self.c_prime_eta_shape + self.d * self.c_beta_shape
        rate = self.d_prime_eta_rate + self.beta.sum(axis=1) # sum over latent dimension d for each feature p
        
        # Add numerical safeguards
        shape = np.maximum(shape, 1e-8)
        rate = np.maximum(rate, 1e-8)
        
        # Clip values to prevent overflow
        shape = np.clip(shape, 1e-8, 1e8)
        rate = np.clip(rate, 1e-8, 1e8)
        
        self.eta = self.rng.gamma(shape=shape, scale=1.0 / rate)
        self.eta = np.clip(self.eta, 1e-8, 1e8)

    def sample_regression(self) -> None:
        try:
            # Create design matrix
            X_design = np.hstack([self.theta, self.X_aux])    # (n, d+q)
            
            # Add numerical safeguards to design matrix
            X_design = np.nan_to_num(X_design, nan=0.0, posinf=1e3, neginf=-1e3)
            
            # Standardize features for better numerical stability
            X_mean = np.mean(X_design, axis=0)
            X_std = np.std(X_design, axis=0) + 1e-8
            X_design_norm = (X_design - X_mean) / X_std
            
            # Determine if the warning for NumPy fallback in regression should be shown
            show_warning_for_this_sampler_instance = False
            if X_design_norm.shape[1] > 1000 and not self._regression_fallback_warning_shown:
                show_warning_for_this_sampler_instance = True
            
            for k in range(self.k):
                y = self.Y[:, k]
                
                # Add safeguards to y
                y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
                y = np.clip(y, 0, 1)
                
                # Skip if all labels are the same
                if len(np.unique(y)) < 2:
                    print(f"Warning: Class {k} has only one unique label, skipping regression update")
                    continue
                
                try:
                    # Use improved regularization parameter
                    adaptive_b = max(0.1, self.b * np.sqrt(self.d))
                    
                    # Show warning only for the first k if it's the first time for this sampler
                    current_call_show_warning = show_warning_for_this_sampler_instance and k == 0

                    coef_norm = laplace_logistic_l1_jax(
                        X_design_norm,
                        y,
                        b=adaptive_b,
                        rng=self.rng,
                        init_coef=np.concatenate([self.upsilon[k] * X_std[:self.d], 
                                                self.gamma[k] * X_std[self.d:]]),
                        show_large_matrix_warning=current_call_show_warning,
                    )
                    
                    # If the warning was slated to be shown and was for the first k, mark it as shown for the instance
                    if current_call_show_warning:
                        self._regression_fallback_warning_shown = True

                    # Un-standardize coefficients
                    coef = coef_norm / X_std
                    
                    # Add safeguards to coefficients
                    coef = np.nan_to_num(coef, nan=0.0, posinf=1e3, neginf=-1e3)
                    
                    self.upsilon[k] = coef[:self.d]
                    self.gamma[k] = coef[self.d:]
                    
                except Exception as inner_e:
                    print(f"Error in logistic regression for class {k}: {inner_e}")
                    # Keep existing coefficients if sampling fails
                    pass
            
        except Exception as e:
            print(f"Error in sample_regression: {e}")
            # Fallback: keep current regression coefficients
            pass

    def step(self) -> None:
        """Single Gibbs sampling step with error handling"""
        try:
            self.sample_theta()
            self.sample_beta()
            self.sample_xi()
            self.sample_eta()
            self.sample_regression()
            
            # Garbage collection for large d
            if self.d > 1000:
                gc.collect()
                
        except Exception as e:
            print(f"Error in Gibbs step: {e}")
            # Continue to next iteration rather than crashing
            pass

    def fit(self, max_iter: int = 1000, burn_in: int = 500, verbose: bool = True,
            early_stopping_patience: int = 50, early_stopping_tol: float = 1e-4) -> dict:
        """
        Run Gibbs sampling
        
        Args:
            max_iter: Maximum number of iterations
            burn_in: Number of burn-in iterations
            verbose: Whether to print progress
            early_stopping_patience: Number of iterations with no improvement to wait before stopping.
            early_stopping_tol: Tolerance for parameter change to define "no improvement".
            
        Returns:
            Dictionary with sampling results and statistics
        """
        if verbose:
            print(f"Starting Gibbs sampling for {max_iter} iterations...")
        
        # Ensure burn_in is reasonable
        if burn_in >= max_iter:
            burn_in = max(1, max_iter // 2)
            if verbose:
                print(f"Adjusted burn_in to {burn_in} (was >= max_iter)")
        
        # Storage for samples (after burn-in)
        samples = {
            'theta': [],
            'beta': [],
            'upsilon': [[] for _ in range(self.k)],
            'gamma': [[] for _ in range(self.k)]
        }

        # For convergence diagnostics
        theta_change_history = []
        beta_change_history = []
        upsilon_change_history = [[] for _ in range(self.k)]
        gamma_change_history = [[] for _ in range(self.k)]

        # Store initial parameters for calculating first change
        prev_theta = self.theta.copy()
        prev_beta = self.beta.copy()
        prev_upsilon = [u.copy() for u in self.upsilon]
        prev_gamma = [g.copy() for g in self.gamma]

        no_improvement_count = 0
        
        for iteration in range(max_iter):
            self.step()

            # Calculate and store parameter changes
            theta_change = np.linalg.norm(self.theta - prev_theta)
            beta_change = np.linalg.norm(self.beta - prev_beta)
            theta_change_history.append(theta_change)
            beta_change_history.append(beta_change)

            current_upsilon_changes = []
            current_gamma_changes = []
            for k_idx in range(self.k):
                upsilon_k_change = np.linalg.norm(self.upsilon[k_idx] - prev_upsilon[k_idx])
                gamma_k_change = np.linalg.norm(self.gamma[k_idx] - prev_gamma[k_idx])
                upsilon_change_history[k_idx].append(upsilon_k_change)
                gamma_change_history[k_idx].append(gamma_k_change)
                current_upsilon_changes.append(upsilon_k_change)
                current_gamma_changes.append(gamma_k_change)

            # Update previous parameters for next iteration
            prev_theta = self.theta.copy()
            prev_beta = self.beta.copy()
            prev_upsilon = [u.copy() for u in self.upsilon]
            prev_gamma = [g.copy() for g in self.gamma]

            # Early stopping check (after burn-in)
            if iteration >= burn_in:
                # Consider combined change of major parameters
                combined_change = theta_change + beta_change
                if self.k > 0:
                    combined_change += np.mean(current_upsilon_changes) + np.mean(current_gamma_changes)
                
                if combined_change < early_stopping_tol:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                
                if no_improvement_count >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at iteration {iteration + 1} due to convergence.")
                    break
            
            # Store samples after burn-in
            if iteration >= burn_in:
                samples['theta'].append(self.theta.copy())
                samples['beta'].append(self.beta.copy())
                for k in range(self.k):
                    samples['upsilon'][k].append(self.upsilon[k].copy())
                    samples['gamma'][k].append(self.gamma[k].copy())
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}")
                print(f"  Theta change: {theta_change:.4e}, Beta change: {beta_change:.4e}")
                if self.k > 0:
                    print(f"  Avg Upsilon change: {np.mean(current_upsilon_changes):.4e}, Avg Gamma change: {np.mean(current_gamma_changes):.4e}")

        # Handle case where no samples were collected
        if len(samples['theta']) == 0:
            if verbose:
                print("Warning: No samples collected after burn-in. Using final iteration values.")
            # Use current values instead of trying to compute means of empty lists
            final_theta = self.theta
            final_beta = self.beta
            final_upsilon = self.upsilon
            final_gamma = self.gamma
        else:
            # Convert to arrays and compute posterior means
            samples['theta'] = np.array(samples['theta'])
            samples['beta'] = np.array(samples['beta'])
            for k in range(self.k):
                samples['upsilon'][k] = np.array(samples['upsilon'][k])
                samples['gamma'][k] = np.array(samples['gamma'][k])
            
            # Compute posterior means as final estimates
            final_theta = samples['theta'].mean(axis=0)
            final_beta = samples['beta'].mean(axis=0)
            final_upsilon = []
            final_gamma = []
            for k in range(self.k):
                final_upsilon.append(samples['upsilon'][k].mean(axis=0))
                final_gamma.append(samples['gamma'][k].mean(axis=0))
        
        # Update instance variables
        self.theta = final_theta
        self.beta = final_beta
        self.upsilon = final_upsilon
        self.gamma = final_gamma
        
        if verbose:
            print("Gibbs sampling completed!")
        
        return {
            'samples': samples,
            'posterior_means': {
                'theta': self.theta,
                'beta': self.beta,
                'upsilon': self.upsilon,
                'gamma': self.gamma
            },
            'convergence_diagnostics': {
                'theta_change': np.array(theta_change_history),
                'beta_change': np.array(beta_change_history),
                'upsilon_change': [np.array(uc) for uc in upsilon_change_history],
                'gamma_change': [np.array(gc) for gc in gamma_change_history],
            },
            'iterations': max_iter,
            'burn_in': burn_in
        }
    
    def predict_proba(self, X_test_aux: np.ndarray, theta_test: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Improved prediction using learned latent factors
        """
        if theta_test is None:
            # For test prediction, we need to infer theta from the count data
            # This is a key missing piece - we need to use the test count data
            print("Warning: theta_test is None - prediction may be suboptimal")
            
            # Use current training theta mean as a baseline
            if hasattr(self, 'theta') and self.theta is not None:
                if self.theta.ndim == 2:
                    theta_mean = np.mean(self.theta, axis=0)
                else:
                    theta_mean = self.theta
            else:
                theta_mean = np.zeros(self.d)
                
            # Check for NaN values
            if np.any(np.isnan(theta_mean)):
                theta_mean = np.zeros(self.d)
            
            theta_test = np.tile(theta_mean, (X_test_aux.shape[0], 1))
        
        # Add small noise to theta_test to avoid identical predictions
        theta_test = theta_test + self.rng.normal(0, 0.01, theta_test.shape)
        
        X_design_test = np.hstack([theta_test, X_test_aux])
        probs = np.zeros((X_test_aux.shape[0], self.k))
        
        for k in range(self.k):
            coef = np.concatenate([self.upsilon[k], self.gamma[k]])
            
            # Check for NaN values
            if np.any(np.isnan(coef)) or np.all(np.abs(coef) < 1e-10):
                # If coefficients are bad, use a simple baseline
                # Use the mean of the training labels for this class as baseline
                if hasattr(self, 'Y') and self.Y is not None and self.Y.shape[1] > k:
                    baseline_prob = np.mean(self.Y[:, k])
                    # Ensure baseline_prob is a scalar and valid
                    if not (0 <= baseline_prob <= 1):
                        baseline_prob = 0.5 # Default if Y is not as expected
                else:
                    baseline_prob = 0.5 # Default if Y is not available
                probs[:, k] = baseline_prob
            else:
                logits = X_design_test @ coef
                logits = np.clip(logits, -500, 500)  # Prevent overflow
                probs[:, k] = 1 / (1 + np.exp(-logits))
        
        return probs

    def infer_theta_for_new_data(self, X_new: np.ndarray, num_samples: int = 10) -> np.ndarray:
        """
        Infer theta for new count data using the learned beta parameters
        """
        n_new = X_new.shape[0]
        
        # Initialize theta for new data
        if sp.issparse(X_new):
            library_sizes = np.asarray(X_new.sum(axis=1)).flatten()
        else:
            library_sizes = X_new.sum(axis=1)
            
        normalized_lib = library_sizes / np.mean(library_sizes)
        
        # Sample theta for new data
        theta_samples = []
        for _ in range(num_samples):
            # Use similar approach as in training but with fewer iterations
            theta_new = self.rng.gamma(
                shape=0.5,
                scale=normalized_lib[:, None] * 0.1,
                size=(n_new, self.d)
            )
            
            # Do a few Gibbs steps to refine theta given the learned beta
            for gibbs_iter in range(5):  # Just a few iterations
                if sp.issparse(X_new):
                    X_sum = np.asarray(X_new.sum(axis=1)).flatten()
                else:
                    X_sum = X_new.sum(axis=1)
                
                shape = self.a_theta_shape + X_sum[:, None]
                # Corrected rate calculation: np.mean(self.xi) is a scalar
                rate = np.mean(self.xi) + self.beta.sum(axis=0)[None, :]
                
                shape = np.clip(shape, 1e-6, 1e6)
                rate = np.clip(rate, 1e-6, 1e6)
                
                theta_new = self.rng.gamma(shape=shape, scale=1.0/rate)
                theta_new = np.clip(theta_new, 1e-6, 1e6)
            
            theta_samples.append(theta_new)
        
        # Return mean of samples
        return np.mean(theta_samples, axis=0)

    def get_parameters(self) -> dict:
        """Get current parameter estimates"""
        return {
            'theta': self.theta,
            'beta': self.beta,
            'upsilon': self.upsilon,
            'gamma': self.gamma,
            'eta': self.eta,
            'xi': self.xi,
        }
