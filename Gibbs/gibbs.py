import numpy as np
from numpy.random import default_rng, Generator
from scipy.special import expit, gammaln, logsumexp, log_expit
from scipy.optimize import minimize
from scipy.stats import norm as norm_dist
from scipy.stats import gamma as gamma_dist

# JAX for vectorised and JIT compiled operations
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random

# Configure JAX for better memory management
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Limit JAX memory usage
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate all memory

# joblib for easy parallelisation across independent updates
from joblib import Parallel, delayed

def spike_slab_logpdf(w, pi, sigma_slab, spike_scale=1e-3):
    # log [ pi * N(w|0,sigma_slab^2) + (1-pi) * N(w|0,spike_scale^2) ]
    # For numerical stability, use logsumexp
    log_slab = norm_dist.logpdf(w, 0, sigma_slab)
    log_spike = norm_dist.logpdf(w, 0, spike_scale)
    return np.logaddexp(np.log(pi) + log_slab, np.log(1 - pi) + log_spike)

def spike_slab_rvs(rng, pi, sigma_slab, size, spike_scale=1e-3):
    # Draw from spike-and-slab: with prob pi from slab, else from spike
    s = rng.binomial(1, pi, size=size)
    w = np.where(
        s == 1,
        rng.normal(0.0, sigma_slab, size=size),
        rng.normal(0.0, spike_scale, size=size)
    )
    return w

def _optimise_theta_worker(i: int, theta_i: np.ndarray, xi_i: float,
                           z_i: np.ndarray, beta: np.ndarray,
                           gamma: np.ndarray, upsilon: np.ndarray,
                           X_aux_i: np.ndarray, Y_i: np.ndarray,
                           alpha_theta: float) -> np.ndarray:
    """Standalone worker for parallel theta updates."""

    def log_likelihood_bernoulli(y: np.ndarray, logits: np.ndarray) -> float:
        log_p1 = log_expit(logits)
        log_p0 = log_expit(-logits)
        return np.sum(y * log_p1 + (1 - y) * log_p0)

    def log_posterior_theta_i(theta_i_vec: np.ndarray) -> float:
        log_prior_gamma = np.sum(
            gamma_dist.logpdf(theta_i_vec, a=alpha_theta, scale=1.0 / xi_i)
        )
        with np.errstate(divide="ignore"):
            log_lik_poisson = np.sum(
                z_i * np.log(beta) - beta * theta_i_vec[:, np.newaxis].T
            )
        logits = X_aux_i @ gamma.T + theta_i_vec @ upsilon.T
        log_lik_logistic = log_likelihood_bernoulli(Y_i, logits)
        return log_prior_gamma + log_lik_poisson + log_lik_logistic

    def objective_func(theta_i_vec: np.ndarray) -> float:
        if np.any(theta_i_vec <= 0):
            return np.inf
        return -log_posterior_theta_i(theta_i_vec)

    result = minimize(fun=objective_func, x0=theta_i, method="Nelder-Mead")
    return result.x if result.success else theta_i


def _optimise_gamma_worker(k: int, gamma_k: np.ndarray, theta: np.ndarray,
                           upsilon_k: np.ndarray, X_aux: np.ndarray,
                           Y_k: np.ndarray, sigma_gamma_sq: float) -> np.ndarray:
    """Standalone worker for parallel gamma updates."""

    def log_likelihood_bernoulli(y: np.ndarray, logits: np.ndarray) -> float:
        log_p1 = log_expit(logits)
        log_p0 = log_expit(-logits)
        return np.sum(y * log_p1 + (1 - y) * log_p0)

    def log_posterior_gamma_k(gamma_k_vec: np.ndarray) -> float:
        log_prior = norm_dist.logpdf(gamma_k_vec, 0,
                                     np.sqrt(sigma_gamma_sq)).sum()
        logits = X_aux @ gamma_k_vec + theta @ upsilon_k
        log_lik = log_likelihood_bernoulli(Y_k, logits)
        return log_prior + log_lik

    def objective_func(gamma_k_vec: np.ndarray) -> float:
        return -log_posterior_gamma_k(gamma_k_vec)

    result = minimize(fun=objective_func, x0=gamma_k, method="Nelder-Mead")
    return result.x if result.success else gamma_k


def _optimise_w_upsilon_worker(k: int, w_upsilon_k: np.ndarray, s_upsilon_k: np.ndarray,
                               theta: np.ndarray, gamma_k: np.ndarray,
                               X_aux: np.ndarray, Y_k: np.ndarray,
                               sigma_slab_sq: float, pi_upsilon: float = 0.5, spike_scale: float = 1e-3) -> np.ndarray:
    """Standalone worker for parallel w_upsilon updates with spike-and-slab prior."""

    def log_likelihood_bernoulli(y: np.ndarray, logits: np.ndarray) -> float:
        log_p1 = log_expit(logits)
        log_p0 = log_expit(-logits)
        return np.sum(y * log_p1 + (1 - y) * log_p0)

    def log_posterior_w_k(w_vec: np.ndarray) -> float:
        # Use spike-and-slab prior for each w
        log_prior = np.sum(spike_slab_logpdf(w_vec, pi_upsilon, np.sqrt(sigma_slab_sq), spike_scale=spike_scale))
        upsilon_k_vec = s_upsilon_k * w_vec
        logits = X_aux @ gamma_k + theta @ upsilon_k_vec
        log_lik = log_likelihood_bernoulli(Y_k, logits)
        return log_prior + log_lik

    if np.any(s_upsilon_k == 1):
        def objective_func(w_vec: np.ndarray) -> float:
            return -log_posterior_w_k(w_vec)

        result = minimize(fun=objective_func, x0=w_upsilon_k, method="BFGS")
        return result.x if result.success else w_upsilon_k
    else:
        rng = default_rng()
        # Draw from spike-and-slab prior: sharp spike at 0, otherwise slab
        return spike_slab_rvs(rng, pi_upsilon, np.sqrt(sigma_slab_sq), size=s_upsilon_k.shape[0], spike_scale=spike_scale)

class SpikeSlabGibbsSampler:

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_aux: np.ndarray,
        n_programs: int,
        *,
        alpha_theta: float = 0.3,
        alpha_beta: float = 0.3,
        alpha_xi: float = 0.3,
        alpha_eta: float = 0.3,
        lambda_xi: float = 0.3,
        lambda_eta: float = 0.3,
        pi_upsilon: float = 0.5,
        sigma_slab_sq: float = 1.0,
        sigma_gamma_sq: float = 1.0,
        spike_scale: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        self.rng: Generator = default_rng(seed)
        self.jax_key = random.PRNGKey(seed if seed is not None else 0)
        self.X = np.asarray(X, dtype=np.int32)
        self.Y = np.asarray(Y, dtype=np.int32)
        self.X_aux = np.asarray(X_aux)
        self.n, self.p = self.X.shape
        self.k = self.Y.shape[1]
        self.d = n_programs
        self.q = self.X_aux.shape[1]
        self.alpha_theta = alpha_theta
        self.alpha_beta = alpha_beta
        self.alpha_xi = alpha_xi
        self.alpha_eta = alpha_eta
        self.lambda_xi = lambda_xi
        self.lambda_eta = lambda_eta
        self.pi_upsilon = pi_upsilon
        self.sigma_slab_sq = sigma_slab_sq
        self.sigma_gamma_sq = sigma_gamma_sq
        self.spike_scale = spike_scale
        self._init_params()

    # ------------------------------------------------------------------
    def _init_params(self) -> None:
        print("[Init] Initialising latent variables and parameters...")
        self.xi = self.rng.gamma(self.alpha_xi, scale=1.0/self.lambda_xi, size=self.n)
        self.eta = self.rng.gamma(self.alpha_eta, scale=1.0/self.lambda_eta, size=self.p)
        self.theta = np.zeros((self.n, self.d))
        for i in range(self.n):
            self.theta[i, :] = self.rng.gamma(self.alpha_theta, scale=1.0/self.xi[i], size=self.d)
        self.beta = np.zeros((self.p, self.d))
        for j in range(self.p):
            self.beta[j, :] = self.rng.gamma(self.alpha_beta, scale=1.0/self.eta[j], size=self.d)
        self.gamma = self.rng.normal(0.0, np.sqrt(self.sigma_gamma_sq), size=(self.k, self.q))
        self.s_upsilon = self.rng.binomial(1, self.pi_upsilon, size=(self.k, self.d))
        self.w_upsilon = spike_slab_rvs(self.rng, self.pi_upsilon, np.sqrt(self.sigma_slab_sq), size=(self.k, self.d), spike_scale=self.spike_scale)
        self.upsilon = self.s_upsilon * self.w_upsilon
        self.z = np.zeros((self.n, self.p, self.d), dtype=np.int32)
        self._update_latent_counts_z()
        print(f"[Init] xi mean: {np.mean(self.xi):.4f}, std: {np.std(self.xi):.4f}")
        print(f"[Init] eta mean: {np.mean(self.eta):.4f}, std: {np.std(self.eta):.4f}")
        print(f"[Init] theta mean: {np.mean(self.theta):.4f}, std: {np.std(self.theta):.4f}")
        print(f"[Init] beta mean: {np.mean(self.beta):.4f}, std: {np.std(self.beta):.4f}")
        print(f"[Init] gamma mean: {np.mean(self.gamma):.4f}, std: {np.std(self.gamma):.4f}")
        print(f"[Init] s_upsilon mean: {np.mean(self.s_upsilon):.4f}")
        print(f"[Init] w_upsilon mean: {np.mean(self.w_upsilon):.4f}, std: {np.std(self.w_upsilon):.4f}")
        print("[Init] Initialization complete.\n")

    def _update_latent_counts_z(self) -> None:
        print("[Update] Sampling latent counts z ...")
        with np.errstate(divide="ignore"):
            log_theta = jnp.log(jnp.asarray(self.theta))
            log_beta = jnp.log(jnp.asarray(self.beta))
        log_rates = log_theta[:, None, :] + log_beta[None, :, :]
        log_probs = log_rates - jsp.special.logsumexp(log_rates, axis=2, keepdims=True)
        probs = jnp.exp(log_probs)
        flat_probs = probs.reshape(-1, self.d)
        flat_counts = jnp.asarray(self.X.reshape(-1), dtype=jnp.int32)
        keys = random.split(self.jax_key, flat_probs.shape[0] + 1)
        self.jax_key = keys[0]
        sample_fun = lambda k, n, p: random.multinomial(k, n=n, p=p)
        samples = jax.vmap(sample_fun)(keys[1:], flat_counts, flat_probs)
        self.z = np.array(samples.reshape(self.n, self.p, self.d), dtype=np.int32)
        print(f"[Update] z mean: {np.mean(self.z):.4f}, std: {np.std(self.z):.4f}")
        print("[Update] Latent counts z update complete.\n")

    # Closed form updates for beta, xi, eta
    def _update_beta(self) -> None:
        print("[Update] Updating beta ...")
        z_sum_over_i = np.sum(self.z, axis=0)
        shape = self.alpha_beta + z_sum_over_i
        rate = self.eta[:, np.newaxis] + np.sum(self.theta, axis=0)
        self.beta = self.rng.gamma(shape, scale=1.0 / rate)
        print(f"[Update] beta mean: {np.mean(self.beta):.4f}, std: {np.std(self.beta):.4f}")
        print("[Update] Beta update complete.\n")

    def _update_eta(self) -> None:
        print("[Update] Updating eta ...")
        shape = self.alpha_eta + self.d * self.alpha_beta
        rate = self.lambda_eta + np.sum(self.beta, axis=1)
        self.eta = self.rng.gamma(shape, scale=1.0 / rate)
        print(f"[Update] eta mean: {np.mean(self.eta):.4f}, std: {np.std(self.eta):.4f}")
        print("[Update] Eta update complete.\n")

    def _update_xi(self) -> None:
        print("[Update] Updating xi ...")
        shape = self.alpha_xi + self.d * self.alpha_theta
        rate = self.lambda_xi + np.sum(self.theta, axis=1)
        self.xi = self.rng.gamma(shape, scale=1.0 / rate)
        print(f"[Update] xi mean: {np.mean(self.xi):.4f}, std: {np.std(self.xi):.4f}")
        print("[Update] Xi update complete.\n")

    @staticmethod
    def _log_likelihood_bernoulli(y: np.ndarray, logits: np.ndarray) -> float:
        """Numerically stable Bernoulli log-likelihood using log_expit."""
        log_p1 = log_expit(logits)  # log(sigmoid(logits))
        log_p0 = log_expit(-logits) # log(1 - sigmoid(logits))
        return np.sum(y * log_p1 + (1 - y) * log_p0)

    # Non-conjugate updates for theta, gamma, upsilon via Laplace Approximation

    def _log_posterior_theta_i(self, i: int, theta_i: np.ndarray) -> float:
        """Calculate log posterior for a single theta_i vector."""
        log_prior_gamma = np.sum(gamma_dist.logpdf(theta_i, a=self.alpha_theta, scale=1.0/self.xi[i]))
        with np.errstate(divide='ignore'):
            log_lik_poisson = np.sum(self.z[i, :, :] * np.log(self.beta) - self.beta * theta_i[:, np.newaxis].T)
        logits = self.X_aux[i] @ self.gamma.T + theta_i @ self.upsilon.T
        log_lik_logistic = self._log_likelihood_bernoulli(self.Y[i, :], logits)
        return log_prior_gamma + log_lik_poisson + log_lik_logistic

    
    def _update_theta(self) -> None:
        print("[Update] Updating theta ...")
        tasks = [
            delayed(_optimise_theta_worker)(
                i,
                self.theta[i],
                self.xi[i],
                self.z[i, :, :],
                self.beta,
                self.gamma,
                self.upsilon,
                self.X_aux[i],
                self.Y[i],
                self.alpha_theta,
            )
            for i in range(self.n)
        ]
        results = Parallel(n_jobs=-1)(tasks)
        self.theta = np.asarray(results)
        print(f"[Update] theta mean: {np.mean(self.theta):.4f}, std: {np.std(self.theta):.4f}")
        print("[Update] Theta update complete.\n")

    def _log_posterior_gamma_k(self, k: int, gamma_k: np.ndarray) -> float:
        """Calculate log posterior for a single gamma_k vector."""
        log_prior = norm_dist.logpdf(gamma_k, 0, np.sqrt(self.sigma_gamma_sq)).sum()
        logits = self.X_aux @ gamma_k + self.theta @ self.upsilon[k]
        log_lik = self._log_likelihood_bernoulli(self.Y[:, k], logits)
        return log_prior + log_lik
                
    def _update_gamma(self) -> None:
        print("[Update] Updating gamma ...")
        tasks = [
            delayed(_optimise_gamma_worker)(
                k,
                self.gamma[k],
                self.theta,
                self.upsilon[k],
                self.X_aux,
                self.Y[:, k],
                self.sigma_gamma_sq,
            )
            for k in range(self.k)
        ]
        results = Parallel(n_jobs=-1)(tasks)
        self.gamma = np.asarray(results)
        print(f"[Update] gamma mean: {np.mean(self.gamma):.4f}, std: {np.std(self.gamma):.4f}")
        print("[Update] Gamma update complete.\n")

    def _update_s_upsilon(self) -> None:
        print("[Update] Updating s_upsilon ...")
        for k in range(self.k):
            logits_0 = self.X_aux @ self.gamma[k]  # Logits when upsilon_k=0
            logits_1 = logits_0 + self.theta @ self.w_upsilon[k]  # Logits when upsilon_k=w_k
            log_post_1 = np.log(self.pi_upsilon) + self._log_likelihood_bernoulli(self.Y[:, k], logits_1)
            log_post_0 = np.log(1 - self.pi_upsilon) + self._log_likelihood_bernoulli(self.Y[:, k], logits_0)
            prob_s1 = expit(log_post_1 - log_post_0)
            self.s_upsilon[k] = self.rng.binomial(1, prob_s1, size=self.d)
        print(f"[Update] s_upsilon mean: {np.mean(self.s_upsilon):.4f}")
        print("[Update] s_upsilon update complete.\n")

    def _log_posterior_w_upsilon_k(self, k: int, w_upsilon_k: np.ndarray) -> float:
        """Calculate log posterior for a single w_upsilon_k vector."""
        # Use spike-and-slab prior for each w
        log_prior = np.sum(spike_slab_logpdf(w_upsilon_k, self.pi_upsilon, np.sqrt(self.sigma_slab_sq), spike_scale=self.spike_scale))
        upsilon_k = self.s_upsilon[k] * w_upsilon_k
        logits = self.X_aux @ self.gamma[k] + self.theta @ upsilon_k
        log_lik = self._log_likelihood_bernoulli(self.Y[:, k], logits)
        return log_prior + log_lik

    def _update_w_upsilon(self) -> None:
        print("[Update] Updating w_upsilon ...")
        tasks = [
            delayed(_optimise_w_upsilon_worker)(
                k,
                self.w_upsilon[k],
                self.s_upsilon[k],
                self.theta,
                self.gamma[k],
                self.X_aux,
                self.Y[:, k],
                self.sigma_slab_sq,
                self.pi_upsilon,
                self.spike_scale,
            )
            for k in range(self.k)
        ]
        results = Parallel(n_jobs=-1)(tasks)
        self.w_upsilon = np.asarray(results)
        print(f"[Update] w_upsilon mean: {np.mean(self.w_upsilon):.4f}, std: {np.std(self.w_upsilon):.4f}")
        print("[Update] w_upsilon update complete.\n")

    def clear_jax_cache(self) -> None:
        """Clear JAX compilation cache to free memory."""
        try:
            # Clear JAX compilation cache
            jax.clear_caches()
        except:
            pass

    def cleanup_memory(self) -> None:
        """Force garbage collection and JAX cache clearing."""
        import gc
        self.clear_jax_cache()
        gc.collect()