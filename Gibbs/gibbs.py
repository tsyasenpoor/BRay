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

# joblib for easy parallelisation across independent updates
from joblib import Parallel, delayed

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
        self._init_params()

    # ------------------------------------------------------------------
    def _init_params(self) -> None:
        """Initialise latent variables and parameters."""
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
        self.w_upsilon = self.rng.normal(0.0, np.sqrt(self.sigma_slab_sq), size=(self.k, self.d))
        self.upsilon = self.s_upsilon * self.w_upsilon
        self.z = np.zeros((self.n, self.p, self.d), dtype=np.int32)
        self._update_latent_counts_z()

    def _update_latent_counts_z(self) -> None:
        """Sample latent counts ``z`` using JAX for vectorisation."""
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

    # Closed form updates for beta, xi, eta
    def _update_beta(self) -> None:
        """Update beta_jl. This calculation does not require log-space."""
        z_sum_over_i = np.sum(self.z, axis=0)
        shape = self.alpha_beta + z_sum_over_i
        rate = self.eta[:, np.newaxis] + np.sum(self.theta, axis=0)
        self.beta = self.rng.gamma(shape, scale=1.0 / rate)

    def _update_eta(self) -> None:
        """Update eta_j. This calculation does not require log-space."""
        shape = self.alpha_eta + self.d * self.alpha_beta
        rate = self.lambda_eta + np.sum(self.beta, axis=1)
        self.eta = self.rng.gamma(shape, scale=1.0 / rate)

    def _update_xi(self) -> None:
        """Update xi_i. This calculation does not require log-space."""
        shape = self.alpha_xi + self.d * self.alpha_theta
        rate = self.lambda_xi + np.sum(self.theta, axis=1)
        self.xi = self.rng.gamma(shape, scale=1.0 / rate)

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
        """Update theta using Laplace Approximation in parallel."""

        def optimise_single(i: int) -> np.ndarray:
            def objective_func(theta_i):
                if np.any(theta_i <= 0):
                    return np.inf
                return -self._log_posterior_theta_i(i, theta_i)

            result = minimize(
                fun=objective_func,
                x0=self.theta[i],
                method="Nelder-Mead",
            )
            return result.x if result.success else self.theta[i]

        results = Parallel(n_jobs=-1)(delayed(optimise_single)(i) for i in range(self.n))
        self.theta = np.asarray(results)

    def _log_posterior_gamma_k(self, k: int, gamma_k: np.ndarray) -> float:
        """Calculate log posterior for a single gamma_k vector."""
        log_prior = norm_dist.logpdf(gamma_k, 0, np.sqrt(self.sigma_gamma_sq)).sum()
        logits = self.X_aux @ gamma_k + self.theta @ self.upsilon[k]
        log_lik = self._log_likelihood_bernoulli(self.Y[:, k], logits)
        return log_prior + log_lik
                
    def _update_gamma(self) -> None:
        """Update gamma using Laplace Approximation in parallel."""

        def optimise_single(k: int) -> np.ndarray:
            def objective_func(gamma_k):
                return -self._log_posterior_gamma_k(k, gamma_k)

            result = minimize(fun=objective_func, x0=self.gamma[k], method="BFGS")
            return result.x if result.success else self.gamma[k]

        results = Parallel(n_jobs=-1)(delayed(optimise_single)(k) for k in range(self.k))
        self.gamma = np.asarray(results)

    def _update_s_upsilon(self) -> None:
        for k in range(self.k):
            logits_0 = self.X_aux @ self.gamma[k] # Logits when upsilon_k=0
            logits_1 = logits_0 + self.theta @ self.w_upsilon[k] # Logits when upsilon_k=w_k
            log_post_1 = np.log(self.pi_upsilon) + self._log_likelihood_bernoulli(self.Y[:, k], logits_1)
            log_post_0 = np.log(1 - self.pi_upsilon) + self._log_likelihood_bernoulli(self.Y[:, k], logits_0)
            
            # Stable calculation of P(s=1) = sigmoid(log_post_1 - log_post_0)
            prob_s1 = expit(log_post_1 - log_post_0)
            self.s_upsilon[k] = self.rng.binomial(1, prob_s1, size=self.d)

    def _log_posterior_w_upsilon_k(self, k: int, w_upsilon_k: np.ndarray) -> float:
        """Calculate log posterior for a single w_upsilon_k vector."""
        log_prior = norm_dist.logpdf(w_upsilon_k, 0, np.sqrt(self.sigma_slab_sq)).sum()
        upsilon_k = self.s_upsilon[k] * w_upsilon_k
        logits = self.X_aux @ self.gamma[k] + self.theta @ upsilon_k
        log_lik = self._log_likelihood_bernoulli(self.Y[:, k], logits)
        return log_prior + log_lik

    def _update_w_upsilon(self) -> None:
        def optimise_single(k: int) -> np.ndarray:
            if np.any(self.s_upsilon[k] == 1):
                def objective_func(w_upsilon_k):
                    return -self._log_posterior_w_upsilon_k(k, w_upsilon_k)

                result = minimize(fun=objective_func, x0=self.w_upsilon[k], method="BFGS")
                return result.x if result.success else self.w_upsilon[k]
            else:
                return self.rng.normal(0.0, np.sqrt(self.sigma_slab_sq), size=self.d)

        results = Parallel(n_jobs=-1)(delayed(optimise_single)(k) for k in range(self.k))
        self.w_upsilon = np.asarray(results)

    
    