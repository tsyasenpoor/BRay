import numpy as np
from numpy.random import default_rng
from scipy.special import expit
import gc
import warnings
from typing import Optional, Union

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False

# Add memory tracking
try:
    import psutil
    MEMORY_TRACKING = True
except ImportError:
    MEMORY_TRACKING = False

def get_memory_usage_mb():
    """Get memory usage in MB."""
    if MEMORY_TRACKING:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # MB
    return 0.0

def log_memory(message=""):
    if MEMORY_TRACKING:
        print(f"MEMORY [{message}]: {get_memory_usage_mb():.1f} MB")

def _compute_rhat(chains: np.ndarray) -> np.ndarray:
    """Compute Gelman-Rubin R-hat statistic.

    Parameters
    ----------
    chains : np.ndarray
        Array of shape (n_chains, n_samples, ...).

    Returns
    -------
    np.ndarray
        R-hat values with shape matching ``chains`` without the first two axes.
    """
    m, n = chains.shape[0], chains.shape[1]
    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean(axis=0)
    B = n * np.sum((chain_means - grand_mean) ** 2, axis=0) / (m - 1)
    W = chains.var(axis=1, ddof=1).mean(axis=0)
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    return np.sqrt(var_hat / (W + 1e-12))


def _effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Estimate effective sample size using autocorrelation."""
    m, n = chains.shape[0], chains.shape[1]
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)

    var_within = chain_vars.mean(axis=0)
    acov_sum = np.zeros_like(var_within)
    for t in range(1, n):
        acov_t = ((chains[:, :-t] - chain_means[:, None]) * (chains[:, t:] - chain_means[:, None])).mean(axis=(0,1))
        rho_t = acov_t / (var_within + 1e-12)
        if np.all(rho_t < 0):
            break
        acov_sum += rho_t
    tau = 1 + 2 * acov_sum
    ess = m * n / tau
    return ess


# Place MemoryEfficientTrace definition above its first use for type checking
class MemoryEfficientTrace:
    """Memory-efficient trace storage that only keeps recent samples."""
    
    def __init__(self, max_samples=1000, dtype=np.float32):
        self.max_samples = max_samples
        self.dtype = dtype
        self.samples = []
        self.total_samples = 0
        
    def append(self, sample):
        """Add a new sample, potentially removing old ones."""
        # Convert to efficient dtype
        if hasattr(sample, 'astype'):
            sample = sample.astype(self.dtype)
        
        self.samples.append(sample)
        self.total_samples += 1
        
        # Keep only recent samples
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)
    
    def get_array(self):
        """Get all stored samples as array."""
        if not self.samples:
            return np.array([])
        return np.array(self.samples, dtype=self.dtype)
    
    def get_recent(self, n_samples):
        """Get the most recent n_samples."""
        if n_samples >= len(self.samples):
            return self.get_array()
        return np.array(self.samples[-n_samples:], dtype=self.dtype)


class SpikeSlabGibbsSampler:
    """Memory-efficient Gibbs sampler using a spike-and-slab prior for ``upsilon`` coefficients.

    This implementation was originally written for experiments without any gene
    program masking.  To support the different experiment configurations used in
    ``run_experiments.py`` we optionally allow a ``mask`` matrix specifying
    which gene--program combinations are active.  If a mask is provided the
    corresponding ``beta`` entries are fixed to zero throughout sampling.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_aux: np.ndarray,
        n_programs: int,
        *,
        a: float = 0.3,
        a_prime: float = 0.3,
        b_prime: float = 0.3,
        c: float = 0.3,
        c_prime: float = 0.3,
        d_prime: float = 0.3,
        tau1_sq: float = 1.0,
        tau0_sq: float = 1e-6,
        pi: float = 0.05,
        pi_alpha: float = 1.0,
        pi_beta: float = 10.0,
        sigma_gamma_sq: float = 1.0,
        seed: Optional[int] = None,
        mask: Optional[np.ndarray] = None,
        use_jax: bool = False,
        memory_efficient: bool = True,
        trace_max_samples: int = 1000,
        use_float32: bool = True,
    ) -> None:
        self.use_jax = bool(use_jax and JAX_AVAILABLE)
        self.memory_efficient = memory_efficient
        self.trace_max_samples = trace_max_samples
        self.use_float32 = use_float32
        
        # Choose dtype based on memory efficiency setting
        self.dtype = np.float32 if use_float32 else np.float64
        
        self.rng = default_rng(seed)
        if self.use_jax:
            self.key = jrandom.PRNGKey(seed or 0)

        # Convert inputs to efficient dtype and handle sparse matrices
        self.X = self._prepare_input(X)
        self.X_aux = self._prepare_input(X_aux)
        self.Y = self._prepare_input(Y)

        self.n, self.p = self.X.shape
        self.q = self.X_aux.shape[1]
        self.k = self.Y.shape[1]
        self.d = n_programs

        self.mask = None
        if mask is not None:
            m = np.asarray(mask, dtype=self.dtype)
            if m.shape != (self.p, self.d):
                raise ValueError(
                    "mask must have shape (p, d) matching (genes, programs)"
                )
            self.mask = m.astype(int)

        # Hyperparameters
        self.a = a
        self.a_prime = a_prime
        self.b_prime = b_prime
        self.c = c
        self.c_prime = c_prime
        self.d_prime = d_prime
        self.tau1_sq = tau1_sq
        self.tau0_sq = tau0_sq
        self.pi_alpha = pi_alpha
        self.pi_beta = pi_beta
        self.pi = self._rng_beta(pi_alpha, pi_beta)
        self.sigma_gamma_sq = sigma_gamma_sq

        self._init_params()
        log_memory("After initialization")

    def _prepare_input(self, arr):
        """Prepare input array with memory-efficient settings."""
        if hasattr(arr, 'toarray'):  # Sparse matrix
            # Keep sparse if memory efficient, otherwise convert
            if self.memory_efficient:
                return arr
            else:
                return arr.toarray().astype(self.dtype)
        else:
            return np.asarray(arr, dtype=self.dtype)

    # ------------------------------------------------------------------
    def _rng_gamma(self, shape, rate):
        if self.use_jax:
            self.key, subkey = jrandom.split(self.key)
            return jrandom.gamma(subkey, shape, shape=shape.shape) / rate
        return self.rng.gamma(shape, 1.0 / rate)

    def _rng_normal(self, mean, std, size):
        if self.use_jax:
            self.key, subkey = jrandom.split(self.key)
            return mean + std * jrandom.normal(subkey, shape=size)
        return self.rng.normal(mean, std, size=size)

    def _rng_binomial(self, n, p, size=None):
        if self.use_jax:
            self.key, subkey = jrandom.split(self.key)
            return jrandom.binomial(subkey, n, p, shape=size)
        return self.rng.binomial(n, p, size=size)

    def _rng_uniform(self, size=None):
        if self.use_jax:
            self.key, subkey = jrandom.split(self.key)
            return jrandom.uniform(subkey, shape=size)
        return self.rng.random(size=size)

    def _rng_beta(self, a, b):
        if self.use_jax:
            self.key, subkey = jrandom.split(self.key)
            return jrandom.beta(subkey, a, b)
        return self.rng.beta(a, b)

    def _rng_multivariate_normal(self, mean, cov, size=None):
        if self.use_jax:
            self.key, subkey = jrandom.split(self.key)
            return jrandom.multivariate_normal(subkey, mean, cov, shape=size)
        return self.rng.multivariate_normal(mean, cov, size=size)

    # ------------------------------------------------------------------
    def _init_params(self) -> None:
        """Initialise latent variables and parameters."""
        self.log_theta = np.log(
            self._rng_gamma(np.ones((self.n, self.d)), np.ones((self.n, self.d))) + 1e-12
        ).astype(self.dtype)
        self.log_beta = np.log(
            self._rng_gamma(np.ones((self.p, self.d)), np.ones((self.p, self.d))) + 1e-12
        ).astype(self.dtype)
        if self.mask is not None:
            self.log_beta = np.where(self.mask, self.log_beta, -np.inf)
        self.log_xi = np.log(self._rng_gamma(np.ones(self.n), np.ones(self.n)) + 1e-12).astype(self.dtype)
        self.log_eta = np.log(self._rng_gamma(np.ones(self.p), np.ones(self.p)) + 1e-12).astype(self.dtype)

        self.gamma = self._rng_normal(0.0, np.sqrt(self.sigma_gamma_sq), size=(self.k, self.q)).astype(self.dtype)

        self.delta = self._rng_binomial(1, self.pi, size=(self.k, self.d))
        self.upsilon = self._rng_normal(0.0, 0.1, size=(self.k, self.d)).astype(self.dtype)

        if self.use_jax:
            self.log_theta = jnp.asarray(self.log_theta)
            self.log_beta = jnp.asarray(self.log_beta)
            self.log_xi = jnp.asarray(self.log_xi)
            self.log_eta = jnp.asarray(self.log_eta)
            self.gamma = jnp.asarray(self.gamma)
            self.delta = jnp.asarray(self.delta)
            self.upsilon = jnp.asarray(self.upsilon)

    # ------------------------------------------------------------------
    @property
    def theta(self) -> np.ndarray:
        return np.exp(self.log_theta)

    @property
    def beta(self) -> np.ndarray:
        return np.exp(self.log_beta)

    @property
    def xi_val(self) -> np.ndarray:
        return np.exp(self.log_xi)

    @property
    def eta_val(self) -> np.ndarray:
        return np.exp(self.log_eta)

    # ------------------------------------------------------------------
    # Memory-efficient conjugate updates for gamma-distributed parameters
    def _update_theta(self) -> None:
        exp_beta = np.exp(self.log_beta)
        rate = np.exp(self.log_xi)[:, None] + exp_beta.sum(axis=0)[None, :]
        
        # Use sparse matrix multiplication if available
        if hasattr(self.X, 'dot'):
            shape = self.a + self.X.dot(exp_beta)
        else:
            shape = self.a + self.X @ exp_beta
            
        theta_new = self._rng_gamma(shape, rate)
        self.log_theta = np.log(theta_new + 1e-12).astype(self.dtype)

    def _update_beta(self) -> None:
        exp_theta = np.exp(self.log_theta)
        rate = np.exp(self.log_eta)[:, None] + exp_theta.sum(axis=0)[None, :]
        
        # Use sparse matrix multiplication if available
        if hasattr(self.X, 'T') and hasattr(self.X.T, 'dot'):
            shape = self.c + self.X.T.dot(exp_theta)
        else:
            shape = self.c + self.X.T @ exp_theta
            
        beta_new = self._rng_gamma(shape, rate)
        if self.mask is not None:
            beta_new = np.where(self.mask, beta_new, 0.0)
        self.log_beta = np.log(beta_new + 1e-12).astype(self.dtype)
        if self.mask is not None:
            self.log_beta = np.where(self.mask, self.log_beta, -np.inf)

    def _update_xi(self) -> None:
        exp_theta = np.exp(self.log_theta)
        rate = self.b_prime + exp_theta.sum(axis=1)
        shape = self.a_prime + self.a * self.d
        xi_new = self._rng_gamma(np.full(self.n, shape), rate)
        self.log_xi = np.log(xi_new + 1e-12).astype(self.dtype)

    def _update_eta(self) -> None:
        exp_beta = np.exp(self.log_beta)
        rate = self.d_prime + exp_beta.sum(axis=1)
        shape = self.c_prime + self.c * self.d
        eta_new = self._rng_gamma(np.full(self.p, shape), rate)
        self.log_eta = np.log(eta_new + 1e-12).astype(self.dtype)

    # ------------------------------------------------------------------
    def _update_gamma(self) -> None:
        XTX = self.X_aux.T @ self.X_aux
        precision = XTX / self.sigma_gamma_sq + np.eye(self.q)
        cov = np.linalg.inv(precision)
        logits = np.exp(self.log_theta) @ self.upsilon.T + self.X_aux @ self.gamma.T
        z = self.Y - expit(logits)
        mean = (cov @ (self.X_aux.T @ z) / self.sigma_gamma_sq).T
        gamma_new = self._rng_multivariate_normal(
            np.zeros(self.q), cov, size=self.k
        ) + mean
        self.gamma = gamma_new.astype(self.dtype)

    # ------------------------------------------------------------------
    @staticmethod
    def _log_likelihood(logits: np.ndarray, y: np.ndarray) -> float:
        """Stable Bernoulli log likelihood under the logit parameterisation."""
        return float(
            np.sum(
                y * logits
                - np.maximum(logits, 0)
                - np.log1p(np.exp(-np.abs(logits)))
            )
        )

    @staticmethod
    def _log_likelihood_vec(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorised log likelihood over multiple outcomes."""
        return (
            y * logits
            - np.maximum(logits, 0)
            - np.log1p(np.exp(-np.abs(logits)))
        ).sum(axis=0)

    def _log_posterior_upsilon(self, k: int, upsilon_k: np.ndarray) -> float:
        logits = np.exp(self.log_theta) @ upsilon_k + self.X_aux @ self.gamma[k]
        log_lik = self._log_likelihood(logits, self.Y[:, k])
        prior_var = np.where(self.delta[k] == 1, self.tau1_sq, self.tau0_sq)
        log_prior = -0.5 * np.sum(upsilon_k**2 / prior_var)
        return log_lik + log_prior

    def _log_posterior_upsilon_all(self, upsilon_mat: np.ndarray) -> np.ndarray:
        logits = np.exp(self.log_theta) @ upsilon_mat.T + self.X_aux @ self.gamma.T
        log_lik = self._log_likelihood_vec(logits, self.Y)
        prior_var = np.where(self.delta == 1, self.tau1_sq, self.tau0_sq)
        log_prior = -0.5 * np.sum(upsilon_mat**2 / prior_var, axis=1)
        return log_lik + log_prior

    def _update_delta(self) -> None:
        v = self.upsilon
        log_p1 = (
            np.log(self.pi)
            - 0.5 * v * v / self.tau1_sq
            - 0.5 * np.log(2 * np.pi * self.tau1_sq)
        )
        log_p0 = (
            np.log(1 - self.pi)
            - 0.5 * v * v / self.tau0_sq
            - 0.5 * np.log(2 * np.pi * self.tau0_sq)
        )
        prob = 1.0 / (1.0 + np.exp(log_p0 - log_p1))
        self.delta = self._rng_binomial(1, prob)

    def _update_upsilon(self, step_size: float = 0.1) -> None:
        current = self.upsilon
        proposal = current + self._rng_normal(0.0, step_size, size=current.shape)
        log_p_curr = self._log_posterior_upsilon_all(current)
        log_p_prop = self._log_posterior_upsilon_all(proposal)
        log_accept = log_p_prop - log_p_curr
        # Use logarithmic comparison to avoid overflow in exp
        rand_log = np.log(self._rng_uniform(size=self.k))
        accept = (log_accept >= 0) | (rand_log < log_accept)
        self.upsilon = np.where(accept[:, None], proposal, current)

    def _update_pi(self) -> None:
        """Update π using conjugate Beta posterior"""
        # Count active components across all k outcomes
        n_active = np.sum(self.delta)
        n_total = self.k * self.d
        
        # Beta posterior parameters
        alpha_post = self.pi_alpha + n_active
        beta_post = self.pi_beta + n_total - n_active
        
        # Sample new π
        self.pi = self._rng_beta(alpha_post, beta_post)

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Run a single Gibbs iteration."""
        self._update_theta()
        self._update_beta()
        self._update_xi()
        self._update_eta()
        self._update_gamma()
        self._update_delta()
        self._update_upsilon()
        self._update_pi()

    def run(
        self,
        n_iter: int,
        *,
        burn_in: int = 0,
        check_convergence: bool = True,
        check_every: int = 50,
        rhat_thresh: float = 1.01,
        ess_thresh: int = 200,
        save_traces: bool = True,
    ) -> dict:
        """Run ``n_iter`` iterations and return traces for ``upsilon`` and ``delta``.

        Parameters
        ----------
        n_iter : int
            Total number of Gibbs iterations to perform.
        burn_in : int, optional
            Number of initial iterations to discard from the returned traces.
        save_traces : bool, optional
            Whether to save parameter traces (memory intensive).
        """
        if burn_in < 0 or burn_in >= n_iter:
            raise ValueError("burn_in must be non-negative and less than n_iter")

        # Initialize memory-efficient trace storage
        if save_traces:
            if self.memory_efficient:
                self.upsilon_trace = MemoryEfficientTrace(max_samples=self.trace_max_samples, dtype=self.dtype)
                self.delta_trace = MemoryEfficientTrace(max_samples=self.trace_max_samples, dtype=self.dtype)
                self.gamma_trace = MemoryEfficientTrace(max_samples=self.trace_max_samples, dtype=self.dtype)
                self.log_beta_trace = MemoryEfficientTrace(max_samples=self.trace_max_samples, dtype=self.dtype)
            else:
                self.upsilon_trace = []
                self.delta_trace = []
                self.gamma_trace = []
                self.log_beta_trace = []
        else:
            self.upsilon_trace = []
            self.delta_trace = []
            self.gamma_trace = []
            self.log_beta_trace = []

        print(f"Running {n_iter} iterations with {burn_in} burn-in...")
        log_memory("Before sampling")
        
        for t in range(n_iter):
            self.step()

            if save_traces:
                if hasattr(self.upsilon_trace, 'append') and callable(getattr(self.upsilon_trace, 'append', None)):
                    self.upsilon_trace.append(self.upsilon.copy())
                    self.delta_trace.append(self.delta.copy())
                    self.gamma_trace.append(self.gamma.copy())
                    self.log_beta_trace.append(self.log_beta.copy())
                else:
                    self.upsilon_trace.append(self.upsilon.copy())
                    self.delta_trace.append(self.delta.copy())
                    self.gamma_trace.append(self.gamma.copy())
                    self.log_beta_trace.append(self.log_beta.copy())

            if check_convergence and (t + 1) % check_every == 0 and (t + 1) > burn_in:
                # Get trace for convergence checking
                if isinstance(self.upsilon_trace, MemoryEfficientTrace):
                    trace = self.upsilon_trace.get_array()[burn_in:]
                else:
                    trace = np.array(self.upsilon_trace[burn_in:])
                
                if trace.shape[0] >= 4:
                    half = trace.shape[0] // 2
                    chains = np.stack([trace[:half], trace[-half:]], axis=0)
                    rhat = _compute_rhat(chains)
                    ess = _effective_sample_size(chains)
                    if np.all(rhat < rhat_thresh) and np.all(ess > ess_thresh):
                        print(
                            f"Converged at iteration {t + 1} "
                            f"(max R-hat {rhat.max():.3f}, min ESS {ess.min():.1f})"
                        )
                        n_iter = t + 1
                        break

            if (t + 1) % 100 == 0:
                print(f"  Completed iteration {t + 1}/{n_iter}")
                if MEMORY_TRACKING:
                    log_memory(f"Iteration {t + 1}")
                # Force garbage collection periodically
                if self.memory_efficient:
                    gc.collect()

        # Convert traces to arrays
        if save_traces:
            if isinstance(self.upsilon_trace, MemoryEfficientTrace):
                trace_upsilon = self.upsilon_trace.get_array()[burn_in:n_iter]
                trace_delta = self.delta_trace.get_array()[burn_in:n_iter]
                trace_gamma = self.gamma_trace.get_array()[burn_in:n_iter]
                trace_log_beta = self.log_beta_trace.get_array()[burn_in:n_iter]
            else:
                trace_upsilon = np.array(self.upsilon_trace)[burn_in:n_iter]
                trace_delta = np.array(self.delta_trace)[burn_in:n_iter]
                trace_gamma = np.array(self.gamma_trace)[burn_in:n_iter]
                trace_log_beta = np.array(self.log_beta_trace)[burn_in:n_iter]
        else:
            trace_upsilon = np.array([])
            trace_delta = np.array([])
            trace_gamma = np.array([])
            trace_log_beta = np.array([])

        print(f"Returning {len(trace_upsilon)} post-burn-in samples")
        log_memory("After sampling")

        return {
            "upsilon": trace_upsilon,
            "delta": trace_delta,
            "gamma": trace_gamma,
            "log_beta": trace_log_beta,
        }
