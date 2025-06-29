import numpy as np
from numpy.random import default_rng
from scipy.special import expit

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


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
    return np.sqrt(var_hat / W)


def _effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Estimate effective sample size using autocorrelation."""
    m, n = chains.shape[0], chains.shape[1]
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)

    var_within = chain_vars.mean(axis=0)
    acov_sum = np.zeros_like(var_within)
    for t in range(1, n):
        acov_t = ((chains[:, :-t] - chain_means[:, None]) * (chains[:, t:] - chain_means[:, None])).mean(axis=(0,1))
        rho_t = acov_t / var_within
        if np.all(rho_t < 0):
            break
        acov_sum += rho_t
    tau = 1 + 2 * acov_sum
    ess = m * n / tau
    return ess


class SpikeSlabGibbsSampler:
    """Gibbs sampler using a spike-and-slab prior for ``upsilon`` coefficients.

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
        seed: int | None = None,
        mask: np.ndarray | None = None,
        use_jax: bool = False,
    ) -> None:
        self.use_jax = bool(use_jax and JAX_AVAILABLE)
        self.rng = default_rng(seed)
        if self.use_jax:
            self.key = jrandom.PRNGKey(seed or 0)

        self.X = np.asarray(X)
        self.X_aux = np.asarray(X_aux)
        self.Y = np.asarray(Y)
        if self.use_jax:
            self.X = jnp.asarray(self.X)
            self.X_aux = jnp.asarray(self.X_aux)
            self.Y = jnp.asarray(self.Y)

        self.n, self.p = self.X.shape
        self.q = self.X_aux.shape[1]
        self.k = self.Y.shape[1]
        self.d = n_programs

        self.mask = None
        if mask is not None:
            m = np.asarray(mask)
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
        )
        self.log_beta = np.log(
            self._rng_gamma(np.ones((self.p, self.d)), np.ones((self.p, self.d))) + 1e-12
        )
        if self.mask is not None:
            self.log_beta = np.where(self.mask, self.log_beta, -np.inf)
        self.log_xi = np.log(self._rng_gamma(np.ones(self.n), np.ones(self.n)) + 1e-12)
        self.log_eta = np.log(self._rng_gamma(np.ones(self.p), np.ones(self.p)) + 1e-12)

        self.gamma = self._rng_normal(0.0, np.sqrt(self.sigma_gamma_sq), size=(self.k, self.q))

        self.delta = self._rng_binomial(1, self.pi, size=(self.k, self.d))
        self.upsilon = self._rng_normal(0.0, 0.1, size=(self.k, self.d))

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
    # Conjugate updates for gamma-distributed parameters
    def _update_theta(self) -> None:
        exp_beta = np.exp(self.log_beta)
        rate = np.exp(self.log_xi)[:, None] + exp_beta.sum(axis=0)[None, :]
        shape = self.a + self.X @ exp_beta
        theta_new = self._rng_gamma(shape, rate)
        self.log_theta = np.log(theta_new + 1e-12)

    def _update_beta(self) -> None:
        exp_theta = np.exp(self.log_theta)
        rate = np.exp(self.log_eta)[:, None] + exp_theta.sum(axis=0)[None, :]
        shape = self.c + self.X.T @ exp_theta
        beta_new = self._rng_gamma(shape, rate)
        if self.mask is not None:
            beta_new = np.where(self.mask, beta_new, 0.0)
        self.log_beta = np.log(beta_new + 1e-12)
        if self.mask is not None:
            self.log_beta = np.where(self.mask, self.log_beta, -np.inf)

    def _update_xi(self) -> None:
        exp_theta = np.exp(self.log_theta)
        rate = self.b_prime + exp_theta.sum(axis=1)
        shape = self.a_prime + self.a * self.d
        xi_new = self._rng_gamma(np.full(self.n, shape), rate)
        self.log_xi = np.log(xi_new + 1e-12)

    def _update_eta(self) -> None:
        exp_beta = np.exp(self.log_beta)
        rate = self.d_prime + exp_beta.sum(axis=1)
        shape = self.c_prime + self.c * self.d
        eta_new = self._rng_gamma(np.full(self.p, shape), rate)
        self.log_eta = np.log(eta_new + 1e-12)

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
        self.gamma = gamma_new

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
        accept = (log_accept >= 0) | (self._rng_uniform(size=self.k) < np.exp(log_accept))
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
    ) -> dict:
        """Run ``n_iter`` iterations and return traces for ``upsilon`` and ``delta``.

        Parameters
        ----------
        n_iter : int
            Total number of Gibbs iterations to perform.
        burn_in : int, optional
            Number of initial iterations to discard from the returned traces.
        """
        if burn_in < 0 or burn_in >= n_iter:
            raise ValueError("burn_in must be non-negative and less than n_iter")

        self.upsilon_trace = []
        self.delta_trace = []
        self.gamma_trace = []
        self.log_beta_trace = []

        print(f"Running {n_iter} iterations with {burn_in} burn-in...")
        
        for t in range(n_iter):
            self.step()

            self.upsilon_trace.append(self.upsilon.copy())
            self.delta_trace.append(self.delta.copy())
            self.gamma_trace.append(self.gamma.copy())
            self.log_beta_trace.append(self.log_beta.copy())

            if check_convergence and (t + 1) % check_every == 0 and (t + 1) > burn_in:
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

        trace_upsilon = np.array(self.upsilon_trace[burn_in:n_iter])
        trace_delta = np.array(self.delta_trace[burn_in:n_iter])
        trace_gamma = np.array(self.gamma_trace[burn_in:n_iter])
        trace_log_beta = np.array(self.log_beta_trace[burn_in:n_iter])

        print(f"Returning {len(trace_upsilon)} post-burn-in samples")

        return {
            "upsilon": trace_upsilon,
            "delta": trace_delta,
            "gamma": trace_gamma,
            "log_beta": trace_log_beta,
        }
