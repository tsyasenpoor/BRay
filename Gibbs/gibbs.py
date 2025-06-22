import numpy as np
from numpy.random import default_rng
from scipy.special import expit, logsumexp


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
        tau0_sq: float = 1e-4,
        pi: float = 0.2,
        sigma_gamma_sq: float = 1.0,
        seed: int | None = None,
        mask: np.ndarray | None = None,
    ) -> None:
        self.rng = default_rng(seed)

        self.X = np.asarray(X)
        self.X_aux = np.asarray(X_aux)
        self.Y = np.asarray(Y)

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
        self.pi = pi
        self.sigma_gamma_sq = sigma_gamma_sq

        self._init_params()

    # ------------------------------------------------------------------
    def _init_params(self) -> None:
        """Initialise latent variables and parameters."""
        self.log_theta = np.log(
            self.rng.gamma(1.0, 1.0, size=(self.n, self.d)) + 1e-12
        )
        self.log_beta = np.log(
            self.rng.gamma(1.0, 1.0, size=(self.p, self.d)) + 1e-12
        )
        if self.mask is not None:
            self.log_beta = np.where(self.mask, self.log_beta, -np.inf)
        self.log_xi = np.log(self.rng.gamma(1.0, 1.0, size=self.n) + 1e-12)
        self.log_eta = np.log(self.rng.gamma(1.0, 1.0, size=self.p) + 1e-12)

        self.gamma = self.rng.normal(0.0, np.sqrt(self.sigma_gamma_sq), size=(self.k, self.q))

        self.delta = self.rng.binomial(1, self.pi, size=(self.k, self.d))
        self.upsilon = self.rng.normal(0.0, 0.1, size=(self.k, self.d))

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
        theta_new = np.zeros_like(self.log_theta)
        for i in range(self.n):
            for l in range(self.d):
                rate = np.exp(self.log_xi[i]) + np.exp(self.log_beta[:, l]).sum()
                shape = self.a + np.dot(self.X[i], np.exp(self.log_beta[:, l]))
                theta_new[i, l] = self.rng.gamma(shape, 1.0 / rate)
        self.log_theta = np.log(theta_new + 1e-12)

    def _update_beta(self) -> None:
        beta_new = np.zeros_like(self.log_beta)
        for j in range(self.p):
            for l in range(self.d):
                if self.mask is not None and self.mask[j, l] == 0:
                    beta_new[j, l] = 0.0
                    continue
                rate = np.exp(self.log_eta[j]) + np.exp(self.log_theta[:, l]).sum()
                shape = self.c + np.dot(self.X[:, j], np.exp(self.log_theta[:, l]))
                beta_new[j, l] = self.rng.gamma(shape, 1.0 / rate)
        self.log_beta = np.log(beta_new + 1e-12)
        if self.mask is not None:
            self.log_beta = np.where(self.mask, self.log_beta, -np.inf)

    def _update_xi(self) -> None:
        xi_new = np.zeros_like(self.log_xi)
        for i in range(self.n):
            rate = self.b_prime + np.exp(self.log_theta[i]).sum()
            shape = self.a_prime + self.a * self.d
            xi_new[i] = self.rng.gamma(shape, 1.0 / rate)
        self.log_xi = np.log(xi_new + 1e-12)

    def _update_eta(self) -> None:
        eta_new = np.zeros_like(self.log_eta)
        for j in range(self.p):
            rate = self.d_prime + np.exp(self.log_beta[j]).sum()
            shape = self.c_prime + self.c * self.d
            eta_new[j] = self.rng.gamma(shape, 1.0 / rate)
        self.log_eta = np.log(eta_new + 1e-12)

    # ------------------------------------------------------------------
    def _update_gamma(self) -> None:
        gamma_new = np.zeros_like(self.gamma)
        XTX = self.X_aux.T @ self.X_aux
        precision = XTX / self.sigma_gamma_sq + np.eye(self.q)
        cov = np.linalg.inv(precision)
        for k in range(self.k):
            logits = np.exp(self.log_theta) @ self.upsilon[k] + self.X_aux @ self.gamma[k]
            z = self.Y[:, k] - expit(logits)
            mean = cov @ self.X_aux.T @ z / self.sigma_gamma_sq
            gamma_new[k] = self.rng.multivariate_normal(mean, cov)
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

    def _log_posterior_upsilon(self, k: int, upsilon_k: np.ndarray) -> float:
        logits = np.exp(self.log_theta) @ upsilon_k + self.X_aux @ self.gamma[k]
        log_lik = self._log_likelihood(logits, self.Y[:, k])
        prior_var = np.where(self.delta[k] == 1, self.tau1_sq, self.tau0_sq)
        log_prior = -0.5 * np.sum(upsilon_k**2 / prior_var)
        return log_lik + log_prior

    def _update_delta(self) -> None:
        delta_new = np.zeros_like(self.delta)
        for k in range(self.k):
            for l in range(self.d):
                v = self.upsilon[k, l]
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
                delta_new[k, l] = self.rng.binomial(1, prob)
        self.delta = delta_new

    def _update_upsilon(self, step_size: float = 0.1) -> None:
        upsilon_new = np.copy(self.upsilon)
        for k in range(self.k):
            current = self.upsilon[k]
            proposal = current + self.rng.normal(0.0, step_size, size=current.shape)
            log_p_curr = self._log_posterior_upsilon(k, current)
            log_p_prop = self._log_posterior_upsilon(k, proposal)
            log_accept = log_p_prop - log_p_curr
            if log_accept >= 0 or self.rng.random() < np.exp(log_accept):
                upsilon_new[k] = proposal
        self.upsilon = upsilon_new

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

    def run(self, n_iter: int) -> dict:
        """Run ``n_iter`` iterations and return traces for ``upsilon`` and ``delta``."""
        trace_upsilon = np.zeros((n_iter, *self.upsilon.shape))
        trace_delta = np.zeros((n_iter, *self.delta.shape))

        for t in range(n_iter):
            self.step()
            trace_upsilon[t] = self.upsilon
            trace_delta[t] = self.delta

        return {
            "upsilon": trace_upsilon,
            "delta": trace_delta,
        }

