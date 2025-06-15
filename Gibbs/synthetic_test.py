import numpy as np
from numpy.random import default_rng
from scipy.special import expit, logsumexp

from gibbs import SpikeSlabGibbsSampler


def generate_synthetic_data(
    n_samples: int,
    n_genes: int,
    n_aux: int,
    n_classes: int,
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
):
    """Generate synthetic data from the spike–slab model."""
    rng = default_rng(seed)

    # latent gamma variables
    xi = rng.gamma(a_prime, 1.0 / b_prime, size=n_samples)
    eta = rng.gamma(c_prime, 1.0 / d_prime, size=n_genes)

    log_theta = np.log(
        rng.gamma(a, 1.0 / xi[:, None], size=(n_samples, n_programs)) + 1e-12
    )
    log_beta = np.log(
        rng.gamma(c, 1.0 / eta[:, None], size=(n_genes, n_programs)) + 1e-12
    )

    log_rate = logsumexp(log_theta[:, None, :] + log_beta[None, :, :], axis=2)
    X = rng.poisson(np.exp(log_rate))

    if n_aux == 1:
        X_aux = np.ones((n_samples, 1))
    else:
        X_aux = rng.normal(size=(n_samples, n_aux))
        X_aux[:, 0] = 1.0

    gamma = rng.normal(0.0, np.sqrt(sigma_gamma_sq), size=(n_classes, n_aux))
    delta = rng.binomial(1, pi, size=(n_classes, n_programs))
    upsilon = rng.normal(0.0, np.sqrt(tau0_sq), size=(n_classes, n_programs))
    inds = delta == 1
    upsilon[inds] = rng.normal(0.0, np.sqrt(tau1_sq), size=np.count_nonzero(inds))

    theta = np.exp(log_theta)
    beta = np.exp(log_beta)
    logits = theta @ upsilon.T + X_aux @ gamma.T
    prob = expit(logits)
    Y = rng.binomial(1, prob)

    params = {
        "theta": theta,
        "beta": beta,
        "xi": xi,
        "eta": eta,
        "gamma": gamma,
        "delta": delta,
        "upsilon": upsilon,
        "log_theta": log_theta,
        "log_beta": log_beta,
        "log_rate": log_rate,
    }
    return X, Y, X_aux, params


if __name__ == "__main__":
    n = 50
    p = 20
    q = 1
    k = 1
    d = 3
    X, Y, X_aux, params = generate_synthetic_data(n, p, q, k, d, seed=0)

    sampler = SpikeSlabGibbsSampler(X, Y, X_aux, n_programs=d, seed=0)
    sampler.run(200)

    log_rate_true = params["log_rate"]
    log_rate_est = logsumexp(
        sampler.log_theta[:, None, :] + sampler.log_beta[None, :, :], axis=2
    )
    corr = np.corrcoef(log_rate_true.flatten(), log_rate_est.flatten())[0, 1]
    print(f"Correlation between true and estimated rate matrix: {corr:.3f}")
