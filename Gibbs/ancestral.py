import numpy as np
from numpy.random import default_rng
from scipy.special import expit

from gibbs import SpikeSlabGibbsSampler


def generate_synthetic_data(n_samples: int, n_genes: int, n_programs: int,
                            n_labels: int = 1, p_aux: int = 1,
                            seed: int = 0):
    """Generate synthetic count data under the spike--slab model."""
    rng = default_rng(seed)

    alpha_theta = 0.3
    alpha_beta = 0.3
    alpha_xi = 0.3
    alpha_eta = 0.3
    lambda_xi = 0.3
    lambda_eta = 0.3
    pi_upsilon = 0.5
    sigma_slab_sq = 1.0
    sigma_gamma_sq = 1.0

    xi = rng.gamma(alpha_xi, scale=1.0 / lambda_xi, size=n_samples)
    eta = rng.gamma(alpha_eta, scale=1.0 / lambda_eta, size=n_genes)

    theta = rng.gamma(alpha_theta, scale=1.0 / xi[:, None], size=(n_samples, n_programs))
    beta = rng.gamma(alpha_beta, scale=1.0 / eta[:, None], size=(n_genes, n_programs))

    gamma = rng.normal(0.0, np.sqrt(sigma_gamma_sq), size=(n_labels, p_aux))
    s_upsilon = rng.binomial(1, pi_upsilon, size=(n_labels, n_programs))
    w_upsilon = rng.normal(0.0, np.sqrt(sigma_slab_sq), size=(n_labels, n_programs))
    upsilon = s_upsilon * w_upsilon

    x_aux = np.ones((n_samples, p_aux))
    rate = theta @ beta.T
    x = rng.poisson(rate)

    logits = x_aux @ gamma.T + theta @ upsilon.T
    y_prob = expit(logits)
    y = rng.binomial(1, y_prob)

    true_params = {
        "theta": theta,
        "beta": beta,
        "xi": xi,
        "eta": eta,
        "gamma": gamma,
        "s_upsilon": s_upsilon,
        "w_upsilon": w_upsilon,
        "upsilon": upsilon,
    }

    return x, y, x_aux, true_params


def run_sampler(x: np.ndarray, y: np.ndarray, x_aux: np.ndarray,
                n_programs: int, iterations: int = 50, seed: int = 0):
    sampler = SpikeSlabGibbsSampler(x, y, x_aux, n_programs, seed=seed)
    print(f"Starting Gibbs sampling for {iterations} iterations...")
    for it in range(iterations):
        if (it + 1) % 10 == 0 or it == 0:
            print(f"  Iteration {it + 1}/{iterations}")
        sampler._update_latent_counts_z()
        sampler._update_beta()
        sampler._update_eta()
        sampler._update_xi()
        sampler._update_theta()
        sampler._update_gamma()
        sampler._update_s_upsilon()
        sampler._update_w_upsilon()
        sampler.upsilon = sampler.s_upsilon * sampler.w_upsilon
    print("Gibbs sampling complete.")
    return sampler


def parameter_recovery(true_params, sampler: SpikeSlabGibbsSampler):
    def corr(a, b):
        a = a.flatten()
        b = b.flatten()
        if a.size == 0 or b.size == 0:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    results = {
        "corr_theta": corr(true_params["theta"], sampler.theta),
        "corr_beta": corr(true_params["beta"], sampler.beta),
        "corr_xi": corr(true_params["xi"], sampler.xi),
        "corr_eta": corr(true_params["eta"], sampler.eta),
        "corr_gamma": corr(true_params["gamma"], sampler.gamma),
        "corr_upsilon": corr(true_params["upsilon"], sampler.w_upsilon * sampler.s_upsilon),
    }
    return results


if __name__ == "__main__":
    X, Y, X_aux, params = generate_synthetic_data(n_samples=50,
                                                  n_genes=20,
                                                  n_programs=3,
                                                  n_labels=1,
                                                  p_aux=1,
                                                  seed=0)
    sampler = run_sampler(X, Y, X_aux, n_programs=3, iterations=20, seed=0)
    recov = parameter_recovery(params, sampler)
    for k, v in recov.items():
        print(f"{k}: {v:.3f}")