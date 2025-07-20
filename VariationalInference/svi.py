# Simplified stochastic variational inference implementation
# for the SupervisedPoissonFactorization model.

import numpy as np
import jax.numpy as jnp
import jax.random as random

from vi_model_complete import (
    SupervisedPoissonFactorization,
    logistic,
    _compute_metrics,
)

from sklearn.model_selection import train_test_split


def fit_svi(model, X, Y, X_aux, n_iter=100, batch_size=64, verbose=False):
    """Run a simple mini-batch stochastic variational inference procedure."""
    n = X.shape[0]
    rng = np.random.default_rng(0)

    params = model.initialize_parameters(X, Y, X_aux)

    for it in range(n_iter):
        batch_idx = rng.choice(n, size=min(batch_size, n), replace=False)
        X_b = X[batch_idx]
        Y_b = Y[batch_idx]
        X_aux_b = X_aux[batch_idx]

        expected = model.expected_values(params)

        z_b = model.update_z_latent(X_b, expected["E_theta"][batch_idx], expected["E_beta"])
        scale = n / X_b.shape[0]

        params.update(model.update_eta(params, expected))
        expected = model.expected_values(params)

        a_xi_new = model.alpha_xi + model.K * model.alpha_theta
        b_xi_new = model.lambda_xi + jnp.sum(expected["E_theta"][batch_idx], axis=1)
        params["a_xi"] = params["a_xi"].at[batch_idx].set(a_xi_new)
        params["b_xi"] = params["b_xi"].at[batch_idx].set(b_xi_new)
        expected = model.expected_values(params)

        beta_update = model.update_beta(params, expected, z_b * scale)
        params.update(beta_update)
        expected = model.expected_values(params)

        params.update(model.update_v_minibatch(params, expected, Y_b, X_aux_b, batch_idx, scale))
        expected = model.expected_values(params)

        params.update(model.update_gamma_minibatch(params, expected, Y_b, X_aux_b, batch_idx, scale))
        expected = model.expected_values(params)

        theta_update = model.update_theta_minibatch(params, expected, z_b * scale, Y_b, X_aux_b, batch_idx)
        params["a_theta"] = params["a_theta"].at[batch_idx].set(theta_update["a_theta"])
        params["b_theta"] = params["b_theta"].at[batch_idx].set(theta_update["b_theta"])
        expected = model.expected_values(params)

        params["zeta"] = params["zeta"].at[batch_idx].set(
            model.update_zeta_minibatch(params, expected, Y_b, X_aux_b, batch_idx)["zeta"]
        )

        if verbose and (it % 10 == 0):
            elbo = model.compute_elbo(
                X_b,
                Y_b,
                X_aux_b,
                params,
                z_b,
                return_components=False,
                batch_idx=batch_idx,
            )
            print(f"SVI iter {it+1}, minibatch ELBO {float(elbo):.3f}")

    return params, model.expected_values(params)


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
    batch_size=64,
    return_probs=True,
    sample_ids=None,
    mask=None,
    scores=None,
    plot_elbo=False,
    plot_prefix=None,
    return_params=False,
    verbose=False,
):
    """Fit the model using SVI and evaluate on data splits."""
    if seed is None:
        seed = 0

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

    params, expected = fit_svi(
        model,
        x_data[train_idx],
        y_data[train_idx],
        x_aux[train_idx],
        n_iter=max_iters,
        batch_size=batch_size,
        verbose=verbose,
    )

    all_probs_train = logistic(
        expected["E_theta"] @ params["mu_v"].T + x_aux[train_idx] @ params["mu_gamma"].T
    )

    E_theta_val = model.infer_theta_for_new_samples(x_data[val_idx], x_aux[val_idx], params, n_iter=20)
    all_probs_val = logistic(
        E_theta_val @ params["mu_v"].T + x_aux[val_idx] @ params["mu_gamma"].T
    )
    E_theta_test = model.infer_theta_for_new_samples(x_data[test_idx], x_aux[test_idx], params, n_iter=20)
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

    return results
