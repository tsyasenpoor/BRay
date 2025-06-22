import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from vi_model_complete import run_model_and_evaluate


def generate_synthetic_data(n_samples, n_genes, n_programs, p_aux=1, seed=0):
    key = random.PRNGKey(seed)
    keys = random.split(key, 10)
    hyperparams = {
        "c_prime": 2.0,
        "d_prime": 2.0,
        "c": 0.8,
        "a_prime": 2.0,
        "b_prime": 2.0,
        "a": 0.8,
        "tau": 2.0,
        "sigma": 2.0,
        "d": n_programs,
    }
    eta_true = random.gamma(keys[0], hyperparams["c_prime"], shape=(n_genes,)) / hyperparams["d_prime"]
    beta_true = random.gamma(keys[1], hyperparams["c"], shape=(n_genes, n_programs)) / jnp.expand_dims(eta_true, axis=1)
    xi_true = random.gamma(keys[2], hyperparams["a_prime"], shape=(n_samples,)) / hyperparams["b_prime"]
    theta_true = random.gamma(keys[3], hyperparams["a"], shape=(n_samples, n_programs)) / jnp.expand_dims(xi_true, axis=1)
    gamma_true = jnp.zeros((1, p_aux))
    upsilon_true = jnp.zeros((1, n_programs))
    rate = jnp.dot(theta_true, beta_true.T)
    x_data = random.poisson(keys[6], rate)
    x_aux = jnp.ones((n_samples, p_aux))
    logits = jnp.dot(x_aux, gamma_true.T) + jnp.dot(theta_true, upsilon_true.T)
    probs = jax.nn.sigmoid(logits)
    y_data = (random.uniform(keys[8], shape=logits.shape) < probs).astype(int)
    return {
        "x_data": x_data,
        "x_aux": x_aux,
        "y_data": y_data,
        "hyperparams": hyperparams,
    }


def main():
    data = generate_synthetic_data(n_samples=30, n_genes=15, n_programs=3, p_aux=1, seed=0)
    x_data = np.array(data['x_data'])
    x_aux = np.array(data['x_aux'])
    y_data = np.array(data['y_data'])
    var_names = [f"gene_{i}" for i in range(x_data.shape[1])]
    hyperparams = data['hyperparams']
    results = run_model_and_evaluate(
        x_data=x_data,
        x_aux=x_aux,
        y_data=y_data,
        var_names=var_names,
        hyperparams=hyperparams,
        seed=0,
        max_iters=10,
        return_probs=True,
        mask=None,
        sample_ids=None,
        scores=None,
        plot_elbo=False,
        plot_prefix=None,
        return_params=False,
    )
    print("Train metrics:", results.get('train_metrics'))
    print("Val metrics:", results.get('val_metrics'))
    print("Test metrics:", results.get('test_metrics'))


if __name__ == "__main__":
    main()
