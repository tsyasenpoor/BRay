import itertools
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import f1_score, brier_score_loss

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate


def find_best_threshold(probs: np.ndarray, y_true: np.ndarray, steps: int = 100):
    """Return threshold giving highest F1 score."""
    thresholds = np.linspace(0.0, 1.0, steps)
    best_t = 0.5
    best_f1 = -np.inf
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t, best_f1


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            bin_acc = y_true[mask].mean()
            bin_conf = probs[mask].mean()
            ece += np.abs(bin_conf - bin_acc) * mask.mean()
    return ece


def grid_search_emtab(param_grid: Dict[str, List[Any]], label: str = "Crohn's disease", max_iters: int = 100):
    """Run grid search over hyperparameters for the EMTAB dataset.

    Parameters
    ----------
    param_grid : Dict[str, List[Any]]
        Dictionary mapping hyperparameter names to lists of values.
    label : str
        Target label column to use from EMTAB.
    max_iters : int
        Maximum iterations for variational inference.

    Returns
    -------
    Dict[str, Any]
        Result dictionary for the best parameter set.
    """
    adata = prepare_and_load_emtab()
    X = adata.X
    y = adata.obs[label].values.astype(float).reshape(-1, 1)
    aux_cols = [c for c in ["age", "sex_female"] if c in adata.obs.columns]
    if aux_cols:
        x_aux = adata.obs[aux_cols].values.astype(float)
        x_aux = np.column_stack([np.ones(x_aux.shape[0]), x_aux])
    else:
        x_aux = np.ones((X.shape[0], 1))

    var_names = list(adata.var_names)
    sample_ids = adata.obs.index.tolist()

    hyper_keys = list(param_grid.keys())
    best = None
    best_f1 = -np.inf

    for values in itertools.product(*[param_grid[k] for k in hyper_keys]):
        hyperparams = dict(zip(hyper_keys, values))
        print(f"Running hyperparams: {hyperparams}")

        results = run_model_and_evaluate(
            x_data=X,
            x_aux=x_aux,
            y_data=y,
            var_names=var_names,
            hyperparams=hyperparams,
            seed=0,
            test_size=0.15,
            val_size=0.15,
            max_iters=max_iters,
            return_probs=True,
            sample_ids=sample_ids,
            mask=None,
            scores=None,
            return_params=False,
            plot_elbo=False,
            plot_prefix=None,
            beta_init=None,
        )

        if "val_probabilities" not in results:
            continue

        val_probs = np.array(results["val_probabilities"]).reshape(-1)
        val_labels = results["val_results_df"]["true_label"].values
        best_t, val_f1 = find_best_threshold(val_probs, val_labels)
        val_brier = brier_score_loss(val_labels, val_probs)
        val_ece = compute_ece(val_labels, val_probs)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best = {
                "hyperparams": hyperparams,
                "best_threshold": best_t,
                "val_f1": val_f1,
                "val_brier": val_brier,
                "val_ece": val_ece,
                "results": results,
            }

        print(
            f"Validation F1: {val_f1:.4f}, Brier: {val_brier:.4f}, ECE: {val_ece:.4f}, threshold: {best_t:.2f}"
        )

    return best
