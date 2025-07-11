import itertools
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.calibration import calibration_curve

from data import prepare_and_load_emtab
from vi_model_complete import run_model_and_evaluate


def compute_best_threshold(probs, labels, n_steps=100):
    best_thr = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.0, 1.0, n_steps + 1):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    return best_thr, best_f1


def grid_search_emtab(param_grid, seed=0, max_iters=100):
    adata = prepare_and_load_emtab()
    Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
    X = adata.X
    x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
    var_names = list(adata.var_names)
    sample_ids = adata.obs.index.tolist()

    grid_keys = list(param_grid.keys())
    all_combos = list(itertools.product(*(param_grid[k] for k in grid_keys)))

    results_summary = []
    for combo in all_combos:
        hyperparams = dict(zip(grid_keys, combo))
        res = run_model_and_evaluate(
            x_data=X,
            x_aux=x_aux,
            y_data=Y,
            var_names=var_names,
            hyperparams=hyperparams,
            seed=seed,
            max_iters=max_iters,
            return_probs=True,
            sample_ids=sample_ids,
            mask=None,
            scores=None,
            plot_elbo=False,
            return_params=False,
        )

        val_probs = np.array(res["val_probabilities"]).reshape(-1)
        val_labels = res["val_results_df"]["true_label"].values
        thr, val_best_f1 = compute_best_threshold(val_probs, val_labels)
        brier = brier_score_loss(val_labels, val_probs)
        results_summary.append({
            "hyperparams": hyperparams,
            "val_f1": res["val_metrics"]["f1"],
            "best_threshold": thr,
            "best_f1": val_best_f1,
            "val_brier": brier,
            "val_probs": val_probs.tolist(),
            "val_labels": val_labels.tolist(),
        })
        print(f"Finished combo {hyperparams}: val_f1={res['val_metrics']['f1']:.3f}, best_thr={thr:.2f}, best_f1={val_best_f1:.3f}, brier={brier:.4f}")

    results_df = pd.DataFrame(results_summary)
    best_idx = results_df['best_f1'].idxmax()
    best_params = results_df.loc[best_idx, 'hyperparams']

    # Compute calibration curve for best parameters
    best_res = results_summary[best_idx]
    best_probs = np.array(best_res['val_probs'])
    best_labels = np.array(best_res['val_labels'])
    frac_pos, mean_pred = calibration_curve(best_labels, best_probs, n_bins=10)
    calibration_data = {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }

    print("Best hyperparameters:", best_params)
    return results_df, best_params, calibration_data


def main():
    param_grid = {
        "c_prime": [2.0],
        "d_prime": [3.0],
        "c": [0.6, 0.8],
        "a_prime": [2.0],
        "b_prime": [3.0],
        "a": [0.6, 0.8],
        "tau": [1.0],
        "sigma": [1.0],
        "d": [25, 50]
    }
    results_df, best_params, calib = grid_search_emtab(param_grid)
    results_df.to_json("grid_search_results.json", orient="records", indent=2)
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    with open("calibration_data.json", "w") as f:
        json.dump(calib, f, indent=2)


if __name__ == "__main__":
    main()

