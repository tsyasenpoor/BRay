import itertools
import json
import numpy as np
import pandas as pd
import argparse
import sys
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
from sklearn.model_selection import train_test_split

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


def _sample_random_params(search_space):
    """Sample a random set of hyperparameters from the search space."""
    params = {}
    for key, spec in search_space.items():
        if isinstance(spec, list):
            params[key] = np.random.choice(spec)
        else:
            low, high = spec.get("low"), spec.get("high")
            if spec.get("type", "float") == "int":
                params[key] = int(np.random.randint(low, high + 1))
            else:
                params[key] = float(np.random.uniform(low, high))
    return params


def _params_to_array(params, keys):
    return np.array([params[k] for k in keys], dtype=float)


def bayesian_search_emtab(
    search_space,
    seed=0,
    max_iters=500,
    output_dir="bayes_search_results",
    n_initial_points=5,
    n_iterations=25,
    verbose=False,
):
    """Run simple Bayesian optimization over the hyperparameter search space."""

    np.random.seed(seed)

    if verbose:
        print("Loading EMTAB dataset...", flush=True)
    adata = prepare_and_load_emtab()

    if verbose:
        print(f"Dataset loaded with shape: {adata.shape}", flush=True)
        print("Preparing data matrices...", flush=True)

    Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
    X = adata.X
    x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
    var_names = list(adata.var_names)
    sample_ids = adata.obs.index.tolist()

    space_keys = list(search_space.keys())
    samples = []
    scores = []
    results_summary = []

    # Initial random evaluations
    for i in range(n_initial_points):
        hyperparams = _sample_random_params(search_space)

        if verbose:
            print(f"Initial sample {i+1}/{n_initial_points}: {hyperparams}", flush=True)

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

        results_summary.append(
            {
                "hyperparams": hyperparams,
                "val_f1": res["val_metrics"]["f1"],
                "best_threshold": thr,
                "best_f1": val_best_f1,
                "val_brier": brier,
                "val_probs": val_probs.tolist(),
                "val_labels": val_labels.tolist(),
            }
        )

        samples.append(_params_to_array(hyperparams, space_keys))
        scores.append(-val_best_f1)

    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)

    for it in range(n_iterations):
        gp.fit(np.array(samples), np.array(scores))

        # Generate candidate points
        candidate_params = [_sample_random_params(search_space) for _ in range(100)]
        candidate_vec = np.array([_params_to_array(c, space_keys) for c in candidate_params])
        mean, std = gp.predict(candidate_vec, return_std=True)
        best_score = np.min(scores)
        improvement = best_score - mean
        Z = improvement / (std + 1e-9)
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        best_idx = int(np.argmax(ei))
        hyperparams = candidate_params[best_idx]

        if verbose:
            print(f"Iteration {it+1}/{n_iterations}: {hyperparams}", flush=True)

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

        results_summary.append(
            {
                "hyperparams": hyperparams,
                "val_f1": res["val_metrics"]["f1"],
                "best_threshold": thr,
                "best_f1": val_best_f1,
                "val_brier": brier,
                "val_probs": val_probs.tolist(),
                "val_labels": val_labels.tolist(),
            }
        )

        samples.append(_params_to_array(hyperparams, space_keys))
        scores.append(-val_best_f1)

    results_df = pd.DataFrame(results_summary)
    best_idx = results_df["best_f1"].idxmax()
    best_params = results_df.loc[best_idx, "hyperparams"]

    best_res = results_summary[best_idx]
    best_probs = np.array(best_res["val_probs"])
    best_labels = np.array(best_res["val_labels"])
    frac_pos, mean_pred = calibration_curve(best_labels, best_probs, n_bins=10)
    calibration_data = {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }

    if verbose:
        print("\nBayesian search completed!", flush=True)
        print(f"Best hyperparameters found: {best_params}", flush=True)
        print(f"Saving results to {output_dir}", flush=True)

    results_df.to_json(f"{output_dir}/bayes_search_results.json", orient="records", indent=2)
    with open(f"{output_dir}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    with open(f"{output_dir}/calibration_data.json", "w") as f:
        json.dump(calibration_data, f, indent=2)

    return results_df, best_params, calibration_data

def grid_search_emtab(param_grid, seed=0, max_iters=500, output_dir='grid_search_results', verbose=False):
    if verbose:
        print("Loading EMTAB dataset...", flush=True)
    adata = prepare_and_load_emtab()
    
    if verbose:
        print(f"Dataset loaded with shape: {adata.shape}", flush=True)
        print("Preparing data matrices...", flush=True)
    
    Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
    X = adata.X
    x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
    var_names = list(adata.var_names)
    sample_ids = adata.obs.index.tolist()

    # Create train/val/test split indices
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    test_size = 0.15
    val_size = 0.15
    train_idx, temp_idx = train_test_split(indices, test_size=val_size + test_size, random_state=seed)
    val_rel = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_rel, random_state=seed)

    grid_keys = list(param_grid.keys())
    all_combos = list(itertools.product(*(param_grid[k] for k in grid_keys)))
    total_combos = len(all_combos)

    if verbose:
        print(f"\nStarting grid search with {total_combos} parameter combinations", flush=True)
        print(f"Data shapes: X={X.shape}, Y={Y.shape}, x_aux={x_aux.shape}", flush=True)
        sys.stdout.flush()

    results_summary = []
    for idx, combo in enumerate(all_combos, 1):
        hyperparams = dict(zip(grid_keys, combo))
        if verbose:
            print(f"\nTrying combination {idx}/{total_combos}:", flush=True)
            print(f"Parameters: {hyperparams}", flush=True)
            sys.stdout.flush()
        
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

        # Get validation probabilities and reshape to match Y's shape
        val_probs = np.array(res["val_probabilities"])
        if Y.shape[1] == 1:
            val_probs = val_probs.reshape(-1)
            val_labels = Y[val_idx].reshape(-1)  # Use first column for single label
        else:
            val_labels = Y[val_idx]  # Use all columns for multiple labels

        # Compute metrics for each label if multiple labels exist
        if Y.shape[1] > 1:
            thresholds = []
            f1_scores = []
            brier_scores = []
            for i in range(Y.shape[1]):
                thr, val_best_f1 = compute_best_threshold(val_probs[:, i], val_labels[:, i])
                brier = brier_score_loss(val_labels[:, i], val_probs[:, i])
                thresholds.append(thr)
                f1_scores.append(val_best_f1)
                brier_scores.append(brier)
            
            # Average the metrics
            thr = np.mean(thresholds)
            val_best_f1 = np.mean(f1_scores)
            brier = np.mean(brier_scores)
        else:
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
        print(f"Finished combo {idx}/{total_combos} {hyperparams}: val_f1={res['val_metrics']['f1']:.3f}, best_thr={thr:.2f}, best_f1={val_best_f1:.3f}, brier={brier:.4f}", flush=True)
        sys.stdout.flush()

    results_df = pd.DataFrame(results_summary)
    best_idx = results_df['best_f1'].idxmax()
    best_params = results_df.loc[best_idx, 'hyperparams']

    # Compute calibration curve for best parameters
    best_res = results_summary[best_idx]
    best_probs = np.array(best_res['val_probs'])
    best_labels = np.array(best_res['val_labels'])
    
    # Handle multiple labels for calibration curve
    if len(best_probs.shape) > 1 and best_probs.shape[1] > 1:
        # Compute calibration curve for each label and average
        frac_pos_list = []
        mean_pred_list = []
        for i in range(best_probs.shape[1]):
            frac_pos, mean_pred = calibration_curve(best_labels[:, i], best_probs[:, i], n_bins=10)
            frac_pos_list.append(frac_pos)
            mean_pred_list.append(mean_pred)
        frac_pos = np.mean(frac_pos_list, axis=0)
        mean_pred = np.mean(mean_pred_list, axis=0)
    else:
        frac_pos, mean_pred = calibration_curve(best_labels, best_probs, n_bins=10)

    calibration_data = {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }

    if verbose:
        print("\nGrid search completed!", flush=True)
        print(f"Best hyperparameters found: {best_params}", flush=True)
        print(f"Saving results to {output_dir}", flush=True)
        sys.stdout.flush()

    # Save results
    results_df.to_json(f"{output_dir}/grid_search_results.json", orient="records", indent=2)
    with open(f"{output_dir}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    with open(f"{output_dir}/calibration_data.json", "w") as f:
        json.dump(calibration_data, f, indent=2)
        
    return results_df, best_params, calibration_data

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for VI model')
    parser.add_argument('--output_dir', type=str, default='search_results',
                        help='Directory to save results')
    parser.add_argument('--max_iters', type=int, default=500,
                        help='Maximum number of VI iterations')
    parser.add_argument('--search_method', choices=['grid', 'bayes'], default='grid',
                        help='Search strategy to use')
    parser.add_argument('--param_grid', type=str,
                        help='JSON string containing parameter grid (for grid search)')
    parser.add_argument('--search_space', type=str,
                        help='JSON string defining search space (for bayesian search)')
    parser.add_argument('--bayes_initial_points', type=int, default=5,
                        help='Number of initial random points for bayesian search')
    parser.add_argument('--bayes_iterations', type=int, default=25,
                        help='Bayesian optimization iterations after initialization')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting hyperparameter search", flush=True)
        print(f"Output directory: {args.output_dir}", flush=True)
        print(f"Max iterations: {args.max_iters}", flush=True)
        print(f"Search method: {args.search_method}", flush=True)
        sys.stdout.flush()

    if args.search_method == 'grid':
        if args.param_grid is None:
            raise ValueError('param_grid must be provided for grid search')
        param_grid = json.loads(args.param_grid)

        if args.verbose:
            print("Parameter grid:", flush=True)
            print(json.dumps(param_grid, indent=2), flush=True)
            sys.stdout.flush()

        results_df, best_params, calib = grid_search_emtab(
            param_grid=param_grid,
            max_iters=args.max_iters,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    else:
        if args.search_space is None:
            raise ValueError('search_space must be provided for bayesian search')
        search_space = json.loads(args.search_space)

        if args.verbose:
            print("Search space:", flush=True)
            print(json.dumps(search_space, indent=2), flush=True)
            sys.stdout.flush()

        results_df, best_params, calib = bayesian_search_emtab(
            search_space=search_space,
            max_iters=args.max_iters,
            output_dir=args.output_dir,
            n_initial_points=args.bayes_initial_points,
            n_iterations=args.bayes_iterations,
            verbose=args.verbose,
        )

if __name__ == "__main__":
    main()

