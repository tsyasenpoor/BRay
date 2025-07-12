import itertools
import json
import numpy as np
import pandas as pd
import argparse
import sys
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
        print(f"Finished combo {idx}/{total_combos} {hyperparams}: val_f1={res['val_metrics']['f1']:.3f}, best_thr={thr:.2f}, best_f1={val_best_f1:.3f}, brier={brier:.4f}", flush=True)
        sys.stdout.flush()

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
    parser = argparse.ArgumentParser(description='Run grid search for VI model hyperparameters')
    parser.add_argument('--output_dir', type=str, default='grid_search_results',
                      help='Directory to save results')
    parser.add_argument('--max_iters', type=int, default=500,
                      help='Maximum number of VI iterations')
    parser.add_argument('--param_grid', type=str, required=True,
                      help='JSON string containing parameter grid')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting grid search script", flush=True)
        print(f"Output directory: {args.output_dir}", flush=True)
        print(f"Max iterations: {args.max_iters}", flush=True)
        sys.stdout.flush()
    
    # Parse param grid from JSON string
    param_grid = json.loads(args.param_grid)
    
    if args.verbose:
        print("Parameter grid:", flush=True)
        print(json.dumps(param_grid, indent=2), flush=True)
        sys.stdout.flush()
    
    # Run grid search
    results_df, best_params, calib = grid_search_emtab(
        param_grid=param_grid,
        max_iters=args.max_iters,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()

