import argparse
import cProfile
import pstats
import numpy as np

from data import (
    prepare_ajm_dataset,
    filter_protein_coding_genes,
    gene_annotation,
    sample_adata,
)
from vi_model_complete import run_model_and_evaluate


def run_subset_vi(n_cells=500, n_genes=500, max_iters=10, seed=0):
    """Run VI on a small random subset of the real dataset."""
    ajm_ap, ajm_cyto = prepare_ajm_dataset()
    adata = filter_protein_coding_genes(ajm_cyto, gene_annotation)
    subset = sample_adata(adata, n_cells=n_cells, n_genes=n_genes, random_state=seed)

    x_data = subset.X.toarray() if hasattr(subset.X, "toarray") else subset.X
    x_aux = np.ones((subset.n_obs, 1))
    y_data = subset.obs['cyto'].values.astype(float).reshape(-1, 1)
    var_names = list(subset.var_names)

    hyperparams = {
        "alpha_eta": 2.0,
        "lambda_eta": 3.0,
        "alpha_beta": 0.6,
        "alpha_xi": 2.0,
        "lambda_xi": 3.0,
        "alpha_theta": 0.6,
        "sigma2_v": 1.0,
        "sigma2_gamma": 1.0,
        "d": 3,
    }

    results = run_model_and_evaluate(
        x_data=x_data,
        x_aux=x_aux,
        y_data=y_data,
        var_names=var_names,
        hyperparams=hyperparams,
        seed=seed,
        max_iters=max_iters,
        return_probs=True,
        mask=None,
        sample_ids=subset.obs.index.tolist(),
        scores=None,
        plot_elbo=False,
        return_params=False,
    )

    print("Train metrics:", results.get('train_metrics'))
    print("Val metrics:", results.get('val_metrics'))
    print("Test metrics:", results.get('test_metrics'))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run variational inference on a small subset of the real dataset"
    )
    parser.add_argument('--cells', type=int, default=500, help='Number of cells to sample')
    parser.add_argument('--genes', type=int, default=500, help='Number of genes to sample')
    parser.add_argument('--iters', type=int, default=10, help='Maximum VI iterations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--profile', action='store_true', help='Run with cProfile')
    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        run_subset_vi(n_cells=args.cells, n_genes=args.genes, max_iters=args.iters, seed=args.seed)
        profiler.disable()
        profiler.dump_stats('subset_vi.prof')
        pstats.Stats(profiler).sort_stats('cumtime').print_stats(20)
        print('Profile saved to subset_vi.prof')
    else:
        run_subset_vi(n_cells=args.cells, n_genes=args.genes, max_iters=args.iters, seed=args.seed)


if __name__ == '__main__':
    main()
