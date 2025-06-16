
"""Run Gibbs sampler experiments with train/val/test splits."""

from __future__ import annotations

import argparse
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from scipy.special import expit

from gibbs import SpikeSlabGibbsSampler
from synthetic_test import generate_synthetic_data


def custom_train_test_split(*arrays: np.ndarray, test_size: float, val_size: float, random_state: int | None = None):
    """Split arrays into train/val/test parts.

    This mirrors the helper used in the variational inference experiments.
    """
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)

    remaining_size = test_size + val_size
    split1 = train_test_split(*arrays, indices, test_size=remaining_size, random_state=random_state)
    train_parts = split1[0::2]
    temp_parts = split1[1::2]

    relative_test_size = test_size / remaining_size
    split2 = train_test_split(*temp_parts, test_size=relative_test_size, random_state=random_state)
    val_parts = split2[0::2]
    test_parts = split2[1::2]

    result = []
    n_arrays = len(arrays)
    for i in range(n_arrays):
        result.append(train_parts[i])
        result.append(val_parts[i])
        result.append(test_parts[i])
    return result


def fold_in_theta(sampler: SpikeSlabGibbsSampler, X_new: np.ndarray, n_iter: int = 50, seed: int | None = None) -> np.ndarray:
    """Infer ``theta`` for new samples with parameters fixed from ``sampler``."""
    rng = default_rng(seed)
    n, _ = X_new.shape
    d = sampler.d

    log_theta = np.log(rng.gamma(1.0, 1.0, size=(n, d)) + 1e-12)
    log_xi = np.log(rng.gamma(1.0, 1.0, size=n) + 1e-12)

    log_beta = sampler.log_beta
    a = sampler.a
    a_prime = sampler.a_prime
    b_prime = sampler.b_prime

    for _ in range(n_iter):
        theta_new = np.zeros_like(log_theta)
        for i in range(n):
            for l in range(d):
                rate = np.exp(log_xi[i]) + np.exp(log_beta[:, l]).sum()
                shape = a + np.dot(X_new[i], np.exp(log_beta[:, l]))
                theta_new[i, l] = rng.gamma(shape, 1.0 / rate)
        log_theta = np.log(theta_new + 1e-12)

        xi_new = np.zeros_like(log_xi)
        for i in range(n):
            rate = b_prime + np.exp(log_theta[i]).sum()
            shape = a_prime + a * d
            xi_new[i] = rng.gamma(shape, 1.0 / rate)
        log_xi = np.log(xi_new + 1e-12)

    return np.exp(log_theta)


def evaluate_set(theta: np.ndarray, X_aux: np.ndarray, Y: np.ndarray, sampler: SpikeSlabGibbsSampler) -> float:
    """Compute classification accuracy for a dataset."""
    logits = theta @ sampler.upsilon.T + X_aux @ sampler.gamma.T
    probs = expit(logits)
    preds = (probs > 0.5).astype(int)
    return float(np.mean(preds == Y))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gibbs sampler experiments")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of synthetic samples")
    parser.add_argument("--n_genes", type=int, default=20, help="Number of genes")
    parser.add_argument("--n_programs", type=int, default=3, help="Number of gene programs")
    parser.add_argument("--n_iter", type=int, default=200, help="Number of Gibbs iterations for training")
    parser.add_argument("--fold_in_iter", type=int, default=50, help="Iterations for fold-in theta inference")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    X, Y, X_aux, _ = generate_synthetic_data(
        n_samples=args.n_samples,
        n_genes=args.n_genes,
        n_aux=1,
        n_classes=1,
        n_programs=args.n_programs,
        seed=args.seed,
    )

    # Ensure arrays have correct shapes
    Y = Y.astype(int)

    splits = custom_train_test_split(X, Y, X_aux, test_size=0.15, val_size=0.15, random_state=args.seed)
    X_tr, X_val, X_te, Y_tr, Y_val, Y_te, X_aux_tr, X_aux_val, X_aux_te = splits

    sampler = SpikeSlabGibbsSampler(X_tr, Y_tr, X_aux_tr, n_programs=args.n_programs, seed=args.seed)
    sampler.run(args.n_iter)

    theta_tr = fold_in_theta(sampler, X_tr, n_iter=args.fold_in_iter, seed=args.seed)
    theta_val = fold_in_theta(sampler, X_val, n_iter=args.fold_in_iter, seed=args.seed)
    theta_te = fold_in_theta(sampler, X_te, n_iter=args.fold_in_iter, seed=args.seed)

    acc_tr = evaluate_set(theta_tr, X_aux_tr, Y_tr, sampler)
    acc_val = evaluate_set(theta_val, X_aux_val, Y_val, sampler)
    acc_te = evaluate_set(theta_te, X_aux_te, Y_te, sampler)

    print(f"Train accuracy: {acc_tr:.3f}")
    print(f"Val accuracy:   {acc_val:.3f}")
    print(f"Test accuracy:  {acc_te:.3f}")


if __name__ == "__main__":
    main()
