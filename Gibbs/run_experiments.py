import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import jax
import jax.numpy as jnp
import gseapy
from gseapy import read_gmt
import mygene  
import argparse
import gc
import psutil  
import pickle  # Add pickle for caching results
import random  # Import for random sampling of pathways
import gzip # Import for gzipping files

from memory_tracking import get_memory_usage, log_memory, log_array_sizes, clear_memory

# Log initial memory
print(f"Initial memory usage: {get_memory_usage():.2f} MB")

from gibbs import SpikeSlabGibbsSampler
from data import *


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from numpy.random import default_rng
from scipy.special import expit


def custom_train_test_split(*arrays, test_size=0.15, val_szie=0.15, random_state=None):
    n_samples = len(arrays[0])
    indices = np.arange(n_samples)

    remaining_size = test_size + val_szie
    
    arrays_and_indices = list(arrays) + [indices]
    split1_results = train_test_split(*arrays_and_indices, test_size=remaining_size, random_state=random_state)
    
    train_parts = split1_results[0::2]
    temp_parts = split1_results[1::2]

    relative_test_size = test_size / remaining_size

    split2_results = train_test_split(*temp_parts, test_size=relative_test_size, random_state=random_state)
    val_parts = split2_results[0::2]
    test_parts = split2_results[1::2]

    final_result = []
    num_arrays = len(arrays)
    for i in range(num_arrays):
        final_result.append(train_parts[i])
        final_result.append(val_parts[i])
        final_result.append(test_parts[i])
    
    return final_result


def _evaluate_predictions(y_true, probs, threshold=0.5, return_probs=True):
    """Compute classification metrics given true labels and predicted probabilities."""
    y_true = np.array(y_true)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    kappa = y_true.shape[1]
    preds = (np.array(probs) >= threshold).astype(int)

    results = {}
    if kappa == 1:
        y_true_f = y_true.ravel()
        y_pred_f = preds.ravel()
        probs_f = np.array(probs).ravel()

        acc = accuracy_score(y_true_f, y_pred_f)
        prec = precision_score(y_true_f, y_pred_f, zero_division=0)
        rec = recall_score(y_true_f, y_pred_f, zero_division=0)
        f1v = f1_score(y_true_f, y_pred_f, zero_division=0)
        try:
            aucv = roc_auc_score(y_true_f, probs_f)
        except ValueError:
            aucv = 0.5
        cm = confusion_matrix(y_true_f, y_pred_f)
        results = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1v,
            "roc_auc": aucv,
            "confusion_matrix": cm.tolist(),
            "threshold": threshold,
        }
    else:
        metrics_per_class = []
        for k in range(kappa):
            yt = y_true[:, k]
            yp = preds[:, k]
            pr = np.array(probs)[:, k]
            acc_k = accuracy_score(yt, yp)
            prec_k = precision_score(yt, yp, zero_division=0)
            rec_k = recall_score(yt, yp, zero_division=0)
            f1_k = f1_score(yt, yp, zero_division=0)
            try:
                auc_k = roc_auc_score(yt, pr)
            except ValueError:
                auc_k = 0.5
            metrics_per_class.append({
                "class": k,
                "accuracy": acc_k,
                "precision": prec_k,
                "recall": rec_k,
                "f1": f1_k,
                "roc_auc": auc_k,
            })

        acc_macro = np.mean([m["accuracy"] for m in metrics_per_class])
        prec_macro = np.mean([m["precision"] for m in metrics_per_class])
        rec_macro = np.mean([m["recall"] for m in metrics_per_class])
        f1_macro = np.mean([m["f1"] for m in metrics_per_class])
        auc_macro = np.mean([m["roc_auc"] for m in metrics_per_class])

        results = {
            "accuracy": acc_macro,
            "precision": prec_macro,
            "recall": rec_macro,
            "f1": f1_macro,
            "roc_auc": auc_macro,
            "per_class_metrics": metrics_per_class,
            "threshold": threshold,
        }

    if return_probs:
        results["probabilities"] = np.array(probs).tolist()
    return results


def _fold_in_theta_gibbs(X_new, sampler, n_iter=30):
    """Estimate latent theta for new data using the trained sampler parameters."""
    rng = default_rng(sampler.rng.integers(1, 1_000_000))
    n, p = X_new.shape
    d = sampler.d
    theta = rng.gamma(1.0, 1.0, size=(n, d))
    xi = rng.gamma(1.0, 1.0, size=n)

    beta = np.exp(sampler.log_beta)
    sum_beta = beta.sum(axis=0)

    for _ in range(n_iter):
        for i in range(n):
            for l in range(d):
                rate = xi[i] + sum_beta[l]
                shape = sampler.a + np.dot(X_new[i], beta[:, l])
                theta[i, l] = rng.gamma(shape, 1.0 / rate)
            xi_rate = sampler.b_prime + theta[i].sum()
            xi_shape = sampler.a_prime + sampler.a * d
            xi[i] = rng.gamma(xi_shape, 1.0 / xi_rate)

    return theta


def run_sampler_and_evaluate(x_data, x_aux, y_data, var_names, hyperparams,
                             seed=None, test_size=0.15, val_size=0.15,
                             max_iters=100, return_probs=True, sample_ids=None,
                             mask=None, scores=None, return_params=False):
    if sample_ids is None:
        sample_ids = np.arange(x_data.shape[0])
    if scores is None:
        scores = np.zeros(x_data.shape[0])

    # Add debugging prints
    print(f"DEBUG: Input x_data shape: {x_data.shape}")
    print(f"DEBUG: Input x_data type: {type(x_data)}")
    print(f"DEBUG: Input y_data shape: {y_data.shape}")
    print(f"DEBUG: Input sample_ids length: {len(sample_ids)}")

    temp_size = val_size + test_size
    X_train, X_temp, XA_train, XA_temp, y_train, y_temp, ids_train, ids_temp, sc_train, sc_temp = train_test_split(
        x_data, x_aux, y_data, sample_ids, scores, test_size=temp_size, random_state=seed
    )
    rel_test = test_size / temp_size
    X_val, X_test, XA_val, XA_test, y_val, y_test, ids_val, ids_test, sc_val, sc_test = train_test_split(
        X_temp, XA_temp, y_temp, ids_temp, sc_temp, test_size=rel_test, random_state=seed
    )

    print(f"DEBUG: X_train shape: {X_train.shape}")
    print(f"DEBUG: X_train type: {type(X_train)}")
    print(f"DEBUG: y_train shape: {y_train.shape}")
    print(f"DEBUG: XA_train shape: {XA_train.shape}")

    # Convert sparse matrices to dense arrays if needed
    if hasattr(X_train, 'toarray'):
        print("Converting X_train from sparse to dense")
        X_train = X_train.toarray()
    if hasattr(X_val, 'toarray'):
        print("Converting X_val from sparse to dense")
        X_val = X_val.toarray()
    if hasattr(X_test, 'toarray'):
        print("Converting X_test from sparse to dense")
        X_test = X_test.toarray()
    
    print(f"DEBUG: After conversion - X_train shape: {X_train.shape}, type: {type(X_train)}")


    sampler = SpikeSlabGibbsSampler(
        X_train, y_train, XA_train, n_programs=hyperparams["d"],
        a=hyperparams.get("a", 0.3), a_prime=hyperparams.get("a_prime", 0.3),
        b_prime=hyperparams.get("b_prime", 0.3), c=hyperparams.get("c", 0.3),
        c_prime=hyperparams.get("c_prime", 0.3), d_prime=hyperparams.get("d_prime", 0.3),
        tau1_sq=hyperparams.get("tau", 1.0), tau0_sq=1e-4,
        pi=0.2, sigma_gamma_sq=hyperparams.get("sigma", 1.0), seed=seed, mask=mask
    )
    sampler.run(max_iters)

    logits_tr = sampler.theta @ sampler.upsilon.T + XA_train @ sampler.gamma.T
    probs_tr = expit(logits_tr)
    theta_val = _fold_in_theta_gibbs(X_val, sampler)
    logits_val = theta_val @ sampler.upsilon.T + XA_val @ sampler.gamma.T
    probs_val = expit(logits_val)
    theta_test = _fold_in_theta_gibbs(X_test, sampler)
    logits_test = theta_test @ sampler.upsilon.T + XA_test @ sampler.gamma.T
    probs_test = expit(logits_test)

    train_metrics = _evaluate_predictions(y_train, probs_tr, return_probs=return_probs)
    val_metrics = _evaluate_predictions(y_val, probs_val, return_probs=return_probs)
    test_metrics = _evaluate_predictions(y_test, probs_test, return_probs=return_probs)

    train_df = pd.DataFrame({
        "sample_id": ids_train,
        "true_label": y_train.reshape(-1),
        "probability": np.round(np.array(probs_tr).reshape(-1), 4),
        "predicted_label": (np.array(probs_tr) >= 0.5).astype(int).reshape(-1),
        "cyto_seed_score": sc_train,
    })

    val_df = pd.DataFrame({
        "sample_id": ids_val,
        "true_label": y_val.reshape(-1),
        "probability": np.round(np.array(probs_val).reshape(-1), 4),
        "predicted_label": (np.array(probs_val) >= 0.5).astype(int).reshape(-1),
        "cyto_seed_score": sc_val,
    })

    test_df = pd.DataFrame({
        "sample_id": ids_test,
        "true_label": y_test.reshape(-1),
        "probability": np.round(np.array(probs_test).reshape(-1), 4),
        "predicted_label": (np.array(probs_test) >= 0.5).astype(int).reshape(-1),
        "cyto_seed_score": sc_test,
    })

    results = {
        "data_info": {
            "n_train": X_train.shape[0],
            "n_val": X_val.shape[0],
            "n_test": X_test.shape[0],
            "p": X_train.shape[1],
            "d": hyperparams["d"],
        },
        "train_metrics": {k: v for k, v in train_metrics.items() if k != "probabilities"},
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "probabilities"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "probabilities"},
        "E_upsilon": sampler.upsilon.tolist(),
        "E_gamma": sampler.gamma.tolist(),
        "train_results_df": train_df,
        "val_results_df": val_df,
        "test_results_df": test_df,
    }

    if return_probs:
        results["train_probabilities"] = train_metrics["probabilities"]
        results["val_probabilities"] = val_metrics["probabilities"]
        results["test_probabilities"] = test_metrics["probabilities"]

    results["E_beta"] = np.exp(sampler.log_beta).tolist()

    if return_params:
        results["delta"] = sampler.delta.tolist()

    return results

def run_all_experiments(datasets, hyperparams_map, output_dir="/labs/Aguiar/SSPA_BRAY/BRay/GibbsResults/unmasked", seed=None, mask=None, max_iter=100, pathway_names=None):
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for dataset_name, (adata, label_col) in datasets.items():
        print(f"\nRunning experiment on dataset {dataset_name}, label={label_col}")
        
        Y = adata.obs[label_col].values.astype(float).reshape(-1,1) 
        X = adata.X  
        var_names = list(adata.var_names)

        x_aux = np.ones((X.shape[0],1)) 
        sample_ids = adata.obs.index.tolist()
        
        log_array_sizes({
            'X': X,
            'Y': Y,
            'x_aux': x_aux
        })
        
        scores = None
        if 'cyto_seed_score' in adata.obs:
            scores = adata.obs['cyto_seed_score'].values
            print(f"Found cyto_seed_score in dataset with mean value: {np.mean(scores):.4f}")

        hyperparams = hyperparams_map[dataset_name].copy()  
        d_values = hyperparams.pop("d")
        for d in d_values:
            print(f"Running with d={d}")
            
            hyperparams["d"] = d
            
            if mask is not None:
                print(f"Using mask with shape: {mask.shape}")
                log_array_sizes({'mask': mask})
            
            clear_memory()

            test_split_size = 0.15
            val_split_size = 0.15
            
            try:
                results = run_sampler_and_evaluate(
                    x_data=X,
                    x_aux=x_aux,
                    y_data=Y,
                    var_names=var_names,
                    hyperparams=hyperparams,
                    seed=seed,
                    test_size=test_split_size,
                    val_size=val_split_size,
                    max_iters=max_iter,
                    return_probs=True,
                    sample_ids=sample_ids,
                    mask=mask,
                    scores=scores,
                    return_params=True
                )

                if "error" in results:
                    print(f"Skipping post-processing for d={d} due to training error.")
                    all_results[f"{dataset_name}_{label_col}_d_{d}"] = results
                    continue 
            
                if "train_results_df" in results:
                    train_csv_path = os.path.join(output_dir, f"{dataset_name}_{label_col}_d_{d}_train_results.csv.gz")
                    val_csv_path = os.path.join(output_dir, f"{dataset_name}_{label_col}_d_{d}_val_results.csv.gz")
                    test_csv_path = os.path.join(output_dir, f"{dataset_name}_{label_col}_d_{d}_test_results.csv.gz")
                    results["train_results_df"].to_csv(train_csv_path, index=False, compression='gzip')
                    if "val_results_df" in results:
                        results["val_results_df"].to_csv(val_csv_path, index=False, compression='gzip')
                    results["test_results_df"].to_csv(test_csv_path, index=False, compression='gzip')
                    
                    train_df = results.pop("train_results_df")
                    val_df = results.pop("val_results_df", None)
                    test_df = results.pop("test_results_df")
                    
                main_results = results.copy()
                if "alpha_beta" in main_results:
                    del main_results["alpha_beta"]
                if "omega_beta" in main_results:
                    del main_results["omega_beta"]
                if "top_genes" in main_results:
                    del main_results["top_genes"]
                
                results_json_path = os.path.join(output_dir, f"{dataset_name}_{label_col}_d_{d}_results_with_scores.json.gz")
                with gzip.open(results_json_path, "wt", encoding="utf-8") as f:
                    json.dump(main_results, f, indent=2)
                    
                if mask is not None:
                    pathway_results = create_pathway_results(results, var_names, mask, pathway_names)
                    pathway_json_path = os.path.join(output_dir, f"{dataset_name}_{label_col}_d_{d}_pathway_results.json.gz")
                    with gzip.open(pathway_json_path, "wt", encoding="utf-8") as f:
                        json.dump(pathway_results, f, indent=2)
                    print(f"Saved pathway-specific results to {pathway_json_path}")
                else:
                    gene_program_results = create_gene_program_results(results, var_names)
                    gp_json_path = os.path.join(output_dir, f"{dataset_name}_{label_col}_d_{d}_gene_program_results.json.gz")
                    with gzip.open(gp_json_path, "wt", encoding="utf-8") as f:
                        json.dump(gene_program_results, f, indent=2)
                    print(f"Saved complete gene program results to {gp_json_path}")

                if "train_results_df" not in results and locals().get('train_df') is not None:
                    results["train_results_df"] = train_df
                    if locals().get('val_df') is not None:
                        results["val_results_df"] = val_df
                    results["test_results_df"] = test_df

                all_results[f"{dataset_name}_{label_col}_d_{d}"] = results
            except Exception as e:
                print(f"--- UNHANDLED EXCEPTION for d={d} ---")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                all_results[f"{dataset_name}_{label_col}_d_{d}"] = {"error": str(e), "status": "crashed"}
            clear_memory()

    return all_results

def create_pathway_results(results, var_names, mask, pathway_names):
    """
    Create a comprehensive dictionary of pathways with their upsilon coefficients
    and all genes in the pathway with their theta weights.
    """
    print("Creating pathway-specific results with upsilon coefficients and gene weights")
    
    # Extract upsilon values (pathway importance for outcome)
    upsilon_values = np.array(results["E_upsilon"])[0]  # Use only first class for simplicity
    
    # Calculate E_beta (gene activity values) from the model parameters
    if "alpha_beta" in results and "omega_beta" in results:
        # Use the model parameters to calculate gene weights
        alpha_beta = np.array(results["alpha_beta"])
        omega_beta = np.array(results["omega_beta"])
        E_beta = alpha_beta / np.maximum(omega_beta, 1e-10)
    else:
        print("Warning: alpha_beta or omega_beta not available in results; trying E_beta directly")
        if "E_beta" in results:
            E_beta = np.array(results["E_beta"])
        else:
            print("Error: Cannot find gene activity values (E_beta). Proceeding without gene activity metrics.")
            E_beta = None
    
    # Create dictionary for results
    pathway_results = {}
    
    # For each pathway, get all genes that are part of it and their weights
    for i, pathway_name in enumerate(pathway_names):
        # Skip if this column in the mask has no genes
        if np.sum(mask[:, i]) == 0:
            continue
            
        # Get all genes in this pathway with their weights
        pathway_genes = []
        gene_indices = np.where(mask[:, i] > 0)[0]
        
        # Get the upsilon value (importance of this pathway for the outcome)
        upsilon_value = float(upsilon_values[i])
        
        # For each gene in the pathway, get its gene name and add to results
        for gene_idx in gene_indices:
            gene_name = var_names[gene_idx]
            gene_info = {
                "gene": gene_name,
                "index": int(gene_idx)
            }
            
            # Add gene activity/contribution metrics if E_beta is available
            if E_beta is not None:
                # Get the activity value for this gene in this pathway
                activity = float(E_beta[gene_idx, i])
                gene_info["activity"] = activity
            
            pathway_genes.append(gene_info)
        
        # Sort genes by activity if activity values are available
        if E_beta is not None:
            pathway_genes.sort(key=lambda x: x.get("activity", 0), reverse=True)
            
            # Add rank information based on sorted activity
            for rank, gene_info in enumerate(pathway_genes):
                gene_info["rank"] = rank + 1
        
        # Add to results dictionary
        pathway_results[pathway_name] = {
            "upsilon": upsilon_value,
            "gene_count": len(gene_indices),
            "genes": pathway_genes
        }
    
    return {
        "num_pathways": len(pathway_results),
        "pathways": pathway_results
    }

def create_gene_program_results(results, var_names):
    """
    Create a comprehensive dictionary of gene programs with all genes and their weights.
    """
    print("Creating complete gene program results with all genes and weights")
    
    # Calculate E_beta from the model parameters
    if "alpha_beta" in results and "omega_beta" in results:
        # Use the model parameters to calculate gene weights
        alpha_beta = np.array(results["alpha_beta"])
        omega_beta = np.array(results["omega_beta"])
        E_beta = alpha_beta / np.maximum(omega_beta, 1e-10)
    else:
        print("Warning: alpha_beta or omega_beta not available in results; trying E_beta directly")
        if "E_beta" in results:
            E_beta = np.array(results["E_beta"])
        else:
            print("Error: Cannot find gene program weights (E_beta). Using top_genes as fallback.")
            return {"gene_programs": results.get("top_genes", {})}
    
    # Extract upsilon values (program importance for outcome)
    upsilon_values = np.array(results["E_upsilon"])[0]  # Use only first class for simplicity
    
    # Create dictionary for results
    gene_program_results = {}
    
    # For each gene program
    for program_idx in range(E_beta.shape[1]):
        program_name = f"program_{program_idx+1}"
        
        # Get upsilon value for this program
        upsilon_value = float(upsilon_values[program_idx])
        
        # Get all genes with their weights for this program
        genes_with_weights = []
        for gene_idx, gene_weight in enumerate(E_beta[:, program_idx]):
            # Only include genes with non-zero weight
            if gene_weight > 1e-10:
                genes_with_weights.append({
                    "gene": var_names[gene_idx],
                    "weight": float(gene_weight),
                    "rank": len(genes_with_weights) + 1
                })
        
        # Sort by weight
        genes_with_weights.sort(key=lambda x: x["weight"], reverse=True)
        
        # Update ranks after sorting
        for i, gene_info in enumerate(genes_with_weights):
            gene_info["rank"] = i + 1
        
        # Add to results dictionary
        gene_program_results[program_name] = {
            "upsilon": upsilon_value,
            "gene_count": len(genes_with_weights),
            "genes": genes_with_weights
        }
    
    return {
        "num_programs": len(gene_program_results),
        "gene_programs": gene_program_results
    }

def run_combined_gp_and_pathway_experiment(dataset_name, adata, label_col, mask, pathway_names, n_gp=500, 
                                  output_dir="/labs/Aguiar/SSPA_BRAY/BRay/GibbsResults/combined",
                                  seed=None, max_iter=100):
    print(f"\nRunning combined pathway+GP experiment on {dataset_name}, label={label_col}, with {n_gp} additional gene programs")
    
    Y = adata.obs[label_col].values.astype(float).reshape(-1,1) 
    X = adata.X  
    var_names = list(adata.var_names)
    
    x_aux = np.ones((X.shape[0],1)) 
    sample_ids = adata.obs.index.tolist()
    
    log_array_sizes({
        'X': X,
        'Y': Y,
        'x_aux': x_aux,
        'mask': mask
    })
    
    scores = None
    if 'cyto_seed_score' in adata.obs:
        scores = adata.obs['cyto_seed_score'].values
        print(f"Found cyto_seed_score in dataset with mean value: {np.mean(scores):.4f}")

    hyperparams = {
        "c_prime": 2.0, "d_prime": 3.0,
        "c":      0.6,
        "a_prime": 2.0, "b_prime": 3.0,
        "a":      0.6,
        "tau":    1.0, "sigma":   1.0
    }
    
    n_pathways = mask.shape[1]
    total_d = n_pathways + n_gp
    hyperparams["d"] = total_d
    
    print(f"Total dimensions: {total_d} = {n_pathways} pathways + {n_gp} gene programs")
    
    n_genes = mask.shape[0]
    extended_mask = np.zeros((n_genes, total_d))
    extended_mask[:, :n_pathways] = mask
    extended_mask[:, n_pathways:] = 1
    
    print(f"Extended mask shape: {extended_mask.shape}")
    print(f"Original mask columns: {n_pathways}, Additional unmasked columns: {n_gp}")
    
    # Use the provided output_dir (timestamped directory) directly
    exp_output_dir = output_dir 
    os.makedirs(exp_output_dir, exist_ok=True) # Ensure it exists, harmless if already there
    
    # Create plot prefix for ELBO plots, ensuring filename uniqueness
    plot_prefix_basename = f"combined_{dataset_name}_pw{n_pathways}_gp{n_gp}"
    plot_prefix = os.path.join(exp_output_dir, plot_prefix_basename)
    
    clear_memory()
    try:
        results = run_sampler_and_evaluate(
            x_data=X,
            x_aux=x_aux,
            y_data=Y,
            var_names=var_names,
            hyperparams=hyperparams,
            seed=seed,
            test_size=0.15,
            val_size=0.15,
            max_iters=max_iter,
            return_probs=True,
            sample_ids=sample_ids,
            mask=extended_mask,
            scores=scores,
            return_params=True
        )
        
        if "train_results_df" in results:
            train_csv_filename = f"{dataset_name}_{label_col}_combined_pw{n_pathways}_gp{n_gp}_train_results.csv.gz"
            val_csv_filename = f"{dataset_name}_{label_col}_combined_pw{n_pathways}_gp{n_gp}_val_results.csv.gz"
            test_csv_filename = f"{dataset_name}_{label_col}_combined_pw{n_pathways}_gp{n_gp}_test_results.csv.gz"
            results["train_results_df"].to_csv(os.path.join(exp_output_dir, train_csv_filename), index=False, compression='gzip')
            if "val_results_df" in results:
                results["val_results_df"].to_csv(os.path.join(exp_output_dir, val_csv_filename), index=False, compression='gzip')
            results["test_results_df"].to_csv(os.path.join(exp_output_dir, test_csv_filename), index=False, compression='gzip')
            
            train_df = results.pop("train_results_df")
            val_df = results.pop("val_results_df", None)
            test_df = results.pop("test_results_df")
        
        main_results = results.copy()
        for large_field in ["alpha_beta", "omega_beta", "E_beta"]:
            if large_field in main_results:
                del main_results[large_field]
        
        results_json_filename = f"{dataset_name}_{label_col}_combined_pw{n_pathways}_gp{n_gp}_results.json.gz"
        out_path = os.path.join(exp_output_dir, results_json_filename)
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(main_results, f, indent=2)
        
        extended_pathway_names = pathway_names.copy() if pathway_names else [f"pathway_{i+1}" for i in range(n_pathways)]
        extended_pathway_names.extend([f"gene_program_{i+1}" for i in range(n_gp)])
        
        combined_analysis_results = create_pathway_results(results, var_names, extended_mask, extended_pathway_names)
        combined_analysis_filename = f"{dataset_name}_{label_col}_combined_pw{n_pathways}_gp{n_gp}_analysis.json.gz" # Renamed for clarity
        combined_analysis_path = os.path.join(exp_output_dir, combined_analysis_filename)
        with gzip.open(combined_analysis_path, "wt", encoding="utf-8") as f:
            json.dump(combined_analysis_results, f, indent=2)
        
        print(f"Saved combined model results to {exp_output_dir}")

        return results

    except Exception as e:
        print(f"--- UNHANDLED EXCEPTION in combined experiment ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        results = {"error": str(e), "status": "crashed"}
        return results

def run_pathway_initialized_experiment(dataset_name, adata, label_col, mask, pathway_names,
                                output_dir="/labs/Aguiar/SSPA_BRAY/BRay/GibbsResults/pathway_initiated",
                                seed=None, max_iter=100):
    """
    Run pathway-initialized experiment.
    Results will be saved directly into the provided output_dir (timestamped directory).
    """
    print(f"\nRunning pathway-initialized experiment on {dataset_name}, label={label_col}")
    print(f"This will initialize gene programs with pathway information, then let them evolve freely")
    
    Y = adata.obs[label_col].values.astype(float).reshape(-1,1) 
    X = adata.X  
    var_names = list(adata.var_names)
    
    x_aux = np.ones((X.shape[0],1)) 
    sample_ids = adata.obs.index.tolist()
    
    log_array_sizes({
        'X': X,
        'Y': Y,
        'x_aux': x_aux,
        'mask': mask
    })
    
    scores = None
    if 'cyto_seed_score' in adata.obs:
        scores = adata.obs['cyto_seed_score'].values
        print(f"Found cyto_seed_score in dataset with mean value: {np.mean(scores):.4f}")

    hyperparams = {
        "c_prime": 2.0, "d_prime": 3.0,
        "c":      0.6,
        "a_prime": 2.0, "b_prime": 3.0,
        "a":      0.6,
        "tau":    1.0, "sigma":   1.0
    }
    
    n_pathways = mask.shape[1]
    hyperparams["d"] = n_pathways
    
    n_genes = mask.shape[0]
    beta_init = np.zeros((n_genes, n_pathways))
    
    for i in range(n_pathways):
        pathway_genes = np.where(mask[:, i] > 0)[0]
        if len(pathway_genes) > 0:
            beta_init[pathway_genes, i] = 1.0
            if seed is not None:
                np.random.seed(seed + i)
            beta_init[pathway_genes, i] += np.random.uniform(0, 0.1, size=len(pathway_genes))
    
    # Use the provided output_dir (timestamped directory) directly
    exp_output_dir = output_dir
    os.makedirs(exp_output_dir, exist_ok=True) # Ensure it exists

    # Create plot prefix for ELBO plots, ensuring filename uniqueness
    plot_prefix_basename = f"initialized_{dataset_name}_pw{n_pathways}"
    plot_prefix = os.path.join(exp_output_dir, plot_prefix_basename)
    
    clear_memory()
    results = run_sampler_and_evaluate(
        x_data=X,
        x_aux=x_aux,
        y_data=Y,
        var_names=var_names,
        hyperparams=hyperparams,
        seed=seed,
        test_size=0.15,
        val_size=0.15,
        max_iters=max_iter,
        return_probs=True,
        sample_ids=sample_ids,
        mask=None,
        scores=scores,
        return_params=True
    )
    
    if "train_results_df" in results:
        train_csv_filename = f"{dataset_name}_{label_col}_initialized_pw{n_pathways}_train_results.csv.gz"
        val_csv_filename = f"{dataset_name}_{label_col}_initialized_pw{n_pathways}_val_results.csv.gz"
        test_csv_filename = f"{dataset_name}_{label_col}_initialized_pw{n_pathways}_test_results.csv.gz"
        results["train_results_df"].to_csv(os.path.join(exp_output_dir, train_csv_filename), index=False, compression='gzip')
        if "val_results_df" in results:
            results["val_results_df"].to_csv(os.path.join(exp_output_dir, val_csv_filename), index=False, compression='gzip')
        results["test_results_df"].to_csv(os.path.join(exp_output_dir, test_csv_filename), index=False, compression='gzip')
        
        train_df = results.pop("train_results_df")
        val_df = results.pop("val_results_df", None)
        test_df = results.pop("test_results_df")
    
    main_results = results.copy()
    for large_field in ["alpha_beta", "omega_beta", "E_beta"]:
        if large_field in main_results:
            del main_results[large_field]
    
    results_json_filename = f"{dataset_name}_{label_col}_initialized_pw{n_pathways}_results.json.gz"
    out_path = os.path.join(exp_output_dir, results_json_filename)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(main_results, f, indent=2)
    
    gene_program_results = create_gene_program_results(results, var_names)
    gp_json_filename = f"{dataset_name}_{label_col}_initialized_pw{n_pathways}_gene_program_results.json.gz"
    gp_path = os.path.join(exp_output_dir, gp_json_filename)
    with gzip.open(gp_path, "wt", encoding="utf-8") as f:
        json.dump(gene_program_results, f, indent=2)
    
    print(f"Saved initialized model results to {exp_output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run experiments with optional mask, custom d, and VI iterations.")
    parser.add_argument("--mask", action="store_true", help="Use mask derived from pathways matrix")
    parser.add_argument("--d", type=int, help="Value of d when mask is not provided")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations for variational inference")
    parser.add_argument("--reduced_pathways", type=int, help="Use only this many pathways from the full set (for testing with mask)")
    
    parser.add_argument("--combined", action="store_true", help="Run combined pathway+gene program configuration")
    parser.add_argument("--n_gp", type=int, default=500, help="Number of gene programs to learn in combined mode")
    parser.add_argument("--initialized", action="store_true", help="Run pathway-initialized unmasked configuration")
    
    parser.add_argument("--max_genes", type=int, help="Use only this many genes (for testing)")
    
    parser.add_argument("--max_samples", type=int, help="Use only this many samples (for testing)")

    parser.add_argument("--profile", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.mask and not args.initialized and args.d is None and not args.combined:
        parser.error("When --mask, --combined, and --initialized flags are not used, --d must be specified.")
    
    # Determine the base output directory (e.g., .../masked, .../unmasked)
    if args.combined:
        base_output_dir_name = "combined"
    elif args.initialized:
        base_output_dir_name = "pathway_initiated"
    elif args.mask:
        base_output_dir_name = "masked"
    else:
        base_output_dir_name = "unmasked"
    
    # Define the root results directory
    root_results_dir = "/labs/Aguiar/SSPA_BRAY/BRay/GibbsResults"
    base_output_dir = os.path.join(root_results_dir, base_output_dir_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create a timestamp-based subdirectory for this specific run
    date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # This run_dir is the single folder where all results for this execution will go.
    run_dir = os.path.join(base_output_dir, date_time_stamp) 
    os.makedirs(run_dir, exist_ok=True)
    
    ajm_ap_samples, ajm_cyto_samples = prepare_ajm_dataset()
    ajm_cyto_filtered = filter_protein_coding_genes(ajm_cyto_samples, gene_annotation)
    
    if args.max_genes and args.max_genes < ajm_cyto_filtered.n_vars:
        print(f"Limiting to {args.max_genes} genes for testing (out of {ajm_cyto_filtered.n_vars})")
        # Select genes with highest variance or expression
        gene_means = np.array(ajm_cyto_filtered.X.mean(axis=0)).flatten()
        top_gene_indices = np.argsort(gene_means)[-args.max_genes:]
        ajm_cyto_filtered = ajm_cyto_filtered[:, top_gene_indices]
        print(f"Filtered dataset shape: {ajm_cyto_filtered.shape}")

    if args.max_samples and args.max_samples < ajm_cyto_filtered.n_obs:
        print(f"Limiting to {args.max_samples} samples for testing (out of {ajm_cyto_filtered.n_obs})")
        cyto_labels = ajm_cyto_filtered.obs['cyto']
        from sklearn.model_selection import train_test_split
        _, sample_subset, _, _ = train_test_split(
            np.arange(ajm_cyto_filtered.n_obs), 
            cyto_labels,
            test_size=args.max_samples/ajm_cyto_filtered.n_obs,
            stratify=cyto_labels,
            random_state=42
        )
        ajm_cyto_filtered = ajm_cyto_filtered[sample_subset]
        print(f"Filtered dataset shape: {ajm_cyto_filtered.shape}")

    del ajm_ap_samples
    del ajm_cyto_samples
    clear_memory()
    
    common_genes = np.intersect1d(ajm_cyto_filtered.var_names, CYTOSEED_ensembl)
    print(f"Found {len(common_genes)} common genes between dataset and CYTOSEED_ensembl")
    cyto_seed_mask = np.array([gene in CYTOSEED_ensembl for gene in ajm_cyto_filtered.var_names])
    cyto_seed_scores = ajm_cyto_filtered.X[:, cyto_seed_mask].sum(axis=1)
    ajm_cyto_filtered.obs['cyto_seed_score'] = cyto_seed_scores
    
    mask_array = None # Renamed from mask to avoid conflict with args.mask
    pathway_names_list = None # Renamed from pathway_names
    
    if args.mask or args.combined or args.initialized:
        gene_names = list(ajm_cyto_filtered.var_names)
        
        if args.reduced_pathways and args.reduced_pathways > 0:
            if args.reduced_pathways >= len(pathways):
                print(f"Warning: Requested {args.reduced_pathways} pathways but only {len(pathways)} are available. Using all pathways.")
                pathway_names_list = list(pathways.keys())
            else:
                print(f"Using a reduced set of {args.reduced_pathways} pathways out of {len(pathways)} total pathways")
                random.seed(42)
                pathway_names_list = random.sample(list(pathways.keys()), args.reduced_pathways)
                print(f"Selected {len(pathway_names_list)} pathways randomly")
        else:
            pathway_names_list = list(pathways.keys())
        
        print(f"Number of genes: {len(gene_names)}")
        print(f"Number of pathways: {len(pathway_names_list)}")
        
        M = pd.DataFrame(0, index=gene_names, columns=pathway_names_list)
        
        print("Filling matrix M...")
        chunk_size = 100
        total_chunks = (len(pathway_names_list) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(pathway_names_list))
            current_pathways = pathway_names_list[start_idx:end_idx]
            
            print(f"Processing pathway chunk {chunk_idx+1}/{total_chunks}, pathways {start_idx} to {end_idx}")
            
            for pathway in current_pathways:
                gene_list = pathways[pathway]
                for gene in gene_list:
                    if gene in M.index:
                        M.loc[gene, pathway] = 1
        
        print(f"Matrix M created with shape {M.shape}")
        log_array_sizes({'M': M.values, 'ajm_cyto_filtered.X': ajm_cyto_filtered.X})
        
        mask_array = M.values
        
        print(f"Mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
        non_zero_entries = np.count_nonzero(mask_array)
        total_entries = mask_array.size
        sparsity = 100 * (1 - non_zero_entries / total_entries)
        print(f"Mask sparsity: {sparsity:.2f}% ({non_zero_entries} non-zero entries out of {total_entries})")
        
        del M
        clear_memory()
    
    hyperparams_cyto = {
        "c_prime": 2.0,  "d_prime": 3.0,
        "c":      0.6,
        "a_prime":2.0,   "b_prime": 3.0,
        "a":      0.6,
        "tau":    1.0,   "sigma":   1.0,
    }
    
    if args.mask:
        hyperparams_cyto["d"] = [mask_array.shape[1]]
        print(f"Using mask-based d value: {mask_array.shape[1]}")
    elif args.combined:
        # d value set within run_combined_gp_and_pathway_experiment
        print(f"Combined config will use total d = {mask_array.shape[1]} + {args.n_gp}")
        # hyperparams_cyto["d"] is not directly used by run_combined_gp_and_pathway_experiment for 'd'
    elif args.initialized:
        # d value set within run_pathway_initialized_experiment
        print(f"Initialized config will use d = {mask_array.shape[1]}")
    else: # Regular unmasked gene program mode
        hyperparams_cyto["d"] = [args.d]
        print(f"Using specified d value: {args.d}")
    
    hyperparams_map = {
        "ajm_cyto": hyperparams_cyto,
    }

    datasets = {
        "ajm_cyto": (ajm_cyto_filtered, "cyto"),
    }

    all_results = {}
    
    if args.combined:
        print("\nRunning COMBINED PATHWAY + GENE PROGRAM configuration:")
        print(f"This will use {mask_array.shape[1]} pathway dimensions plus {args.n_gp} freely learned gene program dimensions")
        dataset_name = "ajm_cyto"
        adata, label_col = datasets[dataset_name]
        combined_results = run_combined_gp_and_pathway_experiment(
            dataset_name, adata, label_col, mask_array, pathway_names_list,
            n_gp=args.n_gp, output_dir=run_dir, # Pass run_dir as output_dir
            seed=None, max_iter=args.max_iter
        )
        # Add debugging to see what's returned
        print(f"DEBUG: Combined results keys: {list(combined_results.keys()) if combined_results else 'None'}")
        if combined_results:
            for key, value in combined_results.items():
                if isinstance(value, dict):
                    print(f"DEBUG: {key} contains: {list(value.keys())}")
        
        all_results["combined_config"] = combined_results
    
    elif args.initialized:
        print("\nRunning PATHWAY-INITIALIZED configuration:")
        print(f"This will initialize {mask_array.shape[1]} gene programs using pathway information, then let them evolve freely")
        dataset_name = "ajm_cyto"
        adata, label_col = datasets[dataset_name]
        initialized_results = run_pathway_initialized_experiment(
            dataset_name, adata, label_col, mask_array, pathway_names_list,
            output_dir=run_dir, seed=None, max_iter=args.max_iter # Pass run_dir
        )
        all_results["initialized_config"] = initialized_results
    
    else: # Standard masked or unmasked configuration
        print("\nRunning standard configuration:")
        current_mask_for_run = mask_array if args.mask else None
        if args.mask:
            print(f"Using MASKED configuration with {mask_array.shape[1]} pathways")
        else:
            print(f"Using UNMASKED configuration with {args.d} gene programs")

        std_results = run_all_experiments(
            datasets, hyperparams_map, output_dir=run_dir, # Pass run_dir
            seed=None, mask=current_mask_for_run, max_iter=args.max_iter,
            pathway_names=pathway_names_list
        )
        all_results.update(std_results)

    print("\nAll experiments completed!")
    print(f"Results saved to: {run_dir}") # Print the timestamped directory

    print("\nSummary of results:")
    print("-" * 80)
    print(f"{'Experiment':<30} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Train F1':<10} {'Val F1':<10} {'Test F1':<10}")
    print("-" * 60)
    for exp_name, res in all_results.items():
        if res and 'train_metrics' in res and 'test_metrics' in res and 'val_metrics' in res: 
            train_acc = res['train_metrics']['accuracy']
            val_acc   = res['val_metrics']['accuracy']
            test_acc  = res['test_metrics']['accuracy']
            train_f1  = res['train_metrics']['f1']
            val_f1    = res['val_metrics']['f1']
            test_f1   = res['test_metrics']['f1']
            print(f"{exp_name:<30} {train_acc:<10.4f} {val_acc:<10.4f} {test_acc:<10.4f} {train_f1:<10.4f} {val_f1:<10} {test_f1:<10.4f}")
        else:
            print(f"{exp_name:<30} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")


if __name__ == "__main__":
    import sys
    import cProfile
    import pstats
    import argparse as _argparse

    parser = _argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true', help='Run with cProfile and save output to profile_output.prof')
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown 

    if args.profile:
        print("Profiling run_experiments.py with cProfile...")
        profile_output = "profile_output.prof"
        cProfile.run('main()', profile_output)
        print(f"Profile data saved to {profile_output}. You can analyze it with 'snakeviz {profile_output}' or 'python -m pstats {profile_output}'")
    else:
        main()

