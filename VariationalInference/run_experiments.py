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

from vi_model_complete import run_model_and_evaluate
from data import *


from sklearn.model_selection import train_test_split

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

def run_all_experiments(datasets, hyperparams_map, output_dir="/labs/Aguiar/SSPA_BRAY/BRay/Results/unmasked", seed=None, mask=None, max_iter=100, pathway_names=None):
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for dataset_name, (adata, label_col) in datasets.items():
        print(f"\nRunning experiment on dataset {dataset_name}, label={label_col}")
        
        if dataset_name == "emtab":
            # For EMTAB, Y contains both Crohn's disease and ulcerative colitis columns
            if label_col == "both_labels":
                Y = adata.obs[['Crohn\'s disease', 'ulcerative colitis']].values.astype(float)
            else:
                # Use specific label column if provided
                Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
            X = adata.X
            var_names = list(adata.var_names)
            
            # For EMTAB, x_aux includes both age and sex_female columns
            x_aux = adata.obs[['age', 'sex_female']].values.astype(float)
            sample_ids = adata.obs.index.tolist()
        else:
            # For AJM datasets (ajm_cyto, ajm_ap), use original logic
            Y = adata.obs[label_col].values.astype(float).reshape(-1,1) 
            X = adata.X  
            var_names = list(adata.var_names)

            x_aux = np.ones((X.shape[0],1)) 
            sample_ids = adata.obs.index.tolist()
        
        scores = None
        if 'cyto_seed_score' in adata.obs:
            scores = adata.obs['cyto_seed_score'].values
            print(f"Found cyto_seed_score in dataset with mean value: {np.mean(scores):.4f}")

        hyperparams = hyperparams_map[dataset_name].copy()  
        d_values = hyperparams.pop("d")
        # Convert to list if it's a single value
        if not isinstance(d_values, list):
            d_values = [d_values]
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
                results = run_model_and_evaluate(
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
                                  output_dir="/labs/Aguiar/SSPA_BRAY/BRay/Results/combined",
                                  seed=None, max_iter=100):
    print(f"\nRunning combined pathway+GP experiment on {dataset_name}, label={label_col}, with {n_gp} additional gene programs")
    
    # Prepare label matrix (Y) and auxiliary matrix (x_aux) depending on the
    # dataset and requested label column.  The EMTAB dataset stores Crohn's
    # disease and ulcerative colitis as separate columns.  When the caller
    # passes "both_labels" we interpret that as using both columns together.
    if dataset_name == "emtab" and label_col == "both_labels":
        Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
        x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
    else:
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        x_aux = np.ones((adata.shape[0], 1))

    X = adata.X
    var_names = list(adata.var_names)
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
        "alpha_eta": 2.0, "lambda_eta": 3.0,
        "alpha_beta": 0.6,
        "alpha_xi": 2.0, "lambda_xi": 3.0,
        "alpha_theta": 0.6,
        "sigma2_v": 1.0, "sigma2_gamma": 1.0
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
        results = run_model_and_evaluate(
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
            return_params=True,
            plot_elbo=True,
            plot_prefix=plot_prefix
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

    except Exception as e:
        print(f"--- UNHANDLED EXCEPTION in combined experiment ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        results = {"error": str(e), "status": "crashed"}
        
        return results

def run_pathway_initialized_experiment(dataset_name, adata, label_col, mask, pathway_names,
                                output_dir="/labs/Aguiar/SSPA_BRAY/BRay/Results/pathway_initiated",
                                seed=None, max_iter=100):
    """
    Run pathway-initialized experiment.
    Results will be saved directly into the provided output_dir (timestamped directory).
    """
    print(f"\nRunning pathway-initialized experiment on {dataset_name}, label={label_col}")
    print(f"This will initialize gene programs with pathway information, then let them evolve freely")
    
    # Prepare Y and x_aux similarly to the combined experiment.  When using the
    # EMTAB dataset the "both_labels" flag indicates that both Crohn's disease
    # and ulcerative colitis columns should be used simultaneously.  In that
    # case we also include age and sex information as auxiliary covariates.
    if dataset_name == "emtab" and label_col == "both_labels":
        Y = adata.obs[["Crohn's disease", "ulcerative colitis"]].values.astype(float)
        x_aux = adata.obs[["age", "sex_female"]].values.astype(float)
    else:
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        x_aux = np.ones((adata.shape[0], 1))

    X = adata.X
    var_names = list(adata.var_names)
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
        "alpha_eta": 2.0, "lambda_eta": 3.0,
        "alpha_beta": 0.6,
        "alpha_xi": 2.0, "lambda_xi": 3.0,
        "alpha_theta": 0.6,
        "sigma2_v": 1.0, "sigma2_gamma": 1.0
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
    results = run_model_and_evaluate(
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
        return_params=True,
        plot_elbo=True,
        plot_prefix=plot_prefix,
        beta_init=beta_init
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
    parser.add_argument("--dataset", type=str, default="cyto", choices=["cyto", "ap", "emtab"], help="Which dataset to use: cyto, ap, or emtab")
    parser.add_argument("--label", type=str, help="Label column to use (for EMTAB dataset)")
    parser.add_argument("--profile", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.mask and not args.initialized and args.d is None and not args.combined and args.dataset != "emtab":
        parser.error("When --mask, --combined, and --initialized flags are not used, --d must be specified (except for emtab).")

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
    root_results_dir = "/labs/Aguiar/SSPA_BRAY/BRay/Results"
    base_output_dir = os.path.join(root_results_dir, base_output_dir_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create a timestamp-based subdirectory for this specific run
    date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, date_time_stamp) 
    os.makedirs(run_dir, exist_ok=True)

    # --- DATASET LOADING LOGIC ---
    datasets = {}
    hyperparams_map = {}
    mask_array = None
    pathway_names_list = None

    if args.dataset == "emtab":
        # Load EMTAB data from emtab_data.py (already imported above)
        emtab_data = prepare_and_load_emtab()
        # For EMTAB, use both label columns by default
        label_col = args.label if args.label else "both_labels"
        print(f"Loaded EMTAB data with shape {emtab_data.shape} and label column '{label_col}'")
        datasets["emtab"] = (emtab_data, label_col)
        
        # Pathway mask logic for EMTAB (same as AJM datasets)
        if args.mask or args.combined or args.initialized:
            gene_names = list(emtab_data.var_names)
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
            log_array_sizes({'M': M.values, 'emtab_data.X': emtab_data.X})
            mask_array = M.values
            print(f"Mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
            non_zero_entries = np.count_nonzero(mask_array)
            total_entries = mask_array.size
            sparsity = 100 * (1 - non_zero_entries / total_entries)
            print(f"Mask sparsity: {sparsity:.2f}% ({non_zero_entries} non-zero entries out of {total_entries})")
            del M
            clear_memory()
        
        # Set up hyperparams for EMTAB
        hyperparams_emtab = {
            "alpha_eta": 2.0,  "lambda_eta": 3.0,
            "alpha_beta": 0.6,
            "alpha_xi": 2.0,   "lambda_xi": 3.0,
            "alpha_theta": 0.6,
            "sigma2_v": 1.0,   "sigma2_gamma": 1.0,
        }
        if args.mask:
            hyperparams_emtab["d"] = mask_array.shape[1]
            print(f"Using mask-based d value: {mask_array.shape[1]}")
        elif args.combined:
            print(f"Combined config will use total d = {mask_array.shape[1]} + {args.n_gp}")
        elif args.initialized:
            print(f"Initialized config will use d = {mask_array.shape[1]}")
        else:
            if args.d is not None:
                hyperparams_emtab["d"] = args.d
                print(f"Using specified d value: {args.d}")
            else:
                hyperparams_emtab["d"] = 50
                print(f"Using default d value: 50")
        hyperparams_map = {"emtab": hyperparams_emtab}

    else:
        # Load AJM data (cyto or ap)
        ajm_ap_samples, ajm_cyto_samples = prepare_ajm_dataset()
        if args.dataset == "cyto":
            ajm_cyto_filtered = filter_protein_coding_genes(ajm_cyto_samples, gene_annotation)
            adata = ajm_cyto_filtered
            label_col = "cyto"
            del ajm_ap_samples
            del ajm_cyto_samples
        elif args.dataset == "ap":
            ajm_ap_filtered = filter_protein_coding_genes(ajm_ap_samples, gene_annotation)
            adata = ajm_ap_filtered
            label_col = "ap"
            del ajm_ap_samples
            del ajm_cyto_samples
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        clear_memory()

        # For cyto, add cyto_seed_score
        if args.dataset == "cyto":
            common_genes = np.intersect1d(adata.var_names, CYTOSEED_ensembl)
            print(f"Found {len(common_genes)} common genes between dataset and CYTOSEED_ensembl")
            cyto_seed_mask = np.array([gene in CYTOSEED_ensembl for gene in adata.var_names])
            cyto_seed_scores = adata.X[:, cyto_seed_mask].sum(axis=1)
            adata.obs['cyto_seed_score'] = cyto_seed_scores

        # Pathway mask logic (for cyto only)
        if args.mask or args.combined or args.initialized:
            gene_names = list(adata.var_names)
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
            log_array_sizes({'M': M.values, 'adata.X': adata.X})
            mask_array = M.values
            print(f"Mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
            non_zero_entries = np.count_nonzero(mask_array)
            total_entries = mask_array.size
            sparsity = 100 * (1 - non_zero_entries / total_entries)
            print(f"Mask sparsity: {sparsity:.2f}% ({non_zero_entries} non-zero entries out of {total_entries})")
            del M
            clear_memory()
        # Set up hyperparams for cyto/ap
        hyperparams_cyto = {
            "alpha_eta": 2.0,  "lambda_eta": 3.0,
            "alpha_beta": 0.6,
            "alpha_xi": 2.0,   "lambda_xi": 3.0,
            "alpha_theta": 0.6,
            "sigma2_v": 1.0,   "sigma2_gamma":   1.0,
        }
        if args.mask:
            hyperparams_cyto["d"] = mask_array.shape[1]
            print(f"Using mask-based d value: {mask_array.shape[1]}")
        elif args.combined:
            print(f"Combined config will use total d = {mask_array.shape[1]} + {args.n_gp}")
        elif args.initialized:
            print(f"Initialized config will use d = {mask_array.shape[1]}")
        else:
            if args.d is not None:
                hyperparams_cyto["d"] = args.d
                print(f"Using specified d value: {args.d}")
            else:
                hyperparams_cyto["d"] = 50
                print(f"Using default d value: 50")
        dataset_key = "ajm_cyto" if args.dataset == "cyto" else "ajm_ap"
        hyperparams_map = {dataset_key: hyperparams_cyto}
        datasets = {dataset_key: (adata, label_col)}

    all_results = {}

    # --- EXPERIMENT RUNNING LOGIC ---
    if args.combined:
        print("\nRunning COMBINED PATHWAY + GENE PROGRAM configuration:")
        print(f"This will use {mask_array.shape[1]} pathway dimensions plus {args.n_gp} freely learned gene program dimensions")
        dataset_name = list(datasets.keys())[0]
        adata, label_col = datasets[dataset_name]
        combined_results = run_combined_gp_and_pathway_experiment(
            dataset_name, adata, label_col, mask_array, pathway_names_list,
            n_gp=args.n_gp, output_dir=run_dir,
            seed=None, max_iter=args.max_iter
        )
        all_results["combined_config"] = combined_results
    
    elif args.initialized:
        print("\nRunning PATHWAY-INITIALIZED configuration:")
        print(f"This will initialize {mask_array.shape[1]} gene programs using pathway information, then let them evolve freely")
        dataset_name = list(datasets.keys())[0]
        adata, label_col = datasets[dataset_name]
        initialized_results = run_pathway_initialized_experiment(
            dataset_name, adata, label_col, mask_array, pathway_names_list,
            output_dir=run_dir, seed=None, max_iter=args.max_iter
        )
        all_results["initialized_config"] = initialized_results
    
    else: # Standard masked or unmasked configuration
        print("\nRunning standard configuration:")
        current_mask_for_run = mask_array if args.mask else None
        
        if args.mask:
            print(f"Using MASKED configuration with {mask_array.shape[1]} pathways")
        else:
            dval = hyperparams_map[list(datasets.keys())[0]]["d"]
            if isinstance(dval, list):
                dval = dval[0]
            print(f"Using UNMASKED configuration with {dval} gene programs")

        std_results = run_all_experiments(
            datasets, hyperparams_map, output_dir=run_dir,
            seed=None, mask=current_mask_for_run, max_iter=args.max_iter,
            pathway_names=pathway_names_list
        )
        all_results.update(std_results)

    print("\nAll experiments completed!")
    print(f"Results saved to: {run_dir}")

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