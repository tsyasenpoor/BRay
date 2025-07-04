import numpy as np
import pandas as pd
import argparse
import os
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.special import expit
import scipy.sparse as sp

from gibbs import SpikeSlabGibbsSampler
from data import (
    prepare_ajm_dataset, 
    prepare_and_load_emtab, 
    filter_protein_coding_genes,
    gene_annotation,
    pathways,
    CYTOSEED_ensembl
)
from memory_tracking import log_memory, clear_memory
from convergence_checking import ConvergenceMonitor, print_convergence_diagnostics


def split_data(X, Y, X_aux, train_ratio=0.6, val_ratio=0.15, test_ratio=0.25, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Gene expression data
    Y : array-like, shape (n_samples, n_labels)
        Labels
    X_aux : array-like, shape (n_samples, n_aux_features)
        Auxiliary features
    train_ratio : float, default=0.6
        Proportion of data for training
    val_ratio : float, default=0.15
        Proportion of data for validation
    test_ratio : float, default=0.25
        Proportion of data for testing
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple of arrays: (X_train, X_val, X_test, Y_train, Y_val, Y_test, 
                     X_aux_train, X_aux_val, X_aux_test, 
                     train_idx, val_idx, test_idx)
    """
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=Y[:, 0] if Y.shape[1] == 1 else None  # Only stratify for single label
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=Y[temp_idx, 0] if Y.shape[1] == 1 else None
    )
    
    # Create splits
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]
    X_aux_train, X_aux_val, X_aux_test = X_aux[train_idx], X_aux[val_idx], X_aux[test_idx]
    
    print(f"Data split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return (X_train, X_val, X_test, Y_train, Y_val, Y_test, 
            X_aux_train, X_aux_val, X_aux_test, 
            train_idx, val_idx, test_idx)


def create_pathway_mask(pathways_dict, gene_names):
    """
    Create a binary mask matrix for pathways.
    
    Parameters:
    -----------
    pathways_dict : dict
        Dictionary mapping pathway names to gene lists
    gene_names : array-like
        List of gene names in the dataset
        
    Returns:
    --------
    mask : array, shape (n_pathways, n_genes)
        Binary mask matrix where mask[i,j] = 1 if gene j is in pathway i
    pathway_names : list
        List of pathway names corresponding to mask rows
    """
    
    gene_names = np.array(gene_names)
    pathway_names = list(pathways_dict.keys())
    n_pathways = len(pathway_names)
    n_genes = len(gene_names)
    
    mask = np.zeros((n_pathways, n_genes), dtype=int)
    
    for i, pathway_name in enumerate(pathway_names):
        pathway_genes = pathways_dict[pathway_name]
        # Find indices of genes that are in this pathway
        gene_indices = np.isin(gene_names, pathway_genes)
        mask[i, gene_indices] = 1
    
    print(f"Created pathway mask: {n_pathways} pathways x {n_genes} genes")
    print(f"Mask sparsity: {np.sum(mask)} / {mask.size} = {np.sum(mask)/mask.size:.4f}")
    
    return mask, pathway_names


def initialize_beta_with_pathways(pathways_dict, gene_names, n_programs, rng):
    """
    Initialize beta matrix with pathway information.
    
    Parameters:
    -----------
    pathways_dict : dict
        Dictionary mapping pathway names to gene lists
    gene_names : array-like
        List of gene names in the dataset
    n_programs : int
        Number of gene programs (should equal number of pathways)
    rng : numpy.random.Generator
        Random number generator
        
    Returns:
    --------
    beta_init : array, shape (n_genes, n_programs)
        Initialized beta matrix
    """
    
    n_genes = len(gene_names)
    gene_names = np.array(gene_names)
    pathway_names = list(pathways_dict.keys())
    
    # Initialize with small random values
    beta_init = rng.gamma(0.1, scale=0.1, size=(n_genes, n_programs))
    
    # Set higher values for genes in corresponding pathways
    for i, pathway_name in enumerate(pathway_names[:n_programs]):
        pathway_genes = pathways_dict[pathway_name]
        gene_indices = np.isin(gene_names, pathway_genes)
        # Set higher initial values for genes in this pathway
        beta_init[gene_indices, i] = rng.gamma(1.0, scale=0.5, size=np.sum(gene_indices))
    
    return beta_init


def infer_theta_for_test(X_test, Y_test, X_aux_test, beta, gamma, upsilon, 
                        sampler_params, n_samples=100, burn_in=50):
    """
    Infer theta for test/validation samples given fixed beta, gamma, and upsilon.
    
    Parameters:
    -----------
    X_test : array, shape (n_test_samples, n_genes)
        Test gene expression data
    Y_test : array, shape (n_test_samples, n_labels)
        Test labels
    X_aux_test : array, shape (n_test_samples, n_aux_features)
        Test auxiliary features
    beta : array, shape (n_genes, n_programs)
        Fixed beta matrix from training
    gamma : array, shape (n_labels, n_aux_features)
        Fixed gamma matrix from training
    upsilon : array, shape (n_labels, n_programs)
        Fixed upsilon matrix from training
    sampler_params : dict
        Dictionary of sampler hyperparameters
    n_samples : int, default=100
        Number of samples to draw
    burn_in : int, default=50
        Number of burn-in samples
        
    Returns:
    --------
    theta_samples : array, shape (n_test_samples, n_programs)
        Inferred theta values (average of effective samples)
    """
    
    n_test_samples, n_genes = X_test.shape
    n_programs = beta.shape[1]
    
    print(f"Inferring theta for {X_test.shape[0]} samples...")
    
    # Create a temporary sampler for theta inference
    temp_sampler = SpikeSlabGibbsSampler(
        X_test, Y_test, X_aux_test, n_programs,
        alpha_theta=sampler_params['alpha_theta'],
        alpha_beta=sampler_params['alpha_beta'],
        alpha_xi=sampler_params['alpha_xi'],
        alpha_eta=sampler_params['alpha_eta'],
        lambda_xi=sampler_params['lambda_xi'],
        lambda_eta=sampler_params['lambda_eta'],
        pi_upsilon=sampler_params['pi_upsilon'],
        sigma_slab_sq=sampler_params['sigma_slab_sq'],
        sigma_gamma_sq=sampler_params['sigma_gamma_sq'],
        seed=sampler_params.get('seed', 42)
    )
    
    # Set fixed parameters
    temp_sampler.beta = beta.copy()
    temp_sampler.gamma = gamma.copy()
    temp_sampler.upsilon = upsilon.copy()
    
    # Store theta samples - use more memory efficient approach
    theta_samples = []
    max_stored_samples = min(50, n_samples - burn_in)  # Limit stored samples
    
    # MCMC sampling for theta only
    for iteration in range(n_samples):
        # Update latent counts z
        temp_sampler._update_latent_counts_z()
        
        # Update xi (hyperparameter for theta)
        temp_sampler._update_xi()
        
        # Update theta
        temp_sampler._update_theta()
        
        # Store samples after burn-in (with memory limit)
        if iteration >= burn_in:
            theta_samples.append(temp_sampler.theta.copy())
            
            # Keep only recent samples to limit memory
            if len(theta_samples) > max_stored_samples:
                theta_samples.pop(0)  # Remove oldest sample
        
        # Periodic cleanup during long runs
        if iteration % 20 == 0:
            temp_sampler.cleanup_memory()
    
    # Average over effective samples
    theta_mean = np.mean(theta_samples, axis=0)
    
    # Aggressive cleanup
    temp_sampler.cleanup_memory()
    del temp_sampler
    del theta_samples
    
    # Force garbage collection and JAX cache clearing
    import gc
    import jax
    try:
        jax.clear_caches()
    except:
        pass
    gc.collect()
    
    print(f"Theta inference completed, shape: {theta_mean.shape}")
    
    return theta_mean


def calculate_cyto_seed_score(X, gene_names, cyto_seed_genes):
    """
    Calculate cytokine seed score for samples.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_genes)
        Gene expression data
    gene_names : array-like
        List of gene names
    cyto_seed_genes : list
        List of cytokine seed genes
        
    Returns:
    --------
    scores : array, shape (n_samples,)
        Cytokine seed scores
    """
    
    gene_names = np.array(gene_names)
    cyto_indices = np.isin(gene_names, cyto_seed_genes)
    
    if np.sum(cyto_indices) == 0:
        print("Warning: No cytokine seed genes found in dataset")
        return np.zeros(X.shape[0])
    
    # Calculate mean expression of cytokine seed genes
    cyto_expression = X[:, cyto_indices]
    scores = np.mean(cyto_expression, axis=1)
    
    return scores


def evaluate_predictions(Y_true, Y_pred_proba, Y_pred_labels):
    """
    Evaluate predictions using various metrics.
    
    Parameters:
    -----------
    Y_true : array, shape (n_samples, n_labels)
        True labels
    Y_pred_proba : array, shape (n_samples, n_labels)
        Predicted probabilities
    Y_pred_labels : array, shape (n_samples, n_labels)
        Predicted labels
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    
    metrics = {}
    n_labels = Y_true.shape[1]
    
    for k in range(n_labels):
        label_metrics = {}
        
        # Calculate metrics for each label
        try:
            label_metrics['auc'] = roc_auc_score(Y_true[:, k], Y_pred_proba[:, k])
        except:
            label_metrics['auc'] = np.nan
            
        label_metrics['accuracy'] = accuracy_score(Y_true[:, k], Y_pred_labels[:, k])
        label_metrics['precision'] = precision_score(Y_true[:, k], Y_pred_labels[:, k], zero_division=0)
        label_metrics['recall'] = recall_score(Y_true[:, k], Y_pred_labels[:, k], zero_division=0)
        label_metrics['f1'] = f1_score(Y_true[:, k], Y_pred_labels[:, k], zero_division=0)
        
        metrics[f'label_{k}'] = label_metrics
    
    return metrics


def save_results(results_dict, output_dir, experiment_name):
    """
    Save experiment results to files.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing all results
    output_dir : str
        Output directory path
    experiment_name : str
        Name of the experiment
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results as pickle
    results_file = os.path.join(output_dir, f"{experiment_name}_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f)
    
    # Save prediction results as gzipped CSV files
    for split in ['train', 'val', 'test']:
        if f'{split}_predictions' in results_dict:
            pred_data = results_dict[f'{split}_predictions']
            
            # Create DataFrame from predictions
            pred_df_data = {}
            
            # Add true labels
            if 'true_labels' in pred_data and pred_data['true_labels'] is not None:
                true_labels = pred_data['true_labels']
                if len(true_labels.shape) == 1:
                    pred_df_data['true_label_0'] = true_labels
                else:
                    for i in range(true_labels.shape[1]):
                        pred_df_data[f'true_label_{i}'] = true_labels[:, i]
            
            # Add predicted labels  
            if 'predicted_labels' in pred_data and pred_data['predicted_labels'] is not None:
                pred_labels = pred_data['predicted_labels']
                if len(pred_labels.shape) == 1:
                    pred_df_data['predicted_label_0'] = pred_labels
                else:
                    for i in range(pred_labels.shape[1]):
                        pred_df_data[f'predicted_label_{i}'] = pred_labels[:, i]
            
            # Add predicted probabilities
            if 'predicted_probabilities' in pred_data and pred_data['predicted_probabilities'] is not None:
                pred_probs = pred_data['predicted_probabilities']
                if len(pred_probs.shape) == 1:
                    pred_df_data['predicted_probability_0'] = pred_probs
                else:
                    for i in range(pred_probs.shape[1]):
                        pred_df_data[f'predicted_probability_{i}'] = pred_probs[:, i]
            
            # Add cytokine seed scores if available
            if 'cyto_seed_scores' in pred_data and pred_data['cyto_seed_scores'] is not None:
                pred_df_data['cyto_seed_score'] = pred_data['cyto_seed_scores']
            
            # Create DataFrame and save as gzipped CSV
            pred_df = pd.DataFrame(pred_df_data)
            pred_file = os.path.join(output_dir, f"{experiment_name}_{split}_predictions.csv.gz")
            pred_df.to_csv(pred_file, index=False, compression='gzip')
            print(f"Saved {split} predictions to {pred_file} (gzipped)")
    
    # Save DRGP matrix as gzipped CSV
    if 'drgp_matrix' in results_dict:
        drgp_file = os.path.join(output_dir, f"{experiment_name}_drgp_matrix.csv.gz")
        results_dict['drgp_matrix'].to_csv(drgp_file, index=False, compression='gzip')
        print(f"Saved DRGP matrix to {drgp_file} (gzipped)")
    
    print(f"All results saved to {output_dir}")


def extract_drgp_matrix(beta, upsilon, gene_names, pathway_names=None):
    """
    Extract DRGP (Data-driven Regulatory Gene Programs) matrix.
    
    Parameters:
    -----------
    beta : array, shape (n_genes, n_programs)
        Beta matrix from the model
    upsilon : array, shape (n_labels, n_programs)
        Upsilon matrix from the model
    gene_names : array-like
        List of gene names
    pathway_names : list, optional
        List of pathway names if using pathways
        
    Returns:
    --------
    drgp_df : DataFrame
        DataFrame containing gene programs with their genes and activities
    """
    
    n_genes, n_programs = beta.shape
    n_labels = upsilon.shape[0]
    
    # Calculate program importance (sum of absolute upsilon values across labels)
    program_importance = np.sum(np.abs(upsilon), axis=0)
    
    # Sort programs by importance
    sorted_program_indices = np.argsort(program_importance)[::-1]
    
    drgp_data = []
    
    for rank, program_idx in enumerate(sorted_program_indices):
        program_name = f"Program_{program_idx}"
        if pathway_names and program_idx < len(pathway_names):
            program_name = pathway_names[program_idx]
        
        # Get top genes for this program
        gene_activities = beta[:, program_idx]
        top_gene_indices = np.argsort(gene_activities)[::-1]
        
        # Get upsilon values for each label
        upsilon_values = upsilon[:, program_idx]
        
        program_data = {
            'program_rank': rank + 1,
            'program_name': program_name,
            'program_index': int(program_idx),
            'program_importance': float(program_importance[program_idx])
        }
        
        # Add upsilon values for each label
        for label_idx in range(n_labels):
            program_data[f'upsilon_label_{label_idx}'] = float(upsilon_values[label_idx])
        
        # Add top genes and their activities
        for gene_rank, gene_idx in enumerate(top_gene_indices):
            gene_data = program_data.copy()
            gene_data.update({
                'gene_name': gene_names[gene_idx],
                'gene_rank': gene_rank + 1,
                'gene_activity': float(gene_activities[gene_idx])
            })
            drgp_data.append(gene_data)
    
    drgp_df = pd.DataFrame(drgp_data)
    return drgp_df


def run_sampler_and_evaluate(X, Y, X_aux, n_programs, configuration='unmasked',
                           pathways_dict=None, gene_names=None, cyto_seed_genes=None,
                           max_iters=1000, burn_in=500, n_theta_samples=100,
                           output_dir='results', experiment_name=None,
                           random_state=42, 
                           n_chains=4, check_convergence=True,
                           convergence_check_interval=100, convergence_patience=3,
                           rhat_threshold=1.1, ess_threshold=400,
                           **sampler_kwargs):
    """
    Run the complete training and evaluation pipeline.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_genes)
        Gene expression data
    Y : array, shape (n_samples, n_labels)
        Labels
    X_aux : array, shape (n_samples, n_aux_features)
        Auxiliary features
    n_programs : int
        Number of gene programs
    configuration : str, default='unmasked'
        Model configuration: 'unmasked', 'masked', 'combined', 'mask_initiated'
    pathways_dict : dict, optional
        Dictionary mapping pathway names to gene lists
    gene_names : array-like, optional
        List of gene names
    cyto_seed_genes : list, optional
        List of cytokine seed genes
    max_iters : int, default=1000
        Maximum number of iterations
    burn_in : int, default=500
        Number of burn-in iterations
    n_theta_samples : int, default=100
        Number of samples for theta inference in test/val
    output_dir : str, default='results'
        Output directory
    experiment_name : str, optional
        Name of the experiment
    random_state : int, default=42
        Random seed
    n_chains : int, default=4
        Number of parallel chains for convergence checking
    check_convergence : bool, default=True
        Whether to check convergence and allow early stopping
    convergence_check_interval : int, default=100
        How often to check convergence (in iterations)
    convergence_patience : int, default=3
        Number of consecutive convergence checks before stopping
    rhat_threshold : float, default=1.1
        R-hat threshold for convergence
    ess_threshold : float, default=400
        Effective sample size threshold
    **sampler_kwargs : dict
        Additional arguments for the sampler
        
    Returns:
    --------
    results : dict
        Dictionary containing all results
    """
    
    log_memory("Starting run_sampler_and_evaluate")
    
    if experiment_name is None:
        experiment_name = f"{configuration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Data splitting
    print("Splitting data...")
    (X_train, X_val, X_test, Y_train, Y_val, Y_test, 
     X_aux_train, X_aux_val, X_aux_test, 
     train_idx, val_idx, test_idx) = split_data(X, Y, X_aux, random_state=random_state)
    
    log_memory("After data splitting")
    
    # Set up sampler parameters
    sampler_params = {
        'alpha_theta': 0.3,
        'alpha_beta': 0.3,
        'alpha_xi': 0.3,
        'alpha_eta': 0.3,
        'lambda_xi': 0.3,
        'lambda_eta': 0.3,
        'pi_upsilon': 0.5,
        'sigma_slab_sq': 1.0,
        'sigma_gamma_sq': 1.0,
        'seed': random_state
    }
    sampler_params.update(sampler_kwargs)
    
    # Handle different configurations
    mask = None
    pathway_names = None
    
    if configuration == 'masked':
        if pathways_dict is None:
            raise ValueError("pathways_dict is required for masked configuration")
        mask, pathway_names = create_pathway_mask(pathways_dict, gene_names)
        n_programs = len(pathway_names)
        print(f"Using masked configuration with {n_programs} pathways")
        
    elif configuration == 'combined':
        if pathways_dict is None:
            raise ValueError("pathways_dict is required for combined configuration")
        if gene_names is None:
            raise ValueError("gene_names is required for combined configuration")
        pathway_mask, pathway_names = create_pathway_mask(pathways_dict, gene_names)
        n_pathway_programs = len(pathway_names)
        total_programs = n_pathway_programs + n_programs
        
        # Create combined mask
        mask = np.zeros((total_programs, len(gene_names)))
        mask[:n_pathway_programs, :] = pathway_mask
        # Additional programs are unmasked (all ones)
        mask[n_pathway_programs:, :] = 1
        
        n_programs = total_programs
        print(f"Using combined configuration with {n_pathway_programs} pathways + {n_programs - n_pathway_programs} free programs")
    
    # Initialize samplers (multiple chains if convergence checking is enabled)
    print(f"Initializing {n_chains if check_convergence else 1} sampler chain(s)...")
    samplers = []
    
    if check_convergence and n_chains > 1:
        # Initialize multiple chains with different random seeds
        for chain_idx in range(n_chains):
            chain_seed = random_state + chain_idx * 1000
            chain_params = sampler_params.copy()
            chain_params['seed'] = chain_seed
            
            sampler = SpikeSlabGibbsSampler(
                X_train, Y_train, X_aux_train, n_programs, **chain_params
            )
            
            # Special initialization for mask_initiated configuration
            if configuration == 'mask_initiated':
                if pathways_dict is None:
                    raise ValueError("pathways_dict is required for mask_initiated configuration")
                sampler.beta = initialize_beta_with_pathways(
                    pathways_dict, gene_names, n_programs, sampler.rng
                )
            
            samplers.append(sampler)
        
        # Initialize convergence monitor
        convergence_monitor = ConvergenceMonitor(
            check_interval=convergence_check_interval,
            rhat_threshold=rhat_threshold,
            ess_threshold=ess_threshold,
            min_samples=max(burn_in, 100),
            patience=convergence_patience
        )
        
        # Storage for chain samples
        chain_samples = {
            'beta': [],
            'gamma': [],
            'upsilon': [],
            'theta': []
        }
        
        print("Multiple chains initialized for convergence checking")
    else:
        # Single chain mode
        sampler = SpikeSlabGibbsSampler(
            X_train, Y_train, X_aux_train, n_programs, **sampler_params
        )
        
        # Special initialization for mask_initiated configuration
        if configuration == 'mask_initiated':
            if pathways_dict is None:
                raise ValueError("pathways_dict is required for mask_initiated configuration")
            print("Initializing beta with pathway information...")
            sampler.beta = initialize_beta_with_pathways(
                pathways_dict, gene_names, n_programs, sampler.rng
            )
        
        samplers.append(sampler)
    
    log_memory("After sampler initialization")
    
    # Training loop
    print(f"Starting training for {max_iters} iterations...")
    actual_iterations = 0
    early_stopped = False
    
    for iteration in range(max_iters):
        actual_iterations = iteration + 1
        
        # Update all chains
        for sampler in samplers:
            # Update latent counts
            sampler._update_latent_counts_z()
            
            # Update beta (with mask if applicable)
            if mask is not None:
                sampler._update_beta()
                # Apply mask to beta
                sampler.beta = sampler.beta * mask.T
            else:
                sampler._update_beta()
            
            # Update other parameters
            sampler._update_eta()
            sampler._update_xi()
            sampler._update_theta()
            sampler._update_gamma()
            sampler._update_s_upsilon()
            sampler._update_w_upsilon()
            sampler.upsilon = sampler.s_upsilon * sampler.w_upsilon
        
        # Store samples for convergence checking (after burn-in)
        if check_convergence and n_chains > 1 and iteration >= burn_in:
            # Store samples from all chains
            try:
                beta_samples = np.array([s.beta for s in samplers])
                gamma_samples = np.array([s.gamma for s in samplers])  
                upsilon_samples = np.array([s.upsilon for s in samplers])
                theta_samples = np.array([s.theta for s in samplers])
                
                # Check for invalid values
                if np.any(np.isnan(beta_samples)) or np.any(np.isinf(beta_samples)):
                    print("Warning: NaN/Inf detected in beta samples - skipping convergence check")
                    continue
                
                chain_samples['beta'].append(beta_samples)
                chain_samples['gamma'].append(gamma_samples)
                chain_samples['upsilon'].append(upsilon_samples)
                chain_samples['theta'].append(theta_samples)
                
                # Limit memory usage - keep only recent samples
                max_samples = 50
                for param_name in chain_samples:
                    if len(chain_samples[param_name]) > max_samples:
                        chain_samples[param_name] = chain_samples[param_name][-max_samples:]
                
                # Check convergence periodically
                if convergence_monitor.should_check(iteration):
                    # Convert to proper format for convergence checking
                    chains_dict = {}
                    for param_name, samples_list in chain_samples.items():
                        if len(samples_list) >= 10:  # Need minimum samples
                            try:
                                # Shape: (n_chains, n_samples, ...)
                                samples_array = np.array(samples_list)
                                chains_dict[param_name] = samples_array.transpose(1, 0, *range(2, samples_array.ndim))
                            except Exception as e:
                                print(f"Warning: Error processing {param_name} samples: {e}")
                                continue
                    
                    if len(chains_dict) > 0:
                        should_stop, diagnostics = convergence_monitor.check_and_update(chains_dict, iteration)
                        
                        if should_stop:
                            print(f"Early stopping at iteration {iteration + 1} due to convergence!")
                            early_stopped = True
                            break
                    else:
                        print("Warning: No valid samples for convergence checking")
                        
            except Exception as e:
                print(f"Warning: Error in convergence checking: {e}")
                continue
        
        # Periodic memory cleanup and logging
        if (iteration + 1) % 50 == 0:
            # Clean up memory every 50 iterations
            for sampler in samplers:
                sampler.cleanup_memory()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{max_iters}")
            log_memory(f"After iteration {iteration + 1}")
    
    # Use the first chain's parameters for final results
    final_sampler = samplers[0]
    
    print(f"Training completed! ({'Early stopped' if early_stopped else 'Full iterations'}: {actual_iterations}/{max_iters})")
    log_memory("After training")
    
    # Make predictions on training data
    print("Making predictions on training data...")
    logits_train = X_aux_train @ final_sampler.gamma.T + final_sampler.theta @ final_sampler.upsilon.T
    Y_pred_proba_train = expit(logits_train)
    Y_pred_labels_train = (Y_pred_proba_train > 0.5).astype(int)
    
    # Calculate cytokine seed scores for training data
    cyto_scores_train = None
    if cyto_seed_genes is not None:
        cyto_scores_train = calculate_cyto_seed_score(X_train, gene_names, cyto_seed_genes)
    
    # IMPORTANT: Infer theta for both validation and test sets
    # Since theta is sample-specific, we need to infer it separately for each dataset
    # using the fixed beta, gamma, and upsilon from training
    
    # Infer theta and make predictions on validation data
    print("Inferring theta and making predictions on validation data...")
    theta_val = infer_theta_for_test(
        X_val, Y_val, X_aux_val, final_sampler.beta, final_sampler.gamma, final_sampler.upsilon,
        sampler_params, n_samples=n_theta_samples, burn_in=burn_in//2
    )
    
    logits_val = X_aux_val @ final_sampler.gamma.T + theta_val @ final_sampler.upsilon.T
    Y_pred_proba_val = expit(logits_val)
    Y_pred_labels_val = (Y_pred_proba_val > 0.5).astype(int)
    
    cyto_scores_val = None
    if cyto_seed_genes is not None:
        cyto_scores_val = calculate_cyto_seed_score(X_val, gene_names, cyto_seed_genes)
    
    # Force cleanup after validation
    import gc
    gc.collect()
    log_memory("After validation predictions")
    
    # Infer theta and make predictions on test data
    print("Inferring theta and making predictions on test data...")
    theta_test = infer_theta_for_test(
        X_test, Y_test, X_aux_test, final_sampler.beta, final_sampler.gamma, final_sampler.upsilon,
        sampler_params, n_samples=n_theta_samples, burn_in=burn_in//2
    )
    
    logits_test = X_aux_test @ final_sampler.gamma.T + theta_test @ final_sampler.upsilon.T
    Y_pred_proba_test = expit(logits_test)
    Y_pred_labels_test = (Y_pred_proba_test > 0.5).astype(int)
    
    cyto_scores_test = None
    if cyto_seed_genes is not None:
        cyto_scores_test = calculate_cyto_seed_score(X_test, gene_names, cyto_seed_genes)
    
    # Force cleanup after test predictions
    gc.collect()
    log_memory("After predictions")
    
    # Evaluate predictions
    print("Evaluating predictions...")
    train_metrics = evaluate_predictions(Y_train, Y_pred_proba_train, Y_pred_labels_train)
    val_metrics = evaluate_predictions(Y_val, Y_pred_proba_val, Y_pred_labels_val)
    test_metrics = evaluate_predictions(Y_test, Y_pred_proba_test, Y_pred_labels_test)
    
    # Extract DRGP matrix
    print("Extracting DRGP matrix...")
    drgp_matrix = extract_drgp_matrix(final_sampler.beta, final_sampler.upsilon, gene_names, pathway_names)
    
    # Prepare results
    results = {
        'experiment_name': experiment_name,
        'configuration': configuration,
        'n_programs': n_programs,
        'max_iters': max_iters,
        'actual_iterations': actual_iterations,
        'early_stopped': early_stopped,
        'burn_in': burn_in,
        'sampler_params': sampler_params,
        'convergence_params': {
            'n_chains': n_chains,
            'check_convergence': check_convergence,
            'rhat_threshold': rhat_threshold,
            'ess_threshold': ess_threshold,
            'convergence_check_interval': convergence_check_interval,
            'convergence_patience': convergence_patience
        },
        
        # Data splits
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        
        # Model parameters
        'beta': final_sampler.beta,
        'gamma': final_sampler.gamma,
        'upsilon': final_sampler.upsilon,
        'theta_train': final_sampler.theta,
        'theta_val': theta_val,
        'theta_test': theta_test,
        
        # Predictions
        'train_predictions': {
            'true_labels': Y_train,
            'predicted_labels': Y_pred_labels_train,
            'predicted_probabilities': Y_pred_proba_train,
            'cyto_seed_scores': cyto_scores_train
        },
        'val_predictions': {
            'true_labels': Y_val,
            'predicted_labels': Y_pred_labels_val,
            'predicted_probabilities': Y_pred_proba_val,
            'cyto_seed_scores': cyto_scores_val
        },
        'test_predictions': {
            'true_labels': Y_test,
            'predicted_labels': Y_pred_labels_test,
            'predicted_probabilities': Y_pred_proba_test,
            'cyto_seed_scores': cyto_scores_test
        },
        
        # Metrics
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        
        # DRGP matrix
        'drgp_matrix': drgp_matrix,
        
        # Configuration-specific info
        'mask': mask,
        'pathway_names': pathway_names
    }
    
    # Save results
    print("Saving results...")
    save_results(results, output_dir, experiment_name)
    
    log_memory("After saving results")
    clear_memory()
    
    print(f"Experiment {experiment_name} completed successfully!")
    
    return results


def main():
    """Main function to run experiments from command line."""
    
    parser = argparse.ArgumentParser(description='Run Gibbs sampler experiments')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ajm_cyto', 'ajm_ap', 'emtab'],
                       help='Dataset to use')
    parser.add_argument('--configuration', type=str, required=True,
                       choices=['unmasked', 'masked', 'combined', 'mask_initiated'],
                       help='Model configuration')
    
    # Optional arguments
    parser.add_argument('--d', type=int, default=10,
                       help='Number of gene programs for unmasked configuration')
    parser.add_argument('--num_gps', type=int, default=5,
                       help='Number of additional gene programs for combined configuration')
    parser.add_argument('--max_iters', type=int, default=1000,
                       help='Maximum number of iterations')
    parser.add_argument('--burn_in', type=int, default=500,
                       help='Number of burn-in iterations')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    # Convergence checking arguments
    parser.add_argument('--n_chains', type=int, default=4,
                       help='Number of parallel chains for convergence checking')
    parser.add_argument('--check_convergence', action='store_true', default=True,
                       help='Enable convergence checking and early stopping')
    parser.add_argument('--no_convergence_check', action='store_false', dest='check_convergence',
                       help='Disable convergence checking')
    parser.add_argument('--convergence_check_interval', type=int, default=100,
                       help='How often to check convergence (in iterations)')
    parser.add_argument('--convergence_patience', type=int, default=3,
                       help='Number of consecutive convergence checks before stopping')
    parser.add_argument('--rhat_threshold', type=float, default=1.1,
                       help='R-hat threshold for convergence')
    parser.add_argument('--ess_threshold', type=float, default=400,
                       help='Effective sample size threshold')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    
    if args.dataset == 'ajm_cyto':
        _, ajm_cyto_samples = prepare_ajm_dataset()
        ajm_cyto_filtered = filter_protein_coding_genes(ajm_cyto_samples, gene_annotation)
        
        X = ajm_cyto_filtered.X.toarray() if sp.issparse(ajm_cyto_filtered.X) else ajm_cyto_filtered.X
        Y = ajm_cyto_filtered.obs['cyto'].values.reshape(-1, 1)
        X_aux = np.zeros((X.shape[0], 1))  # No auxiliary features
        gene_names = ajm_cyto_filtered.var_names.tolist()
        cyto_seed_genes = CYTOSEED_ensembl
        
    elif args.dataset == 'ajm_ap':
        ajm_ap_samples, _ = prepare_ajm_dataset()
        ajm_ap_filtered = filter_protein_coding_genes(ajm_ap_samples, gene_annotation)
        
        X = ajm_ap_filtered.X.toarray() if sp.issparse(ajm_ap_filtered.X) else ajm_ap_filtered.X
        Y = ajm_ap_filtered.obs['ap'].values.reshape(-1, 1)
        X_aux = np.zeros((X.shape[0], 1))  # No auxiliary features
        gene_names = ajm_ap_filtered.var_names.tolist()
        cyto_seed_genes = None
        
    elif args.dataset == 'emtab':
        emtab_data = prepare_and_load_emtab()
        emtab_filtered = filter_protein_coding_genes(emtab_data, gene_annotation)
        
        X = emtab_filtered.X.toarray() if sp.issparse(emtab_filtered.X) else emtab_filtered.X
        Y = emtab_filtered.obs[["Crohn's disease", "ulcerative colitis"]].values
        X_aux = emtab_filtered.obs[["age", "sex_female"]].values
        gene_names = emtab_filtered.var_names.tolist()
        cyto_seed_genes = None
    
    print(f"Data loaded: X shape {X.shape}, Y shape {Y.shape}, X_aux shape {X_aux.shape}")
    
    # Set up configuration parameters
    if args.configuration == 'unmasked':
        n_programs = args.d
    elif args.configuration == 'masked':
        n_programs = len(pathways)  # Will be set in the function
    elif args.configuration == 'combined':
        n_programs = args.num_gps  # Additional programs, pathways will be added
    elif args.configuration == 'mask_initiated':
        n_programs = len(pathways)  # Will be set in the function
    
    # Run experiment
    results = run_sampler_and_evaluate(
        X=X,
        Y=Y,
        X_aux=X_aux,
        n_programs=n_programs,
        configuration=args.configuration,
        pathways_dict=pathways,
        gene_names=gene_names,
        cyto_seed_genes=cyto_seed_genes,
        max_iters=args.max_iters,
        burn_in=args.burn_in,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        random_state=args.random_state,
        n_chains=args.n_chains,
        check_convergence=args.check_convergence,
        convergence_check_interval=args.convergence_check_interval,
        convergence_patience=args.convergence_patience,
        rhat_threshold=args.rhat_threshold,
        ess_threshold=args.ess_threshold
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Configuration: {args.configuration}")
    print(f"Number of programs: {results['n_programs']}")
    print(f"Iterations: {results['actual_iterations']}/{args.max_iters}")
    if results['early_stopped']:
        print("✓ Early stopping due to convergence")
    print(f"Convergence checking: {'Enabled' if args.check_convergence else 'Disabled'}")
    if args.check_convergence:
        print(f"Number of chains: {args.n_chains}")
    
    print("\nTest Set Performance:")
    for label_key, metrics in results['test_metrics'].items():
        print(f"  {label_key}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")
    
    print(f"\nResults saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
