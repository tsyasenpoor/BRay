import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, confusion_matrix, roc_auc_score,
                            classification_report, roc_curve, auc)

from vi_model_complete import (
    initialize_q_params, run_variational_inference, 
    compute_elbo, predict_labels, evaluate_model,
    plot_roc_curve, plot_elbo_convergence
)

def generate_synthetic_data(n_samples, n_genes, n_programs, p_aux=1, seed=42):

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
        "d": n_programs
    }

    eta_true = random.gamma(keys[0], hyperparams["c_prime"], shape=(n_genes,)) / hyperparams["d_prime"]
    
    beta_true = random.gamma(keys[1], hyperparams["c"], shape=(n_genes, n_programs)) / jnp.expand_dims(eta_true, axis=1)
    
    xi_true = random.gamma(keys[2], hyperparams["a_prime"], shape=(n_samples,)) / hyperparams["b_prime"]
    
    theta_true = random.gamma(keys[3], hyperparams["a"], shape=(n_samples, n_programs)) / jnp.expand_dims(xi_true, axis=1)
    
    gamma_true = jnp.zeros((1, p_aux))
    if p_aux > 0:
        gamma_true = gamma_true.at[0, 0].set(0.0)  
    
    upsilon_true = jnp.zeros((1, n_programs))
    for i in range(n_programs):
        if i % 2 == 0:
            upsilon_true = upsilon_true.at[0, i].set(random.normal(keys[5], shape=()) * 0.5)
        else:
            upsilon_true = upsilon_true.at[0, i].set(random.normal(keys[5], shape=()) * -0.5)
    
    rate = jnp.dot(theta_true, beta_true.T)
    x_data = random.poisson(keys[6], rate)
    
    if p_aux == 1:
        x_aux = jnp.ones((n_samples, 1))
    else:
        x_aux = jnp.hstack([jnp.ones((n_samples, 1)), 
                           random.normal(keys[7], shape=(n_samples, p_aux-1))])
    
    logits = jnp.dot(x_aux, gamma_true.T) + jnp.dot(theta_true, upsilon_true.T)
    probs = jax.nn.sigmoid(logits)
    
    y_data = (random.uniform(keys[8], shape=logits.shape) < probs).astype(int)
    
    class_counts = jnp.sum(y_data)
    print(f"Generated {class_counts} positive samples out of {n_samples} total samples.")
    
    if class_counts == 0 or class_counts == n_samples:
        print("Only one class present in generated data. Adjusting and regenerating...")
        return generate_synthetic_data(n_samples, n_genes, n_programs, p_aux, seed+1)
    
    data_dict = {
        "x_data": jnp.array(x_data),
        "x_aux": jnp.array(x_aux),
        "y_data": jnp.array(y_data),
        "true_params": {
            "eta": eta_true,
            "beta": beta_true,
            "xi": xi_true,
            "theta": theta_true,
            "gamma": gamma_true,
            "upsilon": upsilon_true
        },
        "hyperparams": hyperparams
    }
    
    return data_dict

def compare_parameters(true_params, inferred_params):
    E_eta = inferred_params['alpha_eta'] / inferred_params['omega_eta']
    E_beta = inferred_params['alpha_beta'] / inferred_params['omega_beta']
    E_xi = inferred_params['alpha_xi'] / inferred_params['omega_xi']
    E_theta = inferred_params['alpha_theta'] / inferred_params['omega_theta']
    E_gamma = inferred_params['gamma']
    E_upsilon = inferred_params['upsilon']
    
    corr_eta = jnp.corrcoef(true_params['eta'], E_eta)[0, 1]
    
    corr_beta = []
    for j in range(true_params['beta'].shape[0]):
        row_corr = jnp.corrcoef(true_params['beta'][j], E_beta[j])[0, 1]
        if not jnp.isnan(row_corr):
            corr_beta.append(row_corr)
    
    corr_xi = jnp.corrcoef(true_params['xi'], E_xi)[0, 1]
    
    corr_theta = []
    for i in range(true_params['theta'].shape[0]):
        row_corr = jnp.corrcoef(true_params['theta'][i], E_theta[i])[0, 1]
        if not jnp.isnan(row_corr):
            corr_theta.append(row_corr)
    
    corr_gamma = jnp.corrcoef(true_params['gamma'].flatten(), E_gamma.flatten())[0, 1]
    corr_upsilon = jnp.corrcoef(true_params['upsilon'].flatten(), E_upsilon.flatten())[0, 1]
    
    mse_eta = jnp.mean((true_params['eta'] - E_eta) ** 2)
    mse_beta = jnp.mean((true_params['beta'] - E_beta) ** 2)
    mse_xi = jnp.mean((true_params['xi'] - E_xi) ** 2)
    mse_theta = jnp.mean((true_params['theta'] - E_theta) ** 2)
    mse_gamma = jnp.mean((true_params['gamma'] - E_gamma) ** 2)
    mse_upsilon = jnp.mean((true_params['upsilon'] - E_upsilon) ** 2)
    
    results = {
        "correlations": {
            "eta": float(corr_eta),
            "beta": float(jnp.mean(jnp.array(corr_beta))),
            "xi": float(corr_xi),
            "theta": float(jnp.mean(jnp.array(corr_theta))),
            "gamma": float(corr_gamma),
            "upsilon": float(corr_upsilon)
        },
        "mse": {
            "eta": float(mse_eta),
            "beta": float(mse_beta),
            "xi": float(mse_xi),
            "theta": float(mse_theta),
            "gamma": float(mse_gamma),
            "upsilon": float(mse_upsilon)
        }
    }
    
    return results

def plot_parameter_recovery(true_params, inferred_params, param_name, sample_size=100):
    if param_name == 'eta':
        true = true_params['eta']
        inferred = inferred_params['alpha_eta'] / inferred_params['omega_eta']
    elif param_name == 'beta':
        true = true_params['beta'].flatten()
        inferred = (inferred_params['alpha_beta'] / inferred_params['omega_beta']).flatten()
    elif param_name == 'xi':
        true = true_params['xi']
        inferred = inferred_params['alpha_xi'] / inferred_params['omega_xi']
    elif param_name == 'theta':
        true = true_params['theta'].flatten()
        inferred = (inferred_params['alpha_theta'] / inferred_params['omega_theta']).flatten()
    elif param_name == 'gamma':
        true = true_params['gamma'].flatten()
        inferred = inferred_params['gamma'].flatten()
    elif param_name == 'upsilon':
        true = true_params['upsilon'].flatten()
        inferred = inferred_params['upsilon'].flatten()
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")
    
    true = np.array(true)
    inferred = np.array(inferred)
    
    if len(true) > sample_size:
        idx = np.random.choice(len(true), sample_size, replace=False)
        true = true[idx]
        inferred = inferred[idx]
    
    corr = np.corrcoef(true, inferred)[0, 1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=true, y=inferred, alpha=0.6)
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Inferred Values')
    ax.set_title(f'Parameter Recovery: {param_name.capitalize()} (r = {corr:.3f})')
    
    return fig

def run_synthetic_data_test(n_samples=200, n_genes=100, n_programs=10, p_aux=1, 
                           max_iters=200, tol=1e-4, seed=42, output_dir="synthetic_test_results"):
    def safe_evaluate_model(y_true, y_pred, y_prob):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        if len(y_true.shape) > 1 and y_true.shape[1] == 1:
            y_true = y_true.flatten()
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        if len(y_prob.shape) > 1 and y_prob.shape[1] == 1:
            y_prob = y_prob.flatten()
        
        # Check if we have only one class
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 1:
            print(f"Warning: Only one class ({unique_classes[0]}) present in y_true. Some metrics will be undefined.")
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": 0.0 if unique_classes[0] == 0 else 1.0,
                "recall": 0.0 if unique_classes[0] == 0 else 1.0,
                "f1": 0.0 if unique_classes[0] == 0 else 1.0,
                "roc_auc": 0.5,  # Default for undefined case
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            }
        else:
            try:
                metrics = evaluate_model(y_true, y_pred, y_prob)
            except Exception as e:
                print(f"Error in evaluate_model: {e}")
                # Fallback implementation
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred),
                    "recall": recall_score(y_true, y_pred),
                    "f1": f1_score(y_true, y_pred),
                    "roc_auc": roc_auc_score(y_true, y_prob),
                    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
                }
        
        return metrics

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating synthetic data with {n_samples} samples, {n_genes} genes, and {n_programs} programs...")
    data_dict = generate_synthetic_data(n_samples, n_genes, n_programs, p_aux, seed)
    
    x_data = data_dict["x_data"]
    x_aux = data_dict["x_aux"]
    y_data = data_dict["y_data"]
    true_params = data_dict["true_params"]
    hyperparams = data_dict["hyperparams"]
    
    print("Initializing variational parameters...")
    q_params = initialize_q_params(n_samples, n_genes, 1, p_aux, n_programs, seed=seed)
    
    print("Running variational inference...")
    q_params_final, elbo_history = run_variational_inference(
        x_data=x_data,
        y_data=y_data,
        x_aux=x_aux,
        hyperparams=hyperparams,
        q_params=q_params,
        max_iters=max_iters,
        tol=tol,
        verbose=True
    )
    
    print("Comparing true vs. inferred parameters...")
    comparison_results = compare_parameters(true_params, q_params_final)
    
    print("Evaluating prediction performance...")
    y_pred, y_prob = predict_labels(x_data, x_aux, q_params_final)
    metrics = safe_evaluate_model(y_data, y_pred, y_prob)
    
    print("Generating visualizations and saving results...")
    fig_elbo = plot_elbo_convergence(elbo_history, "ELBO Convergence - Synthetic Data")
    fig_elbo.savefig(os.path.join(output_dir, "elbo_convergence.png"))
    plt.close(fig_elbo)
    
    fig_roc = plot_roc_curve(y_data, y_prob, "ROC Curve - Synthetic Data")
    fig_roc.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close(fig_roc)
    
    for param_name in ['eta', 'beta', 'xi', 'theta', 'gamma', 'upsilon']:
        fig = plot_parameter_recovery(true_params, q_params_final, param_name)
        fig.savefig(os.path.join(output_dir, f"{param_name}_recovery.png"))
        plt.close(fig)
    
    corr_data = pd.DataFrame.from_dict(comparison_results["correlations"], orient='index', columns=["Correlation"])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, cmap="viridis", vmin=-1, vmax=1, cbar_kws={"label": "Correlation"})
    plt.title("Parameter Recovery Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    
    mse_data = pd.DataFrame.from_dict(comparison_results["mse"], orient='index', columns=["MSE"])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mse_data.index, y="MSE", data=mse_data)
    plt.title("Parameter Recovery Mean Squared Error")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_barplot.png"))
    plt.close()
    
    print("\n=== PARAMETER RECOVERY SUMMARY ===")
    print("Parameter Correlations:")
    for param, corr in comparison_results["correlations"].items():
        print(f"  {param}: {corr:.4f}")
    
    print("\nPrediction Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    return {
        "parameter_comparison": comparison_results,
        "metrics": metrics,
        "elbo_history": elbo_history,
        "q_params_final": q_params_final,
        "true_params": true_params
    }

def plot_program_weights(true_params, inferred_params, output_dir):
    beta_true = true_params['beta']
    beta_inferred = inferred_params['alpha_beta'] / inferred_params['omega_beta']
    
    upsilon_true = true_params['upsilon']
    upsilon_inferred = inferred_params['upsilon']
    
    n_programs = min(4, beta_true.shape[1])
    
    program_dir = os.path.join(output_dir, "program_weights")
    os.makedirs(program_dir, exist_ok=True)
    
    for prog_idx in range(n_programs):
        true_weights = beta_true[:, prog_idx]
        sorted_indices = np.argsort(-true_weights)[:20]  
        
        inferred_weights = beta_inferred[:, prog_idx]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(sorted_indices)), true_weights[sorted_indices], color='blue', alpha=0.7)
        plt.title(f"True Weights - Program {prog_idx+1}")
        plt.xlabel("Gene Index")
        plt.ylabel("Weight")
        plt.xticks(range(len(sorted_indices)), sorted_indices)
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(sorted_indices)), inferred_weights[sorted_indices], color='red', alpha=0.7)
        plt.title(f"Inferred Weights - Program {prog_idx+1}")
        plt.xlabel("Gene Index")
        plt.ylabel("Weight")
        plt.xticks(range(len(sorted_indices)), sorted_indices)
        
        plt.tight_layout()
        plt.savefig(os.path.join(program_dir, f"program_{prog_idx+1}_weights.png"))
        plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(upsilon_true.shape[1]), upsilon_true[0], color='blue', alpha=0.6, label='True')
    plt.bar(range(upsilon_inferred.shape[1]), upsilon_inferred[0], color='red', alpha=0.6, label='Inferred')
    plt.title("Program Association with Outcome (Upsilon)")
    plt.xlabel("Program Index")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "upsilon_weights.png"))
    plt.close()

if __name__ == "__main__":
    config = {
        "n_samples": 200,       # Number of samples/cells
        "n_genes": 100,         # Number of genes
        "n_programs": 10,       # Number of latent programs
        "p_aux": 1,             # Number of auxiliary covariates (1 = intercept only)
        "max_iters": 200,       # Maximum VI iterations
        "tol": 1e-4,            # Convergence tolerance
        "seed": 42,             # Random seed
        "output_dir": "synthetic_test_results"  # Output directory
    }
    
    
    results = run_synthetic_data_test(**config)
    
    plot_program_weights(
        results['true_params'], 
        results['q_params_final'], 
        config['output_dir']
    )