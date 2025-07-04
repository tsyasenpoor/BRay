import numpy as np
import warnings
from typing import Dict, Tuple, Optional


def _compute_rhat(chains: np.ndarray) -> np.ndarray:
    """Compute Gelman-Rubin R-hat statistic.

    Parameters
    ----------
    chains : np.ndarray
        Array of shape (n_chains, n_samples, ...).

    Returns
    -------
    np.ndarray
        R-hat values with shape matching ``chains`` without the first two axes.
    """
    m, n = chains.shape[0], chains.shape[1]
    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean(axis=0)
    B = n * np.sum((chain_means - grand_mean) ** 2, axis=0) / (m - 1)
    W = chains.var(axis=1, ddof=1).mean(axis=0)
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    
    # Avoid division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rhat = np.sqrt(var_hat / W)
        rhat = np.where(W > 0, rhat, np.nan)
    
    return rhat


def _effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Estimate effective sample size using autocorrelation."""
    m, n = chains.shape[0], chains.shape[1]
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)

    var_within = chain_vars.mean(axis=0)
    acov_sum = np.zeros_like(var_within)
    
    for t in range(1, n):
        if t >= n:
            break
        acov_t = ((chains[:, :-t] - chain_means[:, None]) * 
                  (chains[:, t:] - chain_means[:, None])).mean(axis=(0, 1))
        
        # Avoid division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rho_t = np.where(var_within > 0, acov_t / var_within, 0)
        
        if np.all(rho_t < 0):
            break
        acov_sum += rho_t
    
    tau = 1 + 2 * acov_sum
    ess = m * n / tau
    return ess


def check_convergence(chains_dict: Dict[str, np.ndarray], 
                     rhat_threshold: float = 1.1,
                     ess_threshold: float = 400,
                     min_samples: int = 100) -> Tuple[bool, Dict[str, float]]:
    """
    Check MCMC convergence using R-hat and effective sample size.
    
    Parameters
    ----------
    chains_dict : Dict[str, np.ndarray]
        Dictionary of parameter chains with shape (n_chains, n_samples, ...)
    rhat_threshold : float, default=1.1
        R-hat threshold for convergence (should be close to 1.0)
    ess_threshold : float, default=400
        Minimum effective sample size threshold
    min_samples : int, default=100
        Minimum number of samples before checking convergence
        
    Returns
    -------
    converged : bool
        True if all parameters have converged
    diagnostics : Dict[str, float]
        Dictionary of convergence diagnostics
    """
    
    diagnostics = {}
    converged = True
    
    for param_name, chains in chains_dict.items():
        if chains.shape[1] < min_samples:
            converged = False
            diagnostics[f'{param_name}_rhat'] = np.nan
            diagnostics[f'{param_name}_ess'] = np.nan
            continue
        
        # Compute R-hat
        rhat = _compute_rhat(chains)
        max_rhat = np.nanmax(rhat)
        
        # Compute effective sample size
        ess = _effective_sample_size(chains)
        min_ess = np.nanmin(ess)
        
        # Store diagnostics
        diagnostics[f'{param_name}_rhat'] = float(max_rhat)
        diagnostics[f'{param_name}_ess'] = float(min_ess)
        
        # Check convergence criteria
        if max_rhat > rhat_threshold or min_ess < ess_threshold:
            converged = False
    
    return converged, diagnostics


def print_convergence_diagnostics(diagnostics: Dict[str, float], 
                                rhat_threshold: float = 1.1,
                                ess_threshold: float = 400) -> None:
    """
    Print convergence diagnostics in a formatted way.
    
    Parameters
    ----------
    diagnostics : Dict[str, float]
        Dictionary of convergence diagnostics
    rhat_threshold : float
        R-hat threshold for convergence
    ess_threshold : float
        Effective sample size threshold
    """
    
    print("\n" + "="*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*60)
    
    # Group diagnostics by parameter
    param_names = set()
    for key in diagnostics.keys():
        param_name = key.replace('_rhat', '').replace('_ess', '')
        param_names.add(param_name)
    
    for param_name in sorted(param_names):
        rhat_key = f'{param_name}_rhat'
        ess_key = f'{param_name}_ess'
        
        if rhat_key in diagnostics and ess_key in diagnostics:
            rhat_val = diagnostics[rhat_key]
            ess_val = diagnostics[ess_key]
            
            # Status indicators
            rhat_status = "✓" if rhat_val <= rhat_threshold else "✗"
            ess_status = "✓" if ess_val >= ess_threshold else "✗"
            
            print(f"{param_name:15} | R-hat: {rhat_val:6.3f} {rhat_status} | ESS: {ess_val:8.1f} {ess_status}")
    
    print("-" * 60)
    print(f"Thresholds: R-hat ≤ {rhat_threshold:.1f}, ESS ≥ {ess_threshold:.0f}")
    print("="*60)


def adaptive_thinning(chains: np.ndarray, target_ess: float = 400) -> int:
    """
    Determine appropriate thinning interval to achieve target ESS.
    
    Parameters
    ----------
    chains : np.ndarray
        Array of shape (n_chains, n_samples, ...)
    target_ess : float
        Target effective sample size
        
    Returns
    -------
    thin_interval : int
        Recommended thinning interval
    """
    
    current_ess = _effective_sample_size(chains)
    min_ess = np.nanmin(current_ess)
    
    if min_ess >= target_ess:
        return 1
    
    # Estimate required thinning
    total_samples = chains.shape[0] * chains.shape[1]
    thin_interval = max(1, int(total_samples / target_ess))
    
    return thin_interval


class ConvergenceMonitor:
    """
    Monitor MCMC convergence during sampling.
    """
    
    def __init__(self, 
                 check_interval: int = 100,
                 rhat_threshold: float = 1.1,
                 ess_threshold: float = 400,
                 min_samples: int = 100,
                 patience: int = 3):
        """
        Initialize convergence monitor.
        
        Parameters
        ----------
        check_interval : int
            How often to check convergence (in iterations)
        rhat_threshold : float
            R-hat threshold for convergence
        ess_threshold : float
            Effective sample size threshold
        min_samples : int
            Minimum samples before checking convergence
        patience : int
            Number of consecutive convergence checks before stopping
        """
        self.check_interval = check_interval
        self.rhat_threshold = rhat_threshold
        self.ess_threshold = ess_threshold
        self.min_samples = min_samples
        self.patience = patience
        self.convergence_count = 0
        self.last_check = 0
        self.convergence_history = []
    
    def should_check(self, iteration: int) -> bool:
        """Check if we should run convergence diagnostics."""
        return (iteration - self.last_check) >= self.check_interval
    
    def check_and_update(self, chains_dict: Dict[str, np.ndarray], 
                        iteration: int) -> Tuple[bool, Dict[str, float]]:
        """
        Check convergence and update internal state.
        
        Returns
        -------
        should_stop : bool
            True if sampling should stop
        diagnostics : Dict[str, float]
            Convergence diagnostics
        """
        
        self.last_check = iteration
        converged, diagnostics = check_convergence(
            chains_dict, 
            self.rhat_threshold, 
            self.ess_threshold, 
            self.min_samples
        )
        
        self.convergence_history.append({
            'iteration': iteration,
            'converged': converged,
            'diagnostics': diagnostics
        })
        
        if converged:
            self.convergence_count += 1
            print(f"Convergence achieved at iteration {iteration} ({self.convergence_count}/{self.patience})")
            if self.convergence_count >= self.patience:
                print("Stopping early due to convergence!")
                return True, diagnostics
        else:
            self.convergence_count = 0
            print(f"Convergence not yet achieved at iteration {iteration}")
            print_convergence_diagnostics(diagnostics, self.rhat_threshold, self.ess_threshold)
        
        return False, diagnostics 