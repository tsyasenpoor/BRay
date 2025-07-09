#!/usr/bin/env python3
"""
Minimal test to isolate memory issues - single chain, no convergence checking.
"""

import os
import gc
import numpy as np

def minimal_test():
    """Run minimal test with single chain and memory monitoring."""
    
    print("="*60)
    print("MINIMAL MEMORY TEST")
    print("="*60)
    
    # Import with memory tracking
    from memory_tracking import log_memory
    from data import prepare_test_emtab_dataset
    from run_experiments import run_sampler_and_evaluate
    import scipy.sparse as sp
    
    log_memory("Before creating test data")
    
    # Create very small test dataset
    adata_test = prepare_test_emtab_dataset(n_samples=30, n_genes=100, random_state=42)
    
    X = adata_test.X.toarray() if sp.issparse(adata_test.X) else adata_test.X
    Y = adata_test.obs[["Crohn's disease", "ulcerative colitis"]].values
    X_aux = adata_test.obs[["age", "sex_female"]].values
    gene_names = adata_test.var_names.tolist()
    
    log_memory("After creating test data")
    
    print(f"Test data shape: X={X.shape}, Y={Y.shape}")
    print(f"Data range: {X.min():.3f} - {X.max():.3f}")
    
    # Run with SINGLE chain and NO convergence checking
    print("\nRunning single chain, no convergence checking...")
    
    results = run_sampler_and_evaluate(
        X=X,
        Y=Y,
        X_aux=X_aux,
        n_programs=3,  # Very few programs
        configuration='unmasked',
        pathways_dict=None,
        gene_names=gene_names,
        cyto_seed_genes=None,
        max_iters=5,
        burn_in=1,
        output_dir='minimal_test_results',
        experiment_name='minimal_test',
        random_state=42,
        n_chains=1,  # SINGLE CHAIN
        check_convergence=False,  # NO CONVERGENCE CHECKING
    )
    
    log_memory("After experiment")
    
    print(f"\nTest completed: {results['actual_iterations']} iterations")
    print(f"Memory stable: {results.get('early_stopped', False)}")
    
    # Force cleanup
    del results, X, Y, X_aux, adata_test
    gc.collect()
    log_memory("After cleanup")
    
    return True


if __name__ == "__main__":
    try:
        minimal_test()
        print("✅ Minimal test PASSED")
    except Exception as e:
        print(f"❌ Minimal test FAILED: {e}")
        import traceback
        traceback.print_exc() 