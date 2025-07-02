#!/usr/bin/env python3
"""
Quick test of memory optimizations in Gibbs sampler.
"""

import numpy as np
import gc
from gibbs import SpikeSlabGibbsSampler, log_memory

def quick_test():
    """Quick test with small dataset to verify memory optimizations work."""
    
    print("Creating small test dataset...")
    
    # Small test dataset
    n_samples = 100
    n_genes = 500
    n_programs = 20
    
    # Create sparse data
    from scipy.sparse import random
    X = random(n_samples, n_genes, density=0.1, format='csr', dtype=np.float32)
    X.data = np.abs(X.data)  # Ensure non-negative
    
    X_aux = np.ones((n_samples, 1), dtype=np.float32)
    Y = np.random.binomial(1, 0.3, size=(n_samples, 1)).astype(np.float32)
    
    print(f"Dataset: {X.shape}, sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.3f}")
    
    # Test memory-efficient sampler
    print("\nTesting memory-efficient sampler...")
    log_memory("Before sampler creation")
    
    sampler = SpikeSlabGibbsSampler(
        X, Y, X_aux, n_programs,
        memory_efficient=True,
        use_float32=True,
        trace_max_samples=50
    )
    
    log_memory("After sampler creation")
    
    # Run a few iterations
    results = sampler.run(10, save_traces=True)
    
    log_memory("After sampling")
    
    print(f"Results keys: {list(results.keys())}")
    print(f"Upsilon trace shape: {results['upsilon'].shape}")
    print(f"Gamma trace shape: {results['gamma'].shape}")
    
    # Clean up
    del sampler, results, X, Y, X_aux
    gc.collect()
    log_memory("After cleanup")
    
    print("\nMemory optimization test completed successfully!")

if __name__ == "__main__":
    quick_test() 