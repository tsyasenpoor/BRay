#!/usr/bin/env python3
"""
Test script to demonstrate memory optimizations in the Gibbs sampler.
"""

import numpy as np
import gc
import time
from gibbs import SpikeSlabGibbsSampler, log_memory

def create_test_data(n_samples=1000, n_genes=5000, n_programs=100, sparsity=0.95):
    """Create test data with specified sparsity."""
    print(f"Creating test data: {n_samples} samples, {n_genes} genes, {n_programs} programs")
    
    # Create sparse matrix
    from scipy.sparse import random
    X = random(n_samples, n_genes, density=1-sparsity, format='csr', dtype=np.float32)
    X.data = np.abs(X.data)  # Ensure non-negative
    
    # Create auxiliary variables
    X_aux = np.ones((n_samples, 1), dtype=np.float32)
    
    # Create binary outcomes
    Y = np.random.binomial(1, 0.3, size=(n_samples, 1)).astype(np.float32)
    
    print(f"X shape: {X.shape}, sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.3f}")
    print(f"X memory: {X.data.nbytes / (1024**2):.1f} MB")
    
    return X, Y, X_aux

def test_memory_usage():
    """Test memory usage with different configurations."""
    
    # Test parameters
    n_samples = 2000
    n_genes = 10000
    n_programs = 200
    n_iter = 100
    
    print("=" * 60)
    print("MEMORY OPTIMIZATION TEST")
    print("=" * 60)
    
    # Create test data
    X, Y, X_aux = create_test_data(n_samples, n_genes, n_programs)
    
    # Test 1: Original configuration (memory intensive)
    print("\n" + "=" * 40)
    print("TEST 1: Original Configuration")
    print("=" * 40)
    
    log_memory("Before sampler creation")
    
    sampler1 = SpikeSlabGibbsSampler(
        X, Y, X_aux, n_programs,
        memory_efficient=False,
        use_float32=False,
        trace_max_samples=1000
    )
    
    log_memory("After sampler creation")
    
    start_time = time.time()
    results1 = sampler1.run(n_iter, save_traces=True)
    end_time = time.time()
    
    print(f"Original config time: {end_time - start_time:.1f}s")
    log_memory("After original sampling")
    
    # Clean up
    del sampler1, results1
    gc.collect()
    log_memory("After cleanup")
    
    # Test 2: Memory efficient configuration
    print("\n" + "=" * 40)
    print("TEST 2: Memory Efficient Configuration")
    print("=" * 40)
    
    log_memory("Before memory-efficient sampler creation")
    
    sampler2 = SpikeSlabGibbsSampler(
        X, Y, X_aux, n_programs,
        memory_efficient=True,
        use_float32=True,
        trace_max_samples=100  # Keep fewer samples in memory
    )
    
    log_memory("After memory-efficient sampler creation")
    
    start_time = time.time()
    results2 = sampler2.run(n_iter, save_traces=True)
    end_time = time.time()
    
    print(f"Memory-efficient config time: {end_time - start_time:.1f}s")
    log_memory("After memory-efficient sampling")
    
    # Clean up
    del sampler2, results2
    gc.collect()
    log_memory("After cleanup")
    
    # Test 3: No trace saving (minimal memory)
    print("\n" + "=" * 40)
    print("TEST 3: No Trace Saving")
    print("=" * 40)
    
    log_memory("Before no-trace sampler creation")
    
    sampler3 = SpikeSlabGibbsSampler(
        X, Y, X_aux, n_programs,
        memory_efficient=True,
        use_float32=True
    )
    
    log_memory("After no-trace sampler creation")
    
    start_time = time.time()
    results3 = sampler3.run(n_iter, save_traces=False)
    end_time = time.time()
    
    print(f"No-trace config time: {end_time - start_time:.1f}s")
    log_memory("After no-trace sampling")
    
    # Clean up
    del sampler3, results3, X, Y, X_aux
    gc.collect()
    log_memory("Final cleanup")

def test_sparse_vs_dense():
    """Test memory usage with sparse vs dense matrices."""
    
    print("\n" + "=" * 60)
    print("SPARSE VS DENSE MATRIX TEST")
    print("=" * 60)
    
    n_samples = 1000
    n_genes = 5000
    n_programs = 50
    n_iter = 50
    
    # Create sparse data
    X_sparse, Y, X_aux = create_test_data(n_samples, n_genes, n_programs, sparsity=0.95)
    
    # Convert to dense
    X_dense = X_sparse.toarray()
    
    print(f"\nSparse matrix memory: {X_sparse.data.nbytes / (1024**2):.1f} MB")
    print(f"Dense matrix memory: {X_dense.nbytes / (1024**2):.1f} MB")
    print(f"Memory ratio (dense/sparse): {X_dense.nbytes / X_sparse.data.nbytes:.1f}x")
    
    # Test with sparse matrix
    print("\nTesting with sparse matrix:")
    log_memory("Before sparse test")
    
    sampler_sparse = SpikeSlabGibbsSampler(
        X_sparse, Y, X_aux, n_programs,
        memory_efficient=True,
        use_float32=True
    )
    
    results_sparse = sampler_sparse.run(n_iter, save_traces=False)
    log_memory("After sparse test")
    
    del sampler_sparse, results_sparse
    gc.collect()
    
    # Test with dense matrix
    print("\nTesting with dense matrix:")
    log_memory("Before dense test")
    
    sampler_dense = SpikeSlabGibbsSampler(
        X_dense, Y, X_aux, n_programs,
        memory_efficient=True,
        use_float32=True
    )
    
    results_dense = sampler_dense.run(n_iter, save_traces=False)
    log_memory("After dense test")
    
    del sampler_dense, results_dense, X_sparse, X_dense, Y, X_aux
    gc.collect()
    log_memory("Final cleanup")

if __name__ == "__main__":
    print("Starting memory optimization tests...")
    
    try:
        test_memory_usage()
        test_sparse_vs_dense()
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 