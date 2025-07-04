#!/usr/bin/env python3
"""
Quick test script to validate the Bayesian model on a small subset of data.

This script creates small test datasets and runs quick experiments to validate
that the model is working correctly before running on full datasets.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import quick_test_experiment, prepare_test_emtab_dataset


def test_emtab_unmasked():
    """Test EMTAB dataset with unmasked configuration."""
    print("Testing EMTAB dataset with unmasked configuration...")
    
    results = quick_test_experiment(
        dataset_name='emtab_test',
        n_samples=150,  # Small sample size
        n_genes=500,    # Moderate number of genes
        configuration='unmasked',
        d=8,            # Fewer gene programs for quick test
        max_iters=300,  # Fewer iterations
        burn_in=150,
        random_state=42
    )
    
    return results


def test_emtab_masked():
    """Test EMTAB dataset with masked (pathway) configuration."""
    print("Testing EMTAB dataset with masked configuration...")
    
    results = quick_test_experiment(
        dataset_name='emtab_test',
        n_samples=150,
        n_genes=800,     # More genes to capture pathways better
        configuration='masked',
        max_iters=300,
        burn_in=150,
        random_state=42
    )
    
    return results


def test_ajm_cyto():
    """Test AJM cytokine dataset."""
    print("Testing AJM cytokine dataset...")
    
    results = quick_test_experiment(
        dataset_name='ajm_cyto_test',
        n_samples=100,   # Smaller sample for AJM
        n_genes=400,     # Fewer genes
        configuration='unmasked',
        d=6,
        max_iters=200,
        burn_in=100,
        random_state=42
    )
    
    return results


def test_convergence_checking():
    """Test convergence checking functionality."""
    print("Testing convergence checking with multiple chains...")
    
    # Import here to avoid circular imports
    from run_experiments import run_sampler_and_evaluate
    from data import prepare_test_emtab_dataset
    import scipy.sparse as sp
    
    # Create small test dataset
    adata_test = prepare_test_emtab_dataset(n_samples=100, n_genes=300, random_state=42)
    
    X = adata_test.X.toarray() if sp.issparse(adata_test.X) else adata_test.X
    Y = adata_test.obs[["Crohn's disease", "ulcerative colitis"]].values
    X_aux = adata_test.obs[["age", "sex_female"]].values
    gene_names = adata_test.var_names.tolist()
    
    # Run with convergence checking
    results = run_sampler_and_evaluate(
        X=X,
        Y=Y,
        X_aux=X_aux,
        n_programs=5,  # Very few programs for quick convergence
        configuration='unmasked',
        pathways_dict=None,
        gene_names=gene_names,
        cyto_seed_genes=None,
        max_iters=500,  # Higher max to test early stopping
        burn_in=100,
        output_dir='test_results',
        experiment_name='convergence_test',
        random_state=42,
        n_chains=3,  # Multiple chains for convergence checking
        check_convergence=True,
        convergence_check_interval=25,  # Check frequently
        convergence_patience=2,  # Stop quickly when converged
        rhat_threshold=1.15,  # Slightly more lenient for quick test
        ess_threshold=200   # Lower ESS threshold for quick test
    )
    
    return results


def main():
    """Run all quick tests."""
    
    print("="*80)
    print("QUICK TEST SUITE FOR BAYESIAN MODEL")
    print("="*80)
    print("This will run several small experiments to validate the model functionality.")
    print()
    
    tests_to_run = [
        ("EMTAB Unmasked", test_emtab_unmasked),
        ("EMTAB Masked", test_emtab_masked), 
        ("AJM Cytokine", test_ajm_cyto),
        ("Convergence Checking", test_convergence_checking)
    ]
    
    results = {}
    
    for test_name, test_func in tests_to_run:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print('='*60)
            
            result = test_func()
            results[test_name] = result
            
            print(f"✓ {test_name} completed successfully!")
            
            # Print key metrics
            if 'test_metrics' in result:
                print(f"Final iterations: {result['actual_iterations']}/{result['max_iters']}")
                if result['early_stopped']:
                    print("✓ Early stopping achieved")
                
                for label_key, metrics in result['test_metrics'].items():
                    auc = metrics.get('auc', 'N/A')
                    acc = metrics.get('accuracy', 'N/A')
                    print(f"  {label_key}: AUC={auc:.3f}, Accuracy={acc:.3f}")
            
        except Exception as e:
            print(f"✗ {test_name} failed: {str(e)}")
            results[test_name] = None
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    
    successful_tests = sum(1 for result in results.values() if result is not None)
    total_tests = len(results)
    
    print(f"Tests passed: {successful_tests}/{total_tests}")
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result is not None else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    if successful_tests == total_tests:
        print("\n🎉 All tests passed! The model is ready for full experiments.")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} test(s) failed. Please check the issues above.")
    
    return results


if __name__ == "__main__":
    # You can also run individual tests by calling them directly
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "emtab_unmasked":
            test_emtab_unmasked()
        elif test_name == "emtab_masked":
            test_emtab_masked()
        elif test_name == "ajm_cyto":
            test_ajm_cyto()
        elif test_name == "convergence":
            test_convergence_checking()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: emtab_unmasked, emtab_masked, ajm_cyto, convergence")
    else:
        # Run all tests
        main() 