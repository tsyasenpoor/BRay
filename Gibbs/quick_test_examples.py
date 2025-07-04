"""
Examples of how to use the quick test functionality.

This file contains various examples of how to create small test datasets
and run quick experiments to validate the Bayesian model before running
on full datasets.
"""

import numpy as np
from data import (
    prepare_test_emtab_dataset, 
    create_test_sample,
    quick_test_experiment,
    prepare_ajm_dataset,
    filter_protein_coding_genes,
    gene_annotation,
    pathways
)


def example_1_simple_test():
    """
    Example 1: Simple quick test with default parameters.
    """
    print("Example 1: Simple quick test")
    print("-" * 40)
    
    # Run a quick test with default parameters
    results = quick_test_experiment()
    
    print(f"Test completed in {results['actual_iterations']} iterations")
    print(f"Early stopped: {results['early_stopped']}")
    
    return results


def example_2_custom_test_parameters():
    """
    Example 2: Customized test parameters for specific validation.
    """
    print("Example 2: Custom test parameters")
    print("-" * 40)
    
    # Create a very small test for debugging
    results = quick_test_experiment(
        dataset_name='emtab_test',
        n_samples=50,     # Very small
        n_genes=200,      # Minimal genes
        configuration='unmasked',
        d=5,              # Few gene programs
        max_iters=100,    # Quick iterations
        burn_in=50,
        random_state=123
    )
    
    print(f"Small test completed: {results['actual_iterations']} iterations")
    
    return results


def example_3_test_different_configurations():
    """
    Example 3: Test all different model configurations quickly.
    """
    print("Example 3: Testing all configurations")
    print("-" * 40)
    
    configurations = ['unmasked', 'masked', 'combined', 'mask_initiated']
    results = {}
    
    for config in configurations:
        try:
            print(f"\nTesting {config} configuration...")
            
            if config == 'unmasked':
                result = quick_test_experiment(
                    configuration=config,
                    d=8,
                    n_samples=100,
                    n_genes=400,
                    max_iters=150
                )
            elif config == 'combined':
                result = quick_test_experiment(
                    configuration=config,
                    d=5,  # Additional programs beyond pathways
                    n_samples=100,
                    n_genes=600,  # More genes to capture pathways
                    max_iters=200
                )
            else:
                result = quick_test_experiment(
                    configuration=config,
                    n_samples=100,
                    n_genes=600,
                    max_iters=200
                )
            
            results[config] = result
            print(f"✓ {config}: {result['actual_iterations']} iterations")
            
        except Exception as e:
            print(f"✗ {config} failed: {e}")
            results[config] = None
    
    return results


def example_4_custom_dataset_creation():
    """
    Example 4: Create custom test datasets with specific characteristics.
    """
    print("Example 4: Custom dataset creation")
    print("-" * 40)
    
    # Load full EMTAB dataset
    from data import prepare_and_load_emtab
    emtab_full = prepare_and_load_emtab()
    emtab_filtered = filter_protein_coding_genes(emtab_full, gene_annotation)
    
    print(f"Full dataset shape: {emtab_filtered.shape}")
    
    # Create different sized test datasets
    test_sizes = [
        (50, 200),   # Tiny test
        (100, 500),  # Small test  
        (200, 1000)  # Medium test
    ]
    
    for n_samples, n_genes in test_sizes:
        print(f"\nCreating test dataset: {n_samples} samples, {n_genes} genes")
        
        test_data = create_test_sample(
            emtab_filtered,
            n_samples=n_samples,
            n_genes=n_genes,
            prioritize_pathway_genes=True,
            pathways_dict=pathways,
            random_state=42
        )
        
        print(f"Created: {test_data.shape}")
        
        # Show class distribution
        if "Crohn's disease" in test_data.obs.columns:
            cd_rate = np.mean(test_data.obs["Crohn's disease"])
            uc_rate = np.mean(test_data.obs["ulcerative colitis"])
            print(f"  CD rate: {cd_rate:.2%}, UC rate: {uc_rate:.2%}")


def example_5_convergence_testing():
    """
    Example 5: Test convergence checking with different settings.
    """
    print("Example 5: Convergence testing")
    print("-" * 40)
    
    # Test with strict convergence criteria
    print("Testing with strict convergence criteria...")
    results_strict = quick_test_experiment(
        n_samples=80,
        n_genes=300,
        configuration='unmasked',
        d=4,  # Few programs for quick convergence
        max_iters=400,
        burn_in=100,
        random_state=42
    )
    
    print(f"Strict test: {results_strict['actual_iterations']} iterations")
    print(f"Early stopped: {results_strict['early_stopped']}")
    
    # Compare with a test that should NOT converge quickly
    print("\nTesting with complex model (less likely to converge quickly)...")
    results_complex = quick_test_experiment(
        n_samples=150,
        n_genes=600,
        configuration='unmasked', 
        d=12,  # More programs - harder to converge
        max_iters=200,  # Fewer max iterations
        burn_in=100,
        random_state=42
    )
    
    print(f"Complex test: {results_complex['actual_iterations']} iterations")
    print(f"Early stopped: {results_complex['early_stopped']}")
    
    return results_strict, results_complex


def example_6_ajm_dataset_testing():
    """
    Example 6: Test with AJM datasets.
    """
    print("Example 6: AJM dataset testing")
    print("-" * 40)
    
    # Test AJM cytokine dataset
    print("Testing AJM cytokine dataset...")
    results_cyto = quick_test_experiment(
        dataset_name='ajm_cyto_test',
        n_samples=80,
        n_genes=300,
        configuration='unmasked',
        d=6,
        max_iters=150,
        burn_in=75
    )
    
    print(f"AJM cyto test completed: {results_cyto['actual_iterations']} iterations")
    
    # Show cytokine seed scores if available
    if 'test_predictions' in results_cyto and 'cyto_seed_scores' in results_cyto['test_predictions']:
        cyto_scores = results_cyto['test_predictions']['cyto_seed_scores']
        if cyto_scores is not None:
            print(f"Cytokine seed scores: mean={np.mean(cyto_scores):.3f}, std={np.std(cyto_scores):.3f}")
    
    return results_cyto


def run_all_examples():
    """
    Run all examples in sequence.
    """
    print("="*80)
    print("RUNNING ALL QUICK TEST EXAMPLES")
    print("="*80)
    
    examples = [
        ("Simple Test", example_1_simple_test),
        ("Custom Parameters", example_2_custom_test_parameters), 
        ("All Configurations", example_3_test_different_configurations),
        ("Custom Datasets", example_4_custom_dataset_creation),
        ("Convergence Testing", example_5_convergence_testing),
        ("AJM Testing", example_6_ajm_dataset_testing)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print('='*60)
            
            result = example_func()
            results[name] = result
            
            print(f"✓ {name} completed successfully!")
            
        except Exception as e:
            print(f"✗ {name} failed: {str(e)}")
            results[name] = None
    
    print(f"\n{'='*80}")
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        if example_num == "1":
            example_1_simple_test()
        elif example_num == "2":
            example_2_custom_test_parameters()
        elif example_num == "3":
            example_3_test_different_configurations()
        elif example_num == "4":
            example_4_custom_dataset_creation()
        elif example_num == "5":
            example_5_convergence_testing()
        elif example_num == "6":
            example_6_ajm_dataset_testing()
        else:
            print(f"Unknown example number: {example_num}")
            print("Available examples: 1, 2, 3, 4, 5, 6")
    else:
        # Run all examples
        run_all_examples() 