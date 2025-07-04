# Quick Test Suite for Bayesian Model

This directory contains helper functions and scripts to create small test datasets and run quick validation experiments before running the full Bayesian model on large datasets.

## 🚀 Quick Start

### Run Basic Test
```bash
# Run a simple test with default parameters
python test_quick_experiment.py

# Or run individual tests
python test_quick_experiment.py emtab_unmasked
python test_quick_experiment.py convergence
```

### Use in Python
```python
from data import quick_test_experiment

# Quick test with default parameters  
results = quick_test_experiment()

# Custom test parameters
results = quick_test_experiment(
    dataset_name='emtab_test',
    n_samples=100,
    n_genes=500, 
    configuration='masked',
    max_iters=200
)
```

## 📁 Files

- **`test_quick_experiment.py`** - Main test script with predefined test scenarios
- **`quick_test_examples.py`** - Examples showing different ways to use the functionality
- **`data.py`** - Contains the core helper functions (see functions below)
- **`convergence_checking.py`** - MCMC convergence diagnostics

## 🛠️ Core Functions

### `quick_test_experiment()`
Run a complete quick experiment with sensible defaults.

**Parameters:**
- `dataset_name`: `'emtab_test'`, `'ajm_cyto_test'`, `'ajm_ap_test'`
- `n_samples`: Number of samples (default: 200)
- `n_genes`: Number of genes (default: 1000)
- `configuration`: `'unmasked'`, `'masked'`, `'combined'`, `'mask_initiated'`
- `max_iters`: Maximum iterations (default: 200)
- Other standard model parameters...

### `prepare_test_emtab_dataset()`
Create a small version of the EMTAB dataset.

### `create_test_sample()`
Create a test sample from any AnnData object with smart gene selection.

## 🧪 Test Scenarios

### 1. Basic Validation Tests
```bash
# Test all configurations quickly
python test_quick_experiment.py
```

**What it tests:**
- ✅ EMTAB dataset with unmasked configuration
- ✅ EMTAB dataset with masked (pathway) configuration  
- ✅ AJM cytokine dataset
- ✅ Convergence checking with multiple chains

### 2. Custom Tests
```python
# Very small test for debugging
results = quick_test_experiment(
    n_samples=50,
    n_genes=200,
    max_iters=100,
    d=5
)

# Test specific configuration
results = quick_test_experiment(
    configuration='combined',
    d=5,  # Additional programs beyond pathways
    n_genes=600,
    max_iters=300
)
```

### 3. Dataset Creation Examples
```python
from data import create_test_sample, prepare_and_load_emtab

# Load full dataset and create test sample
full_data = prepare_and_load_emtab()
test_data = create_test_sample(
    full_data,
    n_samples=150,
    n_genes=800,
    prioritize_pathway_genes=True
)
```

## 🎯 Gene Selection Strategy

The test sampling uses a smart gene selection strategy:

1. **Pathway genes** (70% of selection): Genes that appear in the pathway dictionary
2. **High-variance genes** (30% of selection): Genes with highest variance in the dataset
3. **Expression filter**: Only genes with mean expression above threshold

This ensures test datasets contain biologically relevant and informative genes.

## 📊 Sample Output

```
============================================================
QUICK TEST EXPERIMENT  
============================================================
Loading full EMTAB dataset...
Filtering to protein-coding genes...
Creating test sample...
Creating test sample from dataset with shape (1515, 14420)
Sampled 200 samples:
  - Crohn's positive: 50
  - UC positive: 50  
  - Others: 100
Selected 156 pathway genes
Selected 344 high-variance genes
Total genes selected: 500

Running test with:
  - Dataset: emtab_test
  - Data shape: X=(200, 500), Y=(200, 2), X_aux=(200, 2)
  - Configuration: unmasked
  - Max iterations: 200

Iteration 50/200
Convergence achieved at iteration 75 (1/2)
Convergence achieved at iteration 100 (2/2)
Stopping early due to convergence!

✓ Early stopping achieved
Test Performance:
  test label_0: AUC=0.823, Accuracy=0.775
  test label_1: AUC=0.791, Accuracy=0.725
```

## ⚡ Performance Guidelines

### Recommended Test Sizes:

| Purpose | Samples | Genes | Max Iters | Expected Time |
|---------|---------|-------|-----------|---------------|
| **Debug** | 50 | 200 | 100 | 1-2 minutes |
| **Quick validation** | 100-200 | 400-800 | 200-300 | 3-5 minutes |
| **Pre-production test** | 300-500 | 1000-1500 | 500 | 10-15 minutes |

### Configuration Complexity:
- **Fastest**: `unmasked` with d=5-10
- **Medium**: `masked` (depends on pathway count)
- **Slowest**: `combined` and `mask_initiated`

## 🔧 Troubleshooting

### Common Issues:

1. **Out of memory**: Reduce `n_samples` and `n_genes`
2. **Slow convergence**: Reduce `d` (gene programs) or increase `max_iters`
3. **Import errors**: Make sure you're in the Gibbs directory

### Debug Mode:
```python
# Minimal test for debugging
results = quick_test_experiment(
    n_samples=30,
    n_genes=100, 
    max_iters=50,
    d=3,
    random_state=42
)
```

## 📈 Next Steps

Once your quick tests pass:

1. **Scale up gradually**: 500 samples → 1000 samples → full dataset
2. **Tune hyperparameters**: Use test results to optimize burn-in, iterations
3. **Run production experiments**: Use full `run_experiments.py` with optimized settings

## 🎛️ Advanced Usage

### Test Multiple Configurations:
```python
from quick_test_examples import example_3_test_different_configurations
results = example_3_test_different_configurations()
```

### Custom Convergence Testing:
```python
from quick_test_examples import example_5_convergence_testing
results_strict, results_complex = example_5_convergence_testing()
```

### Run All Examples:
```bash
python quick_test_examples.py
```

---

💡 **Tip**: Always run quick tests first to validate your setup before committing to long-running experiments on full datasets! 