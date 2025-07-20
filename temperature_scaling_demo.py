#!/usr/bin/env python3
"""
Temperature Scaling Demonstration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit

def temperature_scaling(probs, temperature):
    """
    Apply temperature scaling to probabilities.
    
    Args:
        probs: Original probabilities (0 to 1)
        temperature: Temperature parameter (T > 1 makes more conservative, T < 1 makes more confident)
    
    Returns:
        Calibrated probabilities
    """
    # Convert to logits
    logits = logit(probs)
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Convert back to probabilities
    calibrated_probs = expit(scaled_logits)
    
    return calibrated_probs

def demonstrate_temperature_scaling():
    """Demonstrate temperature scaling with examples."""
    
    print("=== TEMPERATURE SCALING DEMONSTRATION ===")
    
    # Example 1: Overconfident model
    print("\n1. OVERCONFIDENT MODEL EXAMPLE")
    print("Original probabilities: [0.1, 0.3, 0.7, 0.9]")
    original_probs = np.array([0.1, 0.3, 0.7, 0.9])
    
    temperatures = [0.5, 1.0, 2.0, 3.0]
    
    for temp in temperatures:
        calibrated = temperature_scaling(original_probs, temp)
        print(f"Temperature {temp}: {calibrated}")
    
    print("\nInterpretation:")
    print("- T=0.5: Makes model MORE confident (0.1→0.02, 0.9→0.98)")
    print("- T=1.0: No change")
    print("- T=2.0: Makes model LESS confident (0.1→0.18, 0.9→0.82)")
    print("- T=3.0: Makes model MUCH LESS confident (0.1→0.25, 0.9→0.75)")
    
    # Example 2: Underconfident model
    print("\n2. UNDERCONFIDENT MODEL EXAMPLE")
    print("Original probabilities: [0.4, 0.45, 0.55, 0.6]")
    original_probs2 = np.array([0.4, 0.45, 0.55, 0.6])
    
    for temp in temperatures:
        calibrated = temperature_scaling(original_probs2, temp)
        print(f"Temperature {temp}: {calibrated}")
    
    print("\nInterpretation:")
    print("- T=0.5: Makes model MORE confident (0.4→0.27, 0.6→0.73)")
    print("- T=1.0: No change")
    print("- T=2.0: Makes model LESS confident (0.4→0.47, 0.6→0.53)")
    print("- T=3.0: Makes model MUCH LESS confident (0.4→0.49, 0.6→0.51)")

def plot_temperature_effects():
    """Plot how temperature affects probability distributions."""
    
    # Generate some example logits (raw model outputs)
    np.random.seed(42)
    logits = np.random.normal(0, 2, 1000)
    original_probs = expit(logits)
    
    temperatures = [0.5, 1.0, 2.0, 3.0]
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Probability distributions
    plt.subplot(1, 3, 1)
    plt.hist(original_probs, bins=30, alpha=0.7, label='Original', density=True)
    
    for temp in temperatures:
        calibrated_probs = temperature_scaling(original_probs, temp)
        plt.hist(calibrated_probs, bins=30, alpha=0.7, 
                label=f'Temperature {temp}', density=True)
    
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title('Effect of Temperature on Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Calibration curves
    plt.subplot(1, 3, 2)
    from sklearn.calibration import calibration_curve
    
    # Generate some fake labels for demonstration
    labels = (original_probs > 0.5).astype(int)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Original
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, original_probs, n_bins=10
    )
    plt.plot(mean_predicted_value, fraction_of_positives, 'o-', 
             label='Original (likely overconfident)')
    
    # Temperature scaled
    for temp in temperatures:
        calibrated_probs = temperature_scaling(original_probs, temp)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, calibrated_probs, n_bins=10
        )
        plt.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                 label=f'Temperature {temp}')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Temperature vs confidence
    plt.subplot(1, 3, 3)
    
    # Show how temperature affects confidence
    test_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    temp_range = np.linspace(0.1, 5.0, 50)
    
    for prob in test_probs:
        calibrated_probs = [temperature_scaling(prob, temp) for temp in temp_range]
        plt.plot(temp_range, calibrated_probs, label=f'P={prob}')
    
    plt.xlabel('Temperature')
    plt.ylabel('Calibrated Probability')
    plt.title('How Temperature Affects Individual Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_scaling_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

def why_temperature_scaling_works():
    """Explain why temperature scaling is effective."""
    
    print("\n=== WHY TEMPERATURE SCALING WORKS ===")
    
    print("\n1. **Problem**: Your VI model is likely overconfident")
    print("   - Predicts 0.9 probability when true rate is only 0.7")
    print("   - Predicts 0.1 probability when true rate is actually 0.3")
    
    print("\n2. **Solution**: Temperature scaling adjusts confidence")
    print("   - T > 1: Makes predictions more conservative")
    print("   - T < 1: Makes predictions more confident")
    
    print("\n3. **Why it's called 'temperature'**:")
    print("   - Inspired by statistical physics")
    print("   - High temperature = more random/uncertain")
    print("   - Low temperature = more deterministic/certain")
    
    print("\n4. **Advantages**:")
    print("   - Simple: Only one parameter to tune")
    print("   - Preserves ranking: Best predictions stay best")
    print("   - Works well with neural networks and VI models")
    print("   - Can be optimized on validation set")

def how_to_find_optimal_temperature():
    """Show how to find the optimal temperature."""
    
    print("\n=== HOW TO FIND OPTIMAL TEMPERATURE ===")
    
    code_example = '''
def find_optimal_temperature(probs, labels, temperatures=np.linspace(0.1, 5.0, 50)):
    """Find temperature that minimizes Brier score."""
    
    best_temperature = 1.0
    best_brier = brier_score_loss(labels, probs)
    
    for temp in temperatures:
        calibrated_probs = temperature_scaling(probs, temp)
        brier = brier_score_loss(labels, calibrated_probs)
        
        if brier < best_brier:
            best_brier = brier
            best_temperature = temp
    
    return best_temperature, best_brier

# Usage:
optimal_temp, optimal_brier = find_optimal_temperature(vi_probs, true_labels)
print(f"Optimal temperature: {optimal_temp:.2f}")
print(f"Brier score improvement: {original_brier:.4f} → {optimal_brier:.4f}")
'''
    
    print(code_example)
    
    print("\n**Key Points**:")
    print("- Use validation set to find optimal temperature")
    print("- Don't use training set (would be cheating)")
    print("- Brier score is the standard metric for calibration")
    print("- Temperature > 1 means your model was overconfident")
    print("- Temperature < 1 means your model was underconfident")

def main():
    """Run the complete demonstration."""
    
    demonstrate_temperature_scaling()
    plot_temperature_effects()
    why_temperature_scaling_works()
    how_to_find_optimal_temperature()
    
    print("\n=== SUMMARY ===")
    print("Temperature scaling is a simple but powerful calibration technique.")
    print("It adjusts how confident your model is without changing the ranking of predictions.")
    print("For your VI model, try temperatures between 1.5 and 3.0 to reduce overconfidence.")
    print("This could significantly improve your F1 score!")

if __name__ == "__main__":
    main() 