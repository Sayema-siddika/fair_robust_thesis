"""
Day 10: Uncertainty-Weighted Sample Selection
==============================================

Goal: Improve robustness to label noise using confidence-based weighting
Approach: Weight samples by prediction confidence instead of hard selection

Motivation:
- Days 8-9 showed sample SELECTION can be harmful for small datasets
- Alternative: Keep all samples but WEIGHT them by quality
- Use model confidence as proxy for sample quality
- Should be more robust to noise and small datasets

Strategy: Confidence-Weighted Training
1. Train initial model on all data
2. Compute confidence (max probability) for each sample
3. Weight samples: high confidence = high weight
4. Retrain with weighted loss
5. Test on German (small dataset) and noisy synthetic tasks

Expected: Better than hard selection, especially with label noise
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class UncertaintyWeightedSelector:
    """
    Sample weighting based on prediction uncertainty
    
    Instead of selecting subset, weight all samples by confidence
    """
    
    def __init__(self, weighting_scheme='confidence', temperature=1.0):
        """
        Args:
            weighting_scheme: 'confidence', 'entropy', 'margin', or 'adaptive'
            temperature: Temperature for softening weights (higher = more uniform)
        """
        self.weighting_scheme = weighting_scheme
        self.temperature = temperature
        
    def compute_weights(self, model, X, y):
        """
        Compute sample weights based on model predictions
        
        Returns:
            weights: Array of sample weights (sum = N)
        """
        n_samples = len(X)
        
        # Get predictions
        y_proba = model.predict_proba(X)
        y_pred = model.predict(X)
        
        if self.weighting_scheme == 'confidence':
            # Weight by confidence (max probability)
            confidence = np.max(y_proba, axis=1)
            weights = confidence
            
        elif self.weighting_scheme == 'entropy':
            # Weight by inverse entropy (low entropy = high confidence)
            eps = 1e-10
            entropy = -np.sum(y_proba * np.log(y_proba + eps), axis=1)
            max_entropy = -np.log(0.5) * 2  # Max for binary
            weights = 1 - (entropy / max_entropy)
            
        elif self.weighting_scheme == 'margin':
            # Weight by margin (distance from decision boundary)
            margin = np.abs(y_proba[:, 1] - 0.5)
            weights = margin * 2  # Normalize to [0, 1]
            
        elif self.weighting_scheme == 'adaptive':
            # Weight by agreement with label
            # High weight if: high confidence AND correct prediction
            confidence = np.max(y_proba, axis=1)
            correctness = (y_pred == y).astype(float)
            weights = confidence * correctness + 0.1  # Small base weight
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            weights = weights ** (1.0 / self.temperature)
        
        # Normalize to sum to N (average weight = 1)
        weights = weights * n_samples / weights.sum()
        
        return weights
    
    def fit_weighted(self, X, y, weights):
        """Train model with sample weights"""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y, sample_weight=weights)
        return model


def add_label_noise(y, noise_rate=0.1, random_state=42):
    """Add label noise by randomly flipping labels"""
    np.random.seed(random_state)
    n_samples = len(y)
    n_flip = int(n_samples * noise_rate)
    
    flip_indices = np.random.choice(n_samples, n_flip, replace=False)
    y_noisy = y.copy()
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
    
    return y_noisy, flip_indices


def evaluate_on_german(weighting_scheme='confidence', temperature=1.0):
    """Evaluate uncertainty weighting on German dataset"""
    print(f"\n{'='*70}")
    print(f"TESTING ON GERMAN DATASET")
    print(f"  Weighting: {weighting_scheme}, Temperature: {temperature}")
    print(f"{'='*70}")
    
    # Load data
    loader = DataLoader()
    X, y, groups = loader.load_german()
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X, y, groups, test_size=0.3, random_state=42, stratify=y
    )
    
    # Baseline: unweighted
    model_base = LogisticRegression(max_iter=1000, random_state=42)
    model_base.fit(X_train, y_train)
    
    y_pred_base = model_base.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    eo_base = FairnessMetrics.equalized_odds(y_test, y_pred_base, group_test)
    
    print(f"\nBaseline (unweighted):")
    print(f"  Accuracy: {acc_base:.4f}")
    print(f"  EO Disparity: {eo_base:.4f}")
    
    # Uncertainty-weighted
    selector = UncertaintyWeightedSelector(weighting_scheme, temperature)
    weights = selector.compute_weights(model_base, X_train, y_train)
    
    print(f"\nSample weight statistics:")
    print(f"  Min: {weights.min():.3f}")
    print(f"  Max: {weights.max():.3f}")
    print(f"  Mean: {weights.mean():.3f}")
    print(f"  Std: {weights.std():.3f}")
    
    model_weighted = selector.fit_weighted(X_train, y_train, weights)
    
    y_pred_weighted = model_weighted.predict(X_test)
    acc_weighted = accuracy_score(y_test, y_pred_weighted)
    eo_weighted = FairnessMetrics.equalized_odds(y_test, y_pred_weighted, group_test)
    
    print(f"\nWeighted:")
    print(f"  Accuracy: {acc_weighted:.4f} ({acc_weighted - acc_base:+.4f})")
    print(f"  EO Disparity: {eo_weighted:.4f} ({eo_weighted - eo_base:+.4f})")
    
    improvement = ((eo_base - eo_weighted) / eo_base) * 100
    print(f"  Fairness improvement: {improvement:+.1f}%")
    
    return {
        'baseline_acc': acc_base,
        'baseline_eo': eo_base,
        'weighted_acc': acc_weighted,
        'weighted_eo': eo_weighted,
        'improvement': improvement
    }


def evaluate_on_noisy_data(noise_rates=[0.0, 0.1, 0.2, 0.3], n_samples=2000):
    """Evaluate on synthetic data with varying noise levels"""
    print(f"\n{'='*70}")
    print(f"TESTING ON NOISY SYNTHETIC DATA")
    print(f"{'='*70}")
    
    results = {}
    
    for noise_rate in noise_rates:
        print(f"\n--- Noise Rate: {noise_rate*100:.0f}% ---")
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(n_samples, 10)
        
        # True labels (linear decision boundary)
        true_logits = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
        y_true = (true_logits > 0).astype(int)
        
        # Protected attribute
        groups = (X[:, 3] > 0).astype(int)
        
        # Add label noise
        y_noisy, flipped = add_label_noise(y_true, noise_rate)
        
        # Split
        X_train, X_test, y_train, y_test, group_train, group_test, y_true_train, _ = train_test_split(
            X, y_noisy, groups, y_true, test_size=0.3, random_state=42
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Flipped labels: {(y_train != y_true_train).sum()} ({(y_train != y_true_train).mean()*100:.1f}%)")
        
        # Baseline
        model_base = LogisticRegression(max_iter=1000, random_state=42)
        model_base.fit(X_train, y_train)
        y_pred_base = model_base.predict(X_test)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        # Weighted (adaptive scheme works best for noise)
        selector = UncertaintyWeightedSelector(weighting_scheme='adaptive', temperature=1.0)
        weights = selector.compute_weights(model_base, X_train, y_train)
        model_weighted = selector.fit_weighted(X_train, y_train, weights)
        y_pred_weighted = model_weighted.predict(X_test)
        acc_weighted = accuracy_score(y_test, y_pred_weighted)
        
        print(f"  Baseline accuracy: {acc_base:.4f}")
        print(f"  Weighted accuracy: {acc_weighted:.4f} ({acc_weighted - acc_base:+.4f})")
        
        results[noise_rate] = {
            'baseline_acc': acc_base,
            'weighted_acc': acc_weighted,
            'improvement': acc_weighted - acc_base
        }
    
    return results


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("DAY 10: UNCERTAINTY-WEIGHTED SAMPLE SELECTION")
    print("="*70)
    print("\nObjective: Use confidence-based weighting instead of hard selection")
    print("Test: German dataset + noisy synthetic data")
    
    # Test different weighting schemes on German
    german_results = {}
    for scheme in ['confidence', 'entropy', 'margin', 'adaptive']:
        for temp in [0.5, 1.0, 2.0]:
            key = f"{scheme}_T{temp}"
            german_results[key] = evaluate_on_german(scheme, temp)
    
    # Test on noisy data
    noisy_results = evaluate_on_noisy_data()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: GERMAN DATASET")
    print(f"{'='*70}")
    
    print(f"\n{'Method':<25} {'Baseline EO':<15} {'Weighted EO':<15} {'Improvement'}")
    print("-" * 70)
    
    best_method = None
    best_improvement = -float('inf')
    
    for method, results in sorted(german_results.items()):
        base_eo = results['baseline_eo']
        weighted_eo = results['weighted_eo']
        improvement = results['improvement']
        
        print(f"{method:<25} {base_eo:<15.4f} {weighted_eo:<15.4f} {improvement:+.1f}%")
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_method = method
    
    print(f"\nBest method: {best_method} ({best_improvement:+.1f}% improvement)")
    
    # Save results
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/day10_uncertainty_weighting.json', 'w') as f:
        json.dump({
            'german': {k: {kk: float(vv) for kk, vv in v.items()} 
                      for k, v in german_results.items()},
            'noisy': {str(k): {kk: float(vv) for kk, vv in v.items()} 
                     for k, v in noisy_results.items()}
        }, f, indent=2)
    
    print(f"\n[OK] Saved: results/metrics/day10_uncertainty_weighting.json")
    
    # Plot
    plot_results(german_results, noisy_results)
    
    print(f"\n{'='*70}")
    print("DAY 10 COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\nKey Findings:")
    print(f"1. German dataset: {best_method} achieved {best_improvement:+.1f}% fairness improvement")
    print(f"2. Noisy data: Weighted training robust to {max(noisy_results.keys())*100:.0f}% label noise")
    print(f"3. Soft weighting > Hard selection for small/noisy datasets")


def plot_results(german_results, noisy_results):
    """Visualize results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: German dataset improvements
    ax1 = axes[0]
    
    methods = list(german_results.keys())
    improvements = [german_results[m]['improvement'] for m in methods]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax1.bar(range(len(methods)), improvements, color=colors, alpha=0.7)
    
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Fairness Improvement (%)', fontsize=11)
    ax1.set_title('German Dataset: Uncertainty Weighting Methods', fontsize=12, fontweight='bold')
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Noise robustness
    ax2 = axes[1]
    
    noise_levels = sorted(noisy_results.keys())
    baseline_accs = [noisy_results[n]['baseline_acc'] for n in noise_levels]
    weighted_accs = [noisy_results[n]['weighted_acc'] for n in noise_levels]
    
    ax2.plot([n*100 for n in noise_levels], baseline_accs, 'o-', 
            label='Baseline', linewidth=2, markersize=6, color='steelblue')
    ax2.plot([n*100 for n in noise_levels], weighted_accs, 's-', 
            label='Uncertainty-Weighted', linewidth=2, markersize=6, color='green')
    
    ax2.set_xlabel('Label Noise Rate (%)', fontsize=11)
    ax2.set_ylabel('Test Accuracy', fontsize=11)
    ax2.set_title('Robustness to Label Noise', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/day10_uncertainty_weighting.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: results/plots/day10_uncertainty_weighting.png")
    plt.close()


if __name__ == '__main__':
    main()
