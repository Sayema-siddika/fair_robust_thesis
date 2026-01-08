"""
Day 15: Hybrid Meta-Learning + Uncertainty Weighting
====================================================

Combine the best of both worlds:
1. Meta-learning: Learns selection policy from multiple tasks
2. Uncertainty weighting: Soft weighting based on sample quality

Hypothesis: Hybrid approach can leverage meta-learned selection + 
            adaptive weighting for superior fairness-accuracy trade-off.

Approach:
- Use meta-selector to compute selection probabilities
- Convert probabilities to sample weights (soft version)
- Apply adaptive weighting on top
- Compare: meta-only, adaptive-only, hybrid
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class HybridSelector:
    """Hybrid meta-learning + uncertainty weighting selector."""
    
    def __init__(self, temperature=1.0, alpha=0.5):
        """
        Args:
            temperature: Controls weight smoothness
            alpha: Blend factor (0=pure adaptive, 1=pure meta, 0.5=equal blend)
        """
        self.temperature = temperature
        self.alpha = alpha
        self.results = {}
    
    def compute_adaptive_weights(self, X_train, y_train):
        """Compute uncertainty-based adaptive weights."""
        # Train initial model
        model_init = LogisticRegression(max_iter=1000, random_state=42)
        model_init.fit(X_train, y_train)
        
        # Compute weights
        probs = model_init.predict_proba(X_train)
        max_probs = np.max(probs, axis=1)
        predictions = model_init.predict(X_train)
        correctness = (predictions == y_train).astype(float)
        
        weights = max_probs * correctness + 0.1
        return weights
    
    def compute_meta_weights(self, X_train, y_train):
        """
        Simulate meta-learning weights.
        In practice, this would use pre-trained meta-selector.
        For now, use greedy selection as proxy.
        """
        # Train model for loss computation
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Compute loss-based weights (lower loss = higher weight)
        losses = -model.predict_log_proba(X_train)
        sample_losses = np.array([losses[i, y_train[i]] for i in range(len(y_train))])
        
        # Convert to weights (inverse of loss, normalized)
        weights = 1.0 / (sample_losses + 0.1)
        return weights
    
    def compute_hybrid_weights(self, X_train, y_train):
        """Combine meta and adaptive weights."""
        adaptive_weights = self.compute_adaptive_weights(X_train, y_train)
        meta_weights = self.compute_meta_weights(X_train, y_train)
        
        # Normalize both
        adaptive_weights = adaptive_weights / np.mean(adaptive_weights)
        meta_weights = meta_weights / np.mean(meta_weights)
        
        # Blend
        hybrid_weights = self.alpha * meta_weights + (1 - self.alpha) * adaptive_weights
        
        # Temperature scaling
        hybrid_weights = hybrid_weights ** (1.0 / self.temperature)
        
        # Normalize for training
        hybrid_weights = hybrid_weights / np.sum(hybrid_weights) * len(hybrid_weights)
        
        return hybrid_weights, adaptive_weights, meta_weights
    
    def train_and_evaluate(self, X_train, y_train, z_train, X_test, y_test, z_test, method='hybrid'):
        """Train model with specified weighting method."""
        if method == 'baseline':
            # No weighting
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
        elif method == 'adaptive':
            # Adaptive weighting only
            weights = self.compute_adaptive_weights(X_train, y_train)
            weights = weights ** (1.0 / self.temperature)
            weights = weights / np.sum(weights) * len(weights)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train, sample_weight=weights)
            
        elif method == 'meta':
            # Meta-learning weights only
            weights = self.compute_meta_weights(X_train, y_train)
            weights = weights ** (1.0 / self.temperature)
            weights = weights / np.sum(weights) * len(weights)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train, sample_weight=weights)
            
        elif method == 'hybrid':
            # Hybrid approach
            weights, _, _ = self.compute_hybrid_weights(X_train, y_train)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train, sample_weight=weights)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        eo = FairnessMetrics.equalized_odds(y_test, y_pred, z_test)
        
        return {
            'accuracy': float(acc),
            'eo_disparity': float(eo),
            'model': model
        }


def evaluate_hybrid_methods():
    """Comprehensive evaluation of hybrid approaches."""
    print("\n" + "="*70)
    print("DAY 15: HYBRID META-LEARNING + UNCERTAINTY WEIGHTING")
    print("="*70)
    
    results = {
        'datasets': {},
        'blending_analysis': {},
        'temperature_analysis': {}
    }
    
    # Test on all 3 datasets
    datasets = ['compas', 'adult', 'german']
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Load data
        loader = DataLoader(dataset_name)
        X, y, z = loader.load_dataset()
        
        X_train, X_test, y_train, y_test, z_train, z_test = \
            train_test_split(X, y, z, test_size=0.3, random_state=42, stratify=y)
        
        X_train, X_test = loader.preprocess(X_train, X_test)
        
        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Test different methods
        selector = HybridSelector(temperature=1.0, alpha=0.5)
        
        methods = ['baseline', 'adaptive', 'meta', 'hybrid']
        dataset_results = {}
        
        print(f"\n{'Method':<15} {'Accuracy':<12} {'EO Disparity':<15} {'vs Baseline':<12}")
        print("-" * 70)
        
        baseline_eo = None
        for method in methods:
            metrics = selector.train_and_evaluate(
                X_train, y_train, z_train, X_test, y_test, z_test, method=method
            )
            
            dataset_results[method] = metrics
            
            if method == 'baseline':
                baseline_eo = metrics['eo_disparity']
                improvement = 0.0
            else:
                improvement = (baseline_eo - metrics['eo_disparity']) / baseline_eo * 100
            
            print(f"{method.capitalize():<15} {metrics['accuracy']:<12.4f} "
                  f"{metrics['eo_disparity']:<15.4f} {improvement:+11.1f}%")
        
        results['datasets'][dataset_name] = dataset_results
        
        # Determine winner
        best_method = min(['adaptive', 'meta', 'hybrid'], 
                         key=lambda m: dataset_results[m]['eo_disparity'])
        best_improvement = (baseline_eo - dataset_results[best_method]['eo_disparity']) / baseline_eo * 100
        
        print(f"\n[WINNER] {best_method.capitalize()} with {best_improvement:+.1f}% improvement!")
    
    # Analyze alpha blending on Adult dataset
    print(f"\n{'='*70}")
    print("BLENDING ANALYSIS (Adult Dataset)")
    print(f"{'='*70}")
    
    loader = DataLoader('adult')
    X, y, z = loader.load_dataset()
    X_train, X_test, y_train, y_test, z_train, z_test = \
        train_test_split(X, y, z, test_size=0.3, random_state=42, stratify=y)
    X_train, X_test = loader.preprocess(X_train, X_test)
    
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    blending_results = {}
    
    print(f"\n{'Alpha (α)':<12} {'Meta %':<10} {'Adaptive %':<15} {'Accuracy':<12} {'EO Disparity':<15}")
    print("-" * 70)
    
    for alpha in alphas:
        selector = HybridSelector(temperature=1.0, alpha=alpha)
        metrics = selector.train_and_evaluate(
            X_train, y_train, z_train, X_test, y_test, z_test, method='hybrid'
        )
        
        blending_results[f'alpha_{alpha}'] = metrics
        
        print(f"{alpha:<12.1f} {alpha*100:<10.0f} {(1-alpha)*100:<15.0f} "
              f"{metrics['accuracy']:<12.4f} {metrics['eo_disparity']:<15.4f}")
    
    results['blending_analysis'] = blending_results
    
    # Find best alpha
    best_alpha_key = min(blending_results.keys(), 
                         key=lambda k: blending_results[k]['eo_disparity'])
    best_alpha = float(best_alpha_key.split('_')[1])
    print(f"\n[OPTIMAL] α = {best_alpha} (EO = {blending_results[best_alpha_key]['eo_disparity']:.4f})")
    
    # Analyze temperature on Adult with optimal alpha
    print(f"\n{'='*70}")
    print(f"TEMPERATURE ANALYSIS (Adult, α={best_alpha})")
    print(f"{'='*70}")
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    temp_results = {}
    
    print(f"\n{'Temperature':<15} {'Accuracy':<12} {'EO Disparity':<15} {'Weight Gini':<12}")
    print("-" * 70)
    
    for temp in temperatures:
        selector = HybridSelector(temperature=temp, alpha=best_alpha)
        
        # Get weights for analysis
        weights, _, _ = selector.compute_hybrid_weights(X_train, y_train)
        
        # Compute Gini coefficient
        sorted_weights = np.sort(weights)
        n = len(weights)
        cumsum = np.cumsum(sorted_weights)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_weights)) / (n * cumsum[-1]) - (n + 1) / n
        
        metrics = selector.train_and_evaluate(
            X_train, y_train, z_train, X_test, y_test, z_test, method='hybrid'
        )
        
        metrics['gini'] = float(gini)
        temp_results[f'T_{temp}'] = metrics
        
        print(f"{temp:<15.1f} {metrics['accuracy']:<12.4f} "
              f"{metrics['eo_disparity']:<15.4f} {gini:<12.3f}")
    
    results['temperature_analysis'] = temp_results
    
    # Find best temperature
    best_temp_key = min(temp_results.keys(), 
                        key=lambda k: temp_results[k]['eo_disparity'])
    best_temp = float(best_temp_key.split('_')[1])
    print(f"\n[OPTIMAL] T = {best_temp} (EO = {temp_results[best_temp_key]['eo_disparity']:.4f})")
    
    return results


def create_visualizations(results):
    """Create comprehensive hybrid method visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Day 15: Hybrid Meta-Learning + Uncertainty Weighting Analysis', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Cross-dataset comparison
    ax = axes[0, 0]
    datasets = ['compas', 'adult', 'german']
    methods = ['baseline', 'adaptive', 'meta', 'hybrid']
    colors = {'baseline': 'gray', 'adaptive': 'red', 'meta': 'blue', 'hybrid': 'green'}
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, method in enumerate(methods):
        eos = [results['datasets'][d][method]['eo_disparity'] for d in datasets]
        ax.bar(x + i*width - 0.3, eos, width, label=method.capitalize(), 
               color=colors[method], alpha=0.7)
    
    ax.set_ylabel('EO Disparity (lower is better)', fontsize=10)
    ax.set_title('Method Comparison Across Datasets', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Blending analysis
    ax = axes[0, 1]
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    accs = [results['blending_analysis'][f'alpha_{a}']['accuracy'] for a in alphas]
    eos = [results['blending_analysis'][f'alpha_{a}']['eo_disparity'] for a in alphas]
    
    ax2 = ax.twinx()
    line1 = ax.plot(alphas, accs, 'o-', color='blue', linewidth=2, 
                    markersize=8, label='Accuracy')
    line2 = ax2.plot(alphas, eos, 's-', color='red', linewidth=2, 
                     markersize=8, label='EO Disparity')
    
    ax.set_xlabel('Alpha (α): 0=Adaptive, 1=Meta', fontsize=10)
    ax.set_ylabel('Accuracy', color='blue', fontsize=10)
    ax2.set_ylabel('EO Disparity', color='red', fontsize=10)
    ax.set_title('Blending Factor Analysis (Adult)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=9)
    
    # Plot 3: Temperature analysis
    ax = axes[1, 0]
    temps = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    accs = [results['temperature_analysis'][f'T_{t}']['accuracy'] for t in temps]
    eos = [results['temperature_analysis'][f'T_{t}']['eo_disparity'] for t in temps]
    ginis = [results['temperature_analysis'][f'T_{t}']['gini'] for t in temps]
    
    ax2 = ax.twinx()
    line1 = ax.plot(temps, eos, 'o-', color='red', linewidth=2, 
                    markersize=8, label='EO Disparity')
    line2 = ax2.plot(temps, ginis, 's-', color='purple', linewidth=2, 
                     markersize=8, label='Weight Gini')
    
    ax.set_xlabel('Temperature (T)', fontsize=10)
    ax.set_ylabel('EO Disparity', color='red', fontsize=10)
    ax2.set_ylabel('Weight Gini', color='purple', fontsize=10)
    ax.set_title('Temperature Impact (Hybrid)', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=9)
    
    # Plot 4: Improvement summary
    ax = axes[1, 1]
    
    # Compute average improvements across datasets
    methods = ['adaptive', 'meta', 'hybrid']
    avg_improvements = []
    
    for method in methods:
        improvements = []
        for dataset in ['compas', 'adult', 'german']:
            baseline_eo = results['datasets'][dataset]['baseline']['eo_disparity']
            method_eo = results['datasets'][dataset][method]['eo_disparity']
            imp = (baseline_eo - method_eo) / baseline_eo * 100
            improvements.append(imp)
        avg_improvements.append(np.mean(improvements))
    
    bars = ax.bar(range(len(methods)), avg_improvements, 
                  color=['red', 'blue', 'green'], alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel('Average Fairness Improvement (%)', fontsize=10)
    ax.set_title('Overall Method Performance', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, avg_improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:+.1f}%',
               ha='center', va='bottom' if val > 0 else 'top', 
               fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    plot_path = Path('results/plots/day15_hybrid_methods.png')
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved visualization to {plot_path}")


def main():
    print("\n" + "="*70)
    print("DAY 15: HYBRID META-LEARNING + UNCERTAINTY WEIGHTING")
    print("="*70)
    print("\nObjective: Combine meta-learning and adaptive weighting")
    print("Question: Does hybrid approach beat individual methods?")
    
    # Run evaluation
    results = evaluate_hybrid_methods()
    
    # Create visualizations
    create_visualizations(results)
    
    # Save results
    results_path = Path('results/metrics/day15_hybrid_methods.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        # Remove model objects before saving
        clean_results = {
            'datasets': {},
            'blending_analysis': {},
            'temperature_analysis': {}
        }
        
        # Clean dataset results
        for dataset_name, dataset_data in results['datasets'].items():
            clean_results['datasets'][dataset_name] = {}
            for method, metrics in dataset_data.items():
                clean_results['datasets'][dataset_name][method] = {
                    k: v for k, v in metrics.items() if k != 'model'
                }
        
        # Clean blending results
        for key, metrics in results['blending_analysis'].items():
            clean_results['blending_analysis'][key] = {
                k: v for k, v in metrics.items() if k != 'model'
            }
        
        # Clean temperature results
        for key, metrics in results['temperature_analysis'].items():
            clean_results['temperature_analysis'][key] = {
                k: v for k, v in metrics.items() if k != 'model'
            }
        
        json.dump(clean_results, f, indent=2)
    
    print(f"\n[OK] Saved results to {results_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Hybrid Methods Analysis")
    print("="*70)
    
    # Cross-dataset summary
    print("\nCross-Dataset Performance:")
    for dataset in ['compas', 'adult', 'german']:
        print(f"\n{dataset.upper()}:")
        baseline_eo = results['datasets'][dataset]['baseline']['eo_disparity']
        
        for method in ['adaptive', 'meta', 'hybrid']:
            method_eo = results['datasets'][dataset][method]['eo_disparity']
            imp = (baseline_eo - method_eo) / baseline_eo * 100
            print(f"  {method.capitalize():<10}: {imp:+6.1f}% (EO={method_eo:.4f})")
    
    # Overall winner
    print("\n" + "-"*70)
    print("Overall Winner:")
    
    methods = ['adaptive', 'meta', 'hybrid']
    avg_improvements = {}
    
    for method in methods:
        improvements = []
        for dataset in ['compas', 'adult', 'german']:
            baseline_eo = results['datasets'][dataset]['baseline']['eo_disparity']
            method_eo = results['datasets'][dataset][method]['eo_disparity']
            imp = (baseline_eo - method_eo) / baseline_eo * 100
            improvements.append(imp)
        avg_improvements[method] = np.mean(improvements)
    
    winner = max(avg_improvements.keys(), key=lambda k: avg_improvements[k])
    print(f"  {winner.capitalize()}: {avg_improvements[winner]:+.1f}% average improvement")
    
    print("\n[OK] Day 15 complete - Hybrid methods analyzed!")
    print("Next: Day 16 - Temporal fairness analysis")


if __name__ == '__main__':
    main()
