"""
Day 16: Temporal Fairness Analysis

Research Question:
How does adaptive weighting sample selection evolve over training epochs?
Do the weights converge or keep changing? Which samples matter most over time?

Key Questions:
1. Do high-weight samples stay high throughout training?
2. Does fairness improve monotonically or oscillate?
3. Are there critical epochs where fairness jumps?
4. Which demographic groups get prioritized when?

Experiments:
1. Track sample weights across all epochs
2. Monitor fairness metrics per epoch
3. Visualize weight evolution for specific samples
4. Identify critical samples (always high/low weight)
5. Analyze group-wise weight distribution over time

Expected Insight:
If adaptive weighting works so well, understanding its temporal dynamics
will reveal WHY it's effective and guide future improvements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import json
from datetime import datetime

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class TemporalFairnessTracker:
    """
    Tracks how adaptive weighting affects fairness over training time.
    
    Unlike batch training (one shot), this simulates iterative training
    where weights are recomputed each epoch based on current model.
    """
    
    def __init__(self, temperature=0.5, n_epochs=50):
        self.temperature = temperature
        self.n_epochs = n_epochs
        
        # Storage for temporal data
        self.weight_history = []  # [epoch][sample_idx] -> weight
        self.fairness_history = []  # [epoch] -> metrics dict
        self.accuracy_history = []  # [epoch] -> accuracy
        self.model_history = []  # [epoch] -> model (for analysis)
        
    def compute_adaptive_weights(self, model, X, y, protected):
        """
        Compute adaptive sample weights based on current model.
        
        Weight = (confidence × correctness + 0.1) ** (1/T)
        
        Args:
            model: Current trained model
            X: Features
            y: Labels
            protected: Protected attributes (for group analysis)
            
        Returns:
            weights: Normalized sample weights
            meta: Dictionary with confidence, correctness, etc.
        """
        # Get predictions
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            confidence = np.max(probs, axis=1)
        else:
            # For models without predict_proba, use decision function
            decision = model.decision_function(X)
            confidence = 1.0 / (1.0 + np.exp(-np.abs(decision)))
        
        preds = model.predict(X)
        correctness = (preds == y).astype(float)
        
        # Compute raw weights
        raw_weights = confidence * correctness + 0.1
        
        # Temperature scaling
        if self.temperature != 1.0:
            weights = raw_weights ** (1.0 / self.temperature)
        else:
            weights = raw_weights
        
        # Normalize to sum to N (for weighting interpretation)
        weights = weights / np.sum(weights) * len(weights)
        
        # Compute group-wise statistics
        groups = np.unique(protected)
        group_stats = {}
        for g in groups:
            mask = (protected == g)
            group_stats[int(g)] = {
                'mean_weight': float(np.mean(weights[mask])),
                'std_weight': float(np.std(weights[mask])),
                'mean_confidence': float(np.mean(confidence[mask])),
                'mean_correctness': float(np.mean(correctness[mask])),
                'count': int(np.sum(mask))
            }
        
        meta = {
            'confidence': confidence,
            'correctness': correctness,
            'raw_weights': raw_weights,
            'group_stats': group_stats
        }
        
        return weights, meta
    
    def train_iterative(self, X_train, y_train, protected_train, 
                       X_test, y_test, protected_test):
        """
        Train model iteratively, recomputing weights each epoch.
        
        Unlike standard training (fit once with fixed weights),
        this updates weights based on current model at each epoch.
        
        Process:
        1. Epoch 0: Train baseline model (uniform weights)
        2. Epoch 1+: Compute adaptive weights → retrain → repeat
        
        Args:
            X_train, y_train, protected_train: Training data
            X_test, y_test, protected_test: Test data
            
        Returns:
            results: Dictionary with temporal data
        """
        n_samples = len(X_train)
        
        # Initialize storage
        self.weight_history = []
        self.fairness_history = []
        self.accuracy_history = []
        self.model_history = []
        
        print(f"Training iteratively for {self.n_epochs} epochs with T={self.temperature}")
        print(f"Training samples: {n_samples}")
        
        # Epoch 0: Baseline (uniform weights)
        print("\nEpoch 0: Baseline (uniform weights)")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        eo_disparity = FairnessMetrics.equalized_odds(y_test, y_pred, protected_test)
        dp_disparity = FairnessMetrics.demographic_parity(y_pred, protected_test)
        eop_disparity = FairnessMetrics.equal_opportunity(y_test, y_pred, protected_test)
        
        # Store uniform weights for epoch 0
        uniform_weights = np.ones(n_samples)
        self.weight_history.append(uniform_weights.copy())
        
        self.fairness_history.append({
            'epoch': 0,
            'accuracy': acc,
            'eo_disparity': eo_disparity,
            'demographic_parity': dp_disparity,
            'equal_opportunity': eop_disparity
        })
        self.accuracy_history.append(acc)
        
        print(f"  Accuracy: {acc:.4f}, EO Disparity: {eo_disparity:.4f}")
        
        # Epochs 1+: Adaptive weighting
        for epoch in range(1, self.n_epochs + 1):
            if epoch % 10 == 0 or epoch == 1:
                print(f"\nEpoch {epoch}:")
            
            # Compute adaptive weights based on current model
            weights, meta = self.compute_adaptive_weights(
                model, X_train, y_train, protected_train
            )
            
            # Store weights
            self.weight_history.append(weights.copy())
            
            # Train new model with updated weights
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train, sample_weight=weights)
            
            # Evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            eo_disparity = FairnessMetrics.equalized_odds(y_test, y_pred, protected_test)
            dp_disparity = FairnessMetrics.demographic_parity(y_pred, protected_test)
            eop_disparity = FairnessMetrics.equal_opportunity(y_test, y_pred, protected_test)
            
            # Store metrics
            fairness_entry = {
                'epoch': epoch,
                'accuracy': acc,
                'eo_disparity': eo_disparity,
                'demographic_parity': dp_disparity,
                'equal_opportunity': eop_disparity,
                'mean_weight': float(np.mean(weights)),
                'std_weight': float(np.std(weights)),
                'weight_gini': float(self._compute_gini(weights)),
                'group_stats': meta['group_stats']
            }
            
            self.fairness_history.append(fairness_entry)
            self.accuracy_history.append(acc)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Accuracy: {acc:.4f}, EO Disparity: {eo_disparity:.4f}")
                print(f"  Mean weight: {np.mean(weights):.3f}, Gini: {fairness_entry['weight_gini']:.3f}")
        
        # Analyze weight stability
        weight_stability = self._analyze_weight_stability()
        
        # Find critical samples
        critical_samples = self._find_critical_samples(protected_train)
        
        results = {
            'fairness_history': self.fairness_history,
            'weight_stability': weight_stability,
            'critical_samples': critical_samples,
            'final_epoch': self.n_epochs
        }
        
        return results
    
    def _compute_gini(self, weights):
        """Compute Gini coefficient of weight distribution."""
        sorted_weights = np.sort(weights)
        n = len(weights)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
        return gini
    
    def _analyze_weight_stability(self):
        """
        Analyze how stable sample weights are over time.
        
        Metrics:
        - Weight variance over time (per sample)
        - Rank correlation between epochs
        - Percentage of samples with stable high/low weights
        """
        weight_matrix = np.array(self.weight_history)  # [n_epochs, n_samples]
        n_epochs, n_samples = weight_matrix.shape
        
        # Variance over time (per sample)
        weight_variance = np.var(weight_matrix, axis=0)
        
        # Mean weight over time (per sample)
        mean_weights = np.mean(weight_matrix, axis=0)
        
        # Coefficient of variation (CV = std/mean)
        cv = np.std(weight_matrix, axis=0) / (mean_weights + 1e-8)
        
        # Identify stable samples (CV < 0.2)
        stable_mask = cv < 0.2
        stable_percentage = np.mean(stable_mask) * 100
        
        # Identify consistently high/low weight samples
        # High: mean weight > 75th percentile, CV < 0.2
        # Low: mean weight < 25th percentile, CV < 0.2
        high_threshold = np.percentile(mean_weights, 75)
        low_threshold = np.percentile(mean_weights, 25)
        
        stable_high = np.sum((mean_weights > high_threshold) & stable_mask)
        stable_low = np.sum((mean_weights < low_threshold) & stable_mask)
        
        # Compute rank correlation between consecutive epochs
        rank_correlations = []
        for i in range(1, n_epochs):
            from scipy.stats import spearmanr
            corr, _ = spearmanr(weight_matrix[i-1], weight_matrix[i])
            rank_correlations.append(corr)
        
        stability = {
            'stable_percentage': float(stable_percentage),
            'stable_high_count': int(stable_high),
            'stable_low_count': int(stable_low),
            'mean_rank_correlation': float(np.mean(rank_correlations)),
            'weight_variance_stats': {
                'mean': float(np.mean(weight_variance)),
                'std': float(np.std(weight_variance)),
                'median': float(np.median(weight_variance))
            }
        }
        
        return stability
    
    def _find_critical_samples(self, protected):
        """
        Identify samples that consistently get high/low weights.
        
        Critical high-weight: Top 10% mean weight, low variance
        Critical low-weight: Bottom 10% mean weight, low variance
        """
        weight_matrix = np.array(self.weight_history)
        mean_weights = np.mean(weight_matrix, axis=0)
        std_weights = np.std(weight_matrix, axis=0)
        cv = std_weights / (mean_weights + 1e-8)
        
        # Top 10% high-weight samples with CV < 0.3
        high_threshold = np.percentile(mean_weights, 90)
        critical_high_mask = (mean_weights > high_threshold) & (cv < 0.3)
        critical_high_indices = np.where(critical_high_mask)[0]
        
        # Bottom 10% low-weight samples with CV < 0.3
        low_threshold = np.percentile(mean_weights, 10)
        critical_low_mask = (mean_weights < low_threshold) & (cv < 0.3)
        critical_low_indices = np.where(critical_low_mask)[0]
        
        # Group breakdown
        groups = np.unique(protected)
        high_group_counts = {int(g): int(np.sum(protected[critical_high_indices] == g)) 
                            for g in groups}
        low_group_counts = {int(g): int(np.sum(protected[critical_low_indices] == g)) 
                           for g in groups}
        
        critical = {
            'high_count': int(len(critical_high_indices)),
            'low_count': int(len(critical_low_indices)),
            'high_group_distribution': high_group_counts,
            'low_group_distribution': low_group_counts,
            'high_mean_weight': float(np.mean(mean_weights[critical_high_indices])) if len(critical_high_indices) > 0 else 0.0,
            'low_mean_weight': float(np.mean(mean_weights[critical_low_indices])) if len(critical_low_indices) > 0 else 0.0
        }
        
        return critical


def create_temporal_visualizations(results_dict, output_path):
    """
    Create comprehensive temporal analysis visualizations.
    
    5-panel figure:
    1. Fairness evolution over epochs (all datasets)
    2. Weight stability analysis
    3. Group-wise weight evolution
    4. Critical samples identification
    5. Convergence analysis
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Temporal Fairness Analysis: How Adaptive Weighting Evolves', 
                 fontsize=14, fontweight='bold')
    
    datasets = ['COMPAS', 'Adult', 'German']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # Panel 1: Fairness evolution
    ax = axes[0, 0]
    for dataset, color in zip(datasets, colors):
        history = results_dict[dataset]['fairness_history']
        epochs = [h['epoch'] for h in history]
        eo_disparities = [h['eo_disparity'] for h in history]
        ax.plot(epochs, eo_disparities, '-o', label=dataset, color=color, 
                linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('EO Disparity', fontsize=10)
    ax.set_title('Fairness Evolution Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Target')
    
    # Panel 2: Accuracy evolution
    ax = axes[0, 1]
    for dataset, color in zip(datasets, colors):
        history = results_dict[dataset]['fairness_history']
        epochs = [h['epoch'] for h in history]
        accuracies = [h['accuracy'] for h in history]
        ax.plot(epochs, accuracies, '-s', label=dataset, color=color, 
                linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title('Accuracy Evolution Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Weight Gini evolution
    ax = axes[0, 2]
    for dataset, color in zip(datasets, colors):
        history = results_dict[dataset]['fairness_history']
        epochs = [h['epoch'] for h in history[1:]]  # Skip epoch 0 (uniform)
        gini = [h['weight_gini'] for h in history[1:]]
        ax.plot(epochs, gini, '-^', label=dataset, color=color, 
                linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Weight Gini Coefficient', fontsize=10)
    ax.set_title('Weight Concentration Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Stability metrics
    ax = axes[1, 0]
    stability_data = []
    for dataset in datasets:
        stability = results_dict[dataset]['weight_stability']
        stability_data.append([
            stability['stable_percentage'],
            stability['mean_rank_correlation'] * 100  # Scale to percentage
        ])
    
    x = np.arange(len(datasets))
    width = 0.35
    ax.bar(x - width/2, [s[0] for s in stability_data], width, 
           label='Stable Samples %', color='#3498db')
    ax.bar(x + width/2, [s[1] for s in stability_data], width, 
           label='Rank Correlation %', color='#e74c3c')
    
    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Percentage', fontsize=10)
    ax.set_title('Weight Stability Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: Critical samples distribution
    ax = axes[1, 1]
    critical_high_counts = []
    critical_low_counts = []
    for dataset in datasets:
        critical = results_dict[dataset]['critical_samples']
        critical_high_counts.append(critical['high_count'])
        critical_low_counts.append(critical['low_count'])
    
    x = np.arange(len(datasets))
    width = 0.35
    ax.bar(x - width/2, critical_high_counts, width, 
           label='High-Weight', color='#2ecc71')
    ax.bar(x + width/2, critical_low_counts, width, 
           label='Low-Weight', color='#e67e22')
    
    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Number of Critical Samples', fontsize=10)
    ax.set_title('Critical Samples (Stable High/Low)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 6: Convergence rate
    ax = axes[1, 2]
    for dataset, color in zip(datasets, colors):
        history = results_dict[dataset]['fairness_history']
        epochs = [h['epoch'] for h in history]
        eo_disparities = [h['eo_disparity'] for h in history]
        
        # Compute improvement rate (negative derivative)
        if len(eo_disparities) > 1:
            improvements = -np.diff(eo_disparities)  # Negative because we want decrease
            ax.plot(epochs[1:], improvements, '-o', label=dataset, color=color, 
                    linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('EO Improvement (vs prev epoch)', fontsize=10)
    ax.set_title('Convergence Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.close()


def evaluate_temporal_fairness():
    """
    Main evaluation function: Run temporal analysis on all datasets.
    """
    print("=" * 80)
    print("Day 16: Temporal Fairness Analysis")
    print("=" * 80)
    print("\nResearch Questions:")
    print("1. How do sample weights evolve over training?")
    print("2. Does fairness improve monotonically or oscillate?")
    print("3. Which samples are consistently important?")
    print("4. Are there critical epochs where fairness jumps?")
    print()
    
    # Configuration
    temperature = 0.5  # Optimal from Day 13
    n_epochs = 50
    
    # Results storage
    all_results = {}
    
    # Evaluate each dataset
    datasets = {
        'COMPAS': 'compas',
        'Adult': 'adult',
        'German': 'german'
    }
    
    for name, dataset_name in datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing {name} Dataset")
        print(f"{'='*80}")
        
        # Load data
        loader = DataLoader(dataset_name=dataset_name)
        data = loader.load_and_prepare(noise_rate=0.0, test_size=0.3, seed=42)
        
        X_train = data['X_train']
        y_train = data['y_train']
        protected_train = data['z_train']
        X_test = data['X_test']
        y_test = data['y_test']
        protected_test = data['z_test']
        
        print(f"Train: {len(X_train)} samples")
        print(f"Test: {len(X_test)} samples")
        print(f"Protected group distribution: {np.bincount(protected_train)}")
        
        # Run temporal analysis
        tracker = TemporalFairnessTracker(
            temperature=temperature,
            n_epochs=n_epochs
        )
        
        results = tracker.train_iterative(
            X_train, y_train, protected_train,
            X_test, y_test, protected_test
        )
        
        all_results[name] = results
        
        # Print summary
        print(f"\n{name} Summary:")
        print(f"  Final EO Disparity: {results['fairness_history'][-1]['eo_disparity']:.4f}")
        print(f"  Initial EO Disparity: {results['fairness_history'][0]['eo_disparity']:.4f}")
        improvement = (results['fairness_history'][0]['eo_disparity'] - 
                      results['fairness_history'][-1]['eo_disparity']) / \
                      results['fairness_history'][0]['eo_disparity'] * 100
        print(f"  Improvement: {improvement:+.1f}%")
        
        stability = results['weight_stability']
        print(f"\nWeight Stability:")
        print(f"  Stable samples: {stability['stable_percentage']:.1f}%")
        print(f"  Mean rank correlation: {stability['mean_rank_correlation']:.3f}")
        
        critical = results['critical_samples']
        print(f"\nCritical Samples:")
        print(f"  High-weight: {critical['high_count']} samples")
        print(f"  Low-weight: {critical['low_count']} samples")
        print(f"  High group distribution: {critical['high_group_distribution']}")
        print(f"  Low group distribution: {critical['low_group_distribution']}")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("Creating visualizations...")
    print(f"{'='*80}")
    
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'day16_temporal_fairness.png')
    
    create_temporal_visualizations(all_results, output_path)
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    metrics_dir = 'results/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    output_file = os.path.join(metrics_dir, 'day16_temporal_fairness.json')
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    for name in datasets.keys():
        results = all_results[name]
        history = results['fairness_history']
        stability = results['weight_stability']
        
        print(f"\n{name}:")
        print(f"  Epochs: {len(history)}")
        print(f"  Final accuracy: {history[-1]['accuracy']:.4f}")
        print(f"  Final EO: {history[-1]['eo_disparity']:.4f}")
        print(f"  Improvement: {(history[0]['eo_disparity'] - history[-1]['eo_disparity']) / history[0]['eo_disparity'] * 100:+.1f}%")
        print(f"  Stable samples: {stability['stable_percentage']:.1f}%")
        print(f"  Rank correlation: {stability['mean_rank_correlation']:.3f}")
    
    print(f"\n{'='*80}")
    print("Day 16 Complete!")
    print(f"{'='*80}")
    
    return all_results


if __name__ == '__main__':
    results = evaluate_temporal_fairness()
