"""
Day 18: Calibration + Fairness Analysis

Research Question:
Are fair predictions also well-calibrated?

Calibration measures if predicted probabilities match true frequencies.
Example: If model predicts 70% probability, do 70% of those samples have label=1?

Good calibration is critical for:
- Decision making (thresholds, cost-sensitive applications)
- Trust (users need reliable probabilities)
- Fairness (miscalibrated probabilities can harm specific groups)

Experiments:
1. Measure calibration with Expected Calibration Error (ECE)
2. Plot reliability diagrams (predicted prob vs actual frequency)
3. Compare baseline vs adaptive weighting calibration
4. Check if calibration differs across demographic groups
5. Test if improving fairness hurts calibration

Expected Challenge:
Sample reweighting might hurt calibration (focuses on hard samples, may distort probability estimates).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
import json

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class CalibrationAnalyzer:
    """
    Analyze calibration of fair machine learning models.
    
    Calibration: Do predicted probabilities match true frequencies?
    """
    
    def __init__(self, temperature=0.5, n_bins=10):
        self.temperature = temperature
        self.n_bins = n_bins
        
    def expected_calibration_error(self, y_true, y_prob, n_bins=10):
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = sum over bins of |bin_accuracy - bin_confidence| × bin_size
        
        Args:
            y_true: True labels (binary)
            y_prob: Predicted probabilities for positive class
            n_bins: Number of bins for discretization
            
        Returns:
            ece: Expected calibration error (lower is better)
            bin_stats: Dict with per-bin statistics
        """
        # Bin samples by predicted probability
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_stats = []
        total_ece = 0.0
        
        for bin_idx in range(n_bins):
            mask = (bin_indices == bin_idx)
            bin_size = np.sum(mask)
            
            if bin_size == 0:
                continue
            
            # Average predicted probability in bin
            bin_confidence = np.mean(y_prob[mask])
            
            # Actual accuracy in bin
            bin_accuracy = np.mean(y_true[mask])
            
            # Calibration error for this bin
            bin_error = abs(bin_accuracy - bin_confidence)
            
            # Weight by bin size
            bin_weight = bin_size / len(y_true)
            total_ece += bin_weight * bin_error
            
            bin_stats.append({
                'bin_idx': bin_idx,
                'bin_lower': float(bin_edges[bin_idx]),
                'bin_upper': float(bin_edges[bin_idx + 1]),
                'count': int(bin_size),
                'confidence': float(bin_confidence),
                'accuracy': float(bin_accuracy),
                'error': float(bin_error)
            })
        
        return total_ece, bin_stats
    
    def compute_adaptive_weights(self, model, X, y):
        """Compute adaptive sample weights."""
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            confidence = np.max(probs, axis=1)
        else:
            decision = model.decision_function(X)
            confidence = 1.0 / (1.0 + np.exp(-np.abs(decision)))
        
        preds = model.predict(X)
        correctness = (preds == y).astype(float)
        
        raw_weights = confidence * correctness + 0.1
        
        if self.temperature != 1.0:
            weights = raw_weights ** (1.0 / self.temperature)
        else:
            weights = raw_weights
        
        weights = weights / np.sum(weights) * len(weights)
        
        return weights
    
    def evaluate_calibration(self, X_train, y_train, protected_train,
                            X_test, y_test, protected_test,
                            use_weighting=False):
        """
        Train model and evaluate calibration.
        
        Args:
            X_train, y_train: Training data
            protected_train: Protected attributes
            X_test, y_test: Test data
            protected_test: Protected attributes
            use_weighting: Whether to use adaptive weighting
            
        Returns:
            results: Dict with calibration metrics
        """
        # Train model
        if use_weighting:
            # Train baseline first to get weights
            model_init = LogisticRegression(max_iter=1000, random_state=42)
            model_init.fit(X_train, y_train)
            
            weights = self.compute_adaptive_weights(model_init, X_train, y_train)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Overall metrics
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)  # Lower is better
        
        # Calibration metrics
        ece_overall, bin_stats_overall = self.expected_calibration_error(
            y_test, y_prob, n_bins=self.n_bins
        )
        
        # Fairness metrics
        eo_disparity = FairnessMetrics.equalized_odds(y_test, y_pred, protected_test)
        dp_disparity = FairnessMetrics.demographic_parity(y_pred, protected_test)
        
        # Group-wise calibration
        groups = np.unique(protected_test)
        group_calibration = {}
        
        for g in groups:
            mask = (protected_test == g)
            if np.sum(mask) < 10:  # Skip tiny groups
                continue
            
            ece_g, bin_stats_g = self.expected_calibration_error(
                y_test[mask], y_prob[mask], n_bins=self.n_bins
            )
            
            group_calibration[int(g)] = {
                'ece': float(ece_g),
                'brier': float(brier_score_loss(y_test[mask], y_prob[mask])),
                'count': int(np.sum(mask)),
                'bin_stats': bin_stats_g
            }
        
        # Calibration disparity (max ECE difference between groups)
        group_eces = [gc['ece'] for gc in group_calibration.values()]
        calibration_disparity = max(group_eces) - min(group_eces) if len(group_eces) > 1 else 0.0
        
        # Reliability curve data (for plotting)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_prob, n_bins=self.n_bins, strategy='uniform'
        )
        
        results = {
            'accuracy': float(acc),
            'brier_score': float(brier),
            'ece_overall': float(ece_overall),
            'calibration_disparity': float(calibration_disparity),
            'eo_disparity': float(eo_disparity),
            'dp_disparity': float(dp_disparity),
            'bin_stats': bin_stats_overall,
            'group_calibration': group_calibration,
            'reliability_curve': {
                'fraction_positive': fraction_of_positives.tolist(),
                'mean_predicted': mean_predicted_value.tolist()
            },
            'used_weighting': use_weighting
        }
        
        return results


def create_visualizations(results_dict, output_path):
    """
    Create calibration analysis visualizations.
    
    4-panel figure:
    1. Reliability diagrams (baseline vs adaptive)
    2. ECE comparison (overall and by group)
    3. Calibration disparity vs fairness disparity
    4. Brier score comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Day 18: Calibration + Fairness Analysis', 
                 fontsize=14, fontweight='bold')
    
    datasets = ['COMPAS', 'Adult', 'German']
    colors_base = ['#e74c3c', '#3498db', '#2ecc71']
    colors_adapt = ['#c0392b', '#2980b9', '#27ae60']
    
    # Panel 1: Reliability diagrams
    ax = axes[0, 0]
    
    for i, dataset in enumerate(datasets):
        baseline = results_dict[dataset]['baseline']
        adaptive = results_dict[dataset]['adaptive']
        
        # Baseline
        frac_pos_base = baseline['reliability_curve']['fraction_positive']
        mean_pred_base = baseline['reliability_curve']['mean_predicted']
        ax.plot(mean_pred_base, frac_pos_base, 'o-', 
                color=colors_base[i], alpha=0.6, linewidth=2,
                label=f'{dataset} Baseline')
        
        # Adaptive
        frac_pos_adapt = adaptive['reliability_curve']['fraction_positive']
        mean_pred_adapt = adaptive['reliability_curve']['mean_predicted']
        ax.plot(mean_pred_adapt, frac_pos_adapt, 's--', 
                color=colors_adapt[i], alpha=0.8, linewidth=2,
                label=f'{dataset} Adaptive')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Perfect Calibration')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.set_title('Reliability Diagrams', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Panel 2: ECE comparison
    ax = axes[0, 1]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ece_baseline = [results_dict[d]['baseline']['ece_overall'] for d in datasets]
    ece_adaptive = [results_dict[d]['adaptive']['ece_overall'] for d in datasets]
    
    ax.bar(x - width/2, ece_baseline, width, label='Baseline', 
           color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, ece_adaptive, width, label='Adaptive', 
           color='#2ecc71', alpha=0.7)
    
    ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=11)
    ax.set_title('Overall Calibration Error', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Calibration disparity vs Fairness disparity
    ax = axes[1, 0]
    
    cal_disp_baseline = [results_dict[d]['baseline']['calibration_disparity'] for d in datasets]
    cal_disp_adaptive = [results_dict[d]['adaptive']['calibration_disparity'] for d in datasets]
    
    eo_disp_baseline = [results_dict[d]['baseline']['eo_disparity'] for d in datasets]
    eo_disp_adaptive = [results_dict[d]['adaptive']['eo_disparity'] for d in datasets]
    
    # Scatter plot
    ax.scatter(eo_disp_baseline, cal_disp_baseline, s=150, c='#e74c3c', 
               marker='o', alpha=0.7, label='Baseline', edgecolors='black', linewidth=1.5)
    ax.scatter(eo_disp_adaptive, cal_disp_adaptive, s=150, c='#2ecc71', 
               marker='s', alpha=0.7, label='Adaptive', edgecolors='black', linewidth=1.5)
    
    # Annotate datasets
    for i, dataset in enumerate(datasets):
        ax.annotate(dataset, (eo_disp_baseline[i], cal_disp_baseline[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.annotate(dataset, (eo_disp_adaptive[i], cal_disp_adaptive[i]), 
                   xytext=(5, -10), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('EO Disparity (Fairness)', fontsize=11)
    ax.set_ylabel('Calibration Disparity (max ECE diff)', fontsize=11)
    ax.set_title('Fairness vs Calibration Trade-off', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Brier score comparison
    ax = axes[1, 1]
    
    brier_baseline = [results_dict[d]['baseline']['brier_score'] for d in datasets]
    brier_adaptive = [results_dict[d]['adaptive']['brier_score'] for d in datasets]
    
    x = np.arange(len(datasets))
    ax.bar(x - width/2, brier_baseline, width, label='Baseline', 
           color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, brier_adaptive, width, label='Adaptive', 
           color='#2ecc71', alpha=0.7)
    
    ax.set_ylabel('Brier Score', fontsize=11)
    ax.set_title('Probability Calibration (Brier Score)', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.close()


def evaluate_calibration_fairness():
    """
    Main evaluation: Analyze calibration of fair models.
    """
    print("=" * 80)
    print("Day 18: Calibration + Fairness Analysis")
    print("=" * 80)
    print("\nResearch Questions:")
    print("1. Are fair predictions also well-calibrated?")
    print("2. Does adaptive weighting hurt calibration?")
    print("3. Is calibration fair across demographic groups?")
    print()
    
    # Configuration
    temperature = 0.5
    n_bins = 10
    
    # Results storage
    all_results = {}
    
    # Datasets
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
        
        # Initialize analyzer
        analyzer = CalibrationAnalyzer(temperature=temperature, n_bins=n_bins)
        
        # Baseline
        print("\nBaseline (no weighting):")
        baseline_results = analyzer.evaluate_calibration(
            X_train, y_train, protected_train,
            X_test, y_test, protected_test,
            use_weighting=False
        )
        
        print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
        print(f"  ECE: {baseline_results['ece_overall']:.4f}")
        print(f"  Brier Score: {baseline_results['brier_score']:.4f}")
        print(f"  EO Disparity: {baseline_results['eo_disparity']:.4f}")
        print(f"  Calibration Disparity: {baseline_results['calibration_disparity']:.4f}")
        
        print(f"\n  Group-wise calibration:")
        for gid, gcal in baseline_results['group_calibration'].items():
            print(f"    Group {gid}: ECE={gcal['ece']:.4f}, Brier={gcal['brier']:.4f}, n={gcal['count']}")
        
        # Adaptive
        print("\nAdaptive weighting (T=0.5):")
        adaptive_results = analyzer.evaluate_calibration(
            X_train, y_train, protected_train,
            X_test, y_test, protected_test,
            use_weighting=True
        )
        
        print(f"  Accuracy: {adaptive_results['accuracy']:.4f}")
        print(f"  ECE: {adaptive_results['ece_overall']:.4f}")
        print(f"  Brier Score: {adaptive_results['brier_score']:.4f}")
        print(f"  EO Disparity: {adaptive_results['eo_disparity']:.4f}")
        print(f"  Calibration Disparity: {adaptive_results['calibration_disparity']:.4f}")
        
        print(f"\n  Group-wise calibration:")
        for gid, gcal in adaptive_results['group_calibration'].items():
            print(f"    Group {gid}: ECE={gcal['ece']:.4f}, Brier={gcal['brier']:.4f}, n={gcal['count']}")
        
        # Comparison
        print(f"\nComparison:")
        ece_change = (adaptive_results['ece_overall'] - baseline_results['ece_overall']) / baseline_results['ece_overall'] * 100
        brier_change = (adaptive_results['brier_score'] - baseline_results['brier_score']) / baseline_results['brier_score'] * 100
        eo_change = (baseline_results['eo_disparity'] - adaptive_results['eo_disparity']) / baseline_results['eo_disparity'] * 100
        
        print(f"  ECE change: {ece_change:+.1f}%")
        print(f"  Brier change: {brier_change:+.1f}%")
        print(f"  EO improvement: {eo_change:+.1f}%")
        
        all_results[name] = {
            'baseline': baseline_results,
            'adaptive': adaptive_results,
            'comparison': {
                'ece_change_pct': float(ece_change),
                'brier_change_pct': float(brier_change),
                'eo_improvement_pct': float(eo_change)
            }
        }
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("Creating visualizations...")
    print(f"{'='*80}")
    
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'day18_calibration_fairness.png')
    
    create_visualizations(all_results, output_path)
    
    # Save results
    metrics_dir = 'results/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    output_file = os.path.join(metrics_dir, 'day18_calibration_fairness.json')
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    for name in datasets.keys():
        comp = all_results[name]['comparison']
        print(f"\n{name}:")
        print(f"  ECE change: {comp['ece_change_pct']:+.1f}%")
        print(f"  Brier change: {comp['brier_change_pct']:+.1f}%")
        print(f"  EO improvement: {comp['eo_improvement_pct']:+.1f}%")
    
    print(f"\n{'='*80}")
    print("Day 18 Complete!")
    print(f"{'='*80}")
    
    return all_results


if __name__ == '__main__':
    results = evaluate_calibration_fairness()
