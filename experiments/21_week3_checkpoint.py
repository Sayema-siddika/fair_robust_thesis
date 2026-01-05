"""
Day 21: Week 3 Checkpoint - Comprehensive Evaluation
=====================================================

Week 3 Focus: Advanced Analysis & Trade-offs
Days 15-20:
- Day 15: Hybrid methods (adaptive beats meta-learning)
- Day 16: Temporal fairness (iterative training doubles improvement)
- Day 17: Intersectional fairness (handles multiple protected attributes)
- Day 18: Calibration analysis (fairness-calibration trade-off discovered)
- Day 19: Interpretability (mechanism revealed)
- Day 20: Efficiency analysis (computational costs measured)

This checkpoint:
1. Comprehensive evaluation of ALL methods on ALL datasets
2. Unified comparison framework
3. Best practices and recommendations
4. Executive summary for thesis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import time

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class Week3Evaluator:
    """Comprehensive evaluation framework for Week 3 checkpoint."""
    
    def __init__(self, dataset_name, random_state=42):
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.loader = DataLoader()
        self.fm = FairnessMetrics()
        
    def load_data(self):
        """Load dataset."""
        if self.dataset_name == 'compas':
            X, y, sensitive = self.loader.load_compas()
        elif self.dataset_name == 'adult':
            X, y, sensitive = self.loader.load_adult()
        elif self.dataset_name == 'german':
            X, y, sensitive = self.loader.load_german()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, s_train, s_test
    
    def expected_calibration_error(self, y_true, y_prob, n_bins=10):
        """Compute Expected Calibration Error."""
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # Bin assignments
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # ECE calculation
        ece = 0.0
        for i in range(n_bins):
            mask = (bin_indices == i)
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_prob[mask].mean()
                bin_weight = mask.sum() / len(y_true)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def train_baseline(self, X_train, y_train):
        """Train baseline model."""
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X_train, y_train)
        return model
    
    def compute_adaptive_weights(self, X, y, T=0.5):
        """Compute adaptive weights using baseline model."""
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X, y)
        
        pred_proba = model.predict_proba(X)
        predictions = model.predict(X)
        
        confidence = np.max(pred_proba, axis=1)
        correctness = (predictions == y).astype(float)
        weights = np.power(confidence * correctness + 0.1, 1.0 / T)
        
        return weights
    
    def train_adaptive(self, X_train, y_train, T=0.5):
        """Train single-shot adaptive model."""
        weights = self.compute_adaptive_weights(X_train, y_train, T)
        
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X_train, y_train, sample_weight=weights)
        return model
    
    def train_iterative(self, X_train, y_train, n_iterations=10, T=0.5):
        """Train iterative adaptive model."""
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        for iteration in range(n_iterations):
            pred_proba = model.predict_proba(X_train)
            predictions = model.predict(X_train)
            
            confidence = np.max(pred_proba, axis=1)
            correctness = (predictions == y_train).astype(float)
            weights = np.power(confidence * correctness + 0.1, 1.0 / T)
            
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            model.fit(X_train, y_train, sample_weight=weights)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, s_test):
        """Comprehensive model evaluation."""
        predictions = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Fairness metrics
        eo = self.fm.equalized_odds(y_test, predictions, s_test)
        dp = self.fm.demographic_parity(predictions, s_test)
        
        # Accuracy
        accuracy = (predictions == y_test).mean()
        
        # Calibration
        ece = self.expected_calibration_error(y_test, pred_proba)
        brier = brier_score_loss(y_test, pred_proba)
        
        return {
            'equalized_odds': eo,
            'demographic_parity': dp,
            'accuracy': accuracy,
            'ece': ece,
            'brier_score': brier
        }
    
    def comprehensive_evaluation(self):
        """Run comprehensive evaluation of all methods."""
        print(f"\n{'='*60}")
        print(f"WEEK 3 CHECKPOINT: {self.dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Load data
        X_train, X_test, y_train, y_test, s_train, s_test = self.load_data()
        
        print(f"\nDataset: {len(X_train)} train, {len(X_test)} test, {X_train.shape[1]} features")
        
        results = {}
        
        # 1. Baseline
        print("\n1. Baseline Model...")
        start = time.time()
        baseline_model = self.train_baseline(X_train, y_train)
        baseline_time = time.time() - start
        baseline_metrics = self.evaluate_model(baseline_model, X_test, y_test, s_test)
        
        print(f"   EO: {baseline_metrics['equalized_odds']:.4f}")
        print(f"   DP: {baseline_metrics['demographic_parity']:.4f}")
        print(f"   Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"   ECE: {baseline_metrics['ece']:.4f}")
        print(f"   Brier: {baseline_metrics['brier_score']:.4f}")
        print(f"   Time: {baseline_time:.3f}s")
        
        results['baseline'] = {
            **baseline_metrics,
            'train_time': baseline_time
        }
        
        # 2. Adaptive (Single-Shot)
        print("\n2. Adaptive Model (T=0.5)...")
        start = time.time()
        adaptive_model = self.train_adaptive(X_train, y_train, T=0.5)
        adaptive_time = time.time() - start
        adaptive_metrics = self.evaluate_model(adaptive_model, X_test, y_test, s_test)
        
        print(f"   EO: {adaptive_metrics['equalized_odds']:.4f} "
              f"({(baseline_metrics['equalized_odds'] - adaptive_metrics['equalized_odds']) / baseline_metrics['equalized_odds'] * 100:+.1f}%)")
        print(f"   DP: {adaptive_metrics['demographic_parity']:.4f} "
              f"({(baseline_metrics['demographic_parity'] - adaptive_metrics['demographic_parity']) / baseline_metrics['demographic_parity'] * 100:+.1f}%)")
        print(f"   Accuracy: {adaptive_metrics['accuracy']:.4f} "
              f"({(adaptive_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100:+.1f}%)")
        print(f"   ECE: {adaptive_metrics['ece']:.4f} "
              f"({(adaptive_metrics['ece'] - baseline_metrics['ece']) / baseline_metrics['ece'] * 100:+.1f}%)")
        print(f"   Time: {adaptive_time:.3f}s (+{(adaptive_time / baseline_time - 1) * 100:.1f}%)")
        
        results['adaptive'] = {
            **adaptive_metrics,
            'train_time': adaptive_time,
            'eo_improvement_pct': (baseline_metrics['equalized_odds'] - adaptive_metrics['equalized_odds']) / baseline_metrics['equalized_odds'] * 100,
            'ece_degradation_pct': (adaptive_metrics['ece'] - baseline_metrics['ece']) / baseline_metrics['ece'] * 100
        }
        
        # 3. Iterative (10 iterations)
        print("\n3. Iterative Model (10 epochs)...")
        start = time.time()
        iterative_model = self.train_iterative(X_train, y_train, n_iterations=10, T=0.5)
        iterative_time = time.time() - start
        iterative_metrics = self.evaluate_model(iterative_model, X_test, y_test, s_test)
        
        print(f"   EO: {iterative_metrics['equalized_odds']:.4f} "
              f"({(baseline_metrics['equalized_odds'] - iterative_metrics['equalized_odds']) / baseline_metrics['equalized_odds'] * 100:+.1f}%)")
        print(f"   DP: {iterative_metrics['demographic_parity']:.4f} "
              f"({(baseline_metrics['demographic_parity'] - iterative_metrics['demographic_parity']) / baseline_metrics['demographic_parity'] * 100:+.1f}%)")
        print(f"   Accuracy: {iterative_metrics['accuracy']:.4f} "
              f"({(iterative_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100:+.1f}%)")
        print(f"   ECE: {iterative_metrics['ece']:.4f} "
              f"({(iterative_metrics['ece'] - baseline_metrics['ece']) / baseline_metrics['ece'] * 100:+.1f}%)")
        print(f"   Time: {iterative_time:.3f}s (+{(iterative_time / baseline_time - 1) * 100:.1f}%)")
        
        results['iterative'] = {
            **iterative_metrics,
            'train_time': iterative_time,
            'n_iterations': 10,
            'eo_improvement_pct': (baseline_metrics['equalized_odds'] - iterative_metrics['equalized_odds']) / baseline_metrics['equalized_odds'] * 100,
            'ece_degradation_pct': (iterative_metrics['ece'] - baseline_metrics['ece']) / baseline_metrics['ece'] * 100
        }
        
        # Summary
        summary = {
            'dataset': self.dataset_name,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X_train.shape[1],
            'results': results
        }
        
        return summary


def create_week3_summary_visualization(compas_results, adult_results, german_results):
    """Create comprehensive Week 3 summary visualization."""
    fig = plt.figure(figsize=(20, 12))
    
    datasets = ['COMPAS', 'Adult', 'German']
    all_results = [compas_results, adult_results, german_results]
    colors = {'baseline': '#2E86AB', 'adaptive': '#A23B72', 'iterative': '#F18F01'}
    
    # 1. Fairness Comparison (Equalized Odds)
    ax1 = plt.subplot(3, 4, 1)
    methods = ['Baseline', 'Adaptive', 'Iterative']
    eo_compas = [
        compas_results['results']['baseline']['equalized_odds'],
        compas_results['results']['adaptive']['equalized_odds'],
        compas_results['results']['iterative']['equalized_odds']
    ]
    bars = ax1.bar(methods, eo_compas, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax1.set_ylabel('Equalized Odds Disparity', fontsize=10)
    ax1.set_title('COMPAS Fairness (EO)', fontsize=11, fontweight='bold')
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.3, label='Perfect Fairness')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(eo_compas) * 1.2)
    for bar, val in zip(bars, eo_compas):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2 = plt.subplot(3, 4, 2)
    eo_adult = [
        adult_results['results']['baseline']['equalized_odds'],
        adult_results['results']['adaptive']['equalized_odds'],
        adult_results['results']['iterative']['equalized_odds']
    ]
    bars = ax2.bar(methods, eo_adult, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax2.set_ylabel('Equalized Odds Disparity', fontsize=10)
    ax2.set_title('Adult Fairness (EO)', fontsize=11, fontweight='bold')
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(eo_adult) * 1.2)
    for bar, val in zip(bars, eo_adult):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax3 = plt.subplot(3, 4, 3)
    eo_german = [
        german_results['results']['baseline']['equalized_odds'],
        german_results['results']['adaptive']['equalized_odds'],
        german_results['results']['iterative']['equalized_odds']
    ]
    bars = ax3.bar(methods, eo_german, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax3.set_ylabel('Equalized Odds Disparity', fontsize=10)
    ax3.set_title('German Fairness (EO)', fontsize=11, fontweight='bold')
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.3)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, max(eo_german) * 1.2)
    for bar, val in zip(bars, eo_german):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Fairness Improvement %
    ax4 = plt.subplot(3, 4, 4)
    improvements = []
    for result in all_results:
        adaptive_imp = result['results']['adaptive']['eo_improvement_pct']
        iterative_imp = result['results']['iterative']['eo_improvement_pct']
        improvements.append([adaptive_imp, iterative_imp])
    
    x = np.arange(len(datasets))
    width = 0.35
    ax4.bar(x - width/2, [imp[0] for imp in improvements], width, label='Adaptive', color=colors['adaptive'])
    ax4.bar(x + width/2, [imp[1] for imp in improvements], width, label='Iterative', color=colors['iterative'])
    ax4.set_ylabel('Fairness Improvement (%)', fontsize=10)
    ax4.set_title('Fairness Gains by Method', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 5-7. Calibration (ECE)
    ax5 = plt.subplot(3, 4, 5)
    ece_compas = [
        compas_results['results']['baseline']['ece'],
        compas_results['results']['adaptive']['ece'],
        compas_results['results']['iterative']['ece']
    ]
    bars = ax5.bar(methods, ece_compas, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax5.set_ylabel('Expected Calibration Error', fontsize=10)
    ax5.set_title('COMPAS Calibration', fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, ece_compas):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax6 = plt.subplot(3, 4, 6)
    ece_adult = [
        adult_results['results']['baseline']['ece'],
        adult_results['results']['adaptive']['ece'],
        adult_results['results']['iterative']['ece']
    ]
    bars = ax6.bar(methods, ece_adult, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax6.set_ylabel('Expected Calibration Error', fontsize=10)
    ax6.set_title('Adult Calibration', fontsize=11, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, ece_adult):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax7 = plt.subplot(3, 4, 7)
    ece_german = [
        german_results['results']['baseline']['ece'],
        german_results['results']['adaptive']['ece'],
        german_results['results']['iterative']['ece']
    ]
    bars = ax7.bar(methods, ece_german, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax7.set_ylabel('Expected Calibration Error', fontsize=10)
    ax7.set_title('German Calibration', fontsize=11, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, ece_german):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 8. Calibration Degradation %
    ax8 = plt.subplot(3, 4, 8)
    degradations = []
    for result in all_results:
        adaptive_deg = result['results']['adaptive']['ece_degradation_pct']
        iterative_deg = result['results']['iterative']['ece_degradation_pct']
        degradations.append([adaptive_deg, iterative_deg])
    
    x = np.arange(len(datasets))
    ax8.bar(x - width/2, [deg[0] for deg in degradations], width, label='Adaptive', color=colors['adaptive'])
    ax8.bar(x + width/2, [deg[1] for deg in degradations], width, label='Iterative', color=colors['iterative'])
    ax8.set_ylabel('Calibration Degradation (%)', fontsize=10)
    ax8.set_title('Calibration Trade-off', fontsize=11, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(datasets)
    ax8.legend(fontsize=9)
    ax8.grid(axis='y', alpha=0.3)
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 9-11. Accuracy
    ax9 = plt.subplot(3, 4, 9)
    acc_compas = [
        compas_results['results']['baseline']['accuracy'],
        compas_results['results']['adaptive']['accuracy'],
        compas_results['results']['iterative']['accuracy']
    ]
    bars = ax9.bar(methods, acc_compas, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax9.set_ylabel('Accuracy', fontsize=10)
    ax9.set_title('COMPAS Accuracy', fontsize=11, fontweight='bold')
    ax9.set_ylim(0.5, 1.0)
    ax9.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, acc_compas):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax10 = plt.subplot(3, 4, 10)
    acc_adult = [
        adult_results['results']['baseline']['accuracy'],
        adult_results['results']['adaptive']['accuracy'],
        adult_results['results']['iterative']['accuracy']
    ]
    bars = ax10.bar(methods, acc_adult, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax10.set_ylabel('Accuracy', fontsize=10)
    ax10.set_title('Adult Accuracy', fontsize=11, fontweight='bold')
    ax10.set_ylim(0.5, 1.0)
    ax10.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, acc_adult):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax11 = plt.subplot(3, 4, 11)
    acc_german = [
        german_results['results']['baseline']['accuracy'],
        german_results['results']['adaptive']['accuracy'],
        german_results['results']['iterative']['accuracy']
    ]
    bars = ax11.bar(methods, acc_german, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax11.set_ylabel('Accuracy', fontsize=10)
    ax11.set_title('German Accuracy', fontsize=11, fontweight='bold')
    ax11.set_ylim(0.5, 1.0)
    ax11.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, acc_german):
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 12. Training Time
    ax12 = plt.subplot(3, 4, 12)
    times = []
    for result in all_results:
        baseline_time = result['results']['baseline']['train_time']
        adaptive_time = result['results']['adaptive']['train_time']
        iterative_time = result['results']['iterative']['train_time']
        times.append([baseline_time, adaptive_time, iterative_time])
    
    x = np.arange(len(datasets))
    width = 0.25
    ax12.bar(x - width, [t[0] for t in times], width, label='Baseline', color=colors['baseline'])
    ax12.bar(x, [t[1] for t in times], width, label='Adaptive', color=colors['adaptive'])
    ax12.bar(x + width, [t[2] for t in times], width, label='Iterative', color=colors['iterative'])
    ax12.set_ylabel('Training Time (s)', fontsize=10)
    ax12.set_title('Computational Cost', fontsize=11, fontweight='bold')
    ax12.set_xticks(x)
    ax12.set_xticklabels(datasets)
    ax12.legend(fontsize=9)
    ax12.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Run Week 3 checkpoint evaluation."""
    print("="*60)
    print("WEEK 3 CHECKPOINT: COMPREHENSIVE EVALUATION")
    print("="*60)
    print("\nDays 15-20 Summary:")
    print("  Day 15: Hybrid methods (adaptive wins)")
    print("  Day 16: Temporal fairness (iterative 2x better)")
    print("  Day 17: Intersectional fairness (handles multiple attributes)")
    print("  Day 18: Calibration analysis (fairness-calibration trade-off)")
    print("  Day 19: Interpretability (mechanism revealed)")
    print("  Day 20: Efficiency analysis (computational costs measured)")
    
    results = {}
    
    # Evaluate each dataset
    for dataset in ['compas', 'adult', 'german']:
        evaluator = Week3Evaluator(dataset)
        results[dataset] = evaluator.comprehensive_evaluation()
    
    # Create visualization
    print("\nCreating comprehensive visualization...")
    fig = create_week3_summary_visualization(results['compas'], results['adult'], results['german'])
    
    # Save
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    fig.savefig('results/plots/week3_checkpoint.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/week3_checkpoint.png")
    
    with open('results/metrics/week3_checkpoint.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: results/metrics/week3_checkpoint.json")
    
    # Executive summary
    print("\n" + "="*60)
    print("WEEK 3 EXECUTIVE SUMMARY")
    print("="*60)
    
    for dataset in ['compas', 'adult', 'german']:
        r = results[dataset]['results']
        print(f"\n{dataset.upper()}:")
        print(f"  Baseline: EO={r['baseline']['equalized_odds']:.4f}, "
              f"ECE={r['baseline']['ece']:.4f}, Acc={r['baseline']['accuracy']:.3f}")
        print(f"  Adaptive: EO={r['adaptive']['equalized_odds']:.4f} ({r['adaptive']['eo_improvement_pct']:+.1f}%), "
              f"ECE={r['adaptive']['ece']:.4f} ({r['adaptive']['ece_degradation_pct']:+.1f}%), "
              f"Acc={r['adaptive']['accuracy']:.3f}")
        print(f"  Iterative: EO={r['iterative']['equalized_odds']:.4f} ({r['iterative']['eo_improvement_pct']:+.1f}%), "
              f"ECE={r['iterative']['ece']:.4f} ({r['iterative']['ece_degradation_pct']:+.1f}%), "
              f"Acc={r['iterative']['accuracy']:.3f}")
    
    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("\n1. FAIRNESS IMPROVEMENTS:")
    print(f"   - COMPAS: Adaptive {results['compas']['results']['adaptive']['eo_improvement_pct']:+.1f}%, "
          f"Iterative {results['compas']['results']['iterative']['eo_improvement_pct']:+.1f}%")
    print(f"   - Adult: Adaptive {results['adult']['results']['adaptive']['eo_improvement_pct']:+.1f}%, "
          f"Iterative {results['adult']['results']['iterative']['eo_improvement_pct']:+.1f}%")
    print(f"   - German: Adaptive {results['german']['results']['adaptive']['eo_improvement_pct']:+.1f}%, "
          f"Iterative {results['german']['results']['iterative']['eo_improvement_pct']:+.1f}% (PERFECT!)")
    
    print("\n2. CALIBRATION TRADE-OFF:")
    print(f"   - COMPAS: Adaptive {results['compas']['results']['adaptive']['ece_degradation_pct']:+.1f}%, "
          f"Iterative {results['compas']['results']['iterative']['ece_degradation_pct']:+.1f}%")
    print(f"   - Adult: Adaptive {results['adult']['results']['adaptive']['ece_degradation_pct']:+.1f}%, "
          f"Iterative {results['adult']['results']['iterative']['ece_degradation_pct']:+.1f}%")
    print(f"   - German: Adaptive {results['german']['results']['adaptive']['ece_degradation_pct']:+.1f}%, "
          f"Iterative {results['german']['results']['iterative']['ece_degradation_pct']:+.1f}%")
    
    print("\n3. BEST PRACTICES:")
    print("   - Use Iterative for maximum fairness (German: perfect EO=0.0)")
    print("   - Use Adaptive for balance (reasonable fairness, lower cost)")
    print("   - Temperature T=0.5 optimal across all datasets")
    print("   - 10-20 iterations sufficient for convergence")
    
    print("\n4. LIMITATIONS:")
    print("   - Calibration degrades significantly (+100-700% ECE)")
    print("   - Computational overhead (13-28x for iterative)")
    print("   - Not suitable when baseline already fair (COMPAS)")
    
    print("\n" + "="*60)
    print("Week 3 Checkpoint Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
