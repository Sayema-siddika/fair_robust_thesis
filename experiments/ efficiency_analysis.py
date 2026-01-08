"""
Day 20: Computational Efficiency Analysis
==========================================

Research Questions:
1. What's the computational cost of adaptive weighting?
2. How does it scale with dataset size?
3. Single-shot vs iterative: time/memory trade-offs?
4. Is the fairness improvement worth the computational cost?

Metrics:
- Training time (baseline vs adaptive vs iterative)
- Memory usage (peak RAM)
- Inference time
- Scalability analysis (varying dataset sizes)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import psutil
import tracemalloc

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class EfficiencyAnalyzer:
    """Analyze computational efficiency of fairness methods."""
    
    def __init__(self, dataset_name, random_state=42):
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.loader = DataLoader()
        
    def load_data(self, subsample_fraction=1.0):
        """Load and optionally subsample dataset."""
        if self.dataset_name == 'compas':
            X, y, sensitive = self.loader.load_compas()
        elif self.dataset_name == 'adult':
            X, y, sensitive = self.loader.load_adult()
        elif self.dataset_name == 'german':
            X, y, sensitive = self.loader.load_german()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Subsample if requested
        if subsample_fraction < 1.0:
            n_samples = int(len(X) * subsample_fraction)
            indices = np.random.RandomState(self.random_state).choice(
                len(X), n_samples, replace=False
            )
            X, y, sensitive = X[indices], y[indices], sensitive[indices]
        
        # Split
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, s_train, s_test
    
    def measure_baseline_training(self, X_train, y_train):
        """Measure baseline training time and memory."""
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Train
        start_time = time.time()
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Memory
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024 / 1024
        
        return {
            'train_time': train_time,
            'memory_mb': mem_after - mem_before,
            'peak_memory_mb': peak_mb,
            'model': model
        }
    
    def compute_adaptive_weights(self, X, y, sensitive, T=0.5):
        """Compute adaptive weights (lightweight baseline model)."""
        # Train lightweight model for weight computation
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X, y)
        
        # Get predictions
        pred_proba = model.predict_proba(X)
        predictions = model.predict(X)
        
        # Compute weights
        confidence = np.max(pred_proba, axis=1)
        correctness = (predictions == y).astype(float)
        weights = np.power(confidence * correctness + 0.1, 1.0 / T)
        
        return weights
    
    def measure_adaptive_training(self, X_train, y_train, s_train, T=0.5):
        """Measure adaptive weighting training time and memory."""
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Step 1: Compute weights
        weights = self.compute_adaptive_weights(X_train, y_train, s_train, T)
        
        # Step 2: Train weighted model
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X_train, y_train, sample_weight=weights)
        
        train_time = time.time() - start_time
        
        # Memory
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024 / 1024
        
        return {
            'train_time': train_time,
            'memory_mb': mem_after - mem_before,
            'peak_memory_mb': peak_mb,
            'model': model
        }
    
    def measure_iterative_training(self, X_train, y_train, s_train, 
                                   n_iterations=10, T=0.5):
        """Measure iterative adaptive training time and memory."""
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Initial model
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # Iterative updates
        for iteration in range(n_iterations):
            # Compute weights
            pred_proba = model.predict_proba(X_train)
            predictions = model.predict(X_train)
            confidence = np.max(pred_proba, axis=1)
            correctness = (predictions == y_train).astype(float)
            weights = np.power(confidence * correctness + 0.1, 1.0 / T)
            
            # Retrain
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            model.fit(X_train, y_train, sample_weight=weights)
        
        train_time = time.time() - start_time
        
        # Memory
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024 / 1024
        
        return {
            'train_time': train_time,
            'memory_mb': mem_after - mem_before,
            'peak_memory_mb': peak_mb,
            'n_iterations': n_iterations,
            'model': model
        }
    
    def measure_inference_time(self, model, X_test, n_runs=100):
        """Measure inference time."""
        times = []
        for _ in range(n_runs):
            start = time.time()
            _ = model.predict_proba(X_test)
            times.append(time.time() - start)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'median_time': np.median(times)
        }
    
    def analyze_scalability(self, fractions=[0.1, 0.25, 0.5, 0.75, 1.0]):
        """Analyze scalability across dataset sizes."""
        results = []
        
        for fraction in fractions:
            print(f"\nAnalyzing fraction: {fraction:.2f}")
            
            # Load data
            X_train, X_test, y_train, y_test, s_train, s_test = self.load_data(fraction)
            n_samples = len(X_train)
            
            # Baseline
            print(f"  Baseline (n={n_samples})...")
            baseline = self.measure_baseline_training(X_train, y_train)
            
            # Adaptive
            print(f"  Adaptive (n={n_samples})...")
            adaptive = self.measure_adaptive_training(X_train, y_train, s_train)
            
            # Iterative (5 iterations for speed)
            print(f"  Iterative (n={n_samples})...")
            iterative = self.measure_iterative_training(
                X_train, y_train, s_train, n_iterations=5
            )
            
            # Fairness evaluation
            fm = FairnessMetrics()
            
            baseline_preds = baseline['model'].predict(X_test)
            adaptive_preds = adaptive['model'].predict(X_test)
            iterative_preds = iterative['model'].predict(X_test)
            
            baseline_eo = fm.equalized_odds(y_test, baseline_preds, s_test)
            adaptive_eo = fm.equalized_odds(y_test, adaptive_preds, s_test)
            iterative_eo = fm.equalized_odds(y_test, iterative_preds, s_test)
            
            results.append({
                'fraction': fraction,
                'n_samples': n_samples,
                'baseline': {
                    'train_time': baseline['train_time'],
                    'memory_mb': baseline['memory_mb'],
                    'peak_memory_mb': baseline['peak_memory_mb'],
                    'eo_disparity': baseline_eo
                },
                'adaptive': {
                    'train_time': adaptive['train_time'],
                    'memory_mb': adaptive['memory_mb'],
                    'peak_memory_mb': adaptive['peak_memory_mb'],
                    'eo_disparity': adaptive_eo,
                    'overhead_pct': (adaptive['train_time'] / baseline['train_time'] - 1) * 100
                },
                'iterative': {
                    'train_time': iterative['train_time'],
                    'memory_mb': iterative['memory_mb'],
                    'peak_memory_mb': iterative['peak_memory_mb'],
                    'n_iterations': iterative['n_iterations'],
                    'eo_disparity': iterative_eo,
                    'overhead_pct': (iterative['train_time'] / baseline['train_time'] - 1) * 100
                }
            })
        
        return results
    
    def full_analysis(self):
        """Run full efficiency analysis."""
        print(f"\n{'='*60}")
        print(f"EFFICIENCY ANALYSIS: {self.dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Load full dataset
        X_train, X_test, y_train, y_test, s_train, s_test = self.load_data(1.0)
        
        print(f"\nDataset size: {len(X_train)} train, {len(X_test)} test")
        print(f"Features: {X_train.shape[1]}")
        
        # 1. Baseline
        print("\n1. Baseline Training...")
        baseline = self.measure_baseline_training(X_train, y_train)
        print(f"   Time: {baseline['train_time']:.3f}s")
        print(f"   Memory: {baseline['memory_mb']:.1f} MB (peak: {baseline['peak_memory_mb']:.1f} MB)")
        
        # 2. Adaptive (single-shot)
        print("\n2. Adaptive Training (single-shot)...")
        adaptive = self.measure_adaptive_training(X_train, y_train, s_train)
        print(f"   Time: {adaptive['train_time']:.3f}s")
        print(f"   Memory: {adaptive['memory_mb']:.1f} MB (peak: {adaptive['peak_memory_mb']:.1f} MB)")
        print(f"   Overhead: {(adaptive['train_time'] / baseline['train_time'] - 1) * 100:.1f}%")
        
        # 3. Iterative (10 iterations)
        print("\n3. Iterative Training (10 iterations)...")
        iterative = self.measure_iterative_training(X_train, y_train, s_train, n_iterations=10)
        print(f"   Time: {iterative['train_time']:.3f}s")
        print(f"   Memory: {iterative['memory_mb']:.1f} MB (peak: {iterative['peak_memory_mb']:.1f} MB)")
        print(f"   Overhead: {(iterative['train_time'] / baseline['train_time'] - 1) * 100:.1f}%")
        print(f"   Time per iteration: {iterative['train_time'] / iterative['n_iterations']:.3f}s")
        
        # 4. Inference time
        print("\n4. Inference Time (100 runs)...")
        baseline_inf = self.measure_inference_time(baseline['model'], X_test)
        adaptive_inf = self.measure_inference_time(adaptive['model'], X_test)
        iterative_inf = self.measure_inference_time(iterative['model'], X_test)
        
        print(f"   Baseline: {baseline_inf['mean_time']*1000:.2f} ± {baseline_inf['std_time']*1000:.2f} ms")
        print(f"   Adaptive: {adaptive_inf['mean_time']*1000:.2f} ± {adaptive_inf['std_time']*1000:.2f} ms")
        print(f"   Iterative: {iterative_inf['mean_time']*1000:.2f} ± {iterative_inf['std_time']*1000:.2f} ms")
        
        # 5. Fairness evaluation
        print("\n5. Fairness Performance...")
        fm = FairnessMetrics()
        
        baseline_preds = baseline['model'].predict(X_test)
        adaptive_preds = adaptive['model'].predict(X_test)
        iterative_preds = iterative['model'].predict(X_test)
        
        baseline_eo = fm.equalized_odds(y_test, baseline_preds, s_test)
        adaptive_eo = fm.equalized_odds(y_test, adaptive_preds, s_test)
        iterative_eo = fm.equalized_odds(y_test, iterative_preds, s_test)
        
        print(f"   Baseline EO: {baseline_eo:.4f}")
        print(f"   Adaptive EO: {adaptive_eo:.4f} ({(baseline_eo - adaptive_eo) / baseline_eo * 100:+.1f}%)")
        print(f"   Iterative EO: {iterative_eo:.4f} ({(baseline_eo - iterative_eo) / baseline_eo * 100:+.1f}%)")
        
        # 6. Scalability analysis
        print("\n6. Scalability Analysis...")
        scalability = self.analyze_scalability()
        
        # Summary
        summary = {
            'dataset': self.dataset_name,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X_train.shape[1],
            'baseline': {
                'train_time': baseline['train_time'],
                'memory_mb': baseline['memory_mb'],
                'peak_memory_mb': baseline['peak_memory_mb'],
                'inference_ms': baseline_inf['mean_time'] * 1000,
                'eo_disparity': baseline_eo
            },
            'adaptive': {
                'train_time': adaptive['train_time'],
                'memory_mb': adaptive['memory_mb'],
                'peak_memory_mb': adaptive['peak_memory_mb'],
                'inference_ms': adaptive_inf['mean_time'] * 1000,
                'overhead_pct': (adaptive['train_time'] / baseline['train_time'] - 1) * 100,
                'eo_disparity': adaptive_eo,
                'eo_improvement_pct': (baseline_eo - adaptive_eo) / baseline_eo * 100
            },
            'iterative': {
                'train_time': iterative['train_time'],
                'memory_mb': iterative['memory_mb'],
                'peak_memory_mb': iterative['peak_memory_mb'],
                'inference_ms': iterative_inf['mean_time'] * 1000,
                'n_iterations': 10,
                'time_per_iteration': iterative['train_time'] / 10,
                'overhead_pct': (iterative['train_time'] / baseline['train_time'] - 1) * 100,
                'eo_disparity': iterative_eo,
                'eo_improvement_pct': (baseline_eo - iterative_eo) / baseline_eo * 100
            },
            'scalability': scalability
        }
        
        return summary


def create_visualizations(compas_results, adult_results, german_results):
    """Create comprehensive efficiency visualizations."""
    fig = plt.figure(figsize=(18, 12))
    
    datasets = ['COMPAS', 'Adult', 'German']
    results = [compas_results, adult_results, german_results]
    colors = {'baseline': '#2E86AB', 'adaptive': '#A23B72', 'iterative': '#F18F01'}
    
    # 1. Training Time Comparison
    ax1 = plt.subplot(3, 3, 1)
    methods = ['Baseline', 'Adaptive', 'Iterative']
    times_compas = [
        compas_results['baseline']['train_time'],
        compas_results['adaptive']['train_time'],
        compas_results['iterative']['train_time']
    ]
    bars = ax1.bar(methods, times_compas, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax1.set_ylabel('Training Time (s)', fontsize=10)
    ax1.set_title('COMPAS Training Time', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, time_val in zip(bars, times_compas):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    ax2 = plt.subplot(3, 3, 2)
    times_adult = [
        adult_results['baseline']['train_time'],
        adult_results['adaptive']['train_time'],
        adult_results['iterative']['train_time']
    ]
    bars = ax2.bar(methods, times_adult, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax2.set_ylabel('Training Time (s)', fontsize=10)
    ax2.set_title('Adult Training Time', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, time_val in zip(bars, times_adult):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    ax3 = plt.subplot(3, 3, 3)
    times_german = [
        german_results['baseline']['train_time'],
        german_results['adaptive']['train_time'],
        german_results['iterative']['train_time']
    ]
    bars = ax3.bar(methods, times_german, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax3.set_ylabel('Training Time (s)', fontsize=10)
    ax3.set_title('German Training Time', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, time_val in zip(bars, times_german):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # 2. Memory Usage Comparison
    ax4 = plt.subplot(3, 3, 4)
    mem_compas = [
        compas_results['baseline']['peak_memory_mb'],
        compas_results['adaptive']['peak_memory_mb'],
        compas_results['iterative']['peak_memory_mb']
    ]
    bars = ax4.bar(methods, mem_compas, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax4.set_ylabel('Peak Memory (MB)', fontsize=10)
    ax4.set_title('COMPAS Memory Usage', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar, mem in zip(bars, mem_compas):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}MB', ha='center', va='bottom', fontsize=9)
    
    ax5 = plt.subplot(3, 3, 5)
    mem_adult = [
        adult_results['baseline']['peak_memory_mb'],
        adult_results['adaptive']['peak_memory_mb'],
        adult_results['iterative']['peak_memory_mb']
    ]
    bars = ax5.bar(methods, mem_adult, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax5.set_ylabel('Peak Memory (MB)', fontsize=10)
    ax5.set_title('Adult Memory Usage', fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for bar, mem in zip(bars, mem_adult):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}MB', ha='center', va='bottom', fontsize=9)
    
    ax6 = plt.subplot(3, 3, 6)
    mem_german = [
        german_results['baseline']['peak_memory_mb'],
        german_results['adaptive']['peak_memory_mb'],
        german_results['iterative']['peak_memory_mb']
    ]
    bars = ax6.bar(methods, mem_german, color=[colors['baseline'], colors['adaptive'], colors['iterative']])
    ax6.set_ylabel('Peak Memory (MB)', fontsize=10)
    ax6.set_title('German Memory Usage', fontsize=11, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    for bar, mem in zip(bars, mem_german):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}MB', ha='center', va='bottom', fontsize=9)
    
    # 3. Scalability Curves
    for idx, (dataset_name, result) in enumerate(zip(datasets, results)):
        ax = plt.subplot(3, 3, 7 + idx)
        
        scalability = result['scalability']
        n_samples = [s['n_samples'] for s in scalability]
        baseline_times = [s['baseline']['train_time'] for s in scalability]
        adaptive_times = [s['adaptive']['train_time'] for s in scalability]
        iterative_times = [s['iterative']['train_time'] for s in scalability]
        
        ax.plot(n_samples, baseline_times, 'o-', label='Baseline', 
                color=colors['baseline'], linewidth=2, markersize=6)
        ax.plot(n_samples, adaptive_times, 's-', label='Adaptive', 
                color=colors['adaptive'], linewidth=2, markersize=6)
        ax.plot(n_samples, iterative_times, '^-', label='Iterative (5 iters)', 
                color=colors['iterative'], linewidth=2, markersize=6)
        
        ax.set_xlabel('Training Samples', fontsize=10)
        ax.set_ylabel('Training Time (s)', fontsize=10)
        ax.set_title(f'{dataset_name} Scalability', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Run Day 20 efficiency analysis."""
    print("="*60)
    print("DAY 20: COMPUTATIONAL EFFICIENCY ANALYSIS")
    print("="*60)
    
    results = {}
    
    # Analyze each dataset
    for dataset in ['compas', 'adult', 'german']:
        analyzer = EfficiencyAnalyzer(dataset)
        results[dataset] = analyzer.full_analysis()
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = create_visualizations(results['compas'], results['adult'], results['german'])
    
    # Save
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    fig.savefig('results/plots/day20_efficiency.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/day20_efficiency.png")
    
    with open('results/metrics/day20_efficiency.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: results/metrics/day20_efficiency.json")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: COMPUTATIONAL EFFICIENCY")
    print("="*60)
    
    for dataset in ['compas', 'adult', 'german']:
        r = results[dataset]
        print(f"\n{dataset.upper()}:")
        print(f"  Dataset: {r['n_train']} train, {r['n_test']} test, {r['n_features']} features")
        print(f"  Baseline: {r['baseline']['train_time']:.3f}s, {r['baseline']['peak_memory_mb']:.1f}MB")
        print(f"  Adaptive: {r['adaptive']['train_time']:.3f}s (+{r['adaptive']['overhead_pct']:.1f}%), "
              f"{r['adaptive']['peak_memory_mb']:.1f}MB")
        print(f"  Iterative: {r['iterative']['train_time']:.3f}s (+{r['iterative']['overhead_pct']:.1f}%), "
              f"{r['iterative']['peak_memory_mb']:.1f}MB")
        print(f"  Fairness: Baseline EO={r['baseline']['eo_disparity']:.4f}, "
              f"Adaptive={r['adaptive']['eo_disparity']:.4f} ({r['adaptive']['eo_improvement_pct']:+.1f}%), "
              f"Iterative={r['iterative']['eo_disparity']:.4f} ({r['iterative']['eo_improvement_pct']:+.1f}%)")
    
    print("\n" + "="*60)
    print("Day 20 Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
