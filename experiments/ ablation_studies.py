"""
Day 13: Ablation Studies
========================

Systematically test the impact of each component to understand:
1. Temperature scaling effect (T ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0})
2. Selection ratio sensitivity (τ ∈ {0.3, 0.5, 0.7, 0.9})
3. Weighting scheme comparison (confidence, entropy, margin, adaptive)
4. Model architecture impact (logistic regression vs neural networks)

Goal: Identify which design choices matter most for fairness and robustness.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class AblationAnalyzer:
    """Systematic ablation study framework."""
    
    def __init__(self, dataset_name='adult'):
        self.dataset_name = dataset_name
        self.loader = DataLoader(dataset_name)
        self.results = {
            'dataset': dataset_name,
            'ablations': {}
        }
        
        # Load and prepare data
        print(f"\n[OK] Loading {dataset_name} dataset...")
        X, y, z = self.loader.load_dataset()
        
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test, self.z_train, self.z_test = \
            train_test_split(X, y, z, test_size=0.3, random_state=42, stratify=y)
        
        self.X_train, self.X_test = self.loader.preprocess(self.X_train, self.X_test)
        
        print(f"Train: {len(self.X_train)} samples")
        print(f"Test: {len(self.X_test)} samples")
    
    def compute_adaptive_weights(self, temperature=1.0):
        """Compute adaptive weights with given temperature."""
        model_init = LogisticRegression(max_iter=1000, random_state=42)
        model_init.fit(self.X_train, self.y_train)
        
        probs = model_init.predict_proba(self.X_train)
        max_probs = np.max(probs, axis=1)
        predictions = model_init.predict(self.X_train)
        correctness = (predictions == self.y_train).astype(float)
        
        weights = max_probs * correctness + 0.1
        weights = weights ** (1.0 / temperature)
        weights = weights / np.sum(weights) * len(weights)
        
        return weights
    
    def evaluate_model(self, model, X_test=None, y_test=None, z_test=None):
        """Evaluate model performance."""
        if X_test is None:
            X_test, y_test, z_test = self.X_test, self.y_test, self.z_test
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        eo = FairnessMetrics.equalized_odds(y_test, y_pred, z_test)
        
        return {
            'accuracy': acc,
            'eo_disparity': eo,
            'fairness_improvement': 0.0  # Will compute later
        }
    
    def ablation_temperature_scaling(self):
        """
        Test impact of temperature scaling on adaptive weighting.
        Temperature controls weight distribution:
        - Low T (0.1): Very peaked weights (hard selection-like)
        - High T (10): Very uniform weights (baseline-like)
        """
        print("\n" + "="*60)
        print("ABLATION 1: Temperature Scaling Impact")
        print("="*60)
        
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        results = {}
        
        # Baseline for reference
        baseline_model = LogisticRegression(max_iter=1000, random_state=42)
        baseline_model.fit(self.X_train, self.y_train)
        baseline_metrics = self.evaluate_model(baseline_model)
        
        print(f"\nBaseline: Acc={baseline_metrics['accuracy']:.4f}, "
              f"EO={baseline_metrics['eo_disparity']:.4f}")
        
        results['baseline'] = baseline_metrics
        
        # Test each temperature
        for temp in temperatures:
            print(f"\n--- Temperature T = {temp:.1f} ---")
            
            weights = self.compute_adaptive_weights(temp)
            
            # Train weighted model
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(self.X_train, self.y_train, sample_weight=weights)
            
            metrics = self.evaluate_model(model)
            
            # Compute fairness improvement
            baseline_eo = baseline_metrics['eo_disparity']
            metrics['fairness_improvement'] = (baseline_eo - metrics['eo_disparity']) / baseline_eo * 100
            
            results[f'T_{temp}'] = metrics
            
            # Analyze weight distribution
            weight_stats = {
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'std': float(np.std(weights)),
                'gini': float(self._gini_coefficient(weights))
            }
            results[f'T_{temp}']['weight_stats'] = weight_stats
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"EO Disparity: {metrics['eo_disparity']:.4f}")
            print(f"Fairness Improvement: {metrics['fairness_improvement']:+.1f}%")
            print(f"Weight Gini: {weight_stats['gini']:.3f} (0=uniform, 1=concentrated)")
        
        self.results['ablations']['temperature'] = results
        return results
    
    def ablation_selection_ratio(self):
        """
        Test impact of selection ratio on greedy selection.
        Selection ratio (τ) controls how many samples to keep:
        - Low τ (0.3): Keep only 30% cleanest samples (aggressive)
        - High τ (0.9): Keep 90% samples (conservative)
        """
        print("\n" + "="*60)
        print("ABLATION 2: Selection Ratio Impact")
        print("="*60)
        
        ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = {}
        
        # Baseline
        baseline_model = LogisticRegression(max_iter=1000, random_state=42)
        baseline_model.fit(self.X_train, self.y_train)
        baseline_metrics = self.evaluate_model(baseline_model)
        
        print(f"\nBaseline (τ=1.0): Acc={baseline_metrics['accuracy']:.4f}, "
              f"EO={baseline_metrics['eo_disparity']:.4f}")
        
        results['baseline'] = baseline_metrics
        
        # Initial model for computing losses
        model_init = LogisticRegression(max_iter=1000, random_state=42)
        model_init.fit(self.X_train, self.y_train)
        
        losses = -model_init.predict_log_proba(self.X_train)
        sample_losses = np.array([losses[i, self.y_train[i]] for i in range(len(self.y_train))])
        
        # Test each ratio
        for ratio in ratios:
            print(f"\n--- Selection Ratio τ = {ratio:.1f} ---")
            
            n_select = int(len(sample_losses) * ratio)
            selected_indices = np.argsort(sample_losses)[:n_select]
            
            # Train on selected samples
            X_selected = self.X_train[selected_indices]
            y_selected = self.y_train[selected_indices]
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_selected, y_selected)
            
            metrics = self.evaluate_model(model)
            
            baseline_eo = baseline_metrics['eo_disparity']
            metrics['fairness_improvement'] = (baseline_eo - metrics['eo_disparity']) / baseline_eo * 100
            metrics['samples_used'] = n_select
            metrics['samples_discarded'] = len(self.X_train) - n_select
            
            results[f'tau_{ratio}'] = metrics
            
            print(f"Samples used: {n_select}/{len(self.X_train)} ({ratio*100:.0f}%)")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"EO Disparity: {metrics['eo_disparity']:.4f}")
            print(f"Fairness Improvement: {metrics['fairness_improvement']:+.1f}%")
        
        self.results['ablations']['selection_ratio'] = results
        return results
    
    def ablation_weighting_schemes(self):
        """
        Compare different weighting schemes:
        1. Confidence: weight = max(P(y|x))
        2. Entropy: weight = -sum(p*log(p))
        3. Margin: weight = P(y_true) - P(y_false)
        4. Adaptive: weight = confidence * correctness + 0.1
        """
        print("\n" + "="*60)
        print("ABLATION 3: Weighting Scheme Comparison")
        print("="*60)
        
        schemes = ['confidence', 'entropy', 'margin', 'adaptive']
        results = {}
        
        # Baseline
        baseline_model = LogisticRegression(max_iter=1000, random_state=42)
        baseline_model.fit(self.X_train, self.y_train)
        baseline_metrics = self.evaluate_model(baseline_model)
        
        print(f"\nBaseline: Acc={baseline_metrics['accuracy']:.4f}, "
              f"EO={baseline_metrics['eo_disparity']:.4f}")
        
        results['baseline'] = baseline_metrics
        
        # Initial model for computing weights
        model_init = LogisticRegression(max_iter=1000, random_state=42)
        model_init.fit(self.X_train, self.y_train)
        
        probs = model_init.predict_proba(self.X_train)
        predictions = model_init.predict(self.X_train)
        correctness = (predictions == self.y_train).astype(float)
        
        # Test each scheme
        for scheme in schemes:
            print(f"\n--- Weighting Scheme: {scheme.upper()} ---")
            
            if scheme == 'confidence':
                weights = np.max(probs, axis=1)
            
            elif scheme == 'entropy':
                # Entropy = -sum(p*log(p)), use negative (high entropy = low weight)
                entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                weights = 1.0 / (entropy + 0.1)  # Inverse entropy
            
            elif scheme == 'margin':
                # Margin = difference between top two probabilities
                sorted_probs = np.sort(probs, axis=1)
                margins = sorted_probs[:, -1] - sorted_probs[:, -2]
                weights = margins + 0.1
            
            elif scheme == 'adaptive':
                max_probs = np.max(probs, axis=1)
                weights = max_probs * correctness + 0.1
            
            # Normalize weights
            weights = weights / np.sum(weights) * len(weights)
            
            # Train weighted model
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(self.X_train, self.y_train, sample_weight=weights)
            
            metrics = self.evaluate_model(model)
            
            baseline_eo = baseline_metrics['eo_disparity']
            metrics['fairness_improvement'] = (baseline_eo - metrics['eo_disparity']) / baseline_eo * 100
            
            # Weight statistics
            weight_stats = {
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights))
            }
            metrics['weight_stats'] = weight_stats
            
            results[scheme] = metrics
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"EO Disparity: {metrics['eo_disparity']:.4f}")
            print(f"Fairness Improvement: {metrics['fairness_improvement']:+.1f}%")
            print(f"Weight range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")
        
        self.results['ablations']['weighting_schemes'] = results
        return results
    
    def ablation_model_architecture(self):
        """
        Test if choice of base model matters:
        1. Logistic Regression (linear)
        2. MLP with 1 hidden layer (32 units)
        3. MLP with 2 hidden layers (64, 32 units)
        """
        print("\n" + "="*60)
        print("ABLATION 4: Model Architecture Impact")
        print("="*60)
        
        architectures = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'mlp_small': MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
            'mlp_large': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        }
        
        results = {}
        
        for arch_name, model_class in architectures.items():
            print(f"\n--- Architecture: {arch_name.upper()} ---")
            
            # Baseline (no weighting)
            model_baseline = model_class
            if hasattr(model_baseline, 'max_iter'):
                model_baseline.max_iter = 1000
            model_baseline.fit(self.X_train, self.y_train)
            
            baseline_metrics = self.evaluate_model(model_baseline)
            
            # Adaptive weighting
            if arch_name == 'logistic':
                model_init = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model_init = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
            
            model_init.fit(self.X_train, self.y_train)
            
            probs = model_init.predict_proba(self.X_train)
            max_probs = np.max(probs, axis=1)
            predictions = model_init.predict(self.X_train)
            correctness = (predictions == self.y_train).astype(float)
            weights = max_probs * correctness + 0.1
            weights = weights / np.sum(weights) * len(weights)
            
            # Train weighted model
            if arch_name == 'logistic':
                model_weighted = LogisticRegression(max_iter=1000, random_state=42)
            elif arch_name == 'mlp_small':
                model_weighted = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
            else:
                model_weighted = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
            
            # MLPClassifier doesn't support sample_weight, use resampling instead
            if arch_name == 'logistic':
                model_weighted.fit(self.X_train, self.y_train, sample_weight=weights)
            else:
                # Resample based on weights
                resampled_indices = np.random.choice(
                    len(self.X_train), 
                    size=len(self.X_train), 
                    replace=True, 
                    p=weights/np.sum(weights)
                )
                model_weighted.fit(self.X_train[resampled_indices], self.y_train[resampled_indices])
            
            weighted_metrics = self.evaluate_model(model_weighted)
            
            results[arch_name] = {
                'baseline': baseline_metrics,
                'adaptive_weighted': weighted_metrics,
                'improvement': {
                    'accuracy': (weighted_metrics['accuracy'] - baseline_metrics['accuracy']) * 100,
                    'fairness': (baseline_metrics['eo_disparity'] - weighted_metrics['eo_disparity']) / baseline_metrics['eo_disparity'] * 100
                }
            }
            
            print(f"Baseline:  Acc={baseline_metrics['accuracy']:.4f}, EO={baseline_metrics['eo_disparity']:.4f}")
            print(f"Weighted:  Acc={weighted_metrics['accuracy']:.4f}, EO={weighted_metrics['eo_disparity']:.4f}")
            print(f"Improvement: Acc={results[arch_name]['improvement']['accuracy']:+.2f}%, "
                  f"Fairness={results[arch_name]['improvement']['fairness']:+.1f}%")
        
        self.results['ablations']['model_architecture'] = results
        return results
    
    def _gini_coefficient(self, weights):
        """Compute Gini coefficient (0=equal, 1=concentrated)."""
        sorted_weights = np.sort(weights)
        n = len(weights)
        cumsum = np.cumsum(sorted_weights)
        return (2 * np.sum((np.arange(1, n+1)) * sorted_weights)) / (n * cumsum[-1]) - (n + 1) / n
    
    def plot_ablation_results(self):
        """Create comprehensive ablation visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Ablation Studies - {self.dataset_name.upper()} Dataset', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Temperature scaling
        ax = axes[0, 0]
        if 'temperature' in self.results['ablations']:
            data = self.results['ablations']['temperature']
            temps = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            
            accs = [data[f'T_{t}']['accuracy'] for t in temps]
            eos = [data[f'T_{t}']['eo_disparity'] for t in temps]
            baseline_eo = data['baseline']['eo_disparity']
            
            ax2 = ax.twinx()
            line1 = ax.plot(temps, accs, 'o-', color='blue', linewidth=2, markersize=8, label='Accuracy')
            line2 = ax2.plot(temps, eos, 's-', color='red', linewidth=2, markersize=8, label='EO Disparity')
            ax2.axhline(baseline_eo, color='gray', linestyle='--', alpha=0.5, label='Baseline EO')
            
            ax.set_xlabel('Temperature (T)', fontsize=11)
            ax.set_ylabel('Accuracy', color='blue', fontsize=11)
            ax2.set_ylabel('EO Disparity', color='red', fontsize=11)
            ax.set_title('Temperature Scaling Impact', fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
        
        # Plot 2: Selection ratio
        ax = axes[0, 1]
        if 'selection_ratio' in self.results['ablations']:
            data = self.results['ablations']['selection_ratio']
            ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            accs = [data[f'tau_{r}']['accuracy'] for r in ratios]
            eos = [data[f'tau_{r}']['eo_disparity'] for r in ratios]
            baseline_acc = data['baseline']['accuracy']
            baseline_eo = data['baseline']['eo_disparity']
            
            ax2 = ax.twinx()
            line1 = ax.plot(ratios, accs, 'o-', color='blue', linewidth=2, markersize=8, label='Accuracy')
            line2 = ax2.plot(ratios, eos, 's-', color='red', linewidth=2, markersize=8, label='EO Disparity')
            ax.axhline(baseline_acc, color='blue', linestyle='--', alpha=0.5)
            ax2.axhline(baseline_eo, color='red', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Selection Ratio (τ)', fontsize=11)
            ax.set_ylabel('Accuracy', color='blue', fontsize=11)
            ax2.set_ylabel('EO Disparity', color='red', fontsize=11)
            ax.set_title('Selection Ratio Impact', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
        
        # Plot 3: Weighting schemes
        ax = axes[1, 0]
        if 'weighting_schemes' in self.results['ablations']:
            data = self.results['ablations']['weighting_schemes']
            schemes = ['baseline', 'confidence', 'entropy', 'margin', 'adaptive']
            
            accs = [data[s]['accuracy'] if s in data else data['baseline']['accuracy'] 
                   for s in schemes]
            eos = [data[s]['eo_disparity'] if s in data else data['baseline']['eo_disparity'] 
                  for s in schemes]
            
            x = np.arange(len(schemes))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='blue', alpha=0.7)
            
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, eos, width, label='EO Disparity', color='red', alpha=0.7)
            
            ax.set_xlabel('Weighting Scheme', fontsize=11)
            ax.set_ylabel('Accuracy', color='blue', fontsize=11)
            ax2.set_ylabel('EO Disparity', color='red', fontsize=11)
            ax.set_title('Weighting Scheme Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([s.capitalize() for s in schemes], rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Plot 4: Model architecture
        ax = axes[1, 1]
        if 'model_architecture' in self.results['ablations']:
            data = self.results['ablations']['model_architecture']
            archs = list(data.keys())
            
            baseline_accs = [data[a]['baseline']['accuracy'] for a in archs]
            weighted_accs = [data[a]['adaptive_weighted']['accuracy'] for a in archs]
            
            x = np.arange(len(archs))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline', color='gray', alpha=0.7)
            bars2 = ax.bar(x + width/2, weighted_accs, width, label='Adaptive Weighted', color='green', alpha=0.7)
            
            ax.set_xlabel('Model Architecture', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title('Model Architecture Impact', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([a.upper().replace('_', ' ') for a in archs], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path('results/plots/day13_ablation_studies.png')
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved ablation plot to {plot_path}")
        
        return fig


def main():
    print("\n" + "="*60)
    print("DAY 13: ABLATION STUDIES")
    print("="*60)
    print("\nObjective: Understand which components matter most")
    print("- Temperature scaling (T)")
    print("- Selection ratio (τ)")
    print("- Weighting schemes")
    print("- Model architecture")
    
    # Run ablation studies on Adult dataset
    analyzer = AblationAnalyzer('adult')
    
    # Run all ablations
    analyzer.ablation_temperature_scaling()
    analyzer.ablation_selection_ratio()
    analyzer.ablation_weighting_schemes()
    analyzer.ablation_model_architecture()
    
    # Create visualizations
    analyzer.plot_ablation_results()
    
    # Save results
    results_path = Path('results/metrics/day13_ablation_studies.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(analyzer.results, f, indent=2)
    
    print(f"\n[OK] Saved results to {results_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Ablation Studies Complete")
    print("="*60)
    
    print("\nKey Insights:")
    
    # Temperature
    if 'temperature' in analyzer.results['ablations']:
        data = analyzer.results['ablations']['temperature']
        best_temp = max([k for k in data.keys() if k.startswith('T_')], 
                       key=lambda k: data[k]['fairness_improvement'])
        print(f"\n1. Temperature Scaling:")
        print(f"   Best temperature: {best_temp}")
        print(f"   Fairness improvement: {data[best_temp]['fairness_improvement']:+.1f}%")
    
    # Selection ratio
    if 'selection_ratio' in analyzer.results['ablations']:
        data = analyzer.results['ablations']['selection_ratio']
        best_ratio = max([k for k in data.keys() if k.startswith('tau_')], 
                        key=lambda k: data[k]['fairness_improvement'])
        print(f"\n2. Selection Ratio:")
        print(f"   Best ratio: {best_ratio}")
        print(f"   Fairness improvement: {data[best_ratio]['fairness_improvement']:+.1f}%")
    
    # Weighting schemes
    if 'weighting_schemes' in analyzer.results['ablations']:
        data = analyzer.results['ablations']['weighting_schemes']
        schemes = ['confidence', 'entropy', 'margin', 'adaptive']
        best_scheme = max(schemes, key=lambda s: data[s]['fairness_improvement'])
        print(f"\n3. Weighting Schemes:")
        print(f"   Best scheme: {best_scheme}")
        print(f"   Fairness improvement: {data[best_scheme]['fairness_improvement']:+.1f}%")
    
    print("\n[OK] Day 13 complete - Ablation studies done!")
    print("Next: Day 14 - Week 2 final checkpoint")


if __name__ == '__main__':
    main()
