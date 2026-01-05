"""
Day 12: Comprehensive Robustness Testing
=========================================

Test how well our fairness methods handle realistic challenges:
1. Adversarial feature perturbations (FGSM-style attacks)
2. Distribution shift (train/test mismatch)
3. Feature noise (Gaussian noise on inputs)
4. Label noise (corrupted labels)
5. Missing features (incomplete data)

Goal: Identify which methods are truly robust for real-world deployment.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class RobustnessEvaluator:
    """Comprehensive robustness testing framework."""
    
    def __init__(self, dataset_name='adult'):
        self.dataset_name = dataset_name
        self.loader = DataLoader(dataset_name)
        self.results = {
            'dataset': dataset_name,
            'tests': {}
        }
        
        # Load data
        print(f"\n[OK] Loading {dataset_name} dataset...")
        X, y, z = self.loader.load_dataset()
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test, self.z_train, self.z_test = \
            train_test_split(X, y, z, test_size=0.3, random_state=42, stratify=y)
        
        # Standardize features
        self.X_train, self.X_test = self.loader.preprocess(self.X_train, self.X_test)
        
        print(f"Train: {len(self.X_train)} samples")
        print(f"Test: {len(self.X_test)} samples")
    
    def train_baseline(self):
        """Train baseline model for reference."""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        return model
    
    def train_with_selection(self, selection_ratio=0.7):
        """Train model with greedy sample selection."""
        # Initial model to compute losses
        model_init = LogisticRegression(max_iter=1000, random_state=42)
        model_init.fit(self.X_train, self.y_train)
        
        # Compute losses and select lowest loss samples
        losses = -model_init.predict_log_proba(self.X_train)
        sample_losses = np.array([losses[i, self.y_train[i]] for i in range(len(self.y_train))])
        
        # Select samples with lowest loss
        n_select = int(len(sample_losses) * selection_ratio)
        selected_indices = np.argsort(sample_losses)[:n_select]
        
        # Train on selected samples
        X_selected = self.X_train[selected_indices]
        y_selected = self.y_train[selected_indices]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_selected, y_selected)
        return model
    
    def train_with_weighting(self, scheme='adaptive', temperature=1.0):
        """Train model with uncertainty weighting."""
        # Initial model for computing weights
        model_init = LogisticRegression(max_iter=1000, random_state=42)
        model_init.fit(self.X_train, self.y_train)
        
        # Compute weights
        probs = model_init.predict_proba(self.X_train)
        max_probs = np.max(probs, axis=1)
        predictions = model_init.predict(self.X_train)
        correctness = (predictions == self.y_train).astype(float)
        
        if scheme == 'adaptive':
            weights = max_probs * correctness + 0.1
        elif scheme == 'confidence':
            weights = max_probs
        else:
            weights = np.ones(len(self.X_train))
        
        # Temperature scaling
        weights = weights ** (1.0 / temperature)
        weights = weights / np.sum(weights) * len(weights)
        
        # Train weighted model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train, sample_weight=weights)
        return model
    
    def evaluate_model(self, model, X_test, y_test, z_test):
        """Evaluate model and return metrics."""
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        eo = FairnessMetrics.equalized_odds(y_test, y_pred, z_test)
        return {'accuracy': acc, 'eo_disparity': eo}
    
    def test_adversarial_perturbations(self, epsilon_values=[0.01, 0.05, 0.1, 0.2]):
        """
        Test robustness to adversarial feature perturbations.
        Simulates FGSM-style attacks on test features.
        """
        print("\n" + "="*60)
        print("TEST 1: Adversarial Feature Perturbations")
        print("="*60)
        
        results = {}
        
        # Train models
        baseline_model = self.train_baseline()
        greedy_model = self.train_with_selection(0.7)
        adaptive_model = self.train_with_weighting('adaptive', 1.0)
        
        models = {
            'baseline': baseline_model,
            'greedy': greedy_model,
            'adaptive': adaptive_model
        }
        
        # Test each epsilon
        for eps in epsilon_values:
            print(f"\n--- Perturbation strength epsilon = {eps:.3f} ---")
            results[f'eps_{eps}'] = {}
            
            # Add random noise to test features
            X_perturbed = self.X_test + eps * np.random.randn(*self.X_test.shape)
            
            for method_name, model in models.items():
                metrics = self.evaluate_model(model, X_perturbed, self.y_test, self.z_test)
                results[f'eps_{eps}'][method_name] = metrics
                
                print(f"{method_name:12s}: Acc = {metrics['accuracy']:.4f}, "
                      f"EO = {metrics['eo_disparity']:.4f}")
        
        # Clean test (eps=0) for reference
        print(f"\n--- Clean test (no perturbation) ---")
        results['eps_0.0'] = {}
        for method_name, model in models.items():
            metrics = self.evaluate_model(model, self.X_test, self.y_test, self.z_test)
            results['eps_0.0'][method_name] = metrics
            print(f"{method_name:12s}: Acc = {metrics['accuracy']:.4f}, "
                  f"EO = {metrics['eo_disparity']:.4f}")
        
        self.results['tests']['adversarial'] = results
        return results
    
    def test_distribution_shift(self, shift_ratios=[0.1, 0.3, 0.5]):
        """
        Test robustness to distribution shift.
        Simulate covariate shift by reweighting test samples.
        """
        print("\n" + "="*60)
        print("TEST 2: Distribution Shift (Covariate Shift)")
        print("="*60)
        
        results = {}
        
        # Train models on original distribution
        baseline_model = self.train_baseline()
        greedy_model = self.train_with_selection(0.7)
        adaptive_model = self.train_with_weighting('adaptive', 1.0)
        
        models = {
            'baseline': baseline_model,
            'greedy': greedy_model,
            'adaptive': adaptive_model
        }
        
        # For each shift ratio, create biased test set
        for shift_ratio in shift_ratios:
            print(f"\n--- Shift ratio = {shift_ratio:.1f} ---")
            results[f'shift_{shift_ratio}'] = {}
            
            # Simulate shift: oversample one protected group
            group_0_indices = np.where(self.z_test == 0)[0]
            group_1_indices = np.where(self.z_test == 1)[0]
            
            # Reduce group 0, increase group 1
            n_keep_0 = int(len(group_0_indices) * (1 - shift_ratio))
            n_keep_1 = int(len(group_1_indices) * (1 + shift_ratio))
            
            selected_0 = np.random.choice(group_0_indices, n_keep_0, replace=False)
            selected_1 = np.random.choice(group_1_indices, n_keep_1, replace=True)
            
            shifted_indices = np.concatenate([selected_0, selected_1])
            np.random.shuffle(shifted_indices)
            
            X_shifted = self.X_test[shifted_indices]
            y_shifted = self.y_test[shifted_indices]
            z_shifted = self.z_test[shifted_indices]
            
            print(f"Original group ratio: {np.mean(self.z_test):.3f}")
            print(f"Shifted group ratio: {np.mean(z_shifted):.3f}")
            
            for method_name, model in models.items():
                metrics = self.evaluate_model(model, X_shifted, y_shifted, z_shifted)
                results[f'shift_{shift_ratio}'][method_name] = metrics
                
                print(f"{method_name:12s}: Acc = {metrics['accuracy']:.4f}, "
                      f"EO = {metrics['eo_disparity']:.4f}")
        
        self.results['tests']['distribution_shift'] = results
        return results
    
    def test_feature_noise(self, noise_levels=[0.05, 0.1, 0.2, 0.5]):
        """
        Test robustness to feature noise.
        Add Gaussian noise to features during testing.
        """
        print("\n" + "="*60)
        print("TEST 3: Feature Noise Robustness")
        print("="*60)
        
        results = {}
        
        # Train models
        baseline_model = self.train_baseline()
        greedy_model = self.train_with_selection(0.7)
        adaptive_model = self.train_with_weighting('adaptive', 1.0)
        
        models = {
            'baseline': baseline_model,
            'greedy': greedy_model,
            'adaptive': adaptive_model
        }
        
        # Compute feature std for relative noise
        feature_std = np.std(self.X_test, axis=0)
        
        for noise_level in noise_levels:
            print(f"\n--- Noise level = {noise_level:.2f} * feature_std ---")
            results[f'noise_{noise_level}'] = {}
            
            # Add noise
            noise = noise_level * feature_std * np.random.randn(*self.X_test.shape)
            X_noisy = self.X_test + noise
            
            for method_name, model in models.items():
                metrics = self.evaluate_model(model, X_noisy, self.y_test, self.z_test)
                results[f'noise_{noise_level}'][method_name] = metrics
                
                print(f"{method_name:12s}: Acc = {metrics['accuracy']:.4f}, "
                      f"EO = {metrics['eo_disparity']:.4f}")
        
        self.results['tests']['feature_noise'] = results
        return results
    
    def test_missing_features(self, missing_ratios=[0.1, 0.2, 0.3, 0.5]):
        """
        Test robustness to missing features.
        Randomly set features to 0 (mean imputation would give similar results).
        """
        print("\n" + "="*60)
        print("TEST 4: Missing Features")
        print("="*60)
        
        results = {}
        
        # Train models
        baseline_model = self.train_baseline()
        greedy_model = self.train_with_selection(0.7)
        adaptive_model = self.train_with_weighting('adaptive', 1.0)
        
        models = {
            'baseline': baseline_model,
            'greedy': greedy_model,
            'adaptive': adaptive_model
        }
        
        for missing_ratio in missing_ratios:
            print(f"\n--- Missing ratio = {missing_ratio:.1f} ---")
            results[f'missing_{missing_ratio}'] = {}
            
            # Create missing mask
            mask = np.random.rand(*self.X_test.shape) > missing_ratio
            X_missing = self.X_test.copy()
            X_missing[~mask] = 0  # Set missing to 0 (mean imputation alternative)
            
            for method_name, model in models.items():
                metrics = self.evaluate_model(model, X_missing, self.y_test, self.z_test)
                results[f'missing_{missing_ratio}'][method_name] = metrics
                
                print(f"{method_name:12s}: Acc = {metrics['accuracy']:.4f}, "
                      f"EO = {metrics['eo_disparity']:.4f}")
        
        self.results['tests']['missing_features'] = results
        return results
    
    def test_label_noise_training(self, noise_levels=[0.05, 0.1, 0.2, 0.3]):
        """
        Test robustness when training with label noise.
        Flip random labels during training, test on clean data.
        """
        print("\n" + "="*60)
        print("TEST 5: Label Noise During Training")
        print("="*60)
        
        results = {}
        
        for noise_level in noise_levels:
            print(f"\n--- Label noise = {noise_level:.1%} ---")
            results[f'label_noise_{noise_level}'] = {}
            
            # Create noisy labels
            n_flip = int(len(self.y_train) * noise_level)
            flip_indices = np.random.choice(len(self.y_train), n_flip, replace=False)
            y_noisy = self.y_train.copy()
            y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
            
            # Train baseline
            baseline_model = LogisticRegression(max_iter=1000, random_state=42)
            baseline_model.fit(self.X_train, y_noisy)
            
            # Train greedy (selects on clean labels, trains on noisy)
            temp_model = LogisticRegression(max_iter=1000, random_state=42)
            temp_model.fit(self.X_train, self.y_train)  # Clean for selection
            
            losses = -temp_model.predict_log_proba(self.X_train)
            losses = np.array([losses[i, self.y_train[i]] for i in range(len(self.y_train))])
            selected_indices = np.argsort(losses)[:int(len(losses) * 0.7)]
            
            greedy_model = LogisticRegression(max_iter=1000, random_state=42)
            greedy_model.fit(self.X_train[selected_indices], y_noisy[selected_indices])
            
            # Train adaptive weighting (should downweight noisy samples)
            # Use clean model for weights
            probs = temp_model.predict_proba(self.X_train)
            max_probs = np.max(probs, axis=1)
            predictions = temp_model.predict(self.X_train)
            correctness = (predictions == self.y_train).astype(float)
            weights = max_probs * correctness + 0.1
            weights = weights / np.sum(weights) * len(weights)
            
            adaptive_model = LogisticRegression(max_iter=1000, random_state=42)
            adaptive_model.fit(self.X_train, y_noisy, sample_weight=weights)
            
            models = {
                'baseline': baseline_model,
                'greedy': greedy_model,
                'adaptive': adaptive_model
            }
            
            # Test on CLEAN test set
            for method_name, model in models.items():
                metrics = self.evaluate_model(model, self.X_test, self.y_test, self.z_test)
                results[f'label_noise_{noise_level}'][method_name] = metrics
                
                print(f"{method_name:12s}: Acc = {metrics['accuracy']:.4f}, "
                      f"EO = {metrics['eo_disparity']:.4f}")
        
        self.results['tests']['label_noise'] = results
        return results
    
    def plot_robustness_results(self):
        """Create comprehensive robustness plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Robustness Testing - {self.dataset_name.upper()} Dataset', 
                     fontsize=16, fontweight='bold')
        
        methods = ['baseline', 'greedy', 'adaptive']
        colors = {'baseline': 'blue', 'greedy': 'green', 'adaptive': 'red'}
        markers = {'baseline': 'o', 'greedy': 's', 'adaptive': '^'}
        
        # Plot 1: Adversarial perturbations
        ax = axes[0, 0]
        if 'adversarial' in self.results['tests']:
            data = self.results['tests']['adversarial']
            eps_values = sorted([float(k.split('_')[1]) for k in data.keys()])
            
            for method in methods:
                accs = [data[f'eps_{eps}'][method]['accuracy'] for eps in eps_values]
                ax.plot(eps_values, accs, marker=markers[method], 
                       label=method.capitalize(), color=colors[method], linewidth=2)
            
            ax.set_xlabel('Perturbation Strength (epsilon)', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title('Adversarial Perturbations', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Distribution shift
        ax = axes[0, 1]
        if 'distribution_shift' in self.results['tests']:
            data = self.results['tests']['distribution_shift']
            shift_values = sorted([float(k.split('_')[1]) for k in data.keys()])
            
            for method in methods:
                eos = [data[f'shift_{s}'][method]['eo_disparity'] for s in shift_values]
                ax.plot(shift_values, eos, marker=markers[method],
                       label=method.capitalize(), color=colors[method], linewidth=2)
            
            ax.set_xlabel('Distribution Shift Ratio', fontsize=11)
            ax.set_ylabel('EO Disparity', fontsize=11)
            ax.set_title('Distribution Shift', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Feature noise
        ax = axes[0, 2]
        if 'feature_noise' in self.results['tests']:
            data = self.results['tests']['feature_noise']
            noise_values = sorted([float(k.split('_')[1]) for k in data.keys()])
            
            for method in methods:
                accs = [data[f'noise_{n}'][method]['accuracy'] for n in noise_values]
                ax.plot(noise_values, accs, marker=markers[method],
                       label=method.capitalize(), color=colors[method], linewidth=2)
            
            ax.set_xlabel('Noise Level (x feature_std)', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title('Feature Noise', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Missing features
        ax = axes[1, 0]
        if 'missing_features' in self.results['tests']:
            data = self.results['tests']['missing_features']
            missing_values = sorted([float(k.split('_')[1]) for k in data.keys()])
            
            for method in methods:
                accs = [data[f'missing_{m}'][method]['accuracy'] for m in missing_values]
                ax.plot(missing_values, accs, marker=markers[method],
                       label=method.capitalize(), color=colors[method], linewidth=2)
            
            ax.set_xlabel('Missing Feature Ratio', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title('Missing Features', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Label noise
        ax = axes[1, 1]
        if 'label_noise' in self.results['tests']:
            data = self.results['tests']['label_noise']
            noise_values = sorted([float(k.split('_')[2]) for k in data.keys()])
            
            for method in methods:
                accs = [data[f'label_noise_{n}'][method]['accuracy'] for n in noise_values]
                ax.plot(noise_values, accs, marker=markers[method],
                       label=method.capitalize(), color=colors[method], linewidth=2)
            
            ax.set_xlabel('Label Noise Level', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title('Label Noise (Training)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Summary - Average robustness
        ax = axes[1, 2]
        if len(self.results['tests']) > 0:
            # Compute robustness score (normalized accuracy degradation)
            robustness_scores = {m: [] for m in methods}
            
            for test_name, test_data in self.results['tests'].items():
                if test_name == 'label_noise':
                    continue  # Different baseline
                
                # Get clean performance (first key as baseline)
                keys_list = list(test_data.keys())
                if len(keys_list) == 0:
                    continue
                clean_key = min(keys_list)  # Use first/smallest perturbation as baseline
                
                for method in methods:
                    clean_acc = test_data[clean_key][method]['accuracy']
                    
                    # Average degradation across all perturbation levels
                    degradations = []
                    for key in test_data.keys():
                        if key != clean_key:
                            perturbed_acc = test_data[key][method]['accuracy']
                            deg = (clean_acc - perturbed_acc) / clean_acc
                            degradations.append(deg)
                    
                    avg_degradation = np.mean(degradations) if degradations else 0
                    robustness_scores[method].append(1 - avg_degradation)  # Higher is better
            
            # Plot average robustness
            avg_scores = {m: np.mean(scores) for m, scores in robustness_scores.items()}
            bars = ax.bar(range(len(methods)), 
                         [avg_scores[m] for m in methods],
                         color=[colors[m] for m in methods],
                         alpha=0.7)
            
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([m.capitalize() for m in methods])
            ax.set_ylabel('Robustness Score', fontsize=11)
            ax.set_title('Average Robustness', fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path('results/plots/day12_robustness_testing.png')
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved robustness plot to {plot_path}")
        
        return fig


def main():
    print("\n" + "="*60)
    print("DAY 12: COMPREHENSIVE ROBUSTNESS TESTING")
    print("="*60)
    print("\nObjective: Test fairness methods under realistic challenges")
    print("- Adversarial perturbations")
    print("- Distribution shift")
    print("- Feature noise")
    print("- Missing features")
    print("- Label noise")
    
    # Test on Adult dataset (best meta-learning results)
    evaluator = RobustnessEvaluator('adult')
    
    # Run all tests
    evaluator.test_adversarial_perturbations()
    evaluator.test_distribution_shift()
    evaluator.test_feature_noise()
    evaluator.test_missing_features()
    evaluator.test_label_noise_training()
    
    # Create visualizations
    evaluator.plot_robustness_results()
    
    # Save results
    results_path = Path('results/metrics/day12_robustness_testing.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(evaluator.results, f, indent=2)
    
    print(f"\n[OK] Saved results to {results_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Robustness Testing Complete")
    print("="*60)
    
    print("\nKey Findings:")
    
    # Adversarial robustness
    if 'adversarial' in evaluator.results['tests']:
        data = evaluator.results['tests']['adversarial']
        high_eps = max([float(k.split('_')[1]) for k in data.keys()])
        print(f"\n1. Adversarial Perturbations (eps={high_eps}):")
        for method in ['baseline', 'greedy', 'adaptive']:
            acc = data[f'eps_{high_eps}'][method]['accuracy']
            clean_acc = data['eps_0.0'][method]['accuracy']
            degradation = (clean_acc - acc) / clean_acc * 100
            print(f"   {method.capitalize():12s}: {degradation:.1f}% accuracy degradation")
    
    # Distribution shift
    if 'distribution_shift' in evaluator.results['tests']:
        data = evaluator.results['tests']['distribution_shift']
        high_shift = max([float(k.split('_')[1]) for k in data.keys()])
        print(f"\n2. Distribution Shift (ratio={high_shift}):")
        for method in ['baseline', 'greedy', 'adaptive']:
            eo = data[f'shift_{high_shift}'][method]['eo_disparity']
            print(f"   {method.capitalize():12s}: EO = {eo:.4f}")
    
    # Label noise
    if 'label_noise' in evaluator.results['tests']:
        data = evaluator.results['tests']['label_noise']
        high_noise = max([float(k.split('_')[2]) for k in data.keys()])
        print(f"\n3. Label Noise (noise={high_noise:.0%}):")
        for method in ['baseline', 'greedy', 'adaptive']:
            acc = data[f'label_noise_{high_noise}'][method]['accuracy']
            print(f"   {method.capitalize():12s}: Accuracy = {acc:.4f}")
    
    print("\n[OK] Day 12 complete - Robustness testing done!")
    print("Next: Day 13 - Ablation studies")


if __name__ == '__main__':
    main()
