"""
Day 19: Model Interpretability Analysis

Research Question:
WHY does adaptive weighting work? Which samples and features matter?

Interpretability helps us understand:
1. Which features drive high sample weights?
2. Does adaptive weighting change which features the model relies on?
3. Are high-weight samples clustered in feature space?
4. Can we predict which samples will be important?

Methods:
- Feature importance (coefficients for logistic regression)
- Weight-feature correlation analysis
- High-weight sample profiling
- Visualization of decision boundaries

Expected Insight:
Understanding the mechanism will validate that adaptive weighting makes sense
and isn't just exploiting dataset artifacts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr
import json

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class InterpretabilityAnalyzer:
    """
    Analyze interpretability of adaptive weighting.
    
    Focus on understanding:
    - Which samples get high weights and why
    - How features correlate with weights
    - Whether weighting changes model behavior
    """
    
    def __init__(self, temperature=0.5):
        self.temperature = temperature
        
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
        
        # Return metadata too
        metadata = {
            'confidence': confidence,
            'correctness': correctness,
            'raw_weights': raw_weights
        }
        
        return weights, metadata
    
    def analyze_weight_feature_correlation(self, X, weights, feature_names):
        """
        Analyze correlation between features and sample weights.
        
        Args:
            X: Feature matrix
            weights: Sample weights
            feature_names: List of feature names
            
        Returns:
            correlations: Dict of {feature: correlation_with_weight}
        """
        correlations = {}
        
        for i, fname in enumerate(feature_names):
            feature_values = X[:, i]
            
            # Pearson correlation (linear)
            pearson_corr, pearson_p = pearsonr(feature_values, weights)
            
            # Spearman correlation (monotonic)
            spearman_corr, spearman_p = spearmanr(feature_values, weights)
            
            correlations[fname] = {
                'pearson': float(pearson_corr),
                'pearson_pval': float(pearson_p),
                'spearman': float(spearman_corr),
                'spearman_pval': float(spearman_p)
            }
        
        return correlations
    
    def profile_high_weight_samples(self, X, weights, y, protected, feature_names, percentile=90):
        """
        Profile samples with high weights.
        
        Compare high-weight samples to low-weight samples across features.
        
        Args:
            X: Feature matrix
            weights: Sample weights  
            y: Labels
            protected: Protected attributes
            feature_names: List of feature names
            percentile: Percentile threshold for "high weight"
            
        Returns:
            profile: Dict with statistics
        """
        threshold = np.percentile(weights, percentile)
        high_mask = weights >= threshold
        low_mask = weights < np.percentile(weights, 100 - percentile)
        
        profile = {
            'high_weight_count': int(np.sum(high_mask)),
            'low_weight_count': int(np.sum(low_mask)),
            'threshold': float(threshold),
            'feature_comparison': {},
            'label_distribution': {},
            'protected_distribution': {}
        }
        
        # Feature comparison
        for i, fname in enumerate(feature_names):
            high_mean = float(np.mean(X[high_mask, i]))
            low_mean = float(np.mean(X[low_mask, i]))
            high_std = float(np.std(X[high_mask, i]))
            low_std = float(np.std(X[low_mask, i]))
            
            profile['feature_comparison'][fname] = {
                'high_mean': high_mean,
                'high_std': high_std,
                'low_mean': low_mean,
                'low_std': low_std,
                'difference': high_mean - low_mean
            }
        
        # Label distribution
        profile['label_distribution'] = {
            'high_positive_rate': float(np.mean(y[high_mask])),
            'low_positive_rate': float(np.mean(y[low_mask]))
        }
        
        # Protected attribute distribution
        profile['protected_distribution'] = {
            'high_protected_rate': float(np.mean(protected[high_mask])),
            'low_protected_rate': float(np.mean(protected[low_mask]))
        }
        
        return profile
    
    def compare_model_coefficients(self, X_train, y_train):
        """
        Compare model coefficients trained with/without weighting.
        
        Shows if weighting changes which features the model relies on.
        
        Args:
            X_train, y_train: Training data
            
        Returns:
            comparison: Dict with baseline and adaptive coefficients
        """
        # Baseline model
        model_baseline = LogisticRegression(max_iter=1000, random_state=42)
        model_baseline.fit(X_train, y_train)
        coef_baseline = model_baseline.coef_[0]
        
        # Adaptive model
        weights, _ = self.compute_adaptive_weights(model_baseline, X_train, y_train)
        
        model_adaptive = LogisticRegression(max_iter=1000, random_state=42)
        model_adaptive.fit(X_train, y_train, sample_weight=weights)
        coef_adaptive = model_adaptive.coef_[0]
        
        # Compare
        comparison = {
            'baseline_coef': coef_baseline.tolist(),
            'adaptive_coef': coef_adaptive.tolist(),
            'coefficient_change': (coef_adaptive - coef_baseline).tolist(),
            'relative_change': ((coef_adaptive - coef_baseline) / (np.abs(coef_baseline) + 1e-8) * 100).tolist()
        }
        
        return comparison, model_baseline, model_adaptive
    
    def analyze_interpretability(self, dataset_name, feature_names):
        """
        Full interpretability analysis for a dataset.
        
        Args:
            dataset_name: 'compas', 'adult', or 'german'
            feature_names: List of feature names
            
        Returns:
            results: Dict with all interpretability metrics
        """
        # Load data
        loader = DataLoader(dataset_name=dataset_name)
        data = loader.load_and_prepare(noise_rate=0.0, test_size=0.3, seed=42)
        
        X_train = data['X_train']
        y_train = data['y_train']
        protected_train = data['z_train']
        X_test = data['X_test']
        y_test = data['y_test']
        protected_test = data['z_test']
        
        print(f"Train: {len(X_train)} samples, {X_train.shape[1]} features")
        
        # Model comparison
        print("\nComparing model coefficients...")
        coef_comparison, model_baseline, model_adaptive = self.compare_model_coefficients(
            X_train, y_train
        )
        
        # Get weights
        weights, weight_meta = self.compute_adaptive_weights(
            model_baseline, X_train, y_train
        )
        
        # Weight-feature correlation
        print("Analyzing weight-feature correlation...")
        correlations = self.analyze_weight_feature_correlation(
            X_train, weights, feature_names
        )
        
        # Profile high-weight samples
        print("Profiling high-weight samples...")
        profile = self.profile_high_weight_samples(
            X_train, weights, y_train, protected_train, feature_names
        )
        
        # Evaluate models
        y_pred_baseline = model_baseline.predict(X_test)
        y_pred_adaptive = model_adaptive.predict(X_test)
        
        acc_baseline = accuracy_score(y_test, y_pred_baseline)
        acc_adaptive = accuracy_score(y_test, y_pred_adaptive)
        
        eo_baseline = FairnessMetrics.equalized_odds(y_test, y_pred_baseline, protected_test)
        eo_adaptive = FairnessMetrics.equalized_odds(y_test, y_pred_adaptive, protected_test)
        
        results = {
            'feature_names': feature_names,
            'coefficient_comparison': coef_comparison,
            'weight_feature_correlation': correlations,
            'high_weight_profile': profile,
            'weight_statistics': {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'median': float(np.median(weights)),
                '90th_percentile': float(np.percentile(weights, 90)),
                '10th_percentile': float(np.percentile(weights, 10))
            },
            'performance': {
                'baseline_accuracy': float(acc_baseline),
                'adaptive_accuracy': float(acc_adaptive),
                'baseline_eo': float(eo_baseline),
                'adaptive_eo': float(eo_adaptive)
            },
            'weight_metadata': {
                'mean_confidence': float(np.mean(weight_meta['confidence'])),
                'mean_correctness': float(np.mean(weight_meta['correctness']))
            }
        }
        
        return results


def create_visualizations(results_dict, output_path):
    """
    Create interpretability visualizations.
    
    3x3 grid:
    - Coefficient comparison (3 datasets)
    - Weight-feature correlation heatmap (3 datasets)
    - High vs low weight feature profiles (3 datasets)
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Day 19: Model Interpretability Analysis', 
                 fontsize=14, fontweight='bold')
    
    datasets = ['COMPAS', 'Adult', 'German']
    
    for col_idx, dataset in enumerate(datasets):
        results = results_dict[dataset]
        feature_names = results['feature_names']
        n_features = len(feature_names)
        
        # Row 1: Coefficient comparison
        ax = axes[0, col_idx]
        
        coef_baseline = np.array(results['coefficient_comparison']['baseline_coef'])
        coef_adaptive = np.array(results['coefficient_comparison']['adaptive_coef'])
        
        x = np.arange(n_features)
        width = 0.35
        
        ax.bar(x - width/2, coef_baseline, width, label='Baseline', 
               color='#e74c3c', alpha=0.7)
        ax.bar(x + width/2, coef_adaptive, width, label='Adaptive', 
               color='#2ecc71', alpha=0.7)
        
        ax.set_ylabel('Coefficient Value', fontsize=9)
        ax.set_title(f'{dataset}: Model Coefficients', fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Row 2: Weight-feature correlation
        ax = axes[1, col_idx]
        
        correlations = results['weight_feature_correlation']
        pearson_corrs = [correlations[f]['pearson'] for f in feature_names]
        spearman_corrs = [correlations[f]['spearman'] for f in feature_names]
        
        x = np.arange(n_features)
        width = 0.35
        
        ax.bar(x - width/2, pearson_corrs, width, label='Pearson', 
               color='#3498db', alpha=0.7)
        ax.bar(x + width/2, spearman_corrs, width, label='Spearman', 
               color='#9b59b6', alpha=0.7)
        
        ax.set_ylabel('Correlation with Weight', fontsize=9)
        ax.set_title(f'{dataset}: Feature-Weight Correlation', fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Row 3: High vs low weight feature profiles
        ax = axes[2, col_idx]
        
        profile = results['high_weight_profile']
        feature_comp = profile['feature_comparison']
        
        high_means = [feature_comp[f]['high_mean'] for f in feature_names]
        low_means = [feature_comp[f]['low_mean'] for f in feature_names]
        
        x = np.arange(n_features)
        width = 0.35
        
        ax.bar(x - width/2, high_means, width, label='High Weight', 
               color='#e67e22', alpha=0.7)
        ax.bar(x + width/2, low_means, width, label='Low Weight', 
               color='#95a5a6', alpha=0.7)
        
        ax.set_ylabel('Feature Value (standardized)', fontsize=9)
        ax.set_title(f'{dataset}: Feature Profiles', fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.close()


def evaluate_interpretability():
    """
    Main evaluation: Analyze interpretability across datasets.
    """
    print("=" * 80)
    print("Day 19: Model Interpretability Analysis")
    print("=" * 80)
    print("\nResearch Questions:")
    print("1. Which features drive high sample weights?")
    print("2. Does adaptive weighting change model coefficients?")
    print("3. What characterizes high-weight samples?")
    print()
    
    # Feature names for each dataset
    feature_names_map = {
        'COMPAS': ['age', 'priors_count', 'juv_fel', 'juv_misd', 'juv_other'],
        'Adult': ['age', 'education', 'capital_gain', 'capital_loss', 'hours_per_week'],
        'German': ['duration', 'credit_amount', 'installment', 'residence', 'age', 'num_credits']
    }
    
    datasets = {
        'COMPAS': 'compas',
        'Adult': 'adult',
        'German': 'german'
    }
    
    analyzer = InterpretabilityAnalyzer(temperature=0.5)
    all_results = {}
    
    for name, dataset_name in datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing {name} Dataset")
        print(f"{'='*80}")
        
        feature_names = feature_names_map[name]
        results = analyzer.analyze_interpretability(dataset_name, feature_names)
        
        # Print key findings
        print(f"\nWeight Statistics:")
        wstats = results['weight_statistics']
        print(f"  Mean: {wstats['mean']:.3f}")
        print(f"  Std: {wstats['std']:.3f}")
        print(f"  Range: [{wstats['min']:.3f}, {wstats['max']:.3f}]")
        print(f"  90th percentile: {wstats['90th_percentile']:.3f}")
        
        print(f"\nCoefficient Changes:")
        for i, fname in enumerate(feature_names):
            rel_change = results['coefficient_comparison']['relative_change'][i]
            print(f"  {fname}: {rel_change:+.1f}%")
        
        print(f"\nTop Correlated Features (with weight):")
        corrs = results['weight_feature_correlation']
        sorted_features = sorted(corrs.items(), 
                                key=lambda x: abs(x[1]['pearson']), 
                                reverse=True)
        for fname, corr_data in sorted_features[:3]:
            print(f"  {fname}: r={corr_data['pearson']:.3f} (p={corr_data['pearson_pval']:.4f})")
        
        print(f"\nHigh-Weight Sample Profile:")
        profile = results['high_weight_profile']
        print(f"  Count: {profile['high_weight_count']} (top 10%)")
        print(f"  Positive rate: {profile['label_distribution']['high_positive_rate']:.3f} " +
              f"(vs {profile['label_distribution']['low_positive_rate']:.3f} for low-weight)")
        print(f"  Protected rate: {profile['protected_distribution']['high_protected_rate']:.3f} " +
              f"(vs {profile['protected_distribution']['low_protected_rate']:.3f} for low-weight)")
        
        print(f"\nPerformance:")
        perf = results['performance']
        print(f"  Baseline: Acc={perf['baseline_accuracy']:.4f}, EO={perf['baseline_eo']:.4f}")
        print(f"  Adaptive: Acc={perf['adaptive_accuracy']:.4f}, EO={perf['adaptive_eo']:.4f}")
        
        all_results[name] = results
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("Creating visualizations...")
    print(f"{'='*80}")
    
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'day19_interpretability.png')
    
    create_visualizations(all_results, output_path)
    
    # Save results
    metrics_dir = 'results/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    output_file = os.path.join(metrics_dir, 'day19_interpretability.json')
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    print(f"\n{'='*80}")
    print("Day 19 Complete!")
    print(f"{'='*80}")
    
    return all_results


if __name__ == '__main__':
    results = evaluate_interpretability()
