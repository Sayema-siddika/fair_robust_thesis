"""
Day 17: Intersectional Fairness - Multiple Protected Attributes

Research Question:
Can adaptive weighting handle fairness across MULTIPLE protected attributes simultaneously?

Real-world fairness requires considering intersections:
- Race × Gender (e.g., Black women vs White men)
- Age × Gender
- Multiple dimensions of identity

Current limitation: All previous experiments use single protected attribute.

Experiments:
1. Extend Adult dataset to use Gender AND Race
2. Compute fairness across all intersectional groups
3. Compare single-attribute vs multi-attribute fairness
4. Test if optimizing for one attribute hurts another

Expected Challenges:
- More groups = harder to balance
- Sample size imbalance across intersections
- Potential fairness conflicts between attributes
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
import json
from itertools import product

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class IntersectionalFairnessAnalyzer:
    """
    Analyze fairness across multiple protected attributes simultaneously.
    
    Handles intersectional groups (e.g., Black women, White men, etc.)
    and measures fairness both within single attributes and across intersections.
    """
    
    def __init__(self, temperature=0.5):
        self.temperature = temperature
        
    def create_intersectional_groups(self, protected_attrs):
        """
        Create intersectional group labels from multiple protected attributes.
        
        Args:
            protected_attrs: Dictionary of {attr_name: values_array}
                           e.g., {'gender': [0,1,0,...], 'race': [1,0,1,...]}
        
        Returns:
            group_labels: Array of group IDs (one per sample)
            group_map: Dictionary mapping group_id -> {attr: value, ...}
            group_counts: Dictionary of group sizes
        """
        attr_names = list(protected_attrs.keys())
        attr_arrays = [protected_attrs[name] for name in attr_names]
        
        # Stack attributes into matrix [n_samples, n_attributes]
        attr_matrix = np.column_stack(attr_arrays)
        
        # Create unique group combinations
        unique_combos = np.unique(attr_matrix, axis=0)
        
        # Map each sample to group ID
        group_labels = np.zeros(len(attr_matrix), dtype=int)
        group_map = {}
        group_counts = {}
        
        for group_id, combo in enumerate(unique_combos):
            # Find samples matching this combination
            mask = np.all(attr_matrix == combo, axis=1)
            group_labels[mask] = group_id
            
            # Store mapping
            group_map[group_id] = {
                attr_names[i]: int(combo[i]) 
                for i in range(len(attr_names))
            }
            group_counts[group_id] = int(np.sum(mask))
        
        return group_labels, group_map, group_counts
    
    def compute_intersectional_metrics(self, y_true, y_pred, group_labels, group_map):
        """
        Compute fairness metrics across intersectional groups.
        
        Metrics:
        1. Max disparity across ALL groups (worst-case fairness)
        2. Average pairwise disparity
        3. Per-group metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            group_labels: Intersectional group IDs
            group_map: Mapping of group_id -> attributes
            
        Returns:
            metrics: Dictionary of fairness metrics
        """
        unique_groups = np.unique(group_labels)
        n_groups = len(unique_groups)
        
        # Compute per-group metrics
        group_metrics = {}
        for group_id in unique_groups:
            mask = (group_labels == group_id)
            if np.sum(mask) == 0:
                continue
                
            # True Positive Rate (TPR)
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Positive rate (demographic parity)
            pos_rate = np.mean(y_pred_group)
            
            # TPR and FPR (equalized odds)
            pos_mask = (y_true_group == 1)
            neg_mask = (y_true_group == 0)
            
            tpr = np.mean(y_pred_group[pos_mask]) if np.sum(pos_mask) > 0 else 0.0
            fpr = np.mean(y_pred_group[neg_mask]) if np.sum(neg_mask) > 0 else 0.0
            
            # Accuracy
            acc = accuracy_score(y_true_group, y_pred_group)
            
            group_metrics[group_id] = {
                'attributes': group_map[group_id],
                'size': int(np.sum(mask)),
                'positive_rate': float(pos_rate),
                'tpr': float(tpr),
                'fpr': float(fpr),
                'accuracy': float(acc)
            }
        
        # Compute disparities
        # Max disparity = max difference between any two groups
        pos_rates = [m['positive_rate'] for m in group_metrics.values()]
        tprs = [m['tpr'] for m in group_metrics.values()]
        fprs = [m['fpr'] for m in group_metrics.values()]
        
        dp_disparity = max(pos_rates) - min(pos_rates) if len(pos_rates) > 0 else 0.0
        tpr_disparity = max(tprs) - min(tprs) if len(tprs) > 0 else 0.0
        fpr_disparity = max(fprs) - min(fprs) if len(fprs) > 0 else 0.0
        eo_disparity = max(tpr_disparity, fpr_disparity)
        
        # Average pairwise disparity
        pairwise_dp = []
        pairwise_eo = []
        for i, g1 in enumerate(unique_groups):
            for g2 in unique_groups[i+1:]:
                pairwise_dp.append(abs(group_metrics[g1]['positive_rate'] - 
                                      group_metrics[g2]['positive_rate']))
                pairwise_eo.append(max(
                    abs(group_metrics[g1]['tpr'] - group_metrics[g2]['tpr']),
                    abs(group_metrics[g1]['fpr'] - group_metrics[g2]['fpr'])
                ))
        
        avg_pairwise_dp = np.mean(pairwise_dp) if len(pairwise_dp) > 0 else 0.0
        avg_pairwise_eo = np.mean(pairwise_eo) if len(pairwise_eo) > 0 else 0.0
        
        metrics = {
            'n_groups': n_groups,
            'group_metrics': group_metrics,
            'max_dp_disparity': float(dp_disparity),
            'max_eo_disparity': float(eo_disparity),
            'max_tpr_disparity': float(tpr_disparity),
            'max_fpr_disparity': float(fpr_disparity),
            'avg_pairwise_dp': float(avg_pairwise_dp),
            'avg_pairwise_eo': float(avg_pairwise_eo)
        }
        
        return metrics
    
    def compute_adaptive_weights(self, model, X, y, group_labels, group_map):
        """
        Compute adaptive weights with group-aware rebalancing.
        
        Strategy: Compute base weights, then rebalance across groups
        to ensure no group is systematically downweighted.
        
        Args:
            model: Trained model
            X, y: Features and labels
            group_labels: Intersectional group IDs
            group_map: Group attribute mapping
            
        Returns:
            weights: Sample weights
            meta: Metadata
        """
        # Base adaptive weights
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            confidence = np.max(probs, axis=1)
        else:
            decision = model.decision_function(X)
            confidence = 1.0 / (1.0 + np.exp(-np.abs(decision)))
        
        preds = model.predict(X)
        correctness = (preds == y).astype(float)
        
        raw_weights = confidence * correctness + 0.1
        
        # Temperature scaling
        if self.temperature != 1.0:
            weights = raw_weights ** (1.0 / self.temperature)
        else:
            weights = raw_weights
        
        # Group-aware rebalancing
        # Ensure each group's average weight is at least 0.5
        unique_groups = np.unique(group_labels)
        for group_id in unique_groups:
            mask = (group_labels == group_id)
            group_mean = np.mean(weights[mask])
            
            if group_mean < 0.5:
                # Boost this group's weights
                boost_factor = 0.5 / group_mean
                weights[mask] *= boost_factor
        
        # Normalize
        weights = weights / np.sum(weights) * len(weights)
        
        # Compute group statistics
        group_stats = {}
        for group_id in unique_groups:
            mask = (group_labels == group_id)
            group_stats[int(group_id)] = {
                'mean_weight': float(np.mean(weights[mask])),
                'std_weight': float(np.std(weights[mask])),
                'mean_confidence': float(np.mean(confidence[mask])),
                'mean_correctness': float(np.mean(correctness[mask])),
                'count': int(np.sum(mask)),
                'attributes': group_map[group_id]
            }
        
        meta = {
            'group_stats': group_stats,
            'weight_range': (float(np.min(weights)), float(np.max(weights)))
        }
        
        return weights, meta
    
    def train_and_evaluate(self, X_train, y_train, protected_train,
                          X_test, y_test, protected_test,
                          use_weighting=False):
        """
        Train model and evaluate intersectional fairness.
        
        Args:
            X_train, y_train: Training data
            protected_train: Dict of {attr_name: values} for training
            X_test, y_test: Test data
            protected_test: Dict of {attr_name: values} for testing
            use_weighting: Whether to use adaptive weighting
            
        Returns:
            results: Dictionary with metrics
        """
        # Create intersectional groups
        train_groups, train_map, train_counts = self.create_intersectional_groups(protected_train)
        test_groups, test_map, test_counts = self.create_intersectional_groups(protected_test)
        
        if use_weighting:
            # Train baseline first to get initial weights
            model_init = LogisticRegression(max_iter=1000, random_state=42)
            model_init.fit(X_train, y_train)
            
            # Compute adaptive weights
            weights, weight_meta = self.compute_adaptive_weights(
                model_init, X_train, y_train, train_groups, train_map
            )
            
            # Train with weights
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            # Baseline (no weighting)
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            weights = np.ones(len(X_train))
            weight_meta = {'group_stats': {}}
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Intersectional metrics
        intersectional = self.compute_intersectional_metrics(
            y_test, y_pred, test_groups, test_map
        )
        
        # Single-attribute metrics (for comparison)
        single_attr_metrics = {}
        for attr_name in protected_test.keys():
            attr_values = protected_test[attr_name]
            dp = FairnessMetrics.demographic_parity(y_pred, attr_values)
            eo = FairnessMetrics.equalized_odds(y_test, y_pred, attr_values)
            
            single_attr_metrics[attr_name] = {
                'dp_disparity': float(dp),
                'eo_disparity': float(eo)
            }
        
        results = {
            'accuracy': acc,
            'intersectional': intersectional,
            'single_attribute': single_attr_metrics,
            'used_weighting': use_weighting,
            'weight_meta': weight_meta if use_weighting else None
        }
        
        return results


def load_adult_intersectional():
    """
    Load Adult dataset with BOTH gender and race as protected attributes.
    
    Returns:
        Data dict with multi-attribute protected groups
    """
    print("\nLoading Adult dataset with intersectional attributes...")
    
    # Load raw data to get race information
    data_path = 'data/raw/adult/adult.data'
    
    # Column names for Adult dataset
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    df = pd.read_csv(data_path, names=columns, skipinitialspace=True)
    
    # Create binary target
    df['target'] = (df['income'] == '>50K').astype(int)
    
    # Create race binary: 1 if White, 0 otherwise
    df['race_white'] = (df['race'] == 'White').astype(int)
    
    # Create gender binary: 1 if Female, 0 if Male
    df['gender_female'] = (df['sex'] == 'Female').astype(int)
    
    # Select features (same as standard Adult processing)
    feature_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    X = df[feature_cols].values
    y = df['target'].values
    race = df['race_white'].values
    gender = df['gender_female'].values
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test, race_train, race_test, gender_train, gender_test = train_test_split(
        X, y, race, gender, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Package into result
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'protected_train': {
            'gender': gender_train,
            'race': race_train
        },
        'X_test': X_test,
        'y_test': y_test,
        'protected_test': {
            'gender': gender_test,
            'race': race_test
        }
    }
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Gender distribution (train): {np.bincount(gender_train)}")
    print(f"  Race distribution (train): {np.bincount(race_train)}")
    
    return result


def create_visualizations(results, output_path):
    """
    Visualize intersectional fairness results.
    
    4-panel figure:
    1. Group-wise metrics heatmap
    2. Baseline vs adaptive comparison
    3. Single-attribute vs intersectional disparity
    4. Group size vs fairness scatter
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Extract data
    baseline = results['baseline']
    adaptive = results['adaptive']
    
    # Panel 1: Group-wise positive rates (heatmap)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create matrix: rows=gender, cols=race
    group_metrics_base = baseline['intersectional']['group_metrics']
    group_metrics_adapt = adaptive['intersectional']['group_metrics']
    
    # Build matrices
    matrix_base = np.zeros((2, 2))
    matrix_adapt = np.zeros((2, 2))
    
    for gid, metrics in group_metrics_base.items():
        gender = metrics['attributes']['gender']
        race = metrics['attributes']['race']
        matrix_base[gender, race] = metrics['positive_rate']
    
    for gid, metrics in group_metrics_adapt.items():
        gender = metrics['attributes']['gender']
        race = metrics['attributes']['race']
        matrix_adapt[gender, race] = metrics['positive_rate']
    
    # Plot side-by-side
    x = np.arange(2)
    width = 0.35
    
    labels = ['Male\nNon-White', 'Male\nWhite', 'Female\nNon-White', 'Female\nWhite']
    base_rates = [matrix_base[0,0], matrix_base[0,1], matrix_base[1,0], matrix_base[1,1]]
    adapt_rates = [matrix_adapt[0,0], matrix_adapt[0,1], matrix_adapt[1,0], matrix_adapt[1,1]]
    
    x_pos = np.arange(len(labels))
    ax1.bar(x_pos - width/2, base_rates, width, label='Baseline', color='#e74c3c', alpha=0.7)
    ax1.bar(x_pos + width/2, adapt_rates, width, label='Adaptive', color='#2ecc71', alpha=0.7)
    
    ax1.set_ylabel('Positive Prediction Rate', fontsize=11)
    ax1.set_title('Positive Rates by Intersectional Group', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=np.mean(base_rates), color='red', linestyle='--', alpha=0.5, label='Overall mean')
    
    # Panel 2: Disparity comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    metrics_to_plot = ['Max DP', 'Max EO', 'Avg Pairwise DP', 'Avg Pairwise EO']
    base_values = [
        baseline['intersectional']['max_dp_disparity'],
        baseline['intersectional']['max_eo_disparity'],
        baseline['intersectional']['avg_pairwise_dp'],
        baseline['intersectional']['avg_pairwise_eo']
    ]
    adapt_values = [
        adaptive['intersectional']['max_dp_disparity'],
        adaptive['intersectional']['max_eo_disparity'],
        adaptive['intersectional']['avg_pairwise_dp'],
        adaptive['intersectional']['avg_pairwise_eo']
    ]
    
    x_pos = np.arange(len(metrics_to_plot))
    ax2.bar(x_pos - width/2, base_values, width, label='Baseline', color='#e74c3c', alpha=0.7)
    ax2.bar(x_pos + width/2, adapt_values, width, label='Adaptive', color='#2ecc71', alpha=0.7)
    
    ax2.set_ylabel('Disparity', fontsize=11)
    ax2.set_title('Intersectional Fairness Metrics', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics_to_plot, fontsize=9, rotation=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Single-attribute vs intersectional
    ax3 = fig.add_subplot(gs[1, 0])
    
    attr_names = ['Gender\n(single)', 'Race\n(single)', 'Intersection\n(max)']
    base_eo = [
        baseline['single_attribute']['gender']['eo_disparity'],
        baseline['single_attribute']['race']['eo_disparity'],
        baseline['intersectional']['max_eo_disparity']
    ]
    adapt_eo = [
        adaptive['single_attribute']['gender']['eo_disparity'],
        adaptive['single_attribute']['race']['eo_disparity'],
        adaptive['intersectional']['max_eo_disparity']
    ]
    
    x_pos = np.arange(len(attr_names))
    ax3.bar(x_pos - width/2, base_eo, width, label='Baseline', color='#e74c3c', alpha=0.7)
    ax3.bar(x_pos + width/2, adapt_eo, width, label='Adaptive', color='#2ecc71', alpha=0.7)
    
    ax3.set_ylabel('EO Disparity', fontsize=11)
    ax3.set_title('Single vs Intersectional Fairness', fontweight='bold', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(attr_names, fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Improvement summary
    ax4 = fig.add_subplot(gs[1, 1])
    
    improvements = []
    improvement_labels = []
    
    for attr in ['gender', 'race']:
        base_val = baseline['single_attribute'][attr]['eo_disparity']
        adapt_val = adaptive['single_attribute'][attr]['eo_disparity']
        if base_val > 0:
            improv = (base_val - adapt_val) / base_val * 100
            improvements.append(improv)
            improvement_labels.append(f"{attr.capitalize()}\nEO")
    
    # Intersectional
    base_inter = baseline['intersectional']['max_eo_disparity']
    adapt_inter = adaptive['intersectional']['max_eo_disparity']
    if base_inter > 0:
        improv_inter = (base_inter - adapt_inter) / base_inter * 100
        improvements.append(improv_inter)
        improvement_labels.append("Intersectional\nMax EO")
    
    colors_improv = ['#3498db', '#9b59b6', '#e67e22']
    bars = ax4.bar(improvement_labels, improvements, color=colors_improv, alpha=0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_ylabel('Improvement (%)', fontsize=11)
    ax4.set_title('Fairness Improvement with Adaptive Weighting', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.suptitle('Day 17: Intersectional Fairness Analysis (Gender × Race)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.close()


def evaluate_intersectional_fairness():
    """
    Main evaluation: Compare single vs multi-attribute fairness.
    """
    print("=" * 80)
    print("Day 17: Intersectional Fairness Analysis")
    print("=" * 80)
    print("\nResearch Questions:")
    print("1. Can adaptive weighting handle multiple protected attributes?")
    print("2. Does optimizing for one attribute hurt another?")
    print("3. How does intersectional disparity compare to single-attribute?")
    print()
    
    # Load data with multiple protected attributes
    data = load_adult_intersectional()
    
    X_train = data['X_train']
    y_train = data['y_train']
    protected_train = data['protected_train']
    
    X_test = data['X_test']
    y_test = data['y_test']
    protected_test = data['protected_test']
    
    # Initialize analyzer
    analyzer = IntersectionalFairnessAnalyzer(temperature=0.5)
    
    # Evaluate baseline (no weighting)
    print("\n" + "="*80)
    print("Baseline (No Weighting)")
    print("="*80)
    
    baseline_results = analyzer.train_and_evaluate(
        X_train, y_train, protected_train,
        X_test, y_test, protected_test,
        use_weighting=False
    )
    
    print(f"\nAccuracy: {baseline_results['accuracy']:.4f}")
    print(f"\nSingle-Attribute Metrics:")
    for attr, metrics in baseline_results['single_attribute'].items():
        print(f"  {attr.capitalize()}:")
        print(f"    DP Disparity: {metrics['dp_disparity']:.4f}")
        print(f"    EO Disparity: {metrics['eo_disparity']:.4f}")
    
    print(f"\nIntersectional Metrics:")
    inter = baseline_results['intersectional']
    print(f"  Number of groups: {inter['n_groups']}")
    print(f"  Max DP Disparity: {inter['max_dp_disparity']:.4f}")
    print(f"  Max EO Disparity: {inter['max_eo_disparity']:.4f}")
    print(f"  Avg Pairwise DP: {inter['avg_pairwise_dp']:.4f}")
    print(f"  Avg Pairwise EO: {inter['avg_pairwise_eo']:.4f}")
    
    print(f"\n  Group-wise metrics:")
    for gid, gmetrics in inter['group_metrics'].items():
        attrs = gmetrics['attributes']
        gender_label = "Female" if attrs['gender'] == 1 else "Male"
        race_label = "White" if attrs['race'] == 1 else "Non-White"
        print(f"    {gender_label} × {race_label}: n={gmetrics['size']}, "
              f"pos_rate={gmetrics['positive_rate']:.3f}, "
              f"TPR={gmetrics['tpr']:.3f}, acc={gmetrics['accuracy']:.3f}")
    
    # Evaluate adaptive weighting
    print("\n" + "="*80)
    print("Adaptive Weighting (T=0.5)")
    print("="*80)
    
    adaptive_results = analyzer.train_and_evaluate(
        X_train, y_train, protected_train,
        X_test, y_test, protected_test,
        use_weighting=True
    )
    
    print(f"\nAccuracy: {adaptive_results['accuracy']:.4f}")
    print(f"\nSingle-Attribute Metrics:")
    for attr, metrics in adaptive_results['single_attribute'].items():
        print(f"  {attr.capitalize()}:")
        print(f"    DP Disparity: {metrics['dp_disparity']:.4f}")
        print(f"    EO Disparity: {metrics['eo_disparity']:.4f}")
    
    print(f"\nIntersectional Metrics:")
    inter = adaptive_results['intersectional']
    print(f"  Number of groups: {inter['n_groups']}")
    print(f"  Max DP Disparity: {inter['max_dp_disparity']:.4f}")
    print(f"  Max EO Disparity: {inter['max_eo_disparity']:.4f}")
    print(f"  Avg Pairwise DP: {inter['avg_pairwise_dp']:.4f}")
    print(f"  Avg Pairwise EO: {inter['avg_pairwise_eo']:.4f}")
    
    print(f"\n  Group-wise metrics:")
    for gid, gmetrics in inter['group_metrics'].items():
        attrs = gmetrics['attributes']
        gender_label = "Female" if attrs['gender'] == 1 else "Male"
        race_label = "White" if attrs['race'] == 1 else "Non-White"
        print(f"    {gender_label} × {race_label}: n={gmetrics['size']}, "
              f"pos_rate={gmetrics['positive_rate']:.3f}, "
              f"TPR={gmetrics['tpr']:.3f}, acc={gmetrics['accuracy']:.3f}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON: Baseline vs Adaptive")
    print("="*80)
    
    for attr in ['gender', 'race']:
        base_eo = baseline_results['single_attribute'][attr]['eo_disparity']
        adapt_eo = adaptive_results['single_attribute'][attr]['eo_disparity']
        improvement = (base_eo - adapt_eo) / base_eo * 100 if base_eo > 0 else 0
        print(f"\n{attr.capitalize()} EO:")
        print(f"  Baseline: {base_eo:.4f}")
        print(f"  Adaptive: {adapt_eo:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")
    
    base_inter_eo = baseline_results['intersectional']['max_eo_disparity']
    adapt_inter_eo = adaptive_results['intersectional']['max_eo_disparity']
    inter_improvement = (base_inter_eo - adapt_inter_eo) / base_inter_eo * 100 if base_inter_eo > 0 else 0
    
    print(f"\nIntersectional Max EO:")
    print(f"  Baseline: {base_inter_eo:.4f}")
    print(f"  Adaptive: {adapt_inter_eo:.4f}")
    print(f"  Improvement: {inter_improvement:+.1f}%")
    
    # Save results
    results_all = {
        'baseline': baseline_results,
        'adaptive': adaptive_results,
        'comparison': {
            'gender_improvement': float((baseline_results['single_attribute']['gender']['eo_disparity'] - 
                                        adaptive_results['single_attribute']['gender']['eo_disparity']) / 
                                       baseline_results['single_attribute']['gender']['eo_disparity'] * 100),
            'race_improvement': float((baseline_results['single_attribute']['race']['eo_disparity'] - 
                                      adaptive_results['single_attribute']['race']['eo_disparity']) / 
                                     baseline_results['single_attribute']['race']['eo_disparity'] * 100),
            'intersectional_improvement': float(inter_improvement)
        }
    }
    
    # Create visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)
    
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'day17_intersectional_fairness.png')
    
    create_visualizations(results_all, output_path)
    
    # Save metrics
    metrics_dir = 'results/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    output_file = os.path.join(metrics_dir, 'day17_intersectional_fairness.json')
    
    # Clean results (remove non-serializable objects)
    def convert_keys(obj):
        """Convert numpy int keys to Python int."""
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, (np.integer, np.int32, np.int64)) else k: convert_keys(v) 
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys(item) for item in obj]
        else:
            return obj
    
    clean_results = {
        'baseline': {
            'accuracy': baseline_results['accuracy'],
            'intersectional': convert_keys(baseline_results['intersectional']),
            'single_attribute': baseline_results['single_attribute']
        },
        'adaptive': {
            'accuracy': adaptive_results['accuracy'],
            'intersectional': convert_keys(adaptive_results['intersectional']),
            'single_attribute': adaptive_results['single_attribute']
        },
        'comparison': results_all['comparison']
    }
    
    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    print("\n" + "="*80)
    print("Day 17 Complete!")
    print("="*80)
    
    return results_all


if __name__ == '__main__':
    results = evaluate_intersectional_fairness()
