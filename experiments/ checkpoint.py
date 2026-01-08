"""
Day 7: Week 1 Checkpoint - Multi-Dataset Evaluation
====================================================

Test trained meta-selector on all 3 datasets:
- COMPAS (6K samples)
- Adult (30K samples) 
- German (1K samples) <- KEY TEST: Can meta-selector fix greedy's -85% failure?

Compare:
- Baseline (no selection)
- Greedy selector (loss-based, tau=0.7)
- Meta-Selector (trained on 80 synthetic tasks)

Expected:
- COMPAS: Meta > Greedy (already proven: +11.7% vs +0.7%)
- Adult: Meta ≈ Greedy (both should work on large dataset)
- German: Meta >> Greedy (FIX THE -85% FAILURE!)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics
from src.models.meta_selector import MetaSelector


def evaluate_on_dataset(dataset_name, meta_selector, verbose=True):
    """
    Evaluate all methods on a single dataset
    
    Args:
        dataset_name: 'compas', 'adult', or 'german'
        meta_selector: Trained MetaSelector instance
        verbose: Print results
    
    Returns:
        results: Dictionary with all metrics
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Evaluating on {dataset_name.upper()} Dataset")
        print(f"{'='*80}")
    
    # Load dataset
    loader = DataLoader(dataset_name)
    
    if dataset_name == 'compas':
        X, y, z = loader.load_compas()
    elif dataset_name == 'adult':
        X, y, z = loader.load_adult()
    elif dataset_name == 'german':
        X, y, z = loader.load_german()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Train/test split
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        X, y, z, test_size=0.3, random_state=42, stratify=y
    )
    
    if verbose:
        print(f"\n{dataset_name.upper()}: {len(X_train)} train, {len(X_test)} test samples")
    
    results = {}
    
    # 1. Baseline (all samples)
    if verbose:
        print(f"\n1. Baseline (no selection)...")
    
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)
    
    baseline_pred = baseline_model.predict(X_test)
    baseline_metrics = FairnessMetrics.compute_all_metrics(y_test, baseline_pred, z_test)
    
    results['baseline'] = baseline_metrics
    
    if verbose:
        print(f"   Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"   EO Disparity: {baseline_metrics['eo_disparity']:.4f}")
    
    # 2. Greedy selector (loss-based)
    if verbose:
        print(f"\n2. Greedy Selector (loss-based, tau=0.7)...")
    
    # Train initial model
    temp_model = LogisticRegression(max_iter=1000, random_state=42)
    temp_model.fit(X_train, y_train)
    
    # Get predictions and compute losses (optimized)
    train_probs = temp_model.predict_proba(X_train)[:, 1]
    
    eps = 1e-15
    losses = -(y_train * np.log(train_probs + eps) + (1 - y_train) * np.log(1 - train_probs + eps))
    
    # Select top 70% with lowest loss
    tau = 0.7
    n_select = int(len(X_train) * tau)
    greedy_selected_idx = np.argsort(losses)[:n_select]
    
    # Train on selected
    greedy_model = LogisticRegression(max_iter=1000, random_state=42)
    greedy_model.fit(X_train[greedy_selected_idx], y_train[greedy_selected_idx])
    
    greedy_pred = greedy_model.predict(X_test)
    greedy_metrics = FairnessMetrics.compute_all_metrics(y_test, greedy_pred, z_test)
    
    results['greedy'] = greedy_metrics
    results['greedy_n_selected'] = len(greedy_selected_idx)
    
    if verbose:
        print(f"   Selected: {len(greedy_selected_idx)}/{len(X_train)} samples ({len(greedy_selected_idx)/len(X_train):.1%})")
        print(f"   Accuracy: {greedy_metrics['accuracy']:.4f} ({greedy_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f})")
        print(f"   EO Disparity: {greedy_metrics['eo_disparity']:.4f} ({greedy_metrics['eo_disparity'] - baseline_metrics['eo_disparity']:+.4f})")
    
    # 3. Meta-Selector
    if verbose:
        print(f"\n3. Meta-Selector (trained on synthetic)...")
    
    # Train initial model to get predictions
    temp_model = LogisticRegression(max_iter=1000, random_state=42)
    temp_model.fit(X_train, y_train)
    train_probs = temp_model.predict_proba(X_train)[:, 1]
    
    # Extract 10 meta-features (optimized - batch computation)
    # Compute losses efficiently
    eps = 1e-15
    losses = -(y_train * np.log(train_probs + eps) + (1 - y_train) * np.log(1 - train_probs + eps))
    
    confidence = np.maximum(train_probs, 1 - train_probs)
    
    entropy = -(train_probs * np.log(train_probs + eps) +
                (1 - train_probs) * np.log(1 - train_probs + eps))
    
    group_loss = np.array([
        np.mean(losses[z_train == z_train[i]]) 
        for i in range(len(z_train))
    ])
    
    group_confidence = np.array([
        np.mean(confidence[z_train == z_train[i]])
        for i in range(len(z_train))
    ])
    
    predictions = (train_probs > 0.5).astype(float)
    margin = np.abs(train_probs - 0.5)
    difficulty = np.argsort(np.argsort(losses)) / len(losses)
    
    features = np.column_stack([
        losses, confidence, entropy, z_train,
        group_loss, group_confidence, predictions, y_train,
        margin, difficulty
    ])
    
    # Get selection probabilities from trained policy network
    features_tensor = torch.FloatTensor(features)
    meta_selector.policy_net.eval()
    
    with torch.no_grad():
        selection_probs = meta_selector.policy_net(features_tensor).numpy().flatten()
    
    # Select top 70% (same as greedy for fair comparison)
    n_select = int(len(X_train) * 0.7)
    meta_selected_idx = np.argsort(selection_probs)[-n_select:]
    
    # Train on selected
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(X_train[meta_selected_idx], y_train[meta_selected_idx])
    
    meta_pred = meta_model.predict(X_test)
    meta_metrics = FairnessMetrics.compute_all_metrics(y_test, meta_pred, z_test)
    
    results['meta'] = meta_metrics
    results['meta_n_selected'] = len(meta_selected_idx)
    
    if verbose:
        print(f"   Selected: {len(meta_selected_idx)}/{len(X_train)} samples ({len(meta_selected_idx)/len(X_train):.1%})")
        print(f"   Accuracy: {meta_metrics['accuracy']:.4f} ({meta_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f})")
        print(f"   EO Disparity: {meta_metrics['eo_disparity']:.4f} ({meta_metrics['eo_disparity'] - baseline_metrics['eo_disparity']:+.4f})")
    
    # Comparison
    if verbose:
        print(f"\n{'-'*80}")
        print(f"Summary for {dataset_name.upper()}:")
        print(f"{'-'*80}")
        
        baseline_eo = baseline_metrics['eo_disparity']
        greedy_eo = greedy_metrics['eo_disparity']
        meta_eo = meta_metrics['eo_disparity']
        
        greedy_gain = (baseline_eo - greedy_eo) / baseline_eo * 100
        meta_gain = (baseline_eo - meta_eo) / baseline_eo * 100
        
        print(f"Fairness Improvement:")
        print(f"  Greedy: {greedy_gain:+.1f}%")
        print(f"  Meta:   {meta_gain:+.1f}%")
        print(f"  Meta vs Greedy: {meta_gain - greedy_gain:+.1f} percentage points")
    
    return results


def create_comparison_table(all_results):
    """
    Create comprehensive comparison table
    
    Args:
        all_results: Dictionary with results for each dataset
    """
    print(f"\n{'='*100}")
    print(f"WEEK 1 CHECKPOINT: Multi-Dataset Comparison")
    print(f"{'='*100}")
    
    print(f"\n{'Dataset':<10} {'Method':<15} {'Accuracy':<12} {'EO Disparity':<15} {'Fairness Gain':<15}")
    print(f"{'-'*100}")
    
    for dataset_name in ['compas', 'adult', 'german']:
        if dataset_name not in all_results:
            continue
        
        results = all_results[dataset_name]
        
        # Baseline
        baseline_acc = results['baseline']['accuracy']
        baseline_eo = results['baseline']['eo_disparity']
        
        print(f"{dataset_name.upper():<10} {'Baseline':<15} {baseline_acc:.4f}       {baseline_eo:.4f}          -")
        
        # Greedy
        greedy_acc = results['greedy']['accuracy']
        greedy_eo = results['greedy']['eo_disparity']
        greedy_gain = (baseline_eo - greedy_eo) / baseline_eo * 100
        
        print(f"{'':10} {'Greedy':<15} {greedy_acc:.4f}       {greedy_eo:.4f}          {greedy_gain:+.1f}%")
        
        # Meta
        meta_acc = results['meta']['accuracy']
        meta_eo = results['meta']['eo_disparity']
        meta_gain = (baseline_eo - meta_eo) / baseline_eo * 100
        
        improvement = "✓" if meta_gain > greedy_gain else "✗"
        print(f"{'':10} {'Meta-Selector':<15} {meta_acc:.4f}       {meta_eo:.4f}          {meta_gain:+.1f}% {improvement}")
        
        print(f"{'-'*100}")
    
    print(f"\n{'='*100}")
    print(f"KEY FINDINGS:")
    print(f"{'='*100}")
    
    # Analyze German dataset specifically
    if 'german' in all_results:
        german_results = all_results['german']
        baseline_eo = german_results['baseline']['eo_disparity']
        greedy_eo = german_results['greedy']['eo_disparity']
        meta_eo = german_results['meta']['eo_disparity']
        
        greedy_gain = (baseline_eo - greedy_eo) / baseline_eo * 100
        meta_gain = (baseline_eo - meta_eo) / baseline_eo * 100
        
        print(f"\nGerman Dataset (Small Dataset - 1K samples):")
        print(f"  Greedy:      {greedy_gain:+.1f}% fairness change")
        print(f"  Meta:        {meta_gain:+.1f}% fairness improvement")
        
        if meta_gain > 0 and greedy_gain < 0:
            print(f"  [SUCCESS] Meta-selector FIXED the small dataset problem!")
            print(f"            Greedy failed ({greedy_gain:.1f}%), Meta succeeded ({meta_gain:+.1f}%)")
        elif meta_gain > greedy_gain:
            print(f"  [IMPROVED] Meta-selector better than greedy by {meta_gain - greedy_gain:+.1f}pp")
        else:
            print(f"  [PARTIAL] Meta-selector improvement: {meta_gain - greedy_gain:+.1f}pp")


def plot_comparison(all_results, save_path='results/plots/week1_comparison.png'):
    """
    Create visualization comparing all methods across datasets
    
    Args:
        all_results: Dictionary with results for each dataset
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Week 1: Meta-Selector vs Greedy vs Baseline', fontsize=16, fontweight='bold')
    
    datasets = ['COMPAS', 'Adult', 'German']
    methods = ['Baseline', 'Greedy', 'Meta']
    colors = ['gray', 'orange', 'green']
    
    # Accuracy plot
    for i, method_key in enumerate(['baseline', 'greedy', 'meta']):
        accuracies = [
            all_results[ds.lower()][method_key]['accuracy']
            for ds in datasets if ds.lower() in all_results
        ]
        
        x = np.arange(len(accuracies))
        axes[0].bar(x + i*0.25, accuracies, width=0.25, label=methods[i], color=colors[i], alpha=0.8)
    
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_xticks(np.arange(len(datasets)) + 0.25)
    axes[0].set_xticklabels([ds for ds in datasets if ds.lower() in all_results])
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    
    # Fairness plot (lower is better)
    for i, method_key in enumerate(['baseline', 'greedy', 'meta']):
        disparities = [
            all_results[ds.lower()][method_key]['eo_disparity']
            for ds in datasets if ds.lower() in all_results
        ]
        
        x = np.arange(len(disparities))
        axes[1].bar(x + i*0.25, disparities, width=0.25, label=methods[i], color=colors[i], alpha=0.8)
    
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel('EO Disparity (lower = better)')
    axes[1].set_title('Fairness Comparison')
    axes[1].set_xticks(np.arange(len(datasets)) + 0.25)
    axes[1].set_xticklabels([ds for ds in datasets if ds.lower() in all_results])
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved comparison plot to {save_path}")


def main():
    print("="*100)
    print("DAY 7: WEEK 1 CHECKPOINT - Multi-Dataset Evaluation")
    print("="*100)
    
    # Load trained meta-selector
    print("\nLoading trained meta-selector...")
    meta_selector = MetaSelector(
        input_dim=10,
        hidden_dims=[64, 32],
        meta_lr=0.001,
        inner_lr=0.01
    )
    
    # Load the trained weights
    checkpoint_path = 'results/checkpoints/meta_selector_final.pt'
    if os.path.exists(checkpoint_path):
        meta_selector.load(checkpoint_path)
        print(f"[OK] Loaded trained model from {checkpoint_path}")
    else:
        print(f"[WARNING] No trained model found at {checkpoint_path}")
        print("          Using untrained meta-selector for comparison")
    
    # Evaluate on all datasets
    all_results = {}
    
    for dataset_name in ['compas', 'adult', 'german']:
        results = evaluate_on_dataset(dataset_name, meta_selector, verbose=True)
        all_results[dataset_name] = results
    
    # Create comparison table
    create_comparison_table(all_results)
    
    # Plot comparison
    plot_comparison(all_results)
    
    # Save results
    import json
    
    results_dict = {}
    for dataset_name, results in all_results.items():
        results_dict[dataset_name] = {
            'baseline': {k: float(v) for k, v in results['baseline'].items()},
            'greedy': {k: float(v) for k, v in results['greedy'].items()},
            'meta': {k: float(v) for k, v in results['meta'].items()}
        }
    
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/week1_evaluation.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n[OK] Saved results to results/metrics/week1_evaluation.json")
    
    print("\n" + "="*100)
    print("WEEK 1 CHECKPOINT COMPLETE!")
    print("="*100)
    print("\nNext: Week 2 - Advanced Meta-Learning Techniques")
    print("  - Uncertainty weighting")
    print("  - Pareto optimization") 
    print("  - Transfer learning")


if __name__ == '__main__':
    main()
