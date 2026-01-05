"""
Day 9: Fairness-Constrained Sample Selection
=============================================

Goal: Fix German dataset failure using direct fairness constraints
Approach: Select samples that explicitly minimize EO disparity

Problem with Meta-Selector on German:
- Meta-selector trained on synthetic tasks doesn't transfer to small real datasets
- Fine-tuning failed (still -136% fairness)
- Need alternative approach for small datasets (N < 1,000)

Solution: Fairness-Aware Greedy Selection
- Instead of just selecting low-loss samples (greedy)
- Select samples that minimize group disparity
- Use Lagrangian optimization with fairness penalty
- Directly optimize: Loss + λ × EO_Disparity

Expected: Beat baseline on German dataset (improve from -136% to +20%)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class FairnessConstrainedSelector:
    """
    Sample selector with explicit fairness constraints
    
    Strategy:
    1. Train initial model on all data
    2. Compute loss and group statistics for each sample
    3. Optimize selection to minimize: loss + λ × group_disparity
    4. Use iterative reweighting or subset selection
    """
    
    def __init__(self, select_ratio=0.7, lambda_fairness=1.0, max_iterations=10):
        """
        Initialize fairness-constrained selector
        
        Args:
            select_ratio: Fraction of samples to select (0-1)
            lambda_fairness: Weight for fairness penalty (higher = more fair)
            max_iterations: Max iterations for iterative selection
        """
        self.select_ratio = select_ratio
        self.lambda_fairness = lambda_fairness
        self.max_iterations = max_iterations
        
    def compute_group_disparity(self, y_true, y_pred, groups):
        """Compute EO disparity between groups"""
        return FairnessMetrics.equalized_odds(y_true, y_pred, groups)
    
    def select_balanced(self, X, y, groups, model=None):
        """
        Select samples with fairness constraint
        
        Strategy: Balanced sampling from each group to ensure fair representation
        """
        n_total = len(X)
        n_select = int(n_total * self.select_ratio)
        
        # Train initial model if not provided
        if model is None:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X, y)
        
        # Compute loss for each sample
        y_proba = model.predict_proba(X)
        losses = -np.log(y_proba[np.arange(len(y)), y] + 1e-10)
        
        # Split by group
        group_0_mask = (groups == 0)
        group_1_mask = (groups == 1)
        
        n_group_0 = group_0_mask.sum()
        n_group_1 = group_1_mask.sum()
        
        # Select proportionally from each group (fair representation)
        n_select_0 = int(n_select * (n_group_0 / n_total))
        n_select_1 = n_select - n_select_0
        
        # Within each group, select lowest loss samples
        indices_0 = np.where(group_0_mask)[0]
        indices_1 = np.where(group_1_mask)[0]
        
        losses_0 = losses[indices_0]
        losses_1 = losses[indices_1]
        
        # Select top samples from each group
        selected_0 = indices_0[np.argsort(losses_0)[:n_select_0]]
        selected_1 = indices_1[np.argsort(losses_1)[:n_select_1]]
        
        selected_indices = np.concatenate([selected_0, selected_1])
        
        return selected_indices
    
    def select_lagrangian(self, X, y, groups, model=None):
        """
        Select samples using Lagrangian optimization
        
        Objective: min_w [ sum(w_i * loss_i) + λ * group_disparity(w) ]
        where w_i ∈ {0, 1} are selection weights
        
        Relaxation: Allow w_i ∈ [0, 1] then threshold
        """
        n_total = len(X)
        n_select = int(n_total * self.select_ratio)
        
        # Train initial model
        if model is None:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X, y)
        
        # Compute sample losses
        y_proba = model.predict_proba(X)
        losses = -np.log(y_proba[np.arange(len(y)), y] + 1e-10)
        
        # Normalize losses
        losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-10)
        
        # Compute fairness penalty for each sample
        # Higher weight for minority group samples to balance representation
        group_weights = np.ones(n_total)
        
        group_0_count = (groups == 0).sum()
        group_1_count = (groups == 1).sum()
        
        # Weight minority group higher
        if group_0_count < group_1_count:
            group_weights[groups == 0] = group_1_count / group_0_count
        else:
            group_weights[groups == 1] = group_0_count / group_1_count
        
        # Combined score: low loss + fair representation
        # Lower score = better (prefer low loss + minority group)
        scores = losses - self.lambda_fairness * (group_weights - 1) / group_weights.max()
        
        # Select samples with best scores
        selected_indices = np.argsort(scores)[:n_select]
        
        return selected_indices
    
    def select_iterative(self, X, y, groups, model=None):
        """
        Iterative selection: gradually add samples that improve fairness
        
        Start with empty set, iteratively add samples that minimize:
        - Loss on added sample
        - Group disparity on current selected set
        """
        n_total = len(X)
        n_select = int(n_total * self.select_ratio)
        
        # Train initial model
        if model is None:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X, y)
        
        # Compute losses
        y_proba = model.predict_proba(X)
        losses = -np.log(y_proba[np.arange(len(y)), y] + 1e-10)
        
        # Start with lowest loss samples from each group (seed)
        selected = []
        remaining = set(range(n_total))
        
        # Add one sample from each group
        for group_id in [0, 1]:
            group_mask = (groups == group_id)
            group_indices = np.where(group_mask)[0]
            if len(group_indices) > 0:
                best_in_group = group_indices[np.argmin(losses[group_indices])]
                selected.append(best_in_group)
                remaining.remove(best_in_group)
        
        # Iteratively add samples
        while len(selected) < n_select and remaining:
            best_candidate = None
            best_score = float('inf')
            
            # Try each remaining sample
            for candidate in list(remaining)[:100]:  # Limit for efficiency
                # Simulate adding this candidate
                test_selected = selected + [candidate]
                test_groups = groups[test_selected]
                
                # Compute group balance
                n_group_0 = (test_groups == 0).sum()
                n_group_1 = (test_groups == 1).sum()
                group_imbalance = abs(n_group_0 / len(test_selected) - (groups == 0).sum() / n_total)
                
                # Score = loss + fairness penalty
                score = losses[candidate] + self.lambda_fairness * group_imbalance
                
                if score < best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        # Fill remaining with lowest loss
        if len(selected) < n_select:
            remaining_list = list(remaining)
            remaining_losses = losses[remaining_list]
            n_remaining = n_select - len(selected)
            additional = np.array(remaining_list)[np.argsort(remaining_losses)[:n_remaining]]
            selected.extend(additional)
        
        return np.array(selected)


def evaluate_method(method_name, X_train, X_test, y_train, y_test, group_train, group_test, 
                   selected_indices=None):
    """Evaluate a selection method"""
    print(f"\n{'='*70}")
    print(f"{method_name}")
    print(f"{'='*70}")
    
    if selected_indices is None:
        # Baseline: use all samples
        X_selected = X_train
        y_selected = y_train
        print(f"\nUsing all {len(X_train)} training samples")
    else:
        X_selected = X_train[selected_indices]
        y_selected = y_train[selected_indices]
        group_selected = group_train[selected_indices]
        
        print(f"\nSelected {len(selected_indices)} / {len(X_train)} samples ({len(selected_indices)/len(X_train)*100:.0f}%)")
        print(f"  Group 0: {(group_selected == 0).sum()} samples")
        print(f"  Group 1: {(group_selected == 1).sum()} samples")
        print(f"  Group balance: {(group_selected == 0).sum() / len(selected_indices):.3f} vs {(group_train == 0).sum() / len(group_train):.3f} (original)")
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_selected, y_selected)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    eo_disparity = FairnessMetrics.equalized_odds(y_test, y_pred, group_test)
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  EO Disparity: {eo_disparity:.4f}")
    
    return {
        'accuracy': accuracy,
        'eo_disparity': eo_disparity,
        'n_selected': len(selected_indices) if selected_indices is not None else len(X_train)
    }


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("DAY 9: FAIRNESS-CONSTRAINED SAMPLE SELECTION")
    print("="*70)
    print("\nObjective: Fix German dataset failure using direct fairness constraints")
    print("Approach: Balanced sampling + Lagrangian optimization + Iterative selection")
    
    # Load German dataset
    print("\n" + "="*70)
    print("LOADING GERMAN DATASET")
    print("="*70)
    
    loader = DataLoader()
    X, y, groups = loader.load_german()
    
    # Split
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X, y, groups, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Group 0 (train): {(group_train == 0).sum()} samples ({(group_train == 0).mean()*100:.1f}%)")
    print(f"  Group 1 (train): {(group_train == 1).sum()} samples ({(group_train == 1).mean()*100:.1f}%)")
    
    # Baseline
    baseline_results = evaluate_method(
        "BASELINE: Train on All Samples",
        X_train, X_test, y_train, y_test, group_train, group_test
    )
    
    # Greedy (from Day 8)
    print(f"\n{'='*70}")
    print("GREEDY: Loss-Based Selection (70%)")
    print(f"{'='*70}")
    
    initial_model = LogisticRegression(max_iter=1000, random_state=42)
    initial_model.fit(X_train, y_train)
    y_proba = initial_model.predict_proba(X_train)
    losses = -np.log(y_proba[np.arange(len(y_train)), y_train] + 1e-10)
    n_select = int(len(X_train) * 0.7)
    greedy_indices = np.argsort(losses)[:n_select]
    
    greedy_results = evaluate_method(
        "GREEDY: Loss-Based Selection (70%)",
        X_train, X_test, y_train, y_test, group_train, group_test,
        greedy_indices
    )
    
    # Fairness-constrained methods
    selector = FairnessConstrainedSelector(select_ratio=0.7, lambda_fairness=1.0)
    
    # Method 1: Balanced sampling
    balanced_indices = selector.select_balanced(X_train, y_train, group_train)
    balanced_results = evaluate_method(
        "FAIRNESS-AWARE: Balanced Group Sampling",
        X_train, X_test, y_train, y_test, group_train, group_test,
        balanced_indices
    )
    
    # Method 2: Lagrangian (different lambda values)
    lagrangian_results = {}
    for lambda_val in [0.5, 1.0, 2.0, 5.0]:
        selector_lag = FairnessConstrainedSelector(select_ratio=0.7, lambda_fairness=lambda_val)
        lag_indices = selector_lag.select_lagrangian(X_train, y_train, group_train)
        lag_results = evaluate_method(
            f"LAGRANGIAN: λ = {lambda_val}",
            X_train, X_test, y_train, y_test, group_train, group_test,
            lag_indices
        )
        lagrangian_results[f'lambda_{lambda_val}'] = lag_results
    
    # Method 3: Iterative selection
    iterative_indices = selector.select_iterative(X_train, y_train, group_train)
    iterative_results = evaluate_method(
        "ITERATIVE: Fairness-Guided Selection",
        X_train, X_test, y_train, y_test, group_train, group_test,
        iterative_indices
    )
    
    # Summary table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON: GERMAN DATASET")
    print(f"{'='*70}")
    
    print(f"\n{'Method':<30} {'Accuracy':<12} {'EO Disparity':<15} {'Fairness Improvement'}")
    print("-" * 70)
    
    baseline_eo = baseline_results['eo_disparity']
    
    all_results = {
        'Baseline': baseline_results,
        'Greedy': greedy_results,
        'Balanced Sampling': balanced_results,
        'Lagrangian (λ=0.5)': lagrangian_results['lambda_0.5'],
        'Lagrangian (λ=1.0)': lagrangian_results['lambda_1.0'],
        'Lagrangian (λ=2.0)': lagrangian_results['lambda_2.0'],
        'Lagrangian (λ=5.0)': lagrangian_results['lambda_5.0'],
        'Iterative': iterative_results
    }
    
    best_method = None
    best_improvement = -float('inf')
    
    for method, results in all_results.items():
        accuracy = results['accuracy']
        eo = results['eo_disparity']
        improvement = ((baseline_eo - eo) / baseline_eo) * 100
        
        print(f"{method:<30} {accuracy:<12.4f} {eo:<15.4f} {improvement:+.1f}%")
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_method = method
    
    # Save results
    os.makedirs('results/metrics', exist_ok=True)
    results_path = 'results/metrics/day9_fairness_constrained.json'
    
    with open(results_path, 'w') as f:
        json.dump({
            method: {k: float(v) if isinstance(v, (np.floating, float, int)) else v
                    for k, v in results.items()}
            for method, results in all_results.items()
        }, f, indent=2)
    
    print(f"\n[OK] Saved results: {results_path}")
    
    # Visualization
    plot_path = 'results/plots/day9_fairness_constrained.png'
    plot_comparison(all_results, baseline_eo, plot_path)
    
    # Final summary
    print(f"\n{'='*70}")
    print("DAY 9 COMPLETE!")
    print(f"{'='*70}")
    print(f"\nBest Method: {best_method}")
    print(f"  Fairness Improvement: {best_improvement:+.1f}%")
    print(f"  Baseline EO: {baseline_eo:.4f}")
    print(f"  Best EO: {all_results[best_method]['eo_disparity']:.4f}")
    
    if best_improvement > 0:
        print(f"\n[OK] SUCCESS: Fairness-constrained selection beats baseline!")
    else:
        print(f"\n[WARNING] All methods failed to beat baseline on German dataset")
    
    print(f"\nComparison with Week 1:")
    print(f"  Meta-selector (pre-trained): -136.4%")
    print(f"  Best fairness-constrained: {best_improvement:+.1f}%")
    
    if best_improvement > -136.4:
        print(f"\n[OK] Fairness constraints BETTER than meta-learning for small datasets!")


def plot_comparison(results, baseline_eo, save_path):
    """Visualize comparison of all methods"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in methods]
    eo_disparities = [results[m]['eo_disparity'] for m in methods]
    improvements = [((baseline_eo - eo) / baseline_eo) * 100 for eo in eo_disparities]
    
    # Plot 1: Accuracy vs EO Disparity (scatter)
    ax1 = axes[0]
    
    # Color by improvement
    colors = ['red' if imp < 0 else 'green' for imp in improvements]
    sizes = [100 + abs(imp) * 5 for imp in improvements]
    
    scatter = ax1.scatter(eo_disparities, accuracies, c=improvements, 
                         s=sizes, alpha=0.6, cmap='RdYlGn', vmin=-150, vmax=50)
    
    # Annotate points
    for i, method in enumerate(methods):
        ax1.annotate(method.replace('Lagrangian', 'Lag').replace(' (', '\n('), 
                    (eo_disparities[i], accuracies[i]),
                    fontsize=7, ha='right')
    
    ax1.set_xlabel('EO Disparity (lower is better)', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Accuracy-Fairness Trade-off', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Fairness Improvement (%)', fontsize=10)
    
    # Plot 2: Fairness improvement bar chart
    ax2 = axes[1]
    
    # Sort by improvement
    sorted_indices = np.argsort(improvements)[::-1]
    sorted_methods = [methods[i] for i in sorted_indices]
    sorted_improvements = [improvements[i] for i in sorted_indices]
    
    colors = ['green' if imp > 0 else 'red' for imp in sorted_improvements]
    bars = ax2.barh(range(len(sorted_methods)), sorted_improvements, color=colors, alpha=0.7)
    
    ax2.set_yticks(range(len(sorted_methods)))
    ax2.set_yticklabels([m.replace('Lagrangian', 'Lag') for m in sorted_methods], fontsize=9)
    ax2.set_xlabel('Fairness Improvement (%)', fontsize=11)
    ax2.set_title('Fairness Improvement Over Baseline', fontsize=12, fontweight='bold')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_improvements)):
        ax2.text(val + 1 if val > 0 else val - 1, i, f'{val:+.1f}%',
                va='center', ha='left' if val > 0 else 'right', fontsize=8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved plot: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
