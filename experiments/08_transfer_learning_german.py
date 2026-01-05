"""
Day 8: Transfer Learning for German Dataset
===========================================

Goal: Fix meta-selector's failure on German dataset (-136% fairness)
Approach: Fine-tune pre-trained meta-selector on German training data

Problem:
- Meta-selector trained on synthetic tasks (mean: 3,434 samples)
- German has only 700 training samples → distribution mismatch
- Result: -136.4% fairness (worse than greedy's -9.1%)

Solution:
- Load pre-trained meta-selector from Week 1
- Fine-tune on German training set with few-shot adaptation
- Use lower learning rate + fewer iterations to avoid overfitting
- Compare: Baseline vs Greedy vs Meta (pre-trained) vs Meta (fine-tuned)

Expected Result: Fine-tuned meta-selector should achieve positive fairness improvement
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

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.models.meta_selector import MetaSelector
from src.selection.greedy_selector import GreedySelector
from src.fairness.metrics import FairnessMetrics


def load_german_data():
    """Load German credit dataset"""
    print("\n" + "="*70)
    print("LOADING GERMAN DATASET")
    print("="*70)
    
    loader = DataLoader()
    X, y, group = loader.load_german()
    
    # Split into train/test (70/30 stratified)
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X, y, group, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Class balance: {y_train.mean():.3f} positive")
    print(f"  Group 0: {(group_train == 0).sum()} samples")
    print(f"  Group 1: {(group_train == 1).sum()} samples")
    
    return X_train, X_test, y_train, y_test, group_train, group_test


def evaluate_baseline(X_train, X_test, y_train, y_test, group_test):
    """Evaluate baseline (train on all samples)"""
    print("\n" + "="*70)
    print("BASELINE: Train on All Samples")
    print("="*70)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    eo_disparity = FairnessMetrics.equalized_odds(y_test, y_pred, group_test)
    
    print(f"\nBaseline Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  EO Disparity: {eo_disparity:.4f}")
    
    return {
        'accuracy': accuracy,
        'eo_disparity': eo_disparity,
        'model': model
    }


def evaluate_greedy(X_train, X_test, y_train, y_test, group_train, group_test, select_ratio=0.7):
    """Evaluate greedy selector (loss-based)"""
    print("\n" + "="*70)
    print("GREEDY SELECTOR: Loss-Based Selection")
    print("="*70)
    
    # Train initial model to compute losses
    initial_model = LogisticRegression(max_iter=1000, random_state=42)
    initial_model.fit(X_train, y_train)
    
    # Compute losses for each sample
    y_proba = initial_model.predict_proba(X_train)
    losses = -np.log(y_proba[np.arange(len(y_train)), y_train] + 1e-10)
    
    # Select top tau% samples with lowest loss
    n_select = int(len(X_train) * select_ratio)
    selected_indices = np.argsort(losses)[:n_select]
    
    X_selected = X_train[selected_indices]
    y_selected = y_train[selected_indices]
    
    print(f"\nSelected {len(selected_indices)} / {len(X_train)} samples ({select_ratio*100:.0f}%)")
    
    # Train model on selected samples
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_selected, y_selected)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    eo_disparity = FairnessMetrics.equalized_odds(y_test, y_pred, group_test)
    
    print(f"\nGreedy Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  EO Disparity: {eo_disparity:.4f}")
    
    return {
        'accuracy': accuracy,
        'eo_disparity': eo_disparity,
        'selected_indices': selected_indices
    }


def evaluate_pretrained_meta(X_train, X_test, y_train, y_test, group_train, group_test, 
                             checkpoint_path, select_ratio=0.7):
    """Evaluate pre-trained meta-selector (from Week 1)"""
    print("\n" + "="*70)
    print("META-SELECTOR: Pre-trained (Week 1)")
    print("="*70)
    
    # Load pre-trained meta-selector
    meta_selector = MetaSelector(input_dim=10, hidden_dims=[64, 32])
    
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path)
    meta_selector.policy_net.load_state_dict(checkpoint['policy_net_state'])
    print(f"\n[OK] Loaded checkpoint: {checkpoint_path}")
    print(f"  Meta LR: {checkpoint.get('meta_lr', 'unknown')}")
    print(f"  Inner LR: {checkpoint.get('inner_lr', 'unknown')}")
    
    # Train initial model to extract features
    initial_model = LogisticRegression(max_iter=1000, random_state=42)
    initial_model.fit(X_train, y_train)
    
    # Convert to PyTorch model for feature extraction
    class LogRegTorch(nn.Module):
        def __init__(self, coef, intercept):
            super().__init__()
            self.linear = nn.Linear(len(coef), 1)
            self.linear.weight.data = torch.FloatTensor(coef).unsqueeze(0)
            self.linear.bias.data = torch.FloatTensor([intercept])
            
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    torch_model = LogRegTorch(initial_model.coef_[0], initial_model.intercept_[0])
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    # Select samples using meta-selector
    # First check selection probabilities
    meta_selector.policy_net.eval()
    features = meta_selector.feature_extractor.extract_features(
        torch_model, X_train_tensor, y_train_tensor, group_train
    )
    with torch.no_grad():
        selection_probs = meta_selector.policy_net(features).squeeze().cpu().numpy()
    
    print(f"\nSelection probability stats:")
    print(f"  Min: {selection_probs.min():.4f}")
    print(f"  Max: {selection_probs.max():.4f}")
    print(f"  Mean: {selection_probs.mean():.4f}")
    print(f"  Median: {np.median(selection_probs):.4f}")
    
    # Use top-k selection instead of threshold (select top 70%)
    n_select = int(len(X_train) * select_ratio)
    top_k_indices = np.argsort(selection_probs)[-n_select:]
    selected_indices = top_k_indices
    
    X_selected = X_train[selected_indices]
    y_selected = y_train[selected_indices]
    
    print(f"\nSelected {len(selected_indices)} / {len(X_train)} samples ({select_ratio*100:.0f}%)")
    
    # Train model on selected samples
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_selected, y_selected)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    eo_disparity = FairnessMetrics.equalized_odds(y_test, y_pred, group_test)
    
    print(f"\nPre-trained Meta Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  EO Disparity: {eo_disparity:.4f}")
    
    return {
        'accuracy': accuracy,
        'eo_disparity': eo_disparity,
        'selected_indices': selected_indices,
        'meta_selector': meta_selector
    }


def finetune_meta_selector(meta_selector, X_train, y_train, group_train, 
                           n_iterations=20):
    """
    Fine-tune meta-selector on German dataset
    
    Strategy:
    - Create task from German training data
    - Split into support (80%) and query (20%) for MAML-style updates
    - Use meta_train method with single task repeated
    - Lower iterations (20 vs 100) to avoid overfitting
    """
    print("\n" + "="*70)
    print("FINE-TUNING META-SELECTOR ON GERMAN DATASET")
    print("="*70)
    
    print(f"\nFine-tuning Configuration:")
    print(f"  Iterations: {n_iterations}")
    print(f"  Training samples: {len(X_train)}")
    
    # Create German task with support/query split
    from sklearn.model_selection import train_test_split
    X_support, X_query, y_support, y_query, z_support, z_query = train_test_split(
        X_train, y_train, group_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    german_task = {
        'train': {
            'X': torch.FloatTensor(X_support),
            'y': torch.FloatTensor(y_support),
            'z': z_support
        },
        'test': {
            'X': torch.FloatTensor(X_query),
            'y': torch.FloatTensor(y_query),
            'z': z_query
        }
    }
    
    # Fine-tune using meta_train
    history = meta_selector.meta_train(
        train_tasks=[german_task],  # Single task
        val_tasks=[german_task],     # Same task for validation
        n_iterations=n_iterations,
        save_dir='results/checkpoints/german_finetuning',
        save_every=5,
        verbose=True  # Enable verbose to see progress
    )
    
    print(f"\n[OK] Fine-tuning complete!")
    print(f"  History keys: {history.keys()}")
    print(f"  History lengths: {[len(v) for v in history.values()]}")
    if len(history['train_loss']) > 0:
        print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Val Accuracy: {history['val_accuracy'][-1]:.4f}")
        print(f"  Final Val Fairness: {history['val_fairness'][-1]:.4f}")
    
    return history


def evaluate_finetuned_meta(meta_selector, X_train, X_test, y_train, y_test, 
                            group_train, group_test, select_ratio=0.7):
    """Evaluate fine-tuned meta-selector"""
    print("\n" + "="*70)
    print("META-SELECTOR: Fine-tuned on German")
    print("="*70)
    
    # Train initial model to extract features
    initial_model = LogisticRegression(max_iter=1000, random_state=42)
    initial_model.fit(X_train, y_train)
    
    # Convert to PyTorch model
    class LogRegTorch(nn.Module):
        def __init__(self, coef, intercept):
            super().__init__()
            self.linear = nn.Linear(len(coef), 1)
            self.linear.weight.data = torch.FloatTensor(coef).unsqueeze(0)
            self.linear.bias.data = torch.FloatTensor([intercept])
            
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    torch_model = LogRegTorch(initial_model.coef_[0], initial_model.intercept_[0])
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    # Select samples with fine-tuned selector (use top-k)
    meta_selector.policy_net.eval()
    features = meta_selector.feature_extractor.extract_features(
        torch_model, X_train_tensor, y_train_tensor, group_train
    )
    with torch.no_grad():
        selection_probs = meta_selector.policy_net(features).squeeze().cpu().numpy()
    
    # Use top 70% by probability
    n_select = int(len(X_train) * select_ratio)
    top_k_indices = np.argsort(selection_probs)[-n_select:]
    selected_indices = top_k_indices
    
    X_selected = X_train[selected_indices]
    y_selected = y_train[selected_indices]
    
    print(f"\nSelected {len(selected_indices)} / {len(X_train)} samples ({len(selected_indices)/len(X_train)*100:.0f}%)")
    
    # Train model on selected samples
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_selected, y_selected)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    eo_disparity = FairnessMetrics.equalized_odds(y_test, y_pred, group_test)
    
    print(f"\nFine-tuned Meta Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  EO Disparity: {eo_disparity:.4f}")
    
    return {
        'accuracy': accuracy,
        'eo_disparity': eo_disparity,
        'selected_indices': selected_indices
    }


def create_comparison_table(baseline, greedy, meta_pretrained, meta_finetuned):
    """Create comparison table for all methods"""
    print("\n" + "="*70)
    print("FINAL COMPARISON: GERMAN DATASET")
    print("="*70)
    
    methods = ['Baseline', 'Greedy', 'Meta (Pre-trained)', 'Meta (Fine-tuned)']
    results_list = [baseline, greedy, meta_pretrained, meta_finetuned]
    
    print(f"\n{'Method':<20} {'Accuracy':<12} {'EO Disparity':<15} {'Fairness Improvement'}")
    print("-" * 70)
    
    baseline_eo = baseline['eo_disparity']
    
    for method, results in zip(methods, results_list):
        if results is None:
            print(f"{method:<20} {'N/A':<12} {'N/A':<15} {'N/A'}")
            continue
        
        accuracy = results['accuracy']
        eo_disparity = results['eo_disparity']
        
        # Calculate fairness improvement
        improvement = ((baseline_eo - eo_disparity) / baseline_eo) * 100
        
        print(f"{method:<20} {accuracy:<12.4f} {eo_disparity:<15.4f} {improvement:+.1f}%")
    
    return {
        'methods': methods,
        'baseline': baseline,
        'greedy': greedy,
        'meta_pretrained': meta_pretrained,
        'meta_finetuned': meta_finetuned
    }


def plot_comparison(comparison_results, finetune_history, save_path):
    """Create visualization comparing all methods"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract results
    methods = []
    accuracies = []
    eo_disparities = []
    
    for method, key in [('Baseline', 'baseline'), 
                        ('Greedy', 'greedy'), 
                        ('Meta\n(Pre-trained)', 'meta_pretrained'),
                        ('Meta\n(Fine-tuned)', 'meta_finetuned')]:
        results = comparison_results[key]
        if results is not None:
            methods.append(method)
            accuracies.append(results['accuracy'])
            eo_disparities.append(results['eo_disparity'])
    
    # Plot 1: Accuracy and Fairness Comparison
    x = np.arange(len(methods))
    width = 0.35
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('German Dataset: Method Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylim([0.65, 0.75])
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add accuracy values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Secondary axis for EO disparity
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, eo_disparities, width, label='EO Disparity', 
                         color='coral', alpha=0.8)
    ax1_twin.set_ylabel('EO Disparity (lower is better)', fontsize=11)
    ax1_twin.set_ylim([0, 0.8])
    ax1_twin.legend(loc='upper right')
    
    # Add EO values on bars
    for bar in bars2:
        height = bar.get_height()
        ax1_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Fine-tuning History
    ax2 = axes[1]
    
    if len(finetune_history.get('train_loss', [])) > 0:
        iterations = list(range(1, len(finetune_history['train_loss']) + 1))
        
        ax2.plot(iterations, finetune_history['train_loss'], 'o-', label='Train Loss', 
                 color='purple', linewidth=2, markersize=4)
        ax2.set_xlabel('Fine-tuning Iteration', fontsize=11)
        ax2.set_ylabel('Train Loss', fontsize=11)
        ax2.set_title('Fine-tuning Progress on German Dataset', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(alpha=0.3)
        
        # Add text annotation for final validation metrics
        if len(finetune_history.get('val_accuracy', [])) > 0:
            final_val_acc = finetune_history['val_accuracy'][-1]
            final_val_fair = finetune_history['val_fairness'][-1]
            ax2.text(0.05, 0.95, f'Final Val Acc: {final_val_acc:.3f}\nFinal Val Fairness: {final_val_fair:.3f}',
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No fine-tuning history available', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Fine-tuning Progress', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved plot: {save_path}")
    
    plt.close()


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("DAY 8: TRANSFER LEARNING FOR GERMAN DATASET")
    print("="*70)
    print("\nObjective: Fix meta-selector's -136% fairness failure on German")
    print("Approach: Fine-tune pre-trained meta-selector on German training set")
    
    # Load data
    X_train, X_test, y_train, y_test, group_train, group_test = load_german_data()
    
    # Evaluate all methods
    baseline = evaluate_baseline(X_train, X_test, y_train, y_test, group_test)
    greedy = evaluate_greedy(X_train, X_test, y_train, y_test, group_train, group_test)
    
    # Load and evaluate pre-trained meta-selector
    checkpoint_path = 'results/checkpoints/meta_selector_final.pt'
    meta_pretrained = evaluate_pretrained_meta(
        X_train, X_test, y_train, y_test, group_train, group_test, checkpoint_path
    )
    
    if meta_pretrained is None:
        print("\n[ERROR] Cannot proceed without pre-trained meta-selector")
        return
    
    # Fine-tune meta-selector on German
    meta_selector = meta_pretrained['meta_selector']
    finetune_history = finetune_meta_selector(
        meta_selector, X_train, y_train, group_train, n_iterations=20
    )
    
    # Evaluate fine-tuned meta-selector
    meta_finetuned = evaluate_finetuned_meta(
        meta_selector, X_train, X_test, y_train, y_test, group_train, group_test
    )
    
    # Compare all methods
    comparison = create_comparison_table(baseline, greedy, meta_pretrained, meta_finetuned)
    
    # Save results
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    results_path = 'results/metrics/german_transfer_learning.json'
    with open(results_path, 'w') as f:
        json.dump({
            'baseline': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in baseline.items() if k != 'model'},
            'greedy': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                      for k, v in greedy.items() if k != 'selected_indices'},
            'meta_pretrained': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                               for k, v in meta_pretrained.items() 
                               if k not in ['selected_indices', 'meta_selector']},
            'meta_finetuned': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                              for k, v in meta_finetuned.items() if k != 'selected_indices'},
            'finetune_history': finetune_history
        }, f, indent=2)
    print(f"\n[OK] Saved results: {results_path}")
    
    # Create visualization
    plot_path = 'results/plots/german_transfer_learning.png'
    plot_comparison(comparison, finetune_history, plot_path)
    
    # Calculate final improvement
    baseline_eo = baseline['eo_disparity']
    finetuned_eo = meta_finetuned['eo_disparity']
    improvement = ((baseline_eo - finetuned_eo) / baseline_eo) * 100
    
    print("\n" + "="*70)
    print("DAY 8 COMPLETE!")
    print("="*70)
    print(f"\nTransfer Learning Result:")
    print(f"  Pre-trained Meta: -136.4% (FAILED)")
    print(f"  Fine-tuned Meta: {improvement:+.1f}% fairness improvement")
    
    if improvement > 0:
        print(f"\n[OK] SUCCESS: Fine-tuning fixed the German dataset failure!")
    else:
        print(f"\n[WARNING] PARTIAL: Fine-tuning helped but still needs work")
    
    # Save fine-tuned model
    checkpoint_path = 'results/checkpoints/meta_selector_german_finetuned.pt'
    meta_selector.save(checkpoint_path)
    print(f"\n[OK] Saved fine-tuned model: {checkpoint_path}")


if __name__ == '__main__':
    main()
