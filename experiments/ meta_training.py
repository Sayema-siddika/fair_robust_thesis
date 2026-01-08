"""
Meta-Training Experiment
========================

Train the meta-selector on 80 synthetic tasks, validate on 20 tasks.
Then test on COMPAS dataset to verify transfer to real data.

Expected improvements over greedy selector:
- Better fairness (target: +50% vs greedy's +46%)
- Better accuracy (target: -4% vs greedy's -6%)
- Works on small datasets (fix German's -85% failure!)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from src.utils.synthetic_generator import SyntheticDataGenerator
from src.models.meta_selector import MetaSelector
from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


def load_synthetic_tasks(n_train=80, n_val=20):
    """
    Load synthetic tasks and split into train/val
    
    Args:
        n_train: Number of training tasks
        n_val: Number of validation tasks
    
    Returns:
        train_tasks: List of training tasks
        val_tasks: List of validation tasks
    """
    print("Loading synthetic tasks...")
    
    all_tasks = []
    for task_id in range(n_train + n_val):
        task = SyntheticDataGenerator.load_task(task_id)
        all_tasks.append(task)
    
    # Split into train/val
    train_tasks = all_tasks[:n_train]
    val_tasks = all_tasks[n_train:n_train+n_val]
    
    print(f"[OK] Loaded {len(train_tasks)} training tasks, {len(val_tasks)} validation tasks")
    
    return train_tasks, val_tasks


def test_on_compas(meta_selector, verbose=True):
    """
    Test trained meta-selector on COMPAS dataset
    
    Args:
        meta_selector: Trained MetaSelector instance
        verbose: Print results
    
    Returns:
        results: Dictionary with baseline, greedy, and meta results
    """
    if verbose:
        print("\n" + "="*80)
        print("Testing on COMPAS Dataset")
        print("="*80)
    
    # Load COMPAS
    loader = DataLoader('compas')
    X, y, z = loader.load_compas()
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        X, y, z, test_size=0.3, random_state=42, stratify=y
    )
    
    if verbose:
        print(f"\nCOMPAS: {len(X_train)} train, {len(X_test)} test samples")
    
    # 1. Baseline (all samples)
    if verbose:
        print("\n1. Baseline (no selection)...")
    
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)
    
    baseline_pred = baseline_model.predict(X_test)
    baseline_metrics = FairnessMetrics.compute_all_metrics(y_test, baseline_pred, z_test)
    
    if verbose:
        print(f"   Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"   EO Disparity: {baseline_metrics['eo_disparity']:.4f}")
    
    # 2. Greedy selector (simple loss-based selection)
    if verbose:
        print("\n2. Greedy Selector (loss-based)...")
    
    # Train initial model
    temp_model = LogisticRegression(max_iter=1000, random_state=42)
    temp_model.fit(X_train, y_train)
    
    # Get predictions
    train_probs = temp_model.predict_proba(X_train)[:, 1]
    
    # Compute losses
    from sklearn.metrics import log_loss
    losses = np.array([
        log_loss([y_train[i]], [train_probs[i]], labels=[0, 1])
        for i in range(len(y_train))
    ])
    
    # Select top 70% with lowest loss
    tau = 0.7
    n_select = int(len(X_train) * tau)
    selected_idx = np.argsort(losses)[:n_select]
    
    # Train on selected
    greedy_model = LogisticRegression(max_iter=1000, random_state=42)
    greedy_model.fit(X_train[selected_idx], y_train[selected_idx])
    
    greedy_pred = greedy_model.predict(X_test)
    greedy_metrics = FairnessMetrics.compute_all_metrics(y_test, greedy_pred, z_test)
    
    if verbose:
        print(f"   Selected: {len(selected_idx)}/{len(X_train)} samples ({len(selected_idx)/len(X_train):.1%})")
        print(f"   Accuracy: {greedy_metrics['accuracy']:.4f} ({greedy_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f})")
        print(f"   EO Disparity: {greedy_metrics['eo_disparity']:.4f} ({greedy_metrics['eo_disparity'] - baseline_metrics['eo_disparity']:+.4f})")
    
    # 3. Meta-Selector
    if verbose:
        print("\n3. Meta-Selector (trained)...")
    
    # Train initial model to get predictions
    temp_model = LogisticRegression(max_iter=1000, random_state=42)
    temp_model.fit(X_train, y_train)
    train_probs = temp_model.predict_proba(X_train)[:, 1]
    
    # Extract features manually (sklearn compatibility)
    from sklearn.metrics import log_loss
    
    losses = np.array([
        log_loss([y_train[i]], [train_probs[i]], labels=[0, 1])
        for i in range(len(y_train))
    ])
    
    confidence = np.maximum(train_probs, 1 - train_probs)
    
    eps = 1e-7
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
    
    # Get selection probabilities
    features_tensor = torch.FloatTensor(features)
    meta_selector.policy_net.eval()
    
    with torch.no_grad():
        selection_probs = meta_selector.policy_net(features_tensor).numpy().flatten()
    
    # Select top 70% (same as greedy)
    n_select = int(len(X_train) * 0.7)
    meta_selected_idx = np.argsort(selection_probs)[-n_select:]
    
    # Train on selected
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(X_train[meta_selected_idx], y_train[meta_selected_idx])
    
    meta_pred = meta_model.predict(X_test)
    meta_metrics = FairnessMetrics.compute_all_metrics(y_test, meta_pred, z_test)
    
    if verbose:
        print(f"   Selected: {len(meta_selected_idx)}/{len(X_train)} samples ({len(meta_selected_idx)/len(X_train):.1%})")
        print(f"   Accuracy: {meta_metrics['accuracy']:.4f} ({meta_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f})")
        print(f"   EO Disparity: {meta_metrics['eo_disparity']:.4f} ({meta_metrics['eo_disparity'] - baseline_metrics['eo_disparity']:+.4f})")
    
    # Comparison
    if verbose:
        print("\n" + "="*80)
        print("Comparison Summary")
        print("="*80)
        
        print(f"\n{'Method':<15} {'Accuracy':<12} {'EO Disparity':<15} {'Fairness Gain':<15}")
        print("-"*60)
        
        baseline_acc = baseline_metrics['accuracy']
        baseline_eo = baseline_metrics['eo_disparity']
        
        print(f"{'Baseline':<15} {baseline_acc:.4f}       {baseline_eo:.4f}          -")
        
        greedy_acc = greedy_metrics['accuracy']
        greedy_eo = greedy_metrics['eo_disparity']
        greedy_gain = (baseline_eo - greedy_eo) / baseline_eo * 100
        print(f"{'Greedy':<15} {greedy_acc:.4f}       {greedy_eo:.4f}          {greedy_gain:+.1f}%")
        
        meta_acc = meta_metrics['accuracy']
        meta_eo = meta_metrics['eo_disparity']
        meta_gain = (baseline_eo - meta_eo) / baseline_eo * 100
        print(f"{'Meta-Selector':<15} {meta_acc:.4f}       {meta_eo:.4f}          {meta_gain:+.1f}%")
        
        print("\n" + "="*80)
        print(f"Meta-Selector vs Greedy:")
        print(f"  Accuracy: {meta_acc - greedy_acc:+.4f}")
        print(f"  EO Disparity: {meta_eo - greedy_eo:+.4f}")
        print(f"  Fairness gain: {meta_gain - greedy_gain:+.1f} percentage points")
        print("="*80)
    
    return {
        'baseline': baseline_metrics,
        'greedy': greedy_metrics,
        'meta': meta_metrics
    }


def plot_training_history(history, save_path='results/plots/meta_training_history.png'):
    """
    Plot training history
    
    Args:
        history: Training history from meta_train()
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Meta-Training History', fontsize=16, fontweight='bold')
    
    # Training loss
    axes[0, 0].plot(history['train_loss'], linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(alpha=0.3)
    
    # Validation loss
    val_iterations = np.arange(10, len(history['train_loss']) + 1, 10)
    axes[0, 1].plot(val_iterations, history['val_loss'], linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].grid(alpha=0.3)
    
    # Validation accuracy
    axes[1, 0].plot(val_iterations, history['val_accuracy'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Validation Accuracy')
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].grid(alpha=0.3)
    
    # Validation fairness
    axes[1, 1].plot(val_iterations, history['val_fairness'], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Fairness (DP Gap)')
    axes[1, 1].set_title('Validation Fairness (lower is better)')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved training history plot to {save_path}")


def main():
    print("="*80)
    print("Meta-Training Experiment")
    print("="*80)
    
    # Load synthetic tasks
    train_tasks, val_tasks = load_synthetic_tasks(n_train=80, n_val=20)
    
    # Initialize meta-selector
    print("\nInitializing meta-selector...")
    meta_selector = MetaSelector(
        input_dim=10,
        hidden_dims=[64, 32],
        meta_lr=0.001,
        inner_lr=0.01
    )
    print("[OK] Meta-selector initialized")
    
    # Meta-train
    print("\n" + "="*80)
    print("Starting Meta-Training")
    print("="*80)
    
    history = meta_selector.meta_train(
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        n_iterations=100,
        save_dir='results/checkpoints',
        save_every=20,
        verbose=True
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Test on COMPAS
    results = test_on_compas(meta_selector, verbose=True)
    
    # Save final results
    import json
    results_dict = {
        'baseline': {k: float(v) for k, v in results['baseline'].items()},
        'greedy': {k: float(v) for k, v in results['greedy'].items()},
        'meta': {k: float(v) for k, v in results['meta'].items()},
        'training_history': {
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'final_val_accuracy': float(history['val_accuracy'][-1]),
            'final_val_fairness': float(history['val_fairness'][-1])
        }
    }
    
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/meta_training_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n[OK] Saved results to results/metrics/meta_training_results.json")
    print("\n" + "="*80)
    print("Meta-Training Experiment Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
