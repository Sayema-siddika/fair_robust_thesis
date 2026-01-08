"""
Test Meta-Selector on Synthetic Tasks
======================================

Quick test to verify meta-selector can train on synthetic tasks.
Tests on 3 tasks with different characteristics:
- Task 0: Small dataset (125 samples)
- Task 50: Medium dataset (~3000 samples)
- Task 99: Large dataset (6980 samples)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.utils.synthetic_generator import SyntheticDataGenerator
from src.models.meta_selector import MetaSelector

def test_on_task(task_id, meta_selector):
    """
    Test meta-selector on a single synthetic task.
    
    Args:
        task_id (int): Task ID to test on
        meta_selector (MetaSelector): Meta-selector instance
    
    Returns:
        dict: Test results
    """
    print(f"\n{'=' * 80}")
    print(f"Testing on Task {task_id}")
    print(f"{'=' * 80}")
    
    # Load task
    task = SyntheticDataGenerator.load_task(task_id)
    
    X_train = task['train']['X']
    y_train = task['train']['y']
    z_train = task['train']['z']
    y_clean = task['train']['y_clean']
    
    X_test = task['test']['X']
    y_test = task['test']['y']
    z_test = task['test']['z']
    
    metadata = task['metadata']
    
    print(f"\nTask Characteristics:")
    print(f"  Training samples: {metadata['n_train']}")
    print(f"  Test samples: {metadata['n_test']}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Noise rate: {metadata['noise_rate']:.2%}")
    print(f"  Actual noise: {metadata['train_actual_noise_rate']:.2%}")
    print(f"  Minority rate: {metadata['train_minority_rate']:.2%}")
    print(f"  Positive rate: {metadata['train_pos_rate']:.2%}")
    print(f"  Train DP gap: {metadata['train_dp_gap']:.3f}")
    
    # 1. Baseline: Train on all samples
    print(f"\n1. Baseline (all samples)...")
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)
    
    baseline_acc = baseline_model.score(X_test, y_test)
    baseline_pred = baseline_model.predict(X_test)
    
    # Compute fairness (demographic parity)
    pred_rate_z0 = np.mean(baseline_pred[z_test == 0]) if np.sum(z_test == 0) > 0 else 0
    pred_rate_z1 = np.mean(baseline_pred[z_test == 1]) if np.sum(z_test == 1) > 0 else 0
    baseline_dp = abs(pred_rate_z0 - pred_rate_z1)
    
    print(f"   Accuracy: {baseline_acc:.4f}")
    print(f"   DP gap: {baseline_dp:.4f}")
    
    # 2. Meta-Selector: Select samples using policy network
    print(f"\n2. Meta-Selector (sample selection)...")
    
    # Train a simple model first to get predictions
    temp_model = LogisticRegression(max_iter=1000, random_state=42)
    temp_model.fit(X_train, y_train)
    train_probs = temp_model.predict_proba(X_train)[:, 1]
    
    # Extract features manually (simplified version for sklearn models)
    from sklearn.metrics import log_loss
    
    # Compute per-sample losses
    losses = np.array([
        log_loss([y_train[i]], [train_probs[i]], labels=[0, 1])
        for i in range(len(y_train))
    ])
    
    # Confidence
    confidence = np.maximum(train_probs, 1 - train_probs)
    
    # Entropy
    eps = 1e-7
    entropy = -(train_probs * np.log(train_probs + eps) +
                (1 - train_probs) * np.log(1 - train_probs + eps))
    
    # Group statistics
    group_loss = np.array([
        np.mean(losses[z_train == z_train[i]]) 
        for i in range(len(z_train))
    ])
    
    group_confidence = np.array([
        np.mean(confidence[z_train == z_train[i]])
        for i in range(len(z_train))
    ])
    
    # Predictions and margin
    predictions = (train_probs > 0.5).astype(float)
    margin = np.abs(train_probs - 0.5)
    
    # Difficulty (normalized loss rank)
    difficulty = np.argsort(np.argsort(losses)) / len(losses)
    
    # Stack features: [loss, confidence, entropy, group, group_loss, group_confidence, prediction, label, margin, difficulty]
    features = np.column_stack([
        losses,
        confidence,
        entropy,
        z_train,
        group_loss,
        group_confidence,
        predictions,
        y_train,
        margin,
        difficulty
    ])
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features)
    
    # Get selection probabilities from policy network
    with torch.no_grad():
        selection_probs = meta_selector.policy_net(features_tensor).numpy().flatten()
    
    # Select top samples (similar to greedy selector: top 70%)
    n_select = int(len(X_train) * 0.7)
    selected_idx = np.argsort(selection_probs)[-n_select:]
    
    # Train on selected samples
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(X_train[selected_idx], y_train[selected_idx])
    
    meta_acc = meta_model.score(X_test, y_test)
    meta_pred = meta_model.predict(X_test)
    
    pred_rate_z0 = np.mean(meta_pred[z_test == 0]) if np.sum(z_test == 0) > 0 else 0
    pred_rate_z1 = np.mean(meta_pred[z_test == 1]) if np.sum(z_test == 1) > 0 else 0
    meta_dp = abs(pred_rate_z0 - pred_rate_z1)
    
    print(f"   Selected: {n_select}/{len(X_train)} samples ({n_select/len(X_train):.1%})")
    print(f"   Accuracy: {meta_acc:.4f} ({meta_acc - baseline_acc:+.4f})")
    print(f"   DP gap: {meta_dp:.4f} ({meta_dp - baseline_dp:+.4f})")
    
    # 3. Compare selection quality
    print(f"\n3. Selection Quality Analysis...")
    
    # Compute noise rate in selected samples
    selected_noise = np.mean(y_train[selected_idx] != y_clean[selected_idx])
    overall_noise = np.mean(y_train != y_clean)
    
    print(f"   Overall noise: {overall_noise:.2%}")
    print(f"   Selected noise: {selected_noise:.2%}")
    print(f"   Noise reduction: {(overall_noise - selected_noise) / overall_noise * 100:+.1f}%")
    
    # Group balance in selection
    selected_minority_rate = np.mean(z_train[selected_idx])
    overall_minority_rate = np.mean(z_train)
    
    print(f"   Overall minority: {overall_minority_rate:.2%}")
    print(f"   Selected minority: {selected_minority_rate:.2%}")
    print(f"   Balance change: {(selected_minority_rate - overall_minority_rate) / overall_minority_rate * 100:+.1f}%")
    
    return {
        'task_id': task_id,
        'n_train': metadata['n_train'],
        'baseline_acc': baseline_acc,
        'baseline_dp': baseline_dp,
        'meta_acc': meta_acc,
        'meta_dp': meta_dp,
        'noise_reduction': (overall_noise - selected_noise) / overall_noise if overall_noise > 0 else 0
    }


def main():
    print("=" * 80)
    print("Testing Meta-Selector on Synthetic Tasks")
    print("=" * 80)
    
    # Initialize meta-selector (untrained)
    print("\nInitializing meta-selector...")
    meta_selector = MetaSelector(
        input_dim=10,  # 10 meta-features
        hidden_dims=[64, 32],  # Hidden layer dimensions
        meta_lr=0.001,
        inner_lr=0.01
    )
    print("✓ Meta-selector initialized (untrained)")
    
    # Test on 3 tasks with different sizes
    test_tasks = [0, 50, 99]  # Small, medium, large
    results = []
    
    for task_id in test_tasks:
        result = test_on_task(task_id, meta_selector)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary: Meta-Selector on Synthetic Tasks (Untrained)")
    print("=" * 80)
    
    print(f"\n{'Task':<10} {'Samples':<10} {'Baseline Acc':<15} {'Meta Acc':<15} {'Acc Change':<15}")
    print("-" * 70)
    
    for r in results:
        acc_change = r['meta_acc'] - r['baseline_acc']
        print(f"{r['task_id']:<10} {r['n_train']:<10} {r['baseline_acc']:.4f}          {r['meta_acc']:.4f}          {acc_change:+.4f}")
    
    print(f"\n{'Task':<10} {'Baseline DP':<15} {'Meta DP':<15} {'DP Change':<15} {'Noise Red.':<15}")
    print("-" * 70)
    
    for r in results:
        dp_change = r['meta_dp'] - r['baseline_dp']
        print(f"{r['task_id']:<10} {r['baseline_dp']:.4f}          {r['meta_dp']:.4f}          {dp_change:+.4f}          {r['noise_reduction']:.2%}")
    
    print("\n" + "=" * 80)
    print("Notes:")
    print("=" * 80)
    print("- Meta-selector is UNTRAINED (random initialization)")
    print("- Selection is based on untrained policy network")
    print("- After meta-training (Day 6), expect significant improvements:")
    print("  * Better noise detection → higher noise reduction")
    print("  * Better fairness-aware selection → lower DP gaps")
    print("  * Adaptive selection → better accuracy-fairness trade-offs")
    print("\n✓ Synthetic task testing complete!")
    print("✓ Ready for meta-training (Day 6)!")


if __name__ == '__main__':
    main()
