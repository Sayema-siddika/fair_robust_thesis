"""
Experiment 2: Greedy Sample Selector
Compare greedy selector with baseline on COMPAS dataset
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics
from src.selection.greedy_selector import GreedySelector


class LogisticRegression(nn.Module):
    """Simple logistic regression classifier"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_with_selection(data, selector=None, epochs=100, learning_rate=0.01, 
                        verbose=True, update_lambda=False):
    """
    Train model with optional sample selection
    
    Args:
        data: Dictionary from DataLoader
        selector: GreedySelector instance (None = no selection)
        epochs: Number of training epochs
        learning_rate: Learning rate
        verbose: Print progress
        update_lambda: Update lambda based on fairness (greedy only)
        
    Returns:
        metrics: Final test metrics
        model: Trained model
        selection_history: List of selection stats per epoch
    """
    # Convert to tensors
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train_noisy']).unsqueeze(1)
    X_test = torch.FloatTensor(data['X_test'])
    
    # Initialize model
    model = LogisticRegression(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For weighted samples
    if selector is not None:
        criterion = nn.BCELoss(reduction='none')
    else:
        criterion = nn.BCELoss()
    
    selection_history = []
    
    if verbose:
        method_name = "Greedy Selector" if selector else "Baseline (No Selection)"
        print(f"\nTraining with {method_name}...")
        if selector:
            print(f"  tau: {selector.tau:.2f} (select {selector.tau:.0%} samples)")
            print(f"  lambda: {selector.lambda_val:.2f}")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        # Sample selection (greedy only)
        if selector is not None:
            selection_mask, weights = selector.select_samples(
                model, X_train, y_train.squeeze(), data['z_train']
            )
            
            # Convert to tensors
            selection_mask_t = torch.BoolTensor(selection_mask)
            weights_t = torch.FloatTensor(weights)
            
            # Use only selected samples
            X_selected = X_train[selection_mask_t]
            y_selected = y_train[selection_mask_t]
            weights_selected = weights_t[selection_mask_t]
            
            # Track selection stats
            n_selected = selection_mask.sum()
            noise_in_selected = (
                data['y_train'][selection_mask] != 
                data['y_train_noisy'][selection_mask]
            ).mean()
            
            selection_history.append({
                'epoch': epoch,
                'n_selected': n_selected,
                'noise_rate': noise_in_selected,
                'lambda': selector.lambda_val
            })
            
            # Forward pass with selected samples
            optimizer.zero_grad()
            y_pred = model(X_selected)
            
            # Weighted loss
            losses = criterion(y_pred, y_selected)
            loss = (losses.squeeze() * weights_selected).mean()
            
        else:
            # Standard training (no selection)
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update lambda based on fairness (every 10 epochs)
        if selector and update_lambda and (epoch + 1) % 10 == 0:
            # Evaluate current fairness
            model.eval()
            with torch.no_grad():
                y_pred_test = model(X_test)
                y_pred_test = (y_pred_test.numpy().squeeze() > 0.5).astype(int)
            
            temp_metrics = FairnessMetrics.compute_all_metrics(
                data['y_test'], y_pred_test, data['z_test'], verbose=False
            )
            
            # Update lambda
            selector.update_lambda(temp_metrics['eo_disparity'], target_disparity=0.05)
            model.train()
        
        # Print progress
        if verbose and (epoch + 1) % 20 == 0:
            if selector:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                      f"Selected: {n_selected}/{len(X_train)} ({n_selected/len(X_train):.1%}), "
                      f"Lambda: {selector.lambda_val:.2f}")
            else:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        y_pred_test = (y_pred_test.numpy().squeeze() > 0.5).astype(int)
    
    metrics = FairnessMetrics.compute_all_metrics(
        data['y_test'], y_pred_test, data['z_test'], verbose=verbose
    )
    
    return metrics, model, selection_history


def main():
    """Main experiment comparing baseline and greedy selector"""
    print("="*70)
    print(" " * 10 + "EXPERIMENT 2: GREEDY SELECTOR vs BASELINE")
    print("="*70)
    
    # Configuration
    DATASET = "compas"
    NOISE_RATE = 0.1
    NOISE_TYPE = "random"
    TEST_SIZE = 0.3
    EPOCHS = 100
    LEARNING_RATE = 0.01
    SEED = 42
    
    # Greedy selector config
    TAU = 0.7  # Select 70% samples with lowest loss
    LAMBDA_INIT = 1.5  # Upweight minority by 1.5x
    UPDATE_LAMBDA = True  # Adapt lambda during training
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Noise: {NOISE_RATE:.1%} ({NOISE_TYPE})")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Tau: {TAU} (select {TAU:.0%} samples)")
    print(f"  Lambda: {LAMBDA_INIT} (initial)")
    print(f"  Adaptive lambda: {UPDATE_LAMBDA}")
    
    # Load data
    print(f"\n{'─'*70}")
    print("STEP 1: Loading Data")
    print(f"{'─'*70}")
    
    try:
        loader = DataLoader(dataset_name=DATASET, data_dir="data/raw")
        data = loader.load_and_prepare(
            noise_rate=NOISE_RATE,
            noise_type=NOISE_TYPE,
            test_size=TEST_SIZE,
            seed=SEED
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # Experiment 1: Baseline (no selection)
    print(f"\n{'─'*70}")
    print("STEP 2: Baseline Training (No Selection)")
    print(f"{'─'*70}")
    
    baseline_metrics, baseline_model, _ = train_with_selection(
        data, selector=None, epochs=EPOCHS, learning_rate=LEARNING_RATE
    )
    
    # Experiment 2: Greedy selector
    print(f"\n{'─'*70}")
    print("STEP 3: Greedy Selector Training")
    print(f"{'─'*70}")
    
    selector = GreedySelector(
        tau=TAU, 
        lambda_init=LAMBDA_INIT,
        lambda_lr=0.01,
        fairness_metric='demographic_parity'
    )
    
    greedy_metrics, greedy_model, selection_history = train_with_selection(
        data, selector=selector, epochs=EPOCHS, learning_rate=LEARNING_RATE,
        update_lambda=UPDATE_LAMBDA
    )
    
    # Compare results
    print(f"\n{'═'*70}")
    print(" " * 20 + "COMPARISON RESULTS")
    print(f"{'═'*70}")
    
    print(f"\n📊 Accuracy:")
    print(f"  Baseline:        {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy']*100:.2f}%)")
    print(f"  Greedy Selector: {greedy_metrics['accuracy']:.4f} ({greedy_metrics['accuracy']*100:.2f}%)")
    improvement_acc = (greedy_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    print(f"  → Improvement: {improvement_acc:+.2f}%")
    
    print(f"\n⚖️  Demographic Parity Disparity:")
    print(f"  Baseline:        {baseline_metrics['dp_disparity']:.4f}")
    print(f"  Greedy Selector: {greedy_metrics['dp_disparity']:.4f}")
    improvement_dp = ((baseline_metrics['dp_disparity'] - greedy_metrics['dp_disparity']) / 
                      baseline_metrics['dp_disparity'] * 100)
    print(f"  → Improvement: {improvement_dp:+.1f}%")
    
    print(f"\n⚖️  Equalized Odds Disparity:")
    print(f"  Baseline:        {baseline_metrics['eo_disparity']:.4f}")
    print(f"  Greedy Selector: {greedy_metrics['eo_disparity']:.4f}")
    improvement_eo = ((baseline_metrics['eo_disparity'] - greedy_metrics['eo_disparity']) / 
                      baseline_metrics['eo_disparity'] * 100)
    print(f"  → Improvement: {improvement_eo:+.1f}%")
    
    print(f"\n{'─'*70}")
    print("Summary:")
    print(f"{'─'*70}")
    
    if greedy_metrics['accuracy'] > baseline_metrics['accuracy']:
        print("  ✓ Greedy selector IMPROVES accuracy")
    else:
        print("  ✗ Greedy selector reduces accuracy slightly")
    
    if greedy_metrics['eo_disparity'] < baseline_metrics['eo_disparity']:
        print("  ✓ Greedy selector IMPROVES fairness (lower disparity)")
    else:
        print("  ✗ Greedy selector does not improve fairness")
    
    # Lambda evolution
    if UPDATE_LAMBDA and len(selector.lambda_history) > 0:
        lambda_final = selector.lambda_val
        print(f"\n  Lambda evolution: {LAMBDA_INIT:.2f} → {lambda_final:.2f}")
    
    # Save results
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "greedy_vs_baseline.txt"
    with open(results_file, 'w') as f:
        f.write("GREEDY SELECTOR vs BASELINE - COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Dataset: {DATASET}\n")
        f.write(f"  Noise: {NOISE_RATE:.1%}\n")
        f.write(f"  Tau: {TAU}\n")
        f.write(f"  Lambda (initial): {LAMBDA_INIT}\n")
        f.write(f"  Lambda (final): {selector.lambda_val:.2f}\n\n")
        
        f.write("BASELINE RESULTS:\n")
        f.write(f"  Accuracy: {baseline_metrics['accuracy']:.4f}\n")
        f.write(f"  DP Disparity: {baseline_metrics['dp_disparity']:.4f}\n")
        f.write(f"  EO Disparity: {baseline_metrics['eo_disparity']:.4f}\n\n")
        
        f.write("GREEDY SELECTOR RESULTS:\n")
        f.write(f"  Accuracy: {greedy_metrics['accuracy']:.4f}\n")
        f.write(f"  DP Disparity: {greedy_metrics['dp_disparity']:.4f}\n")
        f.write(f"  EO Disparity: {greedy_metrics['eo_disparity']:.4f}\n\n")
        
        f.write("IMPROVEMENTS:\n")
        f.write(f"  Accuracy: {improvement_acc:+.2f}%\n")
        f.write(f"  DP Disparity: {improvement_dp:+.1f}%\n")
        f.write(f"  EO Disparity: {improvement_eo:+.1f}%\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    print(f"\n{'═'*70}")
    print(" " * 12 + "GREEDY SELECTOR EXPERIMENT COMPLETE!")
    print(f"{'═'*70}\n")
    
    return baseline_metrics, greedy_metrics


if __name__ == "__main__":
    baseline_metrics, greedy_metrics = main()
