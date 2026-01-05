"""
Experiment 3: Multi-Dataset Comparison
Run baseline and greedy selector on COMPAS, Adult, and German datasets
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd

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


def train_model(data, selector=None, epochs=100, learning_rate=0.01, verbose=False):
    """
    Train model with optional sample selection
    
    Returns:
        metrics: Final test metrics
    """
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train_noisy']).unsqueeze(1)
    X_test = torch.FloatTensor(data['X_test'])
    
    model = LogisticRegression(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if selector is not None:
        criterion = nn.BCELoss(reduction='none')
    else:
        criterion = nn.BCELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        if selector is not None:
            # Sample selection
            selection_mask, weights = selector.select_samples(
                model, X_train, y_train.squeeze(), data['z_train']
            )
            
            selection_mask_t = torch.BoolTensor(selection_mask)
            weights_t = torch.FloatTensor(weights)
            
            X_selected = X_train[selection_mask_t]
            y_selected = y_train[selection_mask_t]
            weights_selected = weights_t[selection_mask_t]
            
            optimizer.zero_grad()
            y_pred = model(X_selected)
            losses = criterion(y_pred, y_selected)
            loss = (losses.squeeze() * weights_selected).mean()
        else:
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
        
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        y_pred_test = (y_pred_test.numpy().squeeze() > 0.5).astype(int)
    
    metrics = FairnessMetrics.compute_all_metrics(
        data['y_test'], y_pred_test, data['z_test'], verbose=False
    )
    
    return metrics


def run_dataset_experiments(dataset_name, noise_rate=0.1, epochs=100):
    """
    Run baseline and greedy selector on a dataset
    
    Returns:
        results: Dictionary with baseline and greedy metrics
    """
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    print(f"\n[1/3] Loading data...")
    try:
        loader = DataLoader(dataset_name=dataset_name, data_dir="data/raw")
        data = loader.load_and_prepare(
            noise_rate=noise_rate,
            noise_type='random',
            test_size=0.3,
            seed=42
        )
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        return None
    
    # Baseline
    print(f"\n[2/3] Training baseline...")
    baseline_metrics = train_model(data, selector=None, epochs=epochs, verbose=False)
    print(f"  ✓ Accuracy: {baseline_metrics['accuracy']:.4f}, "
          f"EO: {baseline_metrics['eo_disparity']:.4f}")
    
    # Greedy selector
    print(f"\n[3/3] Training with greedy selector...")
    selector = GreedySelector(tau=0.7, lambda_init=1.5)
    greedy_metrics = train_model(data, selector=selector, epochs=epochs, verbose=False)
    print(f"  ✓ Accuracy: {greedy_metrics['accuracy']:.4f}, "
          f"EO: {greedy_metrics['eo_disparity']:.4f}")
    
    # Calculate improvements
    acc_improvement = (greedy_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    eo_improvement = ((baseline_metrics['eo_disparity'] - greedy_metrics['eo_disparity']) / 
                      baseline_metrics['eo_disparity'] * 100)
    
    results = {
        'dataset': dataset_name,
        'baseline_acc': baseline_metrics['accuracy'],
        'baseline_dp': baseline_metrics['dp_disparity'],
        'baseline_eo': baseline_metrics['eo_disparity'],
        'greedy_acc': greedy_metrics['accuracy'],
        'greedy_dp': greedy_metrics['dp_disparity'],
        'greedy_eo': greedy_metrics['eo_disparity'],
        'acc_improvement': acc_improvement,
        'eo_improvement': eo_improvement
    }
    
    print(f"\n  → Accuracy change: {acc_improvement:+.2f}%")
    print(f"  → Fairness improvement: {eo_improvement:+.1f}%")
    
    return results


def main():
    """Run experiments on all three datasets"""
    print("="*70)
    print(" " * 10 + "MULTI-DATASET COMPARISON EXPERIMENT")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Datasets: COMPAS, Adult, German")
    print(f"  Noise rate: 10%")
    print(f"  Epochs: 100")
    print(f"  Greedy tau: 0.7")
    print(f"  Greedy lambda: 1.5")
    
    # Run experiments on all datasets
    datasets = ['compas', 'adult', 'german']
    all_results = []
    
    for dataset in datasets:
        results = run_dataset_experiments(dataset, noise_rate=0.1, epochs=100)
        if results:
            all_results.append(results)
    
    # Create comparison table
    print(f"\n{'='*70}")
    print(" " * 20 + "SUMMARY TABLE")
    print(f"{'='*70}\n")
    
    # Print table header
    print(f"{'Dataset':<10} {'Method':<10} {'Accuracy':<10} {'DP Disp':<10} {'EO Disp':<10}")
    print(f"{'-'*60}")
    
    for result in all_results:
        dataset = result['dataset'].upper()
        
        # Baseline row
        print(f"{dataset:<10} {'Baseline':<10} "
              f"{result['baseline_acc']:<10.4f} "
              f"{result['baseline_dp']:<10.4f} "
              f"{result['baseline_eo']:<10.4f}")
        
        # Greedy row
        print(f"{'':<10} {'Greedy':<10} "
              f"{result['greedy_acc']:<10.4f} "
              f"{result['greedy_dp']:<10.4f} "
              f"{result['greedy_eo']:<10.4f}")
        
        # Improvement row
        print(f"{'':<10} {'Improve':<10} "
              f"{result['acc_improvement']:+.2f}%     "
              f"{result['eo_improvement']:+.1f}%      ")
        print()
    
    # Save results
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(all_results)
    csv_file = results_dir / "multi_dataset_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"💾 Results saved to: {csv_file}")
    
    # Save as text table
    txt_file = results_dir / "multi_dataset_results.txt"
    with open(txt_file, 'w') as f:
        f.write("MULTI-DATASET COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Noise rate: 10%\n")
        f.write(f"  Epochs: 100\n")
        f.write(f"  Greedy tau: 0.7\n")
        f.write(f"  Greedy lambda: 1.5\n\n")
        
        f.write(f"{'Dataset':<10} {'Method':<10} {'Accuracy':<10} {'DP Disp':<10} {'EO Disp':<10}\n")
        f.write(f"{'-'*60}\n")
        
        for result in all_results:
            dataset = result['dataset'].upper()
            f.write(f"{dataset:<10} {'Baseline':<10} "
                   f"{result['baseline_acc']:<10.4f} "
                   f"{result['baseline_dp']:<10.4f} "
                   f"{result['baseline_eo']:<10.4f}\n")
            f.write(f"{'':<10} {'Greedy':<10} "
                   f"{result['greedy_acc']:<10.4f} "
                   f"{result['greedy_dp']:<10.4f} "
                   f"{result['greedy_eo']:<10.4f}\n")
            f.write(f"{'':<10} {'Improve':<10} "
                   f"{result['acc_improvement']:+.2f}%     "
                   f"{result['eo_improvement']:+.1f}%      \n\n")
    
    print(f"💾 Results saved to: {txt_file}")
    
    # Analysis
    avg_eo_improvement = np.mean([r['eo_improvement'] for r in all_results])
    avg_acc_change = np.mean([r['acc_improvement'] for r in all_results])
    
    print(f"\n{'='*70}")
    print(" " * 25 + "ANALYSIS")
    print(f"{'='*70}")
    print(f"\nAverage across datasets:")
    print(f"  Fairness improvement (EO): {avg_eo_improvement:+.1f}%")
    print(f"  Accuracy change: {avg_acc_change:+.2f}%")
    
    if avg_eo_improvement > 20:
        print(f"\n  ✓ EXCELLENT: Greedy selector significantly improves fairness!")
    elif avg_eo_improvement > 10:
        print(f"\n  ✓ GOOD: Greedy selector moderately improves fairness")
    else:
        print(f"\n  ⚠ MARGINAL: Greedy selector shows limited fairness improvement")
    
    print(f"\n{'='*70}")
    print(" " * 15 + "MULTI-DATASET EXPERIMENT COMPLETE!")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == "__main__":
    results = main()
