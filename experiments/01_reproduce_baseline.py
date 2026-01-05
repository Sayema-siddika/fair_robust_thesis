"""
Baseline Experiment: Simple Logistic Regression
Reproduce baseline results without fairness or robustness interventions
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


class LogisticRegression(nn.Module):
    """Simple logistic regression classifier"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_baseline(data, epochs=100, learning_rate=0.01, verbose=True):
    """
    Train baseline logistic regression model
    
    Args:
        data: Dictionary from DataLoader.load_and_prepare()
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        verbose: Print training progress
        
    Returns:
        metrics: Dictionary of evaluation metrics
        model: Trained model
    """
    # Convert to tensors
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train_noisy']).unsqueeze(1)
    X_test = torch.FloatTensor(data['X_test'])
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = LogisticRegression(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # Training loop
    if verbose:
        print(f"\nTraining baseline logistic regression...")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Training samples: {len(X_train)}")
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Print progress
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        y_pred_test = (y_pred_test.numpy().squeeze() > 0.5).astype(int)
    
    # Compute metrics
    metrics = FairnessMetrics.compute_all_metrics(
        data['y_test'], 
        y_pred_test, 
        data['z_test'],
        verbose=verbose
    )
    
    return metrics, model


def main():
    """Main experiment function"""
    print("="*70)
    print(" " * 10 + "BASELINE EXPERIMENT: LOGISTIC REGRESSION")
    print("="*70)
    
    # Configuration
    DATASET = "compas"
    NOISE_RATE = 0.1
    NOISE_TYPE = "random"
    TEST_SIZE = 0.3
    EPOCHS = 100
    LEARNING_RATE = 0.01
    SEED = 42
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Noise rate: {NOISE_RATE:.1%}")
    print(f"  Noise type: {NOISE_TYPE}")
    print(f"  Test size: {TEST_SIZE:.1%}")
    print(f"  Random seed: {SEED}")
    
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
        print("\n" + "="*70)
        print("TO DOWNLOAD COMPAS DATASET:")
        print("="*70)
        print("1. Visit: https://github.com/propublica/compas-analysis")
        print("2. Download file: compas-scores-two-years.csv")
        print("3. Save to: data/raw/compas/compas-scores-two-years.csv")
        print("\nOR use this direct link:")
        print("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
        print("="*70)
        return
    
    # Train model
    print(f"\n{'─'*70}")
    print("STEP 2: Training Model")
    print(f"{'─'*70}")
    
    metrics, model = train_baseline(
        data, 
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE,
        verbose=True
    )
    
    # Print final results
    print(f"\n{'═'*70}")
    print(" " * 20 + "FINAL RESULTS")
    print(f"{'═'*70}")
    print(f"\n📊 Performance Metrics:")
    print(f"  ✓ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print(f"\n⚖️  Fairness Metrics (lower is better):")
    print(f"  • Demographic Parity Disparity: {metrics['dp_disparity']:.4f}")
    print(f"  • Equalized Odds Disparity: {metrics['eo_disparity']:.4f}")
    print(f"  • Equal Opportunity Disparity: {metrics['eop_disparity']:.4f}")
    
    print(f"\n{'─'*70}")
    print("Interpretation:")
    print(f"{'─'*70}")
    
    if metrics['eo_disparity'] < 0.05:
        fairness_status = "✓ GOOD - Model is relatively fair"
    elif metrics['eo_disparity'] < 0.10:
        fairness_status = "⚠ MODERATE - Some fairness concerns"
    else:
        fairness_status = "✗ POOR - Significant fairness issues"
    
    print(f"  {fairness_status}")
    print(f"  Equalized Odds is the primary fairness metric")
    print(f"  Target: < 0.05 (excellent), < 0.10 (acceptable)")
    
    # Save results
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "baseline_results.txt"
    with open(results_file, 'w') as f:
        f.write("BASELINE EXPERIMENT RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Dataset: {DATASET}\n")
        f.write(f"  Noise rate: {NOISE_RATE:.1%}\n")
        f.write(f"  Noise type: {NOISE_TYPE}\n")
        f.write(f"  Epochs: {EPOCHS}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  DP Disparity: {metrics['dp_disparity']:.4f}\n")
        f.write(f"  EO Disparity: {metrics['eo_disparity']:.4f}\n")
        f.write(f"  EOP Disparity: {metrics['eop_disparity']:.4f}\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    print(f"\n{'═'*70}")
    print(" " * 15 + "BASELINE EXPERIMENT COMPLETE!")
    print(f"{'═'*70}\n")
    
    return metrics, model


if __name__ == "__main__":
    metrics, model = main()
