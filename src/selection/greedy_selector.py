"""
Greedy Sample Selector
Based on Roh et al. (NeurIPS 2021) "Sample Selection for Fair and Robust Training"

Algorithm: Select samples with lowest loss (assumed to be clean) and apply
fairness-aware weighting to balance demographic groups.
"""

import numpy as np
import torch
import torch.nn as nn


class GreedySelector:
    """
    Greedy sample selection for fair and robust training
    
    Core Algorithm:
    1. Compute loss for each sample
    2. Sort samples by loss (ascending)
    3. Select top tau% samples (lowest loss = likely clean)
    4. Apply fairness weighting using lambda ratios
    """
    
    def __init__(self, tau=0.7, lambda_init=1.0, lambda_lr=0.01, 
                 fairness_metric='demographic_parity'):
        """
        Initialize greedy selector
        
        Args:
            tau: Clean sample ratio (0.0 to 1.0)
                 tau=0.7 means select 70% samples with lowest loss
            lambda_init: Initial fairness weight (lambda in paper)
            lambda_lr: Learning rate for lambda updates
            fairness_metric: 'demographic_parity' or 'equalized_odds'
        """
        self.tau = tau
        self.lambda_val = lambda_init
        self.lambda_lr = lambda_lr
        self.fairness_metric = fairness_metric
        
        # Track lambda history
        self.lambda_history = []
    
    def select_samples(self, model, X, y, z, criterion=None):
        """
        Select clean samples based on loss and apply fairness weighting
        
        Args:
            model: Trained model (PyTorch nn.Module)
            X: Training features (torch.Tensor)
            y: Training labels (torch.Tensor)
            z: Sensitive attributes (torch.Tensor or numpy.ndarray)
            criterion: Loss function (default: BCELoss)
            
        Returns:
            selection_mask: Boolean mask of selected samples
            sample_weights: Weights for selected samples (fairness-aware)
        """
        if criterion is None:
            criterion = nn.BCELoss(reduction='none')
        
        # Convert to numpy if needed
        if isinstance(z, torch.Tensor):
            z_np = z.cpu().numpy()
        else:
            z_np = z
        
        # Compute per-sample losses
        model.eval()
        with torch.no_grad():
            y_pred = model(X)
            losses = criterion(y_pred, y.unsqueeze(1) if y.dim() == 1 else y)
            losses = losses.squeeze().cpu().numpy()
        
        # Greedy selection: choose tau% samples with lowest loss
        n_samples = len(losses)
        n_select = int(n_samples * self.tau)
        
        # Sort by loss (ascending)
        sorted_indices = np.argsort(losses)
        selected_indices = sorted_indices[:n_select]
        
        # Create selection mask
        selection_mask = np.zeros(n_samples, dtype=bool)
        selection_mask[selected_indices] = True
        
        # Apply fairness-aware weighting
        sample_weights = self._compute_fairness_weights(
            selection_mask, z_np, y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        )
        
        return selection_mask, sample_weights
    
    def _compute_fairness_weights(self, selection_mask, z, y):
        """
        Compute sample weights to promote fairness
        
        Weighting scheme:
        - Upweight minority group (z=1) by factor lambda
        - Keep majority group (z=0) at weight 1.0
        
        Args:
            selection_mask: Boolean mask of selected samples
            z: Sensitive attributes
            y: Labels (for equalized odds weighting)
            
        Returns:
            weights: Sample weights (n_samples,)
        """
        n_samples = len(selection_mask)
        weights = np.zeros(n_samples)
        
        # Only compute weights for selected samples
        selected_indices = np.where(selection_mask)[0]
        
        if len(selected_indices) == 0:
            return weights
        
        # Get selected z values
        z_selected = z[selected_indices]
        
        # Count samples per group
        n_z0 = (z_selected == 0).sum()
        n_z1 = (z_selected == 1).sum()
        
        if n_z0 == 0 or n_z1 == 0:
            # If only one group, use uniform weights
            weights[selected_indices] = 1.0
            return weights
        
        # Demographic parity weighting
        if self.fairness_metric == 'demographic_parity':
            # Upweight minority to balance positive prediction rates
            for idx in selected_indices:
                if z[idx] == 0:
                    weights[idx] = 1.0
                else:
                    weights[idx] = self.lambda_val
        
        # Equalized odds weighting
        elif self.fairness_metric == 'equalized_odds':
            # Upweight minority within each class (y=0 and y=1)
            y_selected = y[selected_indices]
            
            for idx in selected_indices:
                if z[idx] == 0:
                    weights[idx] = 1.0
                else:
                    # Apply lambda weighting to minority group
                    weights[idx] = self.lambda_val
        
        else:
            # Default: uniform weights
            weights[selected_indices] = 1.0
        
        # Normalize weights to sum to number of selected samples
        if weights.sum() > 0:
            weights = weights * len(selected_indices) / weights.sum()
        
        return weights
    
    def update_lambda(self, disparity, target_disparity=0.05):
        """
        Update fairness parameter lambda based on current disparity
        
        Lambda update rule (simplified):
        - If disparity > target: increase lambda (upweight minority more)
        - If disparity < target: decrease lambda (balance is good)
        
        Args:
            disparity: Current fairness disparity (e.g., EO disparity)
            target_disparity: Target disparity (default: 0.05)
        """
        # Gradient-based update
        disparity_gap = disparity - target_disparity
        
        if disparity_gap > 0:
            # Disparity too high → increase lambda
            self.lambda_val += self.lambda_lr * disparity_gap
        else:
            # Disparity acceptable → decrease lambda slightly
            self.lambda_val -= self.lambda_lr * abs(disparity_gap) * 0.5
        
        # Clip lambda to reasonable range [0.5, 3.0]
        self.lambda_val = np.clip(self.lambda_val, 0.5, 3.0)
        
        # Track history
        self.lambda_history.append(self.lambda_val)
    
    def get_lambda_history(self):
        """Return history of lambda values"""
        return np.array(self.lambda_history)


# Test the greedy selector
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.data_loader import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    
    # Define simple logistic regression model
    class LogisticRegression(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    print("="*70)
    print(" " * 20 + "TESTING GREEDY SELECTOR")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading COMPAS dataset...")
    try:
        loader = DataLoader(dataset_name="compas", data_dir="data/raw")
        data = loader.load_and_prepare(noise_rate=0.1, seed=42)
        
        X_train = torch.FloatTensor(data['X_train'])
        y_train = torch.FloatTensor(data['y_train_noisy'])
        z_train = data['z_train']
        
        print(f"  ✓ Loaded {len(X_train)} training samples")
        
    except FileNotFoundError:
        print("  ✗ COMPAS dataset not found")
        print("  Please run: python experiments/01_reproduce_baseline.py first")
        exit(1)
    
    # Train a simple model
    print("\n[2/4] Training baseline model...")
    model = LogisticRegression(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()
    
    print(f"  ✓ Model trained (loss: {loss.item():.4f})")
    
    # Test greedy selector
    print("\n[3/4] Running greedy sample selection...")
    selector = GreedySelector(tau=0.7, lambda_init=1.5)
    
    selection_mask, weights = selector.select_samples(
        model, X_train, y_train, z_train
    )
    
    n_selected = selection_mask.sum()
    print(f"  ✓ Selected {n_selected}/{len(X_train)} samples ({n_selected/len(X_train):.1%})")
    
    # Analyze selection
    print("\n[4/4] Analyzing selection quality...")
    
    # Check if selected samples are truly cleaner
    clean_labels = torch.FloatTensor(data['y_train'])
    noisy_labels = y_train
    
    # Noise rate in selected vs rejected
    noise_rate_selected = (clean_labels[selection_mask] != noisy_labels[selection_mask]).float().mean()
    noise_rate_rejected = (clean_labels[~selection_mask] != noisy_labels[~selection_mask]).float().mean()
    
    print(f"\n  Noise Rate Analysis:")
    print(f"    Selected samples: {noise_rate_selected:.2%} noise")
    print(f"    Rejected samples: {noise_rate_rejected:.2%} noise")
    print(f"    Improvement: {(noise_rate_rejected - noise_rate_selected):.2%}")
    
    # Group balance
    z_selected = z_train[selection_mask]
    z_rejected = z_train[~selection_mask]
    
    print(f"\n  Group Balance:")
    print(f"    Selected: {z_selected.mean():.1%} minority")
    print(f"    Rejected: {z_rejected.mean():.1%} minority")
    print(f"    Overall: {z_train.mean():.1%} minority")
    
    # Weight statistics
    weights_z0 = weights[selection_mask & (z_train == 0)]
    weights_z1 = weights[selection_mask & (z_train == 1)]
    
    print(f"\n  Sample Weights:")
    print(f"    Majority (z=0): mean={weights_z0.mean():.2f}, std={weights_z0.std():.2f}")
    print(f"    Minority (z=1): mean={weights_z1.mean():.2f}, std={weights_z1.std():.2f}")
    print(f"    Ratio (z1/z0): {weights_z1.mean() / weights_z0.mean():.2f}")
    
    print("\n" + "="*70)
    print(" " * 15 + "GREEDY SELECTOR TEST COMPLETE!")
    print("="*70)
    
    print("\n✓ Key Findings:")
    if noise_rate_selected < noise_rate_rejected:
        print("  • Selector successfully identifies cleaner samples")
    else:
        print("  • Warning: Selected samples not significantly cleaner")
    
    if abs(z_selected.mean() - z_train.mean()) < 0.05:
        print("  • Good demographic balance maintained")
    else:
        print("  • Warning: Some demographic imbalance in selection")
    
    print("\n")
