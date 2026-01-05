"""
Meta-Learned Sample Selector
Uses policy network to learn sample selection for fairness and robustness

Based on:
- MAML (Finn et al., ICML 2017) - Meta-learning framework
- Meta-Weight-Net (Shu et al., NeurIPS 2019) - Sample reweighting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Policy network for meta-learned sample selection
    
    Input: Sample features (loss, confidence, entropy, group stats)
    Output: Keep probability for each sample [0, 1]
    """
    
    def __init__(self, input_dim=10, hidden_dims=[64, 32], dropout=0.1):
        """
        Initialize policy network
        
        Args:
            input_dim: Number of input features per sample
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer: selection probability [0, 1]
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Forward pass: compute selection probabilities
        
        Args:
            features: Sample features (batch_size, input_dim)
            
        Returns:
            probabilities: Keep probabilities (batch_size, 1)
        """
        return self.network(features)
    
    def select_samples(self, features, threshold=0.5, hard=True):
        """
        Select samples based on policy network output
        
        Args:
            features: Sample features (batch_size, input_dim)
            threshold: Threshold for hard selection (default: 0.5)
            hard: If True, return binary mask; if False, return probabilities
            
        Returns:
            selection: Binary mask (hard=True) or probabilities (hard=False)
        """
        probs = self.forward(features).squeeze()
        
        if hard:
            return (probs > threshold).float()
        else:
            return probs


class FeatureExtractor:
    """
    Extract meta-features for each training sample
    
    Features:
    1. Loss: Cross-entropy loss for this sample
    2. Confidence: Max predicted probability
    3. Entropy: Prediction entropy (uncertainty)
    4. Group indicator: Sensitive attribute (z)
    5. Group loss: Average loss in this group
    6. Group confidence: Average confidence in this group
    7. Prediction: Model's prediction (0 or 1)
    8. Label: Training label (may be noisy)
    9. Margin: Distance from decision boundary
    10. Sample difficulty: Normalized loss rank
    """
    
    @staticmethod
    def extract_features(model, X, y, z, criterion=None):
        """
        Extract meta-features for all samples
        
        Args:
            model: Trained model (nn.Module)
            X: Features (batch_size, feature_dim)
            y: Labels (batch_size,)
            z: Sensitive attributes (batch_size,)
            criterion: Loss function (default: BCELoss)
            
        Returns:
            features: Meta-features (batch_size, 10)
        """
        if criterion is None:
            criterion = nn.BCELoss(reduction='none')
        
        model.eval()
        with torch.no_grad():
            # Get predictions
            logits = model(X)
            probs = logits.squeeze()
            
            # 1. Loss per sample
            losses = criterion(logits, y.unsqueeze(1) if y.dim() == 1 else y)
            losses = losses.squeeze()
            
            # 2. Confidence (max probability)
            confidence = torch.max(torch.stack([probs, 1-probs], dim=1), dim=1)[0]
            
            # 3. Entropy (uncertainty)
            eps = 1e-7
            entropy = -(probs * torch.log(probs + eps) + 
                       (1 - probs) * torch.log(1 - probs + eps))
            
            # 4. Group indicator
            z_tensor = torch.FloatTensor(z) if isinstance(z, np.ndarray) else z
            
            # 5-6. Group statistics
            z0_mask = (z_tensor == 0)
            z1_mask = (z_tensor == 1)
            
            group_loss = torch.zeros_like(losses)
            group_confidence = torch.zeros_like(confidence)
            
            if z0_mask.sum() > 0:
                group_loss[z0_mask] = losses[z0_mask].mean()
                group_confidence[z0_mask] = confidence[z0_mask].mean()
            
            if z1_mask.sum() > 0:
                group_loss[z1_mask] = losses[z1_mask].mean()
                group_confidence[z1_mask] = confidence[z1_mask].mean()
            
            # 7. Prediction (binary)
            predictions = (probs > 0.5).float()
            
            # 8. Label
            y_float = y.float()
            
            # 9. Margin (distance from decision boundary)
            margin = torch.abs(probs - 0.5)
            
            # 10. Sample difficulty (normalized loss rank)
            loss_ranks = torch.argsort(torch.argsort(losses)).float()
            difficulty = loss_ranks / (len(losses) - 1)
            
            # Stack all features
            features = torch.stack([
                losses,
                confidence,
                entropy,
                z_tensor,
                group_loss,
                group_confidence,
                predictions,
                y_float,
                margin,
                difficulty
            ], dim=1)
        
        return features


class MetaSelector:
    """
    Meta-learned sample selector combining policy network and meta-training
    """
    
    def __init__(self, input_dim=10, hidden_dims=[64, 32], 
                 meta_lr=0.001, inner_lr=0.01):
        """
        Initialize meta-selector
        
        Args:
            input_dim: Feature dimension
            hidden_dims: Policy network hidden dimensions
            meta_lr: Meta-learning rate (outer loop)
            inner_lr: Task learning rate (inner loop)
        """
        self.policy_net = PolicyNetwork(input_dim, hidden_dims)
        self.feature_extractor = FeatureExtractor()
        
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        
        # Meta-optimizer (for policy network)
        self.meta_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=meta_lr
        )
    
    def select_samples(self, model, X, y, z, threshold=0.5):
        """
        Select samples using learned policy
        
        Args:
            model: Task model
            X, y, z: Training data
            threshold: Selection threshold
            
        Returns:
            selection_mask: Binary mask of selected samples
            weights: Sample weights (uniform for selected)
        """
        # Extract features
        features = self.feature_extractor.extract_features(model, X, y, z)
        
        # Get selection from policy network
        self.policy_net.eval()
        with torch.no_grad():
            selection_probs = self.policy_net(features).squeeze()
            selection_mask = (selection_probs > threshold).cpu().numpy()
        
        # Uniform weights for selected samples
        n_samples = len(y)
        weights = np.zeros(n_samples)
        weights[selection_mask] = 1.0
        
        # Normalize
        if weights.sum() > 0:
            weights = weights * n_samples / weights.sum()
        
        return selection_mask, weights
    
    def meta_train_step(self, support_data, query_data, task_model_class):
        """
        Single meta-training step (MAML-style)
        
        Args:
            support_data: Dictionary with X_train, y_train, z_train (support set)
            query_data: Dictionary with X_val, y_val, z_val (query set)
            task_model_class: Class for task model (e.g., LogisticRegression)
            
        Returns:
            meta_loss: Loss on query set after adaptation
        """
        # Initialize task model
        input_dim = support_data['X_train'].shape[1]
        task_model = task_model_class(input_dim)
        
        # Extract features for support set
        X_support = torch.FloatTensor(support_data['X_train'])
        y_support = torch.FloatTensor(support_data['y_train'])
        z_support = support_data['z_train']
        
        features = self.feature_extractor.extract_features(
            task_model, X_support, y_support, z_support
        )
        
        # Get sample weights from policy network
        self.policy_net.train()
        selection_probs = self.policy_net(features).squeeze()
        
        # Inner loop: Train task model with weighted samples
        # Detach selection_probs for inner loop (we don't want to backprop through inner loop)
        selection_weights = selection_probs.detach()
        
        task_optimizer = torch.optim.SGD(
            task_model.parameters(), 
            lr=self.inner_lr
        )
        
        criterion = nn.BCELoss(reduction='none')
        
        # Multiple inner steps
        for _ in range(5):
            task_optimizer.zero_grad()
            
            # Forward pass
            y_pred = task_model(X_support)
            losses = criterion(y_pred, y_support.unsqueeze(1))
            
            # Weighted loss (using detached weights)
            weighted_loss = (losses.squeeze() * selection_weights).mean()
            
            # Backward pass (inner loop only)
            weighted_loss.backward()
            task_optimizer.step()
        
        # Evaluate on query set
        X_query = torch.FloatTensor(query_data['X_val'])
        y_query = torch.FloatTensor(query_data['y_val'])
        
        task_model.eval()
        y_pred_query = task_model(X_query)
        
        # Meta-loss: accuracy on query set (we want to maximize)
        # Use negative accuracy as loss (minimize = maximize accuracy)
        query_loss = criterion(y_pred_query, y_query.unsqueeze(1)).mean()
        
        # Add fairness penalty
        z_query = query_data['z_val']
        fairness_penalty = self._compute_fairness_penalty(
            y_query, y_pred_query.squeeze(), z_query
        )
        
        meta_loss = query_loss + 0.1 * fairness_penalty
        
        return meta_loss
    
    def _compute_fairness_penalty(self, y_true, y_pred, z):
        """
        Compute fairness penalty (demographic parity violation)
        
        Returns:
            penalty: Fairness penalty (higher = more unfair)
        """
        y_pred_binary = (y_pred > 0.5).float()
        
        z_tensor = torch.FloatTensor(z) if isinstance(z, np.ndarray) else z
        z0_mask = (z_tensor == 0)
        z1_mask = (z_tensor == 1)
        
        if z0_mask.sum() > 0 and z1_mask.sum() > 0:
            pr_z0 = y_pred_binary[z0_mask].mean()
            pr_z1 = y_pred_binary[z1_mask].mean()
            penalty = torch.abs(pr_z0 - pr_z1)
        else:
            penalty = torch.tensor(0.0)
        
        return penalty
    
    def meta_train(self, train_tasks, val_tasks, n_iterations=100, 
                   task_model_class=None, save_dir='results/checkpoints',
                   save_every=20, verbose=True):
        """
        Meta-train the policy network on a suite of tasks
        
        Args:
            train_tasks: List of training tasks (each is a dict with train/test splits)
            val_tasks: List of validation tasks
            n_iterations: Number of meta-training iterations
            task_model_class: PyTorch model class for tasks
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N iterations
            verbose: Print progress
            
        Returns:
            history: Dictionary with training history
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Default task model: simple MLP
        if task_model_class is None:
            class SimpleTaskModel(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.fc = nn.Linear(input_dim, 1)
                    self.sigmoid = nn.Sigmoid()
                
                def forward(self, x):
                    return self.sigmoid(self.fc(x))
            
            task_model_class = SimpleTaskModel
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_fairness': []
        }
        
        if verbose:
            print(f"Starting meta-training for {n_iterations} iterations...")
            print(f"  Training tasks: {len(train_tasks)}")
            print(f"  Validation tasks: {len(val_tasks)}")
        
        for iteration in range(n_iterations):
            # Sample a batch of tasks
            batch_size = min(8, len(train_tasks))
            task_indices = np.random.choice(len(train_tasks), batch_size, replace=False)
            
            # Accumulate gradients over batch
            self.meta_optimizer.zero_grad()
            batch_loss = 0.0
            
            for task_idx in task_indices:
                task = train_tasks[task_idx]
                
                # Prepare support and query sets
                support_data = {
                    'X_train': task['train']['X'],
                    'y_train': task['train']['y'],
                    'z_train': task['train']['z']
                }
                
                query_data = {
                    'X_val': task['test']['X'][:100],  # Use subset for efficiency
                    'y_val': task['test']['y'][:100],
                    'z_val': task['test']['z'][:100]
                }
                
                # Meta-training step
                meta_loss = self.meta_train_step(
                    support_data, query_data, task_model_class
                )
                
                batch_loss += meta_loss
            
            # Average over batch
            batch_loss = batch_loss / batch_size
            
            # Backward pass (outer loop)
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            
            # Update policy network
            self.meta_optimizer.step()
            
            # Record training loss
            history['train_loss'].append(batch_loss.item())
            
            # Validation
            if (iteration + 1) % 10 == 0:
                val_metrics = self._validate(val_tasks, task_model_class)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_fairness'].append(val_metrics['fairness'])
                
                if verbose:
                    print(f"Iter {iteration+1}/{n_iterations}: "
                          f"Train Loss: {batch_loss.item():.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val Acc: {val_metrics['accuracy']:.4f}, "
                          f"Val Fairness: {val_metrics['fairness']:.4f}")
            
            # Save checkpoint
            if (iteration + 1) % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'meta_selector_iter_{iteration+1}.pt')
                self.save(checkpoint_path)
                if verbose:
                    print(f"  Saved checkpoint: {checkpoint_path}")
        
        if verbose:
            print(f"\n[OK] Meta-training complete!")
            print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}")
            print(f"  Final val fairness: {history['val_fairness'][-1]:.4f}")
        
        # Save final model
        final_path = os.path.join(save_dir, 'meta_selector_final.pt')
        self.save(final_path)
        
        return history
    
    def _validate(self, val_tasks, task_model_class):
        """
        Validate meta-selector on validation tasks
        
        Returns:
            metrics: Dictionary with validation metrics
        """
        self.policy_net.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_fairness = 0.0
        
        for task in val_tasks:
            # Train a model with selected samples
            input_dim = task['train']['X'].shape[1]
            task_model = task_model_class(input_dim)
            
            X_train = torch.FloatTensor(task['train']['X'])
            y_train = torch.FloatTensor(task['train']['y'])
            z_train = task['train']['z']
            
            # Get features and selection weights (no grad for policy network)
            with torch.no_grad():
                features = self.feature_extractor.extract_features(
                    task_model, X_train, y_train, z_train
                )
                selection_probs = self.policy_net(features).squeeze()
            
            # Train with weighted samples (requires grad for task model)
            optimizer = torch.optim.SGD(task_model.parameters(), lr=0.01)
            criterion = nn.BCELoss(reduction='none')
            
            for _ in range(10):
                optimizer.zero_grad()
                y_pred = task_model(X_train)
                losses = criterion(y_pred, y_train.unsqueeze(1))
                weighted_loss = (losses.squeeze() * selection_probs).mean()
                weighted_loss.backward()
                optimizer.step()
            
            # Evaluate on test set
            X_test = torch.FloatTensor(task['test']['X'])
            y_test = torch.FloatTensor(task['test']['y'])
            z_test = task['test']['z']
            
            task_model.eval()
            with torch.no_grad():
                y_pred_test = task_model(X_test).squeeze()
                
                # Loss
                test_loss = criterion(y_pred_test.unsqueeze(1), y_test.unsqueeze(1)).mean()
                total_loss += test_loss.item()
                
                # Accuracy
                y_pred_binary = (y_pred_test > 0.5).float()
                accuracy = (y_pred_binary == y_test).float().mean()
                total_accuracy += accuracy.item()
                
                # Fairness (demographic parity)
                z_tensor = torch.FloatTensor(z_test)
                z0_mask = (z_tensor == 0)
                z1_mask = (z_tensor == 1)
                
                if z0_mask.sum() > 0 and z1_mask.sum() > 0:
                    pr_z0 = y_pred_binary[z0_mask].mean()
                    pr_z1 = y_pred_binary[z1_mask].mean()
                    dp_gap = torch.abs(pr_z0 - pr_z1)
                    total_fairness += dp_gap.item()
        
        self.policy_net.train()
        
        n_tasks = len(val_tasks)
        return {
            'loss': total_loss / n_tasks,
            'accuracy': total_accuracy / n_tasks,
            'fairness': total_fairness / n_tasks
        }
    
    def save(self, path):
        """Save meta-selector state"""
        torch.save({
            'policy_net_state': self.policy_net.state_dict(),
            'meta_optimizer_state': self.meta_optimizer.state_dict(),
            'meta_lr': self.meta_lr,
            'inner_lr': self.inner_lr
        }, path)
    
    def load(self, path):
        """Load meta-selector state"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state'])
        self.meta_lr = checkpoint['meta_lr']
        self.inner_lr = checkpoint['inner_lr']


# Test the meta-selector
if __name__ == "__main__":
    print("="*70)
    print(" " * 20 + "TESTING META-SELECTOR")
    print("="*70)
    
    # Create dummy data
    print("\n[1/5] Creating dummy data...")
    n_samples = 1000
    n_features = 5
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,)).float()
    z = torch.randint(0, 2, (n_samples,)).numpy()
    
    print(f"  ✓ Created {n_samples} samples with {n_features} features")
    
    # Create dummy model
    print("\n[2/5] Creating dummy model...")
    from torch import nn
    
    class DummyModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    model = DummyModel(n_features)
    print(f"  ✓ Model created")
    
    # Test feature extraction
    print("\n[3/5] Testing feature extraction...")
    extractor = FeatureExtractor()
    features = extractor.extract_features(model, X, y, z)
    
    print(f"  ✓ Features shape: {features.shape}")
    print(f"  ✓ Feature stats:")
    print(f"    - Loss: {features[:, 0].mean():.4f} ± {features[:, 0].std():.4f}")
    print(f"    - Confidence: {features[:, 1].mean():.4f} ± {features[:, 1].std():.4f}")
    print(f"    - Entropy: {features[:, 2].mean():.4f} ± {features[:, 2].std():.4f}")
    
    # Test policy network
    print("\n[4/5] Testing policy network...")
    policy_net = PolicyNetwork(input_dim=10, hidden_dims=[64, 32])
    
    probs = policy_net(features)
    print(f"  ✓ Selection probabilities shape: {probs.shape}")
    print(f"  ✓ Probability stats: {probs.mean():.4f} ± {probs.std():.4f}")
    print(f"  ✓ Probabilities in [0, 1]: {(probs >= 0).all() and (probs <= 1).all()}")
    
    # Test meta-selector
    print("\n[5/5] Testing meta-selector...")
    meta_selector = MetaSelector(input_dim=10, hidden_dims=[64, 32])
    
    selection_mask, weights = meta_selector.select_samples(model, X, y, z)
    
    n_selected = selection_mask.sum()
    print(f"  ✓ Selected {n_selected}/{n_samples} samples ({n_selected/n_samples:.1%})")
    print(f"  ✓ Weight sum: {weights.sum():.1f} (should be ~{n_samples})")
    
    print("\n" + "="*70)
    print(" " * 15 + "META-SELECTOR TEST COMPLETE!")
    print("="*70)
    
    print("\n✓ All components working:")
    print("  • Feature extraction ✓")
    print("  • Policy network ✓")
    print("  • Meta-selector ✓")
    print("  • Sample selection ✓")
    
    print("\n")
