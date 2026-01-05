"""
Fairness metrics for evaluating model performance across demographic groups
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class FairnessMetrics:
    """
    Calculate various fairness metrics for binary classification
    """
    
    @staticmethod
    def demographic_parity(y_pred, z):
        """
        Demographic Parity (Statistical Parity) Disparity
        
        Measures difference in positive prediction rates between groups:
        DP_disparity = |P(Ŷ=1|Z=0) - P(Ŷ=1|Z=1)|
        
        Args:
            y_pred: Predicted labels (n_samples,)
            z: Sensitive attributes (n_samples,) - binary (0 or 1)
            
        Returns:
            disparity: Float in [0, 1], lower is better (0 = perfect fairness)
        """
        z0_mask = (z == 0)
        z1_mask = (z == 1)
        
        # Positive rate for each group
        pr_z0 = y_pred[z0_mask].mean() if z0_mask.sum() > 0 else 0
        pr_z1 = y_pred[z1_mask].mean() if z1_mask.sum() > 0 else 0
        
        disparity = abs(pr_z0 - pr_z1)
        return disparity
    
    @staticmethod
    def equalized_odds(y_true, y_pred, z):
        """
        Equalized Odds Disparity
        
        Measures max difference in TPR and FPR between groups:
        EO_disparity = max(|TPR_z0 - TPR_z1|, |FPR_z0 - FPR_z1|)
        
        Where:
        - TPR (True Positive Rate) = TP / (TP + FN)
        - FPR (False Positive Rate) = FP / (FP + TN)
        
        Args:
            y_true: True labels (n_samples,)
            y_pred: Predicted labels (n_samples,)
            z: Sensitive attributes (n_samples,)
            
        Returns:
            disparity: Float in [0, 1], lower is better (0 = perfect fairness)
        """
        z0_mask = (z == 0)
        z1_mask = (z == 1)
        
        # Compute confusion matrix for group 0
        if z0_mask.sum() > 0:
            cm0 = confusion_matrix(y_true[z0_mask], y_pred[z0_mask], labels=[0, 1])
            tn0, fp0, fn0, tp0 = cm0.ravel() if cm0.size == 4 else (0, 0, 0, 0)
            tpr_z0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0
            fpr_z0 = fp0 / (fp0 + tn0) if (fp0 + tn0) > 0 else 0
        else:
            tpr_z0, fpr_z0 = 0, 0
        
        # Compute confusion matrix for group 1
        if z1_mask.sum() > 0:
            cm1 = confusion_matrix(y_true[z1_mask], y_pred[z1_mask], labels=[0, 1])
            tn1, fp1, fn1, tp1 = cm1.ravel() if cm1.size == 4 else (0, 0, 0, 0)
            tpr_z1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
            fpr_z1 = fp1 / (fp1 + tn1) if (fp1 + tn1) > 0 else 0
        else:
            tpr_z1, fpr_z1 = 0, 0
        
        # Compute disparities
        tpr_disparity = abs(tpr_z0 - tpr_z1)
        fpr_disparity = abs(fpr_z0 - fpr_z1)
        
        # Equalized odds requires both TPR and FPR to be equal
        disparity = max(tpr_disparity, fpr_disparity)
        
        return disparity
    
    @staticmethod
    def equal_opportunity(y_true, y_pred, z):
        """
        Equal Opportunity Disparity
        
        Measures difference in TPR (True Positive Rate) between groups:
        EOP_disparity = |TPR_z0 - TPR_z1|
        
        This is a relaxed version of Equalized Odds that only considers TPR.
        
        Args:
            y_true: True labels (n_samples,)
            y_pred: Predicted labels (n_samples,)
            z: Sensitive attributes (n_samples,)
            
        Returns:
            disparity: Float in [0, 1], lower is better
        """
        z0_mask = (z == 0)
        z1_mask = (z == 1)
        
        # TPR for group 0
        if z0_mask.sum() > 0:
            cm0 = confusion_matrix(y_true[z0_mask], y_pred[z0_mask], labels=[0, 1])
            tn0, fp0, fn0, tp0 = cm0.ravel() if cm0.size == 4 else (0, 0, 0, 0)
            tpr_z0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0
        else:
            tpr_z0 = 0
        
        # TPR for group 1
        if z1_mask.sum() > 0:
            cm1 = confusion_matrix(y_true[z1_mask], y_pred[z1_mask], labels=[0, 1])
            tn1, fp1, fn1, tp1 = cm1.ravel() if cm1.size == 4 else (0, 0, 0, 0)
            tpr_z1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
        else:
            tpr_z1 = 0
        
        disparity = abs(tpr_z0 - tpr_z1)
        return disparity
    
    @staticmethod
    def compute_all_metrics(y_true, y_pred, z, verbose=False):
        """
        Compute all fairness and performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            z: Sensitive attributes
            verbose: If True, print detailed metrics
            
        Returns:
            metrics: Dictionary containing:
                - accuracy: Overall accuracy
                - dp_disparity: Demographic parity disparity
                - eo_disparity: Equalized odds disparity
                - eop_disparity: Equal opportunity disparity
        """
        # Performance metrics
        acc = accuracy_score(y_true, y_pred)
        
        # Fairness metrics
        dp = FairnessMetrics.demographic_parity(y_pred, z)
        eo = FairnessMetrics.equalized_odds(y_true, y_pred, z)
        eop = FairnessMetrics.equal_opportunity(y_true, y_pred, z)
        
        metrics = {
            'accuracy': acc,
            'dp_disparity': dp,
            'eo_disparity': eo,
            'eop_disparity': eop
        }
        
        if verbose:
            print("\nPerformance Metrics:")
            print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print("\nFairness Metrics (lower is better):")
            print(f"  Demographic Parity Disparity: {dp:.4f}")
            print(f"  Equalized Odds Disparity: {eo:.4f}")
            print(f"  Equal Opportunity Disparity: {eop:.4f}")
            
            # Group-wise statistics
            z0_mask = (z == 0)
            z1_mask = (z == 1)
            
            acc_z0 = accuracy_score(y_true[z0_mask], y_pred[z0_mask])
            acc_z1 = accuracy_score(y_true[z1_mask], y_pred[z1_mask])
            
            print("\nGroup-wise Performance:")
            print(f"  Group 0 (majority) accuracy: {acc_z0:.4f}")
            print(f"  Group 1 (minority) accuracy: {acc_z1:.4f}")
            print(f"  Accuracy gap: {abs(acc_z0 - acc_z1):.4f}")
        
        return metrics
    
    @staticmethod
    def print_confusion_matrices(y_true, y_pred, z):
        """
        Print confusion matrices for each demographic group
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            z: Sensitive attributes
        """
        z0_mask = (z == 0)
        z1_mask = (z == 1)
        
        print("\nGroup 0 (Majority) Confusion Matrix:")
        if z0_mask.sum() > 0:
            cm0 = confusion_matrix(y_true[z0_mask], y_pred[z0_mask])
            print(f"  TN={cm0[0,0]}, FP={cm0[0,1]}")
            print(f"  FN={cm0[1,0]}, TP={cm0[1,1]}")
        else:
            print("  No samples in this group")
        
        print("\nGroup 1 (Minority) Confusion Matrix:")
        if z1_mask.sum() > 0:
            cm1 = confusion_matrix(y_true[z1_mask], y_pred[z1_mask])
            print(f"  TN={cm1[0,0]}, FP={cm1[0,1]}")
            print(f"  FN={cm1[1,0]}, TP={cm1[1,1]}")
        else:
            print("  No samples in this group")


# Test fairness metrics
if __name__ == "__main__":
    print("="*60)
    print("Testing Fairness Metrics")
    print("="*60)
    
    # Create dummy data
    np.random.seed(42)
    n = 1000
    
    # Simulate biased predictions
    y_true = np.random.randint(0, 2, n)
    z = np.random.randint(0, 2, n)
    
    # Create biased predictions (worse for minority group)
    y_pred = y_true.copy()
    minority_mask = (z == 1)
    
    # Add more errors to minority group
    n_errors_minority = int(minority_mask.sum() * 0.3)
    error_indices = np.random.choice(
        np.where(minority_mask)[0], 
        size=n_errors_minority, 
        replace=False
    )
    y_pred[error_indices] = 1 - y_pred[error_indices]
    
    # Add fewer errors to majority group
    majority_mask = (z == 0)
    n_errors_majority = int(majority_mask.sum() * 0.1)
    error_indices = np.random.choice(
        np.where(majority_mask)[0], 
        size=n_errors_majority, 
        replace=False
    )
    y_pred[error_indices] = 1 - y_pred[error_indices]
    
    # Compute metrics
    metrics = FairnessMetrics.compute_all_metrics(
        y_true, y_pred, z, verbose=True
    )
    
    # Print confusion matrices
    FairnessMetrics.print_confusion_matrices(y_true, y_pred, z)
    
    print("\n" + "="*60)
    print("Fairness metrics computed successfully!")
    print("="*60)
