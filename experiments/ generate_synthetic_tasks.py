"""
Generate 100 Synthetic Tasks for Meta-Training
===============================================

This script generates a diverse suite of 100 synthetic classification tasks
for meta-learning. Each task has different characteristics:

- Sample sizes: 100-10,000
- Feature dimensions: 5-20
- Noise rates: 0-30%
- Group imbalances: 10-90%
- Class imbalances: 20-80%
- Separability: 0.5 (hard) to 2.0 (easy)

Tasks are saved to data/synthetic/ for meta-training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.synthetic_generator import SyntheticDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 80)
    print("Generating 100 Synthetic Tasks for Meta-Training")
    print("=" * 80)
    
    # Initialize generator
    generator = SyntheticDataGenerator(seed=42, tasks_dir='data/synthetic')
    
    # Generate 100 tasks
    tasks = generator.generate_task_suite(
        n_tasks=100,
        sample_range=(100, 10000),
        feature_range=(5, 20),
        noise_range=(0.0, 0.3),
        group_imbalance_range=(0.1, 0.9),
        class_imbalance_range=(0.2, 0.8),
        separability_range=(0.5, 2.0),
        save=True
    )
    
    # Compute statistics
    print("\n" + "=" * 80)
    print("Task Suite Statistics")
    print("=" * 80)
    
    n_samples = [t['metadata']['n_train'] for t in tasks]
    n_features = [t['metadata']['n_features'] for t in tasks]
    noise_rates = [t['metadata']['train_actual_noise_rate'] for t in tasks]
    minority_rates = [t['metadata']['train_minority_rate'] for t in tasks]
    positive_rates = [t['metadata']['train_pos_rate'] for t in tasks]
    dp_gaps = [t['metadata']['train_dp_gap'] for t in tasks]
    
    print(f"\nSample Sizes:")
    print(f"  Min: {min(n_samples)}, Max: {max(n_samples)}, Mean: {np.mean(n_samples):.0f}")
    
    print(f"\nFeature Dimensions:")
    print(f"  Min: {min(n_features)}, Max: {max(n_features)}, Mean: {np.mean(n_features):.1f}")
    
    print(f"\nNoise Rates:")
    print(f"  Min: {min(noise_rates):.2%}, Max: {max(noise_rates):.2%}, Mean: {np.mean(noise_rates):.2%}")
    
    print(f"\nMinority Rates:")
    print(f"  Min: {min(minority_rates):.2%}, Max: {max(minority_rates):.2%}, Mean: {np.mean(minority_rates):.2%}")
    
    print(f"\nPositive Class Rates:")
    print(f"  Min: {min(positive_rates):.2%}, Max: {max(positive_rates):.2%}, Mean: {np.mean(positive_rates):.2%}")
    
    print(f"\nDemographic Parity Gaps:")
    print(f"  Min: {min(dp_gaps):.3f}, Max: {max(dp_gaps):.3f}, Mean: {np.mean(dp_gaps):.3f}")
    
    # Create visualization
    print("\n" + "=" * 80)
    print("Creating Visualization...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Synthetic Task Suite Diversity (100 Tasks)', fontsize=16, fontweight='bold')
    
    # Sample sizes
    axes[0, 0].hist(n_samples, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Training Samples')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Sample Sizes\n(Mean: {np.mean(n_samples):.0f})')
    axes[0, 0].grid(alpha=0.3)
    
    # Feature dimensions
    axes[0, 1].hist(n_features, bins=range(5, 22), edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Features')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Feature Dimensions\n(Mean: {np.mean(n_features):.1f})')
    axes[0, 1].grid(alpha=0.3)
    
    # Noise rates
    axes[0, 2].hist(noise_rates, bins=20, edgecolor='black', alpha=0.7, color='red')
    axes[0, 2].set_xlabel('Noise Rate')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title(f'Label Noise\n(Mean: {np.mean(noise_rates):.2%})')
    axes[0, 2].grid(alpha=0.3)
    
    # Minority rates
    axes[1, 0].hist(minority_rates, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Minority Proportion')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Group Imbalance\n(Mean: {np.mean(minority_rates):.2%})')
    axes[1, 0].grid(alpha=0.3)
    
    # Positive class rates
    axes[1, 1].hist(positive_rates, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Positive Class Proportion')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Class Imbalance\n(Mean: {np.mean(positive_rates):.2%})')
    axes[1, 1].grid(alpha=0.3)
    
    # DP gaps
    axes[1, 2].hist(dp_gaps, bins=20, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 2].set_xlabel('Demographic Parity Gap')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title(f'Fairness Violations\n(Mean: {np.mean(dp_gaps):.3f})')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results/plots', exist_ok=True)
    plot_file = 'results/plots/synthetic_task_diversity.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_file}")
    
    # Create correlation matrix
    print("\nCreating correlation matrix...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    data = np.column_stack([
        n_samples,
        n_features,
        noise_rates,
        minority_rates,
        positive_rates,
        dp_gaps
    ])
    
    # Compute correlation
    corr = np.corrcoef(data.T)
    
    # Plot heatmap
    labels = ['Samples', 'Features', 'Noise', 'Minority%', 'Positive%', 'DP Gap']
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=-1, vmax=1, square=True)
    
    ax.set_title('Task Parameter Correlations', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    corr_file = 'results/plots/synthetic_task_correlations.png'
    plt.savefig(corr_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved correlation matrix to {corr_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ Task Generation Complete!")
    print("=" * 80)
    print(f"\nGenerated Files:")
    print(f"  - data/synthetic/task_000.npz ... task_099.npz (100 tasks)")
    print(f"  - data/synthetic/metadata.json (task configurations)")
    print(f"  - {plot_file}")
    print(f"  - {corr_file}")
    
    print(f"\nTask Suite Summary:")
    print(f"  Total tasks: 100")
    print(f"  Total training samples: {sum(n_samples):,}")
    print(f"  Average samples per task: {np.mean(n_samples):.0f}")
    print(f"  Average noise rate: {np.mean(noise_rates):.2%}")
    print(f"  Average fairness violation: {np.mean(dp_gaps):.3f}")
    
    print(f"\n✓ Ready for meta-training!")


if __name__ == '__main__':
    main()
