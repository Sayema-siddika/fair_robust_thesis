"""
Day 14: Week 2 Final Checkpoint & Comprehensive Evaluation
==========================================================

Consolidate all Week 2 findings and run comprehensive evaluation:
1. Cross-dataset validation (test all methods on COMPAS, Adult, German)
2. Best configuration comparison (optimal settings from ablations)
3. Method selection decision tree
4. Performance summary tables
5. Week 3 planning

Goal: Complete Week 2 with actionable recommendations and clear path forward.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.fairness.metrics import FairnessMetrics


class Week2Evaluator:
    """Comprehensive Week 2 evaluation framework."""
    
    def __init__(self):
        self.results = {
            'datasets': {},
            'summary': {},
            'recommendations': {}
        }
    
    def load_dataset(self, dataset_name):
        """Load and prepare dataset."""
        loader = DataLoader(dataset_name)
        X, y, z = loader.load_dataset()
        
        X_train, X_test, y_train, y_test, z_train, z_test = \
            train_test_split(X, y, z, test_size=0.3, random_state=42, stratify=y)
        
        X_train, X_test = loader.preprocess(X_train, X_test)
        
        return {
            'X_train': X_train, 'y_train': y_train, 'z_train': z_train,
            'X_test': X_test, 'y_test': y_test, 'z_test': z_test,
            'n_train': len(X_train), 'n_test': len(X_test)
        }
    
    def train_baseline(self, data):
        """Train baseline model."""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(data['X_train'], data['y_train'])
        return model
    
    def train_greedy(self, data, selection_ratio=0.9):
        """Train greedy selection model (optimal τ=0.9 from ablations)."""
        # Initial model for loss computation
        model_init = LogisticRegression(max_iter=1000, random_state=42)
        model_init.fit(data['X_train'], data['y_train'])
        
        # Compute losses and select
        losses = -model_init.predict_log_proba(data['X_train'])
        sample_losses = np.array([losses[i, data['y_train'][i]] 
                                 for i in range(len(data['y_train']))])
        
        n_select = int(len(sample_losses) * selection_ratio)
        selected_indices = np.argsort(sample_losses)[:n_select]
        
        # Train on selected samples
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(data['X_train'][selected_indices], 
                 data['y_train'][selected_indices])
        
        return model
    
    def train_adaptive(self, data, temperature=0.5):
        """Train adaptive weighting model (optimal T=0.5 from ablations)."""
        # Initial model for weight computation
        model_init = LogisticRegression(max_iter=1000, random_state=42)
        model_init.fit(data['X_train'], data['y_train'])
        
        # Compute adaptive weights
        probs = model_init.predict_proba(data['X_train'])
        max_probs = np.max(probs, axis=1)
        predictions = model_init.predict(data['X_train'])
        correctness = (predictions == data['y_train']).astype(float)
        
        weights = max_probs * correctness + 0.1
        weights = weights ** (1.0 / temperature)
        weights = weights / np.sum(weights) * len(weights)
        
        # Train weighted model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(data['X_train'], data['y_train'], sample_weight=weights)
        
        return model
    
    def evaluate_model(self, model, data):
        """Evaluate model and return all metrics."""
        y_pred = model.predict(data['X_test'])
        
        acc = accuracy_score(data['y_test'], y_pred)
        eo = FairnessMetrics.equalized_odds(data['y_test'], y_pred, data['z_test'])
        
        # Additional metrics
        dp = FairnessMetrics.demographic_parity(y_pred, data['z_test'])
        
        return {
            'accuracy': float(acc),
            'eo_disparity': float(eo),
            'dp_disparity': float(dp)
        }
    
    def evaluate_all_datasets(self):
        """Run comprehensive evaluation on all 3 datasets."""
        print("\n" + "="*70)
        print("COMPREHENSIVE CROSS-DATASET EVALUATION")
        print("="*70)
        
        datasets = ['compas', 'adult', 'german']
        
        for dataset_name in datasets:
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset_name.upper()}")
            print(f"{'='*70}")
            
            # Load data
            data = self.load_dataset(dataset_name)
            print(f"Train: {data['n_train']} samples, Test: {data['n_test']} samples")
            
            # Train all methods with optimal hyperparameters
            print("\nTraining models...")
            baseline_model = self.train_baseline(data)
            greedy_model = self.train_greedy(data, selection_ratio=0.9)  # Optimal from Day 13
            adaptive_model = self.train_adaptive(data, temperature=0.5)  # Optimal from Day 13
            
            # Evaluate
            baseline_metrics = self.evaluate_model(baseline_model, data)
            greedy_metrics = self.evaluate_model(greedy_model, data)
            adaptive_metrics = self.evaluate_model(adaptive_model, data)
            
            # Compute improvements
            baseline_eo = baseline_metrics['eo_disparity']
            
            greedy_improvement = (baseline_eo - greedy_metrics['eo_disparity']) / baseline_eo * 100
            adaptive_improvement = (baseline_eo - adaptive_metrics['eo_disparity']) / baseline_eo * 100
            
            # Display results
            print(f"\n{'Method':<20} {'Accuracy':<12} {'EO Disparity':<15} {'Fairness Δ':<12}")
            print("-" * 70)
            print(f"{'Baseline':<20} {baseline_metrics['accuracy']:<12.4f} "
                  f"{baseline_metrics['eo_disparity']:<15.4f} {'0.0%':<12}")
            print(f"{'Greedy (τ=0.9)':<20} {greedy_metrics['accuracy']:<12.4f} "
                  f"{greedy_metrics['eo_disparity']:<15.4f} {greedy_improvement:+.1f}%")
            print(f"{'Adaptive (T=0.5)':<20} {adaptive_metrics['accuracy']:<12.4f} "
                  f"{adaptive_metrics['eo_disparity']:<15.4f} {adaptive_improvement:+.1f}%")
            
            # Store results
            self.results['datasets'][dataset_name] = {
                'baseline': baseline_metrics,
                'greedy': greedy_metrics,
                'adaptive': adaptive_metrics,
                'improvements': {
                    'greedy': float(greedy_improvement),
                    'adaptive': float(adaptive_improvement)
                }
            }
            
            # Determine winner
            if greedy_improvement > adaptive_improvement:
                winner = 'Greedy'
                improvement = greedy_improvement
            else:
                winner = 'Adaptive'
                improvement = adaptive_improvement
            
            print(f"\n[WINNER] {winner} with {improvement:+.1f}% fairness improvement!")
    
    def create_summary_table(self):
        """Create comprehensive summary table."""
        print("\n" + "="*70)
        print("WEEK 2 COMPREHENSIVE SUMMARY")
        print("="*70)
        
        print("\n" + "-"*70)
        print("FAIRNESS IMPROVEMENTS BY METHOD AND DATASET")
        print("-"*70)
        
        print(f"\n{'Dataset':<12} {'Baseline EO':<15} {'Greedy Δ':<15} {'Adaptive Δ':<15} {'Best Method':<15}")
        print("-" * 70)
        
        for dataset_name in ['compas', 'adult', 'german']:
            if dataset_name not in self.results['datasets']:
                continue
            
            data = self.results['datasets'][dataset_name]
            baseline_eo = data['baseline']['eo_disparity']
            greedy_imp = data['improvements']['greedy']
            adaptive_imp = data['improvements']['adaptive']
            
            best_method = 'Greedy' if greedy_imp > adaptive_imp else 'Adaptive'
            
            print(f"{dataset_name.capitalize():<12} {baseline_eo:<15.4f} "
                  f"{greedy_imp:+14.1f}% {adaptive_imp:+14.1f}% {best_method:<15}")
        
        # Overall statistics
        print("\n" + "-"*70)
        print("AGGREGATE STATISTICS")
        print("-"*70)
        
        all_greedy = [self.results['datasets'][d]['improvements']['greedy'] 
                     for d in self.results['datasets']]
        all_adaptive = [self.results['datasets'][d]['improvements']['adaptive'] 
                       for d in self.results['datasets']]
        
        print(f"\nGreedy (τ=0.9):")
        print(f"  Average improvement: {np.mean(all_greedy):+.1f}%")
        print(f"  Best: {np.max(all_greedy):+.1f}%")
        print(f"  Worst: {np.min(all_greedy):+.1f}%")
        print(f"  Std dev: {np.std(all_greedy):.1f}%")
        
        print(f"\nAdaptive (T=0.5):")
        print(f"  Average improvement: {np.mean(all_adaptive):+.1f}%")
        print(f"  Best: {np.max(all_adaptive):+.1f}%")
        print(f"  Worst: {np.min(all_adaptive):+.1f}%")
        print(f"  Std dev: {np.std(all_adaptive):.1f}%")
        
        # Store summary
        self.results['summary'] = {
            'greedy': {
                'mean': float(np.mean(all_greedy)),
                'max': float(np.max(all_greedy)),
                'min': float(np.min(all_greedy)),
                'std': float(np.std(all_greedy))
            },
            'adaptive': {
                'mean': float(np.mean(all_adaptive)),
                'max': float(np.max(all_adaptive)),
                'min': float(np.min(all_adaptive)),
                'std': float(np.std(all_adaptive))
            }
        }
    
    def create_recommendations(self):
        """Generate method selection recommendations."""
        print("\n" + "="*70)
        print("PRACTITIONER RECOMMENDATIONS")
        print("="*70)
        
        recommendations = {
            'method_selection': {},
            'hyperparameters': {},
            'deployment_scenarios': {}
        }
        
        print("\n1. METHOD SELECTION FLOWCHART:")
        print("-" * 70)
        print("""
        Dataset Size:
        
        Large (N > 10,000):
        ├─ Clean labels? 
        │  ├─ YES → Greedy (τ=0.9) or Adaptive (T=0.5)
        │  └─ NO/UNSURE → Greedy (τ=0.9) [robust to label noise]
        
        Medium (1,000 < N < 10,000):
        ├─ Distribution shift expected?
        │  ├─ YES → Adaptive (T=0.5) [robust to shift]
        │  └─ NO → Greedy (τ=0.9) [better fairness]
        
        Small (N < 1,000):
        └─ Adaptive (T=1.0-2.0) [soft weighting preserves data]
           [From Week 2: German dataset +54.5% with T=1.0]
        """)
        
        recommendations['method_selection'] = {
            'large_datasets': 'Greedy (τ=0.9) if label noise suspected, otherwise either method',
            'medium_datasets': 'Adaptive (T=0.5) for distribution shift, Greedy (τ=0.9) otherwise',
            'small_datasets': 'Adaptive (T=1.0-2.0) - soft weighting essential'
        }
        
        print("\n2. OPTIMAL HYPERPARAMETERS:")
        print("-" * 70)
        print("Greedy Selection:")
        print("  - Selection ratio (τ): 0.9 [Keep 90% of samples]")
        print("  - Rationale: High ratio preserves fairness better than aggressive filtering")
        print("\nAdaptive Weighting:")
        print("  - Temperature (T): 0.5 for medium/large datasets")
        print("  - Temperature (T): 1.0-2.0 for small datasets")
        print("  - Weighting scheme: Adaptive (confidence × correctness + 0.1)")
        print("  - Rationale: Balances weight concentration and sample preservation")
        
        recommendations['hyperparameters'] = {
            'greedy_tau': 0.9,
            'adaptive_T_large': 0.5,
            'adaptive_T_small': 1.0,
            'weighting_scheme': 'adaptive'
        }
        
        print("\n3. DEPLOYMENT SCENARIOS:")
        print("-" * 70)
        print("Scenario 1: High-stakes with label noise (e.g., criminal justice)")
        print("  → Greedy (τ=0.9) [88% fairness improvement under 30% noise]")
        print("\nScenario 2: Deployment across different populations")
        print("  → Adaptive (T=0.5) [robust to distribution shift]")
        print("\nScenario 3: Limited training data")
        print("  → Adaptive (T=1.0-2.0) [preserves all samples via weighting]")
        print("\nScenario 4: Real-time/production constraints")
        print("  → Greedy (τ=0.9) [simpler, no reweighting needed]")
        
        recommendations['deployment_scenarios'] = {
            'label_noise': 'Greedy (τ=0.9)',
            'distribution_shift': 'Adaptive (T=0.5)',
            'limited_data': 'Adaptive (T=1.0-2.0)',
            'production': 'Greedy (τ=0.9)'
        }
        
        self.results['recommendations'] = recommendations
    
    def plot_comprehensive_summary(self):
        """Create comprehensive Week 2 summary visualization."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Week 2 Comprehensive Summary - Fair Sample Selection', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Cross-dataset comparison
        ax1 = fig.add_subplot(gs[0, :2])
        datasets = ['COMPAS', 'Adult', 'German']
        x = np.arange(len(datasets))
        width = 0.25
        
        baseline_eos = [self.results['datasets'][d.lower()]['baseline']['eo_disparity'] 
                       for d in datasets]
        greedy_eos = [self.results['datasets'][d.lower()]['greedy']['eo_disparity'] 
                     for d in datasets]
        adaptive_eos = [self.results['datasets'][d.lower()]['adaptive']['eo_disparity'] 
                       for d in datasets]
        
        ax1.bar(x - width, baseline_eos, width, label='Baseline', color='gray', alpha=0.7)
        ax1.bar(x, greedy_eos, width, label='Greedy (τ=0.9)', color='green', alpha=0.7)
        ax1.bar(x + width, adaptive_eos, width, label='Adaptive (T=0.5)', color='red', alpha=0.7)
        
        ax1.set_ylabel('EO Disparity (lower is better)', fontsize=11)
        ax1.set_title('Fairness Across Datasets', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Improvement percentages
        ax2 = fig.add_subplot(gs[0, 2])
        greedy_imps = [self.results['datasets'][d.lower()]['improvements']['greedy'] 
                      for d in datasets]
        adaptive_imps = [self.results['datasets'][d.lower()]['improvements']['adaptive'] 
                        for d in datasets]
        
        x = np.arange(len(datasets))
        ax2.barh(x - 0.2, greedy_imps, 0.4, label='Greedy', color='green', alpha=0.7)
        ax2.barh(x + 0.2, adaptive_imps, 0.4, label='Adaptive', color='red', alpha=0.7)
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_yticks(x)
        ax2.set_yticklabels(datasets)
        ax2.set_xlabel('Fairness Improvement (%)', fontsize=10)
        ax2.set_title('Improvements', fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Week 2 timeline
        ax3 = fig.add_subplot(gs[1, :])
        days = ['Day 8\nTransfer\nLearning', 'Day 9\nFairness\nConstraints', 
                'Day 10\nUncertainty\nWeighting', 'Day 11\nPareto\nOptimization',
                'Day 12\nRobustness\nTesting', 'Day 13\nAblation\nStudies']
        findings = ['Meta fails\non small\ndata', 'Hard selection\nhurts small\ndata',
                   'BREAKTHROUGH\n+54.5%\nGerman', 'Pareto\nfrontiers\nidentified',
                   'Greedy\n88% better\nunder noise', 'τ=0.9, T=0.5\noptimal']
        
        colors_timeline = ['red', 'red', 'green', 'blue', 'green', 'blue']
        for i, (day, finding, color) in enumerate(zip(days, findings, colors_timeline)):
            ax3.scatter(i, 1, s=500, alpha=0.7, color=color)
            ax3.text(i, 1, day, ha='center', va='center', fontsize=9, fontweight='bold')
            ax3.text(i, 0.5, finding, ha='center', va='top', fontsize=8)
        
        ax3.set_xlim(-0.5, 5.5)
        ax3.set_ylim(0, 1.5)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('Week 2 Research Timeline', fontweight='bold', fontsize=12)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        
        # Plot 4: Key metrics summary
        ax4 = fig.add_subplot(gs[2, 0])
        metrics = ['Avg\nImprovement', 'Best\nResult', 'Robustness\nScore']
        greedy_vals = [self.results['summary']['greedy']['mean'],
                      self.results['summary']['greedy']['max'],
                      19.0]  # From Day 13 τ=0.9
        adaptive_vals = [self.results['summary']['adaptive']['mean'],
                        self.results['summary']['adaptive']['max'],
                        10.9]  # From Day 13 T=0.5
        
        x = np.arange(len(metrics))
        width = 0.35
        ax4.bar(x - width/2, greedy_vals, width, label='Greedy', color='green', alpha=0.7)
        ax4.bar(x + width/2, adaptive_vals, width, label='Adaptive', color='red', alpha=0.7)
        ax4.set_ylabel('Improvement (%)', fontsize=10)
        ax4.set_title('Method Comparison', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, fontsize=9)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Accuracy-Fairness trade-off
        ax5 = fig.add_subplot(gs[2, 1])
        for dataset_name, marker, color in [('compas', 'o', 'blue'), 
                                            ('adult', 's', 'orange'),
                                            ('german', '^', 'purple')]:
            data = self.results['datasets'][dataset_name]
            
            # Baseline
            ax5.scatter(data['baseline']['eo_disparity'], 
                       data['baseline']['accuracy'],
                       marker=marker, s=100, color='gray', alpha=0.5, 
                       label=f'{dataset_name.capitalize()} Baseline')
            
            # Greedy
            ax5.scatter(data['greedy']['eo_disparity'],
                       data['greedy']['accuracy'],
                       marker=marker, s=150, color='green', alpha=0.7)
            
            # Adaptive
            ax5.scatter(data['adaptive']['eo_disparity'],
                       data['adaptive']['accuracy'],
                       marker=marker, s=150, color='red', alpha=0.7)
        
        ax5.set_xlabel('EO Disparity (lower is better)', fontsize=10)
        ax5.set_ylabel('Accuracy', fontsize=10)
        ax5.set_title('Accuracy-Fairness Trade-off', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=7, loc='lower right')
        
        # Plot 6: Recommendations
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.text(0.5, 0.9, 'Best Configurations', ha='center', fontsize=12, 
                fontweight='bold', transform=ax6.transAxes)
        
        text = """
Greedy Selection:
• τ = 0.9 (keep 90%)
• Best for: label noise
• Avg: +{:.1f}% fairness

Adaptive Weighting:
• T = 0.5 (medium/large)
• T = 1.0 (small datasets)
• Best for: dist. shift
• Avg: +{:.1f}% fairness

Use Greedy when:
→ Label quality uncertain
→ Production deployment

Use Adaptive when:
→ Distribution shift
→ Limited training data
        """.format(self.results['summary']['greedy']['mean'],
                  self.results['summary']['adaptive']['mean'])
        
        ax6.text(0.05, 0.75, text, ha='left', va='top', fontsize=8,
                family='monospace', transform=ax6.transAxes)
        ax6.axis('off')
        
        plt.tight_layout()
        
        # Save
        plot_path = Path('results/plots/week2_comprehensive_summary.png')
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved comprehensive summary plot to {plot_path}")
        
        return fig


def main():
    print("\n" + "="*70)
    print("DAY 14: WEEK 2 FINAL CHECKPOINT")
    print("="*70)
    print("\nObjective: Consolidate Week 2 findings and prepare for Week 3")
    
    evaluator = Week2Evaluator()
    
    # Run comprehensive evaluation
    evaluator.evaluate_all_datasets()
    
    # Create summary
    evaluator.create_summary_table()
    
    # Generate recommendations
    evaluator.create_recommendations()
    
    # Create visualization
    evaluator.plot_comprehensive_summary()
    
    # Save results
    results_path = Path('results/metrics/week2_final_checkpoint.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(evaluator.results, f, indent=2)
    
    print(f"\n[OK] Saved comprehensive results to {results_path}")
    
    # Week 3 preview
    print("\n" + "="*70)
    print("WEEK 3 PREVIEW (Days 15-21)")
    print("="*70)
    print("""
Planned Topics:
- Day 15: Hybrid methods (combine meta-learning + uncertainty weighting)
- Day 16: Temporal fairness (fairness over time)
- Day 17: Multiple protected attributes
- Day 18: Calibration and fairness
- Day 19: Interpretability analysis
- Day 20: Scalability experiments
- Day 21: Week 3 checkpoint
    """)
    
    print("\n[OK] Week 2 complete!")
    print(f"Progress: 14/30 days (46.7%)")
    print("Status: ON TRACK with strong results ✓")


if __name__ == '__main__':
    main()
