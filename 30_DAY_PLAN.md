# 30-Day Thesis Roadmap
## Fair and Robust Training with Meta-Learning

---

## 🗓️ Complete Daily Schedule

### WEEK 1: Foundation & Baseline (Days 1-7)

#### ✅ Day 1 (December 6) - COMPLETED
**Goal**: Project setup + core utilities
- [x] Environment setup
- [x] Data loader implementation
- [x] Fairness metrics implementation
- [x] Baseline experiment script

**Next Steps**:
1. Download COMPAS dataset
2. Install dependencies: `.\setup.ps1`

---

#### Day 2 (December 7) - Baseline Reproduction

**Morning (9 AM - 1 PM): 4 hours**
```
✓ Download COMPAS dataset
  → https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
  → Save to: data/raw/compas/compas-scores-two-years.csv

✓ Install dependencies
  → Run: .\setup.ps1
  → Verify: python -c "import torch; print(torch.__version__)"

✓ Test data loader
  → Run: python src/data_loader.py
  → Expected: "Dataset loaded: (7214, 53)"

✓ Run baseline experiment
  → Run: python experiments/01_reproduce_baseline.py
  → Expected: Accuracy ~65%, EO Disparity ~0.12
```

**Afternoon (2 PM - 6 PM): 4 hours**
```
✓ Read base paper Section 3 (Problem Formulation)
  → Understand multidimensional knapsack
  → Note key equations (3.1, 3.2, 3.3)

✓ Read base paper Section 4 (Proposed Method)
  → Greedy algorithm (Algorithm 1)
  → Lambda update mechanism (Equation 4.2)
  → Understand FairBatch integration

✓ Implement greedy_selector.py
  → src/selection/greedy_selector.py
  → Reproduce Algorithm 1 from paper
```

**Evening (7 PM - 9 PM): 2 hours**
```
✓ Document progress in PROGRESS.md
✓ Update TODO.md for Day 3
✓ Commit to Git
```

**Success Criteria**:
- Baseline runs successfully
- Accuracy within 2% of paper (Table 1)
- Greedy selector implemented

---

#### Day 3 (December 8) - Multi-Dataset Support

**Morning**:
```
✓ Download Adult dataset
  → UCI ML Repository
  → Implement load_adult() in data_loader.py

✓ Download German Credit dataset
  → UCI ML Repository
  → Implement load_german() in data_loader.py
```

**Afternoon**:
```
✓ Run baseline on all 3 datasets
  → COMPAS, Adult, German
  → Create comparison table

✓ Reproduce base paper Table 1
  → Match results within 2%
  → Document any differences
```

**Evening**:
```
✓ Create visualization notebook
  → notebooks/01_baseline_analysis.ipynb
  → Plot results comparison
```

---

#### Day 4 (December 9) - Meta-Selector Architecture

**Morning**:
```
✓ Research meta-learning architectures
  → Read MAML paper (Finn et al. 2017)
  → Read Meta-Weight-Net (Shu et al. 2019)

✓ Design policy network
  → Input: sample features (loss, confidence, etc.)
  → Output: keep probability [0, 1]
  → Architecture: MLP with 2 hidden layers
```

**Afternoon**:
```
✓ Implement PolicyNetwork class
  → src/models/meta_selector.py
  → Test forward pass

✓ Implement feature extraction
  → Extract: loss, confidence, entropy, group stats
  → Normalize features
```

**Evening**:
```
✓ Design meta-training loss
  → Validation accuracy as objective
  → REINFORCE algorithm for policy gradient
```

---

#### Day 5 (December 10) - Synthetic Data Generation

**Morning**:
```
✓ Implement synthetic dataset generator
  → src/utils/synthetic_generator.py
  → Generate classification tasks with varying:
    - Number of samples
    - Feature dimensions
    - Class balance
    - Noise rates
```

**Afternoon**:
```
✓ Generate 100 synthetic tasks
  → Save to data/synthetic/
  → Verify diversity (plot statistics)

✓ Test meta-selector on synthetic data
  → Sanity check forward/backward pass
```

**Evening**:
```
✓ Implement meta-training loop
  → src/training/meta_trainer.py
  → Episode-based training
```

---

#### Day 6 (December 11) - Meta-Training

**Full Day (Long Run)**:
```
✓ Meta-train on synthetic datasets
  → Run for ~1000 episodes
  → Monitor loss curves
  → Save checkpoints every 100 episodes

✓ Hyperparameter tuning
  → Learning rate: [0.001, 0.01, 0.1]
  → Hidden dims: [32, 64, 128]
  → Select best config

✓ Test on COMPAS
  → Compare with greedy baseline
  → Target: Match or beat greedy
```

**Success Criteria**:
- Meta-selector converges
- Validation loss decreases
- Beats greedy on at least 1 dataset

---

#### Day 7 (December 12) - Week 1 Checkpoint

**Morning**:
```
✓ Evaluate meta-selector on all datasets
  → COMPAS, Adult, German
  → Create results table

✓ Generate comparison plots
  → Greedy vs Meta-learned
  → Convergence curves
```

**Afternoon**:
```
✓ Week 1 report writing
  → Document what worked
  → Document what didn't work
  → Lessons learned

✓ Update PROGRESS.md
✓ Plan Week 2 in detail
```

**Evening**:
```
✓ Review with supervisor (if available)
✓ Prepare questions
✓ Adjust timeline if needed
```

**Deliverables**:
- Baseline working on 3 datasets
- Meta-selector implemented and tested
- Week 1 report (2-3 pages)

---

### WEEK 2: Core Development (Days 8-14)

#### Day 8 - Adaptive Controller Design

**Tasks**:
```
✓ Design adaptive α algorithm
  → Stuck detection: if disparity unchanged for 50 epochs
  → Oscillation detection: if disparity variance > threshold
  → α adjustment: multiply by 1.2 (stuck) or 0.8 (oscillate)

✓ Implement FairnessController class
  → src/models/fairness_controller.py
  → Track disparity history
  → Automatic α adjustment
```

---

#### Day 9 - Adaptive Controller Testing

**Tasks**:
```
✓ Run experiments: fixed α vs adaptive α
✓ Measure convergence speed
✓ Generate convergence plots
✓ Statistical analysis (t-test)
```

**Success Criteria**:
- Adaptive α converges 2-3× faster
- Reaches target disparity in <1000 epochs

---

#### Day 10 - Full System Integration

**Tasks**:
```
✓ Integrate meta-selector + adaptive controller
✓ Create fair_robust_trainer.py
  → Main training pipeline
  → Combines all components

✓ Test end-to-end on COMPAS
✓ Debug integration issues
```

---

#### Day 11 - Uncertainty Weighting (Optional)

**Tasks**:
```
✓ Implement entropy calculation
✓ Design uncertainty-based weighting
  → High loss + Low entropy = Noisy → Remove
  → High loss + High entropy = Hard → Keep

✓ Run experiments
✓ Decide: keep or drop based on results
```

---

#### Day 12 - Pareto Optimization (Optional)

**Tasks**:
```
✓ Research NSGA-II algorithm
✓ Implement Pareto front generation
  → Multiple objectives: accuracy, fairness

✓ Generate 20 models with different trade-offs
✓ Visualize Pareto front

✓ Decide: keep or drop (time-consuming!)
```

---

#### Day 13 - System Optimization

**Tasks**:
```
✓ Profile code (find bottlenecks)
✓ Optimize slow parts
✓ Parallelize if possible

✓ Hyperparameter tuning
  → Grid search on COMPAS
  → Select best configuration
```

---

#### Day 14 - Week 2 Checkpoint

**Deliverables**:
```
✓ Full system working
✓ Initial improvement over baseline
  → Target: +1% accuracy, -10% disparity

✓ Week 2 report
✓ Plan Week 3 experiments
```

---

### WEEK 3: Experimental Evaluation (Days 15-21)

#### Days 15-16 - Main Experiments

**Experimental Matrix**:
```
Datasets: COMPAS, Adult, German (3)
Noise levels: 0%, 5%, 10%, 15%, 20% (5)
Methods: Baseline, ITLM, FairBatch, Base Paper, Ours (5)

Total experiments: 3 × 5 × 5 = 75 runs
```

**Execution Plan**:
```
Day 15:
  ✓ COMPAS × 5 noise levels × 5 methods = 25 runs
  ✓ Adult × 5 noise levels × 5 methods = 25 runs

Day 16:
  ✓ German × 5 noise levels × 5 methods = 25 runs
  ✓ Verify all results saved correctly
```

---

#### Day 17 - Ablation Studies

**Ablation Matrix**:
```
1. Base Paper (baseline for ablation)
2. Base + Meta Selector
3. Base + Adaptive α
4. Base + Meta + Adaptive
5. Full (+ Uncertainty + Pareto)

Run on COMPAS with 10% noise
```

---

#### Day 18 - Results Analysis

**Tasks**:
```
✓ Compile all results into tables
✓ Statistical significance tests
  → t-test comparing our method vs baseline
  → p < 0.05 threshold

✓ Create comparison tables
  → Main results table (like base paper Table 1)
  → Ablation table
```

---

#### Days 19-20 - Visualization

**Figures to Create**:
```
1. Main results comparison (bar chart)
2. Convergence curves (line plot)
3. Ablation study (grouped bar chart)
4. Fairness-accuracy trade-off (scatter plot)
5. Pareto front (if implemented)
6. Sample selection visualization
7. Meta-selector feature importance
```

---

#### Day 21 - Week 3 Checkpoint

**Deliverables**:
```
✓ All experiments complete
✓ All plots generated (publication quality)
✓ Results interpreted
✓ Ready for thesis writing
```

---

### WEEK 4: Thesis Writing & Defense (Days 22-30)

#### Days 22-23 - Chapters 1-3

**Day 22**:
```
✓ Chapter 1: Introduction (4-5 pages)
  → Motivation
  → Problem statement
  → Contributions
  → Organization

✓ Chapter 2: Literature Review (6-8 pages)
  → Fairness in ML
  → Robust training
  → Meta-learning
  → Research gap
```

**Day 23**:
```
✓ Chapter 3: Methodology (8-10 pages)
  → Problem formulation
  → Proposed framework
  → Meta-selector design
  → Adaptive controller
  → Training algorithm
```

---

#### Days 24-25 - Chapters 4-6

**Day 24**:
```
✓ Chapter 4: Experimental Setup (3-4 pages)
  → Datasets description
  → Baselines
  → Implementation details
  → Hyperparameters

✓ Chapter 5: Results & Analysis (6-8 pages)
  → Main results
  → Ablation studies
  → Convergence analysis
  → Discussion
```

**Day 25**:
```
✓ Chapter 6: Conclusion (2-3 pages)
  → Summary
  → Limitations
  → Future work

✓ References (BibTeX)
✓ Abstract
✓ Acknowledgments
```

---

#### Day 26 - Thesis Finalization

**Tasks**:
```
✓ Proofread entire thesis
✓ Check all citations
✓ Verify all figures/tables
✓ Format consistently
✓ Generate PDF
✓ Submit to supervisor for review
```

---

#### Days 27-28 - Defense Preparation

**Day 27**:
```
✓ Create 20 PowerPoint slides
  → Title slide
  → Problem & motivation (2 slides)
  → Literature review (2 slides)
  → Methodology (4 slides)
  → Results (5 slides)
  → Contributions (2 slides)
  → Future work (1 slide)
  → Q&A (3 slides)

✓ Practice presentation (30 min)
```

**Day 28**:
```
✓ Prepare live demo
  → demo.py script
  → Show convergence comparison
  → Show Pareto front

✓ Anticipate questions
  → Why meta-learning?
  → How does adaptive α work?
  → Limitations?
  → Future work?

✓ Practice Q&A
```

---

#### Day 29 - Rehearsal

**Tasks**:
```
✓ Full rehearsal (3 times)
✓ Time yourself (target: 25-30 min)
✓ Record yourself
✓ Improve based on recording
✓ Sleep well!
```

---

#### Day 30 - Final Submission

**Tasks**:
```
✓ Final thesis PDF
✓ Presentation PDF
✓ Demo code ready
✓ Backup everything (Google Drive)
✓ SUBMIT!
```

---

## 📊 Progress Tracking

### Weekly Checkpoints
- **Week 1**: Baseline + Meta-selector (25% complete)
- **Week 2**: Adaptive + Integration (50% complete)
- **Week 3**: Experiments + Analysis (75% complete)
- **Week 4**: Writing + Defense (100% complete)

### Daily Metrics to Track
- [ ] Lines of code written
- [ ] Experiments completed
- [ ] Pages written
- [ ] Hours worked

---

## ⚡ Quick Commands Reference

```bash
# Activate environment
conda activate thesis

# Run baseline
python experiments/01_reproduce_baseline.py

# Run meta-training
python experiments/02_meta_training.py

# Run full system
python experiments/04_full_system.py

# Run all experiments
python experiments/run_all.py

# Generate plots
python src/utils/visualization.py

# Commit progress
git add .; git commit -m "Day X: ..."
```

---

## 🎯 Success Milestones

### Week 1 ✓
- [ ] Baseline accuracy within 2% of paper
- [ ] Meta-selector implemented
- [ ] Beats greedy on 1+ dataset

### Week 2 ✓
- [ ] Adaptive α shows 2× speedup
- [ ] Full system integrated
- [ ] +1% accuracy improvement

### Week 3 ✓
- [ ] 75 experiments complete
- [ ] Statistical significance (p<0.05)
- [ ] 7 publication-quality figures

### Week 4 ✓
- [ ] 30-40 page thesis complete
- [ ] 20-slide presentation ready
- [ ] Confident in defense

---

**Last Updated**: December 6, 2025
**Next Update**: Daily at 9 PM
