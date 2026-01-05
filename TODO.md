# TODO List - Fair & Robust Thesis

---

## 🎯 IMMEDIATE (Tonight/Tomorrow Morning)

- [ ] Download COMPAS dataset (2 min)
  - URL: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
  - Save to: `data/raw/compas/compas-scores-two-years.csv`

- [ ] Install dependencies (5 min)
  - Run: `.\setup.ps1`
  - OR: Manual installation (see QUICKSTART.md)

- [ ] Verify everything works (3 min)
  - Test: `python src\data_loader.py`
  - Test: `python experiments\01_reproduce_baseline.py`
  - Expected: Accuracy ~65%, EO Disparity ~0.12

---

## 📅 DAY 2 TASKS (December 7)

### Morning (9 AM - 1 PM) - 4 hours

- [ ] **Task 1**: Read base paper (2 hours)
  - Section 3: Problem Formulation (30 min)
  - Section 4: Proposed Method (60 min)
  - Section 5: Experiments (30 min)
  - Take notes in: `notes/base_paper.md`

- [ ] **Task 2**: Implement greedy selector (2 hours)
  - Create: `src/selection/greedy_selector.py`
  - Implement Algorithm 1 from paper
  - Test on COMPAS dataset

### Afternoon (2 PM - 6 PM) - 4 hours

- [ ] **Task 3**: Test greedy vs baseline (1 hour)
  - Create: `experiments/02_greedy_selector.py`
  - Compare results
  - Generate comparison table

- [ ] **Task 4**: Download Adult dataset (1 hour)
  - URL: https://archive.ics.uci.edu/ml/datasets/adult
  - Save to: `data/raw/adult/`
  - Implement `load_adult()` in data_loader.py

- [ ] **Task 5**: Download German dataset (1 hour)
  - URL: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
  - Save to: `data/raw/german/`
  - Implement `load_german()` in data_loader.py

- [ ] **Task 6**: Run baseline on all 3 datasets (1 hour)
  - COMPAS, Adult, German
  - Create comparison table

### Evening (7 PM - 9 PM) - 2 hours

- [ ] **Task 7**: Documentation (1 hour)
  - Update PROGRESS.md
  - Document Day 2 results
  - Note any challenges

- [ ] **Task 8**: Git commit (30 min)
  - `git add .`
  - `git commit -m "Day 2: Greedy selector + multi-dataset"`

- [ ] **Task 9**: Plan Day 3 (30 min)
  - Review 30_DAY_PLAN.md
  - Update TODO.md

---

## 📅 DAY 3 TASKS (December 8)

- [ ] Reproduce base paper Table 1 results
- [ ] Create visualization notebook
- [ ] Start meta-selector architecture design

---

## 📅 WEEK 1 GOALS (Days 1-7)

- [x] Day 1: Project setup + core utilities ✓
- [ ] Day 2: Baseline + greedy selector
- [ ] Day 3: Multi-dataset validation
- [ ] Day 4: Meta-selector architecture
- [ ] Day 5: Synthetic data generation
- [ ] Day 6: Meta-training
- [ ] Day 7: Week 1 checkpoint

**Success Metric**: Reproduce base paper Table 1 (within 2%)

---

## 📅 WEEK 2 GOALS (Days 8-14)

- [ ] Day 8-9: Adaptive α controller
- [ ] Day 10-11: Full system integration
- [ ] Day 12: Uncertainty weighting (optional)
- [ ] Day 13: System optimization
- [ ] Day 14: Week 2 checkpoint

**Success Metric**: Show +1% accuracy, -10% disparity improvement

---

## 📅 WEEK 3 GOALS (Days 15-21)

- [ ] Days 15-16: Main experiments (3 datasets × 5 noise levels)
- [ ] Day 17: Ablation studies
- [ ] Days 18-20: Results analysis & visualization
- [ ] Day 21: Week 3 checkpoint

**Success Metric**: 75 experiments complete, statistical significance shown

---

## 📅 WEEK 4 GOALS (Days 22-30)

- [ ] Days 22-25: Thesis writing (30-40 pages)
- [ ] Days 26-28: Defense preparation (20 slides)
- [ ] Day 29: Rehearsal
- [ ] Day 30: Final submission

**Success Metric**: Thesis complete, confident in defense

---

## 🔧 TECHNICAL DEBT

- [ ] Add unit tests for data_loader.py
- [ ] Add unit tests for fairness/metrics.py
- [ ] Implement logging system
- [ ] Add command-line argument parsing
- [ ] Create config file loader
- [ ] Add progress bars (tqdm)
- [ ] Implement checkpoint saving/loading

---

## 📚 RESEARCH TASKS

- [ ] Read MAML paper (Finn et al. 2017)
- [ ] Read Meta-Weight-Net (Shu et al. 2019)
- [ ] Read FairBatch paper (Roh et al. 2021)
- [ ] Read ITLM paper (Shen & Sanghavi 2019)
- [ ] Survey recent fairness papers (2023-2024)

---

## 📊 EXPERIMENTS TO RUN

### Baseline Experiments
- [x] COMPAS baseline (Day 1)
- [ ] Adult baseline (Day 2)
- [ ] German baseline (Day 2)

### Greedy Selector Experiments
- [ ] COMPAS greedy (Day 2)
- [ ] Adult greedy (Day 3)
- [ ] German greedy (Day 3)

### Meta-Selector Experiments
- [ ] Synthetic data meta-training (Day 6)
- [ ] COMPAS meta-selector (Day 6)
- [ ] Adult meta-selector (Day 7)
- [ ] German meta-selector (Day 7)

### Full System Experiments
- [ ] 3 datasets × 5 noise levels (Days 15-16)
- [ ] Ablation studies (Day 17)
- [ ] Statistical significance tests (Day 18)

---

## 📝 WRITING TASKS

- [ ] Chapter 1: Introduction (4-5 pages) - Day 22
- [ ] Chapter 2: Literature Review (6-8 pages) - Day 22
- [ ] Chapter 3: Methodology (8-10 pages) - Day 23
- [ ] Chapter 4: Experimental Setup (3-4 pages) - Day 24
- [ ] Chapter 5: Results & Analysis (6-8 pages) - Day 24
- [ ] Chapter 6: Conclusion (2-3 pages) - Day 25
- [ ] Abstract - Day 25
- [ ] References - Day 25
- [ ] Acknowledgments - Day 25

---

## 🎨 VISUALIZATION TASKS

- [ ] Main results comparison bar chart
- [ ] Convergence curves (fixed vs adaptive α)
- [ ] Ablation study grouped bar chart
- [ ] Fairness-accuracy scatter plot
- [ ] Pareto front visualization (if implemented)
- [ ] Sample selection heatmap
- [ ] Meta-selector feature importance plot

---

## 🐛 KNOWN ISSUES

- [ ] Intel Arc GPU not detected (using CPU - OK for now)
- [ ] COMPAS dataset not yet downloaded
- [ ] Adult dataset loader not implemented
- [ ] German dataset loader not implemented

---

## 💡 IDEAS / FUTURE WORK

- [ ] Add support for multi-class classification
- [ ] Implement multiple fairness metrics simultaneously
- [ ] Add active learning for sample selection
- [ ] Create web demo using Streamlit
- [ ] Publish code on GitHub with documentation

---

## ✅ COMPLETED TASKS

### Day 1 (December 6)
- [x] Project folder structure created
- [x] requirements.txt created
- [x] .gitignore configured
- [x] README.md written
- [x] src/data_loader.py implemented (262 lines)
- [x] src/fairness/metrics.py implemented (244 lines)
- [x] experiments/01_reproduce_baseline.py implemented (177 lines)
- [x] PROGRESS.md created
- [x] 30_DAY_PLAN.md created
- [x] QUICKSTART.md created
- [x] COMMANDS.md created
- [x] DAY1_SUMMARY.md created

**Total Day 1**: ~700 lines of code, 25+ files created

---

## 🎯 DAILY TARGETS

| Day | Target Tasks | Success Metric |
|-----|-------------|----------------|
| 1 | Setup + core utils | Baseline runs ✓ |
| 2 | Greedy selector | Matches paper |
| 3 | Multi-dataset | 3 datasets work |
| 4 | Meta architecture | Design complete |
| 5 | Synthetic data | 100 tasks generated |
| 6 | Meta-training | Converges |
| 7 | Week 1 checkpoint | Report written |
| ... | ... | ... |
| 30 | Final submission | Thesis complete! |

---

## 🏆 MILESTONES

- [ ] Week 1: Baseline + Meta-selector (25% complete)
- [ ] Week 2: Adaptive + Integration (50% complete)
- [ ] Week 3: Experiments + Analysis (75% complete)
- [ ] Week 4: Writing + Defense (100% complete)

---

## 📞 HELP NEEDED

- [ ] Understand multidimensional knapsack formulation (Day 2)
- [ ] MAML meta-learning algorithm (Day 4)
- [ ] NSGA-II Pareto optimization (Day 12)
- [ ] Statistical significance testing (Day 18)
- [ ] LaTeX thesis template (Day 22)

---

## 🎓 SUPERVISOR MEETINGS

- [ ] Week 1 meeting: Show baseline results
- [ ] Week 2 meeting: Show meta-selector results
- [ ] Week 3 meeting: Show full system results
- [ ] Week 4 meeting: Review thesis draft
- [ ] Defense preparation meeting

---

**Priority Levels**:
- 🔴 Critical (must do today)
- 🟡 Important (do this week)
- 🟢 Nice to have (if time permits)

**Current Priority**: 🔴 Download COMPAS dataset + install dependencies

---

**Last Updated**: December 6, 2025, 11:55 PM
**Next Update**: December 7, 2025, 9:00 PM
