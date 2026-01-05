# Fair and Robust Training with Meta-Learning

## Author
Sayema Siddika

## Abstract
This thesis addresses the challenge of training machine learning models that are both fair (unbiased across demographic groups) and robust (resilient to label noise). We propose a novel framework that combines:
- Meta-learned sample selection policies
- Adaptive fairness controllers
- Uncertainty-guided weighting
- Pareto optimization for multi-objective fairness-accuracy trade-offs

## Project Structure
```
fair_robust_thesis/
├── data/                  # Datasets (COMPAS, Adult, German)
├── src/                   # Source code
│   ├── models/           # Model architectures
│   ├── training/         # Training pipelines
│   ├── selection/        # Sample selection algorithms
│   ├── fairness/         # Fairness metrics & constraints
│   └── utils/            # Helper functions
├── experiments/          # Experiment scripts
├── notebooks/            # Jupyter notebooks for analysis
├── results/              # Experimental results
│   ├── logs/
│   ├── checkpoints/
│   ├── metrics/
│   └── plots/
└── configs/              # Configuration files
```

## Setup

### Create Conda Environment
```bash
conda create -n thesis python=3.8 -y
conda activate thesis
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download COMPAS Dataset
```bash
cd data/raw
mkdir compas
cd compas
# Download from: https://github.com/propublica/compas-analysis
```

### 2. Run Baseline Experiment
```bash
python experiments/01_reproduce_baseline.py
```

### 3. Train Full System
```bash
python experiments/04_full_system.py
```

## Datasets
- **COMPAS**: Recidivism prediction (7K samples, binary classification)
- **Adult**: Income prediction (48K samples, binary classification)
- **German Credit**: Credit risk (1K samples, binary classification)

## Results
See `results/metrics/` for detailed experimental results.

## Base Paper
Roh et al. "Sample Selection for Fair and Robust Training" (NeurIPS 2021)
- Repository: https://github.com/yuji-roh/fair-robust-selection
