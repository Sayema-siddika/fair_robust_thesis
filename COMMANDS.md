# 🔧 Command Cheat Sheet
## Quick Reference for Daily Tasks

---

## 🚀 DAILY STARTUP

```powershell
# 1. Navigate to project
cd d:\Research\fair_robust_thesis

# 2. Activate environment
conda activate thesis

# 3. Check environment
python --version  # Should show Python 3.8.x

# 4. Open VS Code
code .
```

---

## 📥 DOWNLOAD COMPAS DATASET

```powershell
# Method 1: PowerShell (Recommended)
cd data\raw\compas
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv" -OutFile "compas-scores-two-years.csv"
cd ..\..\..

# Method 2: Manual
# 1. Open browser: https://github.com/propublica/compas-analysis
# 2. Download: compas-scores-two-years.csv
# 3. Save to: data\raw\compas\

# Verify download
Get-ChildItem data\raw\compas\compas-scores-two-years.csv
# Should show: ~800 KB file
```

---

## 🔧 SETUP ENVIRONMENT

```powershell
# Option 1: Use setup script
.\setup.ps1

# Option 2: Manual setup
conda create -n thesis python=3.8 -y
conda activate thesis
pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Verify installation
python -c "import torch, numpy, pandas, sklearn; print('✓ All packages installed!')"
```

---

## 🧪 RUN EXPERIMENTS

```powershell
# Test data loader
python src\data_loader.py

# Test fairness metrics
python src\fairness\metrics.py

# Run baseline experiment
python experiments\01_reproduce_baseline.py

# Later: Run greedy selector
python experiments\02_greedy_selector.py

# Later: Run full system
python experiments\04_full_system.py
```

---

## 📊 VIEW RESULTS

```powershell
# View baseline results
Get-Content results\metrics\baseline_results.txt

# View progress
Get-Content PROGRESS.md

# View today's summary
Get-Content DAY1_SUMMARY.md

# List all result files
Get-ChildItem results\metrics\
```

---

## 💾 GIT COMMANDS

```powershell
# Initialize Git (first time only)
git init
git add .
git commit -m "Initial commit: Day 1 complete"

# Daily commits
git add .
git commit -m "Day X: [Brief description]"

# View commit history
git log --oneline

# Check status
git status

# View changes
git diff

# Create remote (optional)
# 1. Create repo on GitHub
# 2. Add remote:
git remote add origin https://github.com/yourusername/fair_robust_thesis.git
git push -u origin main
```

---

## 📝 DAILY WORKFLOW

```powershell
# Morning routine
cd d:\Research\fair_robust_thesis
conda activate thesis
code .

# Check todo list
Get-Content TODO.md

# Work on tasks...

# Evening routine
python experiments\your_experiment.py  # Run today's experiment
git add .
git commit -m "Day X: Completed [tasks]"

# Update progress
notepad PROGRESS.md  # Add today's progress
```

---

## 🔍 DEBUGGING COMMANDS

```powershell
# Check Python packages
pip list | Select-String "torch|numpy|pandas|scikit"

# Check environment
conda env list

# Reinstall package
pip uninstall package_name
pip install package_name

# Check CUDA (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run with verbose output
python -v script.py

# Profile code
python -m cProfile -s cumtime script.py
```

---

## 📦 PACKAGE MANAGEMENT

```powershell
# Update single package
pip install --upgrade package_name

# Update all packages (careful!)
pip install --upgrade -r requirements.txt

# Freeze current environment
pip freeze > requirements_frozen.txt

# Create environment from file
conda env create -f environment.yml
```

---

## 🗂️ FILE NAVIGATION

```powershell
# List all Python files
Get-ChildItem -Path . -Filter *.py -Recurse

# Find files containing text
Select-String -Path .\src\*.py -Pattern "FairnessMetrics"

# Count lines of code
(Get-Content .\src\data_loader.py).Count

# Total lines in project
(Get-ChildItem -Path .\src\ -Filter *.py -Recurse | Get-Content).Count
```

---

## 📊 QUICK TESTS

```powershell
# Test individual components
python -c "from src.data_loader import DataLoader; print('✓ Data loader imports')"
python -c "from src.fairness.metrics import FairnessMetrics; print('✓ Metrics import')"

# Test PyTorch
python -c "import torch; x = torch.rand(5, 3); print(f'✓ PyTorch works: {x.shape}')"

# Test COMPAS data
python -c "import pandas as pd; df = pd.read_csv('data/raw/compas/compas-scores-two-years.csv'); print(f'✓ COMPAS loaded: {df.shape}')"
```

---

## 🧹 CLEANUP

```powershell
# Remove pycache
Get-ChildItem -Path . -Filter __pycache__ -Recurse | Remove-Item -Recurse -Force

# Remove checkpoints
Remove-Item results\checkpoints\*.pth

# Clean old logs
Remove-Item results\logs\*.log -Older (Get-Date).AddDays(-7)
```

---

## 📈 MONITORING

```powershell
# Watch file changes
Get-ChildItem -Path .\experiments\ -Filter *.py | ForEach-Object { $_.LastWriteTime }

# Monitor GPU (if available)
# Install: choco install gpu-z
gpu-z

# Monitor CPU
Get-Counter '\Processor(_Total)\% Processor Time'

# Check disk space
Get-PSDrive D | Select-Object Used,Free
```

---

## 🎨 VS CODE SHORTCUTS

```
Ctrl + `         : Open terminal
Ctrl + B         : Toggle sidebar
Ctrl + P         : Quick file open
Ctrl + Shift + P : Command palette
Ctrl + /         : Comment line
Ctrl + S         : Save file
Ctrl + Shift + S : Save all
F5               : Run/Debug
```

---

## 🔥 EMERGENCY COMMANDS

```powershell
# Kill all Python processes
Get-Process python | Stop-Process -Force

# Remove environment completely
conda env remove -n thesis

# Start fresh
Remove-Item -Recurse -Force d:\Research\fair_robust_thesis
# Then re-clone/recreate

# Fix corrupted package
pip uninstall torch
pip cache purge
pip install torch==1.13.0
```

---

## 📚 DOCUMENTATION COMMANDS

```powershell
# Generate code documentation
# Install: pip install pdoc3
pdoc --html --output-dir docs src

# View documentation
Start-Process docs\index.html

# Generate README from template
# Create markdown files, then view in VS Code
```

---

## 🧪 TESTING COMMANDS

```powershell
# Run all tests (when implemented)
pytest tests\

# Run specific test file
pytest tests\test_data_loader.py

# Run with coverage
pytest --cov=src tests\

# Verbose output
pytest -v tests\
```

---

## 📊 RESULTS ANALYSIS

```powershell
# Convert results to CSV
python -c "import json, csv; data = json.load(open('results/metrics/baseline_metrics.json')); csv.writer(open('results.csv', 'w')).writerows(data)"

# Plot results (after implementing visualization)
python src\utils\visualization.py

# Generate all plots
python scripts\generate_plots.py
```

---

## 🎯 COMMON WORKFLOWS

### Workflow 1: Morning Start
```powershell
cd d:\Research\fair_robust_thesis
conda activate thesis
code .
git pull  # If using remote
Get-Content TODO.md
```

### Workflow 2: Run Experiment
```powershell
python experiments\my_experiment.py
Get-Content results\metrics\my_results.txt
```

### Workflow 3: End of Day
```powershell
git add .
git commit -m "Day X: Completed tasks"
git push  # If using remote
notepad PROGRESS.md  # Update progress
```

---

## 📞 HELP COMMANDS

```powershell
# Python help
python --help

# Conda help
conda --help

# Pip help
pip --help

# Git help
git --help

# Package info
pip show torch
```

---

## 💡 PRODUCTIVITY TIPS

```powershell
# Create alias for common commands
Set-Alias -Name thesis -Value "cd d:\Research\fair_robust_thesis; conda activate thesis"

# Quick experiment run
function Run-Baseline { python experiments\01_reproduce_baseline.py }

# Auto-commit
function Quick-Commit { 
    param($msg)
    git add .
    git commit -m $msg
}

# Usage:
# thesis
# Run-Baseline
# Quick-Commit "Day 2: Greedy selector"
```

---

## 🎓 LEARNING COMMANDS

```powershell
# View base paper
Start-Process https://arxiv.org/pdf/2110.14222.pdf

# View COMPAS dataset info
Start-Process https://github.com/propublica/compas-analysis

# Search papers
Start-Process "https://scholar.google.com/scholar?q=fair+robust+machine+learning"
```

---

**Quick Reference Card**

| Task | Command |
|------|---------|
| Start work | `cd thesis; conda activate thesis` |
| Run baseline | `python experiments\01_reproduce_baseline.py` |
| View results | `cat results\metrics\baseline_results.txt` |
| Commit work | `git add .; git commit -m "msg"` |
| Update progress | `notepad PROGRESS.md` |

---

**Last Updated**: December 6, 2025
**Keep this file open in VS Code for quick reference!**
