# Setup script for thesis environment

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Fair & Robust Thesis - Setup Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create conda environment
Write-Host "[1/5] Creating conda environment..." -ForegroundColor Yellow
conda create -n thesis python=3.8 -y
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Conda environment created" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to create environment" -ForegroundColor Red
    exit 1
}

# Step 2: Activate environment
Write-Host "`n[2/5] Activating environment..." -ForegroundColor Yellow
conda activate thesis
Write-Host "✓ Environment activated" -ForegroundColor Green

# Step 3: Install PyTorch
Write-Host "`n[3/5] Installing PyTorch..." -ForegroundColor Yellow
pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PyTorch installed" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install PyTorch" -ForegroundColor Red
}

# Step 4: Install other dependencies
Write-Host "`n[4/5] Installing other dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
}

# Step 5: Verify installation
Write-Host "`n[5/5] Verifying installation..." -ForegroundColor Yellow
python -c "import torch; import numpy; import pandas; import sklearn; print('✓ All packages imported successfully')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Installation verified" -ForegroundColor Green
} else {
    Write-Host "✗ Verification failed" -ForegroundColor Red
}

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Download COMPAS dataset: cd data\raw\compas" -ForegroundColor White
Write-Host "2. Run baseline: python experiments\01_reproduce_baseline.py" -ForegroundColor White
Write-Host ""
