# Compile LaTeX thesis to PDF
# Run from thesis/ directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Compiling BSc Thesis LaTeX to PDF" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if in correct directory
if (-not (Test-Path "main.tex")) {
    Write-Host "ERROR: main.tex not found!" -ForegroundColor Red
    Write-Host "Please run this script from the thesis/ directory" -ForegroundColor Red
    exit 1
}

# Check for pdflatex
$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
if (-not $pdflatex) {
    Write-Host "ERROR: pdflatex not found!" -ForegroundColor Red
    Write-Host "Please install MiKTeX or TeX Live" -ForegroundColor Red
    exit 1
}

# Check for biber
$biber = Get-Command biber -ErrorAction SilentlyContinue
if (-not $biber) {
    Write-Host "WARNING: biber not found - bibliography may not compile" -ForegroundColor Yellow
}

Write-Host "Step 1/4: First pdflatex pass..." -ForegroundColor Green
pdflatex -interaction=nonstopmode main.tex | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: pdflatex pass 1 had errors" -ForegroundColor Yellow
}

Write-Host "Step 2/4: Running biber for bibliography..." -ForegroundColor Green
if ($biber) {
    biber main | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: biber had errors" -ForegroundColor Yellow
    }
}

Write-Host "Step 3/4: Second pdflatex pass (resolve references)..." -ForegroundColor Green
pdflatex -interaction=nonstopmode main.tex | Out-Null

Write-Host "Step 4/4: Third pdflatex pass (finalize)..." -ForegroundColor Green
pdflatex -interaction=nonstopmode main.tex | Out-Null

# Check if PDF was created
if (Test-Path "main.pdf") {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  SUCCESS: Thesis compiled!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    $pdfInfo = Get-Item main.pdf
    Write-Host "Output file: main.pdf" -ForegroundColor Cyan
    Write-Host "Size: $([math]::Round($pdfInfo.Length / 1MB, 2)) MB" -ForegroundColor Cyan
    Write-Host "Modified: $($pdfInfo.LastWriteTime)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Cleaning up auxiliary files..." -ForegroundColor Yellow
    
    # Clean up auxiliary files
    $auxFiles = @("*.aux", "*.bbl", "*.bcf", "*.blg", "*.log", "*.out", "*.toc", "*.lof", "*.lot", "*.run.xml", "*.fls", "*.fdb_latexmk", "*.synctex.gz")
    foreach ($pattern in $auxFiles) {
        Remove-Item $pattern -ErrorAction SilentlyContinue
    }
    
    Write-Host "Done! You can now open main.pdf" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "ERROR: main.pdf was not created!" -ForegroundColor Red
    Write-Host "Check main.log for detailed error messages" -ForegroundColor Red
    exit 1
}
