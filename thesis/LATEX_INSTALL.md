# LaTeX Installation Guide for Thesis Compilation

## Option 1: MiKTeX (Recommended for Windows)

### Download
- Website: https://miktex.org/download
- Choose: **Basic MiKTeX Installer** (64-bit)
- Size: ~200 MB download, ~800 MB installed

### Installation Steps

1. **Run the installer** (`basic-miktex-x64.exe`)

2. **Setup options**:
   - Installation scope: Choose "Anyone who uses this computer" (if admin) or "Only for: [username]"
   - Install location: Default (`C:\Program Files\MiKTeX`) or custom
   - Settings: 
     - **IMPORTANT**: Set "Install missing packages on-the-fly" to **Yes**
     - Preferred paper: A4

3. **Install** (takes 5-10 minutes)

4. **Restart PowerShell** after installation

5. **Verify installation**:
   ```powershell
   pdflatex --version
   # Should output: MiKTeX-pdfTeX 4.x.x ...
   
   biber --version
   # Should output: biber version x.x
   ```

### First Compilation

1. Navigate to thesis directory:
   ```powershell
   cd d:\Research\fair_robust_thesis\thesis
   ```

2. Run compilation script:
   ```powershell
   .\compile.ps1
   ```

3. **First-time package installation**:
   - MiKTeX will download required packages automatically
   - This may take 10-15 minutes on first run
   - Subsequent compilations will be faster (< 1 minute)

4. **Check output**:
   - Success: `main.pdf` created in `thesis/` directory
   - Errors: Check `main.log` for details

## Option 2: TeX Live (Alternative)

### Download
- Website: https://www.tug.org/texlive/
- Choose: **install-tl-windows.exe**
- Size: ~4 GB (full installation)

### Installation
1. Run installer
2. Choose "Install" (full installation recommended)
3. Installation takes 30-60 minutes (downloads all packages)
4. Restart PowerShell
5. Verify: `pdflatex --version`

### Compilation
Same as MiKTeX: `cd thesis; .\compile.ps1`

## Option 3: Overleaf (Online, No Installation)

### Setup
1. Go to: https://www.overleaf.com/
2. Create free account
3. Click "New Project" → "Upload Project"
4. Upload entire `thesis/` directory as ZIP

### Compilation
- Overleaf compiles automatically on save
- Download PDF: Click "Download PDF" button
- **Pros**: No local installation, collaboration features
- **Cons**: Need internet, slower than local compilation

## Required LaTeX Packages

The thesis uses these packages (auto-installed by MiKTeX):

**Core**:
- `amsmath`, `amssymb` - Mathematical symbols and equations
- `biblatex` - Bibliography management (IEEE style)
- `biber` - Bibliography backend
- `graphicx` - Figure inclusion
- `hyperref` - PDF hyperlinks and cross-references

**Algorithms**:
- `algorithm`, `algorithmic` - Pseudocode formatting

**Code Listings**:
- `listings` - Code syntax highlighting

**Tables**:
- `booktabs` - Professional table formatting
- `multirow` - Multi-row table cells

**Figures**:
- `subfig` - Subfigures (side-by-side plots)

**Formatting**:
- `geometry` - Page layout (margins)
- `setspace` - Line spacing (1.5 spacing)
- `fancyhdr` - Headers and footers

## Troubleshooting

### Error: "pdflatex not found"
**Cause**: LaTeX not installed or not in PATH

**Solution**:
1. Install MiKTeX or TeX Live (see above)
2. Restart PowerShell
3. Check PATH: `$env:Path -split ';' | Select-String MiKTeX`
4. If not in PATH, add manually:
   ```powershell
   $env:Path += ";C:\Program Files\MiKTeX\miktex\bin\x64"
   ```

### Error: "File 'biblatex.sty' not found"
**Cause**: Missing package

**Solution**:
1. MiKTeX: Should auto-install. If not, open MiKTeX Console → Packages → Search "biblatex" → Install
2. TeX Live: Run `tlmgr install biblatex`

### Error: "Citation undefined"
**Cause**: Bibliography not compiled

**Solution**:
Run full compilation cycle:
```powershell
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Or use `compile.ps1` which does this automatically.

### Error: "File 'figures/fairness_comparison.pdf' not found"
**Cause**: Figure files missing

**Solution**:
1. Verify figures exist: `ls thesis/figures/`
2. Run figure generation: `python experiments/25_generate_thesis_figures.py`
3. Check path in LaTeX: should be `figures/filename.pdf` (not `thesis/figures/`)

### Error: "! LaTeX Error: Unknown graphics extension: .pdf"
**Cause**: Wrong compilation engine

**Solution**:
- Use `pdflatex` (not `latex`)
- PDF figures require pdflatex

### Warning: "Overfull \hbox"
**Cause**: Text/equation too wide for page

**Solution**:
- Non-critical: PDF still compiles
- Fix: Break long equations with `split` environment or reword text

## Manual Compilation (if script fails)

```powershell
cd d:\Research\fair_robust_thesis\thesis

# Pass 1: Generate aux files
pdflatex -interaction=nonstopmode main.tex

# Pass 2: Process bibliography
biber main

# Pass 3: Resolve references
pdflatex -interaction=nonstopmode main.tex

# Pass 4: Finalize (ensure all refs resolved)
pdflatex -interaction=nonstopmode main.tex

# Clean up auxiliary files
Remove-Item *.aux, *.bbl, *.bcf, *.blg, *.log, *.out, *.toc, *.lof, *.lot, *.run.xml -ErrorAction SilentlyContinue
```

## Expected Output

**Success**:
```
✓ main.pdf created
✓ Size: 2-5 MB (70-80 pages)
✓ Contains: Title, Abstract, 6 chapters, bibliography, figures
```

**Compilation Time**:
- First run: 10-15 minutes (package installation)
- Subsequent runs: 30-60 seconds

## Viewing the PDF

**Windows**:
- Default: Edge browser (opens automatically)
- Recommended: Adobe Acrobat Reader (better rendering)
- Alternative: SumatraPDF (lightweight, auto-refresh)

**PowerShell**:
```powershell
# Open PDF in default viewer
Invoke-Item main.pdf

# Open in specific app
Start-Process "C:\Program Files\Adobe\Acrobat Reader DC\Reader\AcroRd32.exe" main.pdf
```

## Next Steps After Compilation

1. **Review PDF**:
   - Check all figures render correctly
   - Verify tables are formatted properly
   - Ensure citations appear in bibliography
   - Check cross-references (Chapter X, Figure Y)

2. **Fix errors**:
   - Missing figures: Re-run `25_generate_thesis_figures.py`
   - Citation errors: Check `references.bib` syntax
   - Formatting issues: Adjust LaTeX code in chapter files

3. **Polish**:
   - Spell check: Use Word or Grammarly on chapter .tex files
   - Grammar review: Read PDF, note issues
   - Consistency check: Ensure notation uniform across chapters

4. **Final version**:
   - Run `compile.ps1` one last time
   - Verify page count (target: 70-80 pages)
   - Check PDF file size (should be 2-5 MB)
   - Save final version: `main_final.pdf`

---

**Ready to compile?**
```powershell
cd d:\Research\fair_robust_thesis\thesis
.\compile.ps1
```

**Need help?** Check `main.log` for detailed error messages.
