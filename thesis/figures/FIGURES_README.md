# Thesis Figures Inventory

## Generated Figures (Publication Quality)

1. **fairness_comparison.pdf/.png**
   - Chapter 4 (Results), Figure 4.1
   - Bar chart comparing EO violations across datasets and methods
   - Shows our method achieves perfect fairness on German

2. **reliability_diagrams.pdf/.png**
   - Chapter 4 (Results), Figure 4.2
   - Side-by-side reliability diagrams (Unweighted vs. Ours)
   - Demonstrates calibration degradation (+388% ECE)

3. **weight_distribution.pdf/.png**
   - Chapter 4 (Results), Figure 4.3
   - Histogram showing weight evolution (Iteration 1 vs. 5)
   - Illustrates mechanism: weights become concentrated

4. **scalability.pdf/.png**
   - Chapter 4 (Results), Figure 4.4
   - Log-log plot of training time vs. sample size
   - Confirms O(n) linear scaling

## Copied from Experimental Results

5. **day18_calibration_fairness.png**
   - Chapter 4 (Results), supplementary
   - Fairness-calibration trade-off scatter plots

6. **day19_interpretability.png**
   - Chapter 4 (Results), mechanism section
   - Weight analysis and confidence distributions

7. **day20_efficiency.png**
   - Chapter 4 (Results), efficiency section
   - Training time and iteration analysis

8. **week3_checkpoint.png**
   - Chapter 4 (Results), summary
   - Comprehensive Week 3 results visualization

## LaTeX Integration

All figures saved in both PDF (vector) and PNG (raster) formats:
- **PDF**: Recommended for LaTeX (scalable, publication quality)
- **PNG**: Backup for compatibility (300 DPI)

Usage in LaTeX:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{figures/fairness_comparison.pdf}
\caption{Fairness comparison across datasets...}
\label{fig:fairness_comparison}
\end{figure}
```
