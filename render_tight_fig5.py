#!/usr/bin/env python3
"""
render_tight_fig5.py

Renders an ultra-compact, tightly cropped Fig. 5:
- Combines CALI-PRED and Baseline side-by-side with a SINGLE shared colorbar.
- Eliminates duplicate y-axis labels and dead whitespace.
- Makes the cell numbers huge (13pt bold white).
- Fits perfectly at full column width in LaTeX without pushing references to page 8!
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.autolayout": False,
})

def main():
    multipliers = [0.30, 0.50, 1.00, 2.50]
    sigma_floors = [0.010, 0.100, 0.200]
    
    calipred_data = np.array([
        [25.0, 25.0, 25.0, 25.0],
        [24.7, 25.0, 24.7, 25.0],
        [24.7, 24.7, 25.0, 25.0],
    ])
    
    baseline_data = np.array([
        [25.0, 25.0, 24.7, 24.7],
        [25.0, 24.7, 24.7, 25.0],
        [25.0, 25.0, 24.7, 24.7],
    ])
    
    # 7.5 x 2.6 inches — ultra-compact aspect ratio that maximizes cell area
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6), dpi=300, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.18, top=0.82, wspace=0.12)
    
    # 1. Left Panel: CALI-PRED
    im1 = ax1.imshow(calipred_data, cmap="YlGnBu", origin="lower", aspect="auto", vmin=4, vmax=25)
    ax1.set_xticks(range(len(multipliers)))
    ax1.set_xticklabels([f"{m:.2f}" for m in multipliers], fontsize=10.5, fontweight="semibold")
    ax1.set_yticks(range(len(sigma_floors)))
    ax1.set_yticklabels([f"{f:.3f}" for f in sigma_floors], fontsize=10.5, fontweight="semibold")
    ax1.set_xlabel("Sigma LR Multiplier", fontsize=11, fontweight="bold", labelpad=4)
    ax1.set_ylabel("Sigma Floor", fontsize=11, fontweight="bold", labelpad=4)
    ax1.set_title("CALI-PRED: Mean Best Epoch", fontsize=11.5, fontweight="bold", pad=5)
    
    for i in range(len(sigma_floors)):
        for j in range(len(multipliers)):
            val = calipred_data[i, j]
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=12.5, fontweight="bold")
            
    # 2. Right Panel: Baseline
    im2 = ax2.imshow(baseline_data, cmap="YlGnBu", origin="lower", aspect="auto", vmin=4, vmax=25)
    ax2.set_xticks(range(len(multipliers)))
    ax2.set_xticklabels([f"{m:.2f}" for m in multipliers], fontsize=10.5, fontweight="semibold")
    ax2.set_xlabel("Sigma LR Multiplier", fontsize=11, fontweight="bold", labelpad=4)
    ax2.set_title("Baseline: Mean Best Epoch", fontsize=11.5, fontweight="bold", pad=5)
    
    for i in range(len(sigma_floors)):
        for j in range(len(multipliers)):
            val = baseline_data[i, j]
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=12.5, fontweight="bold")
            
    # 3. Single shared colorbar on far right
    cbar_ax = fig.add_axes([0.90, 0.18, 0.022, 0.64])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("Best Epoch", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9.5)
    
    out_paths = ["sigma_best_epoch_heatmap.png", "checkpoints/sigma_best_epoch_heatmap.png"]
    for p in out_paths:
        plt.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"[SUCCESS] Saved compact, high-legibility heatmap to: {p}")
        
    plt.close()

if __name__ == "__main__":
    main()
