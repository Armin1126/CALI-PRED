#!/usr/bin/env python3
"""
render_exact_original_heatmap.py

Renders the EXACT original two-panel heatmap (CALI-PRED and Baseline)
with your exact experimental values (24.7, 25.0, etc.), but fixing the readability issues:
1. Replaces the invisible black text on dark blue with bold, clear WHITE text.
2. Removes the harsh white gridlines that sliced through the numbers.
3. Uses larger, crisp IEEE serif typography.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "figure.autolayout": True,
})

def main():
    multipliers = [0.30, 0.50, 1.00, 2.50]
    sigma_floors = [0.010, 0.100, 0.200]
    
    # Exact original data from your ablation run:
    # Row 0: floor = 0.010, Row 1: floor = 0.100, Row 2: floor = 0.200
    calipred_data = np.array([
        [25.0, 25.0, 25.0, 25.0],  # 0.010
        [24.7, 25.0, 24.7, 25.0],  # 0.100
        [24.7, 24.7, 25.0, 25.0],  # 0.200
    ])
    
    baseline_data = np.array([
        [25.0, 25.0, 24.7, 24.7],  # 0.010
        [25.0, 24.7, 24.7, 25.0],  # 0.100
        [25.0, 25.0, 24.7, 24.7],  # 0.200
    ])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)
    
    # ----------------- Left Panel: CALI-PRED ----------------- #
    im1 = ax1.imshow(calipred_data, cmap="YlGnBu", origin="lower", aspect="auto", vmin=4, vmax=25)
    ax1.set_xticks(range(len(multipliers)))
    ax1.set_xticklabels([f"{m:.2f}" for m in multipliers])
    ax1.set_yticks(range(len(sigma_floors)))
    ax1.set_yticklabels([f"{f:.3f}" for f in sigma_floors])
    ax1.set_xlabel("Sigma LR Multiplier", fontweight="semibold")
    ax1.set_ylabel("Sigma Floor", fontweight="semibold")
    ax1.set_title("CALI-PRED: Mean Best Epoch\n(>4 = training progressed past warmup)", fontweight="bold")
    
    for i in range(len(sigma_floors)):
        for j in range(len(multipliers)):
            val = calipred_data[i, j]
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=11.5, fontweight="bold")
            
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.85)
    cbar1.ax.tick_params(labelsize=9.5)
    
    # ----------------- Right Panel: Baseline ----------------- #
    im2 = ax2.imshow(baseline_data, cmap="YlGnBu", origin="lower", aspect="auto", vmin=4, vmax=25)
    ax2.set_xticks(range(len(multipliers)))
    ax2.set_xticklabels([f"{m:.2f}" for m in multipliers])
    ax2.set_yticks(range(len(sigma_floors)))
    ax2.set_yticklabels([f"{f:.3f}" for f in sigma_floors])
    ax2.set_xlabel("Sigma LR Multiplier", fontweight="semibold")
    ax2.set_ylabel("Sigma Floor", fontweight="semibold")
    ax2.set_title("Baseline: Mean Best Epoch\n(>4 = training progressed past warmup)", fontweight="bold")
    
    for i in range(len(sigma_floors)):
        for j in range(len(multipliers)):
            val = baseline_data[i, j]
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=11.5, fontweight="bold")
            
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.85)
    cbar2.ax.tick_params(labelsize=9.5)
    
    out_paths = ["sigma_best_epoch_heatmap.png", "checkpoints/sigma_best_epoch_heatmap.png"]
    for p in out_paths:
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[SUCCESS] Saved enhanced original heatmap to: {p}")
        
    plt.close()

if __name__ == "__main__":
    main()
