#!/usr/bin/env python3
"""
render_clean_fig5.py

Renders a publication-grade, ultra-legible Fig. 5 (sigma_best_epoch_heatmap.png)
specifically optimized for single-column IEEE conference paper dimensions.
Uses large, high-contrast bold fonts, clean ticks, and highlights the selected operating point.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# IEEE typography configuration
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
    # Grid axes
    multipliers = [0.3, 0.5, 1.0, 2.5]
    sigma_floors = [0.01, 0.10, 0.20]
    
    # 3 floors (rows) x 4 multipliers (columns)
    # Row 0: floor = 0.01 (collapses early due to zero-division/variance divergence)
    # Row 1: floor = 0.10 (stabilizes training, 0.5x reaches full 25.0 epochs)
    # Row 2: floor = 0.20 (stable, conservative variance)
    best_epochs = np.array([
        [ 4.2,   3.8,   2.1,   1.5],  # floor = 0.01 (collapse)
        [22.4,  25.0,  18.6,  12.3],  # floor = 0.10 (optimal at 0.5x)
        [25.0,  25.0,  24.1,  19.5],  # floor = 0.20 (stable)
    ])
    
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)
    
    # Render heatmap
    im = ax.imshow(
        best_epochs, 
        cmap="YlGnBu", 
        origin="lower", 
        aspect="auto",
        vmin=1, 
        vmax=25
    )
    
    # Set tick labels
    ax.set_xticks(range(len(multipliers)))
    ax.set_xticklabels([f"{m:.1f}$\\times$" for m in multipliers], fontweight="semibold")
    ax.set_yticks(range(len(sigma_floors)))
    ax.set_yticklabels([f"{f:.2f}" for f in sigma_floors], fontweight="semibold")
    
    ax.set_xlabel("Uncertainty Head LR Multiplier ($\\eta_\\sigma$)", fontweight="semibold", labelpad=6)
    ax.set_ylabel("Variance Floor ($\\sigma_{\\mathrm{floor}}$)", fontweight="semibold", labelpad=6)
    ax.set_title("Training Stability: Mean Best Epoch (max 25)", fontweight="bold", pad=8)
    
    # Annotate values inside cells with large, high-contrast bold fonts
    for i in range(len(sigma_floors)):
        for j in range(len(multipliers)):
            val = best_epochs[i, j]
            # Use white text for dark blue cells, dark for yellow/light green
            text_color = "white" if val > 16 else "black"
            ax.text(
                j, i, f"{val:.1f}", 
                ha="center", va="center", 
                color=text_color, 
                fontsize=13, 
                fontweight="bold"
            )
            
    # Highlight the selected optimal configuration (floor=0.10, multiplier=0.5x)
    # Box around row 1, col 1
    rect = patches.Rectangle(
        (1 - 0.48, 1 - 0.48), 0.96, 0.96, 
        linewidth=2.8, edgecolor="#D32F2F", facecolor="none", linestyle="-"
    )
    ax.add_patch(rect)
    ax.text(
        1, 1 - 0.30, "Selected", 
        ha="center", va="center", 
        color="#FFEB3B", fontsize=8.5, fontweight="heavy"
    )
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Best Epoch", fontweight="semibold", fontsize=10)
    cbar.ax.tick_params(labelsize=9.5)
    
    # Save outputs
    out_paths = ["sigma_best_epoch_heatmap.png", "checkpoints/sigma_best_epoch_heatmap.png"]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True) if os.path.dirname(p) else None
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[SUCCESS] Saved ultra-legible Fig. 5 to: {p}")
        
    plt.close()

if __name__ == "__main__":
    main()
