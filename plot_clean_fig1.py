#!/usr/bin/env python3
"""
plot_clean_fig1.py

Generates a clean, publication-grade Fig. 1 (real_data_calibration_comparison.png)
without any misleading 'ECE reduction: -11.8%' text box.
Allows the empirical reliability curves and DTI interval expansion to speak for themselves.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Professional IEEE style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "figure.autolayout": True,
})

def compute_calibration_data(y_true, mu, sigma, dti):
    nominal_levels = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    empirical_coverage = []
    
    sigma = np.maximum(sigma, 1e-6)
    z_scores = np.abs(y_true - mu) / sigma
    
    for p in nominal_levels:
        z_crit = norm.ppf(0.5 + p / 2.0)
        cov = np.mean(z_scores <= z_crit)
        empirical_coverage.append(cov)
        
    empirical_coverage = np.array(empirical_coverage)
    mean_ece = np.mean(np.abs(nominal_levels - empirical_coverage))
    
    # Stratified 90% interval widths across DTI bins
    bins = np.linspace(0.3, 0.9, 6)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    interval_widths = []
    
    z_90 = norm.ppf(0.95) # 1.645
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (dti >= low) & (dti < high)
        if np.sum(mask) > 0:
            width = 2.0 * z_90 * np.mean(sigma[mask])
            interval_widths.append(width)
        else:
            interval_widths.append(np.nan)
            
    return {
        "nominal_levels": nominal_levels,
        "empirical_coverage": empirical_coverage,
        "mean_ece": mean_ece,
        "quality_bins": bin_centers,
        "interval_widths": np.array(interval_widths),
    }

def main():
    candidate_paths = glob.glob("checkpoints/**/test_predictions.npz", recursive=True)
    if not candidate_paths:
        print("[ERROR] No test_predictions.npz found!")
        return
        
    pred_path = candidate_paths[0]
    print(f"Loading predictions from: {pred_path}")
    data = np.load(pred_path)
    
    y_true = data["calipred_y_true"].flatten()
    dti = data["calipred_dti"].flatten()
    
    cali_res = compute_calibration_data(
        y_true, data["calipred_mu"].flatten(), data["calipred_sigma"].flatten(), dti
    )
    base_res = compute_calibration_data(
        data["baseline_y_true"].flatten(), data["baseline_mu"].flatten(), data["baseline_sigma"].flatten(), data["baseline_dti"].flatten()
    )
    
    fig, (ax_rel, ax_w) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    # ---------------- Left Panel: Reliability Diagram ---------------- #
    diag = np.linspace(0.45, 1.0, 100)
    ax_rel.plot(diag, diag, linestyle="--", color="gray", linewidth=1.5, label="Perfect Calibration")
    
    base_color = "#D32F2F" # Crimson Red
    cali_color = "#2E7D32" # Forest Green
    
    ax_rel.plot(
        base_res["nominal_levels"], base_res["empirical_coverage"],
        marker="o", color=base_color, label=f"Baseline Predictor (ECE = {base_res['mean_ece']:.3f})"
    )
    ax_rel.fill_between(
        base_res["nominal_levels"], base_res["nominal_levels"], base_res["empirical_coverage"],
        color=base_color, alpha=0.12
    )
    
    ax_rel.plot(
        cali_res["nominal_levels"], cali_res["empirical_coverage"],
        marker="s", color=cali_color, label=f"CALIPRED (ECE = {cali_res['mean_ece']:.3f})"
    )
    ax_rel.fill_between(
        cali_res["nominal_levels"], cali_res["nominal_levels"], cali_res["empirical_coverage"],
        color=cali_color, alpha=0.12
    )
    
    ax_rel.set_xlabel("Nominal Confidence Level")
    ax_rel.set_ylabel("Empirical Coverage")
    ax_rel.set_title("Empirical Reliability Diagram")
    ax_rel.set_xlim(0.48, 0.98)
    ax_rel.set_ylim(0.50, 1.01)
    ax_rel.grid(True, linestyle=":", alpha=0.5)
    ax_rel.legend(loc="lower right", frameon=True, framealpha=0.9)
    
    # ---------------- Right Panel: Uncertainty Responsiveness --------- #
    ax_w.plot(
        base_res["quality_bins"], base_res["interval_widths"],
        marker="o", color=base_color, label="Baseline Predictor (Flat)"
    )
    ax_w.plot(
        cali_res["quality_bins"], cali_res["interval_widths"],
        marker="s", color=cali_color, label="CALIPRED (Adaptive)"
    )
    
    # Highlight degraded telemetry regime
    ax_w.axvspan(0.30, 0.55, color="orange", alpha=0.10, zorder=0)
    ax_w.text(
        0.05, 0.88, "Degraded Telemetry\n(Adaptive Interval Expansion)",
        transform=ax_w.transAxes, fontsize=9.5, color="#B71C1C", style="italic", fontweight="semibold"
    )
    
    ax_w.set_xlabel("Data Trust Index (DTI)")
    ax_w.set_ylabel("Mean 90% Interval Width")
    ax_w.set_title("Uncertainty Responsiveness vs. DTI")
    ax_w.grid(True, linestyle=":", alpha=0.5)
    ax_w.legend(loc="upper right", frameon=True, framealpha=0.9)
    
    out_path = "real_data_calibration_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Clean Fig. 1 saved to: {out_path}")

if __name__ == "__main__":
    main()
