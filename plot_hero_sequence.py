#!/usr/bin/env python3
"""
plot_hero_sequence.py

Loads saved predictions and plots a paper-ready comparison of CALI-PRED vs. Baseline
prediction intervals during a sensor corruption/degradation event (low DTI).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    pred_path = "checkpoints/test_predictions.npz"
    if not os.path.exists(pred_path):
        print(f"[ERROR] No predictions found at {pred_path}. Run pipeline.py first.")
        return

    data = np.load(pred_path)
    
    # Load CALI-PRED predictions
    y_true = data["calipred_y_true"]
    mu_c = data["calipred_mu"]
    sigma_c = data["calipred_sigma"]
    dti_c = data["calipred_dti"]
    
    # Load Baseline predictions
    mu_b = data["baseline_mu"]
    sigma_b = data["baseline_sigma"]

    print(f"Loaded {len(y_true)} timesteps of predictions.")

    # Find a segment where DTI drops significantly (representing corruption)
    # Let's search for a window of size 60 where DTI goes low (< 0.5)
    window_size = 60
    best_start = -1
    min_dti_val = 1.0
    
    # Simple sliding search for a degraded segment
    for i in range(0, len(y_true) - window_size, 10):
        dti_segment = dti_c[i : i + window_size]
        mean_dti = np.mean(dti_segment)
        if mean_dti < min_dti_val and np.min(dti_segment) < 0.4:
            min_dti_val = mean_dti
            best_start = i

    if best_start == -1:
        # Fallback to a default segment if no low DTI is found
        best_start = 100
        print("Warning: No highly degraded DTI segment found. Using fallback starting index 100.")
    else:
        print(f"Found degraded segment starting at index {best_start} with mean DTI={min_dti_val:.3f}")

    start = best_start
    end = start + window_size

    # Slice the data
    t = np.arange(window_size)
    y_true_seg = y_true[start:end]
    
    mu_c_seg = mu_c[start:end]
    sigma_c_seg = sigma_c[start:end]
    dti_seg = dti_c[start:end]
    
    mu_b_seg = mu_b[start:end]
    sigma_b_seg = sigma_b[start:end]

    # Calculate 90% confidence intervals (z = 1.645)
    ci_mult = 1.645
    
    calipred_lower = mu_c_seg - ci_mult * sigma_c_seg
    calipred_upper = mu_c_seg + ci_mult * sigma_c_seg
    
    baseline_lower = mu_b_seg - ci_mult * sigma_b_seg
    baseline_upper = mu_b_seg + ci_mult * sigma_b_seg

    # Create the paper figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # 1. Baseline Panel
    ax1.plot(t, y_true_seg, color="black", label="True Target", linewidth=1.5)
    ax1.plot(t, mu_b_seg, color="tab:red", linestyle="--", label="Baseline Mean", linewidth=1.2)
    ax1.fill_between(t, baseline_lower, baseline_upper, color="tab:red", alpha=0.15, label="Baseline 90% Interval")
    ax1.set_title("Baseline Model Calibration (Blind to DTI)", fontsize=12, fontweight="bold")
    ax1.ylabel = "Sensor Value"
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. CALI-PRED Panel
    ax2.plot(t, y_true_seg, color="black", label="True Target", linewidth=1.5)
    ax2.plot(t, mu_c_seg, color="tab:green", linestyle="--", label="CALI-PRED Mean", linewidth=1.2)
    ax2.fill_between(t, calipred_lower, calipred_upper, color="tab:green", alpha=0.18, label="CALI-PRED 90% Interval")
    ax2.set_title("CALI-PRED Model Calibration (DTI-Aware Uncertainty Inflation)", fontsize=12, fontweight="bold")
    ax2.ylabel = "Sensor Value"
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    # 3. DTI Trajectory Panel
    ax3.plot(t, dti_seg, color="tab:blue", linewidth=2.0, label="Data Trust Indicator (DTI)")
    ax3.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Injected Missingness: Data Trust Indicator (DTI) Response", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Time steps within prediction window", fontsize=10)
    ax3.ylabel = "DTI Score"
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower left")

    plt.tight_layout()
    plot_path = "checkpoints/hero_sequence_calibration.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n[OK] Paper-ready calibration figure saved to '{plot_path}'")

if __name__ == "__main__":
    main()
