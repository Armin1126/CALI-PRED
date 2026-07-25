#!/usr/bin/env python3
"""
multiseed_coverage_diagnostics.py

Runs multi-seed coverage diagnostics for CALI-PRED vs Baseline across
multiple random seeds (e.g. 42, 123, 456) to compute mean +/- std dev
error bars for DTI-binned ECE, CRPS, and per-level coverage.
"""

import os
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics_engine import expected_calibration_curve, calculate_brier_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("MultiSeedCoverageDiagnostics")

NOMINAL_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def run_multiseed_diagnostics(pred_paths: list[str], save_dir: str = "checkpoints") -> None:
    logger.info("Starting Multi-Seed Coverage Diagnostics across %d seed runs...", len(pred_paths))

    valid_paths = [p for p in pred_paths if os.path.exists(p)]
    if not valid_paths:
        logger.error("No valid prediction files found among %s", pred_paths)
        return

    n_seeds = len(valid_paths)
    logger.info("Found %d valid seed prediction files.", n_seeds)

    # Accumulators for seed results
    seed_per_level_cov_c = []  # shape: (n_seeds, len(NOMINAL_LEVELS))
    seed_per_level_cov_b = []
    
    # 5 quantile bins
    seed_bin_ece_c = []  # shape: (n_seeds, 5)
    seed_bin_ece_b = []
    seed_bin_crps_c = []
    seed_bin_crps_b = []
    seed_bin_centers = []

    for path in valid_paths:
        logger.info("Processing '%s'...", path)
        data = np.load(path)

        y_c, mu_c, sig_c, dti_c = (
            data["calipred_y_true"],
            data["calipred_mu"],
            np.maximum(data["calipred_sigma"], 1e-8),
            data["calipred_dti"],
        )
        y_b, mu_b, sig_b = (
            data["baseline_y_true"],
            data["baseline_mu"],
            np.maximum(data["baseline_sigma"], 1e-8),
        )

        # 1. Per-level coverage
        _, cov_b, _ = expected_calibration_curve(y_b, mu_b, sig_b, NOMINAL_LEVELS)
        _, cov_c, _ = expected_calibration_curve(y_c, mu_c, sig_c, NOMINAL_LEVELS)
        seed_per_level_cov_b.append(cov_b)
        seed_per_level_cov_c.append(cov_c)

        # 2. Quantile DTI binning (5 bins)
        bin_edges = np.percentile(dti_c, [0, 20, 40, 60, 80, 100])
        if len(np.unique(bin_edges)) < len(bin_edges):
            bin_edges = np.unique(bin_edges)

        bin_idx = np.digitize(dti_c, bin_edges[1:-1])
        n_bins = len(bin_edges) - 1

        b_ece_c, b_ece_b, b_crps_c, b_crps_b, b_centers = [], [], [], [], []

        for b in range(n_bins):
            mask = bin_idx == b
            if np.sum(mask) < 30:
                continue

            center = float(np.mean(dti_c[mask]))
            _, _, ece_c = expected_calibration_curve(y_c[mask], mu_c[mask], sig_c[mask], NOMINAL_LEVELS)
            _, _, ece_b = expected_calibration_curve(y_b[mask], mu_b[mask], sig_b[mask], NOMINAL_LEVELS)
            crps_c = calculate_brier_score(y_c[mask], mu_c[mask], sig_c[mask])
            crps_b = calculate_brier_score(y_b[mask], mu_b[mask], sig_b[mask])

            b_centers.append(center)
            b_ece_c.append(ece_c)
            b_ece_b.append(ece_b)
            b_crps_c.append(crps_c)
            b_crps_b.append(crps_b)

        seed_bin_centers.append(b_centers)
        seed_bin_ece_c.append(b_ece_c)
        seed_bin_ece_b.append(b_ece_b)
        seed_bin_crps_c.append(b_crps_c)
        seed_bin_crps_b.append(b_crps_b)

    # Convert to arrays
    cov_b_arr = np.array(seed_per_level_cov_b)  # (n_seeds, 6)
    cov_c_arr = np.array(seed_per_level_cov_c)

    ece_c_arr = np.array(seed_bin_ece_c)  # (n_seeds, 5)
    ece_b_arr = np.array(seed_bin_ece_b)
    crps_c_arr = np.array(seed_bin_crps_c)
    crps_b_arr = np.array(seed_bin_crps_b)
    centers_mean = np.mean(seed_bin_centers, axis=0)

    print("\n" + "=" * 110)
    print(f"  MULTI-SEED COVERAGE DIAGNOSTICS (N={n_seeds} SEEDS, MEAN +/- STD DEV)")
    print("=" * 110)

    print("\n--- 1. PER-LEVEL EMPIRICAL COVERAGE ---")
    print(f"{'Nominal':>8} | {'Baseline Coverage':>22} | {'CALI-PRED Coverage':>22} | {'Cali Gap (Mean +/- Std)':>25}")
    print("-" * 85)
    for i, lvl in enumerate(NOMINAL_LEVELS):
        b_m, b_s = np.mean(cov_b_arr[:, i]), np.std(cov_b_arr[:, i])
        c_m, c_s = np.mean(cov_c_arr[:, i]), np.std(cov_c_arr[:, i])
        gap_m = c_m - lvl
        print(f"{lvl:>8.2f} | {b_m:>10.4f} +/- {b_s:<6.4f} | {c_m:>10.4f} +/- {c_s:<6.4f} | {gap_m:>+12.4f}")

    print("\n--- 2. DTI-BINNED BREAKDOWN (ECE & CRPS WITH ERROR BARS) ---")
    print(
        f"{'Bin':<4} | {'DTI Center':>10} | {'Base ECE':>18} | {'Cali ECE':>18} | "
        f"{'Delta ECE (C-B)':>18} | {'Delta CRPS (C-B)':>18}"
    )
    print("-" * 110)
    for b in range(len(centers_mean)):
        ctr = centers_mean[b]
        eb_m, eb_s = np.mean(ece_b_arr[:, b]), np.std(ece_b_arr[:, b])
        ec_m, ec_s = np.mean(ece_c_arr[:, b]), np.std(ece_c_arr[:, b])
        de_m, de_s = np.mean(ece_c_arr[:, b] - ece_b_arr[:, b]), np.std(ece_c_arr[:, b] - ece_b_arr[:, b])
        dc_m, dc_s = np.mean(crps_c_arr[:, b] - crps_b_arr[:, b]), np.std(crps_c_arr[:, b] - crps_b_arr[:, b])

        print(
            f"#{b:<3} | {ctr:>10.4f} | {eb_m:>8.4f} +/- {eb_s:<6.4f} | {ec_m:>8.4f} +/- {ec_s:<6.4f} | "
            f"{de_m:>+8.4f} +/- {de_s:<6.4f} | {dc_m:>+8.4f} +/- {dc_s:<6.4f}"
        )
    print("=" * 110 + "\n")

    # Save Multi-Seed Error Bar Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left plot: Reliability curves with shaded error bounds
    for i, lvl in enumerate(NOMINAL_LEVELS):
        pass

    ax1.errorbar(
        NOMINAL_LEVELS, np.mean(cov_c_arr, axis=0), yerr=np.std(cov_c_arr, axis=0),
        fmt="-o", color="tab:green", label="CALI-PRED (Mean +/- Std)", capsize=4, linewidth=2
    )
    ax1.errorbar(
        NOMINAL_LEVELS, np.mean(cov_b_arr, axis=0), yerr=np.std(cov_b_arr, axis=0),
        fmt="--x", color="tab:red", label="Baseline (Mean +/- Std)", capsize=4, linewidth=2
    )
    ax1.plot([0.5, 0.95], [0.5, 0.95], "k--", alpha=0.5, label="Perfect Calibration")
    ax1.set_xlabel("Nominal Confidence Level")
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_title("Multi-Seed Per-Level Empirical Coverage")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Right plot: DTI bin ECE Delta with error bars
    ece_delta_m = np.mean(ece_c_arr - ece_b_arr, axis=0)
    ece_delta_s = np.std(ece_c_arr - ece_b_arr, axis=0)
    crps_delta_m = np.mean(crps_c_arr - crps_b_arr, axis=0)
    crps_delta_s = np.std(crps_c_arr - crps_b_arr, axis=0)

    x = np.arange(len(centers_mean))
    width = 0.35
    ax2.bar(x - width/2, ece_delta_m, width, yerr=ece_delta_s, label="ECE Delta (Cali - Base)", color="tab:orange", capsize=4)
    ax2.bar(x + width/2, crps_delta_m, width, yerr=crps_delta_s, label="CRPS Delta (Cali - Base)", color="tab:blue", capsize=4)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c:.2f}" for c in centers_mean])
    ax2.set_xlabel("DTI Bin Center")
    ax2.set_ylabel("Delta (CALI-PRED - Baseline)")
    ax2.set_title("DTI-Binned Metric Deltas with Multi-Seed Error Bars")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "multiseed_coverage_diagnostics.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info("Saved multi-seed diagnostic plot to '%s'.", plot_path)


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Coverage Diagnostics")
    parser.add_argument(
        "--pred-paths", nargs="+",
        default=[
            "checkpoints/test_predictions.npz",
            "checkpoints/test_predictions_seed123.npz",
            "checkpoints/test_predictions_seed456.npz",
        ],
        help="List of prediction NPZ file paths.",
    )
    args = parser.parse_args()
    run_multiseed_diagnostics(args.pred_paths)


if __name__ == "__main__":
    main()
