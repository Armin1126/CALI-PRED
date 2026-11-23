#!/usr/bin/env python3
"""
multiseed_coverage_diagnostics.py

Runs multi-seed coverage diagnostics for CALI-PRED vs Baseline across
multiple random seeds (e.g. 42, 123, 456) to compute mean +/- std dev
error bars for DTI-binned ECE, CRPS, and per-level coverage.
"""

import os
import argparse
import csv
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics_engine import expected_calibration_curve, calculate_gaussian_crps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("MultiSeedCoverageDiagnostics")

NOMINAL_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def run_multiseed_diagnostics(pred_paths: list[str], save_dir: str = "checkpoints") -> None:
    logger.info("Starting Multi-Seed Coverage Diagnostics across %d seed runs...", len(pred_paths))

    missing_paths = [p for p in pred_paths if not os.path.exists(p)]
    if missing_paths:
        raise FileNotFoundError(f"Requested seed prediction artifacts are missing: {missing_paths}")
    valid_paths = list(pred_paths)
    if not valid_paths:
        raise ValueError("At least one prediction artifact is required.")

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
    reference_metadata = None
    reference_timestamps = None
    per_seed_rows = []

    for path in valid_paths:
        logger.info("Processing '%s'...", path)
        with np.load(path, allow_pickle=False) as data:
            if "prediction_unit" not in data or str(data["prediction_unit"].item()) != "one_step_forecast":
                raise ValueError(f"{path} is not a one_step_forecast artifact.")
            needed = ("dataset", "feature_names", "window_size", "stride", "forecast_horizon",
                      "missingness_protocol", "clean_fraction", "max_severity",
                      "dqa_enabled", "iri_enabled", "trust_attention_enabled",
                      "uncertainty_inflation_enabled", "d_model", "n_heads", "n_layers",
                      "seed",
                      "test_target_timestamp", "calipred_y_true", "calipred_mu", "calipred_sigma",
                      "calipred_dti", "baseline_y_true", "baseline_mu", "baseline_sigma")
            absent = [key for key in needed if key not in data]
            if absent:
                raise ValueError(f"{path} is missing required metadata/arrays: {absent}")
            metadata = (
                str(data["dataset"].item()), tuple(data["feature_names"].astype(str).tolist()),
                int(data["window_size"].item()), int(data["stride"].item()),
                int(data["forecast_horizon"].item()),
                str(data["missingness_protocol"].item()), float(data["clean_fraction"].item()),
                float(data["max_severity"].item()), bool(data["dqa_enabled"].item()),
                bool(data["iri_enabled"].item()), bool(data["trust_attention_enabled"].item()),
                bool(data["uncertainty_inflation_enabled"].item()), int(data["d_model"].item()),
                int(data["n_heads"].item()), int(data["n_layers"].item()),
            )
            seed_value = int(data["seed"].item())
            timestamps = np.asarray(data["test_target_timestamp"])
            array_keys = ("test_target_timestamp", "calipred_y_true", "calipred_mu", "calipred_sigma",
                          "calipred_dti", "baseline_y_true", "baseline_mu", "baseline_sigma")
            arrays = {key: np.asarray(data[key]) for key in array_keys}
        if reference_metadata is None:
            reference_metadata = metadata
            reference_timestamps = timestamps
        elif metadata != reference_metadata:
            raise ValueError(f"Seed artifact metadata differ: {reference_metadata} vs {metadata} in {path}.")
        elif not np.array_equal(timestamps, reference_timestamps):
            raise ValueError(f"Seed target timestamps do not align with the first artifact: {path}.")

        y_c, mu_c, sig_c, dti_c = (arrays["calipred_y_true"], arrays["calipred_mu"], arrays["calipred_sigma"], arrays["calipred_dti"])
        y_b, mu_b, sig_b = arrays["baseline_y_true"], arrays["baseline_mu"], arrays["baseline_sigma"]
        shapes = {value.shape for value in (y_c, mu_c, sig_c, dti_c, y_b, mu_b, sig_b, timestamps)}
        if len(shapes) != 1:
            raise ValueError(f"Prediction and timestamp shapes differ in {path}: {shapes}.")
        if any(not np.all(np.isfinite(value)) for value in (y_c, mu_c, sig_c, dti_c, y_b, mu_b, sig_b, timestamps)):
            raise ValueError(f"Non-finite prediction artifact values in {path}.")
        if np.any(sig_c <= 0) or np.any(sig_b <= 0):
            raise ValueError(f"Non-positive sigma in {path}.")
        if not np.array_equal(y_c, y_b):
            raise ValueError(f"CALI-PRED and baseline targets are misaligned in {path}.")

        # 1. Per-level coverage
        _, cov_b, _ = expected_calibration_curve(y_b, mu_b, sig_b, NOMINAL_LEVELS)
        _, cov_c, _ = expected_calibration_curve(y_c, mu_c, sig_c, NOMINAL_LEVELS)
        seed_per_level_cov_b.append(cov_b)
        seed_per_level_cov_c.append(cov_c)

        # 2. Quantile DTI binning (5 bins)
        bin_edges = np.linspace(0.0, 1.0, 6)
        bin_idx = np.digitize(np.clip(dti_c, 0.0, 1.0), bin_edges[1:-1])
        n_bins = 5

        b_ece_c = np.full(n_bins, np.nan)
        b_ece_b = np.full(n_bins, np.nan)
        b_crps_c = np.full(n_bins, np.nan)
        b_crps_b = np.full(n_bins, np.nan)
        b_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for b in range(n_bins):
            mask = bin_idx == b
            if np.sum(mask) < 30:
                continue
            _, _, ece_c = expected_calibration_curve(y_c[mask], mu_c[mask], sig_c[mask], NOMINAL_LEVELS)
            _, _, ece_b = expected_calibration_curve(y_b[mask], mu_b[mask], sig_b[mask], NOMINAL_LEVELS)
            crps_c = calculate_gaussian_crps(y_c[mask], mu_c[mask], sig_c[mask])
            crps_b = calculate_gaussian_crps(y_b[mask], mu_b[mask], sig_b[mask])

            b_ece_c[b] = ece_c
            b_ece_b[b] = ece_b
            b_crps_c[b] = crps_c
            b_crps_b[b] = crps_b

        per_seed_rows.append({"seed": seed_value,
                              "coverage_cali": cov_c.tolist(), "coverage_baseline": cov_b.tolist(),
                              "dti_bin_ece_cali": b_ece_c.tolist(), "dti_bin_ece_baseline": b_ece_b.tolist(),
                              "dti_bin_gaussian_crps_cali": b_crps_c.tolist(), "dti_bin_gaussian_crps_baseline": b_crps_b.tolist()})

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
    centers_mean = (np.linspace(0.0, 1.0, 6)[:-1] + np.linspace(0.0, 1.0, 6)[1:]) / 2
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "multiseed_coverage_per_seed.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "coverage_cali", "coverage_baseline",
                               "dti_bin_ece_cali", "dti_bin_ece_baseline",
                               "dti_bin_gaussian_crps_cali", "dti_bin_gaussian_crps_baseline"))
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    logger.info("Saved per-seed metric rows to '%s'.", csv_path)

    print("\n" + "=" * 110)
    print(f"  MULTI-SEED COVERAGE DIAGNOSTICS (N={n_seeds} SEEDS, MEAN +/- STD DEV)")
    print("=" * 110)

    print("\n--- 1. PER-LEVEL EMPIRICAL COVERAGE ---")
    print(f"{'Nominal':>8} | {'Baseline Coverage':>22} | {'CALI-PRED Coverage':>22} | {'Cali Gap (Mean +/- Std)':>25}")
    print("-" * 85)
    for i, lvl in enumerate(NOMINAL_LEVELS):
        b_m, b_s = np.nanmean(cov_b_arr[:, i]), np.nanstd(cov_b_arr[:, i])
        c_m, c_s = np.nanmean(cov_c_arr[:, i]), np.nanstd(cov_c_arr[:, i])
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
        eb_m, eb_s = np.nanmean(ece_b_arr[:, b]), np.nanstd(ece_b_arr[:, b])
        ec_m, ec_s = np.nanmean(ece_c_arr[:, b]), np.nanstd(ece_c_arr[:, b])
        de_m, de_s = np.nanmean(ece_c_arr[:, b] - ece_b_arr[:, b]), np.nanstd(ece_c_arr[:, b] - ece_b_arr[:, b])
        dc_m, dc_s = np.nanmean(crps_c_arr[:, b] - crps_b_arr[:, b]), np.nanstd(crps_c_arr[:, b] - crps_b_arr[:, b])

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
        NOMINAL_LEVELS, np.nanmean(cov_c_arr, axis=0), yerr=np.nanstd(cov_c_arr, axis=0),
        fmt="-o", color="tab:green", label="CALI-PRED (Mean +/- Std)", capsize=4, linewidth=2
    )
    ax1.errorbar(
        NOMINAL_LEVELS, np.nanmean(cov_b_arr, axis=0), yerr=np.nanstd(cov_b_arr, axis=0),
        fmt="--x", color="tab:red", label="Baseline (Mean +/- Std)", capsize=4, linewidth=2
    )
    ax1.plot([0.5, 0.95], [0.5, 0.95], "k--", alpha=0.5, label="Perfect Calibration")
    ax1.set_xlabel("Nominal Confidence Level")
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_title("Multi-Seed Per-Level Empirical Coverage")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Right plot: DTI bin ECE Delta with error bars
    ece_delta_m = np.nanmean(ece_c_arr - ece_b_arr, axis=0)
    ece_delta_s = np.nanstd(ece_c_arr - ece_b_arr, axis=0)
    crps_delta_m = np.nanmean(crps_c_arr - crps_b_arr, axis=0)
    crps_delta_s = np.nanstd(crps_c_arr - crps_b_arr, axis=0)

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

