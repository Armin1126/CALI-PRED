"""
coverage_diagnostics.py

Diagnostic suite for analyzing CALI-PRED vs Baseline prediction coverage.
Answers:
1. Is higher ECE due to safe OVER-coverage or mixed/under-covering?
2. Is ECE regression concentrated in low-DTI windows or high-DTI windows?
3. Distinguishes DTI-driven over-caution from systemic base sigma miscalibration.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Tuple, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Local imports
from metrics_engine import (
    calculate_brier_score,
    expected_calibration_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("CoverageDiagnostics")


def analyze_dti_distribution(dti_array: np.ndarray, save_dir: str) -> None:
    """
    Compute raw DTI distribution diagnostics and save a log-scale histogram plot.
    """
    logger.info("Computing raw DTI distribution diagnostics...")

    dti_min = float(np.min(dti_array))
    dti_max = float(np.max(dti_array))
    dti_mean = float(np.mean(dti_array))
    dti_median = float(np.median(dti_array))

    percentiles_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles_vals = np.percentile(dti_array, percentiles_keys)
    pct_dict = dict(zip(percentiles_keys, percentiles_vals))

    frac_below_03 = float(np.mean(dti_array < 0.3))
    frac_below_05 = float(np.mean(dti_array < 0.5))
    frac_below_07 = float(np.mean(dti_array < 0.7))

    print("\n" + "=" * 80)
    print("  RAW DTI DISTRIBUTION DIAGNOSTICS")
    print("=" * 80)
    print(f"Min: {dti_min:.6f} | Max: {dti_max:.6f}")
    print(f"Mean: {dti_mean:.6f} | Median: {dti_median:.6f}")
    print("-" * 80)
    print("Percentiles:")
    for k in percentiles_keys:
        print(f"  {k:2d}th: {pct_dict[k]:.6f}")
    print("-" * 80)
    print(f"Fraction below DTI=0.3: {frac_below_03:.4%} ({frac_below_03:.6f})")
    print(f"Fraction below DTI=0.5: {frac_below_05:.4%} ({frac_below_05:.6f})")
    print(f"Fraction below DTI=0.7: {frac_below_07:.4%} ({frac_below_07:.6f})")
    print("-" * 80)

    if frac_below_05 < 0.01:
        print(
            "[WARNING] Fewer than 1% of test samples represent degraded-trust\n"
            "conditions -- the low-DTI regime this architecture targets is barely\n"
            "present in this test set. Consider increasing missing_rate in\n"
            "compute_dti_for_batch, or checking whether DQA/IRI are structurally\n"
            "reluctant to score real MetroPT windows below ~0.5."
        )
    print("=" * 80 + "\n")

    # Generate log-scale histogram plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dti_array, bins=100, color="tab:purple", edgecolor="none", alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(0.3, color="red", linestyle="--", linewidth=1.5, label="DTI = 0.3")
    ax.axvline(0.5, color="orange", linestyle="--", linewidth=1.5, label="DTI = 0.5")
    ax.axvline(0.7, color="green", linestyle="--", linewidth=1.5, label="DTI = 0.7")
    ax.set_xlabel("Data Trust Index (DTI)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Distribution of Data Trust Index (DTI)")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plot_path = os.path.join(save_dir, "dti_distribution.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Saved DTI distribution plot to '%s'.", plot_path)


def analyze_per_level_coverage(data: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """
    Perform per-level coverage diagnostics and print a comparative gap table.
    """
    logger.info("Part A: Analyzing per-level coverage...")
    nominal_levels: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

    y_true_c = data["calipred_y_true"]
    mu_c = data["calipred_mu"]
    sigma_c = data["calipred_sigma"]

    y_true_b = data["baseline_y_true"]
    mu_b = data["baseline_mu"]
    sigma_b = data["baseline_sigma"]

    _, cov_b, _ = expected_calibration_curve(y_true_b, mu_b, sigma_b, nominal_levels)
    _, cov_c, _ = expected_calibration_curve(y_true_c, mu_c, sigma_c, nominal_levels)

    print("\n" + "=" * 85)
    print("  PART A: PER-LEVEL COVERAGE DIAGNOSTICS")
    print("=" * 85)
    print(f"{'Nominal':>8} | {'Base Cov':>10} | {'Base Gap':>10} | {'Cali Cov':>10} | {'Cali Gap':>10} | {'Under-coverage Flags':<20}")
    print("-" * 85)

    under_b = 0
    under_c = 0

    for i, lvl in enumerate(nominal_levels):
        gap_b = cov_b[i] - lvl
        gap_c = cov_c[i] - lvl

        flag_b = gap_b < 0
        flag_c = gap_c < 0

        if flag_b:
            under_b += 1
        if flag_c:
            under_c += 1

        # Determine flags to print
        flags = []
        if flag_b:
            flags.append("Base")
        if flag_c:
            flags.append("Cali")

        flag_str = ""
        if flags:
            flag_str = f"[UNDER-COVERAGE ({'+'.join(flags)})]"

        print(
            f"{lvl:>8.2f} | {cov_b[i]:>10.4f} | {gap_b:>+10.4f} | "
            f"{cov_c[i]:>10.4f} | {gap_c:>+10.4f} | {flag_str:<20}"
        )

    print("-" * 85)
    print(f"Baseline under-covers at {under_b}/6 levels; CALI-PRED under-covers at {under_c}/6 levels.")
    print("=" * 85)

    return under_b, under_c


def analyze_dti_binned_breakdown(
    data: Dict[str, np.ndarray], binning_mode: str
) -> List[Tuple]:
    """
    Split the dataset into DTI bins and evaluate calibration/CRPS delta.
    """
    logger.info("Part B: Analyzing DTI-binned calibration metrics (%s)...", binning_mode)

    y_true_c = data["calipred_y_true"]
    mu_c = data["calipred_mu"]
    sigma_c = data["calipred_sigma"]
    dti_c = data["calipred_dti"]

    y_true_b = data["baseline_y_true"]
    mu_b = data["baseline_mu"]
    sigma_b = data["baseline_sigma"]

    if binning_mode == "fixed":
        bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        # Quantile mode - 5 equal-count bins
        bin_edges = np.percentile(dti_c, [0, 20, 40, 60, 80, 100])
        # Ensure edges are unique to prevent np.digitize error
        if len(np.unique(bin_edges)) < len(bin_edges):
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.array([np.min(dti_c), np.max(dti_c)])

    # Group by real data DTI from CaliPred
    bin_idx = np.digitize(dti_c, bin_edges[1:-1])
    n_bins = len(bin_edges) - 1

    nominal_levels = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

    print("\n" + "=" * 135)
    print(f"  PART B: DTI-BINNED ECE & CRPS BREAKDOWN ({binning_mode.upper()} BINNING)")
    print("=" * 135)
    print(
        f"{'Bin':<5} | {'DTI Range':<22} | {'Center':>6} | {'Samples':>8} | {'Base ECE':>8} | {'Cali ECE':>8} | "
        f"{'ECE Delta':>9} | {'Base CRPS':>9} | {'Cali CRPS':>9} | {'CRPS Delta':>10}"
    )
    print("-" * 135)

    bin_stats = []

    for b in range(n_bins):
        mask = (bin_idx == b)
        n_samples = np.sum(mask)

        if n_samples > 0:
            actual_min = float(np.min(dti_c[mask]))
            actual_max = float(np.max(dti_c[mask]))
            actual_mean = float(np.mean(dti_c[mask]))
        else:
            actual_min, actual_max, actual_mean = 0.0, 0.0, 0.0

        range_str = f"[{actual_min:.4f}, {actual_max:.4f}]"
        center = actual_mean

        if n_samples < 30:
            print(f"#{b:<3} | {range_str:<22} | {center:>6.2f} | {n_samples:>8} | {'insufficient data':<70}")
            bin_stats.append((center, n_samples, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, range_str, actual_min, actual_max))
            continue

        # CALI-PRED evaluation within bin
        _, _, ece_c = expected_calibration_curve(
            y_true_c[mask], mu_c[mask], sigma_c[mask], nominal_levels
        )
        crps_c = calculate_brier_score(y_true_c[mask], mu_c[mask], sigma_c[mask])

        # Baseline evaluation within bin
        _, _, ece_b = expected_calibration_curve(
            y_true_b[mask], mu_b[mask], sigma_b[mask], nominal_levels
        )
        crps_b = calculate_brier_score(y_true_b[mask], mu_b[mask], sigma_b[mask])

        ece_delta = ece_c - ece_b
        crps_delta = crps_c - crps_b

        print(
            f"#{b:<3} | {range_str:<22} | {center:>6.4f} | {n_samples:>8} | {ece_b:>8.4f} | {ece_c:>8.4f} | "
            f"{ece_delta:>+9.4f} | {crps_b:>9.4f} | {crps_c:>9.4f} | {crps_delta:>+10.4f}"
        )
        bin_stats.append(
            (center, n_samples, ece_b, ece_c, ece_delta, crps_b, crps_c, crps_delta, range_str, actual_min, actual_max)
        )

    print("-" * 135)
    print("=" * 135)

    if binning_mode == "quantile":
        any_small_bin = False
        for stat in bin_stats:
            if stat[1] < 100:
                any_small_bin = True
                break
        if any_small_bin:
            print(
                "[WARNING] Even after equal-count binning, insufficient degraded-trust data\n"
                "exists in the current test set to draw a reliable conclusion about\n"
                "low-DTI behavior specifically."
            )
            print("=" * 135 + "\n")

    return bin_stats


def analyze_coverage_at_nominal_05(
    data: Dict[str, np.ndarray],
    bin_stats: List[Tuple],
    binning_mode: str
) -> None:
    """
    Analyze and print absolute coverage gaps at nominal level 0.50 across bins.
    """
    logger.info("Part A-2: Analyzing absolute coverage at nominal level 0.50...")

    y_true_c = data["calipred_y_true"]
    mu_c = data["calipred_mu"]
    sigma_c = data["calipred_sigma"]
    dti_c = data["calipred_dti"]

    y_true_b = data["baseline_y_true"]
    mu_b = data["baseline_mu"]
    sigma_b = data["baseline_sigma"]

    if binning_mode == "fixed":
        bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        bin_edges = np.percentile(dti_c, [0, 20, 40, 60, 80, 100])
        if len(np.unique(bin_edges)) < len(bin_edges):
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.array([np.min(dti_c), np.max(dti_c)])

    bin_idx = np.digitize(dti_c, bin_edges[1:-1])
    centers = [stat[0] for stat in bin_stats]

    print("\n" + "=" * 95)
    print(f"  PART A-2: ABSOLUTE COVERAGE COMPARISON AT NOMINAL LEVEL 0.50 ({binning_mode.upper()} BINNING)")
    print("=" * 95)
    print(
        f"{'Bin':<5} | {'DTI Range':<22} | {'Center':>6} | {'Base Cov':>8} | {'Cali Cov':>8} | "
        f"{'Base Gap':>8} | {'Cali Gap':>8}"
    )
    print("-" * 95)

    target_lvl = 0.50
    last_bin_base_gap = np.nan
    last_bin_cali_gap = np.nan

    for b in range(len(centers)):
        n_samples = bin_stats[b][1]
        range_str = bin_stats[b][8]
        center = bin_stats[b][0]

        if n_samples < 30:
            print(f"#{b:<3} | {range_str:<22} | {center:>6.2f} | {'insufficient data':<55}")
            continue

        mask = (bin_idx == b)

        _, cov_c, _ = expected_calibration_curve(
            y_true_c[mask], mu_c[mask], sigma_c[mask], [target_lvl]
        )
        _, cov_b, _ = expected_calibration_curve(
            y_true_b[mask], mu_b[mask], sigma_b[mask], [target_lvl]
        )

        cov_val_c = cov_c[0]
        cov_val_b = cov_b[0]

        gap_c = cov_val_c - target_lvl
        gap_b = cov_val_b - target_lvl

        # Keep track of the last bin's values (highest-DTI bin)
        last_bin_base_gap = gap_b
        last_bin_cali_gap = gap_c

        print(
            f"#{b:<3} | {range_str:<22} | {center:>6.4f} | {cov_val_b:>8.4f} | {cov_val_c:>8.4f} | "
            f"{gap_b:>+8.4f} | {gap_c:>+8.4f}"
        )

    print("-" * 95)
    print("=" * 95)

    # Systemic vs DTI-Specific Diagnosis
    print("\n" + "=" * 80)
    print("  SYSTEMIC VS DTI-SPECIFIC SIGMA DIAGNOSIS")
    print("=" * 80)

    if np.isnan(last_bin_base_gap):
        print("[INFO] Insufficient samples in the highest-DTI bin to perform diagnosis.")
    else:
        print(f"Highest DTI bin stats (center={centers[-1]:.4f}):")
        print(f"  Baseline gap at 0.50: {last_bin_base_gap:>+7.4f}")
        print(f"  CALI-PRED gap at 0.50: {last_bin_cali_gap:>+7.4f}")
        print("-" * 80)

        if last_bin_base_gap > 0.15:
            print(
                "[SYSTEMIC ISSUE] Baseline also over-covers substantially on clean, high-trust\n"
                "data. This over-coverage is NOT specific to CALI-PRED's trust mechanism --\n"
                "both models share an underlying sigma calibration issue (check\n"
                "TrustCalibratedLoss's pinball calibration_weight, the softplus base-sigma\n"
                "parameterization, or undertraining of the sigma head)."
            )
        elif last_bin_base_gap < 0.05 and last_bin_cali_gap > 0.15:
            print(
                "[CALI-PRED SPECIFIC] Over-coverage at high DTI is specific to CALI-PRED, not\n"
                "shared by baseline. Check whether the inflation factor\n"
                "(1/(DTI+eps))^alpha still exceeds 1.0x meaningfully even when DTI is close\n"
                "to 1.0 -- inspect the actual alpha value learned and the eps term in\n"
                "predictor.py's forward pass."
            )
        else:
            print("[DIAGNOSIS] Gaps are within standard tolerance range in highest-DTI bin.")
    print("=" * 80 + "\n")


def interpret_results(bin_stats: List[Tuple]) -> None:
    """
    Print structural/architectural interpretation based on ECE delta in high DTI bins.
    """
    print("\n" + "=" * 80)
    print("  INTERPRETATION & DIAGNOSIS")
    print("=" * 80)

    # Filter out empty/invalid bins
    valid_stats = [stat for stat in bin_stats if not np.isnan(stat[4])]

    # Define high DTI bins (centers >= 0.70)
    high_dti_stats = [stat for stat in valid_stats if stat[0] >= 0.70]

    has_high_dti_regression = False
    for center, _, _, _, ece_delta, _, _, _, _, _, _ in high_dti_stats:
        if ece_delta > 0.005:  # threshold of 0.5% ECE increase
            has_high_dti_regression = True
            logger.warning(
                "ECE regression detected at clean DTI bin %s: %+0.4f",
                center, ece_delta,
            )

    if has_high_dti_regression:
        print(
            "[WARNING] CALI-PRED is worse-calibrated than baseline even on clean, high-trust\n"
            "data. This suggests the base sigma head itself is miscalibrated,\n"
            "independent of the trust-inflation mechanism, and needs investigation\n"
            "separate from the DTI mechanism."
        )
    else:
        print(
            "[CONSISTENT WITH DESIGN] ECE regression is concentrated in\n"
            "degraded-data bins, where architectural over-caution is intended."
        )
    print("=" * 80 + "\n")


def generate_plots(
    data: Dict[str, np.ndarray], bin_stats: List[Tuple], save_path: str, binning_mode: str
) -> None:
    """
    Generate the 2-panel diagnostics plot.
    """
    logger.info("Generating diagnostic plots (%s)...", binning_mode)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left Panel: reliability curves per DTI bin (Cali and Base) ---------- #
    ax_left = axes[0]
    nominal_levels = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

    y_true_c = data["calipred_y_true"]
    mu_c = data["calipred_mu"]
    sigma_c = data["calipred_sigma"]
    dti_c = data["calipred_dti"]

    y_true_b = data["baseline_y_true"]
    mu_b = data["baseline_mu"]
    sigma_b = data["baseline_sigma"]

    if binning_mode == "fixed":
        bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        bin_edges = np.percentile(dti_c, [0, 20, 40, 60, 80, 100])
        if len(np.unique(bin_edges)) < len(bin_edges):
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.array([np.min(dti_c), np.max(dti_c)])

    bin_idx = np.digitize(dti_c, bin_edges[1:-1])

    # Reference diagonal
    ax_left.plot([0.5, 0.95], [0.5, 0.95], "k-.", alpha=0.5, label="Perfect Calibration")

    # Define standard colors for up to 5 bins
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for b in range(len(bin_edges) - 1):
        center = bin_stats[b][0]
        n_samples = bin_stats[b][1]

        if n_samples < 30:
            continue

        mask = (bin_idx == b)
        color = colors[b % len(colors)]

        # CaliPred Coverage Curve (Solid line)
        _, cov_c, _ = expected_calibration_curve(
            y_true_c[mask], mu_c[mask], sigma_c[mask], nominal_levels
        )
        ax_left.plot(
            nominal_levels, cov_c, "o-", color=color, linewidth=2.0,
            label=f"Cali (bin #{b}, center={center:.2f})",
        )

        # Baseline Coverage Curve (Dashed line)
        _, cov_b, _ = expected_calibration_curve(
            y_true_b[mask], mu_b[mask], sigma_b[mask], nominal_levels
        )
        ax_left.plot(
            nominal_levels, cov_b, "x--", color=color, linewidth=1.5, alpha=0.8,
            label=f"Base (bin #{b}, center={center:.2f})",
        )

    ax_left.set_title(f"Reliability by DTI Bin ({binning_mode.upper()} Binning)")
    ax_left.set_xlabel("Nominal Confidence Level")
    ax_left.set_ylabel("Empirical Coverage")
    ax_left.set_xlim([0.45, 1.0])
    ax_left.set_ylim([0.45, 1.05])
    ax_left.grid(True, linestyle=":", alpha=0.6)
    ax_left.legend(loc="lower right", fontsize=8)

    # --- Right Panel: Grouped bar chart of ECE and CRPS Deltas --------------- #
    ax_right = axes[1]

    valid_bins = [stat for stat in bin_stats if not np.isnan(stat[4])]
    centers = [stat[0] for stat in valid_bins]
    ece_deltas = [stat[4] for stat in valid_bins]
    crps_deltas = [stat[7] for stat in valid_bins]

    x = np.arange(len(centers))
    width = 0.35

    ax_right.bar(
        x - width/2, ece_deltas, width,
        label="ECE Delta (CaliPred - Base)", color="tab:orange", alpha=0.85,
    )
    ax_right.bar(
        x + width/2, crps_deltas, width,
        label="CRPS Delta (CaliPred - Base)", color="tab:blue", alpha=0.85,
    )

    ax_right.set_ylabel("Delta (CALI-PRED - Baseline)")
    ax_right.set_title("ECE and CRPS Deltas by DTI Bin")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([f"{c:.2f}" for c in centers])
    ax_right.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_right.set_xlabel("DTI Bin Center")
    ax_right.legend()
    ax_right.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved diagnostic plots to '%s'.", save_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CALI-PRED Prediction Coverage & Bin Diagnostics Suite",
    )
    parser.add_argument(
        "--pred-path", type=str, default="checkpoints/test_predictions.npz",
        help="Path to prediction NPZ file.",
    )
    parser.add_argument(
        "--binning", type=str, choices=["fixed", "quantile"], default="quantile",
        help="Binning mode for final plots/interpretation.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pred_path):
        logger.error("Prediction NPZ file not found at '%s'. Run pipeline.py first.", args.pred_path)
        return

    logger.info("Loading predictions from '%s'...", args.pred_path)
    data = np.load(args.pred_path)

    # 1. Raw DTI distribution diagnostics (printed before anything else)
    analyze_dti_distribution(data["calipred_dti"], os.path.dirname(args.pred_path))

    # 2. Part A: Per-level coverage diagnostics
    analyze_per_level_coverage(data)

    # 3. Part B: DTI-binned calibration metrics
    # Re-run both fixed-width and quantile-based tables for direct comparison
    print("\n" + "=" * 135)
    print("  COMPARING BINNING SCHEMES")
    print("=" * 135)
    
    fixed_stats = analyze_dti_binned_breakdown(data, "fixed")
    quantile_stats = analyze_dti_binned_breakdown(data, "quantile")

    # Select stats based on requested CLI mode
    selected_stats = quantile_stats if args.binning == "quantile" else fixed_stats

    # 4. Part A-2: Absolute coverage at nominal level 0.50
    analyze_coverage_at_nominal_05(data, selected_stats, args.binning)

    # 5. Interpretation & Diagnosis
    interpret_results(selected_stats)

    # 6. Generate visual diagnostic plots
    plot_path = os.path.join(
        os.path.dirname(args.pred_path), "coverage_diagnostics.png"
    )
    generate_plots(data, selected_stats, plot_path, args.binning)


if __name__ == "__main__":
    main()
