"""
bootstrap_analysis.py

Statistical rigor for CALI-PRED real-data results:
    1. Bootstrap confidence intervals on ECE and CRPS
    2. Multi-severity DTI sweep across corruption levels and seeds

This script operates on saved predictions from pipeline.py (stored in
checkpoints/test_predictions.npz), so it does NOT require re-training.
For the multi-severity sweep, it requires model checkpoints + data.

Usage
-----
    # Bootstrap CI only (from saved predictions):
    python bootstrap_analysis.py --bootstrap-only

    # Full analysis (bootstrap + multi-severity sweep):
    python bootstrap_analysis.py --data-path data/metropt/MetroPT3(chiller).csv

    # Custom parameters:
    python bootstrap_analysis.py --n-bootstrap 5000 --severity-seeds 42 123 456

Python: 3.13+
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("BootstrapAnalysis")


# ------------------------------------------------------------------ #
# Import metric functions from the existing codebase
# ------------------------------------------------------------------ #
from metrics_engine import expected_calibration_curve, calculate_brier_score


# ------------------------------------------------------------------ #
# Bootstrap CI computation
# ------------------------------------------------------------------ #
def bootstrap_metric(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    metric_fn: str,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    window_size: int = 60,
    n_features: int = 15,
) -> dict:
    """
    Compute bootstrap confidence intervals for a calibration metric.

    Parameters
    ----------
    y_true, mu, sigma : np.ndarray, shape (N,)
        Ground truth, predicted mean, predicted std.
    metric_fn : str
        One of "ece" or "crps".
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level for the interval (e.g., 0.95 for 95% CI).
    window_size : int
        Size of prediction windows.
    n_features : int
        Number of features/channels.

    Returns
    -------
    dict with keys: "point_estimate", "mean", "ci_lower", "ci_upper",
    "std", "all_samples".
    """
    if rng is None:
        rng = np.random.default_rng(42)

    block_size = window_size * n_features
    N = len(y_true)
    N_windows = N // block_size

    # Reshape arrays to (N_windows, block_size)
    y_true_reshaped = y_true[:N_windows * block_size].reshape(N_windows, block_size)
    mu_reshaped = mu[:N_windows * block_size].reshape(N_windows, block_size)
    sigma_reshaped = sigma[:N_windows * block_size].reshape(N_windows, block_size)

    samples = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        # Draw window indices with replacement (block bootstrap)
        idx = rng.integers(0, N_windows, size=N_windows)
        y_b = y_true_reshaped[idx].flatten()
        mu_b = mu_reshaped[idx].flatten()
        sigma_b = sigma_reshaped[idx].flatten()

        # Guard against degenerate bootstrap samples
        if np.any(sigma_b <= 0):
            sigma_b = np.maximum(sigma_b, 1e-8)

        if metric_fn == "ece":
            _, _, ece = expected_calibration_curve(y_b, mu_b, sigma_b)
            samples[b] = ece
        elif metric_fn == "crps":
            samples[b] = calculate_brier_score(y_b, mu_b, sigma_b)
        else:
            raise ValueError(f"Unknown metric_fn: {metric_fn}")

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))

    # Point estimate on full data
    if metric_fn == "ece":
        _, _, point = expected_calibration_curve(y_true, mu, sigma)
    else:
        point = calculate_brier_score(y_true, mu, sigma)

    return {
        "point_estimate": float(point),
        "mean": float(np.mean(samples)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(samples)),
        "all_samples": samples,
    }


def bootstrap_difference(
    y_true: np.ndarray,
    mu_a: np.ndarray, sigma_a: np.ndarray,
    mu_b: np.ndarray, sigma_b: np.ndarray,
    metric_fn: str,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    window_size: int = 60,
    n_features: int = 15,
) -> dict:
    """
    Bootstrap the *difference* (A - B) in a metric using a window block bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    block_size = window_size * n_features
    N = len(y_true)
    N_windows = N // block_size

    # Reshape arrays to (N_windows, block_size)
    y_true_reshaped = y_true[:N_windows * block_size].reshape(N_windows, block_size)
    mu_a_reshaped = mu_a[:N_windows * block_size].reshape(N_windows, block_size)
    sigma_a_reshaped = sigma_a[:N_windows * block_size].reshape(N_windows, block_size)
    mu_b_reshaped = mu_b[:N_windows * block_size].reshape(N_windows, block_size)
    sigma_b_reshaped = sigma_b[:N_windows * block_size].reshape(N_windows, block_size)

    diffs = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        # Draw window indices with replacement (block bootstrap)
        idx = rng.integers(0, N_windows, size=N_windows)
        y_b = y_true_reshaped[idx].flatten()
        mu_a_b = mu_a_reshaped[idx].flatten()
        sigma_a_b = np.maximum(sigma_a_reshaped[idx].flatten(), 1e-8)
        mu_b_b = mu_b_reshaped[idx].flatten()
        sigma_b_b = np.maximum(sigma_b_reshaped[idx].flatten(), 1e-8)

        if metric_fn == "ece":
            _, _, val_a = expected_calibration_curve(y_b, mu_a_b, sigma_a_b)
            _, _, val_b = expected_calibration_curve(y_b, mu_b_b, sigma_b_b)
        else:
            val_a = calculate_brier_score(y_b, mu_a_b, sigma_a_b)
            val_b = calculate_brier_score(y_b, mu_b_b, sigma_b_b)

        diffs[b] = val_a - val_b

    alpha = 1.0 - confidence
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(diffs, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        "p_positive": float(np.mean(diffs > 0)),
        "p_negative": float(np.mean(diffs < 0)),
    }


# ------------------------------------------------------------------ #
# Main analysis
# ------------------------------------------------------------------ #
def run_bootstrap_analysis(
    pred_path: str,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    window_size: int = 60,
    n_features: int = 15,
) -> dict:
    """Run bootstrap CI analysis on saved predictions using a block window bootstrap."""

    data = np.load(pred_path)
    y_true_c = data["calipred_y_true"]
    mu_c = data["calipred_mu"]
    sigma_c = data["calipred_sigma"]
    y_true_b = data["baseline_y_true"]
    mu_b = data["baseline_mu"]
    sigma_b = data["baseline_sigma"]

    # Verify y_true is the same for both
    assert np.allclose(y_true_c, y_true_b), (
        "CALI-PRED and Baseline must be evaluated on the same test set."
    )
    y_true = y_true_c
    N = len(y_true)

    print("\n" + "=" * 70)
    print("  WINDOW-BLOCK BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS")
    print(f"  N={N} samples ({N // (window_size * n_features)} window blocks), B={n_bootstrap} resamples, {confidence:.0%} CI")
    print("=" * 70)

    rng = np.random.default_rng(42)
    results = {}

    for metric in ["ece", "crps"]:
        label = "ECE" if metric == "ece" else "CRPS"

        # Individual CIs
        ci_cali = bootstrap_metric(y_true, mu_c, sigma_c, metric, n_bootstrap, confidence, rng, window_size, n_features)
        ci_base = bootstrap_metric(y_true, mu_b, sigma_b, metric, n_bootstrap, confidence, rng, window_size, n_features)

        # Difference CI (CALI-PRED - Baseline)
        ci_diff = bootstrap_difference(
            y_true, mu_c, sigma_c, mu_b, sigma_b, metric, n_bootstrap, confidence, rng, window_size, n_features
        )

        results[metric] = {
            "calipred": ci_cali,
            "baseline": ci_base,
            "difference": ci_diff,
        }

        print(f"\n--- {label} ---")
        print(f"  Baseline:   {ci_base['point_estimate']:.4f}  "
              f"[{ci_base['ci_lower']:.4f}, {ci_base['ci_upper']:.4f}]")
        print(f"  CALI-PRED:  {ci_cali['point_estimate']:.4f}  "
              f"[{ci_cali['ci_lower']:.4f}, {ci_cali['ci_upper']:.4f}]")
        print(f"  Difference (CALI-PRED - Baseline): "
              f"{ci_diff['mean_diff']:+.4f}  "
              f"[{ci_diff['ci_lower']:+.4f}, {ci_diff['ci_upper']:+.4f}]")

        # Interpret significance
        if ci_diff["ci_lower"] > 0:
            print(f"  -> CALI-PRED {label} is SIGNIFICANTLY HIGHER (worse for ECE)")
        elif ci_diff["ci_upper"] < 0:
            print(f"  -> CALI-PRED {label} is SIGNIFICANTLY LOWER (better)")
        else:
            print(f"  -> Difference is NOT statistically significant (CI includes 0)")

        print(f"  P(CALI-PRED > Baseline): {ci_diff['p_positive']:.1%}")
        print(f"  P(CALI-PRED < Baseline): {ci_diff['p_negative']:.1%}")

    print("\n" + "=" * 70)
    return results


def run_severity_sweep(
    data_path: str,
    dataset: str,
    checkpoint_dir: str,
    missing_rates: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
    seeds: Sequence[int] = (42, 123, 456, 789, 1337),
    max_windows: int = 200,
) -> dict:
    """
    Re-evaluate at multiple corruption severities and MAR seeds.

    Requires trained model checkpoints and data access.
    """
    import torch
    from data_loader import IndustrialDataLoader, create_dataloaders
    from dqa_module import UpstreamDQAEngine
    from fusion_engine import TrustFusionEngine
    from iri_module import ImputationReliabilityEngine
    from predictor import CaliPredTransformer
    from pipeline import compute_dti_for_batch, evaluate_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_ds, _, test_ds, _, _, test_loader = create_dataloaders(
        dataset_name=dataset, file_path=data_path,
        window_size=60, stride=10, forecast_horizon=1,
        batch_size=32, random_state=42,
    )
    n_features = train_ds.n_features

    # Limit windows
    if max_windows and len(test_ds) > max_windows:
        from torch.utils.data import Subset, DataLoader
        test_loader = DataLoader(
            Subset(test_ds, range(max_windows)),
            batch_size=32, shuffle=False,
        )

    # Baseline correlation
    baseline_corr = np.corrcoef(train_ds.X.T)
    baseline_corr = np.nan_to_num(baseline_corr, nan=0.0)

    # Load models
    ckpt_cali = os.path.join(checkpoint_dir, "best_model_calipred.pt")
    ckpt_base = os.path.join(checkpoint_dir, "best_model_baseline.pt")

    model_cali = CaliPredTransformer(
        input_dim=n_features, output_dim=n_features,
        d_model=64, n_heads=4, n_layers=3,
    ).to(device)
    model_base = CaliPredTransformer(
        input_dim=n_features, output_dim=n_features,
        d_model=64, n_heads=4, n_layers=3,
    ).to(device)

    if os.path.exists(ckpt_cali):
        model_cali.load_state_dict(
            torch.load(ckpt_cali, map_location=device, weights_only=False)["model_state_dict"]
        )
    else:
        logger.warning("No CALI-PRED checkpoint at '%s'.", ckpt_cali)
        return {}

    if os.path.exists(ckpt_base):
        model_base.load_state_dict(
            torch.load(ckpt_base, map_location=device, weights_only=False)["model_state_dict"]
        )
    else:
        logger.warning("No Baseline checkpoint at '%s'.", ckpt_base)
        return {}

    print("\n" + "=" * 70)
    print("  MULTI-SEVERITY DTI SWEEP")
    print(f"  Rates: {list(missing_rates)}, Seeds: {list(seeds)}")
    print("=" * 70)

    sweep_results = {}

    for rate in missing_rates:
        rate_results = {"ece_cali": [], "ece_base": [],
                        "crps_cali": [], "crps_base": []}

        for seed in seeds:
            # Re-create engines with this seed
            dqa_engine = UpstreamDQAEngine(freshness_tau_seconds=60.0, max_corr_mae=0.5)
            iri_engine = ImputationReliabilityEngine(
                n_features=n_features, epochs=30, holdout_frac=0.15, random_state=seed,
            )
            fusion_engine = TrustFusionEngine(clamp_inputs=True)
            corruption_loader = IndustrialDataLoader(random_state=seed)

            # Evaluate CALI-PRED
            res_cali = evaluate_model(
                model_cali, test_loader,
                dqa_engine, iri_engine, fusion_engine,
                corruption_loader, baseline_corr, n_features,
                device=device, use_real_dti=True,
                label=f"CALI-PRED (rate={rate}, seed={seed})",
                missing_rate=rate,
            )
            # Evaluate Baseline
            res_base = evaluate_model(
                model_base, test_loader,
                dqa_engine, iri_engine, fusion_engine,
                corruption_loader, baseline_corr, n_features,
                device=device, use_real_dti=False,
                label=f"Baseline (rate={rate}, seed={seed})",
                missing_rate=rate,
            )

            rate_results["ece_cali"].append(res_cali["mean_ece"])
            rate_results["ece_base"].append(res_base["mean_ece"])
            rate_results["crps_cali"].append(res_cali["brier_score"])
            rate_results["crps_base"].append(res_base["brier_score"])

        sweep_results[rate] = rate_results

    # Print summary table
    print(f"\n{'Rate':>6} | {'ECE Base':>10} | {'ECE Cali':>10} | "
          f"{'CRPS Base':>10} | {'CRPS Cali':>10} | {'ECE Δ':>8}")
    print("-" * 70)
    for rate in missing_rates:
        r = sweep_results[rate]
        eb = np.mean(r["ece_base"])
        ec = np.mean(r["ece_cali"])
        cb = np.mean(r["crps_base"])
        cc = np.mean(r["crps_cali"])
        delta = ec - eb
        print(f"{rate:>6.2f} | {eb:>10.4f} | {ec:>10.4f} | "
              f"{cb:>10.4f} | {cc:>10.4f} | {delta:>+8.4f}")

    # Save sweep results
    sweep_path = os.path.join(checkpoint_dir, "severity_sweep.npz")
    np.savez(
        sweep_path,
        missing_rates=np.array(missing_rates),
        seeds=np.array(seeds),
        **{f"rate_{rate:.2f}_{k}": np.array(v)
           for rate in missing_rates
           for k, v in sweep_results[rate].items()},
    )
    logger.info("Severity sweep results saved to '%s'.", sweep_path)

    # Plot severity curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle("CALI-PRED: Multi-Severity Corruption Sweep",
                     fontsize=14, fontweight="bold")

        rates = list(missing_rates)
        for ax, metric, label in [(ax1, "ece", "ECE"), (ax2, "crps", "CRPS")]:
            base_means = [np.mean(sweep_results[r][f"{metric}_base"]) for r in rates]
            base_stds = [np.std(sweep_results[r][f"{metric}_base"]) for r in rates]
            cali_means = [np.mean(sweep_results[r][f"{metric}_cali"]) for r in rates]
            cali_stds = [np.std(sweep_results[r][f"{metric}_cali"]) for r in rates]

            ax.errorbar(rates, base_means, yerr=base_stds,
                        marker="o", color="tab:red", linewidth=2,
                        label="Baseline", capsize=4)
            ax.errorbar(rates, cali_means, yerr=cali_stds,
                        marker="s", color="tab:green", linewidth=2,
                        label="CALI-PRED", capsize=4)
            ax.set_xlabel("Missing Rate")
            ax.set_ylabel(f"Mean {label}")
            ax.set_title(f"{label} vs. Corruption Severity")
            ax.legend()

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plot_path = os.path.join(checkpoint_dir, "severity_sweep.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSeverity sweep plot saved to '{plot_path}'.")
    except Exception as e:
        logger.warning("Could not generate severity sweep plot: %s", e)

    print("=" * 70)
    return sweep_results


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CALI-PRED Bootstrap CI + Multi-Severity DTI Sweep",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Number of bootstrap resamples (default: 2000).")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="CI confidence level (default: 0.95).")
    parser.add_argument("--bootstrap-only", action="store_true",
                        help="Only run bootstrap CI (no severity sweep).")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to data CSV (required for severity sweep).")
    parser.add_argument("--dataset", type=str, default="metropt",
                        choices=["metropt", "ai4i2020", "tep"])
    parser.add_argument("--severity-rates", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
    parser.add_argument("--severity-seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1337])
    parser.add_argument("--max-windows", type=int, default=200,
                        help="Max test windows for severity sweep (default: 200).")
    parser.add_argument("--window-size", type=int, default=60,
                        help="Size of prediction windows (default: 60).")
    parser.add_argument("--n-features", type=int, default=None,
                        help="Number of channels/features (default: auto-detected from dataset).")

    args = parser.parse_args()

    # Determine n_features based on dataset if not specified
    n_feat = args.n_features
    if n_feat is None:
        if args.dataset == "ai4i2020":
            n_feat = 5
        elif args.dataset == "tep":
            n_feat = 22
        else:
            n_feat = 15  # metropt default

    # 1. Bootstrap CI (always runs if predictions exist)
    pred_path = os.path.join(args.checkpoint_dir, "test_predictions.npz")
    if os.path.exists(pred_path):
        bootstrap_results = run_bootstrap_analysis(
            pred_path, args.n_bootstrap, args.confidence,
            window_size=args.window_size, n_features=n_feat,
        )
    else:
        print(f"[SKIP] No saved predictions at '{pred_path}'.")
        print("  Run `python pipeline.py` first to generate test predictions.")
        bootstrap_results = None

    # 2. Multi-severity sweep (optional)
    if not args.bootstrap_only and args.data_path is not None:
        sweep_results = run_severity_sweep(
            data_path=args.data_path,
            dataset=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            missing_rates=args.severity_rates,
            seeds=args.severity_seeds,
            max_windows=args.max_windows,
        )
    elif not args.bootstrap_only and args.data_path is None:
        print("\n[SKIP] Severity sweep requires --data-path. "
              "Use --bootstrap-only to skip, or provide --data-path.")

    print("\n[OK] Analysis complete.")
