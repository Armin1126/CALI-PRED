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
def _precompute_window_metrics(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nominal_levels: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
    window_size: int = 60,
    n_features: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute per-window coverage and CRPS arrays for fast vector bootstrap."""
    from scipy.stats import norm
    import math

    block_size = window_size * n_features
    N = len(y_true)
    N_windows = N // block_size

    y_t = y_true[:N_windows * block_size].reshape(N_windows, block_size)
    m = mu[:N_windows * block_size].reshape(N_windows, block_size)
    s = np.maximum(sigma[:N_windows * block_size].reshape(N_windows, block_size), 1e-8)

    # 1. Coverage per window per nominal level: shape (N_windows, L)
    abs_res = np.abs(y_t - m)
    L = len(nominal_levels)
    window_cov = np.empty((N_windows, L), dtype=np.float64)
    for i, lvl in enumerate(nominal_levels):
        z = norm.ppf(0.5 + lvl / 2.0)
        within = (abs_res <= z * s).astype(np.float64)
        window_cov[:, i] = np.mean(within, axis=1)

    # 2. Continuous Ranked Probability Score (CRPS) per window: shape (N_windows,)
    z_std = (y_t - m) / s
    phi = norm.pdf(z_std)
    Phi = norm.cdf(z_std)
    crps_elem = s * (z_std * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    window_crps = np.mean(crps_elem, axis=1)

    return window_cov, window_crps


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
    nominal_levels: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
) -> dict:
    """Fast vectorized window-block bootstrap for individual metric."""
    if rng is None:
        rng = np.random.default_rng(42)

    nom_arr = np.asarray(nominal_levels, dtype=np.float64)
    w_cov, w_crps = _precompute_window_metrics(y_true, mu, sigma, nominal_levels, window_size, n_features)
    N_windows = len(w_crps)

    # Draw all bootstrap resamples in one fast vectorized batch
    idx_matrix = rng.integers(0, N_windows, size=(n_bootstrap, N_windows))

    if metric_fn == "ece":
        # shape: (n_bootstrap, L)
        boot_cov = np.mean(w_cov[idx_matrix], axis=1)
        samples = np.mean(np.abs(boot_cov - nom_arr), axis=1)
        point = float(np.mean(np.abs(np.mean(w_cov, axis=0) - nom_arr)))
    elif metric_fn == "crps":
        samples = np.mean(w_crps[idx_matrix], axis=1)
        point = float(np.mean(w_crps))
    else:
        raise ValueError(f"Unknown metric_fn: {metric_fn}")

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": point,
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
    nominal_levels: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
) -> dict:
    """Fast vectorized window-block bootstrap for paired metric difference (A - B)."""
    if rng is None:
        rng = np.random.default_rng(42)

    nom_arr = np.asarray(nominal_levels, dtype=np.float64)
    w_cov_a, w_crps_a = _precompute_window_metrics(y_true, mu_a, sigma_a, nominal_levels, window_size, n_features)
    w_cov_b, w_crps_b = _precompute_window_metrics(y_true, mu_b, sigma_b, nominal_levels, window_size, n_features)
    N_windows = len(w_crps_a)

    idx_matrix = rng.integers(0, N_windows, size=(n_bootstrap, N_windows))

    if metric_fn == "ece":
        boot_cov_a = np.mean(w_cov_a[idx_matrix], axis=1)
        boot_cov_b = np.mean(w_cov_b[idx_matrix], axis=1)
        samples_a = np.mean(np.abs(boot_cov_a - nom_arr), axis=1)
        samples_b = np.mean(np.abs(boot_cov_b - nom_arr), axis=1)
    elif metric_fn == "crps":
        samples_a = np.mean(w_crps_a[idx_matrix], axis=1)
        samples_b = np.mean(w_crps_b[idx_matrix], axis=1)
    else:
        raise ValueError(f"Unknown metric_fn: {metric_fn}")

    diffs = samples_a - samples_b
    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(diffs)),
        "p_positive": float(np.mean(diffs > 0)),
        "p_negative": float(np.mean(diffs < 0)),
        "all_diffs": diffs,
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
    from pipeline import evaluate_model

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
    parser.add_argument("--pred-paths", type=str, nargs="+", default=None,
                        help="Explicit paths to one or more test_predictions.npz files.")
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

    # 1. Collect prediction paths
    pred_paths = []
    if args.pred_paths:
        pred_paths = args.pred_paths
    else:
        direct_path = os.path.join(args.checkpoint_dir, "test_predictions.npz")
        if os.path.exists(direct_path):
            pred_paths.append(direct_path)
        else:
            import glob
            seed_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "seed_*", "test_predictions.npz")))
            pred_paths.extend(seed_files)

    if pred_paths:
        print(f"\n[INFO] Found {len(pred_paths)} prediction file(s) to analyze.")
        all_results = {}
        for p in pred_paths:
            print(f"\n>>> Running Bootstrap Analysis on: {p}")
            res = run_bootstrap_analysis(
                p, args.n_bootstrap, args.confidence,
                window_size=args.window_size, n_features=n_feat,
            )
            all_results[p] = res

        # If multiple files, print a consolidated LaTeX summary
        if len(pred_paths) > 1:
            print("\n" + "=" * 70)
            print("  CONSOLIDATED MULTI-SEED LATEX TABLE SNIPPET")
            print("=" * 70)
            print("% Copy and paste into LaTeX:")
            print("\\begin{table}[h!]")
            print("\\centering")
            print("\\begin{tabular}{lcccc}")
            print("\\toprule")
            print("Seed & Baseline ECE [95\\% CI] & CALI-PRED ECE [95\\% CI] & Baseline CRPS [95\\% CI] & CALI-PRED CRPS [95\\% CI] \\\\")
            print("\\midrule")
            for p, r in all_results.items():
                s_name = os.path.basename(os.path.dirname(p)) if "seed_" in p else "Overall"
                b_ece = f"{r['ece']['baseline']['point_estimate']:.4f} [{r['ece']['baseline']['ci_lower']:.4f}, {r['ece']['baseline']['ci_upper']:.4f}]"
                c_ece = f"{r['ece']['calipred']['point_estimate']:.4f} [{r['ece']['calipred']['ci_lower']:.4f}, {r['ece']['calipred']['ci_upper']:.4f}]"
                b_crps = f"{r['crps']['baseline']['point_estimate']:.4f} [{r['crps']['baseline']['ci_lower']:.4f}, {r['crps']['baseline']['ci_upper']:.4f}]"
                c_crps = f"{r['crps']['calipred']['point_estimate']:.4f} [{r['crps']['calipred']['ci_lower']:.4f}, {r['crps']['calipred']['ci_upper']:.4f}]"
                print(f"{s_name} & {b_ece} & {c_ece} & {b_crps} & {c_crps} \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
            print("\\caption{Window-block bootstrap 95\\% confidence intervals across multi-seed runs.}")
            print("\\label{tab:bootstrap_ci}")
            print("\\end{table}")
            print("=" * 70)
    else:
        print(f"[SKIP] No saved predictions found in '{args.checkpoint_dir}'.")
        print("  Run `python run_multiseed_pipeline.py` or `pipeline.py` first.")

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

