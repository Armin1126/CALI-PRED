"""
ablation_sigma_lr.py

CALI-PRED Study: Ablation study of sigma head stabilization strategies.

Sweeps a grid of:
  - sigma_lr_multiplier: [0.3, 0.5, 1.0, 2.5]  (slower → faster sigma LR)
  - sigma_floor:         [0.01, 0.1, 0.2]        (architectural sigma lower bound)
  - beta_nll:            [0.0]                    (β-NLL reweighting; extend to [0.0, 0.5, 1.0])
  - seeds:               [42, 456, 789]

Usage:
------
    python ablation_sigma_lr.py --data-path "data/metropt/MetroPT3(AirCompressor).csv" --epochs 25
    python ablation_sigma_lr.py --data-path "data/metropt/MetroPT3(AirCompressor).csv" --epochs 25 --beta-nlls 0.0 0.5 1.0

NOTE: If running on Google Colab, you can specify --save-interval-minutes to
automatically back up progress to Google Drive in case of runtime disconnects.
"""

import argparse
import logging
import os
import sys
import time
import json
import shutil
import threading
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("SigmaLRAblation")

# Mute verbose logs from intermediate engines to avoid Colab console truncation and speed up runs
logging.getLogger("ImputationReliabilityEngine").setLevel(logging.WARNING)
logging.getLogger("TrustFusionEngine").setLevel(logging.WARNING)
logging.getLogger("UpstreamDQAEngine").setLevel(logging.WARNING)
logging.getLogger("IndustrialDataLoader").setLevel(logging.WARNING)

# Local imports
from data_loader import create_dataloaders, IndustrialDataLoader
from dqa_module import UpstreamDQAEngine
from fusion_engine import TrustFusionEngine
from iri_module import ImputationReliabilityEngine
from predictor import CaliPredTransformer, TrustCalibratedLoss
from pipeline import (
    make_severity_sampler,
    precompute_trust_and_imputed,
    TrustCachedDataset,
    train_model,
    evaluate_model,
    fit_validation_sigma_scale,
)

def backup_to_drive_periodically(interval_minutes: float, checkpoint_dir: str):
    """Periodically archives the checkpoints directory to Google Drive if mounted."""
    if interval_minutes <= 0:
        return

    drive_backup_dir = "/content/drive/MyDrive/CALI-PRED-Results/ablation_sigma_lr_backup"

    def run_backup():
        while True:
            time.sleep(interval_minutes * 60)
            if os.path.exists("/content/drive/MyDrive"):
                try:
                    os.makedirs(drive_backup_dir, exist_ok=True)
                    archive_path = os.path.join(drive_backup_dir, "checkpoints_ablation_backup")
                    shutil.make_archive(archive_path, "zip", checkpoint_dir)
                    logger.info("[OK] Periodic backup of checkpoints to Google Drive complete.")
                except Exception as e:
                    logger.warning("[WARNING] Failed to write periodic backup to Google Drive: %s", e)

    t = threading.Thread(target=run_backup, daemon=True)
    t.start()

def compute_val_loss_stability(val_losses: list[float]) -> float:
    """Compute the coefficient of variation (CV) of epoch-to-epoch changes."""
    if len(val_losses) < 2:
        return 0.0
    diffs = np.abs(np.diff(val_losses))
    mean_diff = np.mean(diffs)
    if mean_diff == 0.0:
        return 0.0
    return float(np.std(diffs) / mean_diff)

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Ablation study running on device: %s", device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Dataset Safeguards check
    logger.info("Dataset Safeguards: Loading target CSV metadata check...")
    try:
        raw_df = pd.read_csv(args.data_path, nrows=50)
        logger.info("Dataset Safeguards: Successfully verified target CSV. Shape (first block) = %s, Columns = %s", raw_df.shape, list(raw_df.columns))
    except Exception as e:
        logger.error("Dataset Safeguards: Failed to read CSV from '%s'. Error: %s", args.data_path, e)
        sys.exit(1)

    # Start the periodic drive backup thread
    backup_to_drive_periodically(args.save_interval_minutes, args.checkpoint_dir)

    # Parse grid axes
    seeds = args.seeds
    multipliers = args.multipliers
    sigma_floors = args.sigma_floors
    beta_nlls = args.beta_nlls
    models = ["CALI-PRED", "Baseline"]

    total_runs = len(seeds) * len(multipliers) * len(sigma_floors) * len(beta_nlls) * len(models)
    logger.info(
        "Ablation grid: %d multipliers x %d floors x %d beta_nlls x %d models x %d seeds = %d total runs",
        len(multipliers), len(sigma_floors), len(beta_nlls), len(models), len(seeds), total_runs
    )

    # Clear previous ablation outputs
    json_path = os.path.join(args.checkpoint_dir, "sigma_lr_ablation_results.json")
    csv_path = os.path.join(args.checkpoint_dir, "sigma_lr_ablation_summary.csv")
    if os.path.exists(json_path):
        os.remove(json_path)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    run_count = 0

    for seed in seeds:
        logger.info("-" * 60)
        logger.info("Preparing data cache for Seed %d...", seed)
        logger.info("-" * 60)

        # Load data splits
        train_ds, val_ds, test_ds, _, _, _ = create_dataloaders(
            dataset_name=args.dataset,
            file_path=args.data_path,
            window_size=args.window_size,
            stride=args.stride,
            forecast_horizon=args.forecast_horizon,
            batch_size=args.batch_size,
            random_state=seed,
        )
        n_features = train_ds.n_features

        # Explicit dataset safeguards check
        if hasattr(train_ds, "X"):
            logger.info("Dataset Safeguards: Raw training matrix shape = %s (channels = %d)", train_ds.X.shape, n_features)
        logger.info(
            "Dataset Safeguards: train_ds = %d windows, val_ds = %d windows, test_ds = %d windows",
            len(train_ds), len(val_ds), len(test_ds)
        )

        # Limit windows for quick testing
        if args.max_windows is not None:
            from torch.utils.data import Subset
            max_w = args.max_windows
            if len(train_ds) > max_w:
                train_ds = Subset(train_ds, range(min(max_w, len(train_ds))))
            if len(val_ds) > max_w:
                val_ds = Subset(val_ds, range(min(max_w, len(val_ds))))
            if len(test_ds) > max_w:
                test_ds = Subset(test_ds, range(min(max_w, len(test_ds))))
            logger.info("Limited splits to max %d windows.", max_w)

        # Initialize engines
        dqa_engine = UpstreamDQAEngine(freshness_tau_seconds=60.0, max_corr_mae=0.5)
        iri_engine = ImputationReliabilityEngine(
            n_features=n_features, epochs=30, holdout_frac=0.15, random_state=seed,
        )
        fusion_engine = TrustFusionEngine(clamp_inputs=True)
        corruption_loader = IndustrialDataLoader(random_state=seed)

        # Correlation matrix
        if hasattr(train_ds, "X"):
            train_X = train_ds.X
        else:
            train_X = train_ds.dataset.X
        baseline_corr = np.corrcoef(train_X.T)
        baseline_corr = np.nan_to_num(baseline_corr, nan=0.0)

        # Severity samplers
        train_val_sampler = make_severity_sampler(
            clean_fraction=args.clean_fraction,
            max_severity=args.max_severity,
            random_state=seed,
        )
        test_sampler = make_severity_sampler(
            clean_fraction=args.clean_fraction,
            max_severity=args.max_severity,
            random_state=seed + 58,
        )

        # Precompute cache for splits (only once per seed)
        train_dti, train_imputed = precompute_trust_and_imputed(
            train_ds, dqa_engine, iri_engine, fusion_engine,
            corruption_loader, baseline_corr, n_features,
            missing_rate_sampler=train_val_sampler,
        )
        train_ds = TrustCachedDataset(train_ds, train_dti, train_imputed)

        val_dti, val_imputed = precompute_trust_and_imputed(
            val_ds, dqa_engine, iri_engine, fusion_engine,
            corruption_loader, baseline_corr, n_features,
            missing_rate_sampler=train_val_sampler,
        )
        val_ds = TrustCachedDataset(val_ds, val_dti, val_imputed)

        test_dti, test_imputed = precompute_trust_and_imputed(
            test_ds, dqa_engine, iri_engine, fusion_engine,
            corruption_loader, baseline_corr, n_features,
            missing_rate_sampler=test_sampler,
        )
        test_ds = TrustCachedDataset(test_ds, test_dti, test_imputed)

        # Build loaders
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False,
            drop_last=(len(train_ds) > args.batch_size)
        )
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # Run grid sweep for this seed
        for s_floor in sigma_floors:
            for beta in beta_nlls:
                for mult in multipliers:
                    for m_type in models:
                        run_count += 1

                        # Enforce a deterministic environment
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)

                        logger.info(
                            "[RUN %d/%d] Model=%s | mult=%.2f | floor=%.3f | beta=%.1f | seed=%d",
                            run_count, total_runs, m_type, mult, s_floor, beta, seed
                        )

                        # Output folder for checkpointing
                        ckpt_sub_dir = os.path.join(
                            args.checkpoint_dir,
                            f"ablation_mult_{mult}_floor_{s_floor}_beta_{beta}_seed_{seed}_{m_type.lower()}"
                        )
                        os.makedirs(ckpt_sub_dir, exist_ok=True)

                        # Initialize model with this sigma_floor
                        model = CaliPredTransformer(
                            input_dim=n_features, output_dim=n_features,
                            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                            dropout=0.1, max_uncertainty_inflation=args.max_inflation,
                            alpha_init=args.alpha_init,
                            sigma_floor=s_floor,
                            sigma_init_bias=args.sigma_init_bias,
                        )

                        loss_fn = TrustCalibratedLoss(
                            lower_q=0.05, upper_q=0.95, calibration_weight=0.2,
                            beta_nll=beta,
                        )
                        use_real_dti = (m_type == "CALI-PRED")

                        # Train
                        history = train_model(
                            model=model, loss_fn=loss_fn,
                            train_loader=train_loader, val_loader=val_loader,
                            dqa_engine=dqa_engine, iri_engine=iri_engine, fusion_engine=fusion_engine,
                            corruption_loader=corruption_loader, baseline_corr=baseline_corr,
                            n_features=n_features, epochs=args.epochs, lr=args.lr,
                            device=device, checkpoint_dir=ckpt_sub_dir, use_real_dti=use_real_dti,
                            missing_rate_sampler=train_val_sampler,
                            sigma_lr_multiplier=mult,
                            val_warmup_epochs=getattr(args, "val_warmup_epochs", 3),
                        )

                        # Stability metrics
                        val_losses = history["val_loss"]
                        val_maes = history["val_mae"]
                        stability_cv = compute_val_loss_stability(val_losses)
                        mae_stability_cv = compute_val_loss_stability(val_maes)
                        epochs_trained = len(val_losses)
                        early_stopped = (epochs_trained < args.epochs)

                        # Load best checkpoint for evaluation
                        ckpt_name = "best_model_calipred.pt" if use_real_dti else "best_model_baseline.pt"
                        ckpt_path = os.path.join(ckpt_sub_dir, ckpt_name)
                        if os.path.exists(ckpt_path):
                            model.load_state_dict(
                                torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
                            )

                        # Post-hoc validation scaling
                        val_results = evaluate_model(
                            model, val_loader,
                            dqa_engine, iri_engine, fusion_engine,
                            corruption_loader, baseline_corr, n_features,
                            device=device, use_real_dti=use_real_dti, label=f"{m_type} val",
                            split_name="Validation",
                            missing_rate_sampler=train_val_sampler,
                        )
                        try:
                            sigma_scale = fit_validation_sigma_scale(
                                val_results["y_true"], val_results["mu"], val_results["sigma"]
                            )
                        except Exception:
                            sigma_scale = 1.0

                        # Raw evaluation on test set (sigma_scale=1.0)
                        eval_test_raw = evaluate_model(
                            model, test_loader,
                            dqa_engine, iri_engine, fusion_engine,
                            corruption_loader, baseline_corr, n_features,
                            device=device, use_real_dti=use_real_dti, label=f"{m_type} raw",
                            split_name="Test",
                            missing_rate_sampler=test_sampler,
                            sigma_scale=1.0,
                        )

                        # Scaled evaluation on test set
                        eval_test_scaled = evaluate_model(
                            model, test_loader,
                            dqa_engine, iri_engine, fusion_engine,
                            corruption_loader, baseline_corr, n_features,
                            device=device, use_real_dti=use_real_dti, label=f"{m_type} scaled",
                            split_name="Test",
                            missing_rate_sampler=test_sampler,
                            sigma_scale=sigma_scale,
                        )

                        # Compute mean raw sigma on test data
                        mean_raw_sigma = float(np.mean(eval_test_raw["raw_sigma"]))

                        # --- Incremental persistence ---
                        # A. JSON results append
                        try:
                            with open(json_path, 'r') as f:
                                json_data = json.load(f)
                        except Exception:
                            json_data = []

                        json_data.append({
                            "sigma_lr_multiplier": mult,
                            "sigma_floor": s_floor,
                            "beta_nll": beta,
                            "model": m_type,
                            "seed": seed,
                            "train_loss_curve": history["train_loss"],
                            "val_loss_curve": history["val_loss"],
                            "val_mae_curve": history["val_mae"],
                            "train_sigma_base_curve": history.get("train_sigma_base", []),
                            "val_sigma_base_curve": history.get("val_sigma_base", []),
                            "train_nll_log_sigma_curve": history.get("train_nll_log_sigma", []),
                            "train_nll_residual_curve": history.get("train_nll_residual", []),
                        })

                        with open(json_path, 'w') as f:
                            json.dump(json_data, f, indent=4)

                        # B. CSV summary append
                        row_df = pd.DataFrame([{
                            "sigma_lr_multiplier": mult,
                            "sigma_floor": s_floor,
                            "beta_nll": beta,
                            "model": m_type,
                            "seed": seed,
                            "val_loss_stability_cv": stability_cv,
                            "mae_stability_cv": mae_stability_cv,
                            "epochs_trained": epochs_trained,
                            "early_stopped": early_stopped,
                            "best_epoch": history.get("best_epoch", 1),
                            "ece_raw": float(eval_test_raw["mean_ece"]),
                            "crps_raw": float(eval_test_raw["brier_score"]),
                            "ece_scaled": float(eval_test_scaled["mean_ece"]),
                            "crps_scaled": float(eval_test_scaled["brier_score"]),
                            "sigma_scale": sigma_scale,
                            "sigma_scale_proximity": abs(sigma_scale - 1.0),
                            "mean_raw_sigma": mean_raw_sigma,
                            "final_train_sigma_base": history["train_sigma_base"][-1] if history["train_sigma_base"] else 0.0,
                            "final_val_sigma_base": history["val_sigma_base"][-1] if history["val_sigma_base"] else 0.0,
                        }])

                        header = not os.path.exists(csv_path)
                        row_df.to_csv(csv_path, mode='a', index=False, header=header)
                        logger.info(
                            "  -> Result: ECE_raw=%.4f | CRPS_raw=%.4f | sigma_scale=%.2f | "
                            "mean_raw_sigma=%.4f | best_epoch=%d",
                            float(eval_test_raw["mean_ece"]),
                            float(eval_test_raw["brier_score"]),
                            sigma_scale,
                            mean_raw_sigma,
                            history.get("best_epoch", 1),
                        )

    # ======================================================================== #
    # 5. Read all results back to construct aggregated stats
    # ======================================================================== #
    df = pd.read_csv(csv_path)

    # Calculate Aggregated Stats
    group_cols = ["sigma_lr_multiplier", "sigma_floor", "beta_nll", "model"]
    aggregated = []
    for group_key, group in df.groupby(group_cols):
        mult, s_floor, beta, m_type = group_key
        aggregated.append({
            "sigma_lr_multiplier": mult,
            "sigma_floor": s_floor,
            "beta_nll": beta,
            "model": m_type,
            "n_seeds": len(group),
            "mean_cv": float(np.mean(group["val_loss_stability_cv"].values)),
            "mean_ece_raw": float(np.mean(group["ece_raw"].values)),
            "std_ece_raw": float(np.std(group["ece_raw"].values)),
            "mean_crps_raw": float(np.mean(group["crps_raw"].values)),
            "mean_ece_scaled": float(np.mean(group["ece_scaled"].values)),
            "mean_crps_scaled": float(np.mean(group["crps_scaled"].values)),
            "mean_sigma_scale": float(np.mean(group["sigma_scale"].values)),
            "mean_sigma_proximity": float(np.mean(group["sigma_scale_proximity"].values)),
            "mean_raw_sigma": float(np.mean(group["mean_raw_sigma"].values)),
            "mean_best_epoch": float(np.mean(group["best_epoch"].values)),
            "mean_final_train_sigma": float(np.mean(group["final_train_sigma_base"].values)),
        })

    df_agg = pd.DataFrame(aggregated)
    agg_csv_path = os.path.join(args.checkpoint_dir, "sigma_lr_ablation_aggregated.csv")
    df_agg.to_csv(agg_csv_path, index=False)

    # Print Summary Markdown Table
    print("\n" + "=" * 160)
    print("  AGGREGATED SIGMA STABILITY ABLATION RESULTS")
    print("=" * 160)
    cols = ["sigma_lr_multiplier", "sigma_floor", "beta_nll", "model", "n_seeds",
            "mean_ece_raw", "std_ece_raw", "mean_crps_raw",
            "mean_sigma_scale", "mean_sigma_proximity", "mean_raw_sigma",
            "mean_best_epoch", "mean_final_train_sigma"]
    header = " | ".join(cols)
    sep = " | ".join(["---"] * len(cols))
    print(f"| {header} |")
    print(f"| {sep} |")
    for _, row in df_agg.iterrows():
        row_str = " | ".join(
            f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
            for c in cols
        )
        print(f"| {row_str} |")
    print("=" * 160)

    # Print per-seed detail table
    print("\n" + "-" * 140)
    print("  PER-SEED DETAILS")
    print("-" * 140)
    detail_cols = ["sigma_lr_multiplier", "sigma_floor", "beta_nll", "model", "seed",
                   "best_epoch", "ece_raw", "crps_raw", "sigma_scale",
                   "sigma_scale_proximity", "mean_raw_sigma", "final_train_sigma_base"]
    print("| " + " | ".join(detail_cols) + " |")
    print("| " + " | ".join(["---"] * len(detail_cols)) + " |")
    for _, row in df.iterrows():
        row_str = " | ".join(
            f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
            for c in detail_cols
        )
        print(f"| {row_str} |")
    print("-" * 140)

    # ======================================================================== #
    # Interpretation
    # ======================================================================== #
    print("\n" + "=" * 80)
    print("  INTERPRETATION SUMMARY")
    print("=" * 80)

    # Find the configuration with sigma_scale closest to 1.0 (CALI-PRED only)
    cali_rows = df_agg[df_agg["model"] == "CALI-PRED"]
    if len(cali_rows) > 0:
        best_row = cali_rows.loc[cali_rows["mean_sigma_proximity"].idxmin()]
        print(
            f"\n[BEST SIGMA CALIBRATION] CALI-PRED configuration with sigma_scale closest to 1.0:\n"
            f"  mult={best_row['sigma_lr_multiplier']:.2f}, floor={best_row['sigma_floor']:.3f}, "
            f"beta={best_row['beta_nll']:.1f}\n"
            f"  -> sigma_scale={best_row['mean_sigma_scale']:.4f} "
            f"(proximity={best_row['mean_sigma_proximity']:.4f})\n"
            f"  -> ECE_raw={best_row['mean_ece_raw']:.4f}, CRPS_raw={best_row['mean_crps_raw']:.4f}\n"
            f"  -> mean_best_epoch={best_row['mean_best_epoch']:.1f} "
            f"(>4 means training actually progressed past warmup)\n"
            f"  -> mean_raw_sigma={best_row['mean_raw_sigma']:.4f} "
            f"(should be ~0.2-0.5 for well-calibrated z-scored data)"
        )

    # Check if sigma floor prevents collapse (train sigma stays above floor)
    if len(cali_rows) > 0:
        for _, row in cali_rows.iterrows():
            if row["mean_final_train_sigma"] < row["sigma_floor"] * 1.2:
                print(
                    f"\n[FLOOR LOAD-BEARING WARNING] floor={row['sigma_floor']:.3f}: "
                    f"final train sigma ({row['mean_final_train_sigma']:.4f}) is near or "
                    f"at the floor. Check that DTI uncertainty responsiveness (Fig. 1 right) "
                    f"still shows meaningful separation across strata — the floor may be "
                    f"compressing your core dynamic range result."
                )

    # Stability comparison
    if len(cali_rows) > 1:
        most_stable = cali_rows.loc[cali_rows["mean_cv"].idxmin()]
        least_stable = cali_rows.loc[cali_rows["mean_cv"].idxmax()]
        print(
            f"\n[STABILITY] Most stable config: mult={most_stable['sigma_lr_multiplier']:.2f}, "
            f"floor={most_stable['sigma_floor']:.3f} (CV={most_stable['mean_cv']:.4f})\n"
            f"            Least stable:      mult={least_stable['sigma_lr_multiplier']:.2f}, "
            f"floor={least_stable['sigma_floor']:.3f} (CV={least_stable['mean_cv']:.4f})"
        )

    print("=" * 80)

    # ======================================================================== #
    # Diagnostic Plots
    # ======================================================================== #
    # Plot 1: Sigma scale proximity heatmap (multiplier x floor)
    if len(cali_rows) > 0 and len(multipliers) > 1 and len(sigma_floors) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax_idx, model_name in enumerate(["CALI-PRED", "Baseline"]):
            ax = axes[ax_idx]
            model_df = df_agg[df_agg["model"] == model_name]
            if len(model_df) == 0:
                continue

            # Build heatmap matrix
            pivot = model_df.pivot_table(
                values="mean_sigma_proximity",
                index="sigma_floor", columns="sigma_lr_multiplier",
                aggfunc="mean"
            )
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.3f}" for v in pivot.index])
            ax.set_xlabel("Sigma LR Multiplier")
            ax.set_ylabel("Sigma Floor")
            ax.set_title(f"{model_name}: |sigma_scale - 1.0|\n(lower = better calibrated)")

            # Annotate cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                color="black" if val < 3 else "white", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        heatmap_path = os.path.join(args.checkpoint_dir, "sigma_calibration_heatmap.png")
        plt.savefig(heatmap_path, dpi=150)
        logger.info("Sigma calibration heatmap saved to '%s'.", heatmap_path)
        plt.close()

    # Plot 2: Best epoch heatmap (did training progress past warmup?)
    if len(cali_rows) > 0 and len(multipliers) > 1 and len(sigma_floors) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax_idx, model_name in enumerate(["CALI-PRED", "Baseline"]):
            ax = axes[ax_idx]
            model_df = df_agg[df_agg["model"] == model_name]
            if len(model_df) == 0:
                continue

            pivot = model_df.pivot_table(
                values="mean_best_epoch",
                index="sigma_floor", columns="sigma_lr_multiplier",
                aggfunc="mean"
            )
            im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", origin="lower",
                          vmin=4, vmax=args.epochs)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.3f}" for v in pivot.index])
            ax.set_xlabel("Sigma LR Multiplier")
            ax.set_ylabel("Sigma Floor")
            ax.set_title(f"{model_name}: Mean Best Epoch\n(>4 = training progressed past warmup)")

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                                color="black", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        epoch_path = os.path.join(args.checkpoint_dir, "sigma_best_epoch_heatmap.png")
        plt.savefig(epoch_path, dpi=150)
        logger.info("Best epoch heatmap saved to '%s'.", epoch_path)
        plt.close()

    logger.info("[OK] Ablation study complete. Results in '%s'.", args.checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study: Sigma Head Stabilization (LR x Floor x beta-NLL)",
    )
    parser.add_argument(
        "--dataset", type=str, default="metropt",
        choices=["metropt", "ai4i2020", "tep"],
    )
    parser.add_argument(
        "--data-path", type=str, default="data/metropt/MetroPT3(AirCompressor).csv",
        help="Path to the raw CSV file.",
    )
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=200)  # Stride 200 for speed
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--max-inflation", type=float, default=10.0)
    parser.add_argument("--alpha-init", type=float, default=0.5)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--clean-fraction", type=float, default=0.25)
    parser.add_argument("--max-severity", type=float, default=0.45)
    parser.add_argument(
        "--save-interval-minutes", type=float, default=5.0,
        help="Minutes between periodic checkpoints zips back to Google Drive (default: 5.0).",
    )
    parser.add_argument(
        "--val-warmup-epochs", type=int, default=3,
        help="Number of initial warmup epochs before checkpoint eligibility (default: 3).",
    )
    parser.add_argument("--max-windows", type=int, default=None)

    # Ablation grid axes
    parser.add_argument(
        "--multipliers", type=float, nargs="+", default=[0.3, 0.5, 1.0, 2.5],
        help="Sigma LR multiplier values to sweep (default: 0.3 0.5 1.0 2.5).",
    )
    parser.add_argument(
        "--sigma-floors", type=float, nargs="+", default=[0.01, 0.1, 0.2],
        help="Sigma floor values to sweep (default: 0.01 0.1 0.2).",
    )
    parser.add_argument(
        "--beta-nlls", type=float, nargs="+", default=[0.0],
        help="Beta-NLL values to sweep (default: 0.0; add 0.5 1.0 for full sweep).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 456, 789],
        help="Random seeds to sweep (default: 42 456 789).",
    )
    parser.add_argument(
        "--sigma-init-bias", type=float, default=0.5,
        help="Initial bias for sigma head (default: 0.5).",
    )

    args = parser.parse_args()
    main(args)


