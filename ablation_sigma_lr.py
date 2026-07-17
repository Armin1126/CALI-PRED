"""
ablation_sigma_lr.py

CALI-PRED Study: Ablation study of the decoupled sigma learning rate multiplier
to evaluate its impact on training/validation stability and seed-to-seed result variance.

Usage:
------
    python ablation_sigma_lr.py --data-path "data/metropt/MetroPT3(AirCompressor).csv" --epochs 25
    python ablation_sigma_lr.py --data-path "data/metropt/MetroPT3(chiller).csv" --epochs 25

NOTE: If running on Google Colab, make sure to periodically check the drive backup.
You can specify --save-interval-minutes to automatically back up progress to your
Google Drive in case of runtime disconnects.
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

    # 1. Dataset Safeguards check: immediately load CSV via pandas to verify true MetroPT dataset dimensions
    logger.info("Dataset Safeguards: Loading target CSV metadata check...")
    try:
        raw_df = pd.read_csv(args.data_path, nrows=50)
        logger.info("Dataset Safeguards: Successfully verified target CSV. Shape (first block) = %s, Columns = %s", raw_df.shape, list(raw_df.columns))
    except Exception as e:
        logger.error("Dataset Safeguards: Failed to read CSV from '%s'. Error: %s", args.data_path, e)
        sys.exit(1)

    # Start the periodic drive backup thread
    backup_to_drive_periodically(args.save_interval_minutes, args.checkpoint_dir)

    seeds = [42, 456]
    multipliers = [1.0, 2.5]
    models = ["CALI-PRED", "Baseline"]

    # Clear previous ablation outputs to ensure fresh starts
    json_path = os.path.join(args.checkpoint_dir, "sigma_lr_ablation_results.json")
    csv_path = os.path.join(args.checkpoint_dir, "sigma_lr_ablation_summary.csv")
    if os.path.exists(json_path):
        os.remove(json_path)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Keep track of loss histories for plotting
    plot_histories = {}

    print("\n" + "=" * 80)
    print("Note: ablation runs use reduced epoch count (25 vs 40) for speed; confirm trend holds with full training before finalizing conclusions.")
    print("=" * 80)

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
            train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # Run models & multipliers
        for mult in multipliers:
            for m_type in models:
                # Enforce a deterministic environment at the beginning of each run
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                logger.info(
                    "[RUN] Running: Model=%s, Multiplier=%.1f, Seed=%d",
                    m_type, mult, seed
                )

                # Output folder for checkpointing
                ckpt_sub_dir = os.path.join(args.checkpoint_dir, f"ablation_mult_{mult}_seed_{seed}_{m_type.lower()}")
                os.makedirs(ckpt_sub_dir, exist_ok=True)

                # Initialize model
                model = CaliPredTransformer(
                    input_dim=n_features, output_dim=n_features,
                    d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                    dropout=0.1, max_uncertainty_inflation=args.max_inflation,
                    alpha_init=args.alpha_init,
                )

                loss_fn = TrustCalibratedLoss(lower_q=0.05, upper_q=0.95, calibration_weight=0.2)
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
                )

                # Keep track of val_loss history for plots
                plot_histories[(mult, seed, m_type)] = history["val_loss"]

                # Stability metrics
                val_losses = history["val_loss"]
                stability_cv = compute_val_loss_stability(val_losses)
                epochs_trained = len(val_losses)
                early_stopped = (epochs_trained < args.epochs)

                # Calculate spiked_3x: check if val_loss ever exceeds 3x its value from 2 epochs prior
                spiked_3x = False
                for e in range(2, len(val_losses)):
                    if val_losses[e] > 3.0 * val_losses[e - 2]:
                        spiked_3x = True
                        break

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
                    missing_rate_sampler=train_val_sampler,
                )
                try:
                    sigma_scale = fit_validation_sigma_scale(
                        val_results["y_true"], val_results["mu"], val_results["sigma"]
                    )
                except Exception:
                    sigma_scale = 1.0

                # Final Evaluation on Held-out test set
                eval_test = evaluate_model(
                    model, test_loader,
                    dqa_engine, iri_engine, fusion_engine,
                    corruption_loader, baseline_corr, n_features,
                    device=device, use_real_dti=use_real_dti, label=m_type,
                    missing_rate_sampler=test_sampler,
                    sigma_scale=sigma_scale,
                )

                # --- Defensive Persistence: Write incrementally to files ---
                # A. JSON results append
                try:
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                except Exception:
                    json_data = []

                json_data.append({
                    "sigma_lr_multiplier": mult,
                    "model": m_type,
                    "seed": seed,
                    "train_loss_curve": history["train_loss"],
                    "val_loss_curve": history["val_loss"],
                })

                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=4)

                # B. CSV summary append
                row_df = pd.DataFrame([{
                    "sigma_lr_multiplier": mult,
                    "model": m_type,
                    "seed": seed,
                    "val_loss_stability_cv": stability_cv,
                    "spiked_3x": spiked_3x,
                    "epochs_trained": epochs_trained,
                    "early_stopped": early_stopped,
                    "final_ece": float(eval_test["mean_ece"]),
                    "final_crps": float(eval_test["brier_score"]),
                    "sigma_scale": sigma_scale,
                }])

                header = not os.path.exists(csv_path)
                row_df.to_csv(csv_path, mode='a', index=False, header=header)
                logger.info("Saved incremental results for mult=%.1f, model=%s, seed=%d.", mult, m_type, seed)

    # 5. Read all results back to construct aggregated stats
    df = pd.read_csv(csv_path)

    # Calculate Aggregated Stats
    aggregated = []
    for (mult, m_type), group in df.groupby(["sigma_lr_multiplier", "model"]):
        ece_vals = group["final_ece"].values
        crps_vals = group["final_crps"].values
        cv_vals = group["val_loss_stability_cv"].values
        scale_vals = group["sigma_scale"].values
        
        ece_range = float(np.max(ece_vals) - np.min(ece_vals))
        
        aggregated.append({
            "sigma_lr_multiplier": mult,
            "model": m_type,
            "mean_cv": float(np.mean(cv_vals)),
            "std_cv": float(np.std(cv_vals)),
            "mean_ece": float(np.mean(ece_vals)),
            "std_ece": float(np.std(ece_vals)),
            "ece_range": ece_range,
            "mean_crps": float(np.mean(crps_vals)),
            "std_crps": float(np.std(crps_vals)),
            "mean_scale": float(np.mean(scale_vals)),
            "std_scale": float(np.std(scale_vals)),
        })

    df_agg = pd.DataFrame(aggregated)

    # Print Summary Markdown Table
    print("\n" + "=" * 120)
    print("  AGGERGATED SIGMA LR DECOUPLING ABLATION RESULTS")
    print("=" * 120)
    cols = ["sigma_lr_multiplier", "model", "mean_cv", "mean_ece", "ece_range", "mean_crps", "mean_scale"]
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
    print("=" * 120)

    # Direct Interpretation logic
    cali_10 = df_agg[(df_agg["sigma_lr_multiplier"] == 1.0) & (df_agg["model"] == "CALI-PRED")].iloc[0]
    cali_25 = df_agg[(df_agg["sigma_lr_multiplier"] == 2.5) & (df_agg["model"] == "CALI-PRED")].iloc[0]

    cv_drop = (cali_25["mean_cv"] - cali_10["mean_cv"]) / (cali_25["mean_cv"] + 1e-8)
    range_drop = (cali_25["ece_range"] - cali_10["ece_range"]) / (cali_25["ece_range"] + 1e-8)

    print("\n" + "=" * 80)
    print("  INTERPRETATION SUMMARY")
    print("=" * 80)
    if cv_drop >= 0.50 and range_drop >= 0.50:
        print(
            "[CONFIRMED] The decoupled sigma learning rate is a primary driver "
            "of both training instability and seed-to-seed result variance. "
            "Recommend adopting sigma_lr_multiplier=1.0 (or a smaller value like "
            "1.5) as the new default, and re-running the full multi-seed "
            "benchmark before finalizing paper results."
        )
    else:
        print(
            "[NOT CONFIRMED] Instability persists even without the sigma LR decoupling. "
            "The root cause is elsewhere -- investigate the cosine annealing schedule's "
            "interaction with the NLL loss's log(sigma^2) term, the sigma_floor value in "
            "CaliPredTransformer, gradient clipping threshold, or whether "
            "TrustCalibratedLoss's calibration_weight is contributing (see the prior "
            "calibration_weight ablation)."
        )

    # Check Baseline vs CALI-PRED
    base_25 = df_agg[(df_agg["sigma_lr_multiplier"] == 2.5) & (df_agg["model"] == "Baseline")].iloc[0]
    if base_25["std_ece"] < cali_25["std_ece"] * 0.5:
        print(
            "\nNote: Baseline (DTI=1.0) shows substantially less variance than CALI-PRED "
            "at the 2.5x multiplier. This suggests an interaction between the sigma LR "
            "and the DTI trust-inflation mechanism specifically, not a general "
            "sigma-head training issue."
        )
    else:
        print(
            "\nNote: Baseline and CALI-PRED exhibit similar variance levels at 2.5x, "
            "pointing to a general issue with the sigma-head learning rate itself."
        )
    print("=" * 80)

    # Plot loss curves in a 2x2 grid (Seeds x Multipliers)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    
    # Grid indexing: 
    # Row 0: Seed 42, Col 0: mult 1.0, Col 1: mult 2.5
    # Row 1: Seed 456, Col 0: mult 1.0, Col 1: mult 2.5
    seed_idx_map = {42: 0, 456: 1}
    mult_idx_map = {1.0: 0, 2.5: 1}
    
    for (mult, seed, m_type), losses in plot_histories.items():
        row = seed_idx_map[seed]
        col = mult_idx_map[mult]
        axes[row, col].plot(range(1, len(losses) + 1), losses, label=f"{m_type}", marker='o')
        axes[row, col].set_title(f"Seed {seed} | mult={mult}")
        axes[row, col].set_xlabel("Epoch")
        axes[row, col].set_ylabel("Validation Loss")
        axes[row, col].legend()
        axes[row, col].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(args.checkpoint_dir, "sigma_lr_ablation.png")
    plt.savefig(plot_path)
    logger.info("Diagnostic plot saved to '%s'.", plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study: Decoupled Sigma Learning Rate Multiplier",
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
    parser.add_argument("--stride", type=int, default=100) # Stride 100 for speed
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
    parser.add_argument("--max-windows", type=int, default=None)

    args = parser.parse_args()
    main(args)
