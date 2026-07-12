"""
ablation_calibration_weight.py

CALI-PRED Study: Ablation study of the TrustCalibratedLoss's pinball calibration weight
to isolate systemic over-coverage on high-trust data.

Usage:
------
    python ablation_calibration_weight.py --data-path "data/metropt/MetroPT3(AirCompressor).csv" --epochs 15
    python ablation_calibration_weight.py --data-path "data/metropt/MetroPT3(chiller).csv" --epochs 15
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("CalibrationAblation")

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
)

def run_ablation(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Ablation study running on device: %s", device)

    # 1. Load data
    logger.info("Loading dataset '%s' from '%s'...", args.dataset, args.data_path)
    train_ds, val_ds, test_ds, _, _, _ = create_dataloaders(
        dataset_name=args.dataset,
        file_path=args.data_path,
        window_size=args.window_size,
        stride=args.stride,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        random_state=42,
    )
    n_features = train_ds.n_features
    logger.info(
        "Data loaded: %d features, train=%d windows, val=%d windows, test=%d windows.",
        n_features, len(train_ds), len(val_ds), len(test_ds),
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

    # 2. Initialize pipeline components
    dqa_engine = UpstreamDQAEngine(freshness_tau_seconds=60.0, max_corr_mae=0.5)
    iri_engine = ImputationReliabilityEngine(
        n_features=n_features, epochs=30, holdout_frac=0.15, random_state=42,
    )
    fusion_engine = TrustFusionEngine(clamp_inputs=True)
    corruption_loader = IndustrialDataLoader(random_state=42)

    # Compute baseline correlation matrix
    if hasattr(train_ds, "X"):
        train_X = train_ds.X
    else:
        train_X = train_ds.dataset.X
    baseline_corr = np.corrcoef(train_X.T)
    baseline_corr = np.nan_to_num(baseline_corr, nan=0.0)

    # Initialize severity samplers
    train_val_sampler = make_severity_sampler(
        clean_fraction=args.clean_fraction,
        max_severity=args.max_severity,
        random_state=42,
    )
    test_sampler = make_severity_sampler(
        clean_fraction=args.clean_fraction,
        max_severity=args.max_severity,
        random_state=100,
    )

    # 3. Precompute DTI and Imputations once
    logger.info("Precomputing DTI and Imputations for splits...")
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

    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    weights_to_test = [0.0, 0.1, 0.2]
    results = []

    # 4. Loop over weights
    for cw in weights_to_test:
        logger.info("=" * 80)
        logger.info("Testing calibration_weight = %.1f", cw)
        logger.info("=" * 80)

        # Output subdirectory
        ckpt_dir = os.path.join(args.checkpoint_dir, f"ablation_cw_{cw}")
        os.makedirs(ckpt_dir, exist_ok=True)

        loss_fn = TrustCalibratedLoss(lower_q=0.05, upper_q=0.95, calibration_weight=cw)

        # Run 1: CALI-PRED
        logger.info("Training CALI-PRED with calibration_weight=%.1f...", cw)
        model_calipred = CaliPredTransformer(
            input_dim=n_features, output_dim=n_features,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            dropout=0.1, max_uncertainty_inflation=args.max_inflation,
            alpha_init=args.alpha_init,
        )
        train_model(
            model=model_calipred, loss_fn=loss_fn,
            train_loader=train_loader, val_loader=val_loader,
            dqa_engine=dqa_engine, iri_engine=iri_engine, fusion_engine=fusion_engine,
            corruption_loader=corruption_loader, baseline_corr=baseline_corr,
            n_features=n_features, epochs=args.epochs, lr=args.lr,
            device=device, checkpoint_dir=ckpt_dir, use_real_dti=True,
            missing_rate_sampler=train_val_sampler,
        )
        # Load best and evaluate
        ckpt_path_cali = os.path.join(ckpt_dir, "best_model_calipred.pt")
        if os.path.exists(ckpt_path_cali):
            model_calipred.load_state_dict(
                torch.load(ckpt_path_cali, map_location=device, weights_only=False)["model_state_dict"]
            )
        eval_cali = evaluate_model(
            model_calipred, test_loader,
            dqa_engine, iri_engine, fusion_engine,
            corruption_loader, baseline_corr, n_features,
            device=device, use_real_dti=True, label="CALI-PRED",
            missing_rate_sampler=test_sampler,
        )
        cov_50_cali = float(eval_cali["empirical_coverage"][0])
        gap_50_cali = cov_50_cali - 0.50
        results.append({
            "calibration_weight": cw,
            "model": "CALI-PRED",
            "mean_ece": float(eval_cali["mean_ece"]),
            "brier_score": float(eval_cali["brier_score"]),
            "coverage_at_nominal_50": cov_50_cali,
            "gap_at_nominal_50": gap_50_cali
        })

        # Run 2: Baseline
        logger.info("Training Baseline with calibration_weight=%.1f...", cw)
        model_baseline = CaliPredTransformer(
            input_dim=n_features, output_dim=n_features,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            dropout=0.1, max_uncertainty_inflation=args.max_inflation,
            alpha_init=args.alpha_init,
        )
        train_model(
            model=model_baseline, loss_fn=loss_fn,
            train_loader=train_loader, val_loader=val_loader,
            dqa_engine=dqa_engine, iri_engine=iri_engine, fusion_engine=fusion_engine,
            corruption_loader=corruption_loader, baseline_corr=baseline_corr,
            n_features=n_features, epochs=args.epochs, lr=args.lr,
            device=device, checkpoint_dir=ckpt_dir, use_real_dti=False,
            missing_rate_sampler=train_val_sampler,
        )
        # Load best and evaluate
        ckpt_path_base = os.path.join(ckpt_dir, "best_model_baseline.pt")
        if os.path.exists(ckpt_path_base):
            model_baseline.load_state_dict(
                torch.load(ckpt_path_base, map_location=device, weights_only=False)["model_state_dict"]
            )
        eval_base = evaluate_model(
            model_baseline, test_loader,
            dqa_engine, iri_engine, fusion_engine,
            corruption_loader, baseline_corr, n_features,
            device=device, use_real_dti=False, label="Baseline",
            missing_rate_sampler=test_sampler,
        )
        cov_50_base = float(eval_base["empirical_coverage"][0])
        gap_50_base = cov_50_base - 0.50
        results.append({
            "calibration_weight": cw,
            "model": "Baseline",
            "mean_ece": float(eval_base["mean_ece"]),
            "brier_score": float(eval_base["brier_score"]),
            "coverage_at_nominal_50": cov_50_base,
            "gap_at_nominal_50": gap_50_base
        })

    # 5. Output summary table
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.checkpoint_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Ablation results saved to '%s'.", csv_path)

    # Print summary table
    print("\n" + "=" * 100)
    print("  CALIBRATION WEIGHT ABLATION RESULTS SUMMARY")
    print("=" * 100)
    # Manual markdown formatting to avoid tabulate dependency
    cols = list(df.columns)
    header = " | ".join(cols)
    separator = " | ".join(["---"] * len(cols))
    print(f"| {header} |")
    print(f"| {separator} |")
    for _, row in df.iterrows():
        row_str = " | ".join(
            f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating))
            else str(row[c])
            for c in cols
        )
        print(f"| {row_str} |")
    print("=" * 100 + "\n")

    # 6. Explicit Interpretation
    # Extract gaps at 0.2 and 0.0
    gap_cali_02 = df[(df["calibration_weight"] == 0.2) & (df["model"] == "CALI-PRED")]["gap_at_nominal_50"].values[0]
    gap_base_02 = df[(df["calibration_weight"] == 0.2) & (df["model"] == "Baseline")]["gap_at_nominal_50"].values[0]
    gap_cali_00 = df[(df["calibration_weight"] == 0.0) & (df["model"] == "CALI-PRED")]["gap_at_nominal_50"].values[0]
    gap_base_00 = df[(df["calibration_weight"] == 0.0) & (df["model"] == "Baseline")]["gap_at_nominal_50"].values[0]

    print("=" * 100)
    print("  INTERPRETATION & DIAGNOSTIC REPORT")
    print("=" * 100)
    print(f"CALI-PRED gap at cw=0.2: {gap_cali_02:+.4f} | cw=0.0: {gap_cali_00:+.4f}")
    print(f"Baseline gap at cw=0.2:  {gap_base_02:+.4f} | cw=0.0: {gap_base_00:+.4f}")
    print("-" * 100)

    # Check if gap shrinks by more than half for both
    # Shrunk criteria: gap at 0.0 is <= 0.5 * gap at 0.2
    # Since they are positive gaps, we can compare absolute gaps.
    shrunk_cali = abs(gap_cali_00) <= 0.5 * abs(gap_cali_02)
    shrunk_base = abs(gap_base_00) <= 0.5 * abs(gap_base_02)

    if shrunk_cali and shrunk_base:
        print("[CONFIRMED] The pinball calibration term in TrustCalibratedLoss is a primary driver of the systemic "
              "over-coverage. Consider reducing calibration_weight or reweighting the 0.05/0.95 pinball terms asymmetrically.")
    else:
        # Check if gap stays large (>0.15) even at 0.0
        large_cali = abs(gap_cali_00) > 0.15
        large_base = abs(gap_base_00) > 0.15
        if large_cali or large_base:
            print("[NOT CONFIRMED] Over-coverage persists even with pure NLL loss (no pinball term). The systemic "
                  "issue is NOT primarily the calibration term -- investigate the softplus base-sigma parameterization, "
                  "sigma_floor value, or whether the sigma head is undertrained relative to the mu head (check if "
                  "sigma converges much slower than mu during training by logging them separately).")
        else:
            print("[MIXED FINDINGS] Over-coverage decreased but did not meet full confirmation thresholds. Investigate "
                  "both the calibration weight and base parameterization.")

    print("-" * 100)
    # Check if CALI-PRED gap is proportionally larger than Baseline gap at each weight
    # proportional: CALI-PRED gap > Baseline gap (in absolute terms)
    gaps_larger = []
    for cw in weights_to_test:
        g_cali = abs(df[(df["calibration_weight"] == cw) & (df["model"] == "CALI-PRED")]["gap_at_nominal_50"].values[0])
        g_base = abs(df[(df["calibration_weight"] == cw) & (df["model"] == "Baseline")]["gap_at_nominal_50"].values[0])
        if g_cali > g_base:
            gaps_larger.append(f"cw={cw:.1f} (Cali={g_cali:.4f} > Base={g_base:.4f})")
    
    if len(gaps_larger) == len(weights_to_test):
        print(f"CALI-PRED's gap is proportionally larger than Baseline's gap at ALL weights tested: {', '.join(gaps_larger)}. "
              "This suggests that the DTI inflation mechanism compounds the already-present systemic issue, rather than "
              "being unrelated to it.")
    elif len(gaps_larger) > 0:
        print(f"CALI-PRED's gap is proportionally larger than Baseline's gap at SOME weights: {', '.join(gaps_larger)}. "
              "The DTI mechanism has a compounding effect under certain loss configurations.")
    else:
        print("CALI-PRED's gap is NOT larger than Baseline's gap at any weight tested. The DTI mechanism is unrelated "
              "to the systemic over-coverage issue.")

    print("-" * 100)
    print("Note: ablation runs use reduced epoch count for speed; confirm trend holds with full training "
          "before finalizing conclusions in the report.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CALI-PRED Calibration Weight Ablation Study")
    parser.add_argument("--dataset", type=str, default="metropt", choices=["metropt", "ai4i2020", "tep"])
    parser.add_argument("--data-path", type=str, default="data/metropt/MetroPT3(AirCompressor).csv")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--max-inflation", type=float, default=10.0)
    parser.add_argument("--alpha-init", type=float, default=0.5)
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--clean-fraction", type=float, default=0.25)
    parser.add_argument("--max-severity", type=float, default=0.45)

    args = parser.parse_args()
    run_ablation(args)
