"""
run_multiseed_pipeline.py

Orchestrates full-pipeline training for CALI-PRED and Baseline across multiple
random seeds, saves per-seed predictions, and automatically runs multi-seed
coverage diagnostics and bootstrap analysis.

Usage:
------
    python run_multiseed_pipeline.py --data-path "/content/CALI-PRED/data/metropt/MetroPT3(AirCompressor).csv" --seeds 42 123 456
"""

import argparse
import logging
import os
import subprocess
import sys
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("MultiSeedOrchestrator")

from multiseed_coverage_diagnostics import run_multiseed_diagnostics

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed CALI-PRED Pipeline Runner")
    parser.add_argument(
        "--data-path", type=str, default="data/metropt/MetroPT3(AirCompressor).csv",
        help="Path to the raw CSV file.",
    )
    parser.add_argument("--dataset", type=str, default="metropt")
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-windows", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--sigma-floor", type=float, default=0.10)
    parser.add_argument("--sigma-lr-multiplier", type=float, default=0.50)
    parser.add_argument("--sigma-init-bias", type=float, default=0.50)
    parser.add_argument("--calibration-weight", type=float, default=0.2)
    parser.add_argument("--iri-epochs", type=int, default=30)
    parser.add_argument("--apply-validation-sigma-scaling", action="store_true",
                        help="Apply validation-fitted uncertainty scaling in the prediction stage.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Root directory for per-seed artifacts; defaults to --checkpoint-dir.")
    parser.add_argument("--mini-run-report", type=str, required=True,
                        help="Successful development run_config.json required before full experiments.")
    args = parser.parse_args()

    artifact_root = args.artifact_dir or args.checkpoint_dir
    os.makedirs(artifact_root, exist_ok=True)
    pred_paths = []

    for i, seed in enumerate(args.seeds):
        logger.info("=" * 80)
        logger.info("STARTING SEED RUN %d/%d (Seed = %d)", i + 1, len(args.seeds), seed)
        logger.info("=" * 80)

        seed_artifact_dir = os.path.join(artifact_root, f"seed_{seed}")
        seed_ckpt_dir = os.path.join(seed_artifact_dir, "checkpoints")
        os.makedirs(seed_ckpt_dir, exist_ok=True)

        cmd = [
            sys.executable, "pipeline.py",
            "--dataset", args.dataset,
            "--data-path", args.data_path,
            "--stride", str(args.stride),
            "--epochs", str(args.epochs),
            "--checkpoint-dir", seed_ckpt_dir,
            "--artifact-dir", seed_artifact_dir,
            "--train-only",
            "--mini-run-report", args.mini_run_report,
            "--sigma-floor", str(args.sigma_floor),
            "--sigma-lr-multiplier", str(args.sigma_lr_multiplier),
            "--sigma-init-bias", str(args.sigma_init_bias),
            "--calibration-weight", str(args.calibration_weight),
            "--iri-epochs", str(args.iri_epochs),
            "--seed", str(seed),
        ]
        if args.max_windows is not None:
            cmd.extend(["--max-windows", str(args.max_windows)])

        logger.info("Executing: %s", " ".join(cmd))
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            logger.error("Seed %d failed with return code %d", seed, ret.returncode)
            raise subprocess.CalledProcessError(ret.returncode, cmd)

        predict_cmd = [
            sys.executable, "pipeline.py",
            "--dataset", args.dataset,
            "--data-path", args.data_path,
            "--stride", str(args.stride),
            "--checkpoint-dir", seed_ckpt_dir,
            "--artifact-dir", seed_artifact_dir,
            "--predict-only",
            "--sigma-floor", str(args.sigma_floor),
            "--sigma-lr-multiplier", str(args.sigma_lr_multiplier),
            "--sigma-init-bias", str(args.sigma_init_bias),
            "--calibration-weight", str(args.calibration_weight),
            "--iri-epochs", str(args.iri_epochs),
            "--seed", str(seed),
        ]
        if args.apply_validation_sigma_scaling:
            predict_cmd.append("--apply-validation-sigma-scaling")
        if args.max_windows is not None:
            predict_cmd.extend(["--max-windows", str(args.max_windows)])
        logger.info("Predicting frozen seed %d checkpoints: %s", seed, " ".join(predict_cmd))
        ret = subprocess.run(predict_cmd)
        if ret.returncode != 0:
            logger.error("Prediction for seed %d failed with return code %d", seed, ret.returncode)
            raise subprocess.CalledProcessError(ret.returncode, predict_cmd)

        pred_file = os.path.join(seed_artifact_dir, "predictions", "test_predictions.npz")
        if not os.path.isfile(pred_file):
            raise FileNotFoundError(f"Seed {seed} completed without prediction artifact: {pred_file}")
        pred_paths.append(pred_file)

    logger.info("=" * 80)
    logger.info("ALL SEED RUNS COMPLETE. Running Multi-Seed Coverage Diagnostics...")
    logger.info("=" * 80)

    if len(pred_paths) != len(args.seeds):
        raise RuntimeError(f"Expected {len(args.seeds)} seed artifacts; found {len(pred_paths)}.")
    run_multiseed_diagnostics(pred_paths, save_dir=artifact_root)
    logger.info("[OK] Multi-seed diagnostics complete. Figure saved in '%s'.", artifact_root)

if __name__ == "__main__":
    main()

