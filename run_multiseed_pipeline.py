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
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    pred_paths = []

    for i, seed in enumerate(args.seeds):
        logger.info("=" * 80)
        logger.info("STARTING SEED RUN %d/%d (Seed = %d)", i + 1, len(args.seeds), seed)
        logger.info("=" * 80)

        seed_ckpt_dir = os.path.join(args.checkpoint_dir, f"seed_{seed}")
        os.makedirs(seed_ckpt_dir, exist_ok=True)

        cmd = [
            sys.executable, "pipeline.py",
            "--dataset", args.dataset,
            "--data-path", args.data_path,
            "--stride", str(args.stride),
            "--epochs", str(args.epochs),
            "--checkpoint-dir", seed_ckpt_dir,
            "--sigma-floor", str(args.sigma_floor),
            "--sigma-lr-multiplier", str(args.sigma_lr_multiplier),
            "--sigma-init-bias", str(args.sigma_init_bias),
            "--apply-validation-sigma-scaling",
            "--seed", str(seed),
        ]
        if args.max_windows is not None:
            cmd.extend(["--max-windows", str(args.max_windows)])

        logger.info("Executing: %s", " ".join(cmd))
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            logger.error("Seed %d failed with return code %d", seed, ret.returncode)
            sys.exit(ret.returncode)

        pred_file = os.path.join(seed_ckpt_dir, "test_predictions.npz")
        if os.path.exists(pred_file):
            pred_paths.append(pred_file)

    logger.info("=" * 80)
    logger.info("ALL SEED RUNS COMPLETE. Running Multi-Seed Coverage Diagnostics...")
    logger.info("=" * 80)

    if pred_paths:
        run_multiseed_diagnostics(pred_paths, save_dir=args.checkpoint_dir)
        logger.info("[OK] Multi-seed diagnostics complete. Figure saved in '%s'.", args.checkpoint_dir)

if __name__ == "__main__":
    main()
