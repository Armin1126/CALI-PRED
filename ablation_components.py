#!/usr/bin/env python3
"""Run component ablations from scratch with matched seeds and data splits."""

import argparse
import json
import os
import subprocess
import sys


VARIANTS = {
    "dqa_only": ["--disable-iri"],
    "iri_only": ["--disable-dqa"],
    "dqa_iri_only": ["--disable-trust-attention", "--disable-uncertainty-inflation"],
    "dqa_iri_sigma": ["--disable-trust-attention"],
    "dqa_iri_attention": ["--disable-uncertainty-inflation"],
    "full_calipred": [],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("metropt", "ai4i2020", "tep"), default="metropt")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/component_ablations")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for variant, switches in VARIANTS.items():
        for seed in args.seeds:
            run_dir = os.path.join(args.checkpoint_dir, variant, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            manifest = {
                "variant": variant,
                "dataset": args.dataset,
                "seed": seed,
                "epochs": args.epochs,
                "stride": args.stride,
                "max_windows": args.max_windows,
                "flags": switches,
                "trained_from_scratch": True,
                "checkpoint_selection": "minimum validation MAE after warmup",
            }
            manifest_path = os.path.join(run_dir, "ablation_config.json")
            temporary = f"{manifest_path}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
            os.replace(temporary, manifest_path)

            command = [
                sys.executable, "pipeline.py",
                "--dataset", args.dataset,
                "--data-path", args.data_path,
                "--stride", str(args.stride),
                "--epochs", str(args.epochs),
                "--checkpoint-dir", run_dir,
                "--seed", str(seed),
                *switches,
            ]
            if args.max_windows is not None:
                command.extend(("--max-windows", str(args.max_windows)))
            print(f"Running {variant} seed={seed}: {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)
            artifact = os.path.join(run_dir, "test_predictions.npz")
            if not os.path.isfile(artifact):
                raise FileNotFoundError(f"Run completed without predictions: {artifact}")


if __name__ == "__main__":
    main()
