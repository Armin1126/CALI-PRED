"""
verify_training_history.py

Audit the saved training history for the epoch-4 duplicate value bug.
Loads checkpoints/training_history.npz and checks for suspicious exact
matches between CALI-PRED and Baseline val_loss at any epoch.

Usage
-----
    python verify_training_history.py
    python verify_training_history.py --history-path checkpoints/training_history.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def verify(history_path: str) -> bool:
    """
    Load training history and audit for duplicate val_loss values.

    Returns True if no issues found, False if suspicious duplicates detected.
    """
    try:
        data = np.load(history_path)
    except FileNotFoundError:
        print(f"[ERROR] History file not found: '{history_path}'")
        print("  Run `python pipeline.py` first to generate training history.")
        return False

    cali_train = data.get("calipred_train_loss")
    cali_val = data.get("calipred_val_loss")
    base_train = data.get("baseline_train_loss")
    base_val = data.get("baseline_val_loss")

    if any(x is None for x in [cali_train, cali_val, base_train, base_val]):
        print("[ERROR] History file is missing expected keys.")
        return False

    n_epochs = len(cali_val)
    print(f"Training history: {n_epochs} epochs\n")

    # Print full table
    print(f"{'Epoch':>6} | {'Baseline Train':>14} | {'Baseline Val':>12} | "
          f"{'CALI-PRED Train':>15} | {'CALI-PRED Val':>13} | {'Match?':>7}")
    print("-" * 85)

    issues_found = False
    for epoch in range(n_epochs):
        bt = float(base_train[epoch]) if epoch < len(base_train) else float("nan")
        bv = float(base_val[epoch]) if epoch < len(base_val) else float("nan")
        ct = float(cali_train[epoch]) if epoch < len(cali_train) else float("nan")
        cv = float(cali_val[epoch]) if epoch < len(cali_val) else float("nan")

        # Check for exact float equality between val losses
        is_exact_match = np.isclose(bv, cv, rtol=0, atol=1e-12)
        match_flag = " ⚠ YES" if is_exact_match else ""

        if is_exact_match:
            issues_found = True

        print(f"{epoch + 1:>6} | {bt:>14.6f} | {bv:>12.6f} | "
              f"{ct:>15.6f} | {cv:>13.6f} | {match_flag:>7}")

    print()

    # Summary statistics
    print("=" * 60)
    if issues_found:
        matches = np.where(
            np.isclose(base_val[:n_epochs], cali_val[:n_epochs], rtol=0, atol=1e-12)
        )[0]
        print(f"[WARNING] SUSPICIOUS: Exact val_loss matches at epoch(s): "
              f"{[int(m) + 1 for m in matches]}")
        print("  The probability of two independently-trained models producing")
        print("  exactly identical float-precision val_loss is effectively zero.")
        print("  Possible causes:")
        print("    1. Models shared state (e.g., same checkpoint was loaded for both)")
        print("    2. Training loop bug (e.g., the baseline was accidentally trained")
        print("       with real DTI at that epoch, or vice versa)")
        print("    3. Values were manually transcribed incorrectly into the report")
        print("    4. Both models hit the same early-stopping checkpoint edge case")
    else:
        print("[OK] No suspicious duplicate val_loss values found.")

    # Check for non-monotonic convergence (another potential issue)
    for name, vals in [("Baseline", base_val), ("CALI-PRED", cali_val)]:
        increases = []
        for i in range(1, len(vals)):
            if vals[i] > vals[i - 1]:
                increases.append((i, float(vals[i - 1]), float(vals[i])))
        if increases:
            print(f"\n[INFO] {name} val_loss increased at epochs:")
            for epoch, prev, curr in increases:
                print(f"  Epoch {epoch + 1}: {prev:.6f} -> {curr:.6f} "
                      f"(+{curr - prev:.6f})")

    print("=" * 60)
    return not issues_found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit CALI-PRED training history for the epoch-4 duplicate bug.",
    )
    parser.add_argument(
        "--history-path", type=str, default="checkpoints/training_history.npz",
        help="Path to the training history NPZ file.",
    )
    args = parser.parse_args()

    ok = verify(args.history_path)
    sys.exit(0 if ok else 1)
