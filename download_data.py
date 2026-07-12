"""
download_data.py

Industrial Anomaly Prediction Framework — Dataset Acquisition Utility
==============================================================================

Downloads and validates the two target benchmark datasets:

    1. **AI4I 2020 Predictive Maintenance** (UCI ML Repository)
       ~10 000 rows, tabular sensor readings + failure-type labels.
       URL: https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip

    2. **MetroPT-3** (Kaggle / UCI)
       Continuous air-production-unit compressor sensor data, well-suited
       for sequence-based models with 60-timestep sliding windows.
       URL: https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip

Files are placed under ``data/ai4i2020/`` and ``data/metropt/`` relative to
the project root.

Usage
-----
    python download_data.py                    # download both
    python download_data.py --dataset metropt  # download only MetroPT
    python download_data.py --dataset ai4i2020 # download only AI4I 2020

Python: 3.13+
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from urllib.error import URLError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("DatasetDownloader")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
_DATASET_CONFIGS = {
    "ai4i2020": {
        "url": "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        "target_dir": DATA_DIR / "ai4i2020",
        "expected_csv": "ai4i2020.csv",
        "expected_columns": [
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        ],
        "min_rows": 9000,
        "description": "AI4I 2020 Predictive Maintenance Dataset (~10k rows, tabular)",
    },
    "metropt": {
        "url": "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip",
        "target_dir": DATA_DIR / "metropt",
        "expected_csv": "MetroPT3(chiller).csv",
        "expected_columns": [
            "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
            "Oil_temperature", "Motor_current",
        ],
        "min_rows": 50000,
        "description": "MetroPT-3 Dataset (compressor sensor time-series)",
    },
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Simple download progress reporter."""
    if total_size > 0:
        downloaded = block_num * block_size
        pct = min(100.0, downloaded / total_size * 100.0)
        sys.stdout.write(f"\r  Downloading: {pct:.1f}% ({downloaded // 1024}KB / {total_size // 1024}KB)")
        sys.stdout.flush()


def download_and_extract(dataset_name: str) -> Optional[Path]:
    """
    Download a dataset ZIP from UCI and extract it into the target directory.

    Parameters
    ----------
    dataset_name : str
        One of ``{"ai4i2020", "metropt"}``.

    Returns
    -------
    Optional[Path]
        Path to the extracted CSV if successful, ``None`` on failure.
    """
    if dataset_name not in _DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {list(_DATASET_CONFIGS.keys())}"
        )

    config = _DATASET_CONFIGS[dataset_name]
    target_dir: Path = config["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    existing_csvs = list(target_dir.glob("*.csv"))
    if existing_csvs:
        logger.info(
            "Found existing CSV(s) in '%s': %s. Skipping download.",
            target_dir, [f.name for f in existing_csvs],
        )
        return existing_csvs[0]

    logger.info(
        "Downloading %s from:\n  %s", config["description"], config["url"]
    )

    zip_path = target_dir / f"{dataset_name}.zip"
    try:
        urlretrieve(config["url"], str(zip_path), reporthook=_progress_hook)
        print()  # newline after progress bar
    except (URLError, OSError) as exc:
        logger.error(
            "Failed to download '%s': %s\n"
            "Please manually download from:\n  %s\n"
            "and place the CSV file(s) into:\n  %s",
            dataset_name, exc, config["url"], target_dir,
        )
        return None

    # Extract
    logger.info("Extracting '%s' to '%s'...", zip_path.name, target_dir)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile as exc:
        logger.error("Bad ZIP file: %s", exc)
        return None
    finally:
        zip_path.unlink(missing_ok=True)

    # Find the CSV — it may be in a subdirectory within the ZIP
    csv_files = list(target_dir.rglob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found after extraction in '%s'.", target_dir)
        return None

    # Move CSVs to the target_dir root if they're nested
    for csv_file in csv_files:
        if csv_file.parent != target_dir:
            dest = target_dir / csv_file.name
            csv_file.rename(dest)
            logger.info("  Moved '%s' → '%s'", csv_file.name, dest)

    final_csvs = list(target_dir.glob("*.csv"))
    logger.info(
        "Extraction complete. CSV files in '%s': %s",
        target_dir, [f.name for f in final_csvs],
    )
    return final_csvs[0] if final_csvs else None


def validate_dataset(dataset_name: str) -> bool:
    """
    Validate that a downloaded dataset has the expected structure.

    Parameters
    ----------
    dataset_name : str
        One of ``{"ai4i2020", "metropt"}``.

    Returns
    -------
    bool
        ``True`` if validation passes, ``False`` otherwise.
    """
    import pandas as pd

    config = _DATASET_CONFIGS[dataset_name]
    target_dir: Path = config["target_dir"]

    csv_files = list(target_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in '%s'.", target_dir)
        return False

    csv_path = csv_files[0]
    logger.info("Validating '%s'...", csv_path)

    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception as exc:
        logger.error("Failed to read CSV '%s': %s", csv_path, exc)
        return False

    # Check for expected columns (flexible: check substrings since
    # UCI column names may vary slightly across versions)
    found_cols = set(df.columns)
    expected = config["expected_columns"]
    missing = []
    for exp_col in expected:
        # Try exact match first, then substring match
        if exp_col not in found_cols:
            matches = [c for c in found_cols if exp_col.lower() in c.lower()]
            if not matches:
                missing.append(exp_col)

    if missing:
        logger.warning(
            "Dataset '%s' is missing expected columns: %s\n"
            "Available columns: %s\n"
            "This may be a different version; the pipeline will attempt "
            "to map columns defensively.",
            dataset_name, missing, list(found_cols),
        )

    # Check row count
    try:
        row_count = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
    except Exception:
        row_count = len(pd.read_csv(csv_path))

    if row_count < config["min_rows"]:
        logger.warning(
            "Dataset '%s' has %d rows (expected >= %d). "
            "May be truncated or a different version.",
            dataset_name, row_count, config["min_rows"],
        )
    else:
        logger.info(
            "Dataset '%s': %d rows, columns look good [OK]",
            dataset_name, row_count,
        )

    return True


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and validate CALI-PRED benchmark datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["metropt", "ai4i2020", "both"],
        default="both",
        help="Which dataset(s) to download (default: both).",
    )
    args = parser.parse_args()

    datasets_to_download = (
        list(_DATASET_CONFIGS.keys()) if args.dataset == "both"
        else [args.dataset]
    )

    all_ok = True
    for ds_name in datasets_to_download:
        print(f"\n{'='*60}")
        print(f"  Dataset: {_DATASET_CONFIGS[ds_name]['description']}")
        print(f"{'='*60}")

        csv_path = download_and_extract(ds_name)
        if csv_path is not None:
            valid = validate_dataset(ds_name)
            if not valid:
                all_ok = False
        else:
            all_ok = False

    if all_ok:
        print("\n[OK] All requested datasets downloaded and validated successfully.")
    else:
        print("\n[WARN] Some datasets had issues. Check the logs above.")
