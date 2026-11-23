"""Evaluate a saved validation or test prediction artifact without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from metrics_engine import calculate_gaussian_crps, expected_calibration_curve


def _load_arrays(path: str | Path, split: str, model: str):
    with np.load(path, allow_pickle=False) as data:
        if str(data["prediction_unit"].item()) != "one_step_forecast":
            raise ValueError("Expected a one_step_forecast prediction artifact.")
        if split == "validation":
            prefix = "" if model == "calipred" else "baseline_"
            keys = (prefix + "y_true", prefix + "mu", prefix + "sigma")
        else:
            prefix = "calipred_" if model == "calipred" else "baseline_"
            keys = (prefix + "y_true", prefix + "mu", prefix + "sigma")
        missing = [key for key in keys if key not in data]
        if missing:
            raise ValueError(f"Artifact does not contain {split}/{model} arrays: {missing}")
        y, mu, sigma = (np.asarray(data[key], dtype=np.float64).reshape(-1) for key in keys)
    if not (y.shape == mu.shape == sigma.shape) or y.size == 0:
        raise ValueError("Prediction and target arrays must have the same non-empty shape.")
    if not all(np.isfinite(value).all() for value in (y, mu, sigma)) or np.any(sigma <= 0):
        raise ValueError("Predictions must be finite and sigma must be positive.")
    return y, mu, sigma


def evaluate(y, mu, sigma, sigma_scale: float = 1.0, conformal_q: float | None = None) -> dict:
    if not np.isfinite(sigma_scale) or sigma_scale <= 0:
        raise ValueError("sigma_scale must be finite and positive.")
    sigma = sigma * sigma_scale
    levels = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    _, _, ece = expected_calibration_curve(y, mu, sigma, nominal_levels=levels)
    z90 = float(norm.ppf(0.95))
    radius = z90 * sigma if conformal_q is None else conformal_q * sigma
    result = {"mae": float(np.mean(np.abs(y - mu))), "ece": float(ece),
              "ece_definition": "mean absolute nominal-vs-empirical central Gaussian interval coverage gap across levels 0.50–0.95",
              "gaussian_crps": calculate_gaussian_crps(y, mu, sigma),
              "interval_nominal_coverage": 0.90,
              "interval_coverage": float(np.mean(np.abs(y - mu) <= radius)),
              "mean_interval_width": float(np.mean(2 * radius)),
              "interval_method": "central Gaussian 90%" if conformal_q is None else "validation-fitted normalized-residual split conformal"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--model", choices=("calipred", "baseline"), default="calipred")
    parser.add_argument("--calibration-file", help="Optional JSON produced by calibrate_saved_predictions.py")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    y, mu, sigma = _load_arrays(args.prediction_file, args.split, args.model)
    scale, q = 1.0, None
    if args.calibration_file:
        if args.split != "test":
            raise ValueError("Apply validation-fitted calibration only when evaluating the frozen test split.")
        calibration = json.loads(Path(args.calibration_file).read_text(encoding="utf-8"))
        if calibration.get("source_split") != "validation":
            raise ValueError("Calibration file must declare validation as its source split.")
        model_cal = calibration["models"][args.model]
        scale = model_cal["nll_sigma_scale"]
        q_record = model_cal["split_conformal"]["0.9"]["normalized_residual_quantile"]
        if q_record is None:
            raise ValueError("Validation sample is too small for a finite 90% split-conformal quantile.")
        q = q_record
    metrics = evaluate(y, mu, sigma, scale, q)
    metrics.update({"split": args.split, "model": args.model, "sample_count": int(y.size),
                    "prediction_file": str(args.prediction_file),
                    "calibration_file": args.calibration_file})
    output = json.dumps(metrics, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
