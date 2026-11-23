"""Fit calibration transforms using validation predictions only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


def fit_validation_calibration(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if str(data["prediction_unit"].item()) != "one_step_forecast":
            raise ValueError("Expected a one_step_forecast validation artifact.")
        required = ("y_true", "mu", "sigma", "baseline_y_true", "baseline_mu", "baseline_sigma")
        if any(key not in data for key in required):
            raise ValueError("Calibration requires the standalone validation_predictions.npz artifact.")
        arrays = {key: np.asarray(data[key], dtype=np.float64) for key in required}
        dataset = str(data["dataset"].item()) if "dataset" in data else None
        seed = int(data["seed"].item()) if "seed" in data else None

    result = {"source_split": "validation", "prediction_unit": "one_step_forecast",
              "dataset": dataset, "seed": seed, "models": {}}
    for model, prefix in (("calipred", ""), ("baseline", "baseline_")):
        y = arrays[prefix + "y_true"]
        mu = arrays[prefix + "mu"]
        sigma = arrays[prefix + "sigma"]
        if not (y.shape == mu.shape == sigma.shape) or y.size == 0:
            raise ValueError(f"Invalid {model} validation array shapes or empty arrays.")
        if not all(np.isfinite(value).all() for value in (y, mu, sigma)) or np.any(sigma <= 0):
            raise ValueError(f"Invalid {model} validation predictions or uncertainty values.")
        scale = float(np.sqrt(np.mean(np.square((y - mu) / sigma))))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid validation sigma scale for {model}: {scale}.")
        scores = np.sort(np.abs(y - mu) / (sigma * scale))
        conformal = {}
        for level in (0.5, 0.8, 0.9, 0.95):
            k = int(np.ceil((scores.size + 1) * level))
            q = float(scores[k - 1]) if k <= scores.size else None
            conformal[str(level)] = {"normalized_residual_quantile": q,
                                     "sample_count": int(scores.size),
                                     "one_based_order_statistic": k}
        result["models"][model] = {"nll_sigma_scale": scale, "split_conformal": conformal}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output", default="artifacts/results/calibration.json")
    args = parser.parse_args()
    calibration = fit_validation_calibration(args.validation_file)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(f"Validation-only calibration saved to {output}")


if __name__ == "__main__":
    main()
