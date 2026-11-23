import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from calibrate_saved_predictions import fit_validation_calibration
from data_loader import IndustrialDataLoader, _SUPPORTED_DATASETS, create_dataloaders
from evaluate_saved_predictions import _load_arrays, evaluate
from metrics_engine import calculate_ece, calculate_gaussian_crps, expected_calibration_curve, gaussian_interval
from pipeline import validate_prediction_artifact
from predictor import CaliPredTransformer, TrustCalibratedLoss


def _write_metropt_csv(path, rows=240, include_missing=False):
    frame = pd.DataFrame({
        name: np.sin(np.arange(rows, dtype=float) / (i + 2)) + i
        for i, name in enumerate(_SUPPORTED_DATASETS["metropt"])
    })
    frame.insert(0, "timestamp", pd.date_range("2025-01-01", periods=rows, freq="min"))
    if include_missing:
        frame.loc[5, _SUPPORTED_DATASETS["metropt"][0]] = np.nan
    frame.to_csv(path, index=False)


class DevelopmentWorkflowTests(unittest.TestCase):
    def test_loader_schema_timestamps_shapes_and_train_only_scaler(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metropt.csv"
            _write_metropt_csv(path)
            train, val, test, train_loader, _, _ = create_dataloaders(
                "metropt", str(path), window_size=12, stride=5, batch_size=4, random_state=3
            )
            x, target, timestamps = next(iter(train_loader))
            self.assertEqual(tuple(x.shape), (4, 12, 15))
            self.assertEqual(tuple(target.shape), (4, 12, 15))
            self.assertEqual(tuple(timestamps.shape), (4, 12))
            self.assertEqual(train.sensor_cols, list(_SUPPORTED_DATASETS["metropt"]))
            self.assertLess(train.timestamps[-1], val.timestamps[0])
            self.assertLess(val.timestamps[-1], test.timestamps[0])
            expected = StandardScaler().fit(pd.read_csv(path).loc[:167, list(_SUPPORTED_DATASETS["metropt"])].to_numpy())
            np.testing.assert_allclose(train.scaler.mean_, expected.mean_)

    def test_missingness_mask_and_natural_missingness_row_alignment(self):
        loader = IndustrialDataLoader(random_state=7)
        clean = np.arange(120, dtype=np.float32).reshape(40, 3)
        corrupted, mask = loader.inject_missingness(clean, mechanism="MCAR", missing_rate=0.2, block_size=3)
        self.assertEqual(corrupted.shape, mask.shape)
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))
        self.assertTrue(np.array_equal(np.isnan(corrupted), mask == 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            _write_metropt_csv(path, include_missing=True)
            train, val, test, *_ = create_dataloaders(
                "metropt", str(path), window_size=12, stride=5, batch_size=4,
            )
            self.assertTrue(np.isfinite(train.X).all())
            self.assertTrue(np.isfinite(val.X).all())
            self.assertTrue(np.isfinite(test.X).all())
            self.assertLess(train.timestamps[-1], val.timestamps[0])
            self.assertLess(val.timestamps[-1], test.timestamps[0])

    def test_model_loss_gradients_and_checkpoint_round_trip(self):
        torch.manual_seed(4)
        model = CaliPredTransformer(input_dim=3, output_dim=3, d_model=8, n_heads=2, n_layers=1)
        x = torch.randn(2, 6, 3)
        dti = torch.full((2, 6), 0.8)
        mu, sigma, _ = model(x, dti)
        self.assertEqual(tuple(mu.shape), (2, 6, 3))
        self.assertEqual(tuple(sigma.shape), (2, 6, 3))
        self.assertTrue(torch.isfinite(mu).all() and torch.isfinite(sigma).all() and torch.all(sigma > 0))
        loss, _ = TrustCalibratedLoss()(mu[:, -1:], sigma[:, -1:], torch.randn(2, 1, 3))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients and all(torch.isfinite(grad).all() for grad in gradients))
        model.eval()
        with torch.no_grad():
            mu, sigma, _ = model(x, dti)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict()}, path)
            restored = CaliPredTransformer(input_dim=3, output_dim=3, d_model=8, n_heads=2, n_layers=1)
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
            restored.eval()
            with torch.no_grad():
                mu2, sigma2, _ = restored(x, dti)
            torch.testing.assert_close(mu, mu2)
            torch.testing.assert_close(sigma, sigma2)

    def test_metrics_crps_ece_and_interval_coverage_width(self):
        y, mu, sigma = np.array([0.0]), np.array([0.0]), np.array([1.0])
        expected_crps = 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
        self.assertAlmostEqual(calculate_gaussian_crps(y, mu, sigma), expected_crps, places=10)
        self.assertAlmostEqual(calculate_ece(y, gaussian_interval(mu, sigma, 0.9), 0.9), 0.1, places=10)
        levels, coverage, ece = expected_calibration_curve(y, mu, sigma, nominal_levels=(0.9,))
        self.assertEqual(levels.tolist(), [0.9])
        self.assertEqual(coverage.tolist(), [1.0])
        self.assertAlmostEqual(ece, 0.1, places=10)
        interval = gaussian_interval(mu, sigma, 0.9)
        self.assertLess(interval[0, 0], 0)
        self.assertGreater(interval[0, 1], 0)
        self.assertAlmostEqual(interval[0, 1] - interval[0, 0], 2 * 1.6448536269514722, places=10)

    def test_prediction_artifact_contract_rejects_invalid_sigma_and_shapes(self):
        good = {"y_true": np.zeros(2), "mu": np.zeros(2), "sigma": np.ones(2),
                "dti": np.ones(2), "target_timestamp": np.arange(2)}
        validate_prediction_artifact(good, "one_step_forecast")
        invalid = dict(good, sigma=np.array([1.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            validate_prediction_artifact(invalid, "one_step_forecast")
        invalid_shape = dict(good, dti=np.ones(3))
        with self.assertRaisesRegex(ValueError, "different shapes"):
            validate_prediction_artifact(invalid_shape, "one_step_forecast")

    def test_saved_prediction_calibration_uses_validation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation_predictions.npz"
            y = np.linspace(-1, 1, 40)
            mu = y + 0.1
            sigma = np.full_like(y, 0.5)
            np.savez(path, prediction_unit=np.array("one_step_forecast"), dataset=np.array("metropt"),
                     seed=np.array(42), y_true=y, mu=mu, sigma=sigma,
                     baseline_y_true=y, baseline_mu=mu + 0.2, baseline_sigma=sigma * 1.2)
            calibration = fit_validation_calibration(path)
            self.assertEqual(calibration["source_split"], "validation")
            self.assertIn("0.9", calibration["models"]["calipred"]["split_conformal"])
            loaded = _load_arrays(path, "validation", "calipred")
            scored = evaluate(*loaded)
            self.assertIn("gaussian_crps", scored)
            self.assertIn("interval_coverage", scored)

