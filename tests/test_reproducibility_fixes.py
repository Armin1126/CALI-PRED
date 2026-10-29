import numpy as np
import pandas as pd
import pytest

from data_loader import IndustrialDataLoader, RealDataset, create_dataloaders, _SUPPORTED_DATASETS
from dqa_module import UpstreamDQAEngine
from iri_module import ImputationReliabilityEngine


def test_missing_dataset_fails_closed(tmp_path):
    loader = IndustrialDataLoader()
    with pytest.raises(FileNotFoundError, match="fail closed"):
        loader.load_iiot_data("metropt", str(tmp_path / "missing.csv"))


def test_real_dataset_timestamps_follow_dropped_sensor_rows(tmp_path):
    n = 12
    cols = list(_SUPPORTED_DATASETS["metropt"])
    df = pd.DataFrame({c: np.arange(n, dtype=float) + j for j, c in enumerate(cols)})
    df.insert(0, "timestamp", pd.date_range("2026-01-01", periods=n, freq="s"))
    df.loc[4, cols[0]] = np.nan
    path = tmp_path / "metropt.csv"
    df.to_csv(path, index=False)
    ds = RealDataset("metropt", str(path), window_size=3, stride=1, forecast_horizon=1)
    assert ds.timestamps[4] == pytest.approx(pd.Timestamp("2026-01-01 00:00:05").timestamp())
    assert len(ds.timestamps) == len(ds.X) == n - 1
    x, target, _ = ds[0]
    assert target[-1, 0].item() == pytest.approx(ds.X[3, 0].item())
    assert x[-1, 0].item() == pytest.approx(ds.X[2, 0].item())


def test_freshness_penalizes_a_stale_channel():
    engine = UpstreamDQAEngine(freshness_tau_seconds=1.0)
    ts = np.array([0.0, 1.0, 2.0])
    mask = np.array([[1, 1], [1, 1], [1, 0]], dtype=np.int8)
    score = engine.freshness(ts, inference_time=2.0, observation_mask=mask)
    assert score == pytest.approx((1.0 + np.exp(-1.0)) / 2.0)


def test_iri_reconstruction_error_changes_absolute_score():
    ensemble = np.array([[[0.0], [0.0]], [[0.2], [0.2]]], dtype=np.float64)
    holdout = np.array([[1], [0]], dtype=np.int8)
    good_truth = np.array([[0.1], [0.0]])
    bad_truth = np.array([[10.0], [0.0]])
    good = ImputationReliabilityEngine.compute_iri(ensemble, good_truth, holdout)
    bad = ImputationReliabilityEngine.compute_iri(ensemble, bad_truth, holdout)
    assert np.all(good >= bad)
    assert not np.allclose(good, bad)


def test_bootstrap_supports_one_step_forecast_artifacts(tmp_path):
    from bootstrap_analysis import run_bootstrap_analysis

    rng = np.random.default_rng(4)
    y = rng.normal(size=40)
    mu_c = y + rng.normal(scale=0.2, size=40)
    mu_b = y + rng.normal(scale=0.3, size=40)
    path = tmp_path / "test_predictions.npz"
    np.savez(
        path,
        calipred_y_true=y, calipred_mu=mu_c, calipred_sigma=np.full(40, 0.5),
        baseline_y_true=y, baseline_mu=mu_b, baseline_sigma=np.full(40, 0.6),
        prediction_unit=np.asarray("one_step_forecast"), n_features=np.asarray(2),
    )
    result = run_bootstrap_analysis(str(path), n_bootstrap=10, n_features=15)
    assert "ece" in result and "crps" in result


def test_create_dataloaders_keeps_timestamps_aligned_after_dropna(tmp_path):
    n = 40
    cols = list(_SUPPORTED_DATASETS["metropt"])
    rng = np.random.default_rng(12)
    df = pd.DataFrame({c: rng.normal(size=n) for c in cols})
    df.insert(0, "timestamp", pd.date_range("2026-01-01", periods=n, freq="s"))
    df.loc[5, cols[2]] = np.nan
    path = tmp_path / "metropt_split.csv"
    df.to_csv(path, index=False)
    train, val, test, *_ = create_dataloaders(
        "metropt", str(path), window_size=3, stride=1, forecast_horizon=1,
        batch_size=4, train_frac=0.6, val_frac=0.2,
    )
    assert train.timestamps[5] == pytest.approx(pd.Timestamp("2026-01-01 00:00:06").timestamp())
    assert len(train.timestamps) == len(train.X)
    assert len(val.timestamps) == len(val.X)
    assert len(test.timestamps) == len(test.X)



def test_reusable_iri_ensemble_fits_once_and_supports_batch_inference():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(4, 6, 2)).astype(np.float32)
    engine = ImputationReliabilityEngine(
        n_features=2, d_model=8, epochs=1, holdout_frac=0.2,
        random_state=7, device="cpu",
    )
    engine.fit_reusable_ensemble(X, batch_size=2, epochs=1)
    assert engine._fitted_models is not None

    mask = np.ones_like(X, dtype=np.int8)
    mask[:, 2, 0] = 0
    corrupted = X.copy()
    corrupted[mask == 0] = np.nan
    out = engine.impute_ensemble_batch(corrupted, mask)
    assert out.shape == (2, 4, 6, 2)
    assert np.isfinite(out).all()
    for member in range(2):
        assert np.allclose(out[member][mask.astype(bool)], X[mask.astype(bool)])
