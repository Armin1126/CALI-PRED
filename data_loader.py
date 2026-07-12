"""
data_loader.py

Industrial Anomaly Prediction Framework — Data Ingestion & Corruption Module
==============================================================================

This module provides `IndustrialDataLoader`, a utility class responsible for
the first stage of an industrial time-series anomaly-detection pipeline:

    1. Ingesting multivariate sensor time series from common IIoT benchmark
       formats (MetroPT-3 air-production-unit compressor data, AI4I 2020
       predictive-maintenance data, or the Tennessee Eastman Process (TEP)
       simulation dataset).
    2. Normalizing sensor channels to zero-mean, unit-variance to stabilize
       gradients for downstream neural architectures (e.g., Transformers,
       GRU-D, BRITS-style imputers).
    3. Programmatically injecting *synthetic* missingness with a known
       ground-truth mask, so that imputation / robustness modules can be
       validated against a controlled corruption process rather than
       naturally-occurring (and therefore unverifiable) gaps.

Design notes
------------
- All array-producing methods operate on ``X in R^{T x K}`` — T timesteps,
  K sensor channels — which is the standard layout expected by sequence
  models (batch dimension added later by the caller / DataLoader).
- Missingness mechanisms:
    * MCAR (Missing Completely At Random): block start positions are drawn
      uniformly at random, independent of the data values or channel.
    * MAR (Missing At Random): block start positions are biased towards
      specific channels/timesteps, simulating a plausible industrial
      failure mode such as a sensor's transmitter dropping packets more
      often when a correlated "load" channel is elevated. Critically, the
      missingness depends only on *observed* covariates, not on the
      missing values themselves (which would be MNAR).
- Missingness is injected in **contiguous blocks**, not independent
  point-wise dropouts, because real industrial packet loss / sensor
  dropout manifests as runs of missing samples (e.g., a PLC losing link
  for several scan cycles), not isolated single-sample gaps.

Author: Industrial Anomaly Prediction Framework
Python: 3.13+
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("IndustrialDataLoader")


# --------------------------------------------------------------------------- #
# Supported dataset registry
# --------------------------------------------------------------------------- #

_SUPPORTED_DATASETS: dict[str, Tuple[str, ...]] = {
    "metropt": (
        "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
        "Motor_current", "COMP", "DV_eletric", "Towers", "MPG", "LPS",
        "Pressure_switch", "Oil_level", "Caudal_impulses",
    ),
    "ai4i2020": (
        "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
        "Torque_Nm", "Tool_wear_min",
    ),
    "tep": tuple(f"XMEAS_{i}" for i in range(1, 23)),
}

_MECHANISMS: Tuple[str, ...] = ("MCAR", "MAR")


@dataclass(frozen=True)
class MissingnessReport:
    """Lightweight summary of an injection run, useful for logging/tests."""

    mechanism: str
    requested_rate: float
    achieved_rate: float
    n_blocks: int
    block_size: int
    shape: Tuple[int, int]


class IndustrialDataLoader:
    """
    Ingestion and corruption utility for multivariate industrial time series.

    This class is intentionally stateless across calls (no persisted internal
    buffers) so that it can be safely reused across multiple datasets /
    experiment folds within the same process. The one piece of state that
    *is* useful to retain between calls is the fitted ``StandardScaler``,
    which callers should hold onto (returned by :meth:`normalize_data`) in
    order to invert the transform at inference/reporting time.

    Parameters
    ----------
    random_state : Optional[int]
        Seed controlling the reproducibility of missingness injection.
        If ``None``, injection is non-deterministic across runs.
    """

    def __init__(self, random_state: Optional[int] = 42) -> None:
        self.random_state = random_state
        self._rng: np.random.Generator = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ #
    # 1. Ingestion
    # ------------------------------------------------------------------ #
    def load_iiot_data(self, dataset_name: str, file_path: str) -> pd.DataFrame:
        """
        Load a multivariate IIoT time series from disk, or synthesize a
        structurally-faithful mock dataset when the real file is unavailable
        (useful for unit tests / CI pipelines without data-access secrets).

        Parameters
        ----------
        dataset_name : str
            One of ``{"metropt", "ai4i2020", "tep"}`` (case-insensitive).
            Determines the expected sensor-channel schema used both to
            validate a real file and to synthesize a mock one.
        file_path : str
            Path to a CSV file containing the raw sensor log. If the path
            does not exist, a warning is logged and a mock DataFrame with
            the correct schema is generated instead, so downstream code can
            still be exercised end-to-end.

        Returns
        -------
        pd.DataFrame
            A DataFrame of shape ``(T, K)`` (plus an implicit index acting
            as the timestamp axis), where T is the number of timesteps and
            K is the number of sensor channels for the requested dataset.

        Raises
        ------
        ValueError
            If ``dataset_name`` is not one of the supported datasets.
        """
        key = dataset_name.strip().lower()
        if key not in _SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset_name '{dataset_name}'. "
                f"Expected one of {list(_SUPPORTED_DATASETS.keys())}."
            )

        expected_cols = list(_SUPPORTED_DATASETS[key])

        try:
            df = pd.read_csv(file_path)
            logger.info("Loaded real data from '%s' (shape=%s).", file_path, df.shape)
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                "Could not read '%s' (%s). Falling back to mock %s data for "
                "pipeline validation.",
                file_path, exc, dataset_name,
            )
            df = self._generate_mock_dataframe(key, expected_cols)

        missing_expected = [c for c in expected_cols if c not in df.columns]
        if missing_expected:
            logger.warning(
                "Loaded '%s' data is missing expected columns %s; "
                "downstream steps should select columns defensively.",
                dataset_name, missing_expected,
            )

        return df

    def _generate_mock_dataframe(
        self, key: str, expected_cols: list[str], n_timesteps: int = 2000
    ) -> pd.DataFrame:
        """Synthesize a plausible mock time series matching a schema."""
        t = np.arange(n_timesteps)
        data: dict[str, np.ndarray] = {}

        for i, col in enumerate(expected_cols):
            # Superpose a slow trend, a periodic operating cycle, and
            # sensor-appropriate Gaussian noise so correlation structure
            # roughly resembles real plant telemetry.
            trend = 0.001 * i * t
            cycle = (5 + i) * np.sin(2 * np.pi * t / (200 + 10 * i))
            noise = self._rng.normal(loc=0.0, scale=1.0, size=n_timesteps)
            data[col] = trend + cycle + noise

        df = pd.DataFrame(data)
        df.insert(0, "timestamp", pd.date_range("2024-01-01", periods=n_timesteps, freq="min"))
        logger.info("Generated mock '%s' DataFrame with shape %s.", key, df.shape)
        return df

    # ------------------------------------------------------------------ #
    # 2. Normalization
    # ------------------------------------------------------------------ #
    def normalize_data(
        self, df: pd.DataFrame, target_cols: list[str]
    ) -> Tuple[np.ndarray, StandardScaler]:
        """
        Apply zero-mean, unit-variance scaling to the specified sensor
        columns.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame containing at least ``target_cols``.
        target_cols : list[str]
            Names of the numeric sensor columns to scale. Order is
            preserved in the output array (column j of X corresponds to
            ``target_cols[j]``).

        Returns
        -------
        Tuple[np.ndarray, StandardScaler]
            - ``X``: array of shape ``(T, K)``, ``K = len(target_cols)``,
              scaled to zero mean / unit variance per channel.
            - ``scaler``: the fitted ``StandardScaler`` instance, retained
              by the caller to invert the transform later
              (``scaler.inverse_transform``).

        Raises
        ------
        ValueError
            If ``target_cols`` is empty, or any requested column is absent
            from ``df``, or the resulting slice contains non-numeric data.
        """
        if not target_cols:
            raise ValueError("target_cols must be a non-empty list of column names.")

        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame.")

        subset = df.loc[:, target_cols]
        if not all(pd.api.types.is_numeric_dtype(subset[c]) for c in target_cols):
            raise ValueError(
                "All target_cols must be numeric; found non-numeric dtype(s) "
                f"in {[c for c in target_cols if not pd.api.types.is_numeric_dtype(subset[c])]}."
            )

        scaler = StandardScaler()
        X = scaler.fit_transform(subset.to_numpy(dtype=np.float64))

        logger.info(
            "Normalized %d channels over %d timesteps (mean~0, var~1).",
            X.shape[1], X.shape[0],
        )
        return X, scaler

    # ------------------------------------------------------------------ #
    # 3. Synthetic missingness injection
    # ------------------------------------------------------------------ #
    def inject_missingness(
        self,
        X: np.ndarray,
        mechanism: str = "MAR",
        missing_rate: float = 0.2,
        block_size: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject contiguous blocks of synthetic missingness into a clean
        multivariate time series, establishing an exact ground-truth mask
        for downstream imputation/robustness validation.

        Parameters
        ----------
        X : np.ndarray
            Clean input array of shape ``(T, K)`` with no NaNs (T timesteps,
            K channels). Typically the output of :meth:`normalize_data`.
        mechanism : str, default "MAR"
            Missingness mechanism, one of ``{"MCAR", "MAR"}``:
              - "MCAR": block start positions are chosen uniformly at
                random across all (timestep, channel) pairs, independent
                of the data.
              - "MAR": block start positions are biased using an
                *observed* auxiliary signal — here, channels with higher
                average magnitude (a proxy for "under higher load") are
                assigned proportionally more dropout blocks, mimicking
                sensors that fail more often under stress. This bias uses
                only observed values, satisfying the MAR assumption
                (missingness depends on observed data, not on the values
                that go missing).
        missing_rate : float, default 0.2
            Target fraction of entries in X to mark as missing, in
            ``(0, 1)``. The achieved rate will approximate this value but
            may differ slightly due to block quantization and boundary
            clipping (reported rate is logged).
        block_size : int, default 5
            Length (in timesteps) of each contiguous missing block along
            the time axis, applied independently per selected channel.
            Must be a positive integer <= T.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - ``X_corrupted``: array of shape ``(T, K)``, identical to X
              except with ``np.nan`` placed at injected missing positions.
            - ``mask``: binary array of shape ``(T, K)``, dtype
              ``np.int8``, where ``1`` = observed and ``0`` = injected
              missing. This is the ground-truth ``M`` used to evaluate
              imputation quality (e.g., masked MSE/MAE).

        Raises
        ------
        ValueError
            If ``X`` is not 2-D, ``mechanism`` is unsupported, or
            ``missing_rate`` / ``block_size`` are out of valid range.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (T, K); got shape {X.shape}.")
        if mechanism not in _MECHANISMS:
            raise ValueError(f"mechanism must be one of {_MECHANISMS}; got '{mechanism}'.")
        if not (0.0 < missing_rate < 1.0):
            raise ValueError(f"missing_rate must be in (0, 1); got {missing_rate}.")

        T, K = X.shape
        if block_size <= 0 or block_size > T:
            raise ValueError(f"block_size must be in [1, T={T}]; got {block_size}.")
        if np.isnan(X).any():
            raise ValueError(
                "Input X already contains NaNs; inject_missingness expects a "
                "clean array so the ground-truth mask is unambiguous."
            )

        mask = np.ones((T, K), dtype=np.int8)
        total_entries = T * K
        target_missing_entries = int(round(missing_rate * total_entries))
        n_blocks = max(1, target_missing_entries // block_size)

        # Determine per-channel sampling weights for block placement.
        if mechanism == "MCAR":
            channel_weights = np.full(K, fill_value=1.0 / K)
        else:  # MAR: bias towards channels with higher observed magnitude
            channel_energy = np.mean(np.abs(X), axis=0)
            # Add a small epsilon floor so zero-energy channels can still
            # occasionally be selected (avoids fully deterministic bias).
            channel_weights = channel_energy + 1e-6
            channel_weights = channel_weights / channel_weights.sum()

        valid_starts = T - block_size + 1
        channels = self._rng.choice(K, size=n_blocks, p=channel_weights)
        starts = self._rng.integers(low=0, high=valid_starts, size=n_blocks)

        for ch, start in zip(channels, starts):
            mask[start:start + block_size, ch] = 0

        X_corrupted = X.copy()
        X_corrupted[mask == 0] = np.nan

        achieved_rate = float(1.0 - mask.mean())
        report = MissingnessReport(
            mechanism=mechanism,
            requested_rate=missing_rate,
            achieved_rate=achieved_rate,
            n_blocks=n_blocks,
            block_size=block_size,
            shape=(T, K),
        )
        logger.info(
            "Injected %s missingness: requested=%.3f achieved=%.3f "
            "(%d blocks x %d, shape=%s).",
            report.mechanism, report.requested_rate, report.achieved_rate,
            report.n_blocks, report.block_size, report.shape,
        )

        return X_corrupted, mask

    # ------------------------------------------------------------------ #
    # Convenience: tensor conversion for downstream PyTorch models
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_torch(
        X_corrupted: np.ndarray, mask: np.ndarray, fill_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a corrupted array + mask pair into PyTorch tensors suitable
        for a sequence model, replacing NaNs with ``fill_value`` (models
        should rely on the mask, not the fill value, to identify gaps).

        Parameters
        ----------
        X_corrupted : np.ndarray
            Array of shape ``(T, K)`` containing NaNs at missing positions.
        mask : np.ndarray
            Binary array of shape ``(T, K)`` (1 = observed, 0 = missing).
        fill_value : float, default 0.0
            Value used to replace NaNs so the tensor is safe for arithmetic
            (e.g., matrix multiplication) without propagating NaNs.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(x_tensor, mask_tensor)``, both of shape ``(T, K)`` and dtype
            ``torch.float32``.
        """
        x_filled = np.nan_to_num(X_corrupted, nan=fill_value)
        x_tensor = torch.as_tensor(x_filled, dtype=torch.float32)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
        return x_tensor, mask_tensor


# --------------------------------------------------------------------------- #
# Real-data windowed Dataset for PyTorch DataLoader
# --------------------------------------------------------------------------- #
class RealDataset(torch.utils.data.Dataset):
    """
    A PyTorch ``Dataset`` that loads a real IIoT CSV, normalizes sensor
    channels, and serves overlapping sliding windows of shape
    ``(window_size, n_features)`` for sequence-model training.

    This class is the bridge between the raw CSV on disk and the batched
    ``(B, T, K)`` tensors consumed by ``CaliPredTransformer``. It handles:

    1. Loading the CSV via :class:`IndustrialDataLoader`.
    2. Selecting and normalizing the sensor columns.
    3. Extracting overlapping windows with configurable stride.
    4. Providing ``(input_window, target_window, timestamp_window)`` tuples
       where the target is shifted by ``forecast_horizon`` steps (default 1)
       for 1-step-ahead forecasting.

    Parameters
    ----------
    dataset_name : str
        One of ``{"metropt", "ai4i2020", "tep"}`` — selects the column
        schema from ``_SUPPORTED_DATASETS``.
    file_path : str
        Path to the raw CSV file.
    window_size : int, default 60
        Number of timesteps per sliding window (sequence length ``T``).
    stride : int, default 10
        Step size between consecutive window start positions. Stride=1
        maximizes overlap; larger strides reduce dataset size for faster
        iteration.
    forecast_horizon : int, default 1
        Number of steps the target is shifted ahead of the input. 0 means
        self-reconstruction (target = input); 1 means 1-step-ahead
        forecasting.
    scaler : Optional[StandardScaler], default None
        If provided, this pre-fitted scaler is used instead of fitting a
        new one (used for val/test sets to prevent data leakage — the
        scaler must be fitted on the training set only).
    random_state : int, default 42
        Seed for the underlying ``IndustrialDataLoader``.
    """

    def __init__(
        self,
        dataset_name: str,
        file_path: str,
        window_size: int = 60,
        stride: int = 10,
        forecast_horizon: int = 1,
        scaler: Optional[StandardScaler] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__()

        key = dataset_name.strip().lower()
        if key not in _SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset_name '{dataset_name}'. "
                f"Expected one of {list(_SUPPORTED_DATASETS.keys())}."
            )

        self.dataset_name = key
        self.window_size = window_size
        self.stride = stride
        self.forecast_horizon = forecast_horizon

        # --- Load raw data -------------------------------------------------- #
        loader = IndustrialDataLoader(random_state=random_state)
        df = loader.load_iiot_data(dataset_name, file_path)

        # Resolve sensor columns: use the expected schema, keeping only
        # columns actually present in the loaded DataFrame.
        expected_cols = list(_SUPPORTED_DATASETS[key])
        # Map AI4I 2020 column name variants (UCI versions differ)
        _COLUMN_ALIASES = {
            "Air_temperature_K": ["Air temperature [K]", "Air_temperature_K"],
            "Process_temperature_K": ["Process temperature [K]", "Process_temperature_K"],
            "Rotational_speed_rpm": ["Rotational speed [rpm]", "Rotational_speed_rpm"],
            "Torque_Nm": ["Torque [Nm]", "Torque_Nm"],
            "Tool_wear_min": ["Tool wear [min]", "Tool_wear_min"],
        }
        resolved_cols = []
        col_renames = {}
        for ecol in expected_cols:
            if ecol in df.columns:
                resolved_cols.append(ecol)
            elif ecol in _COLUMN_ALIASES:
                for alias in _COLUMN_ALIASES[ecol]:
                    if alias in df.columns:
                        col_renames[alias] = ecol
                        resolved_cols.append(ecol)
                        break
            else:
                # Try substring matching as fallback
                for dc in df.columns:
                    if ecol.lower().replace("_", " ") in dc.lower():
                        col_renames[dc] = ecol
                        resolved_cols.append(ecol)
                        break

        if col_renames:
            df = df.rename(columns=col_renames)

        if not resolved_cols:
            raise ValueError(
                f"No matching sensor columns found in '{file_path}' for "
                f"dataset '{dataset_name}'. Expected: {expected_cols}. "
                f"Found: {list(df.columns)}."
            )

        self.sensor_cols = resolved_cols
        self.n_features = len(resolved_cols)

        # --- Extract timestamps if available -------------------------------- #
        timestamp_col = None
        for candidate in ("timestamp", "Timestamp", "datetime", "date", "time"):
            if candidate in df.columns:
                timestamp_col = candidate
                break

        if timestamp_col is not None:
            try:
                ts_series = pd.to_datetime(df[timestamp_col])
                # Convert to Unix seconds (float64)
                self.timestamps = (
                    ts_series.astype(np.int64).values / 1e9
                ).astype(np.float64)
            except Exception:
                self.timestamps = np.arange(len(df), dtype=np.float64)
        else:
            self.timestamps = np.arange(len(df), dtype=np.float64)

        # --- Normalize ------------------------------------------------------ #
        sensor_df = df[resolved_cols].copy()
        # Drop rows with NaNs in sensor columns (real data may have some)
        sensor_df = sensor_df.dropna().reset_index(drop=True)

        if scaler is not None:
            self.scaler = scaler
            self.X = scaler.transform(
                sensor_df.to_numpy(dtype=np.float64)
            ).astype(np.float32)
        else:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(
                sensor_df.to_numpy(dtype=np.float64)
            ).astype(np.float32)

        # Trim timestamps to match cleaned data
        self.timestamps = self.timestamps[: len(self.X)]

        # --- Create windows ------------------------------------------------- #
        self._create_windows()

        logger.info(
            "RealDataset '%s': %d windows of size %d (stride=%d, "
            "horizon=%d) from %d total timesteps across %d channels.",
            self.dataset_name, len(self.windows), self.window_size,
            self.stride, self.forecast_horizon,
            len(self.X), self.n_features,
        )

    def _create_windows(self) -> None:
        """Slice the normalized time series into overlapping windows."""
        total_len = len(self.X)
        max_end = total_len - self.forecast_horizon  # need room for target

        self.windows = []
        start = 0
        while start + self.window_size <= max_end:
            self.windows.append(start)
            start += self.stride

        if not self.windows:
            raise ValueError(
                f"Cannot create any windows: total_len={total_len}, "
                f"window_size={self.window_size}, "
                f"forecast_horizon={self.forecast_horizon}."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - ``x``: input window, shape ``(window_size, n_features)``
            - ``target``: target window, shape ``(window_size, n_features)``,
              shifted by ``forecast_horizon``
            - ``ts``: timestamp window, shape ``(window_size,)``
        """
        start = self.windows[idx]
        end = start + self.window_size

        x = torch.as_tensor(self.X[start:end], dtype=torch.float32)

        if self.forecast_horizon == 0:
            target = x.clone()
        else:
            t_start = start + self.forecast_horizon
            t_end = end + self.forecast_horizon
            target = torch.as_tensor(
                self.X[t_start:t_end], dtype=torch.float32
            )

        ts = torch.as_tensor(
            self.timestamps[start:end], dtype=torch.float64
        )

        return x, target, ts


def create_dataloaders(
    dataset_name: str,
    file_path: str,
    window_size: int = 60,
    stride: int = 10,
    forecast_horizon: int = 1,
    batch_size: int = 32,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
    num_workers: int = 0,
) -> Tuple["RealDataset", "RealDataset", "RealDataset",
           torch.utils.data.DataLoader, torch.utils.data.DataLoader,
           torch.utils.data.DataLoader]:
    """
    Create time-based train/val/test splits and return DataLoaders.

    **Time-based split, not random shuffle**: for time-series data, random
    shuffling creates data leakage (future data informing past predictions).
    Instead, the first ``train_frac`` of the timeline goes to training, the
    next ``val_frac`` to validation, and the remainder to testing.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier (e.g., ``"metropt"``, ``"ai4i2020"``).
    file_path : str
        Path to the raw CSV.
    window_size : int, default 60
        Sequence length per window.
    stride : int, default 10
        Step between consecutive windows.
    forecast_horizon : int, default 1
        Target shift (0 = self-reconstruction, 1 = 1-step-ahead).
    batch_size : int, default 32
        Batch size for all DataLoaders.
    train_frac : float, default 0.70
        Fraction of total timesteps for training.
    val_frac : float, default 0.15
        Fraction for validation. Test gets the remainder.
    random_state : int, default 42
        Seed for reproducibility.
    num_workers : int, default 0
        Number of DataLoader worker processes (0 = main process only).

    Returns
    -------
    Tuple[RealDataset, RealDataset, RealDataset,
          DataLoader, DataLoader, DataLoader]
        ``(train_ds, val_ds, test_ds, train_loader, val_loader, test_loader)``
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1); got {train_frac}.")
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}.")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1.0; "
            f"got {train_frac} + {val_frac} = {train_frac + val_frac}."
        )

    # Step 1: Load the full dataset to get total length and fit the scaler
    # on the training portion only.
    key = dataset_name.strip().lower()
    loader = IndustrialDataLoader(random_state=random_state)
    df = loader.load_iiot_data(dataset_name, file_path)

    # Resolve columns (same logic as RealDataset)
    expected_cols = list(_SUPPORTED_DATASETS[key])
    _COLUMN_ALIASES = {
        "Air_temperature_K": ["Air temperature [K]", "Air_temperature_K"],
        "Process_temperature_K": ["Process temperature [K]", "Process_temperature_K"],
        "Rotational_speed_rpm": ["Rotational speed [rpm]", "Rotational_speed_rpm"],
        "Torque_Nm": ["Torque [Nm]", "Torque_Nm"],
        "Tool_wear_min": ["Tool wear [min]", "Tool_wear_min"],
    }
    col_renames = {}
    resolved_cols = []
    for ecol in expected_cols:
        if ecol in df.columns:
            resolved_cols.append(ecol)
        elif ecol in _COLUMN_ALIASES:
            for alias in _COLUMN_ALIASES[ecol]:
                if alias in df.columns:
                    col_renames[alias] = ecol
                    resolved_cols.append(ecol)
                    break
        else:
            for dc in df.columns:
                if ecol.lower().replace("_", " ") in dc.lower():
                    col_renames[dc] = ecol
                    resolved_cols.append(ecol)
                    break

    if col_renames:
        df = df.rename(columns=col_renames)

    sensor_df = df[resolved_cols].dropna().reset_index(drop=True)
    total_len = len(sensor_df)

    train_end = int(total_len * train_frac)
    val_end = int(total_len * (train_frac + val_frac))

    # Fit scaler on training portion ONLY (prevent data leakage)
    train_scaler = StandardScaler()
    train_scaler.fit(sensor_df.iloc[:train_end].to_numpy(dtype=np.float64))

    # Step 2: Write temporary split CSVs (or use index slicing)
    # More efficient: create datasets that operate on array slices.
    # We'll create a helper that builds RealDataset from pre-split arrays.

    class _SlicedDataset(torch.utils.data.Dataset):
        """Lightweight dataset operating on a pre-sliced, pre-scaled array."""

        def __init__(
            self, X: np.ndarray, timestamps: np.ndarray,
            window_size: int, stride: int, forecast_horizon: int,
            n_features: int, scaler: StandardScaler,
        ):
            self.X = X.astype(np.float32)
            self.timestamps = timestamps
            self.window_size = window_size
            self.stride = stride
            self.forecast_horizon = forecast_horizon
            self.n_features = n_features
            self.scaler = scaler
            self.sensor_cols = resolved_cols

            max_end = len(X) - forecast_horizon
            self.windows = []
            s = 0
            while s + window_size <= max_end:
                self.windows.append(s)
                s += stride

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, idx):
            start = self.windows[idx]
            end = start + self.window_size
            x = torch.as_tensor(self.X[start:end], dtype=torch.float32)
            if self.forecast_horizon == 0:
                target = x.clone()
            else:
                t_start = start + self.forecast_horizon
                t_end = end + self.forecast_horizon
                target = torch.as_tensor(
                    self.X[t_start:t_end], dtype=torch.float32
                )
            ts = torch.as_tensor(
                self.timestamps[start:end], dtype=torch.float64
            )
            return x, target, ts

    # Scale each split using the train-fitted scaler
    full_array = sensor_df.to_numpy(dtype=np.float64)

    # Timestamps
    timestamp_col = None
    for candidate in ("timestamp", "Timestamp", "datetime", "date", "time"):
        if candidate in df.columns:
            timestamp_col = candidate
            break
    if timestamp_col is not None:
        try:
            ts_all = (
                pd.to_datetime(df[timestamp_col])
                .astype(np.int64).values / 1e9
            ).astype(np.float64)
        except Exception:
            ts_all = np.arange(total_len, dtype=np.float64)
    else:
        ts_all = np.arange(total_len, dtype=np.float64)
    ts_all = ts_all[:total_len]

    X_train = train_scaler.transform(full_array[:train_end])
    X_val = train_scaler.transform(full_array[train_end:val_end])
    X_test = train_scaler.transform(full_array[val_end:])

    ts_train = ts_all[:train_end]
    ts_val = ts_all[train_end:val_end]
    ts_test = ts_all[val_end:]

    n_features = len(resolved_cols)
    train_ds = _SlicedDataset(
        X_train, ts_train, window_size, stride,
        forecast_horizon, n_features, train_scaler,
    )
    val_ds = _SlicedDataset(
        X_val, ts_val, window_size, stride,
        forecast_horizon, n_features, train_scaler,
    )
    test_ds = _SlicedDataset(
        X_test, ts_test, window_size, stride,
        forecast_horizon, n_features, train_scaler,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False,
    )

    logger.info(
        "Created time-based splits: train=%d windows, val=%d windows, "
        "test=%d windows (total timesteps=%d, split at [%d, %d]).",
        len(train_ds), len(val_ds), len(test_ds),
        total_len, train_end, val_end,
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# Real corruption injector for fault-validation experiments
# --------------------------------------------------------------------------- #
class RealCorruptionInjector:
    """
    Injects realistic industrial sensor faults into real data windows for
    validating that the DTI → sigma inflation chain works correctly.

    Each method corrupts a clean ``(T, K)`` window and returns both the
    corrupted window and a binary mask (1 = uncorrupted, 0 = corrupted),
    so downstream DQA/IRI/DTI impact can be measured against the known
    corruption ground truth.

    Parameters
    ----------
    random_state : Optional[int], default 42
        Seed for reproducible corruption placement.
    """

    def __init__(self, random_state: Optional[int] = 42) -> None:
        self._rng = np.random.default_rng(random_state)

    def sensor_dropout(
        self,
        window: np.ndarray,
        channels: Optional[list[int]] = None,
        duration: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a sensor going completely dead (producing NaN) for a
        contiguous block of timesteps on specified channels.

        Parameters
        ----------
        window : np.ndarray, shape (T, K)
            Clean sensor window.
        channels : Optional[list[int]], default None
            Which channel indices to drop out. If ``None``, randomly
            selects 2 channels.
        duration : int, default 20
            Length of the contiguous dropout block (in timesteps).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(corrupted_window, mask)`` — corrupted window has ``np.nan``
            at dropout positions; mask is 1 = OK, 0 = dropped.
        """
        T, K = window.shape
        corrupted = window.copy()
        mask = np.ones_like(window, dtype=np.int8)

        if channels is None:
            channels = self._rng.choice(K, size=min(2, K), replace=False).tolist()
        duration = min(duration, T)
        start = self._rng.integers(0, max(1, T - duration + 1))

        for ch in channels:
            corrupted[start:start + duration, ch] = np.nan
            mask[start:start + duration, ch] = 0

        return corrupted, mask

    def gaussian_noise(
        self,
        window: np.ndarray,
        snr_db: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add calibrated Gaussian noise to all channels, simulating
        electromagnetic interference or sensor degradation.

        Parameters
        ----------
        window : np.ndarray, shape (T, K)
            Clean sensor window.
        snr_db : float, default 5.0
            Signal-to-noise ratio in decibels. Lower = more noise.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(noisy_window, mask)`` — mask is all-ones (no missing data,
            but the values are corrupted — DQA consistency should detect
            the distribution shift).
        """
        T, K = window.shape
        signal_power = np.mean(window ** 2, axis=0, keepdims=True)
        signal_power = np.maximum(signal_power, 1e-10)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = self._rng.normal(0.0, np.sqrt(noise_power), size=(T, K))

        noisy = window + noise
        mask = np.ones((T, K), dtype=np.int8)
        return noisy, mask

    def stuck_at_value(
        self,
        window: np.ndarray,
        channels: Optional[list[int]] = None,
        duration: int = 30,
        value: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a stuck/frozen sensor: a channel reports the same constant
        value (e.g., its mean or last-known reading) for a contiguous block.

        Parameters
        ----------
        window : np.ndarray, shape (T, K)
            Clean sensor window.
        channels : Optional[list[int]], default None
            Which channel indices to freeze. If ``None``, randomly selects 1.
        duration : int, default 30
            Length of the stuck-at block.
        value : Optional[float], default None
            The constant value to freeze at. If ``None``, uses each
            channel's mean (simulating a common PLC fault mode).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(stuck_window, mask)`` — mask is 1 = OK, 0 = stuck position.
            The stuck values are *not* NaN (the sensor is "reporting")
            but carry no information.
        """
        T, K = window.shape
        stuck = window.copy()
        mask = np.ones((T, K), dtype=np.int8)

        if channels is None:
            channels = [int(self._rng.integers(0, K))]
        duration = min(duration, T)
        start = self._rng.integers(0, max(1, T - duration + 1))

        for ch in channels:
            freeze_val = value if value is not None else float(np.mean(window[:, ch]))
            stuck[start:start + duration, ch] = freeze_val
            mask[start:start + duration, ch] = 0

        return stuck, mask


# ------------------------------------------------------------------------- #
# Demonstration / smoke test
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("Running IndustrialDataLoader demonstration with dummy data.")

    loader = IndustrialDataLoader(random_state=7)

    # --- Build dummy clean data directly (bypassing file I/O) ------------- #
    T_DEMO, K_DEMO = 500, 8
    rng = np.random.default_rng(7)
    dummy_df = pd.DataFrame(
        rng.normal(loc=50.0, scale=10.0, size=(T_DEMO, K_DEMO)),
        columns=[f"sensor_{i}" for i in range(K_DEMO)],
    )

    # --- Normalize --------------------------------------------------------- #
    X_scaled, fitted_scaler = loader.normalize_data(
        dummy_df, target_cols=list(dummy_df.columns)
    )
    assert X_scaled.shape == (T_DEMO, K_DEMO), "Unexpected shape after scaling."
    print(f"[OK] Scaled data shape: {X_scaled.shape} "
          f"(mean~{X_scaled.mean():.4f}, std~{X_scaled.std():.4f})")

    # --- Inject MCAR missingness ------------------------------------------- #
    X_mcar, mask_mcar = loader.inject_missingness(
        X_scaled, mechanism="MCAR", missing_rate=0.2, block_size=5
    )
    assert X_mcar.shape == X_scaled.shape == mask_mcar.shape
    mcar_rate = 1.0 - mask_mcar.mean()
    print(f"[OK] MCAR corruption -> achieved missing rate: {mcar_rate:.3f}")
    assert np.isnan(X_mcar[mask_mcar == 0]).all(), "NaNs must align with mask==0."
    assert not np.isnan(X_mcar[mask_mcar == 1]).any(), "Observed entries must not be NaN."

    # --- Inject MAR missingness --------------------------------------------- #
    X_mar, mask_mar = loader.inject_missingness(
        X_scaled, mechanism="MAR", missing_rate=0.3, block_size=10
    )
    mar_rate = 1.0 - mask_mar.mean()
    print(f"[OK] MAR corruption -> achieved missing rate: {mar_rate:.3f}")

    # --- Convert to PyTorch tensors ----------------------------------------- #
    x_tensor, mask_tensor = IndustrialDataLoader.to_torch(X_mar, mask_mar)
    print(f"[OK] Torch tensors -> x: {tuple(x_tensor.shape)} "
          f"({x_tensor.dtype}), mask: {tuple(mask_tensor.shape)} ({mask_tensor.dtype})")

    print("\nAll smoke tests passed.")
