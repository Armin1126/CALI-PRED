"""
pipeline.py

Industrial Anomaly Prediction Framework — End-to-End Real-Data Pipeline
==============================================================================

Wires all CALI-PRED modules together on real benchmark data:

    1. Load real CSV → window into (B, 60, K) sequences → time-based split
    2. Compute real DQA/IRI/DTI per window (data-driven, not hardcoded)
    3. Train CaliPredTransformer with real batches + real DTI + NLL loss
    4. Evaluate calibration on held-out test data
    5. Compare against a DTI-blind baseline

Usage
-----
    python pipeline.py --dataset metropt --data-path data/metropt/MetroPT3(chiller).csv
    python pipeline.py --dataset metropt --data-path data/metropt/MetroPT3(chiller).csv --max-windows 50 --epochs 5

Python: 3.13+
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("CaliPredPipeline")


def _atomic_torch_save(payload: dict, path: str) -> None:
    temporary_path = f"{path}.tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)

# Mute verbose logs from intermediate engines to avoid Colab console truncation and speed up runs
logging.getLogger("ImputationReliabilityEngine").setLevel(logging.WARNING)
logging.getLogger("TrustFusionEngine").setLevel(logging.WARNING)
logging.getLogger("UpstreamDQAEngine").setLevel(logging.WARNING)
logging.getLogger("IndustrialDataLoader").setLevel(logging.WARNING)

# Local imports
from data_loader import (
    IndustrialDataLoader,
    create_dataloaders,
)
from dqa_module import UpstreamDQAEngine
from fusion_engine import TrustFusionEngine
from iri_module import ImputationReliabilityEngine
from predictor import CaliPredTransformer, TrustCalibratedLoss
from metrics_engine import (
    expected_calibration_curve,
    calculate_gaussian_crps,
    gaussian_interval,
    plot_reliability_diagram,
    ModelCalibrationOutputs,
)


# --------------------------------------------------------------------------- #
def fit_validation_sigma_scale(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Fit a positive global sigma multiplier using validation Gaussian NLL.

    With fixed ``mu`` and ``sigma``, the NLL-optimal multiplier is the root
    mean squared standardized residual. Fitting it on validation outputs only
    avoids test leakage and isolates global sigma mis-scaling from DTI effects.
    """
    if not (y_true.shape == mu.shape == sigma.shape):
        raise ValueError("y_true, mu, and sigma must have identical shapes.")
    sigma_safe = np.clip(sigma.astype(np.float64), 1e-8, None)
    scale = float(np.sqrt(np.mean(((y_true - mu) / sigma_safe) ** 2)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid validation sigma scale: {scale!r}")
    return scale


def batch_uncertainty_diagnostics(
    model: CaliPredTransformer, mu: Tensor, sigma: Tensor, target: Tensor, dti: Tensor,
) -> Tuple[float, float, float]:
    """Return mean base sigma, mean DTI inflation, and standardized RMSE."""
    with torch.no_grad():
        inflation = model.uncertainty_inflation(dti)
        sigma_base = sigma / inflation.unsqueeze(-1)
        standardized_rmse = torch.sqrt(
            torch.mean(((target - mu) / sigma.clamp_min(1e-6)) ** 2)
        )
    return (
        float(sigma_base.mean().item()),
        float(inflation.mean().item()),
        float(standardized_rmse.item()),
    )

# DTI computation for a batch of windows
# --------------------------------------------------------------------------- #
def compute_dti_for_batch(
    x_batch: np.ndarray,
    ts_batch: np.ndarray,
    n_features: int,
    dqa_engine: UpstreamDQAEngine,
    iri_engine: ImputationReliabilityEngine,
    fusion_engine: TrustFusionEngine,
    corruption_loader: IndustrialDataLoader,
    baseline_corr: np.ndarray,
    missing_rate: float = 0.15,
    block_size: int = 5,
    missing_rate_sampler: Optional[Callable[[], float]] = None,
    use_dqa: bool = True,
    use_iri: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute real DTI for a batch of clean windows by:
    1. Injecting synthetic missingness (controlled corruption with known GT)
    2. Running DQA on each corrupted window
    3. Running IRI ensemble imputation + IRI scoring
    4. Fusing DQA * IRI → DTI

    Parameters
    ----------
    x_batch : np.ndarray, shape (B, T, K)
        Clean (pre-corruption) sensor windows.
    ts_batch : np.ndarray, shape (B, T)
        Timestamps for each window.
    n_features : int
        Number of sensor channels K.
    dqa_engine : UpstreamDQAEngine
    iri_engine : ImputationReliabilityEngine
    fusion_engine : TrustFusionEngine
    corruption_loader : IndustrialDataLoader
    baseline_corr : np.ndarray, shape (K, K)
    missing_rate : float, default 0.15
    block_size : int, default 5
    missing_rate_sampler : Optional[Callable[[], float]], default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - dti_batch: shape (B, T), per-timestep DTI averaged across channels
        - x_imputed_batch: shape (B, T, K), ensemble-mean imputed windows
    """
    B, T, K = x_batch.shape
    dti_batch = np.empty((B, T), dtype=np.float64)
    x_imputed_batch = np.empty_like(x_batch, dtype=np.float32)

    for i in range(B):
        window = x_batch[i]  # (T, K)
        ts_window = ts_batch[i]  # (T,)

        # Step 1: Inject missingness
        window_missing_rate = missing_rate
        if missing_rate_sampler is not None:
            window_missing_rate = missing_rate_sampler()

        if window_missing_rate == 0.0:
            x_corrupted = window.copy()
            mask = np.ones_like(window, dtype=np.int8)
        else:
            try:
                x_corrupted, mask = corruption_loader.inject_missingness(
                    window, mechanism="MAR", missing_rate=window_missing_rate,
                    block_size=block_size,
                )
            except ValueError:
                # Edge case: window might already have issues
                x_corrupted = window.copy()
                mask = np.ones_like(window, dtype=np.int8)

        # Step 2: DQA
        inference_time = float(ts_window[-1]) + 0.5
        if not use_dqa:
            dqa_score = np.ones(T, dtype=np.float32)
        else:
            try:
                dqa_score = dqa_engine.compute_dqa_score(
                    mask=mask,
                    timestamps=ts_window,
                    inference_time=inference_time,
                    X_corrupted=x_corrupted,
                    baseline_corr_matrix=baseline_corr,
                )
            except Exception:
                dqa_score = 0.5  # fallback for edge cases

        # Step 3: IRI (ensemble imputation + reliability scoring)
        if not use_iri:
            x_imputed = x_corrupted.copy()
            for channel in range(K):
                observed = mask[:, channel].astype(bool)
                fill_value = float(np.mean(x_corrupted[observed, channel])) if np.any(observed) else 0.0
                x_imputed[~observed, channel] = fill_value
            iri_grid = np.ones((T, K), dtype=np.float32)
        else:
            try:
                ensemble_out = iri_engine.impute_ensemble(x_corrupted, mask)  # (2, T, K)
                eval_mask = (1 - mask).astype(np.int8)
                iri_grid = iri_engine.compute_iri(
                    ensemble_out, window, eval_mask,
                )  # (T, K)
                x_imputed = ensemble_out.mean(axis=0).astype(np.float32)  # (T, K)
            except Exception:
                iri_grid = np.full((T, K), 0.5)
                x_imputed = x_corrupted.copy()
                x_imputed = np.nan_to_num(x_imputed, nan=0.0).astype(np.float32)

        # Step 4: Fuse DQA * IRI → DTI
        dti_grid = fusion_engine.compute_dti(dqa_score, iri_grid)  # (T, K)
        # Average across channels to get per-timestep DTI for the Transformer
        dti_per_timestep = np.mean(dti_grid, axis=1)  # (T,)

        dti_batch[i] = dti_per_timestep
        x_imputed_batch[i] = x_imputed

    return dti_batch, x_imputed_batch


def make_severity_sampler(
    clean_fraction: float = 0.25,
    max_severity: float = 0.45,
    random_state: int = 42,
) -> Callable[[], float]:
    """
    Returns a closure that samples a missing_rate per window:
    - With probability clean_fraction, returns exactly 0.0.
    - Otherwise, returns a draw from U(0.0, max_severity].
    """
    rng = np.random.default_rng(random_state)

    def sampler() -> float:
        if rng.random() < clean_fraction:
            return 0.0
        val = rng.uniform(1e-9, max_severity)
        return float(val)

    return sampler


# --------------------------------------------------------------------------- #
# TrustCachedDataset and Precomputation Helper
# --------------------------------------------------------------------------- #
class TrustCachedDataset(torch.utils.data.Dataset):
    """
    A Dataset wrapper that holds precomputed DTI and imputed sensor sequences.
    Returns: (x_imputed, target, ts, dti)
    """
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        cached_dti: np.ndarray,
        cached_imputed: np.ndarray,
    ) -> None:
        self.base_dataset = base_dataset
        self.cached_dti = torch.as_tensor(cached_dti, dtype=torch.float32)
        self.cached_imputed = torch.as_tensor(cached_imputed, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # base_dataset returns: (x_clean, target, ts)
        _, target, ts = self.base_dataset[idx]
        x_imputed = self.cached_imputed[idx]
        dti = self.cached_dti[idx]
        return x_imputed, target, ts, dti


def precompute_trust_and_imputed(
    dataset: torch.utils.data.Dataset,
    dqa_engine: UpstreamDQAEngine,
    iri_engine: ImputationReliabilityEngine,
    fusion_engine: TrustFusionEngine,
    corruption_loader: IndustrialDataLoader,
    baseline_corr: np.ndarray,
    n_features: int,
    missing_rate: float = 0.15,
    batch_size: int = 128,
    missing_rate_sampler: Optional[Callable[[], float]] = None,
    use_dqa: bool = True,
    use_iri: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs DTI and Imputation precomputation on a full dataset in batches
    to avoid running them repeatedly during epochs.
    """
    # Create a batch-oriented loader (shuffle=False to match index ordering)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    all_dti = []
    all_imputed = []

    logger.info(
        "Precomputing DTI and Imputation for dataset of size %d (batch_size=%d)...",
        len(dataset), batch_size,
    )
    start_time = time.time()

    for x_batch, _, ts_batch in loader:
        dti_batch, x_imputed_batch = compute_dti_for_batch(
            x_batch.numpy(), ts_batch.numpy(), n_features,
            dqa_engine, iri_engine, fusion_engine,
            corruption_loader, baseline_corr,
            missing_rate=missing_rate,
            missing_rate_sampler=missing_rate_sampler,
            use_dqa=use_dqa,
            use_iri=use_iri,
        )
        all_dti.append(dti_batch)
        all_imputed.append(x_imputed_batch)

    elapsed = time.time() - start_time
    logger.info("Finished precomputation in %.2f seconds.", elapsed)

    return np.concatenate(all_dti, axis=0), np.concatenate(all_imputed, axis=0)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_model(
    model: CaliPredTransformer,
    loss_fn: TrustCalibratedLoss,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    dqa_engine: UpstreamDQAEngine,
    iri_engine: ImputationReliabilityEngine,
    fusion_engine: TrustFusionEngine,
    corruption_loader: IndustrialDataLoader,
    baseline_corr: np.ndarray,
    n_features: int,
    epochs: int = 30,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: str = "checkpoints",
    use_real_dti: bool = True,
    missing_rate_sampler: Optional[Callable[[], float]] = None,
    sigma_lr_multiplier: float = 0.5,
    val_warmup_epochs: int = 3,
    sigma_weight_decay: float = 1e-3,
    checkpoint_metadata: Optional[dict] = None,
    use_dqa: bool = True,
    use_iri: bool = True,
) -> dict:
    """
    Real DataLoader-based training loop with:
    - Real DTI computation per batch (or uniform DTI=1.0 for baseline)
    - Cosine annealing LR scheduler
    - Gradient clipping
    - Validation MAE tracking + early stopping (decoupled from raw NLL
      to prevent sigma collapse from gaming the checkpoint criterion)
    - Model checkpointing on validation MAE

    Returns
    -------
    dict
        Training history with keys "train_loss", "val_loss", "best_epoch".
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Decoupled learning rates for sigma head and trust parameters.
    # Default multiplier is 0.5 (slower sigma LR) to prevent the sigma
    # head from overfitting to shrinking training residuals faster than
    # the mu head generalizes.  Weight decay on sigma params further
    # regularizes against extreme weight magnitudes that produce
    # collapsed sigma values.
    if hasattr(model, "sigma_head") and (sigma_lr_multiplier != 1.0 or sigma_weight_decay > 0.0):
        sigma_lr_mult = sigma_lr_multiplier
        sigma_params = list(model.sigma_head.parameters())
        if hasattr(model, "sigma_temperature_raw"):
            sigma_params.append(model.sigma_temperature_raw)
        if hasattr(model, "trust_sensitivity_raw"):
            sigma_params.append(model.trust_sensitivity_raw)

        sigma_param_ids = {id(p) for p in sigma_params}
        other_params = [p for p in model.parameters() if id(p) not in sigma_param_ids]

        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": lr},
            {"params": sigma_params, "lr": lr * sigma_lr_mult,
             "weight_decay": sigma_weight_decay}
        ])
        logger.info(
            "Optimizer: decoupled learning rates. General LR = %.6f, "
            "Sigma LR = %.6f (multiplier = %.2f), Sigma WD = %.6f",
            lr, lr * sigma_lr_mult, sigma_lr_mult, sigma_weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    history = {
        "train_loss": [], "val_loss": [], "best_epoch": 0,
        "train_nll": [], "val_nll": [], "train_pinball": [], "val_pinball": [],
        "train_nll_log_sigma": [], "val_nll_log_sigma": [],
        "train_nll_residual": [], "val_nll_residual": [],
        "train_sigma_base": [], "val_sigma_base": [],
        "train_inflation": [], "val_inflation": [], "train_z_rmse": [], "val_z_rmse": [],
        "train_mae": [], "val_mae": [],
    }
    best_val_mae = float("inf")
    patience = 10
    patience_counter = 0

    model.to(device)
    model_label = "CALI-PRED" if use_real_dti else "Baseline (DTI=1.0)"

    for epoch in range(epochs):
        # --- Training phase ------------------------------------------------ #
        model.train()
        epoch_losses = []
        epoch_nlls, epoch_pinballs, epoch_base_sigmas, epoch_inflations, epoch_z_rmses, epoch_maes = [], [], [], [], [], []
        epoch_nll_log_sigmas, epoch_nll_residuals = [], []

        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 4:
                x_imputed, target, ts, dti_tensor = batch
                if not use_real_dti:
                    dti_tensor = torch.ones_like(dti_tensor)
            else:
                x, target, ts = batch
                B, T, K = x.shape
                if use_real_dti:
                    x_np = x.numpy()
                    ts_np = ts.numpy()
                    dti_np, x_imputed_np = compute_dti_for_batch(
                        x_np, ts_np, n_features,
                        dqa_engine, iri_engine, fusion_engine,
                        corruption_loader, baseline_corr,
                        missing_rate_sampler=missing_rate_sampler,
                        use_dqa=use_dqa, use_iri=use_iri,
                    )
                    dti_tensor = torch.as_tensor(dti_np, dtype=torch.float32)
                    x_imputed = torch.as_tensor(x_imputed_np, dtype=torch.float32)
                else:
                    # To be fair, baseline gets corrupted/imputed sequence as well
                    x_np = x.numpy()
                    ts_np = ts.numpy()
                    _, x_imputed_np = compute_dti_for_batch(
                        x_np, ts_np, n_features,
                        dqa_engine, iri_engine, fusion_engine,
                        corruption_loader, baseline_corr,
                        missing_rate_sampler=missing_rate_sampler,
                        use_dqa=use_dqa, use_iri=use_iri,
                    )
                    dti_tensor = torch.ones(B, T, dtype=torch.float32)
                    x_imputed = torch.as_tensor(x_imputed_np, dtype=torch.float32)

            x_imputed = x_imputed.to(device)
            target = target.to(device)
            dti_tensor = dti_tensor.to(device)

            optimizer.zero_grad()
            mu, sigma, _ = model(x_imputed, dti_tensor)
            mu, sigma, target = mu[:, -1:, :], sigma[:, -1:, :], target[:, -1:, :]
            dti_tensor = dti_tensor[:, -1:]
            loss, diagnostics = loss_fn(mu, sigma, target)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}, batch {batch_idx}.")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(
                        f"Non-finite gradient for {name} at epoch {epoch + 1}, batch {batch_idx}."
                    )
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_maes.append(torch.mean(torch.abs(target - mu)).item())
            epoch_nlls.append(diagnostics["nll"].item())
            epoch_pinballs.append(diagnostics["calibration"].item())
            epoch_nll_log_sigmas.append(diagnostics["nll_log_sigma"].item())
            epoch_nll_residuals.append(diagnostics["nll_residual"].item())
            base_sigma, inflation, z_rmse = batch_uncertainty_diagnostics(
                model, mu, sigma, target, dti_tensor,
            )
            epoch_base_sigmas.append(base_sigma); epoch_inflations.append(inflation); epoch_z_rmses.append(z_rmse)
        if len(epoch_losses) > 0:
            scheduler.step()
            avg_train_loss = float(np.mean(epoch_losses))
            avg_train_mae = float(np.mean(epoch_maes))
            avg_train_sigma_base = float(np.mean(epoch_base_sigmas))
            avg_train_inflation = float(np.mean(epoch_inflations))
        else:
            avg_train_loss = 0.0
            avg_train_mae = 0.0
            avg_train_sigma_base = 0.0
            avg_train_inflation = 1.0

        history["train_loss"].append(avg_train_loss)
        history["train_nll"].append(float(np.mean(epoch_nlls)) if epoch_nlls else 0.0)
        history["train_pinball"].append(float(np.mean(epoch_pinballs)) if epoch_pinballs else 0.0)
        history["train_nll_log_sigma"].append(float(np.mean(epoch_nll_log_sigmas)) if epoch_nll_log_sigmas else 0.0)
        history["train_nll_residual"].append(float(np.mean(epoch_nll_residuals)) if epoch_nll_residuals else 0.0)
        history["train_sigma_base"].append(float(avg_train_sigma_base))
        history["train_inflation"].append(float(avg_train_inflation))
        history["train_z_rmse"].append(float(np.mean(epoch_z_rmses)) if epoch_z_rmses else 0.0)
        history["train_mae"].append(float(avg_train_mae))

        # --- Validation phase ---------------------------------------------- #
        model.eval()
        val_losses = []
        val_nlls, val_pinballs, val_base_sigmas, val_inflations, val_z_rmses, val_maes = [], [], [], [], [], []
        val_nll_log_sigmas, val_nll_residuals = [], []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    x_imputed, target, ts, dti_tensor = batch
                    if not use_real_dti:
                        dti_tensor = torch.ones_like(dti_tensor)
                else:
                    x, target, ts = batch
                    B, T, K = x.shape
                    if use_real_dti:
                        x_np = x.numpy()
                        ts_np = ts.numpy()
                        dti_np, x_imputed_np = compute_dti_for_batch(
                            x_np, ts_np, n_features,
                            dqa_engine, iri_engine, fusion_engine,
                            corruption_loader, baseline_corr,
                            missing_rate_sampler=missing_rate_sampler,
                            use_dqa=use_dqa, use_iri=use_iri,
                        )
                        dti_tensor = torch.as_tensor(dti_np, dtype=torch.float32)
                        x_imputed = torch.as_tensor(x_imputed_np, dtype=torch.float32)
                    else:
                        # To be fair, baseline gets corrupted/imputed sequence as well
                        x_np = x.numpy()
                        ts_np = ts.numpy()
                        _, x_imputed_np = compute_dti_for_batch(
                            x_np, ts_np, n_features,
                            dqa_engine, iri_engine, fusion_engine,
                            corruption_loader, baseline_corr,
                            missing_rate_sampler=missing_rate_sampler,
                            use_dqa=use_dqa, use_iri=use_iri,
                        )
                        dti_tensor = torch.ones(B, T, dtype=torch.float32)
                        x_imputed = torch.as_tensor(x_imputed_np, dtype=torch.float32)

                x_imputed = x_imputed.to(device)
                target = target.to(device)
                dti_tensor = dti_tensor.to(device)
                mu, sigma, _ = model(x_imputed, dti_tensor)
                mu, sigma, target = mu[:, -1:, :], sigma[:, -1:, :], target[:, -1:, :]
                dti_tensor = dti_tensor[:, -1:]
                loss, diagnostics = loss_fn(mu, sigma, target)
                val_losses.append(loss.item())
                val_nlls.append(diagnostics["nll"].item())
                val_pinballs.append(diagnostics["calibration"].item())
                val_nll_log_sigmas.append(diagnostics["nll_log_sigma"].item())
                val_nll_residuals.append(diagnostics["nll_residual"].item())
                base_sigma, inflation, z_rmse = batch_uncertainty_diagnostics(
                    model, mu, sigma, target, dti_tensor,
                )
                val_base_sigmas.append(base_sigma)
                val_inflations.append(inflation)
                val_z_rmses.append(z_rmse)
                val_maes.append(torch.mean(torch.abs(target - mu)).item())

        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
        avg_val_mae = np.mean(val_maes) if val_maes else float("inf")
        avg_val_sigma_base = np.mean(val_base_sigmas) if val_base_sigmas else 0.0
        avg_val_inflation = np.mean(val_inflations) if val_inflations else 1.0

        history["val_loss"].append(avg_val_loss)
        history["val_nll"].append(float(np.mean(val_nlls)))
        history["val_pinball"].append(float(np.mean(val_pinballs)))
        history["val_nll_log_sigma"].append(float(np.mean(val_nll_log_sigmas)))
        history["val_nll_residual"].append(float(np.mean(val_nll_residuals)))
        history["val_sigma_base"].append(float(avg_val_sigma_base))
        history["val_inflation"].append(float(avg_val_inflation))
        history["val_z_rmse"].append(float(np.mean(val_z_rmses)))
        history["val_mae"].append(float(avg_val_mae))

        # --- Logging ------------------------------------------------------- #
        lr_current = scheduler.get_last_lr()[0]
        logger.info(
            "[%s] Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | val_mae=%.4f | lr=%.6f",
            model_label, epoch + 1, epochs, avg_train_loss, avg_val_loss, avg_val_mae, lr_current,
        )
        logger.info(
            "  -> Train: MAE=%.4f | BaseSigma=%.4f | Inflation=%.4f | NLL(logσ=%.4f, resid=%.4f)",
            avg_train_mae, avg_train_sigma_base, avg_train_inflation,
            history["train_nll_log_sigma"][-1], history["train_nll_residual"][-1],
        )
        logger.info(
            "  -> Val:   MAE=%.4f | BaseSigma=%.4f | Inflation=%.4f | NLL(logσ=%.4f, resid=%.4f)",
            avg_val_mae, avg_val_sigma_base, avg_val_inflation,
            history["val_nll_log_sigma"][-1], history["val_nll_residual"][-1],
        )

        # --- Early stopping + checkpointing on validation MAE ------------- #
        # Checkpoint on val MAE instead of raw NLL so that a healthy,
        # non-collapsed sigma doesn't get penalized relative to a
        # collapsing one that's gaming the NLL loss.  This decouples
        # mu quality from sigma stability in the checkpoint criterion.
        if (epoch + 1) <= val_warmup_epochs:
            logger.info("  Warmup epoch %d/%d: checkpointing & early stopping disabled.", epoch + 1, val_warmup_epochs)
        else:
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                history["best_epoch"] = epoch + 1
                patience_counter = 0
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"best_model_{'calipred' if use_real_dti else 'baseline'}.pt",
                )
                _atomic_torch_save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": best_val_mae,
                    "val_loss": avg_val_loss,
                    "metadata": checkpoint_metadata or {},
                }, ckpt_path)
                logger.info("  Saved best checkpoint (val_mae=%.4f) → '%s'", best_val_mae, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("  Early stopping at epoch %d (patience=%d).", epoch + 1, patience)
                    break

    if history["best_epoch"] == 0:
        history["best_epoch"] = epochs
        ckpt_path = os.path.join(
            checkpoint_dir,
            f"best_model_{'calipred' if use_real_dti else 'baseline'}.pt",
        )
        _atomic_torch_save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": avg_val_mae,
            "val_loss": avg_val_loss,
            "metadata": checkpoint_metadata or {},
        }, ckpt_path)
        logger.info("  No checkpoint saved during warmup; saved final epoch %d as fallback checkpoint → '%s'", epochs, ckpt_path)

    return history


# --------------------------------------------------------------------------- #
# Evaluation on held-out test set
# --------------------------------------------------------------------------- #
def evaluate_model(
    model: CaliPredTransformer,
    test_loader: torch.utils.data.DataLoader,
    dqa_engine: UpstreamDQAEngine,
    iri_engine: ImputationReliabilityEngine,
    fusion_engine: TrustFusionEngine,
    corruption_loader: IndustrialDataLoader,
    baseline_corr: np.ndarray,
    n_features: int,
    device: torch.device = torch.device("cpu"),
    use_real_dti: bool = True,
    label: str = "CALI-PRED",
    split_name: str = "Test",
    missing_rate: float = 0.15,
    missing_rate_sampler: Optional[Callable[[], float]] = None,
    sigma_scale: float = 1.0,
    target_timestamps: Optional[np.ndarray] = None,
    use_dqa: bool = True,
    use_iri: bool = True,
) -> dict:
    """
    Run inference on the test set and compute calibration metrics.

    Returns
    -------
    dict
        Keys: "y_true", "mu", "sigma", "dti", "ece", "brier",
        "nominal_levels", "empirical_coverage", "quality_bins",
        "interval_widths".
    """
    if not np.isfinite(sigma_scale) or sigma_scale <= 0.0:
        raise ValueError("sigma_scale must be finite and positive.")

    model.eval()
    model.to(device)

    K = n_features
    all_mu, all_sigma, all_target, all_dti, all_origin_ts = [], [], [], [], []
    all_base_sigma, all_inflation = [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                x_imputed, target, ts, dti_tensor = batch
                if not use_real_dti:
                    dti_tensor = torch.ones_like(dti_tensor)
            else:
                x, target, ts = batch
                B, T, K = x.shape
                if use_real_dti:
                    x_np = x.numpy()
                    ts_np = ts.numpy()
                    dti_np, x_imputed_np = compute_dti_for_batch(
                        x_np, ts_np, n_features,
                        dqa_engine, iri_engine, fusion_engine,
                        corruption_loader, baseline_corr,
                        missing_rate=missing_rate,
                        missing_rate_sampler=missing_rate_sampler,
                        use_dqa=use_dqa, use_iri=use_iri,
                    )
                    dti_tensor = torch.as_tensor(dti_np, dtype=torch.float32)
                    x_imputed = torch.as_tensor(x_imputed_np, dtype=torch.float32)
                else:
                    # To be fair, baseline gets corrupted/imputed sequence as well
                    x_np = x.numpy()
                    ts_np = ts.numpy()
                    _, x_imputed_np = compute_dti_for_batch(
                        x_np, ts_np, n_features,
                        dqa_engine, iri_engine, fusion_engine,
                        corruption_loader, baseline_corr,
                        missing_rate=missing_rate,
                        missing_rate_sampler=missing_rate_sampler,
                        use_dqa=use_dqa, use_iri=use_iri,
                    )
                    dti_tensor = torch.ones(B, T, dtype=torch.float32)
                    x_imputed = torch.as_tensor(x_imputed_np, dtype=torch.float32)

            x_imputed = x_imputed.to(device)
            target = target.to(device)
            dti_tensor = dti_tensor.to(device)
            mu, sigma, _ = model(x_imputed, dti_tensor)
            inflation = model.uncertainty_inflation(dti_tensor).unsqueeze(-1)
            base_sigma = sigma / inflation.clamp_min(1e-8)

            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_base_sigma.append(base_sigma.cpu().numpy())
            all_inflation.append(inflation.cpu().numpy())
            all_target.append(target.cpu().numpy())
            all_dti.append(dti_tensor.cpu().numpy())
            all_origin_ts.append(ts[:, -1].cpu().numpy())

    # Score one forecast per window: the final output position.
    mu_all = np.concatenate(all_mu)        # (N, T, K)
    sigma_all = np.concatenate(all_sigma)  # (N, T, K)
    target_all = np.concatenate(all_target)  # (N, T, K)
    dti_all = np.concatenate(all_dti)      # (N, T)
    base_sigma_all = np.concatenate(all_base_sigma)[:, -1:, :]
    inflation_all = np.concatenate(all_inflation)[:, -1:, :]
    origin_ts = np.concatenate(all_origin_ts)
    mu_all = mu_all[:, -1:, :]
    sigma_all = sigma_all[:, -1:, :]
    target_all = target_all[:, -1:, :]
    dti_all = dti_all[:, -1:]

    # Flatten to 1-D for metric computation
    y_flat = target_all.flatten()
    mu_flat = mu_all.flatten()
    raw_sigma_flat = sigma_all.flatten()
    sigma_flat = raw_sigma_flat * sigma_scale
    base_sigma_flat = base_sigma_all.flatten()
    inflation_flat = inflation_all.flatten()
    # Per-sample DTI: broadcast (N, T) → (N, T, K) then flatten
    dti_expanded = np.repeat(dti_all[:, :, np.newaxis], K, axis=2)
    dti_flat = dti_expanded.flatten()
    origin_ts_flat = np.repeat(origin_ts, K)
    target_ts_flat = None
    if target_timestamps is not None:
        target_timestamps = np.asarray(target_timestamps)
        if target_timestamps.shape != origin_ts.shape:
            raise ValueError(
                "Target timestamp count does not match forecast origins: "
                f"{target_timestamps.shape} vs {origin_ts.shape}."
            )
        target_ts_flat = np.repeat(target_timestamps, K)

    # ECE
    nominal_levels = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    levels, coverage, mean_ece = expected_calibration_curve(
        y_flat, mu_flat, sigma_flat, nominal_levels=nominal_levels,
    )

    gaussian_crps = calculate_gaussian_crps(y_flat, mu_flat, sigma_flat)

    # Interval width vs DTI bins
    quality_bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    quality_bin_centers = (quality_bin_edges[:-1] + quality_bin_edges[1:]) / 2.0
    bin_indices = np.digitize(dti_flat, quality_bin_edges[1:-1])

    intervals_90 = gaussian_interval(mu_flat, sigma_flat, 0.90)
    widths = intervals_90[:, 1] - intervals_90[:, 0]
    interval_widths = np.array([
        widths[bin_indices == b].mean() if np.any(bin_indices == b) else np.nan
        for b in range(len(quality_bin_centers))
    ])
    standardized_residuals = (y_flat - mu_flat) / sigma_flat
    dti_bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    dti_bin_idx = np.digitize(np.clip(dti_flat, 0.0, 1.0), dti_bin_edges[1:-1])
    residuals_by_dti = []
    for bin_id in range(len(dti_bin_edges) - 1):
        bin_mask = dti_bin_idx == bin_id
        residuals_by_dti.append({
            "dti_lower": float(dti_bin_edges[bin_id]),
            "dti_upper": float(dti_bin_edges[bin_id + 1]),
            "count": int(np.sum(bin_mask)),
            "standardized_residual_mean": float(np.mean(standardized_residuals[bin_mask])) if np.any(bin_mask) else None,
            "standardized_residual_std": float(np.std(standardized_residuals[bin_mask])) if np.any(bin_mask) else None,
        })
    floor = float(getattr(model, "sigma_floor", 0.0))
    floor_dominance = float(np.mean(base_sigma_flat <= floor * (1.0 + 1e-6))) if floor > 0 else 0.0
    saturation = float(np.mean(inflation_flat >= float(model.max_uncertainty_inflation) * (1.0 - 1e-6)))
    logger.info(
        "[%s] uncertainty diagnostics: mean base sigma=%.4f, mean DTI inflation=%.4f, "
        "sigma-floor-dominated=%.2f%%, max-inflation-saturated=%.2f%%",
        label, float(np.mean(base_sigma_flat)), float(np.mean(inflation_flat)),
        100 * floor_dominance, 100 * saturation,
    )

    logger.info(
        "[%s] %s set: ECE=%.4f, Gaussian CRPS=%.4f, N_samples=%d",
        label, split_name, mean_ece, gaussian_crps, len(y_flat),
    )

    return {
        "label": label,
        "y_true": y_flat,
        "mu": mu_flat,
        "sigma": sigma_flat,
        "raw_sigma": raw_sigma_flat,
        "base_sigma": base_sigma_flat,
        "inflation": inflation_flat,
        "standardized_residuals_by_dti": residuals_by_dti,
        "dti": dti_flat,
        "forecast_origin_timestamp": origin_ts_flat,
        "target_timestamp": target_ts_flat,
        "mean_ece": mean_ece,
        "gaussian_crps": gaussian_crps,
        "brier_score": gaussian_crps,  # compatibility for existing analysis scripts
        "nominal_levels": levels,
        "empirical_coverage": coverage,
        "quality_bins": quality_bin_centers,
        "interval_widths": interval_widths,
    }


def _dataset_target_timestamps(dataset) -> np.ndarray:
    """Get one target timestamp per window through cache/subset wrappers."""
    if hasattr(dataset, "base_dataset"):
        return _dataset_target_timestamps(dataset.base_dataset)
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return _dataset_target_timestamps(dataset.dataset)[list(dataset.indices)]
    timestamps = getattr(dataset, "target_timestamps", None)
    if timestamps is not None:
        return np.asarray(timestamps)

    # RealDataset and create_dataloaders' _SlicedDataset expose row timestamps
    # plus each window's start index. The pipeline scores the final output
    # position, whose target is shifted by forecast_horizon rows.
    row_timestamps = getattr(dataset, "timestamps", None)
    windows = getattr(dataset, "windows", None)
    window_size = getattr(dataset, "window_size", None)
    horizon = getattr(dataset, "forecast_horizon", None)
    if row_timestamps is not None and windows is not None and window_size is not None and horizon is not None:
        target_indices = np.asarray(windows, dtype=np.int64) + int(window_size) - 1 + int(horizon)
        if np.any(target_indices < 0) or np.any(target_indices >= len(row_timestamps)):
            raise ValueError("Dataset window metadata produces out-of-range target timestamps.")
        return np.asarray(row_timestamps)[target_indices]
    raise ValueError("Dataset does not expose target timestamps or window timestamp metadata.")


def _dataset_sensor_cols(dataset):
    if hasattr(dataset, "base_dataset"):
        return _dataset_sensor_cols(dataset.base_dataset)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return _dataset_sensor_cols(dataset.dataset)
    return list(dataset.sensor_cols)


def _dataset_time_bounds(dataset):
    if hasattr(dataset, "base_dataset"):
        return _dataset_time_bounds(dataset.base_dataset)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return _dataset_time_bounds(dataset.dataset)
    timestamps = np.asarray(dataset.timestamps)
    return {"start": float(timestamps[0]), "end": float(timestamps[-1])}


def _dataset_scaler(dataset):
    if hasattr(dataset, "base_dataset"):
        return _dataset_scaler(dataset.base_dataset)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return _dataset_scaler(dataset.dataset)
    return dataset.scaler


def _artifact_paths(args):
    root = args.artifact_dir or args.checkpoint_dir
    if args.artifact_dir:
        predictions = os.path.join(root, "predictions")
        results = os.path.join(root, "results")
        preprocessing = os.path.join(root, "preprocessing")
    else:
        predictions = results = args.checkpoint_dir
        preprocessing = os.path.join(root, "preprocessing")
    for path in (root, args.checkpoint_dir, predictions, results, preprocessing):
        os.makedirs(path, exist_ok=True)
    return {"root": root, "predictions": predictions, "results": results,
            "preprocessing": preprocessing}


def _write_run_config(args, datasets, histories=None, runtime_seconds=None):
    train_ds, val_ds, test_ds = datasets
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_commit = None
    histories = histories or {}
    checkpoint_hashes = {}
    for filename in ("best_model_calipred.pt", "best_model_baseline.pt"):
        checkpoint_path = os.path.join(args.checkpoint_dir, filename)
        if os.path.isfile(checkpoint_path):
            digest = hashlib.sha256()
            with open(checkpoint_path, "rb") as checkpoint_file:
                for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    digest.update(block)
            checkpoint_hashes[filename] = digest.hexdigest()
    config = {
        "repository": "Armin1126/CALI-PRED", "git_commit": git_commit,
        "dataset": args.dataset, "seed": args.seed,
        "feature_names": _dataset_sensor_cols(train_ds),
        "train_windows": len(train_ds), "validation_windows": len(val_ds),
        "test_windows": len(test_ds), "window_size": args.window_size,
        "stride": args.stride, "forecast_horizon": args.forecast_horizon,
        "prediction_unit": "one_step_forecast", "development_run": bool(args.development),
        "training_only": bool(args.train_only), "prediction_only": bool(args.predict_only),
        "epochs": args.epochs, "seed": args.seed,
        "calibration_weight": 0.2, "missingness_protocol": "synthetic_MAR",
        "components": {"dqa": not args.disable_dqa, "iri": not args.disable_iri,
                       "trust_attention": not args.disable_trust_attention,
                       "uncertainty_inflation": not args.disable_uncertainty_inflation},
        "clean_fraction": args.clean_fraction, "max_severity": args.max_severity,
        "apply_validation_sigma_scaling": args.apply_validation_sigma_scaling,
        "python_version": platform.python_version(), "pytorch_version": torch.__version__,
        "numpy_version": np.__version__, "pandas_version": __import__("pandas").__version__,
        "sklearn_version": sklearn_version, "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_selection_rule": "minimum validation MAE after warmup",
        "checkpoint_sha256": checkpoint_hashes,
        "best_epoch_calipred": histories.get("calipred", {}).get("best_epoch"),
        "best_epoch_baseline": histories.get("baseline", {}).get("best_epoch"),
        "split_time_boundaries": {"train": _dataset_time_bounds(train_ds),
                                  "validation": _dataset_time_bounds(val_ds),
                                  "test": _dataset_time_bounds(test_ds)},
        "runtime_seconds": runtime_seconds, "arguments": vars(args),
    }
    paths = _artifact_paths(args)
    config_path = os.path.join(paths["root"], "run_config.json")
    temporary = f"{config_path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)
    os.replace(temporary, config_path)
    return config


def _validate_mini_run_report(report_path: str, dataset_name: str, feature_names: list[str]) -> None:
    """Require a completed development run on the same dataset/schema before full training."""
    if not report_path:
        raise ValueError(
            "Full training requires --mini-run-report pointing to a successful development run_config.json."
        )
    report = os.path.abspath(report_path)
    if not os.path.isfile(report):
        raise FileNotFoundError(f"Mini-run report not found: {report}")
    with open(report, encoding="utf-8") as handle:
        config = json.load(handle)
    if not config.get("development_run") or not 2 <= int(config.get("epochs", 0)) <= 5:
        raise ValueError("The supplied run_config.json does not describe a valid 2–5 epoch development run.")
    if config.get("dataset") != dataset_name or config.get("feature_names") != feature_names:
        raise ValueError("Mini-run dataset or feature schema does not match this training run.")
    root = os.path.dirname(report)
    required_files = (
        os.path.join(root, "results", "metrics.json"),
        os.path.join(root, "predictions", "validation_predictions.npz"),
        os.path.join(root, "checkpoints", "best_model_calipred.pt"),
        os.path.join(root, "checkpoints", "best_model_baseline.pt"),
    )
    missing = [path for path in required_files if not os.path.isfile(path)]
    if missing:
        # Legacy/default artifact layout keeps predictions and results beside checkpoints.
        legacy = (
            os.path.join(root, "metrics.json"), os.path.join(root, "validation_predictions.npz"),
            os.path.join(root, "best_model_calipred.pt"), os.path.join(root, "best_model_baseline.pt"),
        )
        if not all(os.path.isfile(path) for path in legacy):
            raise ValueError(f"Mini-run is incomplete; missing required artifacts: {missing}")
    logger.info("Verified successful mini-run report: %s", report)


def _save_preprocessing_artifacts(args, train_ds):
    paths = _artifact_paths(args)
    import joblib
    scaler = _dataset_scaler(train_ds)
    joblib.dump(scaler, os.path.join(paths["preprocessing"], "scaler.joblib"))
    metadata = {
        "dataset": args.dataset, "feature_names": _dataset_sensor_cols(train_ds),
        "scaler_fit_scope": "training split only",
        "scaler_mean": np.asarray(scaler.mean_).tolist(),
        "scaler_scale": np.asarray(scaler.scale_).tolist(),
        "forecast_horizon": args.forecast_horizon,
        "prediction_unit": "one_step_forecast",
    }
    with open(os.path.join(paths["preprocessing"], "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _atomic_save_npz(path, **arrays):
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def validate_prediction_artifact(results: dict, prediction_unit: str) -> None:
    """Fail closed before writing mismatched or invalid prediction arrays."""
    if prediction_unit != "one_step_forecast":
        raise ValueError(
            "Primary prediction artifacts must use prediction_unit="
            "'one_step_forecast'."
        )
    required = ("y_true", "mu", "sigma", "dti", "target_timestamp")
    arrays = {key: np.asarray(results[key]) for key in required}
    shapes = {key: value.shape for key, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Prediction artifact arrays have different shapes: {shapes}.")
    for key, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Prediction artifact field '{key}' contains non-finite values.")
    if np.any(arrays["sigma"] <= 0):
        raise ValueError("Prediction artifact sigma must be strictly positive.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    """Execute the full pipeline: data → DTI → train → evaluate → plot."""

    run_started = time.perf_counter()
    if args.train_only and args.predict_only:
        raise ValueError("--train-only and --predict-only cannot be combined.")
    if args.development and (args.train_only or args.predict_only):
        raise ValueError("--development is a mini-training run and cannot be combined with stage-only flags.")
    if args.development:
        if not 2 <= args.dev_epochs <= 5:
            raise ValueError("--dev-epochs must be between 2 and 5.")
        if args.dev_windows < 2:
            raise ValueError("--dev-windows must be at least 2.")
        args.epochs = args.dev_epochs
        args.max_windows = args.dev_windows
    elif not args.predict_only:
        # Enforce the workflow gate for all new training runs; inference-only runs
        # use an already selected checkpoint and do not need to repeat this check.
        if not args.mini_run_report:
            raise ValueError(
                "Run --development successfully first, then pass its run_config.json via --mini-run-report."
            )
    paths = _artifact_paths(args)

    if args.forecast_horizon != 1:
        raise ValueError("The primary evaluation protocol requires --forecast-horizon 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info("Using random seed: %d", args.seed)

    # ------------------------------------------------------------------ #
    # 1. Load data and create DataLoaders
    # ------------------------------------------------------------------ #
    logger.info("Loading dataset '%s' from '%s'...", args.dataset, args.data_path)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        file_path=args.data_path,
        window_size=args.window_size,
        stride=args.stride,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        random_state=args.seed,
    )

    n_features = train_ds.n_features
    if not args.development and not args.predict_only:
        _validate_mini_run_report(
            args.mini_run_report, args.dataset, _dataset_sensor_cols(train_ds),
        )
    logger.info(
        "Data loaded: %d features, train=%d windows, val=%d windows, test=%d windows.",
        n_features, len(train_ds), len(val_ds), len(test_ds),
    )

    # Optionally limit the number of windows for quick testing
    if args.max_windows is not None:
        from torch.utils.data import Subset
        max_w = args.max_windows
        if len(train_ds) > max_w:
            train_ds = Subset(train_ds, range(min(max_w, len(train_ds))))
        if len(val_ds) > max_w:
            val_ds = Subset(val_ds, range(min(max_w, len(val_ds))))
        if len(test_ds) > max_w:
            test_ds = Subset(test_ds, range(min(max_w, len(test_ds))))
        logger.info("Limited to max %d windows per split.", max_w)

    _save_preprocessing_artifacts(args, train_ds)

    # ------------------------------------------------------------------ #
    # 2. Initialize pipeline components
    # ------------------------------------------------------------------ #
    dqa_engine = UpstreamDQAEngine(
        freshness_tau_seconds=60.0,
        max_corr_mae=0.5,
    )
    iri_engine = ImputationReliabilityEngine(
        n_features=n_features,
        epochs=3 if args.development else args.iri_epochs,
        holdout_frac=0.15,
        random_state=args.seed,
    )
    fusion_engine = TrustFusionEngine(clamp_inputs=True)
    corruption_loader = IndustrialDataLoader(random_state=args.seed)

    # Baseline correlation: compute from training data
    if hasattr(train_ds, "X"):
        train_X = train_ds.X  # (T_train, K), already scaled
    else:
        train_X = train_ds.dataset.X
    baseline_corr = np.corrcoef(train_X.T)  # (K, K)
    baseline_corr = np.nan_to_num(baseline_corr, nan=0.0)
    logger.info(
        "Computed baseline correlation matrix from training data: shape=%s",
        baseline_corr.shape,
    )

    # 2b. Initialize severity samplers for dynamic missingness
    train_val_sampler = make_severity_sampler(
        clean_fraction=args.clean_fraction,
        max_severity=args.max_severity,
        random_state=args.seed,
    )
    test_sampler = make_severity_sampler(
        clean_fraction=args.clean_fraction,
        max_severity=args.max_severity,
        random_state=args.seed + 58,
    )

    # ------------------------------------------------------------------ #
    # 3. Precompute DTI and Imputed Sequences (Caching)
    # ------------------------------------------------------------------ #
    # Precompute train split
    train_dti, train_imputed = precompute_trust_and_imputed(
        train_ds, dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        missing_rate_sampler=train_val_sampler,
        use_dqa=not args.disable_dqa, use_iri=not args.disable_iri,
    )
    train_ds = TrustCachedDataset(train_ds, train_dti, train_imputed)

    # Precompute val split
    val_dti, val_imputed = precompute_trust_and_imputed(
        val_ds, dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        missing_rate_sampler=train_val_sampler,
        use_dqa=not args.disable_dqa, use_iri=not args.disable_iri,
    )
    val_ds = TrustCachedDataset(val_ds, val_dti, val_imputed)

    # Precompute test split
    if not (args.development or args.train_only):
        test_dti, test_imputed = precompute_trust_and_imputed(
            test_ds, dqa_engine, iri_engine, fusion_engine,
            corruption_loader, baseline_corr, n_features,
            missing_rate_sampler=test_sampler,
            use_dqa=not args.disable_dqa, use_iri=not args.disable_iri,
        )
        test_ds = TrustCachedDataset(test_ds, test_dti, test_imputed)

    # Recreate DataLoaders to yield (x_imputed, target, ts, dti)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=(len(train_ds) > args.batch_size)
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )

    logger.info("Analyzing DTI distribution on the precomputed training set...")
    mean_dtis = train_dti.mean(axis=1) # mean per window
    p33, p67 = np.percentile(mean_dtis, [33, 67])
    logger.info(
        "DTI distribution (N=%d): mean=%.4f, std=%.4f, "
        "33rd-pct=%.4f, 67th-pct=%.4f, min=%.4f, max=%.4f",
        len(mean_dtis), mean_dtis.mean(), mean_dtis.std(),
        p33, p67, mean_dtis.min(), mean_dtis.max(),
    )
    logger.info(
        "Data-driven DTI buckets: LOW=[0, %.4f), MED=[%.4f, %.4f), HIGH=[%.4f, 1.0]",
        p33, p33, p67, p67,
    )

    # ------------------------------------------------------------------ #
    # 4. Train CALI-PRED (real DTI)
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Training CALI-PRED (with real DTI)...")
    logger.info("=" * 60)

    model_calipred = CaliPredTransformer(
        input_dim=n_features,
        output_dim=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=0.1,
        max_uncertainty_inflation=args.max_inflation,
        alpha_init=args.alpha_init,
        use_temperature=args.use_temperature,
        sigma_floor=args.sigma_floor,
        sigma_init_bias=args.sigma_init_bias,
        use_trust_attention=not args.disable_trust_attention,
        use_uncertainty_inflation=not args.disable_uncertainty_inflation,
    )
    model_calipred.sigma_lr_multiplier = args.sigma_lr_multiplier

    loss_fn = TrustCalibratedLoss(
        lower_q=0.05, upper_q=0.95, calibration_weight=args.calibration_weight,
        beta_nll=args.beta_nll,
    )

    try:
        code_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        code_revision = None
    checkpoint_metadata = {
        "dataset": args.dataset,
        "feature_names": _dataset_sensor_cols(train_ds),
        "seed": args.seed,
        "model_config": {"input_dim": n_features, "output_dim": n_features,
                         "d_model": args.d_model, "n_heads": args.n_heads,
                         "n_layers": args.n_layers, "max_inflation": args.max_inflation,
                         "alpha_init": args.alpha_init, "use_temperature": args.use_temperature,
                         "dropout": 0.1,
                         "sigma_floor": args.sigma_floor,
                         "sigma_init_bias": args.sigma_init_bias,
                         "use_trust_attention": not args.disable_trust_attention,
                         "use_uncertainty_inflation": not args.disable_uncertainty_inflation},
        "components": {"dqa": not args.disable_dqa, "iri": not args.disable_iri},
        "training_config": {"epochs": args.epochs, "learning_rate": args.lr,
                            "sigma_lr_multiplier": args.sigma_lr_multiplier,
                            "calibration_weight": 0.2, "forecast_horizon": args.forecast_horizon},
        "code_revision": code_revision,
    }

    history_calipred = None
    if not args.predict_only:
        history_calipred = train_model(
            model=model_calipred,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            dqa_engine=dqa_engine,
            iri_engine=iri_engine,
            fusion_engine=fusion_engine,
            corruption_loader=corruption_loader,
            baseline_corr=baseline_corr,
            n_features=n_features,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            use_real_dti=True,
            missing_rate_sampler=train_val_sampler,
            sigma_lr_multiplier=args.sigma_lr_multiplier,
            val_warmup_epochs=args.val_warmup_epochs,
            checkpoint_metadata=checkpoint_metadata,
            use_dqa=not args.disable_dqa, use_iri=not args.disable_iri,
        )

    # ------------------------------------------------------------------ #
    # 5. Train Baseline (DTI=1.0 everywhere)
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Training Baseline (DTI=1.0, trust-blind)...")
    logger.info("=" * 60)

    model_baseline = CaliPredTransformer(
        input_dim=n_features,
        output_dim=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=0.1,
        max_uncertainty_inflation=args.max_inflation,
        alpha_init=args.alpha_init,
        use_temperature=args.use_temperature,
        sigma_floor=args.sigma_floor,
        sigma_init_bias=args.sigma_init_bias,
        use_trust_attention=not args.disable_trust_attention,
        use_uncertainty_inflation=not args.disable_uncertainty_inflation,
    )
    model_baseline.sigma_lr_multiplier = args.sigma_lr_multiplier

    history_baseline = None
    if not args.predict_only:
        history_baseline = train_model(
            model=model_baseline,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            dqa_engine=dqa_engine,
            iri_engine=iri_engine,
            fusion_engine=fusion_engine,
            corruption_loader=corruption_loader,
            baseline_corr=baseline_corr,
            n_features=n_features,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            use_real_dti=False,
            missing_rate_sampler=train_val_sampler,
            sigma_lr_multiplier=args.sigma_lr_multiplier,
            val_warmup_epochs=args.val_warmup_epochs,
            checkpoint_metadata=checkpoint_metadata,
            use_dqa=not args.disable_dqa, use_iri=not args.disable_iri,
        )

    # ------------------------------------------------------------------ #
    # 6. Evaluate on held-out test data
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Loading trained checkpoints for prediction stages...")
    logger.info("=" * 60)

    # Load best checkpoints
    ckpt_cali = os.path.join(args.checkpoint_dir, "best_model_calipred.pt")
    ckpt_base = os.path.join(args.checkpoint_dir, "best_model_baseline.pt")

    if not os.path.isfile(ckpt_cali) or not os.path.isfile(ckpt_base):
        raise FileNotFoundError("Both trained checkpoints are required before prediction/evaluation.")
    checkpoint_calipred = torch.load(ckpt_cali, map_location=device, weights_only=False)
    checkpoint_baseline = torch.load(ckpt_base, map_location=device, weights_only=False)
    model_calipred.load_state_dict(checkpoint_calipred["model_state_dict"])
    model_baseline.load_state_dict(checkpoint_baseline["model_state_dict"])
    logger.info("Loaded trained checkpoints and verified both state dictionaries.")
    if args.predict_only:
        for name, checkpoint in (("CALI-PRED", checkpoint_calipred), ("Baseline", checkpoint_baseline)):
            metadata = checkpoint.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("dataset") != args.dataset:
                raise ValueError(f"{name} checkpoint metadata does not match dataset {args.dataset!r}.")
            if list(metadata.get("feature_names", [])) != _dataset_sensor_cols(train_ds):
                raise ValueError(f"{name} checkpoint feature schema does not match this data split.")
            if metadata.get("seed") != args.seed:
                raise ValueError(f"{name} checkpoint seed does not match --seed {args.seed}.")
            if metadata.get("model_config") != checkpoint_metadata["model_config"]:
                raise ValueError(f"{name} checkpoint model configuration does not match the prediction arguments.")
            if metadata.get("components") != checkpoint_metadata["components"]:
                raise ValueError(f"{name} checkpoint component settings do not match the prediction arguments.")

    if args.train_only:
        _write_run_config(args, (train_ds, val_ds, test_ds),
                          {"calipred": history_calipred or {}, "baseline": history_baseline or {}},
                          time.perf_counter() - run_started)
        history_path = os.path.join(paths["results"], "training_history.npz")
        if history_calipred and history_baseline:
            np.savez(history_path,
                     calipred_train_loss=history_calipred["train_loss"],
                     calipred_val_loss=history_calipred["val_loss"],
                     baseline_train_loss=history_baseline["train_loss"],
                     baseline_val_loss=history_baseline["val_loss"])
        print(f"Training complete. Checkpoints saved under {args.checkpoint_dir}.")
        return

    # Fit global uncertainty correction on validation predictions only.  The
    # scalar is frozen before evaluating the held-out test set.
    val_results_calipred = evaluate_model(
        model_calipred, val_loader,
        dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        device=device, use_real_dti=True, label="CALI-PRED validation",
        split_name="Validation",
        missing_rate_sampler=train_val_sampler,
        target_timestamps=_dataset_target_timestamps(val_ds),
        use_dqa=not args.disable_dqa, use_iri=not args.disable_iri,
    )
    val_results_baseline = evaluate_model(
        model_baseline, val_loader,
        dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        device=device, use_real_dti=False, label="Baseline validation",
        split_name="Validation",
        missing_rate_sampler=train_val_sampler,
        target_timestamps=_dataset_target_timestamps(val_ds),
        use_dqa=not args.disable_dqa, use_iri=not args.disable_iri,
    )
    if not np.array_equal(val_results_calipred["target_timestamp"], val_results_baseline["target_timestamp"]):
        raise ValueError("Validation forecast timestamps differ between models.")
    if not np.array_equal(val_results_calipred["y_true"], val_results_baseline["y_true"]):
        raise ValueError("Validation targets differ between models.")
    calipred_sigma_scale = fit_validation_sigma_scale(
        val_results_calipred["y_true"],
        val_results_calipred["mu"],
        val_results_calipred["sigma"],
    )
    baseline_sigma_scale = fit_validation_sigma_scale(
        val_results_baseline["y_true"],
        val_results_baseline["mu"],
        val_results_baseline["sigma"],
    )
    logger.info(
        "Validation-only NLL sigma scales: CALI-PRED=%.4f, Baseline=%.4f.",
        calipred_sigma_scale, baseline_sigma_scale,
    )

    if args.development:
        validate_prediction_artifact(val_results_calipred, "one_step_forecast")
        validate_prediction_artifact(val_results_baseline, "one_step_forecast")
        dev_validation_path = os.path.join(paths["predictions"], "validation_predictions.npz")
        _atomic_save_npz(
            dev_validation_path,
            prediction_unit=np.array("one_step_forecast"), dataset=np.array(args.dataset),
            seed=np.array(args.seed), feature_names=np.asarray(_dataset_sensor_cols(train_ds)),
            y_true=val_results_calipred["y_true"], mu=val_results_calipred["mu"],
            sigma=val_results_calipred["raw_sigma"], dti=val_results_calipred["dti"],
            target_timestamp=val_results_calipred["target_timestamp"],
            baseline_y_true=val_results_baseline["y_true"], baseline_mu=val_results_baseline["mu"],
            baseline_sigma=val_results_baseline["raw_sigma"],
        )
        z90 = 1.6448536269514722
        dev_metrics = {}
        for label, output in (("calipred", val_results_calipred), ("baseline", val_results_baseline)):
            residual = output["y_true"] - output["mu"]
            sigma = output["raw_sigma"]
            dev_metrics[label] = {
                "mae": float(np.mean(np.abs(residual))),
                "ece": float(output["mean_ece"]),
                "gaussian_crps": float(calculate_gaussian_crps(output["y_true"], output["mu"], sigma)),
                "nominal_90_coverage": float(np.mean(np.abs(residual) <= z90 * sigma)),
                "mean_90_interval_width": float(np.mean(2.0 * z90 * sigma)),
                "forecast_origin_count": int(np.unique(output["target_timestamp"]).size),
            }
        dev_metrics["warning"] = "Development metrics are for debugging only; no test data was evaluated."
        metrics_path = os.path.join(paths["results"], "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(dev_metrics, handle, indent=2)
        histories = {"calipred": history_calipred or {}, "baseline": history_baseline or {}}
        history_path = os.path.join(paths["results"], "training_history.npz")
        if history_calipred and history_baseline:
            np.savez(history_path,
                     calipred_train_loss=history_calipred["train_loss"],
                     calipred_val_loss=history_calipred["val_loss"],
                     baseline_train_loss=history_baseline["train_loss"],
                     baseline_val_loss=history_baseline["val_loss"])
        runtime = time.perf_counter() - run_started
        _write_run_config(args, (train_ds, val_ds, test_ds), histories, runtime)
        logger.info("Development run finished in %.1f seconds; validation metrics saved to '%s'.", runtime, metrics_path)
        print("[OK] Development training and validation complete. The test split was not evaluated.")
        return

    K = n_features
    results_calipred = evaluate_model(
        model_calipred, test_loader,
        dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        device=device, use_real_dti=True, label="CALI-PRED",
        split_name="Test",
        missing_rate_sampler=test_sampler,
        sigma_scale=calipred_sigma_scale if args.apply_validation_sigma_scaling else 1.0,
        target_timestamps=_dataset_target_timestamps(test_ds),
        use_dqa=not args.disable_dqa, use_iri=not args.disable_iri,
    )
    results_baseline = evaluate_model(
        model_baseline, test_loader,
        dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        device=device, use_real_dti=False, label="Baseline",
        split_name="Test",
        missing_rate_sampler=test_sampler,
        sigma_scale=baseline_sigma_scale if args.apply_validation_sigma_scaling else 1.0,
        target_timestamps=_dataset_target_timestamps(test_ds),
        use_dqa=not args.disable_dqa, use_iri=not args.disable_iri,
    )
    validate_prediction_artifact(val_results_calipred, "one_step_forecast")
    validate_prediction_artifact(val_results_baseline, "one_step_forecast")
    validate_prediction_artifact(results_calipred, "one_step_forecast")
    validate_prediction_artifact(results_baseline, "one_step_forecast")
    for split_label, cali_results, baseline_results in (
        ("validation", val_results_calipred, val_results_baseline),
        ("test", results_calipred, results_baseline),
    ):
        if not np.array_equal(cali_results["target_timestamp"], baseline_results["target_timestamp"]):
            raise ValueError(f"{split_label.title()} forecast timestamps differ between models.")
        if not np.array_equal(cali_results["y_true"], baseline_results["y_true"]):
            raise ValueError(f"{split_label.title()} targets differ between models.")
    val_target_origins = np.unique(val_results_calipred["target_timestamp"])
    test_target_origins = np.unique(results_calipred["target_timestamp"])
    if np.intersect1d(val_target_origins, test_target_origins).size:
        raise ValueError("Validation and test target timestamps overlap.")
    if val_target_origins.size and test_target_origins.size and val_target_origins.max() >= test_target_origins.min():
        raise ValueError("Validation targets must precede all test targets chronologically.")

    # ------------------------------------------------------------------ #
    # 7. Report and plot
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  CALIBRATION COMPARISON: Real Data Results")
    print("=" * 60)
    print(f"  Baseline ECE:    {results_baseline['mean_ece']:.4f}")
    print(f"  CALI-PRED ECE:   {results_calipred['mean_ece']:.4f}")
    if results_baseline['mean_ece'] > 0:
        reduction = (
            (results_baseline['mean_ece'] - results_calipred['mean_ece'])
            / results_baseline['mean_ece'] * 100.0
        )
        print(f"  ECE Reduction:   {reduction:.1f}%")
    print(f"  Baseline Gaussian CRPS:   {results_baseline['gaussian_crps']:.4f}")
    print(f"  CALI-PRED Gaussian CRPS:  {results_calipred['gaussian_crps']:.4f}")
    print("=" * 60)

    print("\nMean 90% interval width by DTI bin:")
    print(f"{'DTI bin center':>15} | {'Baseline':>10} | {'CALI-PRED':>10}")
    for center, wb, wc in zip(
        results_baseline['quality_bins'],
        results_baseline['interval_widths'],
        results_calipred['interval_widths'],
    ):
        wb_str = f"{wb:.3f}" if not np.isnan(wb) else "N/A"
        wc_str = f"{wc:.3f}" if not np.isnan(wc) else "N/A"
        print(f"{center:>15.2f} | {wb_str:>10} | {wc_str:>10}")

    # Plot reliability diagram
    calipred_outputs = ModelCalibrationOutputs(
        label="CALI-PRED",
        nominal_levels=results_calipred["nominal_levels"],
        empirical_coverage=results_calipred["empirical_coverage"],
        mean_ece=results_calipred["mean_ece"],
        brier_score=results_calipred["brier_score"],
        quality_bins=results_calipred["quality_bins"],
        interval_widths=results_calipred["interval_widths"],
        color="tab:green",
    ).to_dict()

    baseline_outputs = ModelCalibrationOutputs(
        label="Baseline Predictor",
        nominal_levels=results_baseline["nominal_levels"],
        empirical_coverage=results_baseline["empirical_coverage"],
        mean_ece=results_baseline["mean_ece"],
        brier_score=results_baseline["brier_score"],
        quality_bins=results_baseline["quality_bins"],
        interval_widths=results_baseline["interval_widths"],
        color="tab:red",
    ).to_dict()

    plot_path = os.path.join(paths["results"], "real_data_calibration_comparison.png")
    plot_reliability_diagram(calipred_outputs, baseline_outputs, save_path=plot_path)
    print(f"\nReliability diagram saved to '{plot_path}'.")

    # Save loss history
    history_path = os.path.join(paths["results"], "training_history.npz")
    np.savez(
        history_path,
        calipred_train_loss=history_calipred["train_loss"],
        calipred_val_loss=history_calipred["val_loss"],
        baseline_train_loss=history_baseline["train_loss"],
        baseline_val_loss=history_baseline["val_loss"],
    )
    logger.info("Training history saved to '%s'.", history_path)

    # --- Post-training sanity check: duplicate val_loss detection ---------- #
    cali_vals = np.array(history_calipred["val_loss"])
    base_vals = np.array(history_baseline["val_loss"])
    min_len = min(len(cali_vals), len(base_vals))
    exact_matches = np.where(
        np.isclose(cali_vals[:min_len], base_vals[:min_len], rtol=0, atol=1e-12)
    )[0]
    if len(exact_matches) > 0:
        logger.warning(
            "SUSPICIOUS: Epochs %s have identical val_loss between CALI-PRED "
            "and Baseline (probability of exact float equality between two "
            "independent models is effectively zero — check for shared state "
            "or logging bugs).",
            [int(e) + 1 for e in exact_matches],
        )

    # --- Save raw predictions for bootstrap analysis ----------------------- #
    pred_path = os.path.join(paths["predictions"], "test_predictions.npz")
    _atomic_save_npz(
        pred_path,
        prediction_unit=np.array("one_step_forecast"),
        dataset=np.array(args.dataset),
        seed=np.array(args.seed),
        feature_names=np.asarray(_dataset_sensor_cols(train_ds)),
        window_size=np.array(args.window_size),
        stride=np.array(args.stride),
        forecast_horizon=np.array(args.forecast_horizon),
        missingness_protocol=np.array("synthetic_MAR"),
        clean_fraction=np.array(args.clean_fraction),
        max_severity=np.array(args.max_severity),
        dqa_enabled=np.array(not args.disable_dqa),
        iri_enabled=np.array(not args.disable_iri),
        trust_attention_enabled=np.array(not args.disable_trust_attention),
        uncertainty_inflation_enabled=np.array(not args.disable_uncertainty_inflation),
        d_model=np.array(args.d_model),
        n_heads=np.array(args.n_heads),
        n_layers=np.array(args.n_layers),
        calipred_y_true=results_calipred["y_true"],
        calipred_mu=results_calipred["mu"],
        calipred_sigma=results_calipred["raw_sigma"],
        calipred_dti=results_calipred["dti"],
        baseline_y_true=results_baseline["y_true"],
        baseline_mu=results_baseline["mu"],
        baseline_sigma=results_baseline["raw_sigma"],
        baseline_dti=results_baseline["dti"],
        val_y_true=val_results_calipred["y_true"],
        val_mu=val_results_calipred["mu"],
        val_sigma=val_results_calipred["raw_sigma"],
        val_base_sigma=val_results_calipred["base_sigma"],
        val_inflation=val_results_calipred["inflation"],
        val_dti=val_results_calipred["dti"],
        val_target_timestamp=val_results_calipred["target_timestamp"],
        val_baseline_y_true=val_results_baseline["y_true"],
        val_baseline_mu=val_results_baseline["mu"],
        val_baseline_sigma=val_results_baseline["raw_sigma"],
        val_baseline_dti=val_results_baseline["dti"],
        test_target_timestamp=results_calipred["target_timestamp"],
        test_forecast_origin_timestamp=results_calipred["forecast_origin_timestamp"],
        val_forecast_origin_timestamp=val_results_calipred["forecast_origin_timestamp"],
        calipred_base_sigma=results_calipred["base_sigma"],
        calipred_inflation=results_calipred["inflation"],
        )
    logger.info("Raw test predictions saved to '%s' (for bootstrap analysis).", pred_path)
    validation_path = os.path.join(paths["predictions"], "validation_predictions.npz")
    _atomic_save_npz(
        validation_path,
        prediction_unit=np.array("one_step_forecast"), dataset=np.array(args.dataset),
        seed=np.array(args.seed), feature_names=np.asarray(_dataset_sensor_cols(train_ds)),
        y_true=val_results_calipred["y_true"], mu=val_results_calipred["mu"],
        sigma=val_results_calipred["raw_sigma"], dti=val_results_calipred["dti"],
        target_timestamp=val_results_calipred["target_timestamp"],
        baseline_y_true=val_results_baseline["y_true"],
        baseline_mu=val_results_baseline["mu"], baseline_sigma=val_results_baseline["raw_sigma"],
        baseline_dti=val_results_baseline["dti"],
    )
    z90 = 1.6448536269514722
    metric_summary = {}
    for split_name, model_results in (
        ("validation", {"calipred": val_results_calipred, "baseline": val_results_baseline}),
        ("test", {"calipred": results_calipred, "baseline": results_baseline}),
    ):
        metric_summary[split_name] = {}
        for model_name, output in model_results.items():
            residual = output["y_true"] - output["mu"]
            sigma = output["sigma"]
            metric_summary[split_name][model_name] = {
                "mae": float(np.mean(np.abs(residual))),
                "ece": float(output["mean_ece"]),
                "gaussian_crps": float(calculate_gaussian_crps(output["y_true"], output["mu"], sigma)),
                "nominal_90_coverage": float(np.mean(np.abs(residual) <= z90 * sigma)),
                "mean_90_interval_width": float(np.mean(2.0 * z90 * sigma)),
                "forecast_origin_count": int(np.unique(output["target_timestamp"]).size),
            }
    metrics_path = os.path.join(paths["results"], "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metric_summary, handle, indent=2)
    _write_run_config(args, (train_ds, val_ds, test_ds),
                      {"calipred": history_calipred, "baseline": history_baseline},
                      time.perf_counter() - run_started)

    print("\n[OK] Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CALI-PRED End-to-End Real-Data Pipeline",
    )
    parser.add_argument(
        "--dataset", type=str, default="metropt",
        choices=["metropt", "ai4i2020", "tep"],
        help="Dataset name (default: metropt).",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/metropt/MetroPT3(AirCompressor).csv",
        help="Path to the raw CSV file.",
    )
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--max-inflation", type=float, default=10.0,
                        help="Ceiling on DTI-driven sigma inflation (default: 10.0).")
    parser.add_argument("--alpha-init", type=float, default=0.5,
                        help="Initial value for trust-sensitivity exponent (default: 0.5).")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Limit windows per split for quick testing.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Root for run_config.json and separated preprocessing/prediction/result artifacts.")
    parser.add_argument("--development", action="store_true",
                        help="Run a 2–5 epoch, single-seed mini-train and validation only; never evaluate test data.")
    parser.add_argument("--dev-epochs", type=int, default=3,
                        help="Epoch count for --development (must be 2–5).")
    parser.add_argument("--dev-windows", type=int, default=16,
                        help="Maximum windows per split for --development.")
    parser.add_argument("--train-only", action="store_true",
                        help="Train and select checkpoints using validation; do not score the test split.")
    parser.add_argument("--predict-only", action="store_true",
                        help="Load existing compatible checkpoints and produce validation/test predictions.")
    parser.add_argument("--mini-run-report", type=str, default=None,
                        help="Successful development run_config.json required before non-development training.")
    parser.add_argument("--clean-fraction", type=float, default=0.25,
                        help="Fraction of clean windows (missing_rate = 0.0) sampled.")
    parser.add_argument("--max-severity", type=float, default=0.45,
                        help="Maximum missingness rate for corrupted windows.")
    parser.add_argument(
        "--apply-validation-sigma-scaling", action="store_true",
        help=("Apply the global sigma scale fitted on validation predictions "
              "only; intended for the calibration experiment."),
    )
    parser.add_argument(
        "--sigma-lr-multiplier", type=float, default=0.5,
        help="Multiplier for the learning rate of the sigma head (default: 0.5). "
             "Values < 1.0 slow sigma training to prevent NLL collapse.",
    )
    parser.add_argument(
        "--val-warmup-epochs", type=int, default=3,
        help="Number of initial warmup epochs before checkpoint eligibility (default: 3).",
    )
    parser.add_argument(
        "--use-temperature", action="store_true",
        help="Use a global learnable temperature scaling on predicted base sigma.",
    )
    parser.add_argument(
        "--sigma-floor", type=float, default=0.1,
        help="Additive floor on base sigma to prevent NLL collapse (default: 0.1).",
    )
    parser.add_argument(
        "--sigma-init-bias", type=float, default=0.5,
        help="Initial bias for sigma head; softplus(0.5) ~ 1.07 (default: 0.5).",
    )
    parser.add_argument(
        "--calibration-weight", type=float, default=0.2,
        help="Weight for the pinball calibration term; keep fixed within an experiment.",
    )
    parser.add_argument(
        "--iri-epochs", type=int, default=30,
        help="Self-supervised imputer epochs per window (development mode uses 3).",
    )
    parser.add_argument("--disable-trust-attention", action="store_true",
                        help="Ablation switch: use ordinary all-trust attention weights.")
    parser.add_argument("--disable-uncertainty-inflation", action="store_true",
                        help="Ablation switch: disable DTI-conditioned sigma inflation.")
    parser.add_argument("--disable-dqa", action="store_true",
                        help="Ablation switch: replace DQA quality scores with full trust.")
    parser.add_argument("--disable-iri", action="store_true",
                        help="Ablation switch: use mean imputation and full IRI reliability.")
    parser.add_argument(
        "--beta-nll", type=float, default=0.0,
        help="Beta-NLL reweighting exponent (Seitzer et al. 2022). "
             "0.0 = standard NLL, 0.5 = balanced dampening (default: 0.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic initialization and splits (default: 42).",
    )

    args = parser.parse_args()
    main(args)


