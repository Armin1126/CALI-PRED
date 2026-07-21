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
import logging
import math
import os
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("CaliPredPipeline")

# Local imports
from data_loader import (
    IndustrialDataLoader,
    RealCorruptionInjector,
    create_dataloaders,
)
from dqa_module import UpstreamDQAEngine
from fusion_engine import TrustFusionEngine
from iri_module import ImputationReliabilityEngine
from predictor import CaliPredTransformer, TrustCalibratedLoss
from metrics_engine import (
    expected_calibration_curve,
    calculate_brier_score,
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
    sigma_lr_multiplier: float = 2.5,
    val_warmup_epochs: int = 3,
) -> dict:
    """
    Real DataLoader-based training loop with:
    - Real DTI computation per batch (or uniform DTI=1.0 for baseline)
    - Cosine annealing LR scheduler
    - Gradient clipping
    - Validation loss tracking + early stopping
    - Model checkpointing

    Returns
    -------
    dict
        Training history with keys "train_loss", "val_loss", "best_epoch".
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Decoupled learning rates for sigma head and trust parameters if multiplier != 1.0
    if hasattr(model, "sigma_head") and sigma_lr_multiplier != 1.0:
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
            {"params": sigma_params, "lr": lr * sigma_lr_mult}
        ])
        logger.info(
            "Optimizer: decoupled learning rates. General LR = %.6f, Sigma LR = %.6f (multiplier = %.2f)",
            lr, lr * sigma_lr_mult, sigma_lr_mult
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    history = {
        "train_loss": [], "val_loss": [], "ema_val_loss": [], "best_epoch": 0,
        "train_nll": [], "val_nll": [], "train_pinball": [], "val_pinball": [],
        "train_sigma_base": [], "val_sigma_base": [],
        "train_inflation": [], "val_inflation": [], "train_z_rmse": [], "val_z_rmse": [],
        "train_mae": [], "val_mae": [],
    }
    best_val_loss = float("inf")
    ema_val_loss = None
    ema_alpha = 0.3  # EMA smoothing: dampen single-epoch spikes in non-monotonic NLL
    patience = 10
    patience_counter = 0

    model.to(device)
    model_label = "CALI-PRED" if use_real_dti else "Baseline (DTI=1.0)"

    for epoch in range(epochs):
        # --- Training phase ------------------------------------------------ #
        model.train()
        epoch_losses = []
        epoch_nlls, epoch_pinballs, epoch_base_sigmas, epoch_inflations, epoch_z_rmses, epoch_maes = [], [], [], [], [], []

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
                    )
                    dti_tensor = torch.ones(B, T, dtype=torch.float32)
                    x_imputed = torch.as_tensor(x_imputed_np, dtype=torch.float32)

            x_imputed = x_imputed.to(device)
            target = target.to(device)
            dti_tensor = dti_tensor.to(device)

            optimizer.zero_grad()
            mu, sigma, _ = model(x_imputed, dti_tensor)
            loss, diagnostics = loss_fn(mu, sigma, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_nlls.append(diagnostics["nll"].item())
            epoch_pinballs.append(diagnostics["calibration"].item())
            base_sigma, inflation, z_rmse = batch_uncertainty_diagnostics(
                model, mu, sigma, target, dti_tensor,
            )
            epoch_base_sigmas.append(base_sigma); epoch_inflations.append(inflation); epoch_z_rmses.append(z_rmse)
            epoch_maes.append(torch.mean(torch.abs(target - mu)).item())

        scheduler.step()
        avg_train_loss = np.mean(epoch_losses)
        avg_train_mae = np.mean(epoch_maes)
        avg_train_sigma_base = np.mean(epoch_base_sigmas)
        avg_train_inflation = np.mean(epoch_inflations)

        history["train_loss"].append(avg_train_loss)
        history["train_nll"].append(float(np.mean(epoch_nlls)))
        history["train_pinball"].append(float(np.mean(epoch_pinballs)))
        history["train_sigma_base"].append(float(avg_train_sigma_base))
        history["train_inflation"].append(float(avg_train_inflation))
        history["train_z_rmse"].append(float(np.mean(epoch_z_rmses)))
        history["train_mae"].append(float(avg_train_mae))

        # --- Validation phase ---------------------------------------------- #
        model.eval()
        val_losses = []
        val_nlls, val_pinballs, val_base_sigmas, val_inflations, val_z_rmses, val_maes = [], [], [], [], [], []

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
                        )
                        dti_tensor = torch.ones(B, T, dtype=torch.float32)
                        x_imputed = torch.as_tensor(x_imputed_np, dtype=torch.float32)

                x_imputed = x_imputed.to(device)
                target = target.to(device)
                dti_tensor = dti_tensor.to(device)
                mu, sigma, _ = model(x_imputed, dti_tensor)
                loss, diagnostics = loss_fn(mu, sigma, target)
                val_losses.append(loss.item())
                val_nlls.append(diagnostics["nll"].item())
                val_pinballs.append(diagnostics["calibration"].item())
                base_sigma, inflation, z_rmse = batch_uncertainty_diagnostics(
                    model, mu, sigma, target, dti_tensor,
                )
                val_base_sigmas.append(base_sigma)
                val_inflations.append(inflation)
                val_z_rmses.append(z_rmse)
                val_maes.append(torch.mean(torch.abs(target - mu)).item())

        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
        avg_val_mae = np.mean(val_maes) if val_maes else 0.0
        avg_val_sigma_base = np.mean(val_base_sigmas) if val_base_sigmas else 0.0
        avg_val_inflation = np.mean(val_inflations) if val_inflations else 1.0

        # Update EMA of validation loss for smoothed checkpoint selection
        if ema_val_loss is None:
            ema_val_loss = avg_val_loss
        else:
            ema_val_loss = ema_alpha * avg_val_loss + (1.0 - ema_alpha) * ema_val_loss

        history["val_loss"].append(avg_val_loss)
        history["ema_val_loss"].append(ema_val_loss)
        history["val_nll"].append(float(np.mean(val_nlls)))
        history["val_pinball"].append(float(np.mean(val_pinballs)))
        history["val_sigma_base"].append(float(avg_val_sigma_base))
        history["val_inflation"].append(float(avg_val_inflation))
        history["val_z_rmse"].append(float(np.mean(val_z_rmses)))
        history["val_mae"].append(float(avg_val_mae))

        # --- Logging ------------------------------------------------------- #
        lr_current = scheduler.get_last_lr()[0]
        logger.info(
            "[%s] Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | ema_val=%.4f | lr=%.6f",
            model_label, epoch + 1, epochs, avg_train_loss, avg_val_loss, ema_val_loss, lr_current,
        )
        logger.info(
            "  -> Train: MAE=%.4f | BaseSigma=%.4f | Inflation=%.4f",
            avg_train_mae, avg_train_sigma_base, avg_train_inflation
        )
        logger.info(
            "  -> Val:   MAE=%.4f | BaseSigma=%.4f | Inflation=%.4f",
            avg_val_mae, avg_val_sigma_base, avg_val_inflation
        )

        # --- Early stopping + checkpointing (EMA-smoothed) ---------------- #
        if (epoch + 1) <= val_warmup_epochs:
            logger.info("  Warmup epoch %d/%d: checkpointing & early stopping disabled.", epoch + 1, val_warmup_epochs)
        else:
            if ema_val_loss < best_val_loss:
                best_val_loss = ema_val_loss
                history["best_epoch"] = epoch + 1
                patience_counter = 0
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"best_model_{'calipred' if use_real_dti else 'baseline'}.pt",
                )
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                }, ckpt_path)
                logger.info("  Saved best checkpoint (ema_val=%.4f) → '%s'", best_val_loss, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("  Early stopping at epoch %d (patience=%d).", epoch + 1, patience)
                    break

    if history["best_epoch"] == 0:
        history["best_epoch"] = epochs
        best_val_loss = ema_val_loss if ema_val_loss is not None else float("inf")
        ckpt_path = os.path.join(
            checkpoint_dir,
            f"best_model_{'calipred' if use_real_dti else 'baseline'}.pt",
        )
        torch.save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
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
    all_mu, all_sigma, all_target, all_dti = [], [], [], []

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
                    )
                    dti_tensor = torch.ones(B, T, dtype=torch.float32)
                    x_imputed = torch.as_tensor(x_imputed_np, dtype=torch.float32)

            x_imputed = x_imputed.to(device)
            target = target.to(device)
            dti_tensor = dti_tensor.to(device)
            mu, sigma, _ = model(x_imputed, dti_tensor)

            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_target.append(target.cpu().numpy())
            all_dti.append(dti_tensor.cpu().numpy())

    # Concatenate and flatten for scalar metrics
    mu_all = np.concatenate(all_mu)        # (N, T, K)
    sigma_all = np.concatenate(all_sigma)  # (N, T, K)
    target_all = np.concatenate(all_target)  # (N, T, K)
    dti_all = np.concatenate(all_dti)      # (N, T)

    # Flatten to 1-D for metric computation
    y_flat = target_all.flatten()
    mu_flat = mu_all.flatten()
    raw_sigma_flat = sigma_all.flatten()
    sigma_flat = raw_sigma_flat * sigma_scale
    # Per-sample DTI: broadcast (N, T) → (N, T, K) then flatten
    dti_expanded = np.repeat(dti_all[:, :, np.newaxis], K, axis=2)
    dti_flat = dti_expanded.flatten()

    # ECE
    nominal_levels = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    levels, coverage, mean_ece = expected_calibration_curve(
        y_flat, mu_flat, sigma_flat, nominal_levels=nominal_levels,
    )

    # Brier / CRPS
    brier = calculate_brier_score(y_flat, mu_flat, sigma_flat)

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

    logger.info(
        "[%s] %s set: ECE=%.4f, Brier(CRPS)=%.4f, N_samples=%d",
        label, split_name, mean_ece, brier, len(y_flat),
    )

    return {
        "label": label,
        "y_true": y_flat,
        "mu": mu_flat,
        "sigma": sigma_flat,
        "raw_sigma": raw_sigma_flat,
        "dti": dti_flat,
        "mean_ece": mean_ece,
        "brier_score": brier,
        "nominal_levels": levels,
        "empirical_coverage": coverage,
        "quality_bins": quality_bin_centers,
        "interval_widths": interval_widths,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    """Execute the full pipeline: data → DTI → train → evaluate → plot."""

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

    # ------------------------------------------------------------------ #
    # 2. Initialize pipeline components
    # ------------------------------------------------------------------ #
    dqa_engine = UpstreamDQAEngine(
        freshness_tau_seconds=60.0,
        max_corr_mae=0.5,
    )
    iri_engine = ImputationReliabilityEngine(
        n_features=n_features,
        epochs=30,  # lighter for pipeline use
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
    )
    train_ds = TrustCachedDataset(train_ds, train_dti, train_imputed)

    # Precompute val split
    val_dti, val_imputed = precompute_trust_and_imputed(
        val_ds, dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        missing_rate_sampler=train_val_sampler,
    )
    val_ds = TrustCachedDataset(val_ds, val_dti, val_imputed)

    # Precompute test split
    test_dti, test_imputed = precompute_trust_and_imputed(
        test_ds, dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        missing_rate_sampler=test_sampler,
    )
    test_ds = TrustCachedDataset(test_ds, test_dti, test_imputed)

    # Recreate DataLoaders to yield (x_imputed, target, ts, dti)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
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
    )
    model_calipred.sigma_lr_multiplier = args.sigma_lr_multiplier

    loss_fn = TrustCalibratedLoss(
        lower_q=0.05, upper_q=0.95, calibration_weight=0.2,
    )

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
    )
    model_baseline.sigma_lr_multiplier = args.sigma_lr_multiplier

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
    )

    # ------------------------------------------------------------------ #
    # 6. Evaluate on held-out test data
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Evaluating on held-out test set...")
    logger.info("=" * 60)

    # Load best checkpoints
    ckpt_cali = os.path.join(args.checkpoint_dir, "best_model_calipred.pt")
    ckpt_base = os.path.join(args.checkpoint_dir, "best_model_baseline.pt")

    if os.path.exists(ckpt_cali):
        model_calipred.load_state_dict(
            torch.load(ckpt_cali, map_location=device, weights_only=False)["model_state_dict"]
        )
        logger.info("Loaded best CALI-PRED checkpoint from '%s'.", ckpt_cali)

    if os.path.exists(ckpt_base):
        model_baseline.load_state_dict(
            torch.load(ckpt_base, map_location=device, weights_only=False)["model_state_dict"]
        )
        logger.info("Loaded best Baseline checkpoint from '%s'.", ckpt_base)

    # Fit global uncertainty correction on validation predictions only.  The
    # scalar is frozen before evaluating the held-out test set.
    val_results_calipred = evaluate_model(
        model_calipred, val_loader,
        dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        device=device, use_real_dti=True, label="CALI-PRED validation",
        split_name="Validation",
        missing_rate_sampler=train_val_sampler,
    )
    val_results_baseline = evaluate_model(
        model_baseline, val_loader,
        dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        device=device, use_real_dti=False, label="Baseline validation",
        split_name="Validation",
        missing_rate_sampler=train_val_sampler,
    )
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

    K = n_features
    results_calipred = evaluate_model(
        model_calipred, test_loader,
        dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        device=device, use_real_dti=True, label="CALI-PRED",
        split_name="Test",
        missing_rate_sampler=test_sampler,
        sigma_scale=calipred_sigma_scale if args.apply_validation_sigma_scaling else 1.0,
    )
    results_baseline = evaluate_model(
        model_baseline, test_loader,
        dqa_engine, iri_engine, fusion_engine,
        corruption_loader, baseline_corr, n_features,
        device=device, use_real_dti=False, label="Baseline",
        split_name="Test",
        missing_rate_sampler=test_sampler,
        sigma_scale=baseline_sigma_scale if args.apply_validation_sigma_scaling else 1.0,
    )

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
    print(f"  Baseline CRPS:   {results_baseline['brier_score']:.4f}")
    print(f"  CALI-PRED CRPS:  {results_calipred['brier_score']:.4f}")
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

    plot_path = os.path.join(
        args.checkpoint_dir, "real_data_calibration_comparison.png"
    )
    plot_reliability_diagram(calipred_outputs, baseline_outputs, save_path=plot_path)
    print(f"\nReliability diagram saved to '{plot_path}'.")

    # Save loss history
    history_path = os.path.join(args.checkpoint_dir, "training_history.npz")
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
    pred_path = os.path.join(args.checkpoint_dir, "test_predictions.npz")
    np.savez(
        pred_path,
        calipred_y_true=results_calipred["y_true"],
        calipred_mu=results_calipred["mu"],
        calipred_sigma=results_calipred["sigma"],
        calipred_dti=results_calipred["dti"],
        baseline_y_true=results_baseline["y_true"],
        baseline_mu=results_baseline["mu"],
        baseline_sigma=results_baseline["sigma"],
        baseline_dti=results_baseline["dti"],
    )
    logger.info("Raw test predictions saved to '%s' (for bootstrap analysis).", pred_path)

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
        "--sigma-lr-multiplier", type=float, default=2.5,
        help="Multiplier for the learning rate of the sigma head (default: 2.5).",
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
        "--seed", type=int, default=42,
        help="Random seed for deterministic initialization and splits (default: 42).",
    )

    args = parser.parse_args()
    main(args)
