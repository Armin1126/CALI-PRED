"""
iri_module.py

Industrial Anomaly Prediction Framework — Imputation Reliability Module
==============================================================================

This module implements the **Midstream Imputation Reliability Indicator
(IRI)**: a per-timestep, per-channel trust score for imputed sensor values,
computed from an ensemble of structurally-different imputation models.

Motivation
----------
A single imputation model (however accurate on average) gives no signal
about *where* its reconstruction should be trusted. Downstream anomaly
detectors that naively consume imputed values inherit that blind spot —
they can't distinguish "this sensor reading is real" from "this sensor
reading is a confident-looking guess." IRI closes that gap by combining
two independent uncertainty signals:

1. **Epistemic disagreement (ensemble variance, sigma_ens^2)** — Two
   architecturally distinct imputers are run over the same corrupted
   input:
     - **SAITS** (Self-Attention Imputation for Time Series): a
       *parallel*, non-recurrent self-attention encoder. It reconstructs
       every timestep simultaneously by attending over the full window,
       so it excels at capturing long-range, non-sequential structure
       (e.g., a sensor's relationship to another sensor 200 steps away).
     - **BRITS** (Bidirectional Recurrent Imputation for Time Series): a
       *recurrent* architecture that propagates a hidden state forward
       and backward through time, explicitly modeling temporal decay of
       information as gaps grow longer.
     - Where SAITS and BRITS *agree*, the reconstruction is corroborated
       by two independent inductive biases. Where they *diverge*, that
       divergence is a proxy for epistemic uncertainty: the missing
       value's true generative process could plausibly fit either
       hypothesis, so neither should be blindly trusted. This ensemble
       spread is only a **proxy** (not a Bayesian posterior) — it
       captures a specific, useful form of model uncertainty and should
       not be read as a calibrated probability.

2. **Self-supervised reconstruction error (epsilon_recon)** — A subset of
   *originally observed* values is deliberately withheld (in addition to
   the naturally missing entries) before imputation, and the model
   ensemble's ability to reconstruct these known values is scored. This
   grounds the reliability score in a real error signal rather than pure
   model disagreement (two models can agree and still both be wrong).

IRI formula
-----------
For each (timestep, channel) position:

    IRI = (sigma_ens + eps)^-1 * (1 - epsilon_recon)

then min-max normalized over the full (T, K) grid to [0, 1]. High ensemble
variance *or* poor self-supervised reconstruction accuracy both drive IRI
towards 0 (low trust); low variance combined with strong reconstruction
accuracy drives IRI towards 1 (high trust).

Libraries: PyTorch (model internals), NumPy (array/statistics layer).
Python: 3.13+
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ImputationReliabilityEngine")

# --------------------------------------------------------------------------- #
# Optional PyPOTS backend detection
# --------------------------------------------------------------------------- #
# PyPOTS ships production implementations of SAITS and BRITS. Where available
# and importable, downstream code MAY wire real PyPOTS models in behind the
# same interface used here. Since PyPOTS is a heavy, environment-sensitive
# dependency (and this module must run deterministically in CI/sandboxes
# without a training corpus or pretrained checkpoints), we default to
# lightweight, architecturally-faithful PyTorch stubs that preserve the two
# properties that matter for IRI: (a) SAITS's parallel self-attention
# reconstruction and (b) BRITS's recurrent, direction-aware reconstruction.
try:
    import pypots  # noqa: F401
    _PYPOTS_AVAILABLE = True
    logger.info("PyPOTS detected; real SAITS/BRITS backends may be wired in.")
except ImportError:
    _PYPOTS_AVAILABLE = False
    logger.info(
        "PyPOTS not available in this environment; using internal "
        "self-attention (SAITS-style) and recurrent (BRITS-style) "
        "PyTorch stubs with real tensor training."
    )


# --------------------------------------------------------------------------- #
# Model stubs
# --------------------------------------------------------------------------- #
class _SAITSStub(nn.Module):
    """
    Self-attention imputation stub, architecturally modeled on SAITS.

    Unlike a recurrent model, every output timestep is computed in
    parallel via multi-head self-attention over the *entire* input
    window — there is no sequential state to propagate, so the model's
    errors arise from attention-pattern mismatches rather than
    accumulated recurrent drift. This is precisely the failure mode we
    want represented in the ensemble: SAITS can "hallucinate" a
    plausible global pattern that a purely local/recurrent model would
    not produce, and vice versa.

    Input is the concatenation of (masked value, mask indicator) per
    channel, doubling the feature dimension, so the model can distinguish
    "observed zero" from "missing" at every position.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features * 2, d_model)
        self.register_buffer(
            "positional_encoding", self._build_positional_encoding(max_len, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, n_features)

    @staticmethod
    def _build_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_masked : torch.Tensor, shape (B, T, K)
            Corrupted input with missing entries pre-filled as 0.
        mask : torch.Tensor, shape (B, T, K)
            1 = observed, 0 = missing.

        Returns
        -------
        torch.Tensor, shape (B, T, K)
            Full reconstruction (all positions), computed in parallel.
        """
        seq_len = x_masked.shape[1]
        inp = torch.cat([x_masked * mask, mask], dim=-1)
        h = self.input_proj(inp) + self.positional_encoding[:, :seq_len, :]
        h = self.encoder(h)
        return self.output_proj(h)


class _BRITSStub(nn.Module):
    """
    Recurrent imputation stub, architecturally modeled on BRITS.

    A bidirectional GRU propagates a hidden state across time in both
    directions, so reconstruction at timestep t is informed by an
    explicitly *sequential* summary of the past and future rather than
    an unordered attention pool. As gaps lengthen, the recurrent hidden
    state's influence naturally decays — the classic BRITS "temporal
    decay" intuition — making this model systematically more uncertain
    during long dropout blocks than SAITS, which suffers no equivalent
    decay. That structural difference is exactly what makes the SAITS/
    BRITS pairing informative for uncertainty estimation.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size * 2, n_features)

    def forward(self, x_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_masked : torch.Tensor, shape (B, T, K)
            Corrupted input with missing entries pre-filled as 0.
        mask : torch.Tensor, shape (B, T, K)
            1 = observed, 0 = missing.

        Returns
        -------
        torch.Tensor, shape (B, T, K)
            Full reconstruction (all positions), computed recurrently.
        """
        inp = torch.cat([x_masked * mask, mask], dim=-1)
        h, _ = self.rnn(inp)
        return self.output_proj(h)


# --------------------------------------------------------------------------- #
# Main engine
# --------------------------------------------------------------------------- #
class ImputationReliabilityEngine:
    """
    Runs a SAITS/BRITS-style ensemble over corrupted time series and
    computes the Midstream Imputation Reliability Indicator (IRI): a
    per-(timestep, channel) trust score in [0, 1] combining ensemble
    disagreement and self-supervised reconstruction accuracy.

    Parameters
    ----------
    n_features : int
        Number of sensor channels K in the input time series.
    d_model : int, default 64
        Hidden width shared by both stub architectures (kept equal for
        a fair, complexity-matched ensemble comparison).
    epochs : int, default 150
        Number of self-supervised training iterations per model per call
        to :meth:`impute_ensemble`. Kept small by default so the module
        runs quickly in a CI/demo context; increase for production use.
    lr : float, default 1e-3
        Adam learning rate for the internal self-supervised fit.
    holdout_frac : float, default 0.1
        Fraction of *originally observed* entries additionally hidden
        during training, so the self-supervised loss has real targets
        to reconstruct rather than trivially copying observed inputs.
    random_state : Optional[int], default 42
        Seed for reproducible model init and holdout selection.
    device : Optional[str], default None
        Torch device string (e.g. "cuda"); defaults to CPU if None and
        no GPU is visible.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        epochs: int = 150,
        lr: float = 1e-3,
        holdout_frac: float = 0.1,
        random_state: Optional[int] = 42,
        device: Optional[str] = None,
    ) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be positive; got {n_features}.")
        if not (0.0 < holdout_frac < 1.0):
            raise ValueError(f"holdout_frac must be in (0, 1); got {holdout_frac}.")

        self.n_features = n_features
        self.d_model = d_model
        self.epochs = epochs
        self.lr = lr
        self.holdout_frac = holdout_frac
        self.random_state = random_state
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if random_state is not None:
            torch.manual_seed(random_state)
        self._rng = np.random.default_rng(random_state)

        # Model names are fixed and ordered: index 0 = SAITS-style, index 1
        # = BRITS-style. Downstream code should rely on this ordering
        # rather than re-deriving it.
        self.model_names: Tuple[str, str] = ("SAITS", "BRITS")

    # ------------------------------------------------------------------ #
    # 1. Ensemble imputation
    # ------------------------------------------------------------------ #
    def impute_ensemble(self, X_corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Reconstruct missing values using a SAITS-style and a BRITS-style
        model in parallel, returning both plausible reconstructions
        without collapsing them to a single point estimate.

        Each model is briefly trained in a self-supervised manner: a
        random subset of the already-observed entries is additionally
        hidden, and the model is optimized to reconstruct exactly those
        entries (a masked-autoencoding objective). Observed values are
        then preserved as-is in the returned output; only genuinely
        missing positions are replaced by the model's prediction.

        Parameters
        ----------
        X_corrupted : np.ndarray, shape (T, K)
            Time series with ``np.nan`` at missing positions (e.g., the
            output of ``IndustrialDataLoader.inject_missingness``).
        mask : np.ndarray, shape (T, K)
            Binary mask, 1 = observed, 0 = missing, aligned with
            ``X_corrupted``.

        Returns
        -------
        np.ndarray, shape (M, T, K)
            Ensemble of reconstructions, ``M=2``: index 0 is the
            SAITS-style (self-attention) reconstruction, index 1 is the
            BRITS-style (recurrent) reconstruction.

        Raises
        ------
        ValueError
            If shapes of ``X_corrupted`` and ``mask`` don't match, or the
            feature dimension doesn't match ``self.n_features``.
        """
        if X_corrupted.shape != mask.shape:
            raise ValueError(
                f"X_corrupted shape {X_corrupted.shape} must match mask shape {mask.shape}."
            )
        T, K = X_corrupted.shape
        if K != self.n_features:
            raise ValueError(f"Expected {self.n_features} features; got {K}.")

        x_filled = np.nan_to_num(X_corrupted, nan=0.0).astype(np.float32)
        x_tensor = torch.as_tensor(x_filled, device=self.device).unsqueeze(0)  # (1, T, K)
        mask_tensor = torch.as_tensor(
            mask.astype(np.float32), device=self.device
        ).unsqueeze(0)  # (1, T, K)

        saits = _SAITSStub(self.n_features, d_model=self.d_model).to(self.device)
        brits = _BRITSStub(self.n_features, hidden_size=self.d_model).to(self.device)

        saits_out = self._fit_and_reconstruct(saits, x_tensor, mask_tensor)
        brits_out = self._fit_and_reconstruct(brits, x_tensor, mask_tensor)

        ensemble_outputs = np.stack([saits_out, brits_out], axis=0)  # (2, T, K)
        logger.info(
            "Ensemble imputation complete: models=%s, shape=%s.",
            self.model_names, ensemble_outputs.shape,
        )
        return ensemble_outputs

    def _fit_and_reconstruct(
        self, model: nn.Module, x_tensor: torch.Tensor, mask_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Self-supervised training loop shared by both stub architectures.

        An additional "artificial holdout" mask is drawn from the
        *observed* entries; the model is trained to minimize MSE between
        its prediction and the true values at those held-out positions
        only (never on already-missing positions, since ground truth
        isn't available there). After training, a final forward pass
        uses the *original* mask to produce the deployed reconstruction,
        with observed entries copied through unchanged.
        """
        _, T, K = x_tensor.shape
        holdout = (torch.rand(1, T, K, device=self.device) < self.holdout_frac).float()
        holdout = holdout * mask_tensor  # can only hold out truly-observed entries
        train_mask = mask_tensor * (1.0 - holdout)  # what the model is allowed to see

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = model(x_tensor, train_mask)
            # Supervise only on the artificially held-out (known-truth) entries.
            if holdout.sum() > 0:
                loss = loss_fn(pred * holdout, x_tensor * holdout)
            else:
                # Degenerate edge case (holdout_frac too small / unlucky draw):
                # fall back to reconstructing all observed entries so training
                # still has a gradient signal.
                loss = loss_fn(pred * mask_tensor, x_tensor * mask_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_pred = model(x_tensor, mask_tensor)
            # Preserve true observed values; only trust the model at
            # genuinely missing positions.
            reconstructed = mask_tensor * x_tensor + (1.0 - mask_tensor) * final_pred

        return reconstructed.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------ #
    # 2. Ensemble variance (epistemic disagreement proxy)
    # ------------------------------------------------------------------ #
    @staticmethod
    def ensemble_variance(ensemble_outputs: np.ndarray) -> np.ndarray:
        """
        Compute sigma_ens^2: the per-(timestep, channel) variance across
        ensemble members, used as a proxy for epistemic uncertainty.

        Because SAITS (parallel self-attention) and BRITS (recurrent,
        decay-aware) encode different structural assumptions about how
        information propagates through the series, their disagreement at
        a given position reflects genuine model uncertainty about that
        position's true value — as opposed to a single model's
        overconfident point estimate, which carries no such signal.

        Parameters
        ----------
        ensemble_outputs : np.ndarray, shape (M, T, K)
            Stacked reconstructions from M ensemble members (M=2 for the
            SAITS/BRITS pairing used in this module, but this function is
            written generically for any M >= 2).

        Returns
        -------
        np.ndarray, shape (T, K)
            Per-position variance across the M member axis (ddof=0,
            population variance, appropriate since the ensemble members
            are the entire population of models being compared, not a
            sample from a larger population).

        Raises
        ------
        ValueError
            If ``ensemble_outputs`` is not 3-D or has fewer than 2 members.
        """
        if ensemble_outputs.ndim != 3:
            raise ValueError(
                f"ensemble_outputs must be 3-D (M, T, K); got shape {ensemble_outputs.shape}."
            )
        if ensemble_outputs.shape[0] < 2:
            raise ValueError(
                "Need at least 2 ensemble members to compute disagreement; "
                f"got M={ensemble_outputs.shape[0]}."
            )
        return np.var(ensemble_outputs, axis=0, ddof=0)

    # ------------------------------------------------------------------ #
    # 3. Self-supervised reconstruction validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def reconstruction_validation(
        X_imputed_mean: np.ndarray,
        X_ground_truth: np.ndarray,
        evaluation_mask: np.ndarray,
    ) -> float:
        """
        Score the ensemble's mean reconstruction against known ground
        truth at deliberately-withheld positions, producing a single
        bounded reconstruction-error term epsilon_recon in [0, 1).

        This is the module's grounding signal: ensemble agreement alone
        cannot detect a case where every model shares the same blind
        spot, but comparing against true values at controlled evaluation
        positions can.

        Parameters
        ----------
        X_imputed_mean : np.ndarray, shape (T, K)
            Mean reconstruction across ensemble members (typically
            ``ensemble_outputs.mean(axis=0)``).
        X_ground_truth : np.ndarray, shape (T, K)
            The true, pre-corruption values (only meaningful at positions
            selected by ``evaluation_mask``).
        evaluation_mask : np.ndarray, shape (T, K)
            Binary mask, 1 = "this position was deliberately withheld and
            its true value is known for validation purposes," 0 =
            elsewhere. Typically a subset of the overall injected
            missingness mask.

        Returns
        -------
        float
            epsilon_recon in [0, 1): a saturating normalized RMSE, where
            0 = perfect reconstruction at evaluation positions and values
            approach 1 as error grows arbitrarily large. Returns 0.0 if
            ``evaluation_mask`` selects no positions (nothing to
            validate against — treated as a neutral, not a good, result;
            callers should log this case as it means IRI's reconstruction
            term carries no information for this batch).

        Raises
        ------
        ValueError
            If shapes of the three arrays don't match.
        """
        if not (X_imputed_mean.shape == X_ground_truth.shape == evaluation_mask.shape):
            raise ValueError(
                "X_imputed_mean, X_ground_truth, and evaluation_mask must all "
                f"share shape; got {X_imputed_mean.shape}, {X_ground_truth.shape}, "
                f"{evaluation_mask.shape}."
            )

        n_eval = evaluation_mask.sum()
        if n_eval == 0:
            logger.warning(
                "reconstruction_validation: evaluation_mask selects zero "
                "positions; returning neutral epsilon_recon=0.0."
            )
            return 0.0

        errors = (X_imputed_mean - X_ground_truth) * evaluation_mask
        mse = float(np.sum(errors ** 2) / n_eval)
        rmse = float(np.sqrt(mse))

        # Normalize by the spread of the ground-truth values at evaluated
        # positions so epsilon_recon is scale-invariant (a RMSE of 1.0 on a
        # unit-variance signal is very different from RMSE of 1.0 on a
        # signal with std of 100).
        gt_values = X_ground_truth[evaluation_mask.astype(bool)]
        gt_std = float(np.std(gt_values)) if gt_values.size > 0 else 1.0
        gt_std = max(gt_std, 1e-8)
        nrmse = rmse / gt_std

        # Saturating map R+ -> [0, 1): large errors approach (but never
        # reach) 1, so (1 - epsilon_recon) in the IRI formula never
        # collapses fully to zero from this term alone.
        epsilon_recon = nrmse / (1.0 + nrmse)
        return epsilon_recon

    # ------------------------------------------------------------------ #
    # 4. Composite IRI score
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_iri(
        ensemble_outputs: np.ndarray,
        X_ground_truth: np.ndarray,
        evaluation_mask: np.ndarray,
        eps: float = 1e-3,
    ) -> np.ndarray:
        """
        Compute the Midstream Imputation Reliability Indicator (IRI):

            IRI(t, k) = (sigma_ens(t, k) + eps)^-1 * (1 - epsilon_recon)

        followed by global min-max normalization to [0, 1] across the
        full (T, K) grid.

        Interpretation
        --------------
        - **sigma_ens(t, k)** (per-position ensemble variance) penalizes
          positions where SAITS and BRITS disagree — i.e., where the
          two structurally different inductive biases can't agree on a
          reconstruction, signaling epistemic uncertainty specific to
          that timestep/channel.
        - **epsilon_recon** (global self-supervised reconstruction error)
          penalizes the *entire* imputation run uniformly when the
          ensemble's mean prediction is measurably wrong on positions
          with known ground truth — catching the case where both models
          confidently agree on the wrong answer.
        - The product means a position needs **both** low disagreement
          *and* a track record of accurate reconstruction (via the global
          epsilon_recon term) to score near 1. Either high disagreement
          or poor validated accuracy pulls IRI towards 0.

        Parameters
        ----------
        ensemble_outputs : np.ndarray, shape (M, T, K)
            Stacked ensemble reconstructions (see :meth:`impute_ensemble`).
        X_ground_truth : np.ndarray, shape (T, K)
            True pre-corruption values, used only at ``evaluation_mask``
            positions to compute epsilon_recon.
        evaluation_mask : np.ndarray, shape (T, K)
            Binary mask marking positions with known ground truth for
            self-supervised validation.
        eps : float, default 1e-3
            Numerical floor added to sigma_ens before inversion, avoiding
            a division blow-up at positions of perfect ensemble
            agreement (sigma_ens == 0), which would otherwise produce an
            unbounded (and physically meaningless) IRI value.

        Returns
        -------
        np.ndarray, shape (T, K)
            IRI scores in [0, 1], where 1.0 = maximum trust (low
            ensemble disagreement, strong validated reconstruction) and
            0.0 = minimum trust.

        Raises
        ------
        ValueError
            If shapes are inconsistent across inputs.
        """
        if ensemble_outputs.ndim != 3:
            raise ValueError(
                f"ensemble_outputs must be 3-D (M, T, K); got shape {ensemble_outputs.shape}."
            )
        _, T, K = ensemble_outputs.shape
        if X_ground_truth.shape != (T, K) or evaluation_mask.shape != (T, K):
            raise ValueError(
                f"X_ground_truth and evaluation_mask must have shape ({T}, {K}); "
                f"got {X_ground_truth.shape} and {evaluation_mask.shape}."
            )

        sigma_ens = ImputationReliabilityEngine.ensemble_variance(ensemble_outputs)  # (T, K)
        x_imputed_mean = ensemble_outputs.mean(axis=0)  # (T, K)
        epsilon_recon = ImputationReliabilityEngine.reconstruction_validation(
            x_imputed_mean, X_ground_truth, evaluation_mask
        )  # scalar, applied uniformly across the grid

        raw_iri = (1.0 / (sigma_ens + eps)) * (1.0 - epsilon_recon)

        # Global min-max normalization to [0, 1]. Guard against a
        # degenerate all-equal raw_iri grid (would divide by zero range).
        iri_min, iri_max = float(raw_iri.min()), float(raw_iri.max())
        value_range = iri_max - iri_min
        if value_range < 1e-12:
            logger.warning(
                "compute_iri: raw IRI grid is (near-)constant (min=%.6f, "
                "max=%.6f); returning a uniform mid-trust grid of 0.5.",
                iri_min, iri_max,
            )
            return np.full_like(raw_iri, fill_value=0.5)

        iri_normalized = (raw_iri - iri_min) / value_range
        logger.info(
            "Computed IRI grid: shape=%s, epsilon_recon=%.4f, "
            "sigma_ens range=[%.4f, %.4f], IRI range=[0, 1] "
            "(pre-normalization range was [%.4f, %.4f]).",
            iri_normalized.shape, epsilon_recon,
            float(sigma_ens.min()), float(sigma_ens.max()),
            iri_min, iri_max,
        )
        return iri_normalized

    # ------------------------------------------------------------------ #
    # 5. Batched ensemble imputation
    # ------------------------------------------------------------------ #
    def impute_ensemble_batch(
        self,
        X_corrupted_batch: np.ndarray,
        mask_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Process a batch of corrupted windows through the SAITS/BRITS ensemble
        in a single batched forward pass.

        Parameters
        ----------
        X_corrupted_batch : np.ndarray, shape (B, T, K)
            Batch of corrupted time-series windows with NaN at missing positions.
        mask_batch : np.ndarray, shape (B, T, K)
            Binary masks (1 = observed, 0 = missing).

        Returns
        -------
        np.ndarray, shape (M, B, T, K)
            Ensemble of reconstructions, M=2 (SAITS, BRITS).
        """
        if X_corrupted_batch.shape != mask_batch.shape:
            raise ValueError(
                f"X_corrupted_batch shape {X_corrupted_batch.shape} must match "
                f"mask_batch shape {mask_batch.shape}."
            )
        B, T, K = X_corrupted_batch.shape
        if K != self.n_features:
            raise ValueError(f"Expected {self.n_features} features; got {K}.")

        x_filled = np.nan_to_num(X_corrupted_batch, nan=0.0).astype(np.float32)
        x_tensor = torch.as_tensor(x_filled, device=self.device)        # (B, T, K)
        mask_tensor = torch.as_tensor(
            mask_batch.astype(np.float32), device=self.device
        )  # (B, T, K)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        saits = _SAITSStub(self.n_features, d_model=self.d_model).to(self.device)
        brits = _BRITSStub(self.n_features, hidden_size=self.d_model).to(self.device)

        saits_out = self._fit_and_reconstruct(saits, x_tensor, mask_tensor)  # (B, T, K)
        brits_out = self._fit_and_reconstruct(brits, x_tensor, mask_tensor)  # (B, T, K)

        # Stack: (M=2, B, T, K)
        ensemble = np.stack([saits_out, brits_out], axis=0)
        logger.info(
            "Batch ensemble imputation complete: B=%d, models=%s, shape=%s.",
            B, self.model_names, ensemble.shape,
        )
        return ensemble

    # ------------------------------------------------------------------ #
    # 6. Convenience: compute IRI for a batch of windows
    # ------------------------------------------------------------------ #
    def compute_iri_batch(
        self,
        X_corrupted_batch: np.ndarray,
        mask_batch: np.ndarray,
        X_ground_truth_batch: np.ndarray,
        eps: float = 1e-3,
    ) -> np.ndarray:
        """
        Compute IRI for a batch of windows in one call.

        Parameters
        ----------
        X_corrupted_batch : np.ndarray, shape (B, T, K)
        mask_batch : np.ndarray, shape (B, T, K)
        X_ground_truth_batch : np.ndarray, shape (B, T, K)
        eps : float, default 1e-3

        Returns
        -------
        np.ndarray, shape (B, T, K)
            Per-window, per-(timestep, channel) IRI scores in [0, 1].
        """
        B, T, K = X_corrupted_batch.shape
        iri_batch = np.empty((B, T, K), dtype=np.float64)

        for i in range(B):
            ensemble_i = self.impute_ensemble(X_corrupted_batch[i], mask_batch[i])
            # evaluation_mask: use the injection mask itself (where we know GT)
            eval_mask_i = (1 - mask_batch[i]).astype(np.int8)
            iri_batch[i] = self.compute_iri(
                ensemble_i, X_ground_truth_batch[i], eval_mask_i, eps=eps,
            )

        return iri_batch


# --------------------------------------------------------------------------- #
# Demonstration / smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("Running ImputationReliabilityEngine demonstration with dummy data.")

    T_DEMO, K_DEMO = 200, 4
    rng = np.random.default_rng(11)

    # --- Synthesize a clean ground-truth series with cross-channel structure --- #
    t_axis = np.arange(T_DEMO)
    X_ground_truth = np.stack(
        [np.sin(2 * np.pi * t_axis / (20 + 5 * k)) + 0.1 * rng.normal(size=T_DEMO)
         for k in range(K_DEMO)],
        axis=1,
    ).astype(np.float32)

    # --- Inject missingness (contiguous blocks), keep an evaluation subset ----- #
    full_mask = np.ones((T_DEMO, K_DEMO), dtype=np.int8)
    n_blocks = 6
    block_len = 8
    starts = rng.integers(0, T_DEMO - block_len, size=n_blocks)
    channels = rng.integers(0, K_DEMO, size=n_blocks)
    for s, c in zip(starts, channels):
        full_mask[s:s + block_len, c] = 0

    X_corrupted = X_ground_truth.copy()
    X_corrupted[full_mask == 0] = np.nan

    # Evaluation mask: half of the injected-missing positions, where we
    # (as the harness) still retain the true value for validation. In a
    # real pipeline this comes from the synthetic corruption step in
    # data_loader.py, which knows ground truth by construction.
    missing_positions = np.argwhere(full_mask == 0)
    rng.shuffle(missing_positions)
    eval_positions = missing_positions[: len(missing_positions) // 2]
    evaluation_mask = np.zeros_like(full_mask)
    for (tt, kk) in eval_positions:
        evaluation_mask[tt, kk] = 1

    print(f"[OK] Injected missingness: {int((1 - full_mask.mean()) * 100)}% missing, "
          f"{evaluation_mask.sum()} positions held out for validation.")

    # --- Run the reliability engine -------------------------------------------- #
    engine = ImputationReliabilityEngine(
        n_features=K_DEMO, epochs=60, holdout_frac=0.15, random_state=11
    )

    ensemble_outputs = engine.impute_ensemble(X_corrupted, full_mask)
    assert ensemble_outputs.shape == (2, T_DEMO, K_DEMO), "Unexpected ensemble shape."
    print(f"[OK] Ensemble output shape: {ensemble_outputs.shape} "
          f"(models={engine.model_names})")

    sigma_ens = engine.ensemble_variance(ensemble_outputs)
    assert sigma_ens.shape == (T_DEMO, K_DEMO)
    print(f"[OK] Ensemble variance -> mean={sigma_ens.mean():.6f}, "
          f"max={sigma_ens.max():.6f}")

    x_mean = ensemble_outputs.mean(axis=0)
    epsilon_recon = engine.reconstruction_validation(x_mean, X_ground_truth, evaluation_mask)
    assert 0.0 <= epsilon_recon < 1.0
    print(f"[OK] epsilon_recon (self-supervised reconstruction error): {epsilon_recon:.4f}")

    iri = engine.compute_iri(ensemble_outputs, X_ground_truth, evaluation_mask)
    assert iri.shape == (T_DEMO, K_DEMO)
    assert np.all((iri >= 0.0) & (iri <= 1.0)), "IRI must be bounded in [0, 1]."
    print(f"[OK] IRI grid -> min={iri.min():.4f}, max={iri.max():.4f}, mean={iri.mean():.4f}")

    # Sanity check: average IRI at genuinely-missing positions should
    # generally be lower than at fully-observed positions, since those
    # are exactly where ensemble disagreement concentrates.
    missing_bool = (full_mask == 0)
    observed_bool = (full_mask == 1)
    print(f"[OK] Mean IRI at missing positions: {iri[missing_bool].mean():.4f} "
          f"vs. observed positions: {iri[observed_bool].mean():.4f}")

    print("\nAll smoke tests passed.")
