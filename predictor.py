"""
predictor.py

Industrial Anomaly Prediction Framework — Quality-Conditioned Transformer
==============================================================================

This module implements ``CaliPredTransformer``: a Transformer forecasting
backbone whose internal attention mechanism and output uncertainty are both
explicitly conditioned on the Continuous Trust Signal (the Data Trust Index,
DTI) produced by ``fusion_engine.py``.

Why condition the *architecture* on trust, not just the *loss*?
------------------------------------------------------------------
A conventional approach bolts uncertainty estimation onto a trust-blind
backbone: train normally, then post-hoc widen confidence intervals wherever
DTI is low. That's strictly weaker than what this module does, for two
reasons:

1. **Attention should stop aggregating information from low-trust
   timesteps in the first place.** If a stretch of the input window sits
   on a stale/corrupted sensor segment (low DTI), a trust-blind
   self-attention layer will still happily attend to it and blend its
   (untrustworthy) representation into every other timestep's context —
   contaminating predictions that shouldn't have been affected at all.
   ``TrustAwareAttention`` (below) directly down-weights low-trust key
   positions *inside* the attention weight matrix, so untrustworthy
   segments are prevented from propagating their influence through the
   network, rather than being cleaned up after the fact.

2. **Predictive uncertainty should be architecturally *forced* to inflate
   under low trust, not just learned to (hopefully) correlate with it.**
   ``CaliPredTransformer``'s output head multiplies its learned base
   variance by an explicit, monotonically-decreasing function of DTI —
   as DTI -> 0, predicted sigma is driven towards its configured ceiling
   *regardless of what the network otherwise believes*, giving a hard
   architectural guarantee (not just a training-time incentive) that low
   trust always yields wider intervals.

Components
----------
- ``TrustAwareAttention``: multi-head self-attention with DTI-modulated,
  renormalized attention weights.
- ``TrustAwareEncoderLayer``: standard pre-norm residual Transformer block
  built around ``TrustAwareAttention``.
- ``CaliPredTransformer``: the full stacked encoder plus a heteroscedastic
  (mu, sigma) output head with DTI-conditioned variance inflation.
- ``TrustCalibratedLoss``: a Gaussian NLL loss combined with a pinball
  (quantile) calibration term for the 5% / 95% predictive bounds.

Tensor dimension legend used throughout: ``B`` = batch size, ``T`` =
sequence length (timesteps), ``K`` = input feature/channel count, ``D`` =
Transformer hidden width (``d_model``), ``H`` = number of attention heads,
``O`` = output/target dimension.

Python: 3.13+
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("CaliPredTransformer")


# --------------------------------------------------------------------------- #
# Trust-aware attention
# --------------------------------------------------------------------------- #
class TrustAwareAttention(nn.Module):
    """
    Multi-head self-attention whose attention weight matrix is modulated
    by a per-timestep Data Trust Index (DTI) signal, so the network
    explicitly down-weights low-trust time segments during temporal
    aggregation instead of treating every timestep as equally reliable
    context.

    Mechanism
    ---------
    After the standard scaled dot-product softmax produces attention
    weights ``attn[b, h, t_query, t_key]``, each column ``t_key`` is
    scaled by that key position's trust value ``dti[b, t_key]``. Because
    this multiplicative gating no longer sums to 1 along the key axis,
    the weights are renormalized so every query still produces a valid
    convex combination of value vectors — one that has been reweighted
    *away from* low-trust keys and *towards* high-trust keys, rather than
    one that has simply been scaled down in magnitude. A hard attention
    mask (fully excluding a key) is the ``dti -> 0`` limit of this
    mechanism; intermediate trust values interpolate smoothly between
    "fully attend" and "fully ignore."

    Parameters
    ----------
    d_model : int
        Transformer hidden width. Must be divisible by ``n_heads``.
    n_heads : int
        Number of attention heads.
    dropout : float, default 0.1
        Dropout applied to the (trust-modulated, renormalized) attention
        weights before the value aggregation.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, dti: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, D]
            Hidden representations for the current layer.
        dti : Tensor, shape [B, T]
            Per-timestep Data Trust Index in ``[0, 1]``; ``1`` = fully
            trusted, ``0`` = fully untrusted. Applied along the *key*
            (context) axis of attention.
        eps : float, default 1e-8
            Numerical floor added to the renormalization denominator, so
            a query whose entire trusted-key mass collapses to zero
            (every key in the window is fully untrusted) doesn't produce
            a divide-by-zero.

        Returns
        -------
        Tuple[Tensor, Tensor]
            - ``context`` : Tensor, shape [B, T, D] — attention output,
              same shape as the input.
            - ``attn_weights`` : Tensor, shape [B, H, T, T] — the final,
              trust-modulated and renormalized attention weight matrix
              (post-dropout), returned for inspection/diagnostics (e.g.,
              visualizing which timesteps a prediction actually drew on).
        """
        B, T, D = x.shape  # [B, T, D]

        # Project and reshape into per-head query/key/value tensors.
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]

        # Standard scaled dot-product attention scores.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, T_q, T_k]
        attn = torch.softmax(scores, dim=-1)  # [B, H, T_q, T_k], sums to 1 over T_k

        # --- Trust modulation -------------------------------------------------- #
        # dti: [B, T] -> [B, 1, 1, T] so it broadcasts over heads (H) and the
        # query axis (T_q), applying uniformly along the key axis (T_k).
        trust_gate = dti.clamp(0.0, 1.0).view(B, 1, 1, T)  # [B, 1, 1, T_k]
        attn_trust = attn * trust_gate  # [B, H, T_q, T_k], low-trust keys downweighted

        # Renormalize so each query's weights are a valid convex combination
        # again (sum to 1 over T_k), redistributing mass towards trusted keys
        # rather than merely shrinking the aggregated value's magnitude.
        normalizer = attn_trust.sum(dim=-1, keepdim=True).clamp_min(eps)  # [B, H, T_q, 1]
        attn_trust = attn_trust / normalizer  # [B, H, T_q, T_k]

        attn_trust = self.dropout(attn_trust)

        context = torch.matmul(attn_trust, V)  # [B, H, T_q, d_k]
        context = context.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        context = self.out_proj(context)  # [B, T, D]

        return context, attn_trust


# --------------------------------------------------------------------------- #
# Encoder layer
# --------------------------------------------------------------------------- #
class TrustAwareEncoderLayer(nn.Module):
    """
    A pre-norm Transformer encoder block built around
    :class:`TrustAwareAttention`, with a standard position-wise
    feed-forward sub-layer and residual connections.

    Parameters
    ----------
    d_model : int
        Hidden width.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward inner dimension.
    dropout : float, default 0.1
        Dropout applied within attention and the feed-forward sub-layer,
        and to both residual branches.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = TrustAwareAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, dti: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, D]
        dti : Tensor, shape [B, T]

        Returns
        -------
        Tuple[Tensor, Tensor]
            - Updated hidden states, shape [B, T, D].
            - Attention weights from this layer, shape [B, H, T, T].
        """
        attn_out, attn_weights = self.attn(x, dti)  # [B, T, D], [B, H, T, T]
        x = self.norm1(x + self.dropout(attn_out))  # [B, T, D]
        ff_out = self.feed_forward(x)  # [B, T, D]
        x = self.norm2(x + self.dropout(ff_out))  # [B, T, D]
        return x, attn_weights


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class CaliPredTransformer(nn.Module):
    """
    Quality-Conditioned Transformer for probabilistic industrial
    time-series forecasting: a ``TrustAwareAttention``-based encoder
    stack, topped with a heteroscedastic (mu, sigma) output head whose
    variance is architecturally inflated as the Data Trust Index (DTI)
    degrades.

    Uncertainty scaling
    -------------------
    The output head first predicts a data-driven "base" variance
    ``sigma_base`` in the usual heteroscedastic-regression sense
    (capturing ordinary aleatoric noise the network has learned from
    data). This is then scaled by an explicit trust-inflation factor:

        inflation(t) = min( (1 / (DTI(t) + eps)) ** alpha, max_inflation )
        sigma(t)     = sigma_base(t) * inflation(t)

    ``alpha`` is a learned, strictly-positive scalar (parameterized via
    softplus) controlling how aggressively low trust inflates
    uncertainty; ``max_inflation`` is a fixed ceiling preventing the
    inflation factor from diverging as DTI approaches exactly 0. This
    realizes ``Final_Uncertainty ∝ f(DTI^-1)`` with ``f(z) = z^alpha``,
    clipped for numerical stability — at ``DTI = 1`` the inflation factor
    is ``1.0`` (no effect); as ``DTI -> 0`` it saturates at
    ``max_inflation``.

    .. note:: The default ``max_inflation`` is 10.0, not 50.0. For
       normalized data (sigma_base ~ 1.0), an inflation of 50× produces
       intervals of ±164σ, which are absurdly wide and guaranteed to
       over-cover at every nominal level, *hurting* ECE rather than
       helping it. 10× is aggressive enough to force meaningful widening
       under low trust while remaining in a regime where the pinball
       calibration loss can still usefully shape the distribution.

    Parameters
    ----------
    input_dim : int
        Number of input sensor channels, ``K``.
    output_dim : Optional[int], default None
        Number of target variables to forecast per timestep, ``O``.
        Defaults to ``input_dim`` (full multivariate reconstruction /
        forecasting) if not specified.
    d_model : int, default 64
        Transformer hidden width.
    n_heads : int, default 4
        Number of attention heads per layer.
    n_layers : int, default 3
        Number of stacked ``TrustAwareEncoderLayer`` blocks.
    d_ff : int, default 256
        Feed-forward inner dimension per layer.
    dropout : float, default 0.1
        Dropout probability used throughout the encoder.
    max_len : int, default 4096
        Maximum sequence length supported by the sinusoidal positional
        encoding buffer.
    max_uncertainty_inflation : float, default 10.0
        Ceiling on the DTI-driven variance inflation factor (see above).
    alpha_init : float, default 0.5
        Initial value for the learned trust-sensitivity exponent
        ``alpha`` (in raw/unconstrained space, before softplus). A value
        of 0.5 maps to ``softplus(0.5) ≈ 0.974``, producing moderate
        inflation at DTI = 0.5 (~1.97×) while leaving DTI = 1.0 at
        exactly 1.0×. Set to 0.0 to recover the original behavior
        (``softplus(0) ≈ 0.693``).
    sigma_floor : float, default 1e-3
        Additive floor on the base (pre-inflation) predicted sigma,
        preventing a degenerate zero-variance (infinitely confident)
        prediction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 4096,
        max_uncertainty_inflation: float = 10.0,
        alpha_init: float = 0.5,
        sigma_floor: float = 1e-3,
        use_temperature: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.d_model = d_model
        self.max_uncertainty_inflation = max_uncertainty_inflation
        self.sigma_floor = sigma_floor
        self.use_temperature = use_temperature

        self.input_proj = nn.Linear(input_dim, d_model)
        self.register_buffer(
            "positional_encoding", self._build_positional_encoding(max_len, d_model)
        )
        self.layers = nn.ModuleList(
            [TrustAwareEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Separate heads for mu (predictions) and sigma (base uncertainty)
        self.mu_head = nn.Linear(d_model, self.output_dim)
        self.sigma_head = nn.Linear(d_model, self.output_dim)

        # Learnable global scale factor for base sigma (log space, exp(0.0) = 1.0)
        self.sigma_temperature_raw = nn.Parameter(torch.tensor([0.0]))

        # Learnable, strictly-positive exponent controlling how sharply the
        # inflation factor grows as DTI decreases. Parameterized in raw
        # (unconstrained) space and passed through softplus at use time.
        self.trust_sensitivity_raw = nn.Parameter(torch.tensor([float(alpha_init)]))

    @staticmethod
    def _build_positional_encoding(max_len: int, d_model: int) -> Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, D]

    def forward(
        self, x: Tensor, dti: Tensor, eps: float = 1e-4
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, K]
            Input multivariate sensor window (typically post-imputation
            output from the midstream stage of the pipeline).
        dti : Tensor, shape [B, T]
            Per-timestep Data Trust Index in ``[0, 1]`` (the fused
            output of ``fusion_engine.TrustFusionEngine.compute_dti``).
        eps : float, default 1e-4
            Numerical floor added to ``dti`` before inversion, preventing
            a division blow-up at ``dti == 0`` beyond the configured
            ``max_uncertainty_inflation`` ceiling.

        Returns
        -------
        Tuple[Tensor, Tensor, List[Tensor]]
            - ``mu`` : Tensor, shape [B, T, O] — point predictions.
            - ``sigma`` : Tensor, shape [B, T, O] — trust-inflated
              predictive standard deviation, strictly positive.
            - ``attn_maps`` : List[Tensor], one entry per encoder layer,
              each of shape [B, H, T, T] — trust-modulated attention
              weights, useful for diagnosing which timesteps informed a
              given prediction.
        """
        if x.shape[0] != dti.shape[0] or x.shape[1] != dti.shape[1]:
            raise ValueError(
                f"x (shape {tuple(x.shape)}) and dti (shape {tuple(dti.shape)}) "
                "must share the same batch size and sequence length."
            )

        B, T, K = x.shape  # [B, T, K]
        h = self.input_proj(x) + self.positional_encoding[:, :T, :]  # [B, T, D]

        attn_maps: List[Tensor] = []
        for layer in self.layers:
            h, attn_weights = layer(h, dti)  # [B, T, D], [B, H, T, T]
            attn_maps.append(attn_weights)

        mu = self.mu_head(h)  # [B, T, O]
        sigma_raw = self.sigma_head(h)  # [B, T, O]

        # Base (data-driven) heteroscedastic variance: softplus keeps it
        # strictly positive; the floor prevents a degenerate near-zero
        # variance that would make the NLL loss numerically unstable.
        sigma_base = F.softplus(sigma_raw) + self.sigma_floor  # [B, T, O]

        # Apply global learnable temperature scaling if enabled
        if self.use_temperature:
            temp = torch.exp(self.sigma_temperature_raw)
            sigma_base = sigma_base * temp

        # --- DTI-driven uncertainty inflation ------------------------------ #
        inflation_factor = self.uncertainty_inflation(dti, eps=eps)  # [B, T]
        inflation_factor = inflation_factor.unsqueeze(-1)  # [B, T, 1], broadcasts over O

        sigma = sigma_base * inflation_factor  # [B, T, O]

        return mu, sigma, attn_maps

    def uncertainty_inflation(self, dti: Tensor, eps: float = 1e-4) -> Tensor:
        """Return the per-timestep DTI-driven sigma multiplier.

        This public helper is shared by ``forward`` and diagnostics so reported
        base sigma and inflation are exactly the quantities used by the model.
        """
        alpha = F.softplus(self.trust_sensitivity_raw) + 1e-3
        dti_safe = dti.clamp(0.0, 1.0) + eps
        inflation_factor = torch.pow(1.0 / dti_safe, alpha)
        return torch.clamp(inflation_factor, max=self.max_uncertainty_inflation)


# --------------------------------------------------------------------------- #
# Loss function
# --------------------------------------------------------------------------- #
class TrustCalibratedLoss(nn.Module):
    """
    Combined Gaussian Negative Log-Likelihood + pinball (quantile)
    calibration loss for training :class:`CaliPredTransformer`'s dual
    (mu, sigma) output head.

    The NLL term drives ``mu`` towards accurate point predictions and
    ``sigma`` towards a well-calibrated Gaussian predictive distribution.
    The pinball term directly supervises the *specific* 5% / 95% quantile
    bounds implied by that Gaussian (via its inverse CDF), so the dynamic
    confidence interval used downstream is calibrated as an explicit
    training objective rather than only an incidental byproduct of the
    Gaussian assumption.

    Parameters
    ----------
    lower_q : float, default 0.05
        Lower quantile level for the calibration term / reported bound.
    upper_q : float, default 0.95
        Upper quantile level for the calibration term / reported bound.
    calibration_weight : float, default 0.2
        Weight on the pinball calibration term relative to the NLL term.
    eps : float, default 1e-6
        Numerical floor on sigma before computing NLL / the Normal
        distribution's inverse CDF.
    """

    def __init__(
        self,
        lower_q: float = 0.05,
        upper_q: float = 0.95,
        calibration_weight: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not (0.0 < lower_q < upper_q < 1.0):
            raise ValueError(f"Require 0 < lower_q < upper_q < 1; got ({lower_q}, {upper_q}).")
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.calibration_weight = calibration_weight
        self.eps = eps

    @staticmethod
    def _pinball_loss(target: Tensor, quantile_pred: Tensor, tau: float) -> Tensor:
        """
        Pinball (quantile) loss: penalizes under- and over-prediction of
        a given quantile asymmetrically, such that the minimizer of its
        expectation is the true ``tau``-quantile of the target
        distribution.

        Parameters
        ----------
        target : Tensor, shape [B, T, O]
        quantile_pred : Tensor, shape [B, T, O]
        tau : float
            Target quantile level in ``(0, 1)``.

        Returns
        -------
        Tensor
            Scalar mean pinball loss.
        """
        diff = target - quantile_pred
        return torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))

    def forward(
        self, mu: Tensor, sigma: Tensor, target: Tensor
    ) -> Tuple[Tensor, dict[str, Tensor]]:
        """
        Parameters
        ----------
        mu : Tensor, shape [B, T, O]
            Predicted mean.
        sigma : Tensor, shape [B, T, O]
            Predicted (trust-inflated) standard deviation, strictly
            positive.
        target : Tensor, shape [B, T, O]
            Ground-truth target values.

        Returns
        -------
        Tuple[Tensor, dict[str, Tensor]]
            - Scalar total loss (NLL + weighted calibration term),
              differentiable w.r.t. both ``mu`` and ``sigma``.
            - Diagnostics dict with keys ``"nll"``, ``"calibration"``,
              ``"lower_bound"`` (shape [B, T, O]), and ``"upper_bound"``
              (shape [B, T, O]) for logging/inspection.

        Raises
        ------
        ValueError
            If ``mu``, ``sigma``, and ``target`` shapes don't match.
        """
        if not (mu.shape == sigma.shape == target.shape):
            raise ValueError(
                f"mu {tuple(mu.shape)}, sigma {tuple(sigma.shape)}, and target "
                f"{tuple(target.shape)} must all share the same shape."
            )

        sigma_safe = sigma.clamp_min(self.eps)  # [B, T, O]

        # Gaussian NLL: 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2).
        # Larger sigma (as forced by low DTI) directly reduces the second
        # term's penalty for a given residual, which is the mechanism by
        # which the model is "allowed" to be wrong on low-trust segments
        # without being over-penalized -- provided it also honestly widens
        # its interval, which the calibration term below enforces.
        nll_elementwise = (
            0.5 * torch.log(2.0 * math.pi * sigma_safe ** 2)
            + (target - mu) ** 2 / (2.0 * sigma_safe ** 2)
        )  # [B, T, O]
        nll_loss = nll_elementwise.mean()

        dist = torch.distributions.Normal(mu, sigma_safe)
        lower_bound = dist.icdf(torch.tensor(self.lower_q, device=mu.device, dtype=mu.dtype))
        upper_bound = dist.icdf(torch.tensor(self.upper_q, device=mu.device, dtype=mu.dtype))

        pinball_lower = self._pinball_loss(target, lower_bound, self.lower_q)
        pinball_upper = self._pinball_loss(target, upper_bound, self.upper_q)
        calibration_loss = pinball_lower + pinball_upper

        total_loss = nll_loss + self.calibration_weight * calibration_loss

        diagnostics = {
            "nll": nll_loss.detach(),
            "calibration": calibration_loss.detach(),
            "lower_bound": lower_bound.detach(),
            "upper_bound": upper_bound.detach(),
        }
        return total_loss, diagnostics


# --------------------------------------------------------------------------- #
# Mock training loop / Real-data training (CLI-driven)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CaliPredTransformer training")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to real CSV. If omitted, runs mock demo.")
    parser.add_argument("--dataset", type=str, default="metropt",
                        choices=["metropt", "ai4i2020", "tep"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)

    args = parser.parse_args()

    if args.data_path is not None:
        # ------------------------------------------------------------------ #
        # Real-data training path: delegates to pipeline.py
        # ------------------------------------------------------------------ #
        logger.info("Real data path provided. Delegating to pipeline.py...")
        print("Use `python pipeline.py` for full real-data training with "
              "DTI computation, baseline comparison, and calibration evaluation.")
        print(f"  Example: python pipeline.py --dataset {args.dataset} "
              f"--data-path {args.data_path} --epochs {args.epochs}")
    else:
        # ------------------------------------------------------------------ #
        # Original mock training demo (preserved for CI / quick tests)
        # ------------------------------------------------------------------ #
        logger.info("Running CaliPredTransformer mock training loop.")

        torch.manual_seed(0)

        B, T, K = 8, 60, 5           # batch, timesteps, input channels
        O = K                        # forecasting/reconstructing all channels
        LOW_TRUST_START, LOW_TRUST_END = 20, 35  # a contiguous low-trust segment

        # --- Synthesize a dummy batch ----------------------------------------- #
        t_axis = torch.arange(T).float()
        base_signal = torch.stack(
            [torch.sin(2 * math.pi * t_axis / (10 + 3 * k)) for k in range(K)], dim=-1
        )  # [T, K]
        X = (base_signal.unsqueeze(0) + 0.05 * torch.randn(B, T, K))  # [B, T, K]
        target = X.clone()  # simple self-reconstruction target for this mock demo

        # DTI signal: high trust everywhere except one contiguous degraded segment
        dti = torch.ones(B, T)
        dti[:, LOW_TRUST_START:LOW_TRUST_END] = 0.05

        model = CaliPredTransformer(input_dim=K, output_dim=O, d_model=32, n_heads=4, n_layers=2)
        loss_fn = TrustCalibratedLoss(lower_q=0.05, upper_q=0.95, calibration_weight=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        n_epochs = args.epochs
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            mu, sigma, _attn_maps = model(X, dti)
            loss, diagnostics = loss_fn(mu, sigma, target)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(
                    f"[epoch {epoch:02d}] total_loss={loss.item():.4f} "
                    f"nll={diagnostics['nll'].item():.4f} "
                    f"calibration={diagnostics['calibration'].item():.4f}"
                )

        # --- Verify architectural guarantee: low trust -> inflated sigma ------ #
        model.eval()
        with torch.no_grad():
            mu, sigma, attn_maps = model(X, dti)

        low_trust_sigma = sigma[:, LOW_TRUST_START:LOW_TRUST_END, :].mean().item()
        high_trust_sigma = torch.cat(
            [sigma[:, :LOW_TRUST_START, :], sigma[:, LOW_TRUST_END:, :]], dim=1
        ).mean().item()

        print(f"\nMean predicted sigma during LOW-trust segment:  {low_trust_sigma:.4f}")
        print(f"Mean predicted sigma during HIGH-trust segment: {high_trust_sigma:.4f}")
        assert low_trust_sigma > high_trust_sigma, (
            "Architectural guarantee violated: predicted uncertainty must be "
            "higher during the low-DTI segment than the high-DTI segment."
        )
        print("[OK] Low-trust segment correctly produced inflated predictive uncertainty.")

        # --- Verify calibration bounds widen accordingly ----------------------- #
        with torch.no_grad():
            dist = torch.distributions.Normal(mu, sigma.clamp_min(1e-6))
            lower = dist.icdf(torch.tensor(0.05))
            upper = dist.icdf(torch.tensor(0.95))
            interval_width = (upper - lower)

        low_trust_width = interval_width[:, LOW_TRUST_START:LOW_TRUST_END, :].mean().item()
        high_trust_width = torch.cat(
            [interval_width[:, :LOW_TRUST_START, :], interval_width[:, LOW_TRUST_END:, :]], dim=1
        ).mean().item()
        print(f"\nMean 90% interval width during LOW-trust segment:  {low_trust_width:.4f}")
        print(f"Mean 90% interval width during HIGH-trust segment: {high_trust_width:.4f}")
        assert low_trust_width > high_trust_width, "Confidence interval must widen under low trust."
        print("[OK] Confidence bounds correctly widened during the low-trust segment.")

        # --- Sanity check output shapes ---------------------------------------- #
        assert mu.shape == (B, T, O)
        assert sigma.shape == (B, T, O)
        assert len(attn_maps) == len(model.layers)
        print(f"\n[OK] Output shapes verified: mu={tuple(mu.shape)}, sigma={tuple(sigma.shape)}, "
              f"attn_maps={len(attn_maps)} layers each shaped {tuple(attn_maps[0].shape)}.")

        print("\nAll predictor tests passed.")

