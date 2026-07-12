"""
dqa_module.py

Industrial Anomaly Prediction Framework — Upstream Data Quality Assessment
==============================================================================

This module implements the **Upstream Data Quality Assessment (DQA) Score**:
a single, interpretable health signal computed on the *raw* sensor stream
**before** any imputation or modeling takes place.

Why upstream, before imputation?
---------------------------------
The IRI module (see ``iri_module.py``) scores *how much to trust an imputed
value* — it presupposes that imputation has already happened and is a
midstream signal. DQA answers a logically prior question: *should we even
be running imputation and downstream inference on this window of data at
all?* A stream that has gone stale, lost most of its readings, or started
producing physically implausible cross-sensor relationships (e.g., a
pressure/flow pair that should move together suddenly decorrelating) is a
sign of a sensor fault, a network partition, or a mis-wired PLC channel —
problems that no imputation model can meaningfully "fix," only paper over.
Surfacing this *before* imputation lets an operator (or an automated
circuit-breaker) decide whether to trust the pipeline's output at all.

Formulation
-----------
    DQA = w1 * C_comp + w2 * C_fresh + w3 * C_cons

where each component is normalized to ``[0, 1]`` (1 = perfectly healthy):

    - **C_comp (Completeness)**: fraction of expected readings actually
      observed within a sliding window (derived from the binary
      observation mask).
    - **C_fresh (Freshness)**: an exponentially-decaying penalty on the
      latency between the most recent sensor timestamp and the current
      inference time — a stream that hasn't reported in a while is
      operationally stale even if historically complete.
    - **C_cons (Consistency)**: agreement between the *current* rolling
      cross-sensor correlation structure and a known-good
      ``baseline_corr_matrix`` representing expected physical
      relationships (e.g., temperature and pressure in a closed vessel
      should stay positively correlated under normal operation).

Weights default to equal thirds (summing to 1.0) but are fully
configurable, letting an operator emphasize, e.g., freshness over
completeness for a latency-sensitive control loop.

Python: 3.13+
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("UpstreamDQAEngine")


@dataclass(frozen=True)
class DQABreakdown:
    """Per-component breakdown of a single DQA score, useful for logging,
    dashboards, or root-causing why a stream's DQA dropped."""

    completeness: float
    freshness: float
    consistency: float
    weights: Tuple[float, float, float]
    dqa_score: float


class UpstreamDQAEngine:
    """
    Computes the Upstream Data Quality Assessment (DQA) score for a
    multivariate sensor stream, evaluated on raw (pre-imputation) data.

    Parameters
    ----------
    weights : Optional[Tuple[float, float, float]], default None
        ``(w1, w2, w3)`` weights for (completeness, freshness,
        consistency) respectively. Must sum to 1.0 (within a small
        numerical tolerance). Defaults to equal thirds
        ``(1/3, 1/3, 1/3)`` if not provided.
    freshness_tau_seconds : float, default 60.0
        Time constant controlling how quickly the freshness score decays
        with latency. Freshness = ``exp(-latency / tau)``, so latency
        equal to ``tau`` seconds yields a freshness score of ``~0.368``,
        and latency of ``3*tau`` yields ``~0.050``. Choose ``tau``
        relative to the sensor's expected reporting cadence (e.g., a
        1 Hz sensor might use ``tau`` on the order of a few seconds; a
        slow batch-process sensor might use minutes).
    max_corr_mae : float, default 0.5
        The mean absolute correlation deviation (between current and
        baseline correlation matrices) at which consistency bottoms out
        at 0.0. Correlation differences range in ``[0, 2]`` in the worst
        case (e.g., baseline +1.0 vs. observed -1.0), but a
        physically-plausible fault condition rarely pushes MAE anywhere
        near that extreme, so a smaller practical ceiling (default 0.5)
        keeps the score sensitive in the realistic operating range.
    default_window : Optional[int], default None
        Default sliding window size (in timesteps, counted from the end
        of the array) used by :meth:`completeness` and :meth:`consistency`
        when no explicit ``window`` is passed to those calls. ``None``
        means "use the entire provided array" (no windowing).

    Raises
    ------
    ValueError
        If ``weights`` is provided but doesn't sum to 1.0 within
        tolerance, or contains negative values.
    """

    _WEIGHT_SUM_TOLERANCE = 1e-6

    def __init__(
        self,
        weights: Optional[Tuple[float, float, float]] = None,
        freshness_tau_seconds: float = 60.0,
        max_corr_mae: float = 0.5,
        default_window: Optional[int] = None,
    ) -> None:
        self.weights = self._validate_weights(weights or (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))

        if freshness_tau_seconds <= 0:
            raise ValueError(f"freshness_tau_seconds must be positive; got {freshness_tau_seconds}.")
        if max_corr_mae <= 0:
            raise ValueError(f"max_corr_mae must be positive; got {max_corr_mae}.")

        self.freshness_tau_seconds = freshness_tau_seconds
        self.max_corr_mae = max_corr_mae
        self.default_window = default_window

    @staticmethod
    def _validate_weights(weights: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if len(weights) != 3:
            raise ValueError(f"weights must have exactly 3 elements (w1, w2, w3); got {len(weights)}.")
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must be non-negative; got {weights}.")
        total = sum(weights)
        if abs(total - 1.0) > UpstreamDQAEngine._WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"weights must sum to 1.0 (got sum={total:.6f}) for weights={weights}.")
        return (float(weights[0]), float(weights[1]), float(weights[2]))

    # ------------------------------------------------------------------ #
    # C_comp: Completeness
    # ------------------------------------------------------------------ #
    def completeness(self, mask: np.ndarray, window: Optional[int] = None) -> float:
        """
        Compute C_comp: the ratio of observed readings to expected
        readings within a sliding time window.

        Parameters
        ----------
        mask : np.ndarray, shape (T, K)
            Binary observation mask, 1 = observed, 0 = missing.
        window : Optional[int], default None
            Number of most-recent timesteps to evaluate (a trailing
            sliding window). If ``None``, falls back to
            ``self.default_window``; if that is also ``None``, the
            entire mask is used.

        Returns
        -------
        float
            Completeness ratio in ``[0, 1]``, where 1.0 means every
            expected reading in the window was observed.

        Raises
        ------
        ValueError
            If ``mask`` is empty, or the effective window resolves to
            zero expected readings.
        """
        if mask.size == 0:
            raise ValueError("mask must be non-empty.")

        effective_window = window if window is not None else self.default_window
        windowed_mask = mask[-effective_window:] if effective_window is not None else mask

        total_expected = windowed_mask.size
        if total_expected == 0:
            raise ValueError("Resolved window contains zero expected readings.")

        observed = float(np.sum(windowed_mask))
        c_comp = observed / total_expected
        return float(np.clip(c_comp, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # C_fresh: Freshness
    # ------------------------------------------------------------------ #
    def freshness(self, timestamps: np.ndarray, inference_time: float) -> float:
        """
        Compute C_fresh: an exponentially-decaying normalized penalty on
        operational latency — the delay between the most recent sensor
        timestamp and the current inference execution time.

        Parameters
        ----------
        timestamps : np.ndarray, shape (T,)
            Unix (or otherwise monotonic, seconds-scale) timestamps for
            each observed reading in the current window. May contain
            ``np.nan`` for timesteps with no reading at all; these are
            ignored when finding the latest timestamp.
        inference_time : float
            The current wall-clock time (same units/epoch as
            ``timestamps``) at which inference is being executed.

        Returns
        -------
        float
            Freshness score in ``[0, 1]``: ``exp(-latency / tau)``,
            clamped defensively to ``[0, 1]``. A latency of 0 yields a
            score of 1.0 (perfectly fresh); latency growing large drives
            the score towards 0.0 (stale).

        Raises
        ------
        ValueError
            If ``timestamps`` is empty or contains only NaNs.
        """
        if timestamps.size == 0:
            raise ValueError("timestamps must be non-empty.")

        valid_timestamps = timestamps[~np.isnan(timestamps)] if np.issubdtype(
            timestamps.dtype, np.floating
        ) else timestamps

        if valid_timestamps.size == 0:
            raise ValueError("timestamps contains no valid (non-NaN) entries.")

        latest_timestamp = float(np.max(valid_timestamps))
        # Clamp negative latency (e.g., minor clock skew, or a timestamp
        # slightly ahead of inference_time) to zero rather than letting it
        # produce a freshness score > 1.
        latency = max(0.0, inference_time - latest_timestamp)

        c_fresh = float(np.exp(-latency / self.freshness_tau_seconds))
        return float(np.clip(c_fresh, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # C_cons: Consistency
    # ------------------------------------------------------------------ #
    def consistency(self, X_corrupted: np.ndarray, baseline_corr_matrix: np.ndarray) -> float:
        """
        Compute C_cons: agreement between the current rolling cross-sensor
        correlation matrix and a known-good baseline correlation matrix
        representing expected physical relationships between channels.

        Missing values (``np.nan``) are handled gracefully via
        pairwise-complete observations: for each pair of channels, the
        correlation is computed using only the timesteps where *both*
        channels have an observed (non-NaN) value, rather than dropping
        any timestep with *any* missing channel (which would needlessly
        discard usable data in a wide multivariate stream).

        Parameters
        ----------
        X_corrupted : np.ndarray, shape (T, K)
            Raw (pre-imputation) sensor readings, with ``np.nan`` at
            missing positions.
        baseline_corr_matrix : np.ndarray, shape (K, K)
            Known-good reference correlation matrix (e.g., estimated
            from a historical healthy-operation period, or derived from
            first-principles physical relationships between sensors).

        Returns
        -------
        float
            Consistency score in ``[0, 1]``, where 1.0 means the current
            correlation structure exactly matches baseline, and 0.0 means
            the mean absolute correlation deviation has reached or
            exceeded ``self.max_corr_mae``.

        Raises
        ------
        ValueError
            If ``baseline_corr_matrix`` is not square, or its dimension
            doesn't match the number of channels in ``X_corrupted``.
        """
        T, K = X_corrupted.shape
        if baseline_corr_matrix.shape != (K, K):
            raise ValueError(
                f"baseline_corr_matrix must have shape ({K}, {K}) to match "
                f"X_corrupted's {K} channels; got {baseline_corr_matrix.shape}."
            )

        # pandas' .corr() uses pairwise-complete observations by default:
        # each pairwise correlation is computed from the subset of rows
        # where both columns are non-NaN, rather than a single global
        # complete-case filter across all K channels.
        current_corr = pd.DataFrame(X_corrupted).corr(method="pearson").to_numpy()

        # A channel with zero variance in the available window (e.g., a
        # stuck sensor) produces NaN correlations against every other
        # channel. Treat these as maximal disagreement (0.0 correlation)
        # rather than silently ignoring them, since a stuck sensor is
        # itself an inconsistency signal worth penalizing.
        current_corr = np.nan_to_num(current_corr, nan=0.0)

        off_diagonal = ~np.eye(K, dtype=bool)
        abs_diff = np.abs(current_corr - baseline_corr_matrix)
        mae = float(np.mean(abs_diff[off_diagonal])) if K > 1 else 0.0

        c_cons = 1.0 - (mae / self.max_corr_mae)
        return float(np.clip(c_cons, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Composite score
    # ------------------------------------------------------------------ #
    def compute_dqa_score(
        self,
        mask: np.ndarray,
        timestamps: np.ndarray,
        inference_time: float,
        X_corrupted: np.ndarray,
        baseline_corr_matrix: np.ndarray,
        weights: Optional[Tuple[float, float, float]] = None,
        window: Optional[int] = None,
        return_breakdown: bool = False,
    ) -> float | Tuple[float, DQABreakdown]:
        """
        Compute the unified Upstream DQA score:

            DQA = w1 * C_comp + w2 * C_fresh + w3 * C_cons

        clamped to ``[0.0, 1.0]``.

        Parameters
        ----------
        mask : np.ndarray, shape (T, K)
            Binary observation mask (see :meth:`completeness`).
        timestamps : np.ndarray, shape (T,)
            Per-timestep timestamps (see :meth:`freshness`).
        inference_time : float
            Current wall-clock inference time (see :meth:`freshness`).
        X_corrupted : np.ndarray, shape (T, K)
            Raw sensor readings with NaNs at missing positions (see
            :meth:`consistency`).
        baseline_corr_matrix : np.ndarray, shape (K, K)
            Reference correlation matrix (see :meth:`consistency`).
        weights : Optional[Tuple[float, float, float]], default None
            Override for this call only; must sum to 1.0. If ``None``,
            uses ``self.weights`` (set at construction time).
        window : Optional[int], default None
            Sliding window override passed through to :meth:`completeness`.
        return_breakdown : bool, default False
            If ``True``, also return a :class:`DQABreakdown` with the
            individual component scores for logging/diagnostics.

        Returns
        -------
        float
            The composite DQA score in ``[0.0, 1.0]``, OR
        Tuple[float, DQABreakdown]
            if ``return_breakdown=True``.

        Raises
        ------
        ValueError
            Propagated from component methods, or if ``weights`` (when
            provided) doesn't sum to 1.0.
        """
        w1, w2, w3 = self._validate_weights(weights) if weights is not None else self.weights

        c_comp = self.completeness(mask, window=window)
        c_fresh = self.freshness(timestamps, inference_time)
        c_cons = self.consistency(X_corrupted, baseline_corr_matrix)

        raw_dqa = w1 * c_comp + w2 * c_fresh + w3 * c_cons
        dqa_score = float(np.clip(raw_dqa, 0.0, 1.0))

        logger.info(
            "DQA=%.4f (C_comp=%.4f, C_fresh=%.4f, C_cons=%.4f, weights=(%.3f, %.3f, %.3f)).",
            dqa_score, c_comp, c_fresh, c_cons, w1, w2, w3,
        )

        if return_breakdown:
            breakdown = DQABreakdown(
                completeness=c_comp,
                freshness=c_fresh,
                consistency=c_cons,
                weights=(w1, w2, w3),
                dqa_score=dqa_score,
            )
            return dqa_score, breakdown

        return dqa_score

    # ------------------------------------------------------------------ #
    # Batch DQA computation
    # ------------------------------------------------------------------ #
    def compute_dqa_batch(
        self,
        masks: np.ndarray,
        timestamps_batch: np.ndarray,
        inference_times: np.ndarray,
        X_corrupted_batch: np.ndarray,
        baseline_corr_matrix: np.ndarray,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Vectorized DQA computation over a batch of windows.

        Parameters
        ----------
        masks : np.ndarray, shape (B, T, K)
            Binary observation masks for each window in the batch.
        timestamps_batch : np.ndarray, shape (B, T)
            Per-timestep timestamps for each window.
        inference_times : np.ndarray, shape (B,)
            Inference time for each window.
        X_corrupted_batch : np.ndarray, shape (B, T, K)
            Raw sensor readings with NaNs at missing positions.
        baseline_corr_matrix : np.ndarray, shape (K, K)
            Reference correlation matrix.
        window : Optional[int], default None
            Sliding window override for completeness.

        Returns
        -------
        np.ndarray, shape (B,)
            Per-window DQA scores in [0, 1].
        """
        B = masks.shape[0]
        dqa_scores = np.empty(B, dtype=np.float64)

        for i in range(B):
            dqa_scores[i] = self.compute_dqa_score(
                mask=masks[i],
                timestamps=timestamps_batch[i],
                inference_time=float(inference_times[i]),
                X_corrupted=X_corrupted_batch[i],
                baseline_corr_matrix=baseline_corr_matrix,
                window=window,
            )

        return dqa_scores


# --------------------------------------------------------------------------- #
# Demonstration / simulation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("Running UpstreamDQAEngine simulation.")

    rng = np.random.default_rng(5)
    T_DEMO, K_DEMO = 300, 4

    # --- Baseline correlation structure: two correlated pairs ---------------- #
    # Channels 0 & 1 move together (e.g., temperature & pressure), channels
    # 2 & 3 move together (e.g., current & vibration), the two pairs are
    # roughly independent of each other.
    baseline_corr_matrix = np.array([
        [1.00, 0.85, 0.05, 0.05],
        [0.85, 1.00, 0.05, 0.05],
        [0.05, 0.05, 1.00, 0.80],
        [0.05, 0.05, 0.80, 1.00],
    ])

    # --- Generate a healthy synthetic stream honoring that structure --------- #
    latent_pair_a = rng.normal(size=T_DEMO)
    latent_pair_b = rng.normal(size=T_DEMO)
    X_healthy = np.stack([
        latent_pair_a + 0.3 * rng.normal(size=T_DEMO),
        latent_pair_a + 0.3 * rng.normal(size=T_DEMO),
        latent_pair_b + 0.4 * rng.normal(size=T_DEMO),
        latent_pair_b + 0.4 * rng.normal(size=T_DEMO),
    ], axis=1)

    mask_healthy = np.ones((T_DEMO, K_DEMO), dtype=np.int8)
    timestamps_healthy = np.arange(T_DEMO, dtype=np.float64) * 1.0  # 1-second cadence
    inference_time_fresh = float(timestamps_healthy[-1] + 0.5)  # 0.5s after last reading

    engine = UpstreamDQAEngine(freshness_tau_seconds=10.0, max_corr_mae=0.5)

    # --- Scenario 1: fully healthy baseline ----------------------------------- #
    dqa_baseline, breakdown_baseline = engine.compute_dqa_score(
        mask_healthy, timestamps_healthy, inference_time_fresh,
        X_healthy, baseline_corr_matrix, return_breakdown=True,
    )
    print(f"[Scenario 1: Healthy]        DQA={dqa_baseline:.4f}  {breakdown_baseline}")

    # --- Scenario 2: increased latency (stream has gone stale) --------------- #
    inference_time_stale = float(timestamps_healthy[-1] + 120.0)  # 2 minutes late
    dqa_stale, breakdown_stale = engine.compute_dqa_score(
        mask_healthy, timestamps_healthy, inference_time_stale,
        X_healthy, baseline_corr_matrix, return_breakdown=True,
    )
    print(f"[Scenario 2: Stale/latency]  DQA={dqa_stale:.4f}  {breakdown_stale}")
    assert dqa_stale < dqa_baseline, "Increased latency must reduce DQA."
    print("  [OK] Increased latency correctly reduced the DQA score.")

    # --- Scenario 3: mismatched correlation (e.g., a sensor fault decorrelates it) --- #
    X_faulty = X_healthy.copy()
    # Channel 1 stops tracking channel 0 and instead reports near-independent noise,
    # simulating a decoupled/faulty sensor.
    X_faulty[:, 1] = rng.normal(size=T_DEMO)

    dqa_faulty, breakdown_faulty = engine.compute_dqa_score(
        mask_healthy, timestamps_healthy, inference_time_fresh,
        X_faulty, baseline_corr_matrix, return_breakdown=True,
    )
    print(f"[Scenario 3: Decorrelated]   DQA={dqa_faulty:.4f}  {breakdown_faulty}")
    assert dqa_faulty < dqa_baseline, "Mismatched correlation must reduce DQA."
    print("  [OK] Mismatched cross-sensor correlation correctly reduced the DQA score.")

    # --- Scenario 4: reduced completeness (dropped packets) ------------------- #
    mask_incomplete = mask_healthy.copy()
    mask_incomplete[100:200, :] = 0  # a large dropout block across all channels
    X_incomplete = X_healthy.copy()
    X_incomplete[mask_incomplete == 0] = np.nan

    dqa_incomplete, breakdown_incomplete = engine.compute_dqa_score(
        mask_incomplete, timestamps_healthy, inference_time_fresh,
        X_incomplete, baseline_corr_matrix, return_breakdown=True,
    )
    print(f"[Scenario 4: Incomplete]     DQA={dqa_incomplete:.4f}  {breakdown_incomplete}")
    assert dqa_incomplete < dqa_baseline, "Reduced completeness must reduce DQA."
    print("  [OK] Reduced completeness correctly reduced the DQA score.")

    # --- Scenario 5: custom weights emphasizing freshness --------------------- #
    dqa_custom = engine.compute_dqa_score(
        mask_healthy, timestamps_healthy, inference_time_stale,
        X_healthy, baseline_corr_matrix, weights=(0.1, 0.8, 0.1),
    )
    print(f"[Scenario 5: Custom weights, stale + freshness-weighted] DQA={dqa_custom:.4f}")
    assert dqa_custom < dqa_baseline, "Freshness-weighted stale scenario must still score below healthy baseline."

    print("\nAll simulation checks passed.")
