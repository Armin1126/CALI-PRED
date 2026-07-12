"""
fusion_engine.py

Industrial Anomaly Prediction Framework — Trust Fusion Module
==============================================================================

This module establishes the **Continuous Trust Signal**: a single, unified
per-timestep (and per-channel, where applicable) measure of how much the
entire pipeline's output should be trusted, computed by fusing:

    - **DQA** (Upstream Data Quality Assessment, ``dqa_module.py``) — is the
      *raw sensor stream* healthy (complete, fresh, physically consistent)?
    - **IRI** (Midstream Imputation Reliability Indicator, ``iri_module.py``)
      — is the *imputed reconstruction* trustworthy (low ensemble
      disagreement, validated against known ground truth)?

Mathematical formulation — the "weak-link" philosophy
------------------------------------------------------
    DTI = DQA * IRI

The Data Trust Index (DTI) is computed via **element-wise multiplication**,
not averaging or any other linear combination, and this is a deliberate
architectural choice, not an implementation detail.

Consider the alternative — a weighted average, ``DTI = 0.5*DQA + 0.5*IRI``.
Under that formulation, a catastrophically unhealthy upstream stream
(``DQA = 0.05``, e.g. a sensor that has been offline for ten minutes) paired
with a confident-looking imputation (``IRI = 0.95``) would still average out
to ``DTI = 0.50`` — a "medium trust" signal for data that is, in truth,
almost entirely fabricated. An additive/averaging formulation lets a strong
score in one dimension *compensate* for a catastrophic failure in the other.
That compensatory behavior is exactly wrong for a trust signal feeding an
industrial anomaly-detection system: a confidently-imputed value built on
top of a dead sensor is not "medium-trust" data, it is data that should not
be acted on.

Multiplication enforces the opposite property — a **weak-link** (or
"weakest-link") property, analogous to a chain being only as strong as its
weakest link, or a series circuit where any open switch cuts all current
regardless of how closed the other switches are:

    - If **either** DQA **or** IRI is low, DTI is driven towards zero,
      regardless of how high the other term is.
    - DTI can only be high when **both** the upstream stream **and** the
      midstream reconstruction are simultaneously trustworthy.
    - Because both terms live in ``[0, 1]``, their product is bounded by
      the smaller of the two: ``DTI <= min(DQA, IRI)``. Trust can never
      exceed the weaker of its two contributing signals — it can only be
      dragged down further by disagreement between them.

This is the correct behavior for a safety/reliability gate: it should be
*hard* to earn high trust (both signals must independently check out) and
*easy* to lose it (a single failure anywhere in the pipeline is sufficient).

Python: 3.13+
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("TrustFusionEngine")

# DQA and IRI signals may arrive as plain Python floats (a single
# whole-window summary score) or as NumPy arrays (a per-timestep and/or
# per-channel signal). compute_dti accepts either and relies on NumPy
# broadcasting to reconcile them.
TrustSignal = Union[float, np.ndarray]


class TrustFusionEngine:
    """
    Fuses upstream (DQA) and midstream (IRI) trust signals into a single
    Continuous Trust Signal (the Data Trust Index, DTI) via strict
    element-wise multiplication.

    This class is intentionally minimal — trust fusion is a critical,
    frequently-invoked step (potentially once per inference tick in a
    real-time pipeline) and should have as small and predictable a
    surface area as possible. All the interesting judgment calls (how to
    weigh completeness vs. freshness, how to size an imputation ensemble)
    belong further upstream/midstream, in ``dqa_module.py`` and
    ``iri_module.py`` respectively; this engine's only job is the fusion
    arithmetic and the safety checks around it.

    Parameters
    ----------
    clamp_inputs : bool, default True
        If ``True``, ``dqa_signal`` and ``iri_signal`` are defensively
        clipped to ``[0, 1]`` before fusion — protecting against minor
        floating-point overshoot from upstream computations (e.g., a DQA
        or IRI value of ``1.0000000002``) without raising an error. If
        ``False``, out-of-bounds inputs raise a ``ValueError`` instead of
        being silently corrected — appropriate for a strict validation
        context where an out-of-range trust component indicates a bug
        upstream that should be surfaced, not masked.
    """

    def __init__(self, clamp_inputs: bool = True) -> None:
        self.clamp_inputs = clamp_inputs

    def compute_dti(self, dqa_signal: TrustSignal, iri_signal: TrustSignal) -> np.ndarray:
        """
        Compute the Data Trust Index (DTI) via DTI = DQA * IRI.

        Parameters
        ----------
        dqa_signal : float or np.ndarray
            Upstream Data Quality Assessment score(s), in ``[0, 1]``.
            May be a single scalar (one score for the whole window) or an
            array (e.g., a per-timestep DQA signal of shape ``(T,)``, or
            a per-timestep-per-channel signal of shape ``(T, K)``).
        iri_signal : float or np.ndarray
            Midstream Imputation Reliability Indicator score(s), in
            ``[0, 1]``. Commonly shape ``(T, K)`` (see
            ``iri_module.ImputationReliabilityEngine.compute_iri``), but
            any shape broadcastable against ``dqa_signal`` is accepted.

        Returns
        -------
        np.ndarray
            ``DTI_signal``: the element-wise product of the two inputs
            after NumPy broadcasting, clipped to ``[0.0, 1.0]``. The
            output shape is the broadcast shape of the two inputs (e.g.,
            a scalar DQA fused with a ``(T, K)`` IRI signal yields a
            ``(T, K)`` DTI signal, since the single DQA score is applied
            uniformly across the whole window).

        Raises
        ------
        ValueError
            - If either input contains NaN or Inf (a non-finite trust
              component cannot be meaningfully fused and should never be
              silently treated as "trustworthy" or "untrustworthy").
            - If ``clamp_inputs=False`` and either input falls outside
              ``[0, 1]``.
            - If the two input shapes are not broadcast-compatible under
              standard NumPy broadcasting rules (e.g., shapes ``(10, 3)``
              and ``(7, 3)`` cannot be reconciled).

        Notes on the weak-link property
        --------------------------------
        Because both inputs are bounded in ``[0, 1]`` and combined by
        multiplication (never addition), the output satisfies
        ``DTI <= min(DQA, IRI)`` at every position — the fused trust score
        can never exceed the weaker of its two contributing signals. This
        is verified explicitly in this module's ``__main__`` test block.
        """
        dqa_arr = np.asarray(dqa_signal, dtype=np.float64)
        iri_arr = np.asarray(iri_signal, dtype=np.float64)

        self._validate_finite(dqa_arr, "dqa_signal")
        self._validate_finite(iri_arr, "iri_signal")

        if self.clamp_inputs:
            dqa_arr = np.clip(dqa_arr, 0.0, 1.0)
            iri_arr = np.clip(iri_arr, 0.0, 1.0)
        else:
            self._validate_unit_bounds(dqa_arr, "dqa_signal")
            self._validate_unit_bounds(iri_arr, "iri_signal")

        try:
            dti_signal = dqa_arr * iri_arr
        except ValueError as exc:
            raise ValueError(
                f"dqa_signal (shape {dqa_arr.shape}) and iri_signal "
                f"(shape {iri_arr.shape}) are not broadcast-compatible: {exc}"
            ) from exc

        # Defensive final clamp: even with validated [0, 1] inputs, this
        # guards against floating-point edge cases (e.g., 1.0 * 1.0000000001
        # from upstream rounding) so the contract "DTI in [0, 1]" is never
        # violated regardless of input provenance.
        dti_signal = np.clip(dti_signal, 0.0, 1.0)

        logger.info(
            "Computed DTI: shape=%s, mean=%.4f, min=%.4f, max=%.4f "
            "(dqa mean=%.4f, iri mean=%.4f).",
            dti_signal.shape,
            float(np.mean(dti_signal)), float(np.min(dti_signal)), float(np.max(dti_signal)),
            float(np.mean(dqa_arr)), float(np.mean(iri_arr)),
        )
        return dti_signal

    # ------------------------------------------------------------------ #
    # Safety checks
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_finite(arr: np.ndarray, name: str) -> None:
        """Reject NaN/Inf trust components rather than silently propagating them."""
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            raise ValueError(
                f"{name} contains {n_bad} non-finite value(s) (NaN/Inf). "
                "A trust signal cannot be fused when undefined; upstream "
                "callers should resolve or explicitly mask these positions "
                "before calling compute_dti."
            )

    @staticmethod
    def _validate_unit_bounds(arr: np.ndarray, name: str) -> None:
        """Reject out-of-[0,1] trust components when clamp_inputs=False."""
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError(
                f"{name} must be within [0, 1] when clamp_inputs=False; "
                f"got range [{float(np.min(arr)):.6f}, {float(np.max(arr)):.6f}]. "
                "Either fix the upstream signal or construct this engine "
                "with clamp_inputs=True to tolerate minor overshoot."
            )


# --------------------------------------------------------------------------- #
# Demonstration / test block
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("Running TrustFusionEngine demonstration.")

    engine = TrustFusionEngine(clamp_inputs=True)

    # --- Core requirement: DQA=0.1, IRI=0.9 must register LOW trust ---------- #
    dqa_low, iri_high = 0.1, 0.9
    dti_weak_link = engine.compute_dti(dqa_low, iri_high)
    print(f"[Weak-link check] DQA={dqa_low}, IRI={iri_high} -> DTI={float(dti_weak_link):.4f}")
    expected_product = round(dqa_low * iri_high, 10)
    assert round(float(dti_weak_link), 10) == expected_product, \
        "DTI must equal the exact product of DQA and IRI."
    assert float(dti_weak_link) < 0.15, (
        "A low DQA must drag DTI down to a low-trust regime even when IRI is "
        "high — the whole point of the multiplicative weak-link design."
    )
    print("  [OK] Low upstream quality correctly collapsed overall trust, "
          "despite high midstream reliability.")

    # --- Symmetric check: the reverse (high DQA, low IRI) must also collapse -- #
    dqa_high, iri_low = 0.95, 0.08
    dti_weak_link_2 = engine.compute_dti(dqa_high, iri_low)
    print(f"[Weak-link check] DQA={dqa_high}, IRI={iri_low} -> DTI={float(dti_weak_link_2):.4f}")
    assert float(dti_weak_link_2) < 0.15, "A low IRI must also drag DTI down, symmetrically."
    print("  [OK] Low midstream reliability correctly collapsed overall trust, "
          "despite high upstream quality.")

    # --- Compensation must NOT occur: verify DTI <= min(DQA, IRI) always ------ #
    rng = np.random.default_rng(0)
    dqa_random = rng.uniform(0.0, 1.0, size=1000)
    iri_random = rng.uniform(0.0, 1.0, size=1000)
    dti_random = engine.compute_dti(dqa_random, iri_random)
    assert np.all(dti_random <= np.minimum(dqa_random, iri_random) + 1e-9), (
        "DTI must never exceed the smaller of its two contributing signals "
        "(no compensation between upstream and midstream trust)."
    )
    print("[OK] Verified DTI <= min(DQA, IRI) across 1000 random samples "
          "(no compensatory behavior).")

    # --- Broadcasting: scalar DQA applied uniformly across a per-timestep, ---- #
    # --- per-channel IRI signal ------------------------------------------------ #
    T_DEMO, K_DEMO = 50, 4
    dqa_scalar_window_score = 0.7  # one DQA score for the whole window
    iri_timeseries = rng.uniform(0.5, 1.0, size=(T_DEMO, K_DEMO))
    dti_broadcast = engine.compute_dti(dqa_scalar_window_score, iri_timeseries)
    assert dti_broadcast.shape == (T_DEMO, K_DEMO), "Broadcasting must preserve the IRI signal's shape."
    print(f"[OK] Broadcast scalar DQA={dqa_scalar_window_score} across IRI shape "
          f"{iri_timeseries.shape} -> DTI shape {dti_broadcast.shape}, "
          f"mean={dti_broadcast.mean():.4f}.")

    # --- Bounds: output must always land in [0, 1] ----------------------------- #
    assert np.all((dti_random >= 0.0) & (dti_random <= 1.0))
    assert np.all((dti_broadcast >= 0.0) & (dti_broadcast <= 1.0))
    print("[OK] All DTI outputs bounded within [0.0, 1.0].")

    # --- Safety: non-finite input must raise ------------------------------------ #
    try:
        engine.compute_dti(np.array([0.5, np.nan]), np.array([0.5, 0.5]))
        raise AssertionError("Expected ValueError for NaN input, but none was raised.")
    except ValueError as e:
        print(f"[OK] Correctly rejected non-finite input: {e}")

    # --- Safety: incompatible shapes must raise --------------------------------- #
    try:
        engine.compute_dti(np.ones((10, 3)), np.ones((7, 3)))
        raise AssertionError("Expected ValueError for shape mismatch, but none was raised.")
    except ValueError as e:
        print(f"[OK] Correctly rejected incompatible shapes: {e}")

    # --- Safety: strict mode rejects out-of-bounds input without clamping ------- #
    strict_engine = TrustFusionEngine(clamp_inputs=False)
    try:
        strict_engine.compute_dti(1.2, 0.5)
        raise AssertionError("Expected ValueError for out-of-bounds input under strict mode.")
    except ValueError as e:
        print(f"[OK] Strict mode correctly rejected out-of-bounds input: {e}")

    # Lenient mode should tolerate the same overshoot via clamping.
    lenient_result = engine.compute_dti(1.2, 0.5)
    assert float(lenient_result) == 0.5, "clamp_inputs=True should clip DQA=1.2 down to 1.0 before fusion."
    print(f"[OK] Lenient mode correctly clamped DQA=1.2 -> 1.0 before fusion, DTI={float(lenient_result):.4f}.")

    print("\nAll fusion engine tests passed.")
