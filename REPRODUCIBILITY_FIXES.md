# CALI-PRED reproducibility fixes

This change defines a new evaluation contract. Existing prediction artifacts and paper metrics generated before these fixes are **not directly comparable** and must be regenerated.

## Forecast target

Each window is an input history of length `window_size`; the primary prediction is the next timestep immediately after that history. Training, validation, and test metrics use only the final output position (`mu[:, -1, :]`, `sigma[:, -1, :]`, and `target[:, -1, :]`). Intermediate sequence outputs are not scored because bidirectional imputation of an entire window can use later observations within that window.

## IRI validation

IRI reconstruction error is measured on a random holdout drawn from entries that were originally observed. Those entries are hidden from the imputer before reconstruction. The original known values are used only for scoring that holdout and are not fed to the imputer for the trust calculation. Original observed values remain unchanged in the final model input; only the controlled artificial missing entries are imputed.

IRI no longer uses per-window min-max normalization. Its bounded score is `(1 - reconstruction_error) / (1 + ensemble_variance)`, with reconstruction error bounded by the existing validation function. Inputs are standardized using training-split statistics.

## DQA freshness

Freshness uses the observation mask to calculate the most recent valid timestamp separately for each channel, then averages channel-level freshness. Offline experiments retain the explicit assumption `inference_time = last_timestamp + 0.5 seconds`; this is a synthetic latency assumption, not measured wall-clock latency.

## Data integrity

Missing or unreadable CSV paths now fail rather than silently triggering synthetic fallback. Timestamp rows are filtered with the same valid-sensor-row mask as the sensor matrix. Trust calculation errors are no longer converted to neutral default scores in the primary pipeline.

## Artifacts and bootstrap

`test_predictions.npz` stores `prediction_unit=one_step_forecast` and `n_features`; the bootstrap reader uses one forecast-origin block per window for these artifacts. Legacy sequence-output artifacts are still interpreted using the configured `window_size`.

## Experiment discipline

The default primary setup retains the 15-channel schema and calibration weight 0.2. Seven-channel experiments and alternate calibration weights must be run as separately identified ablations. Runtime source patching is disabled. Post-hoc validation sigma scaling is opt-in and should be reported separately from unscaled model results.

## Validation scope

The automated tests cover target alignment, timestamp preservation, stale-channel freshness, IRI reconstruction-penalty sensitivity, and the one-step prediction artifact contract. A full MetroPT run is still required before reporting corrected research metrics.


## Runtime optimization: reusable IRI ensemble

The pipeline fits the SAITS-style and BRITS-style imputers once using only
training-split windows and self-supervised random holdouts. Fitted models are
reused for batched inference while DQA remains window-specific. This removes
per-window neural-network initialization and optimization from preprocessing.
It is a methodological change from the legacy per-window fitting protocol:
training-window IRI is estimated by models fitted on the training split, while
validation/test windows are processed by inference only. Regenerate metrics and
report results under this updated protocol.

Precomputation logs report completed windows, percentage, elapsed time,
throughput, and ETA. Reusable IRI fitting reports batch progress and elapsed
time. Full MetroPT runs remain necessary before reporting research results.
