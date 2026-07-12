#!/usr/bin/env python3
"""
profile_latency.py

Profiles the millisecond-level latency of each stage of the CALI-PRED pipeline
per window (length T=60, features K=15) to demonstrate real-time edge viability.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from data_loader import IndustrialDataLoader
from dqa_module import UpstreamDQAEngine
from iri_module import ImputationReliabilityEngine
from fusion_engine import TrustFusionEngine
from predictor import CaliPredTransformer

def main():
    print("=" * 70)
    print("           CALI-PRED EDGE INFERENCE LATENCY PROFILER")
    print("=" * 70)

    # 1. Setup sizes
    T = 60
    K = 15
    print(f"Configurations: Sequence length (T) = {T}, Features (K) = {K}")

    # 2. Initialize engines
    dqa_engine = UpstreamDQAEngine(freshness_tau_seconds=60.0, max_corr_mae=0.5)
    iri_engine = ImputationReliabilityEngine(n_features=K, epochs=5, holdout_frac=0.15, random_state=42)
    fusion_engine = TrustFusionEngine(clamp_inputs=True)
    corruption_loader = IndustrialDataLoader(random_state=42)

    # Create dummy correlation matrix
    baseline_corr = np.eye(K)

    # Initialize predictor model
    model = CaliPredTransformer(
        input_dim=K,
        output_dim=K,
        d_model=64,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        max_uncertainty_inflation=10.0,
        alpha_init=0.5,
        use_temperature=True
    )
    model.eval()

    # 3. Create dummy inputs
    window = np.random.randn(T, K)
    ts_window = np.arange(T, dtype=np.float64)

    # Warm-up runs
    print("\nPerforming warm-up runs...")
    for _ in range(5):
        # Missingness Injection
        x_corrupted, mask = corruption_loader.inject_missingness(window, mechanism="MAR", missing_rate=0.15, block_size=5)
        # DQA
        dqa_score = dqa_engine.compute_dqa_score(mask, ts_window, ts_window[-1] + 0.5, x_corrupted, baseline_corr)
        # IRI
        ensemble_out = iri_engine.impute_ensemble(x_corrupted, mask)
        iri_grid = iri_engine.compute_iri(ensemble_out, window, 1 - mask)
        x_imputed = ensemble_out.mean(axis=0)
        # Fusion
        dti_grid = fusion_engine.compute_dti(dqa_score, iri_grid)
        dti_per_timestep = np.mean(dti_grid, axis=1)
        # Model forward
        with torch.no_grad():
            x_tensor = torch.as_tensor(x_imputed, dtype=torch.float32).unsqueeze(0)
            dti_tensor = torch.as_tensor(dti_per_timestep, dtype=torch.float32).unsqueeze(0)
            _ = model(x_tensor, dti_tensor)

    # 4. Profile loop
    N_trials = 100
    print(f"\nRunning profiling across {N_trials} iterations...")

    t_corruption = []
    t_dqa = []
    t_iri = []
    t_fusion = []
    t_transformer = []
    t_total = []

    for _ in range(N_trials):
        t0 = time.perf_counter()

        # Step 1: Corruption Injection (simulation of packets dropping on the network)
        t_start = time.perf_counter()
        x_corrupted, mask = corruption_loader.inject_missingness(window, mechanism="MAR", missing_rate=0.15, block_size=5)
        t_corruption.append(time.perf_counter() - t_start)

        # Step 2: DQA
        t_start = time.perf_counter()
        dqa_score = dqa_engine.compute_dqa_score(mask, ts_window, ts_window[-1] + 0.5, x_corrupted, baseline_corr)
        t_dqa.append(time.perf_counter() - t_start)

        # Step 3: IRI
        t_start = time.perf_counter()
        ensemble_out = iri_engine.impute_ensemble(x_corrupted, mask)
        iri_grid = iri_engine.compute_iri(ensemble_out, window, (1 - mask).astype(np.int8))
        x_imputed = ensemble_out.mean(axis=0).astype(np.float32)
        t_iri.append(time.perf_counter() - t_start)

        # Step 4: Fusion
        t_start = time.perf_counter()
        dti_grid = fusion_engine.compute_dti(dqa_score, iri_grid)
        dti_per_timestep = np.mean(dti_grid, axis=1)
        t_fusion.append(time.perf_counter() - t_start)

        # Step 5: Transformer forward pass
        t_start = time.perf_counter()
        with torch.no_grad():
            x_tensor = torch.as_tensor(x_imputed, dtype=torch.float32).unsqueeze(0)
            dti_tensor = torch.as_tensor(dti_per_timestep, dtype=torch.float32).unsqueeze(0)
            _ = model(x_tensor, dti_tensor)
        t_transformer.append(time.perf_counter() - t_start)

        t_total.append(time.perf_counter() - t0)

    # 5. Summarize metrics
    print("\n" + "=" * 70)
    print("              LATENCY SUMMARY STATISTICS (per window)")
    print("=" * 70)
    print(f"  Corruption Injection: {np.mean(t_corruption)*1000:6.3f} ms  (std: {np.std(t_corruption)*1000:5.3f} ms)")
    print(f"  Upstream DQA:         {np.mean(t_dqa)*1000:6.3f} ms  (std: {np.std(t_dqa)*1000:5.3f} ms)")
    print(f"  Midstream IRI:         {np.mean(t_iri)*1000:6.3f} ms  (std: {np.std(t_iri)*1000:5.3f} ms)")
    print(f"  Trust Fusion:         {np.mean(t_fusion)*1000:6.3f} ms  (std: {np.std(t_fusion)*1000:5.3f} ms)")
    print(f"  CaliPred Transformer: {np.mean(t_transformer)*1000:6.3f} ms  (std: {np.std(t_transformer)*1000:5.3f} ms)")
    print("-" * 70)
    print(f"  Total pipeline:       {np.mean(t_total)*1000:6.3f} ms  (std: {np.std(t_total)*1000:5.3f} ms)")
    print(f"  Maximum Frequency:    {1.0/np.mean(t_total):6.1f} Hz")
    print("=" * 70)

if __name__ == "__main__":
    main()
