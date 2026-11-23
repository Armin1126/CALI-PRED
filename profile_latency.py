#!/usr/bin/env python3
"""
profile_latency.py

Profiles the millisecond-level latency of each stage of the CALI-PRED pipeline
per window (length T=60, features K=15) to demonstrate real-time edge viability.
"""

import argparse
import platform
import time
import numpy as np
import torch

from data_loader import IndustrialDataLoader
from dqa_module import UpstreamDQAEngine
from iri_module import ImputationReliabilityEngine
from fusion_engine import TrustFusionEngine
from predictor import CaliPredTransformer

def main():
    parser = argparse.ArgumentParser(description="Profile synthetic per-window CALI-PRED latency.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 1:
        raise ValueError("warmup must be >= 0 and repetitions must be >= 1.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    print("=" * 70)
    print("        CALI-PRED SYNTHETIC LATENCY PROFILER")
    print("=" * 70)

    # 1. Setup sizes
    T = 60
    K = 15
    print(f"Configurations: Sequence length (T) = {T}, Features (K) = {K}")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine()
    print(f"Device: {device} ({device_name}); batch size: 1; warm-up iterations: {args.warmup}; timed repetitions: {args.repetitions}")
    print(f"Python: {platform.python_version()}; PyTorch: {torch.__version__}; NumPy: {np.__version__}; data loading/preprocessing excluded")

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
    model.to(device)
    model.eval()

    # 3. Create dummy inputs
    window = np.random.randn(T, K)
    ts_window = np.arange(T, dtype=np.float64)

    # Warm-up runs
    print("\nPerforming warm-up runs...")
    for _ in range(args.warmup):
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
            x_tensor = torch.as_tensor(x_imputed, dtype=torch.float32).unsqueeze(0).to(device)
            dti_tensor = torch.as_tensor(dti_per_timestep, dtype=torch.float32).unsqueeze(0).to(device)
            _ = model(x_tensor, dti_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # 4. Profile loop
    N_trials = args.repetitions
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
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
            x_tensor = torch.as_tensor(x_imputed, dtype=torch.float32).unsqueeze(0).to(device)
            dti_tensor = torch.as_tensor(dti_per_timestep, dtype=torch.float32).unsqueeze(0).to(device)
            _ = model(x_tensor, dti_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_transformer.append(time.perf_counter() - t_start)

        t_total.append(time.perf_counter() - t0)

    # 5. Summarize metrics
    print("\n" + "=" * 70)
    print("              LATENCY SUMMARY STATISTICS (per window)")
    print("=" * 70)
    def report(label, values):
        ms = np.asarray(values) * 1000
        print(f"  {label:<23} median={np.median(ms):7.3f} mean={np.mean(ms):7.3f} p90={np.percentile(ms,90):7.3f} p95={np.percentile(ms,95):7.3f} ms")
    report("Corruption Injection", t_corruption)
    report("Upstream DQA", t_dqa)
    report("Midstream IRI", t_iri)
    report("Trust Fusion", t_fusion)
    report("Transformer Forward", t_transformer)
    print("-" * 70)
    report("Total synthetic path", t_total)
    print("  This synthetic host measurement is not an edge-device latency claim.")
    if device.type == "cuda":
        print(f"  Peak allocated GPU memory: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.1f} MiB")
    print("=" * 70)

if __name__ == "__main__":
    main()

