#!/usr/bin/env python3
"""
benchmark_external_baselines.py

Evaluates CALIPRED against standard external calibration/UQ baselines on saved test predictions:
1. Trust-Blind Baseline (Uncalibrated)
2. Post-Hoc Temperature / Variance Scaling (Guo et al., ICML 2017)
3. Standard Conformal Prediction (Romano et al., NeurIPS 2019)
4. CALIPRED (Proposed Quality-Aware Dynamic Inflation)

Runs in < 2 seconds using vectorized numpy operations.
"""

import os
import glob
import numpy as np

def evaluate_baselines_on_predictions(npz_path: str):
    data = np.load(npz_path)
    
    y_true = data["calipred_y_true"].flatten()
    dti = data["calipred_dti"].flatten()
    
    # Baseline raw predictions
    mu_base = data["baseline_mu"].flatten()
    sigma_base = data["baseline_sigma"].flatten()
    
    # CALIPRED predictions
    mu_cali = data["calipred_mu"].flatten()
    sigma_cali = data["calipred_sigma"].flatten()
    
    n_samples = len(y_true)
    
    # Stratification masks
    clean_mask = dti >= 0.70
    degraded_mask = dti <= 0.50
    
    z_95 = 1.960
    
    # Gaussian CRPS calculation helper
    def compute_crps(y, mu, sigma):
        from scipy.stats import norm
        sigma = np.maximum(sigma, 1e-6)
        zeta = (y - mu) / sigma
        crps = sigma * (zeta * (2 * norm.cdf(zeta) - 1) + 2 * norm.pdf(zeta) - 1.0 / np.sqrt(np.pi))
        return np.mean(crps)

    # 1. Uncalibrated Baseline
    cov_95_uncal = np.mean(np.abs(y_true - mu_base) <= z_95 * sigma_base)
    cov_deg_uncal = np.mean(np.abs(y_true[degraded_mask] - mu_base[degraded_mask]) <= z_95 * sigma_base[degraded_mask])
    crps_clean_uncal = compute_crps(y_true[clean_mask], mu_base[clean_mask], sigma_base[clean_mask])
    
    # 2. Post-Hoc Temperature/Variance Scaling (Guo et al., 2017)
    # Fit scalar s on first 20% (proxy validation split) to achieve 95% nominal coverage
    n_val = int(0.20 * n_samples)
    y_val, mu_val, sig_val = y_true[:n_val], mu_base[:n_val], sigma_base[:n_val]
    y_test, mu_test, sig_test = y_true[n_val:], mu_base[n_val:], sigma_base[n_val:]
    dti_test = dti[n_val:]
    clean_test = dti_test >= 0.70
    deg_test = dti_test <= 0.50
    
    # Search optimal global scale for 95% nominal target on validation
    scales = np.linspace(0.5, 3.0, 250)
    best_scale = 1.0
    min_diff = 1.0
    for s in scales:
        c = np.mean(np.abs(y_val - mu_val) <= z_95 * (s * sig_val))
        if abs(c - 0.95) < min_diff:
            min_diff = abs(c - 0.95)
            best_scale = s
            
    sigma_scaled = best_scale * sig_test
    cov_95_scaled = np.mean(np.abs(y_test - mu_test) <= z_95 * sigma_scaled)
    cov_deg_scaled = np.mean(np.abs(y_test[deg_test] - mu_test[deg_test]) <= z_95 * sigma_scaled[deg_test])
    crps_clean_scaled = compute_crps(y_test[clean_test], mu_test[clean_test], sigma_scaled[clean_test])
    
    # 3. Standard Conformal Prediction (CQR / Romano et al., 2019)
    # Non-conformity scores: R_i = |y - mu| / sigma
    scores_val = np.abs(y_val - mu_val) / np.maximum(sig_val, 1e-6)
    q_conf = np.quantile(scores_val, 0.95 * (1.0 + 1.0 / n_val))
    
    cov_95_conf = np.mean(np.abs(y_test - mu_test) <= q_conf * sig_test)
    cov_deg_conf = np.mean(np.abs(y_test[deg_test] - mu_test[deg_test]) <= q_conf * sig_test[deg_test])
    crps_clean_conf = compute_crps(y_test[clean_test], mu_test[clean_test], (q_conf / z_95) * sig_test[clean_test])
    
    # 4. CALIPRED
    cov_95_cali = np.mean(np.abs(y_test - mu_cali[n_val:]) <= z_95 * sigma_cali[n_val:])
    cov_deg_cali = np.mean(np.abs(y_test[deg_test] - mu_cali[n_val:][deg_test]) <= z_95 * sigma_cali[n_val:][deg_test])
    crps_clean_cali = compute_crps(y_test[clean_test], mu_cali[n_val:][clean_test], sigma_cali[n_val:][clean_test])
    
    return {
        "Uncalibrated": (cov_95_uncal, cov_deg_uncal, crps_clean_uncal),
        "PostHoc_Scaling": (cov_95_scaled, cov_deg_scaled, crps_clean_scaled),
        "Conformal_Prediction": (cov_95_conf, cov_deg_conf, crps_clean_conf),
        "CALIPRED": (cov_95_cali, cov_deg_cali, crps_clean_cali),
    }

def main():
    candidate_paths = glob.glob("checkpoints/seed_*/test_predictions.npz")
    if not candidate_paths:
        candidate_paths = glob.glob("checkpoints/test_predictions.npz")
        
    if not candidate_paths:
        print("[ERROR] No prediction files found in checkpoints/")
        return
        
    print(f"Found {len(candidate_paths)} prediction files. Evaluating external baselines...")
    
    all_results = {"Uncalibrated": [], "PostHoc_Scaling": [], "Conformal_Prediction": [], "CALIPRED": []}
    
    for p in candidate_paths:
        print(f"\n--> Processing: {p}")
        res = evaluate_baselines_on_predictions(p)
        for k in all_results:
            all_results[k].append(res[k])
            
    print("\n" + "="*85)
    print("  EXTERNAL CALIBRATION & UQ BASELINE COMPARISON TABLE (Mean across Seeds)")
    print("="*85)
    print(f"{'Method / Baseline':<32} | {'95% Overall Cov':<15} | {'Degraded Cov (DTI<=0.5)':<23} | {'Clean CRPS (DTI>=0.7)':<20}")
    print("-" * 85)
    
    for k, v in all_results.items():
        arr = np.array(v)
        mean_cov95 = np.mean(arr[:, 0])
        std_cov95 = np.std(arr[:, 0])
        mean_cov_deg = np.mean(arr[:, 1])
        mean_crps_clean = np.mean(arr[:, 2])
        
        print(f"{k:<32} | {mean_cov95:.4f} ± {std_cov95:.4f}  | {mean_cov_deg:.4f}                  | {mean_crps_clean:.4f}")
        
    print("="*85)

if __name__ == "__main__":
    main()
