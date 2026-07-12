# Full-Scale Benchmark Run (MetroPT & AI4I 2020)
# This script executes the academic validation strategy across datasets and seeds.

Write-Output "=========================================================="
Write-Output "Starting Full-Scale Benchmark Run for CALI-PRED"
Write-Output "=========================================================="

Write-Output "`n[1/6] Training MetroPT (Seed 42)..."
python pipeline.py --dataset metropt --epochs 40 --seed 42 --checkpoint-dir checkpoints_metropt_42

Write-Output "`n[2/6] Training MetroPT (Seed 123)..."
python pipeline.py --dataset metropt --epochs 40 --seed 123 --checkpoint-dir checkpoints_metropt_123

Write-Output "`n[3/6] Training MetroPT (Seed 456)..."
python pipeline.py --dataset metropt --epochs 40 --seed 456 --checkpoint-dir checkpoints_metropt_456

Write-Output "`n[4/6] Training AI4I 2020 (Seed 42)..."
python pipeline.py --dataset ai4i2020 --data-path data/ai4i2020/ai4i2020.csv --epochs 40 --seed 42 --checkpoint-dir checkpoints_ai4i_42

Write-Output "`n[5/6] Training AI4I 2020 (Seed 123)..."
python pipeline.py --dataset ai4i2020 --data-path data/ai4i2020/ai4i2020.csv --epochs 40 --seed 123 --checkpoint-dir checkpoints_ai4i_123

Write-Output "`n[6/6] Running Block Bootstrap Analysis on MetroPT Seed 42 (2000 resamples)..."
python bootstrap_analysis.py --bootstrap-only --n-bootstrap 2000 --checkpoint-dir checkpoints_metropt_42 --dataset metropt

Write-Output "`n=========================================================="
Write-Output "Benchmark runs completed successfully!"
Write-Output "=========================================================="
