#!/bin/zsh
# Phase A sequence (STRICTLY SEQUENTIAL - timing integrity):
#   1. tau-sweep benches at 127^2 with the improved (ft2) correctors
#   2. N=191 corrector training + residual fine-tune
#   3. tau-sweep bench at 191^2
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde

POL="classical,oracle_ca,hints10,hints25,hints50,hints100,hints200,hints400"
TOLS127="1e-2,1e-3,1e-4,6.2e-5,3.1e-5,1e-5,6.2e-6,1e-6,1e-8"
# N=191: h^2 = 1/191^2 = 2.74e-5; h^2/2 = 1.37e-5; h^2/10 = 2.74e-6
TOLS191="1e-2,1e-3,1e-4,2.74e-5,1.37e-5,1e-5,2.74e-6,1e-6,1e-8"

echo "=== [1/5] tau-sweep bench Poisson 127 ==="
$PY bench_wallclock.py --equation Poisson --N 127 --solvers jacobi --n_test 32 \
  --ckp checkpoints/fast_deeponet_Poisson_127_ft_best.pth \
  --policies $POL --tols $TOLS127 --time_cap 200 --res_floor 1e-10 \
  --out_dir results_wallclock_ph1 2>&1 | grep -E "median|per-op|saved"

echo "=== [2/5] tau-sweep bench ConvDiff 127 ==="
$PY bench_wallclock.py --equation ConvDiff --N 127 --solvers jacobi --n_test 32 \
  --ckp checkpoints/fast_deeponet_ConvDiff_127_ft_best.pth \
  --policies $POL --tols $TOLS127 --time_cap 200 --res_floor 1e-10 \
  --out_dir results_wallclock_ph1 2>&1 | grep -E "median|per-op|saved"

echo "=== [3/5] train N=191 correctors ==="
for eq in Poisson ConvDiff; do
  $PY train_fast_deeponet.py --equation $eq --N 191 --n_train 6000 --n_val 800 \
      --epochs 220 > logs/deeponet_${eq}_191.log 2>&1
  tail -1 logs/deeponet_${eq}_191.log
done

echo "=== [4/5] residual fine-tune N=191 ==="
for eq in Poisson ConvDiff; do
  $PY finetune_deeponet_residual.py --equation $eq --N 191 --solver jacobi \
      --rollout_iters 2400 --n_col 128 --n_orig 5000 > logs/ft_${eq}_191.log 2>&1
  tail -1 logs/ft_${eq}_191.log
done

echo "=== [5/5] tau-sweep bench 191 ==="
for eq in Poisson ConvDiff; do
  $PY bench_wallclock.py --equation $eq --N 191 --solvers jacobi --n_test 32 \
    --ckp checkpoints/fast_deeponet_${eq}_191_ft_best.pth \
    --policies $POL --tols $TOLS191 --time_cap 400 --res_floor 1e-10 \
    --out_dir results_wallclock_ph1 2>&1 | grep -E "median|per-op|saved"
done
echo PHASEA_QUEUE_DONE
