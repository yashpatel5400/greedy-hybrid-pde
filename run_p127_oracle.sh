#!/bin/zsh
# 127^2 campaign, phase 1: paper's pure error-greedy oracle vs HINTS(tau=25),
# all 5 solver pairings x 2 PDEs. Riskiest pairings first for early signal.
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
TOLS="1e-2,1e-3,1e-4,6.2e-5,3.1e-5,1e-5,6.2e-6,1e-6,1e-8"
for S in sor_1.5 ssor gs jacobi_0.67 jacobi; do
  for EQ in Poisson ConvDiff; do
    echo "=== [P1-127] $EQ $S ==="
    $PY bench_wallclock.py --equation $EQ --N 127 --solvers $S --n_test 32 \
      --ckp checkpoints/fast_deeponet_${EQ}_127_ft_best.pth \
      --policies classical,oracle,hints25 --tols $TOLS \
      --time_cap 200 --res_floor 1e-10 --out_dir results_wallclock_ph2 \
      2>&1 | grep --line-buffered -E "median|per-iteration|saved|Traceback"
  done
done
echo P127_ORACLE_DONE
