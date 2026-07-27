#!/bin/zsh
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
POL="classical,oracle_ca,hints10,hints25,hints50,hints100,hints200,hints400"
POL63="classical,oracle_ca,hints5,hints10,hints25,hints50,hints100,hints200"
TOLS127="1e-2,1e-3,1e-4,6.2e-5,3.1e-5,1e-5,6.2e-6,1e-6,1e-8"
TOLS191="1e-2,1e-3,1e-4,2.74e-5,1.37e-5,1e-5,2.74e-6,1e-6,1e-8"
TOLS63="1e-2,1e-3,2.52e-4,1e-4,2.52e-5,1e-5,1e-6,1e-7,1e-8"
bench() { # eq N solver tols timecap
  $PY bench_wallclock.py --equation $1 --N $2 --solvers $3 --n_test 32 \
    --ckp checkpoints/fast_deeponet_${1}_${2}_ft_best.pth \
    --policies $6 --tols $4 --time_cap $5 --res_floor 1e-10 \
    --out_dir results_wallclock_ph1 2>&1 | grep --line-buffered -E "median|per-iteration|saved|Traceback"
}
echo "=== [S1] 63^2 tau-sweep benches ==="
bench Poisson  63 jacobi $TOLS63 120 $POL63
bench ConvDiff 63 jacobi $TOLS63 120 $POL63
echo "=== [S2] 191^2 tau-sweep benches ==="
bench Poisson  191 jacobi $TOLS191 400 $POL
bench ConvDiff 191 jacobi $TOLS191 400 $POL
echo "=== [S3] jacobi_0.67 127^2 tau-sweep benches ==="
bench Poisson  127 jacobi_0.67 $TOLS127 200 $POL
bench ConvDiff 127 jacobi_0.67 $TOLS127 200 $POL
echo "=== [S4] router training ==="
r() { $PY router_lite.py --equation $1 --N $2 --solver $3 --n_inst $4 --max_roll $5 > logs/router2_$1_$2_$3.log 2>&1; tail -1 logs/router2_$1_$2_$3.log }
r Poisson  63  jacobi      192 8000
r ConvDiff 63  jacobi      192 8000
r Poisson  127 jacobi      96  14000
r ConvDiff 127 jacobi      96  14000
r Poisson  127 jacobi_0.67 96  10000
r ConvDiff 127 jacobi_0.67 96  10000
r Poisson  191 jacobi      48  10000
r ConvDiff 191 jacobi      48  10000
echo RESUME2_DONE
