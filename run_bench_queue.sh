#!/bin/zsh
# Final wall-clock benchmark queue. Run on an otherwise-idle machine (timing).
# Tolerance grids include the truncation-level tolerance h^2 = 1/N^2 per grid.
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde

TOLS31="1e-2,1.04e-3,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8"
TOLS63="1e-2,1e-3,2.52e-4,1e-4,1e-5,1e-6,1e-7,1e-8"
TOLS127="1e-2,1e-3,1e-4,6.2e-5,1e-5,1e-6,1e-7,1e-8"

bench() { # eq N solver n_test tols extra_policies
  eq=$1; n=$2; sol=$3; nt=$4; tols=$5; pol=$6
  echo "=== bench $eq N=$n $sol ==="
  $PY bench_wallclock.py --equation $eq --N $n --solvers $sol --n_test $nt \
    --ckp checkpoints/fast_deeponet_${eq}_${n}_ft_best.pth \
    --policies $pol --tols $tols --time_cap 150 --res_floor 1e-10 \
    2>&1 | grep -E "median|per-op|saved"
}

P="classical,oracle_ca,router,hints25"
# headline: jacobi pairing across equations and grids
bench Poisson  31  jacobi 32 $TOLS31  "$P"
bench ConvDiff 31  jacobi 32 $TOLS31  "$P"
bench Poisson  63  jacobi 32 $TOLS63  "$P,hints5,hints10,hints50,hints100"
bench ConvDiff 63  jacobi 32 $TOLS63  "$P"
bench Poisson  127 jacobi 32 $TOLS127 "$P"
bench ConvDiff 127 jacobi 32 $TOLS127 "$P"
# solver-strength axis at N=63 / 127
bench Poisson  63  gs          16 $TOLS63  "$P"
bench Poisson  63  jacobi_0.67 16 $TOLS63  "$P"
bench Poisson  63  sor_1.5     16 $TOLS63  "$P"
bench ConvDiff 63  sor_1.5     16 $TOLS63  "$P"
bench Poisson  127 sor_1.5     16 $TOLS127 "$P"
echo BENCH_QUEUE_DONE
