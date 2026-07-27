#!/bin/zsh
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
until grep -q ROUTERS_A_DONE logs/phaseA_queue.log && grep -q ROUTERS_B_DONE logs/phaseA_queue.log; do sleep 60; done
TOLS="1e-2,1e-3,1e-4,6.2e-5,3.1e-5,1e-5,6.2e-6,1e-6,1e-8"
OLDG=checkpoints/oldgen_fast_deeponet_ConvDiff_127_ft_best.pth
echo "=== [F1] ConvDiff final oracle benches (selected correctors) ==="
for S in sor_1.5 ssor gs jacobi_0.67; do
  $PY bench_wallclock.py --equation ConvDiff --N 127 --solvers $S --n_test 32 \
    --ckp $OLDG --policies classical,oracle,hints25 --tols $TOLS --time_cap 200 \
    --res_floor 1e-10 --out_dir results_wallclock_final \
    2>&1 | grep --line-buffered -E "median|saved|Traceback"
done
$PY bench_wallclock.py --equation ConvDiff --N 127 --solvers jacobi --n_test 32 \
  --ckp checkpoints/capgen_fast_deeponet_ConvDiff_127_ft_best.pth \
  --policies classical,oracle,hints25 --tols $TOLS --time_cap 200 \
  --res_floor 1e-10 --out_dir results_wallclock_final \
  2>&1 | grep --line-buffered -E "median|saved|Traceback"
echo "=== [F2] Poisson final oracle benches (shared ft corrector) ==="
for S in jacobi jacobi_0.67 gs ssor sor_1.5; do
  $PY bench_wallclock.py --equation Poisson --N 127 --solvers $S --n_test 32 \
    --ckp checkpoints/fast_deeponet_Poisson_127_ft_best.pth \
    --policies classical,oracle,hints25 --tols $TOLS --time_cap 200 \
    --res_floor 1e-10 --out_dir results_wallclock_final \
    2>&1 | grep --line-buffered -E "median|saved|Traceback"
done
echo "=== [F3] ConvDiff routers (selected correctors) ==="
r() { $PY router_lite.py --equation ConvDiff --N 127 --solver $1 --n_inst 96 --max_roll $2 \
  --ckp $3 > logs/router3_ConvDiff_127_$1.log 2>&1; tail -1 logs/router3_ConvDiff_127_$1.log }
r jacobi 14000 checkpoints/capgen_fast_deeponet_ConvDiff_127_ft_best.pth
r jacobi_0.67 12000 $OLDG
r gs 6000 $OLDG
r ssor 5000 $OLDG
r sor_1.5 4000 $OLDG
echo FINAL_CHAIN_STAGE1_DONE
