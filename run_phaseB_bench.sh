#!/bin/zsh
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
until grep -q FINAL_CHAIN_STAGE1_DONE logs/phaseA_queue.log; do sleep 60; done
TOLS="1e-2,1e-3,1e-4,6.2e-5,3.1e-5,1e-5,6.2e-6,1e-6,1e-8"
OLDG=checkpoints/oldgen_fast_deeponet_ConvDiff_127_ft_best.pth
b() { # eq solver ckp
  $PY bench_wallclock.py --equation $1 --N 127 --solvers $2 --n_test 32 \
    --ckp $3 --policies classical,oracle,router,hints25 --tols $TOLS \
    --time_cap 200 --res_floor 1e-10 --out_dir results_wallclock_phaseB \
    2>&1 | grep --line-buffered -E "median|router decision|saved|Traceback"
}
echo "=== [PB] Phase B 4-policy benches ==="
b Poisson jacobi       checkpoints/fast_deeponet_Poisson_127_ft_best.pth
b Poisson jacobi_0.67  checkpoints/fast_deeponet_Poisson_127_ft_best.pth
b Poisson gs           checkpoints/fast_deeponet_Poisson_127_ft_best.pth
b Poisson ssor         checkpoints/fast_deeponet_Poisson_127_ft_best.pth
b Poisson sor_1.5      checkpoints/fast_deeponet_Poisson_127_ft_best.pth
b ConvDiff jacobi      checkpoints/capgen_fast_deeponet_ConvDiff_127_ft_best.pth
b ConvDiff jacobi_0.67 $OLDG
b ConvDiff gs          $OLDG
b ConvDiff ssor        $OLDG
b ConvDiff sor_1.5     $OLDG
echo PHASEB_BENCH_DONE
