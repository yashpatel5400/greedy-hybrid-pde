#!/bin/zsh
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
TOLS="1e-2,1e-3,1e-4,6.2e-5,3.1e-5,1e-5,6.2e-6,1e-6,1e-8"
bench1() { $PY bench_wallclock.py --equation $1 --N 127 --solvers $2 --n_test 32 \
  --ckp $3 --policies classical,oracle,hints25 --tols $TOLS --time_cap 200 \
  --res_floor 1e-10 --out_dir results_wallclock_ph3 \
  2>&1 | grep --line-buffered -E "median|saved|Traceback" }
ftp() { $PY finetune_deeponet_residual.py --equation $1 --N 127 --solver $2 \
  --collect_policy oracle --p_no 0.05 --rollout_iters 1600 --n_col 128 --n_orig 5000 \
  > logs/ft_$1_127_$2.log 2>&1; tail -1 logs/ft_$1_127_$2.log }
echo "=== [ON2b] remaining ConvDiff fine-tunes ==="
for S in sor_1.5 ssor jacobi_0.67; do ftp ConvDiff $S; done
echo "=== [ON3] ConvDiff re-benches ==="
bench1 ConvDiff jacobi      checkpoints/fast_deeponet_ConvDiff_127_ft_best.pth
bench1 ConvDiff gs          checkpoints/fast_deeponet_ConvDiff_127_ft_gs_best.pth
bench1 ConvDiff sor_1.5     checkpoints/fast_deeponet_ConvDiff_127_ft_sor_1.5_best.pth
bench1 ConvDiff ssor        checkpoints/fast_deeponet_ConvDiff_127_ft_ssor_best.pth
bench1 ConvDiff jacobi_0.67 checkpoints/fast_deeponet_ConvDiff_127_ft_jacobi_0.67_best.pth
echo "=== [ON4] Poisson pairing polish ==="
for S in gs sor_1.5 ssor; do ftp Poisson $S; done
bench1 Poisson gs      checkpoints/fast_deeponet_Poisson_127_ft_gs_best.pth
bench1 Poisson sor_1.5 checkpoints/fast_deeponet_Poisson_127_ft_sor_1.5_best.pth
bench1 Poisson ssor    checkpoints/fast_deeponet_Poisson_127_ft_ssor_best.pth
echo OVERNIGHT_FIX_DONE
