#!/bin/zsh
# Targeted per-pairing fine-tune + single-cell re-bench for failing cells.
# Usage: zsh run_ft_targeted.sh EQ SOLVER [P_NO] [ROLLOUT]
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
EQ=$1; S=$2; PNO=${3:-0.05}; ROLL=${4:-1600}
TOLS="1e-2,1e-3,1e-4,6.2e-5,3.1e-5,1e-5,6.2e-6,1e-6,1e-8"
$PY finetune_deeponet_residual.py --equation $EQ --N 127 --solver $S \
    --collect_policy oracle --p_no $PNO --rollout_iters $ROLL --n_col 128 --n_orig 5000 \
    > logs/ft_${EQ}_127_${S}.log 2>&1
tail -1 logs/ft_${EQ}_127_${S}.log
$PY bench_wallclock.py --equation $EQ --N 127 --solvers $S --n_test 32 \
    --ckp checkpoints/fast_deeponet_${EQ}_127_ft_${S}_best.pth \
    --policies classical,oracle,hints25 --tols $TOLS \
    --time_cap 200 --res_floor 1e-10 --out_dir results_wallclock_ph2 \
    2>&1 | grep --line-buffered -E "median|saved|Traceback"
