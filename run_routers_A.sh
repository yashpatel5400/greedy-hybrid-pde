#!/bin/zsh
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
r() { $PY router_lite.py --equation $1 --N 127 --solver $2 --n_inst 96 --max_roll $3 \
  --ckp checkpoints/fast_deeponet_$1_127_ft_best.pth > logs/router3_$1_127_$2.log 2>&1
  tail -1 logs/router3_$1_127_$2.log }
r Poisson jacobi 14000
r Poisson gs 6000
r Poisson ssor 5000
echo ROUTERS_A_DONE
