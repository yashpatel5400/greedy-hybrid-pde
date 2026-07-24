#!/bin/zsh
# Sequential DeepONet training queue for the wall-clock study.
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
mkdir -p logs
for cfg in "Poisson 63 10000 250" "ConvDiff 63 10000 250" "Poisson 127 8000 250" "ConvDiff 127 8000 250" "Poisson 31 10000 250" "ConvDiff 31 10000 250"; do
  set -- ${=cfg}
  eq=$1; n=$2; ntr=$3; ep=$4
  echo "=== training $eq N=$n ==="
  $PY train_fast_deeponet.py --equation $eq --N $n --n_train $ntr --epochs $ep \
      > logs/deeponet_${eq}_${n}.log 2>&1
  tail -2 logs/deeponet_${eq}_${n}.log
done
echo "ALL DONE"
