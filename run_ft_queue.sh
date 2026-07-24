#!/bin/zsh
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
for cfg in "ConvDiff 63" "Poisson 127" "ConvDiff 127" "Poisson 31" "ConvDiff 31"; do
  set -- ${=cfg}
  eq=$1; n=$2
  echo "=== finetuning $eq N=$n ==="
  $PY finetune_deeponet_residual.py --equation $eq --N $n --solver jacobi > logs/ft_${eq}_${n}.log 2>&1
  tail -1 logs/ft_${eq}_${n}.log
done
echo FT_QUEUE_DONE
