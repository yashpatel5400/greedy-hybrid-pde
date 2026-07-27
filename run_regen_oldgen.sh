#!/bin/zsh
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
echo "=== [RG1] regenerate old-gen ConvDiff 127 base (original recipe) ==="
$PY train_fast_deeponet.py --equation ConvDiff --N 127 --n_train 8000 --n_val 800 \
  --epochs 250 > logs/deeponet_ConvDiff_127_oldgen.log 2>&1
tail -1 logs/deeponet_ConvDiff_127_oldgen.log
echo "=== [RG2] old-gen shared fine-tune (ft2 recipe: random policy, p_no 0.25) ==="
$PY finetune_deeponet_residual.py --equation ConvDiff --N 127 --solver jacobi \
  --rollout_iters 1600 --n_col 256 > logs/ft2_ConvDiff_127_oldgen.log 2>&1
tail -1 logs/ft2_ConvDiff_127_oldgen.log
# park old-gen under distinct names; restore capgen as the live base
cd checkpoints
cp fast_deeponet_ConvDiff_127_best.pth oldgen_fast_deeponet_ConvDiff_127_best.pth
cp fast_deeponet_ConvDiff_127_ft_best.pth oldgen_fast_deeponet_ConvDiff_127_ft_best.pth
cp capgen_fast_deeponet_ConvDiff_127_best.pth fast_deeponet_ConvDiff_127_best.pth
cp capgen_fast_deeponet_ConvDiff_127_ft_best.pth fast_deeponet_ConvDiff_127_ft_best.pth
echo REGEN_DONE
