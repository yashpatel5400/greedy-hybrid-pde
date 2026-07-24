#!/bin/zsh
PY=/Users/yash/miniconda3/envs/l2dseq/bin/python
cd /Users/yash/Documents/Personal/greedy-hybrid-pde
run() { echo "=== router $1 $2 $3 ==="; $PY router_lite.py --equation $1 --N $2 --solver $3 --n_inst $4 --max_roll $5 > logs/router_$1_$2_$3.log 2>&1; tail -1 logs/router_$1_$2_$3.log; }
run Poisson 31 jacobi 192 4000
run ConvDiff 31 jacobi 192 4000
run ConvDiff 63 jacobi 192 8000
run Poisson 63 gs 192 6000
run Poisson 63 jacobi_0.67 192 8000
run Poisson 63 sor_1.5 192 6000
run ConvDiff 63 sor_1.5 192 6000
run Poisson 127 jacobi 96 14000
run ConvDiff 127 jacobi 96 14000
run Poisson 127 sor_1.5 96 8000
echo ROUTER_QUEUE_DONE
