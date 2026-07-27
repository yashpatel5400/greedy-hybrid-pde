"""Untimed diagnostic: effective interleaved floor of a corrector under the
cost-aware oracle policy (batched, no wall-clock accounting). Used to compare
corrector variants before committing to a timed benchmark run.

  python fast_floor_diag.py --equation Poisson --N 127 \
      --ckps checkpoints/fast_deeponet_Poisson_127_ft_best.pth,...
"""

import argparse

import numpy as np

from fast_pde import FastStencilPDE, GRF2D, demean, l2, make_solver
from fast_hybrid import DeepONetCorrector

parser = argparse.ArgumentParser()
parser.add_argument("--equation", type=str, default="Poisson")
parser.add_argument("--N", type=int, default=127)
parser.add_argument("--solver", type=str, default="jacobi")
parser.add_argument("--ckps", type=str, required=True, help="comma-separated checkpoints")
parser.add_argument("--n_test", type=int, default=32)
parser.add_argument("--iters", type=int, default=6400)
parser.add_argument("--cost_ratio", type=float, default=4.0,
                    help="per-iteration (residual-inclusive) c_NO / c_classical")
parser.add_argument("--b_vel", type=float, default=20.0)
# NOTE: deliberately NOT the test seed (72) -- this diagnostic is used for
# corrector/recipe selection, which must not touch the benchmark test set
parser.add_argument("--seed", type=int, default=7)
args = parser.parse_args()

pde = FastStencilPDE(args.N, equation=args.equation, b_vec=(args.b_vel, args.b_vel))
solver = make_solver(pde, args.solver)
grf = GRF2D(args.N, rng=np.random.default_rng(args.seed))
f = grf.sample(args.n_test)
u_star = pde.solve_direct(f)
un = l2(demean(u_star))
checkpoints = [100, 200, 400, 800, 1600, 3200, 6400]
tiny = 1e-300

for ckp in args.ckps.split(","):
    corr = DeepONetCorrector(ckp, threads=8)
    du = corr.correct(f)
    one_shot = np.median(l2(demean(du - u_star)) / un)
    u = np.zeros_like(f)
    n_no = 0
    print(f"[{ckp.split('/')[-1]}] one-shot med rel err: {one_shot:.4f}")
    for it in range(1, args.iters + 1):
        r = pde.residual(u, f)
        u_c = solver.step(u, f, r)
        u_n = u + corr.correct(r)
        e_prev = np.maximum(l2(demean(u - u_star)), tiny)
        e_c = np.maximum(l2(demean(u_c - u_star)), tiny)
        e_n = np.maximum(l2(demean(u_n - u_star)), tiny)
        rate_c = np.log(e_prev / e_c)
        rate_n = np.log(e_prev / e_n) / args.cost_ratio
        pick = rate_n > rate_c
        n_no += int(pick.sum())
        u = np.where(pick[:, None, None], u_n, u_c)
        if it in checkpoints:
            rel = l2(demean(u - u_star)) / un
            print(f"    iter {it:5d}: rel err med {np.median(rel):.3e} "
                  f"p90 {np.quantile(rel, 0.9):.3e} | cum NO calls/inst "
                  f"{n_no/args.n_test:.1f}")
