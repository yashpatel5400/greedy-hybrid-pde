"""Collect (coarse residual, exact coarse error) pairs from hybrid rollouts for
residual-distribution fine-tuning of the corrector (DAgger-style correction of
the corrector's input distribution).

Rollouts follow the cost-aware oracle with epsilon-exploration for each of the
given solver pairings; at every decision epoch the band-limited residual seen
by the corrector is recorded together with its exact error (FFT ground truth),
in the corrector's own normalisation (x = R r / ||R r||, y = R e / ||R r||).

  python collect_corrector_pairs.py --equation Poisson --N 128 --solvers jacobi,gs
  python corrector.py --equation Poisson --N 128 --init_from checkpoints/deeponet_Poisson_128_best.pth \
      --extra_pairs checkpoints/rollout_pairs_Poisson_128.npz --suffix _ft --lr 3e-4 --steps 20000
"""

import argparse
import time

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D, demean, l2
from corrector import DeepONetCorrector
from hybrid import Env, measure_costs

p = argparse.ArgumentParser()
p.add_argument("--equation", default="Poisson")
p.add_argument("--N", type=int, default=128)
p.add_argument("--solvers", default="jacobi,gs,sor_1.5")
p.add_argument("--n_inst", type=int, default=96)
p.add_argument("--eps", type=float, default=0.15)
p.add_argument("--max_epochs", type=int, default=400)
p.add_argument("--err_stop", type=float, default=1e-10)
p.add_argument("--seed", type=int, default=777)
p.add_argument("--b_vel", type=float, default=20.0)
p.add_argument("--ckp", default=None)
p.add_argument("--out", default=None)
args = p.parse_args()

torch.set_num_threads(8)
rng = np.random.default_rng(args.seed)
pde = FastStencilPDE(args.N, equation=args.equation, b_vec=(args.b_vel, args.b_vel))
corr = DeepONetCorrector(args.ckp or f"checkpoints/deeponet_{args.equation}_{args.N}_best.pth", threads=8)
T = corr.transfer
X, Y = [], []
t0 = time.time()
for spec in args.solvers.split(","):
    env = Env(pde, [spec], corr)
    env.costs = measure_costs(env, GRF2D(args.N, rng=np.random.default_rng(1)).sample(1))
    env.set_macro_sizes()
    f = GRF2D(args.N, rng=rng).sample(args.n_inst)
    u_star = pde.solve_direct(f)
    un = l2(demean(u_star))
    u = np.zeros_like(f)
    active = np.ones(args.n_inst, dtype=bool)
    n_rec = 0
    for ep in range(args.max_epochs):
        r = pde.residual(u, f)
        rel_err = l2(demean(u - u_star)) / un
        active &= rel_err > args.err_stop
        if not active.any():
            break
        # record the corrector's input/target at this state
        rc = T.restrict(r[active])
        ec = T.restrict(demean(u_star[active] - u[active]))
        rn = np.sqrt((rc ** 2).sum(axis=(-2, -1), keepdims=True))
        ok = rn[:, 0, 0] > 1e-300
        X.append((rc[ok] / rn[ok]).reshape(ok.sum(), -1).astype(np.float32))
        Y.append((ec[ok] / rn[ok]).reshape(ok.sum(), -1).astype(np.float32))
        n_rec += int(ok.sum())
        # oracle (cost-aware) action with exploration
        cands = [env.apply_macro(j, u, f, r) for j in range(env.K)]
        errs = np.stack([l2(demean(c - u_star)) for c in cands], axis=1)
        act = np.argmin(errs, axis=1)
        flip = rng.random(args.n_inst) < args.eps
        act[flip] = rng.integers(env.K, size=flip.sum())
        u = np.stack([cands[act[b]][b] for b in range(args.n_inst)])
    print(f"  {spec}: {ep+1} epochs, {n_rec} pairs ({time.time()-t0:.0f}s)", flush=True)
X = np.concatenate(X)
Y = np.concatenate(Y)
out = args.out or f"checkpoints/rollout_pairs_{args.equation}_{args.N}.npz"
np.savez_compressed(out, X=X, Y=Y)
# how well does the current corrector do on these states?
with torch.no_grad():
    pred = corr.branch(torch.tensor(X * corr.in_scale)) @ corr.basis_T
pred = pred.numpy() * corr.inv_scale
rel = np.linalg.norm(pred - Y, axis=1) / np.maximum(np.linalg.norm(Y, axis=1), 1e-300)
print(f"saved {len(X)} pairs to {out}; current corrector rel err on them: "
      f"median {np.median(rel):.3f} mean {rel.mean():.3f} p90 {np.quantile(rel, 0.9):.3f} max {rel.max():.3f}")
