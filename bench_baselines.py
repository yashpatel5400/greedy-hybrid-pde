"""Time-to-tolerance of strong classical baselines (FFT, multigrid, Krylov)
on the same test instances as bench.py. Stationary baselines (fft, mg) go
through hybrid.run_untimed/run_timed as 'classical' policies; Krylov methods
through baselines.run_krylov_untimed / time_krylov.

  python bench_baselines.py --equation Poisson --N 128 --n_test 64
writes results/baselines_<eq>_<N>.json
"""

import argparse
import gc
import json
import os
import time

import numpy as np

from fast_pde import FastStencilPDE, GRF2D, demean, l2, make_solver
from hybrid import Env, run_untimed, run_timed, time_to_tol
from baselines import run_krylov_untimed, time_krylov, KRYLOV

p = argparse.ArgumentParser()
p.add_argument("--equation", default="Poisson", choices=["Poisson", "ConvDiff"])
p.add_argument("--N", type=int, default=128)
p.add_argument("--n_test", type=int, default=64)
p.add_argument("--seed", type=int, default=72)
p.add_argument("--methods", default=None)
p.add_argument("--tols", default="1e-2,1e-3,h2,1e-5,1e-6,1e-8")
p.add_argument("--max_iter", type=int, default=5000)
p.add_argument("--time_cap", type=float, default=120.0, help="cap (s) on the untimed Krylov run")
p.add_argument("--out_dir", default="./results")
args = p.parse_args()

h2 = 1.0 / args.N ** 2
tols = [h2 if t == "h2" else float(t) for t in args.tols.split(",")]
if args.methods:
    methods = args.methods.split(",")
elif args.equation == "Poisson":
    methods = ["fft", "mg", "cg", "pcg_ssor", "pcg_mg"]
else:
    methods = ["fft", "mg", "bicgstab", "bicgstab_mg", "gmres"]

pde = FastStencilPDE(args.N, equation=args.equation)
f_test = GRF2D(args.N, rng=np.random.default_rng(args.seed)).sample(args.n_test)
u_truth = pde.solve_direct(f_test)
os.makedirs(args.out_dir, exist_ok=True)
res = {"args": vars(args), "tols": tols, "h2": h2, "methods": {m: [] for m in methods}}

for m in methods:
    t_start = time.time()
    if m in KRYLOV:
        for i in range(args.n_test):
            f1, u1 = f_test[i:i + 1], u_truth[i:i + 1]
            errs, hit = run_krylov_untimed(pde, f1, u1, m, tols, max_iter=args.max_iter)
            gc.collect(); gc.disable()
            unit = 20 if m.startswith("gmres") else 1
            row = {"n_iters_total": int((len(errs) - 1) * unit), "final_rel_err": float(errs[-1]), "tol": {}}
            for tol in tols:
                k = hit[tol]
                t = time_krylov(pde, f1, m, k) if k is not None else np.inf
                row["tol"][f"{tol:.6g}"] = {"iters": None if k is None else k * unit,
                                            "t_live": None if not np.isfinite(t) else float(t)}
            gc.enable()
            res["methods"][m].append(row)
    else:
        env = Env(pde, [m], None)
        _ = env.solvers[0].step(np.zeros_like(f_test[:1]), f_test[:1])
        env.costs = {m: 1.0, "_residual": 0.0}
        env.set_macro_sizes()
        for i in range(args.n_test):
            f1, u1 = f_test[i:i + 1], u_truth[i:i + 1]
            tr = run_untimed(env, f1, u1, "classical", max_ops=args.max_iter, err_stop=1e-9)
            gc.collect(); gc.disable()
            t, _ = run_timed(env, f1, tr, "classical")
            gc.enable()
            tt = time_to_tol(tr, t, tols)
            row = {"n_iters_total": int(len(tr["op"])), "final_rel_err": float(tr["rel_err"][-1]), "tol": {}}
            for tol in tols:
                tl, il = tt[tol]
                row["tol"][f"{tol:.6g}"] = {"iters": None if not np.isfinite(il) else int(il),
                                            "t_live": None if not np.isfinite(tl) else float(tl)}
            res["methods"][m].append(row)
    med = {f"{tol:g}": np.median([r["tol"][f"{tol:.6g}"]["t_live"] or np.inf for r in res["methods"][m]]) for tol in tols}
    its = {f"{tol:g}": np.median([r["tol"][f"{tol:.6g}"]["iters"] if r["tol"][f"{tol:.6g}"]["iters"] is not None else np.inf for r in res["methods"][m]]) for tol in tols}
    print(f"[{args.equation} N={args.N} {m:12s}] median time (ms): " + " ".join(f"{k}:{v*1e3:.2f}" for k, v in med.items())
          + " | iters: " + " ".join(f"{k}:{v:.0f}" for k, v in its.items()) + f"  ({time.time()-t_start:.0f}s)", flush=True)
    out = f"{args.out_dir}/baselines_{args.equation}_{args.N}.json"
    json.dump(res, open(out, "w"))
print("saved", out)
