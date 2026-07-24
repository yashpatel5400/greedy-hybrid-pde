"""Wall-clock time-to-tolerance benchmark: classical solver only vs hybrid
(oracle-greedy / HINTS / learned router) with a DeepONet corrector.

Example:
  python bench_wallclock.py --equation Poisson --N 63 --solvers jacobi,gs --n_test 16
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D, demean, l2, make_solver
from fast_hybrid import DeepONetCorrector, run_rollout, time_to_tol

parser = argparse.ArgumentParser()
parser.add_argument("--equation", type=str, default="Poisson", choices=["Poisson", "ConvDiff"])
parser.add_argument("--N", type=int, default=63)
parser.add_argument("--solvers", type=str, default="jacobi")
parser.add_argument("--n_test", type=int, default=16)
parser.add_argument("--b_vel", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=72)
parser.add_argument("--max_iters", type=int, default=200000)
parser.add_argument("--time_cap", type=float, default=120.0)
parser.add_argument("--res_floor", type=float, default=1e-10)
parser.add_argument("--tols", type=str, default="1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8")
parser.add_argument("--policies", type=str, default="classical,oracle,hints25")
parser.add_argument("--ckp", type=str, default=None)
parser.add_argument("--router_ckp", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./results_wallclock")
parser.add_argument("--diag", action="store_true", help="print NO diagnostics first")
args = parser.parse_args()

torch.set_num_threads(1)  # stable single-instance timing; conservative for the NO
os.makedirs(args.out_dir, exist_ok=True)
tols = [float(t) for t in args.tols.split(",")]

pde = FastStencilPDE(args.N, equation=args.equation, b_vec=(args.b_vel, args.b_vel))
ckp_path = args.ckp or f"./checkpoints/fast_deeponet_{args.equation}_{args.N}_best.pth"
corrector = DeepONetCorrector(ckp_path)

grf = GRF2D(args.N, rng=np.random.default_rng(args.seed))
f_test = grf.sample(args.n_test)
u_truth = pde.solve_direct(f_test)

# warmup (torch lazy init, splu factorizations)
_ = corrector.correct(f_test[:1])

router = None
if "router" in args.policies:
    from router_lite import LiteRouter
    router = LiteRouter.load(args.router_ckp or
                             f"./checkpoints/lite_router_{args.equation}_{args.N}_{args.solvers.split(',')[0]}.pth")

if args.diag:
    # (1) one-shot relative error of the NO on the test forcings
    du = corrector.correct(f_test)
    rel = l2(demean(du - u_truth)) / l2(demean(u_truth))
    print(f"[diag] one-shot NO rel err: median {np.median(rel):.3f}, "
          f"min {rel.min():.3f}, max {rel.max():.3f}")
    # (2) pure NO recursion: u <- u + NO(r); how far can repeated calls go?
    u = np.zeros_like(f_test)
    print("[diag] pure-NO recursion rel err per call:")
    for k in range(8):
        u = u + corrector.correct(pde.residual(u, f_test))
        rel = l2(demean(u - u_truth)) / l2(demean(u_truth))
        print(f"    call {k+1}: median {np.median(rel):.2e}  max {rel.max():.2e}")

results = {}
for spec in args.solvers.split(","):
    solver = make_solver(pde, spec)
    _ = solver.step(np.zeros_like(f_test[:1]), f_test[:1])  # warm splu path

    # stable per-op cost estimates for the cost-aware policies
    u_w = np.zeros_like(f_test[:1])
    reps = []
    for _k in range(25):
        t0 = time.perf_counter(); _ = solver.step(u_w, f_test[:1]); reps.append(time.perf_counter() - t0)
    t_classical = float(np.median(reps))
    reps = []
    for _k in range(25):
        t0 = time.perf_counter(); _ = corrector.correct(f_test[:1]); reps.append(time.perf_counter() - t0)
    t_no = float(np.median(reps))
    print(f"[{spec}] per-op cost: classical {t_classical*1e6:.0f}us, NO {t_no*1e6:.0f}us "
          f"(ratio {t_no/t_classical:.1f}x)")

    results[spec] = {"op_costs": {"classical": t_classical, "no": t_no}}
    for policy in args.policies.split(","):
        rows = []
        t_start = time.time()
        for i in range(args.n_test):
            tr = run_rollout(pde, solver, f_test[i:i + 1], u_truth[i:i + 1], policy,
                             corrector=corrector, router=router,
                             max_iters=args.max_iters, time_cap=args.time_cap,
                             res_floor=args.res_floor, op_costs=(t_classical, t_no))
            rows.append({
                "res": {str(t): time_to_tol(tr, [t], "rel_res")[t] for t in tols},
                "err": {str(t): time_to_tol(tr, [t], "rel_err")[t] for t in tols},
                "n_iters": int(len(tr["decision"])),
                "n_no_calls": int((tr["decision"] == 1).sum()),
                "final_rel_res": float(tr["rel_res"][-1]),
                "final_rel_err": float(tr["rel_err"][-1]),
            })
        results[spec][policy] = rows
        med = {f"{t:g}": float(np.median([r["res"][str(t)][0] for r in rows])) for t in tols}
        its = {f"{t:g}": float(np.median([r["res"][str(t)][1] for r in rows])) for t in tols}
        nno = float(np.median([r["n_no_calls"] for r in rows]))
        print(f"[{args.equation} N={args.N} {spec} | {policy:>9s}] "
              f"median time-to-tol (s): " +
              " ".join(f"{k}:{v:.3g}" for k, v in med.items()) +
              f" | iters@1e-6: {its.get('1e-06', float('nan')):.0f}"
              f" | NO calls: {nno:.0f} | bench took {time.time()-t_start:.0f}s", flush=True)

out_path = f"{args.out_dir}/{args.equation}_{args.N}_{args.solvers.replace(',', '+')}.json"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and np.isinf(o):
        return "inf"
    return o


with open(out_path, "w") as fh:
    json.dump(_clean({"args": vars(args), "results": results}), fh)
print("saved", out_path)
