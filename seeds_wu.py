"""Backfill timer-free work-unit times (and per-tolerance iteration counts) into
results/seeds_<eq>_<N>.json by replaying each seed router untimed on the test set."""
import argparse, json
import numpy as np, torch
from fast_pde import FastStencilPDE, GRF2D
from corrector import DeepONetCorrector
from hybrid import Env, run_untimed, time_to_tol, work_units
from router import Router
p = argparse.ArgumentParser(); p.add_argument("--equation", default="Poisson"); p.add_argument("--N", type=int, default=128)
a = p.parse_args(); torch.set_num_threads(4)
path = f"results/seeds_{a.equation}_{a.N}.json"; res = json.load(open(path)); tols = res["tols"]
pde = FastStencilPDE(a.N, equation=a.equation); corr = DeepONetCorrector(f"checkpoints/deeponet_{a.equation}_{a.N}_best.pth", threads=4)
f_test = GRF2D(a.N, rng=np.random.default_rng(res["args"]["seed"])).sample(res["args"]["n_test"]); u_truth = pde.solve_direct(f_test)
costs = json.load(open(f"checkpoints/costs_{a.equation}_{a.N}.json"))
for spec, seeds in res["groups"].items():
    env = Env(pde, [spec], corr, costs=costs[spec])
    for sd, blk in seeds.items():
        rt = Router.load(f"checkpoints/router_{a.equation}_{a.N}_{spec}_seed{sd}.pth")
        for i, row in enumerate(blk["rows"]):
            tr = run_untimed(env, f_test[i:i+1], u_truth[i:i+1], "router", max_ops=res["args"]["max_ops"], err_stop=1e-9, router=rt)
            tw = time_to_tol(tr, work_units(env, tr, "router", router_cost=5e-6), tols)
            for tol in tols:
                row["tol"][f"{tol:.6g}"]["t_wu"] = None if not np.isfinite(tw[tol][0]) else float(tw[tol][0])
                row["tol"][f"{tol:.6g}"]["iters"] = None if not np.isfinite(tw[tol][1]) else int(tw[tol][1])
        print(f"  {spec} seed {sd}: backfilled", flush=True)
json.dump(res, open(path, "w")); print("saved", path)
