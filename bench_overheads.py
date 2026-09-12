"""Per-operation overheads versus grid size, for the cost-aware discussion:

  * classical iterations (Jacobi, GS, SymGS, multigrid V-cycle, FFT solve)
  * the corrector call (FFT transfers + matrix-vector product)
  * the lightweight router's decision (features + numpy MLP)
  * one decision of the paper's LSTM router (input N^2, hidden 256, 4 layers)
    as used in Section 7, to show why it cannot be deployed inside a
    wall-clock loop at these grid sizes
  * corrector / router training times (from the logs / checkpoints)

  python bench_overheads.py --Ns 32,64,128,256,512
writes results/overheads.json
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D, make_solver
from corrector import DeepONetCorrector
from hybrid import Env, FeatureState
from router import Router

p = argparse.ArgumentParser()
p.add_argument("--Ns", default="32,64,128,256,512")
p.add_argument("--reps", type=int, default=40)
p.add_argument("--out", default="results/overheads.json")
args = p.parse_args()
torch.set_num_threads(1)


def tm(fn, reps, warm=5):
    for _ in range(warm):
        fn()
    ts = np.empty(reps)
    for k in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        ts[k] = time.perf_counter_ns() - t0
    return float(np.median(ts)) * 1e-9


out = {"per_op": {}, "training": {}}
for N in [int(x) for x in args.Ns.split(",")]:
    pde = FastStencilPDE(N, "Poisson")
    f = GRF2D(N, rng=np.random.default_rng(3)).sample(1)
    u = np.zeros_like(f)
    r = pde.residual(u, f)
    row = {}
    row["residual"] = tm(lambda: pde.residual(u, f), args.reps)
    for spec in ["jacobi", "gs", "ssor", "mg", "fft"]:
        try:
            sv = make_solver(pde, spec)
            sv.step(u, f, r)
            row[spec] = row["residual"] + tm(lambda: sv.step(u, f, r), args.reps if spec != "mg" else 10)
        except Exception as e:  # e.g. MG needs N >= 64
            row[spec] = None
    ck = f"checkpoints/deeponet_Poisson_{N}_best.pth"
    if os.path.exists(ck):
        corr = DeepONetCorrector(ck)
        corr.correct(r)
        row["corrector"] = row["residual"] + tm(lambda: corr.correct(r), args.reps)
        row["corrector_n_c"] = corr.n_c
    # lightweight router decision
    K = 2
    fs = FeatureState(K, 1)
    rp = glob.glob(f"checkpoints/router_Poisson_{N}_gs.pth")
    if rp:
        rt = Router.load(rp[0])
    else:
        rt = Router([np.zeros((6 + K, 64)), np.zeros(64), np.zeros((64, 64)), np.zeros(64), np.zeros((64, K)), np.zeros(K)], K)
    def dec():
        x = fs.features(1e-3)
        d = rt.decide(x)
        fs.update(d, 1)
    row["router_decision"] = tm(dec, 300)
    # the paper's LSTM router (Section 7 / Table 4: hidden 256, 4 layers) fed the
    # full field state (a_h, f_h, u_h, residual) -- one decision at this N
    lstm = torch.nn.LSTM(input_size=4 * N * N, hidden_size=256, num_layers=4, batch_first=True)
    head = torch.nn.Linear(256, K)
    x = torch.randn(1, 1, 4 * N * N)
    h0 = (torch.zeros(4, 1, 256), torch.zeros(4, 1, 256))
    with torch.no_grad():
        def lstm_dec():
            y, _ = lstm(x, h0)
            return head(y[:, -1])
        row["lstm_router_decision"] = tm(lstm_dec, 20)
    out["per_op"][N] = row
    print(f"N={N}: " + ", ".join(f"{k} {v*1e6:.0f}us" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)

# training times: corrector (data generation + LS fit) from logs, routers from logs
for path in glob.glob("logs/deeponet_*_*.log"):
    txt = open(path).read()
    m1 = re.search(r"data: .* in ([0-9.]+)s", txt)
    m2 = re.search(r"least squares in ([0-9.]+)s", txt)
    key = os.path.basename(path)[len("deeponet_"):-4]
    if m1 and m2:
        out["training"][f"corrector_{key}"] = {"data_s": float(m1.group(1)), "fit_s": float(m2.group(1))}
for path in glob.glob("logs/routers_*.log"):
    txt = open(path).read()
    for m in re.finditer(r"=== (\S+) N=(\d+) ensemble (\S+) ===.*?trained router in (\d+)s", txt, re.S):
        out["training"][f"router_{m.group(1)}_{m.group(2)}_{m.group(3)}"] = {"train_s": float(m.group(4))}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print("saved", args.out)
