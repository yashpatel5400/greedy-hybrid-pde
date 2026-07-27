"""Residual-distribution fine-tuning of the DeepONet corrector.

The base corrector is trained on GRF forcings f, but inside a hybrid solve it is
applied to *residuals*, whose distribution drifts away from the forcing
distribution after the first few corrections (this is why pure-NO recursion
stalls, and why the paper's DeepONet is only selected early). Since the
operator is linear and the exact error e = A^{-1} r is available in closed form
(FFT) for any residual, we can fine-tune on the states the hybrid solver
actually visits (a DAgger-style correction of the input distribution):

  1. roll out hybrid solves on fresh training instances under an exploratory
     policy (random Jacobi/NO mixture) with the base corrector;
  2. snapshot residuals at log-spaced iterations, label them exactly;
  3. fine-tune on a mixture of the original (f -> u) pairs and the collected
     (r -> e) pairs, keeping the same scale-equivariant normalization.
"""

import argparse
import json
import time

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D, demean, l2, make_solver
from fast_hybrid import DeepONetCorrector
from ml_solver import DeepONet

parser = argparse.ArgumentParser()
parser.add_argument("--equation", type=str, default="Poisson", choices=["Poisson", "ConvDiff"])
parser.add_argument("--N", type=int, default=63)
parser.add_argument("--solver", type=str, default="jacobi")
parser.add_argument("--n_col", type=int, default=512, help="instances for residual collection")
parser.add_argument("--rollout_iters", type=int, default=400)
parser.add_argument("--p_no", type=float, default=0.25, help="explore prob of NO call")
parser.add_argument("--collect_policy", type=str, default="random",
                    choices=["random", "oracle"],
                    help="advance rollouts randomly (p_no) or with the paper's "
                         "pure error-greedy oracle (+ p_no exploration)")
parser.add_argument("--n_orig", type=int, default=8000, help="original f->u pairs kept in the mix")
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--b_vel", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
parser.add_argument("--ckp_dir", type=str, default="./checkpoints")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
base_tag = f"fast_deeponet_{args.equation}_{args.N}"
# per-pairing fine-tunes get solver-tagged checkpoints (jacobi keeps the
# legacy untagged name for backwards compatibility)
out_tag = base_tag + ("_ft" if args.solver == "jacobi" else f"_ft_{args.solver}")
pde = FastStencilPDE(args.N, equation=args.equation, b_vec=(args.b_vel, args.b_vel))
solver = make_solver(pde, args.solver)
corr = DeepONetCorrector(f"{args.ckp_dir}/{base_tag}_best.pth", threads=8)
ckp = torch.load(f"{args.ckp_dir}/{base_tag}_best.pth", map_location="cpu", weights_only=False)
target_scale = ckp["target_scale"]

# ---------------------------------------------------------------- collection
# log-spaced snapshots covering the whole rollout (early phase densely, late
# phase sparsely) so the corrector sees the full residual distribution that a
# deployed hybrid solve visits -- at fine grids trajectories run into the
# thousands of iterations and the late-phase states matter
snap_iters = sorted(set(
    list(range(1, 12)) +
    np.unique(np.round(np.geomspace(12, max(args.rollout_iters, 13),
                                    24)).astype(int)).tolist()))
snap_iters = [s for s in snap_iters if s <= args.rollout_iters]
print(f"collecting residuals: {args.n_col} instances x {len(snap_iters)} snapshots "
      f"(explore p_no={args.p_no})")
t0 = time.time()
grf = GRF2D(args.N, rng=rng)
f_col = grf.sample(args.n_col)
u_star = pde.solve_direct(f_col)

X, Y = [], []
u = np.zeros_like(f_col)
for it in range(1, args.rollout_iters + 1):
    r = pde.residual(u, f_col)
    if args.collect_policy == "oracle":
        # advance with the paper's pure error-greedy (Alg. 1), with p_no
        # exploration so nearby off-policy states are also covered
        u_c = solver.step(u, f_col, r)
        u_n = u + corr.correct(r)
        e_c = l2(demean(u_c - u_star))
        e_n = l2(demean(u_n - u_star))
        pick_no = (e_n < e_c)
        flip = rng.random(args.n_col) < args.p_no
        pick_no = np.where(flip, ~pick_no, pick_no)
        u = np.where(pick_no[:, None, None], u_n, u_c)
    else:
        pick_no = rng.random(args.n_col) < args.p_no
        if pick_no.any():
            u[pick_no] = u[pick_no] + corr.correct(r[pick_no])
        if (~pick_no).any():
            u[~pick_no] = solver.step(u[~pick_no], f_col[~pick_no], r[~pick_no])
    if it in snap_iters:
        r_now = pde.residual(u, f_col)
        e_now = demean(u_star - u)
        rn = l2(r_now)[:, None, None]
        keep = (rn[:, 0, 0] > 1e-13 * l2(f_col))  # skip converged instances
        if keep.any():
            X.append((r_now[keep] / rn[keep]).reshape(int(keep.sum()), -1).astype(np.float32))
            Y.append((e_now[keep] / rn[keep] * target_scale).reshape(int(keep.sum()), -1).astype(np.float32))
        else:
            break  # every instance converged; later snapshots are empty too
X = np.concatenate(X)
Y = np.concatenate(Y)
print(f"  collected {len(X)} residual pairs in {time.time()-t0:.0f}s")

# original pairs (same normalization) regenerated with the training seed
grf0 = GRF2D(args.N, rng=np.random.default_rng(1234))
f0 = grf0.sample(args.n_orig)
u0 = pde.solve_direct(f0)
fn0 = l2(f0)[:, None, None]
X0 = (f0 / fn0).reshape(args.n_orig, -1).astype(np.float32)
Y0 = (u0 / fn0 * target_scale).reshape(args.n_orig, -1).astype(np.float32)

Xa = np.concatenate([X, X0])
Ya = np.concatenate([Y, Y0])
perm = np.random.default_rng(0).permutation(len(Xa))
Xa, Ya = Xa[perm], Ya[perm]
n_val = 1000
X_tr, Y_tr = torch.tensor(Xa[n_val:]), torch.tensor(Ya[n_val:])
X_va, Y_va = torch.tensor(Xa[:n_val]), torch.tensor(Ya[:n_val])

# ---------------------------------------------------------------- fine-tune
device = torch.device(args.device)
a = ckp["args"]
model = DeepONet(N=a["N"], dim=2, device=device, in_channels=1, boundary="Periodic",
                 branch_dim=a["branch_dim"], hidden_branch=a["hidden"],
                 num_branch_layers=a["layers"], hidden_trunk=a["hidden"],
                 num_trunk_layers=a["layers"]).to(device)
model.load_state_dict(ckp["model"])
X_va, Y_va = X_va.to(device), Y_va.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(args.epochs // 2, 1), gamma=0.3)


def val_rel():
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X_va), 512):
            pred = model(X_va[i:i + 512]).reshape(-1, args.N * args.N)
            tgt = Y_va[i:i + 512]
            errs.append((torch.linalg.norm(pred - tgt, dim=1)
                         / torch.linalg.norm(tgt, dim=1)).cpu())
    return torch.cat(errs).median().item()


best = np.inf
n_tr = len(X_tr)
nb = int(np.ceil(n_tr / args.batch_size))
for ep in range(args.epochs):
    model.train()
    perm = torch.randperm(n_tr)
    for b in range(nb):
        idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
        xb, yb = X_tr[idx].to(device), Y_tr[idx].to(device)
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((model(xb).reshape(len(idx), -1) - yb) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()
    if ep % 10 == 0 or ep == args.epochs - 1:
        v = val_rel()
        mark = ""
        if v < best:
            best = v
            torch.save({"model": model.state_dict(), "args": a, "target_scale": target_scale},
                       f"{args.ckp_dir}/{out_tag}_best.pth")
            mark = " *"
        print(f"  ep {ep:4d} val_relL2(mixed) {v:.4f}{mark}", flush=True)

with open(f"{args.ckp_dir}/{out_tag}_meta.json", "w") as fh:
    json.dump({"best_val_median_relL2": best, "n_residual_pairs": int(len(X)), **vars(args)}, fh, indent=1)
print(f"[{out_tag}] done, best mixed val rel L2 {best:.4f}")
