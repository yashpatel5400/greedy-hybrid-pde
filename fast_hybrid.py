"""Hybrid solver rollouts with wall-clock accounting.

All methods share identical accounting rules:
  * charged per iteration: residual computation + the update that is actually
    executed (+ router feature/inference cost for the learned router);
  * NOT charged: ground-truth error diagnostics (benchmark-only), and the
    oracle-greedy decision itself (the oracle is an idealized upper bound --
    the learned router row is the deployable method and pays for its decisions).

The DeepONet is applied scale-equivariantly:  du = ||r|| * net(r / ||r||) / s,
with the constant s stored in the checkpoint at training time. This preserves
the zero-preservation property (C(0) = 0) assumed by Proposition 4.2.
"""

import time

import numpy as np
import torch

from fast_pde import FastStencilPDE, demean, l2
from ml_solver import DeepONet


class DeepONetCorrector:
    """Loads a train_fast_deeponet.py checkpoint; caches the trunk basis
    (the grid is fixed, so the trunk MLP never needs re-evaluation) and applies
    the scale-equivariant correction."""

    def __init__(self, ckp_path, device="cpu", threads=1):
        torch.set_num_threads(threads)
        ckp = torch.load(ckp_path, map_location="cpu", weights_only=False)
        a = ckp["args"]
        dev = torch.device(device)
        model = DeepONet(N=a["N"], dim=2, device=dev, in_channels=1, boundary="Periodic",
                         branch_dim=a["branch_dim"], hidden_branch=a["hidden"],
                         num_branch_layers=a["layers"], hidden_trunk=a["hidden"],
                         num_trunk_layers=a["layers"]).to(dev)
        model.load_state_dict(ckp["model"])
        model.eval()
        with torch.no_grad():
            trunk = model.trunk_net(model.coords)          # (N^2, branch_dim)
        self.trunk_T = trunk.transpose(0, 1).contiguous().to(dev)  # (branch_dim, N^2)
        self.branch = model.branch_net
        self.inv_scale = 1.0 / ckp["target_scale"]
        self.N = a["N"]
        self.device = dev

    def correct(self, r):
        """r: (B, N, N) float64 residual -> additive correction du (B, N, N)."""
        B = r.shape[0]
        rn = np.sqrt((r ** 2).sum(axis=(-2, -1), keepdims=True))
        rn = np.maximum(rn, 1e-300)
        x = torch.from_numpy((r / rn).reshape(B, -1).astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = self.branch(x) @ self.trunk_T
        du = out.cpu().numpy().astype(np.float64).reshape(r.shape) * rn * self.inv_scale
        return du - du.mean(axis=(-2, -1), keepdims=True)


# ---------------------------------------------------------------------------
# Rollouts (single instance, per-iteration wall-clock accounting)
# ---------------------------------------------------------------------------

def _make_trace():
    return {"t": [], "rel_res": [], "rel_err": [], "decision": []}


def run_rollout(pde: FastStencilPDE, solver, f, u_truth, policy, corrector=None,
                router=None, max_iters=100000, time_cap=180.0, res_floor=1e-10,
                op_costs=None):
    """Generic single-instance rollout.

    policy: 'classical' | 'oracle' | 'oracle_ca' | 'hints<tau>' | 'router'
      oracle    - greedy on one-step error reduction (paper's Alg. 1)
      oracle_ca - cost-aware greedy: argmax_j log(||e||/||e_j||) / cost_j,
                  with per-op costs from `op_costs` = (t_classical, t_no)
    Returns a trace with per-iteration cumulative charged time, relative
    residual (at iterate BEFORE this iteration's update) and true relative error.
    """
    N = pde.N
    u = np.zeros((1, N, N))
    fn = float(l2(f)[0])
    un = float(l2(demean(u_truth))[0])
    trace = _make_trace()
    t_cum = 0.0
    hints_tau = int(policy[5:]) if policy.startswith("hints") else None
    if router is not None:
        router.reset()
    prev_decision = 0

    for it in range(max_iters):
        # residual + convergence check (charged; needed by every method)
        t0 = time.perf_counter()
        r = pde.residual(u, f)
        rel_res = float(l2(r)[0]) / fn
        t_cum += time.perf_counter() - t0

        # ground-truth diagnostic (not charged)
        rel_err = float(l2(demean(u - u_truth))[0]) / un

        # record the state seen by the stopping test, at the time it is seen
        trace["t"].append(t_cum)
        trace["rel_res"].append(rel_res)
        trace["rel_err"].append(rel_err)
        if rel_res <= res_floor or t_cum > time_cap:
            trace["decision"].append(-1)
            break

        if policy == "classical":
            decision = 0
        elif policy in ("oracle", "oracle_ca"):
            decision = None  # decided below from true errors
        elif hints_tau is not None:
            decision = 1 if (it + 1) % hints_tau == 0 else 0
        elif policy == "router":
            t0 = time.perf_counter()
            decision = router.decide(r, rel_res, it, prev_decision)
            t_cum += time.perf_counter() - t0
        else:
            raise ValueError(policy)

        if policy in ("oracle", "oracle_ca"):
            t0 = time.perf_counter()
            u_c = solver.step(u, f, r)
            t_c = time.perf_counter() - t0
            t0 = time.perf_counter()
            u_n = u + corrector.correct(r)
            t_n = time.perf_counter() - t0
            e_c = float(l2(demean(u_c - u_truth))[0])
            e_n = float(l2(demean(u_n - u_truth))[0])
            if policy == "oracle":
                pick_no = e_n < e_c
            else:
                # error-reduction rate per unit time, with stable cost estimates
                cc, cn = op_costs if op_costs is not None else (t_c, t_n)
                e_prev = float(l2(demean(u - u_truth))[0])
                tiny = 1e-300
                gain_c = np.log(max(e_prev, tiny) / max(e_c, tiny)) / cc
                gain_n = np.log(max(e_prev, tiny) / max(e_n, tiny)) / cn
                pick_no = gain_n > gain_c
            if pick_no:
                u, t_cum, decision = u_n, t_cum + t_n, 1
            else:
                u, t_cum, decision = u_c, t_cum + t_c, 0
        else:
            t0 = time.perf_counter()
            if decision == 1:
                u = u + corrector.correct(r)
            else:
                u = solver.step(u, f, r)
            t_cum += time.perf_counter() - t0

        prev_decision = decision
        trace["decision"].append(decision)
    for k in trace:
        trace[k] = np.asarray(trace[k])
    return trace


def time_to_tol(trace, tols, criterion="rel_res"):
    """First cumulative charged time at which the criterion drops below tol.
    Returns dict tol -> (time, iterations) with (inf, inf) if never reached."""
    vals = trace[criterion]
    out = {}
    for tol in tols:
        idx = np.nonzero(vals <= tol)[0]
        if len(idx) == 0:
            out[tol] = (np.inf, np.inf)
        else:
            i = int(idx[0])
            out[tol] = (float(trace["t"][i]), i)
    return out
