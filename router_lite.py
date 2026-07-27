"""Lightweight learned router for wall-clock-efficient hybrid solving.

The paper's LSTM router consumes the full field state (O(N^2 * hidden) work per
decision, ~1 ms at N=63), which can exceed the cost of the classical iteration
it is routing. This router uses five *scalar* observable features and a 2x32
MLP evaluated in pure numpy (~5 us per decision, independent of N), and is
trained with the paper's cost-weighted surrogate recipe, with costs measured in
*error-reduction rate per second*, so the imitated policy is the cost-aware
greedy oracle. A DAgger round (roll out the router, relabel with the oracle)
removes exposure bias.

Features per step (all observable; no ground-truth access; all O(1) given the
residual norm, which every stopping test already computes):
  1. log10 relative residual
  2. one-step change of log10 residual (progress-rate signal)
  3. EMA of (2) (smoothed progress rate)
  4. log(1 + iteration index)
  5. previous routing decision
  6. log(1 + number of NO calls so far)
  7. log(1 + iterations since the last NO call)
"""

import argparse
import math
import time

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D, demean, l2, make_solver

FEAT_DIM = 7
EMA = 0.7


class LiteRouter:
    """Numpy-inference MLP: FEAT_DIM -> 32 -> 32 -> 2."""

    def __init__(self, weights):
        self.weights = [np.ascontiguousarray(w, dtype=np.float64) for w in weights]
        self.reset()

    def reset(self):
        self._prev_log_res = None
        self._ema = 0.0
        self._n_no = 0
        self._since_no = 10 ** 4
        self._x = np.zeros(FEAT_DIM)

    def features(self, rel_res, it, prev_decision):
        lr = math.log10(rel_res) if rel_res > 1e-16 else -16.0
        if self._prev_log_res is None:
            dlr = 0.0
        else:
            dlr = max(-2.0, min(2.0, lr - self._prev_log_res))
        self._ema = EMA * self._ema + (1 - EMA) * dlr
        self._prev_log_res = lr
        x = self._x
        x[0] = lr / 8.0
        x[1] = dlr
        x[2] = self._ema
        x[3] = math.log1p(it) / 8.0
        x[4] = prev_decision
        x[5] = math.log1p(self._n_no) / 3.0
        x[6] = math.log1p(min(self._since_no, 9999)) / 8.0
        return x

    def decide(self, r, rel_res, it, prev_decision):
        """Single-instance decision used inside run_rollout (r is unused;
        the router is O(1) given the residual norm)."""
        x = self.features(rel_res, it, prev_decision)
        W1, b1, W2, b2, W3, b3 = self.weights
        h = np.maximum(x @ W1 + b1, 0.0, )
        h = np.maximum(h @ W2 + b2, 0.0)
        o = h @ W3 + b3
        d = int(o[1] > o[0])
        if d:
            self._n_no += 1
            self._since_no = 0
        else:
            self._since_no += 1
        return d

    def save(self, path):
        torch.save({"weights": [torch.tensor(w) for w in self.weights]}, path)

    @classmethod
    def load(cls, path):
        ckp = torch.load(path, map_location="cpu", weights_only=False)
        return cls([w.numpy() for w in ckp["weights"]])


def batch_features(lr, dlr, ema, it, prev_dec, n_no, since_no):
    return np.stack([lr / 8.0, np.clip(dlr, -2, 2), ema,
                     np.full_like(lr, math.log1p(it) / 8.0), prev_dec,
                     np.log1p(n_no) / 3.0,
                     np.log1p(np.minimum(since_no, 9999)) / 8.0], axis=-1)


class _TorchMLP(torch.nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(FEAT_DIM, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)

    def export_weights(self):
        ws = []
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                ws += [m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()]
        return ws


# ---------------------------------------------------------------------------
# Training: batched rollouts labeled by the cost-aware oracle
# ---------------------------------------------------------------------------

def collect_states(pde, solver, corrector, n_inst, op_costs, rng, max_roll=6000,
                   eps=0.15, res_floor=1e-10, behavior=None):
    """Roll out batched hybrid solves; label every visited state with the
    cost-aware oracle decision and the wall-clock regret weight.

    behavior: None -> eps-greedy around the oracle; LiteRouter -> follow the
    router's own policy (DAgger round).
    """
    grf = GRF2D(pde.N, rng=rng)
    f = grf.sample(n_inst)
    u_star = pde.solve_direct(f)
    fn = l2(f)
    u = np.zeros_like(f)
    prev_log = None
    ema = np.zeros(n_inst)
    prev_dec = np.zeros(n_inst)
    n_no = np.zeros(n_inst)
    since_no = np.full(n_inst, 1e4)
    cc, cn = op_costs
    tiny = 1e-300
    X, rows = [], []
    active = np.ones(n_inst, dtype=bool)
    for it in range(max_roll):
        r = pde.residual(u, f)
        rel = l2(r) / fn
        active = active & (rel > res_floor)
        if not active.any():
            break
        lr = np.log10(np.maximum(rel, 1e-16))
        dlr = np.zeros(n_inst) if prev_log is None else np.clip(lr - prev_log, -2, 2)
        ema = EMA * ema + (1 - EMA) * dlr
        feats = batch_features(lr, dlr, ema, it, prev_dec, n_no, since_no)
        prev_log = lr

        u_c = solver.step(u, f, r)
        u_n = u + corrector.correct(r)
        e_prev = l2(demean(u - u_star))
        e_c = l2(demean(u_c - u_star))
        e_n = l2(demean(u_n - u_star))
        rate_c = np.log(np.maximum(e_prev, tiny) / np.maximum(e_c, tiny)) / cc
        rate_n = np.log(np.maximum(e_prev, tiny) / np.maximum(e_n, tiny)) / cn
        label = (rate_n > rate_c).astype(np.float64)
        # relative regret: bounded and symmetric between go/stop mistakes
        w = np.abs(rate_n - rate_c) / (np.maximum(np.abs(rate_n), np.abs(rate_c)) + 1e-12)

        record = active & ((it < 300) | (it % 10 == 0))
        if record.any():
            X.append(np.concatenate([feats[record], label[record, None],
                                     w[record, None]], axis=1))
        if behavior is None:
            act = np.where(rng.random(n_inst) < eps, 1 - label, label).astype(bool)
        else:
            W1, b1, W2, b2, W3, b3 = behavior.weights
            h = np.maximum(feats @ W1 + b1, 0.0)
            h = np.maximum(h @ W2 + b2, 0.0)
            o = h @ W3 + b3
            act = o[:, 1] > o[:, 0]
        u = np.where(act[:, None, None], u_n, u_c)
        prev_dec = act.astype(np.float64)
        n_no = n_no + prev_dec
        since_no = np.where(act, 0.0, since_no + 1.0)
    data = np.concatenate(X)
    return data[:, :FEAT_DIM], data[:, FEAT_DIM], data[:, FEAT_DIM + 1]


def train_router(feats, labels, weights, epochs=80, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = _TorchMLP()
    w = np.clip(weights, 0.0, 2.0)
    X = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    wt = torch.tensor(w, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, 4096):
            idx = perm[b:b + 4096]
            opt.zero_grad()
            logits = model(X[idx])
            loss = (torch.nn.functional.cross_entropy(logits, y[idx], reduction="none")
                    * wt[idx]).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 20 == 0 or ep == epochs - 1:
            with torch.no_grad():
                pred = model(X).argmax(1)
                acc = (pred == y).float().mean().item()
                wacc = (wt * (pred == y).float()).sum().item() / wt.sum().item()
            print(f"    router ep {ep}: loss {tot/n:.4f} acc {acc:.3f} w-acc {wacc:.3f}",
                  flush=True)
    return LiteRouter(model.export_weights())


def fit(equation, N, solver_spec, ckp_path, n_inst=192, max_roll=6000, b_vel=20.0,
        seed=555, dagger_rounds=2, ckp_dir="./checkpoints"):
    from fast_hybrid import DeepONetCorrector

    pde = FastStencilPDE(N, equation=equation, b_vec=(b_vel, b_vel))
    solver = make_solver(pde, solver_spec)
    corr = DeepONetCorrector(ckp_path, threads=8)

    # per-ITERATION arm costs at the boundaries run_rollout charges (residual is
    # paid every iteration by both arms; classical step gets the precomputed r)
    warm = np.zeros((1, N, N))
    fwarm = GRF2D(N, rng=np.random.default_rng(1)).sample(1)
    rwarm = pde.residual(warm, fwarm)
    _ = solver.step(warm, fwarm, rwarm); _ = corr.correct(rwarm)
    reps_r, reps_c, reps_n = [], [], []
    for _k in range(40):
        t0 = time.perf_counter(); rwarm = pde.residual(warm, fwarm); reps_r.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); _ = solver.step(warm, fwarm, rwarm); reps_c.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); _ = warm + corr.correct(rwarm); reps_n.append(time.perf_counter() - t0)
    t_res = float(np.median(reps_r))
    costs = (t_res + float(np.median(reps_c)), t_res + float(np.median(reps_n)))
    print(f"per-iteration arm costs: classical {costs[0]*1e6:.0f}us NO {costs[1]*1e6:.0f}us "
          f"(residual {t_res*1e6:.0f}us)")

    rng = np.random.default_rng(seed)
    t0 = time.time()
    F, L, W = collect_states(pde, solver, corr, n_inst, costs, rng, max_roll=max_roll)
    print(f"round 0: {len(F)} states ({L.mean()*100:.2f}% NO-preferred) "
          f"in {time.time()-t0:.0f}s", flush=True)
    router = train_router(F, L, W)
    for rd in range(dagger_rounds):
        F2, L2, W2 = collect_states(pde, solver, corr, n_inst, costs, rng,
                                    max_roll=max_roll, behavior=router)
        F, L, W = (np.concatenate([F, F2]), np.concatenate([L, L2]),
                   np.concatenate([W, W2]))
        print(f"DAgger round {rd+1}: +{len(F2)} on-policy states "
              f"({L2.mean()*100:.2f}% NO-preferred)", flush=True)
        router = train_router(F, L, W)
    out = f"{ckp_dir}/lite_router_{equation}_{N}_{solver_spec}.pth"
    router.save(out)
    print("saved", out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--equation", type=str, default="Poisson")
    p.add_argument("--N", type=int, default=63)
    p.add_argument("--solver", type=str, default="jacobi")
    p.add_argument("--ckp", type=str, default=None)
    p.add_argument("--n_inst", type=int, default=192)
    p.add_argument("--max_roll", type=int, default=6000)
    p.add_argument("--b_vel", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=555)
    p.add_argument("--dagger_rounds", type=int, default=2)
    p.add_argument("--ckp_dir", type=str, default="./checkpoints")
    a = p.parse_args()
    ckp = a.ckp or f"{a.ckp_dir}/fast_deeponet_{a.equation}_{a.N}_ft_best.pth"
    fit(a.equation, a.N, a.solver, ckp, n_inst=a.n_inst, max_roll=a.max_roll,
        b_vel=a.b_vel, seed=a.seed, dagger_rounds=a.dagger_rounds, ckp_dir=a.ckp_dir)
