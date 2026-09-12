"""Learned cost-aware router: features -> macro-action, trained with the paper's
cost-weighted surrogate (Eq. 10) to imitate the cost-aware greedy oracle.

Training recipe (Section 5 of the paper, adapted to macro-actions):
  * roll out hybrid solves on training instances; at every decision epoch
    record the router features and, for every macro-action j, the true error
    after applying it, c~_j = ||e after j||^2 (exact, from the FFT ground
    truth);
  * minimise Psi(g, x) = - sum_j w_j log softmax_j(g(x)) with
    w_j = sum_{k != j} c~_k / sum_k c~_k (the per-state normalisation does not
    change the Bayes decision, see the remark after Theorem 5.1);
  * scheduled sampling / DAgger: round 0 collects states along oracle rollouts
    with epsilon-exploration, later rounds roll out the current router and
    relabel its own states with the oracle (removes exposure bias).
The router is a 2-hidden-layer MLP evaluated in numpy at deployment (a few
microseconds per decision).
"""

import argparse
import math
import time

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D, demean, l2
from hybrid import Env, FeatureState, EMA, measure_costs, n_features


class Router:
    """numpy-inference MLP: F -> H -> H -> K."""

    def __init__(self, weights, K):
        self.weights = [np.ascontiguousarray(w, dtype=np.float64) for w in weights]
        self.K = K

    def reset(self):
        pass

    def logits(self, x):
        W1, b1, W2, b2, W3, b3 = self.weights
        h = np.maximum(x @ W1 + b1, 0.0)
        h = np.maximum(h @ W2 + b2, 0.0)
        return h @ W3 + b3

    def decide(self, x):
        return int(np.argmax(self.logits(x)))

    def decide_batch(self, X):
        W1, b1, W2, b2, W3, b3 = self.weights
        h = np.maximum(X @ W1 + b1, 0.0)
        h = np.maximum(h @ W2 + b2, 0.0)
        return np.argmax(h @ W3 + b3, axis=1)

    def save(self, path, meta=None):
        torch.save({"weights": [torch.tensor(w) for w in self.weights], "K": self.K,
                    "meta": meta or {}}, path)

    @classmethod
    def load(cls, path):
        ckp = torch.load(path, map_location="cpu", weights_only=False)
        r = cls([w.numpy() for w in ckp["weights"]], ckp["K"])
        r.meta = ckp.get("meta", {})
        return r


class _TorchMLP(torch.nn.Module):
    def __init__(self, F, K, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(F, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, K))

    def forward(self, x):
        return self.net(x)

    def export(self):
        ws = []
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                ws += [m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()]
        return ws


# ---------------------------------------------------------------------------
# Batched data collection
# ---------------------------------------------------------------------------

class BatchFeatureState:
    """Vectorised FeatureState over B instances (same feature definition)."""

    def __init__(self, B, K, no_index):
        self.B, self.K, self.no_index = B, K, no_index
        self.prev_log = None
        self.prev_ops = np.ones(B)
        self.ema = np.zeros(B)
        self.prev_action = -np.ones(B, dtype=int)
        self.n_no = np.zeros(B)
        self.since_no = np.full(B, 1e4)
        self.epoch = 0

    def features(self, rel_res):
        lr = np.log10(np.maximum(rel_res, 1e-300))
        dlr = np.zeros(self.B) if self.prev_log is None else np.clip((lr - self.prev_log) / self.prev_ops, -2, 2)
        self.ema = EMA * self.ema + (1 - EMA) * dlr
        self.prev_log = lr
        X = np.zeros((self.B, n_features(self.K)))
        X[:, 0] = lr / 8.0
        X[:, 1] = dlr
        X[:, 2] = self.ema
        X[:, 3] = math.log1p(self.epoch) / 8.0
        has = self.prev_action >= 0
        X[np.flatnonzero(has), 4 + self.prev_action[has]] = 1.0
        X[:, 4 + self.K] = np.log1p(self.n_no) / 3.0
        X[:, 5 + self.K] = np.log1p(np.minimum(self.since_no, 9999)) / 8.0
        return X

    def update(self, action, n_ops):
        self.prev_action = action.copy()
        self.prev_ops = n_ops.astype(float)
        self.epoch += 1
        is_no = action == self.no_index
        self.n_no = self.n_no + np.where(is_no, n_ops, 0)
        self.since_no = np.where(is_no, 0.0, self.since_no + n_ops)


def collect_states(env: Env, n_inst, rng, behavior=None, eps=0.1, max_epochs=3000,
                   err_stop=1e-9, res_floor=1e-13, record_every=1, rate=False):
    """Batched rollouts. Returns X (n, F), C (n, K) = true squared errors after
    each macro-action, and A (n,) = action taken by the behaviour policy.
    behavior None -> oracle with eps-exploration; Router -> DAgger (router acts,
    oracle labels). With rate=True the rollout advances one operation at a
    time and the labels are the cost-adjusted errors ||e_j||^(2 c_max/c_j)
    (per-iteration form of the cost-aware rule)."""
    pde, K = env.pde, env.K
    grf = GRF2D(env.N, rng=rng)
    f = grf.sample(n_inst)
    u_star = pde.solve_direct(f)
    fn = l2(f)
    un = l2(demean(u_star))
    u = np.zeros_like(f)
    fs = BatchFeatureState(n_inst, K, env.no_index)
    X, C, A = [], [], []
    active = np.ones(n_inst, dtype=bool)
    for ep in range(max_epochs):
        r = pde.residual(u, f)
        rel_res = l2(r) / fn
        rel_err = l2(demean(u - u_star)) / un
        active &= (rel_err > err_stop) & (rel_res > res_floor)
        if not active.any():
            break
        feats = fs.features(rel_res)
        # all (macro-)action outcomes (batched)
        cands = []
        for j in range(K):
            uj = env.apply_op(j, u, f, r) if rate else env.apply_macro(j, u, f, r)
            cands.append(uj)
        errs2 = np.stack([l2(demean(uj - u_star)) ** 2 for uj in cands], axis=1)  # (B, K)
        if rate:
            # cost-adjusted errors, computed in the log domain relative to the
            # current error so the per-state normalisation stays well scaled
            e0 = np.maximum(rel_err * un, 1e-300)[:, None]
            logratio = np.log(np.maximum(errs2, 1e-300) / e0 ** 2)          # log(e_j^2/e^2) <= 0
            errs2 = np.exp(np.array(env.rate_exp)[None, :] * logratio)      # (e_j/e)^(2 c_max/c_j)
        # normalise per state (scale-free) -- keeps weights O(1)
        lab_err = errs2 / np.maximum(errs2.sum(axis=1, keepdims=True), 1e-300)
        if ep % record_every == 0:
            X.append(feats[active])
            C.append(lab_err[active])
        oracle = np.argmin(errs2, axis=1)
        if behavior is None:
            act = oracle.copy()
            flip = rng.random(n_inst) < eps
            act[flip] = rng.integers(K, size=flip.sum())
        else:
            act = behavior.decide_batch(feats)
        if ep % record_every == 0:
            A.append(act[active])
        u = np.stack([cands[act[b]][b] for b in range(n_inst)])
        m = np.ones(n_inst, dtype=int) if rate else np.array([env.m[a] for a in act])
        fs.update(act, m)
    return np.concatenate(X), np.concatenate(C), np.concatenate(A)


def surrogate_loss(logits, c_norm):
    """Paper's Eq. 10 with per-state normalised c~: w_j = sum_{k!=j} c_k."""
    logp = torch.nn.functional.log_softmax(logits, dim=1)
    w = c_norm.sum(dim=1, keepdim=True) - c_norm
    return -(w * logp).sum(dim=1).mean()


def train_router(X, C, K, epochs=150, lr=2e-3, hidden=64, seed=0, verbose=True):
    torch.manual_seed(seed)
    F = X.shape[1]
    model = _TorchMLP(F, K, hidden)
    Xt = torch.tensor(X, dtype=torch.float32)
    Ct = torch.tensor(C, dtype=torch.float32)
    y = Ct.argmin(dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Xt)
    bs = 2048
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            opt.zero_grad()
            loss = surrogate_loss(model(Xt[idx]), Ct[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            with torch.no_grad():
                pred = model(Xt).argmax(1)
                acc = (pred == y).float().mean().item()
                # regret: normalised error of the chosen action vs the oracle's
                chosen = Ct[torch.arange(n), pred]
                best = Ct.min(dim=1).values
                regret = ((chosen - best) / best.clamp_min(1e-30)).median().item()
            print(f"    router ep {ep}: loss {tot/n:.4f} acc {acc:.3f} median regret {regret:.3f}",
                  flush=True)
    return Router(model.export(), K)


def fit_router(env: Env, n_inst=128, seed=555, dagger_rounds=2, eps=0.1, epochs=150,
               hidden=64, verbose=True, rate=False, max_epochs=3000, err_stop=1e-9):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    X, C, A = collect_states(env, n_inst, rng, behavior=None, eps=eps, rate=rate, max_epochs=max_epochs,
                             err_stop=err_stop)
    frac_no = (C.argmin(1) == env.no_index).mean() if env.no_index is not None else 0.0
    print(f"  round 0: {len(X)} states ({frac_no*100:.1f}% corrector-preferred) in {time.time()-t0:.0f}s",
          flush=True)
    router = train_router(X, C, env.K, epochs=epochs, hidden=hidden, seed=seed, verbose=verbose)
    for rd in range(dagger_rounds):
        X2, C2, A2 = collect_states(env, n_inst, rng, behavior=router, rate=rate, max_epochs=max_epochs,
                                    err_stop=err_stop)
        X, C = np.concatenate([X, X2]), np.concatenate([C, C2])
        agree = (A2 == C2.argmin(1)).mean()
        print(f"  DAgger round {rd+1}: +{len(X2)} on-policy states (agreement with oracle {agree*100:.1f}%)",
              flush=True)
        router = train_router(X, C, env.K, epochs=epochs, hidden=hidden, seed=seed + rd + 1,
                              verbose=verbose)
    return router
