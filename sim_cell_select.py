"""Offline (untimed) cell simulator: rolls oracle and HINTS-25 trajectories on
VALIDATION instances (seed 7) and charges both from measured per-arm constants.
Used to select the shipping corrector per cell without touching the test set."""
import sys, json
import numpy as np
from fast_pde import FastStencilPDE, GRF2D, demean, l2, make_solver
from fast_hybrid import DeepONetCorrector

eq, spec, ckp = sys.argv[1], sys.argv[2], sys.argv[3]
COSTS = json.load(open('results_wallclock_ph3/%s_127_%s.json' % (eq, spec)))['results'][spec]['op_costs']
cc, cn = COSTS['classical'], COSTS['no']
pde = FastStencilPDE(127, equation=eq)
solver = make_solver(pde, spec)
corr = DeepONetCorrector(ckp, threads=8)
grf = GRF2D(127, rng=np.random.default_rng(7))
f = grf.sample(24); u_star = pde.solve_direct(f)
un = l2(demean(u_star)); tiny = 1e-300
TOLS = [6.2e-5, 6.2e-6, 1e-6]

def roll(policy, max_it=60000):
    u = np.zeros_like(f); t = np.zeros(24); done = {tol: np.full(24, np.inf) for tol in TOLS}
    for it in range(max_it):
        r = pde.residual(u, f)
        rel = l2(demean(u - u_star)) / un
        for tol in TOLS:
            hit = (rel <= tol) & ~np.isfinite(done[tol])
            done[tol][hit] = t[hit]
        if all(np.isfinite(done[t_]).all() for t_ in TOLS): break
        if policy == 'oracle':
            u_c = solver.step(u, f, r); u_n = u + corr.correct(r)
            e_c = l2(demean(u_c - u_star)); e_n = l2(demean(u_n - u_star))
            pick = e_n < e_c
        else:
            pick = np.full(24, (it + 1) % 25 == 0)
            u_c = solver.step(u, f, r)
            u_n = (u + corr.correct(r)) if pick.any() else u_c
        u = np.where(pick[:, None, None], u_n, u_c)
        t = t + np.where(pick, cn, cc)
    return done

o = roll('oracle'); h = roll('hints')
for tol in TOLS:
    ratio = np.median(h[tol] / np.maximum(o[tol], tiny))
    print(f"  {eq} {spec} tol {tol:.1e}: sim paired hints/oracle = {ratio:.3f}")
