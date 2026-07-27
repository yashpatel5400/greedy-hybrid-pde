"""Recompute all policies' time-to-tolerance under uniform work-unit accounting
(measured per-arm constants x executed ops) from stored bench rows.
oracle rows are already constants-charged; classical = iters*cc;
hints = iters*cc + floor(iters/tau)*(cn-cc). Also prints live-vs-WU deltas for
the live-timed rows (machine-noise diagnostic)."""
import glob, json, sys
import numpy as np

pat = sys.argv[1] if len(sys.argv) > 1 else 'results_wallclock_final/*.json'
TOLS = [("1e-3",1e-3),("h2",6.2e-5),("h2/10",6.2e-6),("1e-6",1e-6),("1e-8",1e-8)]
for path in sorted(glob.glob(pat)):
    dd = json.load(open(path)); spec = list(dd['results'].keys())[0]
    d = dd['results'][spec]; eq = dd['args']['equation']
    cc, cn = d['op_costs']['classical'], d['op_costs']['no']
    def stats(pol, tolf):
        rows = d[pol]
        for k in rows[0]['err']:
            if abs(float(k)-tolf) < 0.03*tolf:
                t = np.array([np.inf if r['err'][k][0]=='inf' else r['err'][k][0] for r in rows])
                i = np.array([np.inf if r['err'][k][1]=='inf' else r['err'][k][1] for r in rows])
                return t, i
    line = []
    for nm, tolf in TOLS:
        _, ic = stats('classical', tolf)
        th_live, ih = stats('hints25', tolf)
        to, io = stats('oracle', tolf)
        wu_h = ih*cc + np.floor(ih/25.0)*(cn-cc)
        wu_c = ic*cc
        line.append(f"{nm}:{np.median(wu_h/to):.2f}x")
    # live-vs-WU delta for hints at 1e-6 (noise diagnostic)
    th_live, ih = stats('hints25', 1e-6)
    delta = np.median(th_live/(ih*cc + np.floor(ih/25.0)*(cn-cc)))
    print(f"{eq:9s} {spec:12s} WU-paired hints/oracle: " + " ".join(line) +
          f"   [hints live/WU at 1e-6: {delta:.2f}]")
