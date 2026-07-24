"""Aggregate results_wallclock/*.json into summary tables (console + LaTeX rows).

Primary metric: median wall-clock time to reach a target relative L2 *error*
(vs the exact discrete solution), over test instances. Also reports iteration
counts, NO-call counts, and speedup vs the classical-only baseline.
"""

import argparse
import glob
import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", type=str, default="results_wallclock/*.json")
parser.add_argument("--criterion", type=str, default="err", choices=["err", "res"])
parser.add_argument("--tols", type=str, default="1e-4,1e-6,1e-8")
parser.add_argument("--latex", action="store_true")
args = parser.parse_args()

TOL_KEYS = {1e-2: "0.01", 1e-3: "0.001", 1e-4: "0.0001", 1e-5: "1e-05",
            1e-6: "1e-06", 1e-7: "1e-07", 1e-8: "1e-08"}
tols = [float(t) for t in args.tols.split(",")]


def get_times(rows, tol, crit):
    key = TOL_KEYS[tol]
    return np.array([np.inf if r[crit][key][0] == "inf" else r[crit][key][0] for r in rows])


def fmt_t(x):
    if not np.isfinite(x):
        return "--"
    if x < 1e-3:
        return f"{x*1e6:.0f}$\\mu$s" if args.latex else f"{x*1e6:.0f}us"
    if x < 1.0:
        return f"{x*1e3:.1f}ms"
    return f"{x:.2f}s"


for path in sorted(glob.glob(args.pattern)):
    d = json.load(open(path))
    a = d["args"]
    for spec, polres in d["results"].items():
        pols = [p for p in polres if p != "op_costs"]
        if "classical" not in pols:
            continue
        base_rows = polres["classical"]
        print(f"\n== {a['equation']} N={a['N']} solver={spec} "
              f"(n={a['n_test']}, criterion={args.criterion}) ==")
        header = f"{'policy':<12}" + "".join(f"{f'tol={t:g}':>22}" for t in tols) + f"{'NO calls':>10}"
        print(header)
        for pol in pols:
            rows = polres[pol]
            cells = []
            for tol in tols:
                ts = get_times(rows, tol, args.criterion)
                tb = get_times(base_rows, tol, args.criterion)
                med = np.median(ts)
                # paired speedup vs classical on the same instances
                sp = np.median(tb / ts) if np.isfinite(med) else np.nan
                n_conv = int(np.isfinite(ts).sum())
                cell = f"{fmt_t(med)} ({sp:.1f}x)" if np.isfinite(med) else "--"
                if n_conv < len(ts):
                    cell += f" [{n_conv}/{len(ts)}]"
                cells.append(f"{cell:>22}")
            nno = np.median([r["n_no_calls"] for r in rows])
            print(f"{pol:<12}" + "".join(cells) + f"{nno:>10.0f}")
