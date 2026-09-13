"""Compare the larger-capacity ensemble routers (results_ens_big/, run_phase6.sh)
with the default ones (results/): median time-to-h^2 in work units and live
time, the paired per-instance ratio, and the gap to the cost-aware oracle.

  python compare_ens_big.py
"""

import glob
import json
import os

import numpy as np


def med_tt(rows, key, tol):
    return np.array([r["tol"][tol][key] if r["tol"][tol][key] is not None else np.inf for r in rows])


for path in sorted(glob.glob("results_ens_big/*_ens_*.json")):
    base = os.path.join("results", os.path.basename(path))
    if not os.path.exists(base):
        continue
    big, ref = json.load(open(path)), json.load(open(base))
    tol = f"{big['h2']:.6g}"
    for grp in big["groups"]:
        gb, gr = big["groups"][grp]["policies"], ref["groups"][grp]["policies"]
        out = [os.path.basename(path)[:-5]]
        for key in ("t_wu", "t_live"):
            b, r, o = (med_tt(gb["router"], key, tol), med_tt(gr["router"], key, tol), med_tt(gr["oracle"], key, tol))
            out.append(f"{key}: big {np.median(b)*1e3:.3f} ms | base {np.median(r)*1e3:.3f} ms | oracle {np.median(o)*1e3:.3f} ms"
                       f" | paired base/big {np.median(r / b):.3f} | big/oracle {np.median(b / o):.3f}")
        miss_b = int(np.sum(~np.isfinite(med_tt(gb["router"], "t_wu", tol))))
        miss_r = int(np.sum(~np.isfinite(med_tt(gr["router"], "t_wu", tol))))
        out.append(f"missed h2: big {miss_b}, base {miss_r}")
        print("\n  ".join(out))
