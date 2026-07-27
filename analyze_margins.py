"""Decision analysis for the HINTS comparison, with conservative statistics.

For each setting and tolerance, the claim margin of OURS over HINTS is computed
against EVERY tau in the run and reported for the WORST tau for us, using the
more conservative of two statistics (median of paired ratios; ratio of
medians). Censoring: if OURS never reaches the tolerance but HINTS does, the
pair counts fully against us (ratio 0); the reverse counts as ratio +inf.

A setting 'meets the bar' at a tolerance when the conservative margin over the
worst tau is >= --bar and the bootstrap 95% CI of the worst-tau paired-median
ratio excludes 1.0.
"""

import argparse
import glob
import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", type=str, default="results_wallclock_ph1/*.json")
parser.add_argument("--ours", type=str, default="oracle_ca")
parser.add_argument("--criterion", type=str, default="err")
parser.add_argument("--bar", type=float, default=1.10)
parser.add_argument("--n_boot", type=int, default=4000)
args = parser.parse_args()

rng = np.random.default_rng(0)


def get(rows, key, crit):
    return np.array([np.inf if r[crit][key][0] == "inf" else r[crit][key][0]
                     for r in rows])


def paired_ratios(h, o):
    """Censoring-aware paired ratios HINTS/OURS (higher = better for us)."""
    r = np.empty(len(h))
    both = np.isfinite(h) & np.isfinite(o)
    r[both] = h[both] / o[both]
    r[np.isfinite(h) & ~np.isfinite(o)] = 0.0      # we failed, HINTS didn't
    r[~np.isfinite(h) & np.isfinite(o)] = np.inf   # HINTS failed, we didn't
    r[~np.isfinite(h) & ~np.isfinite(o)] = 1.0     # both failed: tie
    return r


for path in sorted(glob.glob(args.pattern)):
    d = json.load(open(path))
    a = d["args"]
    for spec, pr in d["results"].items():
        if args.ours not in pr:
            continue
        taus = sorted([p for p in pr if p.startswith("hints")],
                      key=lambda s: int(s[5:]))
        if not taus:
            continue
        keys = sorted(pr[args.ours][0][args.criterion].keys(), key=float,
                      reverse=True)
        print(f"\n== {a['equation']} N={a['N']} {spec} | {args.ours} vs HINTS "
              f"taus={[int(t[5:]) for t in taus]} (n={a['n_test']}) ==")
        for k in keys:
            ours = get(pr[args.ours], k, args.criterion)
            stats = {}
            for t in taus:
                h = get(pr[t], k, args.criterion)
                pratio = paired_ratios(h, ours)
                med_paired = np.median(pratio)
                mh, mo = np.median(h), np.median(ours)
                rom = (mh / mo) if np.isfinite(mo) else (1.0 if not np.isfinite(mh) else 0.0)
                stats[t] = (min(med_paired, rom), med_paired, rom, pratio)
            worst_t = min(stats, key=lambda t: stats[t][0])
            cons, medp, rom, pratio = stats[worst_t]
            finite = pratio[np.isfinite(pratio)]
            boots = np.array([np.median(rng.choice(pratio, len(pratio)))
                              for _ in range(args.n_boot)])
            lo, hi = np.quantile(boots, [0.025, 0.975])
            flag = ("MEETS BAR" if (cons >= args.bar and lo > 1.0)
                    else ("close" if cons >= 1.0 else "LOSES"))
            print(f"  tol {float(k):8.2e}: worst-tau={worst_t:9s} conservative "
                  f"x{cons:5.2f} (paired {medp:5.2f} [CI {lo:.2f},{hi:.2f}], "
                  f"ratio-of-medians {rom:5.2f}) | {flag}")
