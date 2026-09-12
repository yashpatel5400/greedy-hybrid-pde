"""Generate the LaTeX tables of the cost-aware wall-clock study from
results/*.json (written by bench.py) into paper/costaware_tables.tex.

Macros:
  \\cawcmain          time to eps = h^2, all solvers x both PDEs (pairwise)
  \\cawctol<eq>       tolerance sweep per solver (appendix)
  \\caauc             AUC / final error over T iterations (paper-style)
  \\causage           corrector-call counts and iteration counts
  \\caens             ensemble results
  \\cacosts           measured per-iteration costs and macro sizes
"""

import glob
import json
import math
import os

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

SOLVER_NAMES = {"jacobi": "Jacobi", "jacobi_0.67": "Jacobi (0.67)", "gs": "GS",
                "ssor": "SymGS", "sor_1.5": "SOR (1.5)"}
SOLVER_ORDER = ["jacobi", "jacobi_0.67", "gs", "ssor", "sor_1.5"]
POL_NAMES = {"classical": "Solver only", "hints25": "HINTS ($\\tau{=}25$)",
             "hints5": "HINTS ($\\tau{=}5$)", "hints10": "HINTS ($\\tau{=}10$)",
             "hints50": "HINTS ($\\tau{=}50$)", "greedy": "Greedy oracle (Alg.~1)",
             "oracle": "Cost-aware oracle", "router": "Learned router (ours)"}
EQS = ["Poisson", "ConvDiff"]


RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
OUT_TEX = os.environ.get("OUT_TEX", "paper/costaware_tables.tex")


def load(pattern=None):
    pattern = pattern or f"{RESULTS_DIR}/*.json"
    R = {}
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path))
        a = d["args"]
        for gkey, g in d["groups"].items():
            R[(a["equation"], a["N"], gkey, bool(a["ensemble"]))] = (d, g)
    return R


def tkey(d, tol):
    for k in d["tols"]:
        if abs(k - tol) <= 1e-9 * max(1.0, tol) or abs(k - tol) < 0.02 * tol:
            return f"{k:.6g}"
    raise KeyError(tol)


def times(rows, key, field="t_live"):
    return np.array([np.inf if r["tol"][key][field] is None else r["tol"][key][field] for r in rows])


def iters(rows, key):
    return np.array([np.inf if r["tol"][key]["iters"] is None else r["tol"][key]["iters"] for r in rows])


def fmt_time(x):
    if not np.isfinite(x):
        return "--"
    if x < 1e-3:
        return f"{x*1e6:.0f}\\,$\\mu$s"
    if x < 1.0:
        return f"{x*1e3:.2f}\\,ms" if x < 0.01 else f"{x*1e3:.1f}\\,ms"
    return f"{x:.2f}\\,s"


def paired_speedup(base, ours):
    """Censoring-aware paired median speedup base/ours (inf-safe)."""
    both = np.isfinite(base) & np.isfinite(ours)
    r = np.full(len(base), np.nan)
    r[both] = base[both] / ours[both]
    r[np.isfinite(base) & ~np.isfinite(ours)] = 0.0
    r[~np.isfinite(base) & np.isfinite(ours)] = np.inf
    r[~np.isfinite(base) & ~np.isfinite(ours)] = 1.0
    return float(np.median(r)), r


def fmt_sp(sp, censored_frac=0.0):
    if not np.isfinite(sp):
        return "$>10^{3}\\times$"
    if sp >= 100:
        s = f"{sp:.0f}$\\times$"
    elif sp >= 10:
        s = f"{sp:.1f}$\\times$"
    else:
        s = f"{sp:.2f}$\\times$"
    return s


def pval_str(p):
    if p < 1e-3:
        return "$<10^{-3}$"
    return f"{p:.3f}"


def cell_time(rows, base_rows, key, bold=False):
    ts = times(rows, key)
    tb = times(base_rows, key)
    med = np.median(ts)
    cens = int((~np.isfinite(ts)).sum())
    if cens > len(ts) / 2:
        return "--"
    sp, _ = paired_speedup(tb, ts)
    s = fmt_time(med)
    if rows is not base_rows:
        s += f" ({fmt_sp(sp)})"
    if cens:
        s += f"$^{{\\dagger {cens}}}$"
    return f"\\textbf{{{s}}}" if bold else s


def main():
    R = load()
    out = []
    N = None

    # ------------------------------------------------------------ main table
    out.append("\\newcommand{\\cawcmain}{")
    out.append("\\begin{tabular}{llcccccc}\n\\toprule")
    out.append("Equation & Solver & Solver only & HINTS ($\\tau{=}25$) & HINTS (best $\\tau$) & "
               "Greedy oracle & Cost-aware oracle & Learned router (ours) \\\\ \\midrule")
    for eq in EQS:
        first = True
        for spec in SOLVER_ORDER:
            keys = [k for k in R if k[0] == eq and k[2] == spec and not k[3]]
            if not keys:
                continue
            d, g = R[keys[0]]
            N = keys[0][1]
            key = tkey(d, d["h2"])
            P = g["policies"]
            base = P["classical"]
            # best fixed tau by median time
            taus = [p for p in P if p.startswith("hints")]
            best_tau = min(taus, key=lambda p: np.median(times(P[p], key)))
            # deployable comparison: router vs every HINTS and classical
            t_r = times(P["router"], key)
            best_dep = "router"
            for p in taus + ["classical"]:
                if np.median(times(P[p], key)) < np.median(t_r):
                    best_dep = p
            row = [eq if first else "", SOLVER_NAMES[spec],
                   cell_time(base, base, key),
                   cell_time(P["hints25"], base, key, bold=(best_dep == "hints25")),
                   cell_time(P[best_tau], base, key, bold=(best_dep == best_tau)).replace(
                       ")", f"; $\\tau{{=}}{best_tau[5:]}$)", 1),
                   cell_time(P["greedy"], base, key) if "greedy" in P else "--",
                   "\\textit{" + cell_time(P["oracle"], base, key) + "}",
                   cell_time(P["router"], base, key, bold=(best_dep == "router"))]
            out.append(" & ".join(row) + " \\\\")
            first = False
        if eq == "Poisson":
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # -------------------------------------------- router vs HINTS speedup table
    out.append("\\newcommand{\\cavshints}{")
    out.append("\\begin{tabular}{llcccccc}\n\\toprule")
    out.append("& & \\multicolumn{3}{c}{vs.\\ HINTS ($\\tau{=}25$)} & \\multicolumn{3}{c}{vs.\\ best fixed $\\tau$} \\\\")
    out.append("\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}")
    out.append("Equation & Solver & $\\varepsilon{=}10^{-3}$ & $\\varepsilon{=}h^2$ & $\\varepsilon{=}10^{-8}$ & "
               "$\\varepsilon{=}10^{-3}$ & $\\varepsilon{=}h^2$ & $\\varepsilon{=}10^{-8}$ \\\\ \\midrule")
    for eq in EQS:
        first = True
        for spec in SOLVER_ORDER:
            keys = [k for k in R if k[0] == eq and k[2] == spec and not k[3]]
            if not keys:
                continue
            d, g = R[keys[0]]
            P = g["policies"]
            taus = [p for p in P if p.startswith("hints")]
            cells = []
            for ref in ["hints25", "best"]:
                for tol in [1e-3, d["h2"], 1e-8]:
                    key = tkey(d, tol)
                    t_r = times(P["router"], key)
                    if ref == "best":
                        bt = min(taus, key=lambda p: np.median(times(P[p], key)))
                        t_h = times(P[bt], key)
                        lab = f" ($\\tau{{=}}{bt[5:]}$)"
                    else:
                        t_h = times(P["hints25"], key)
                        lab = ""
                    sp, r = paired_speedup(t_h, t_r)
                    ok = np.isfinite(r) & (r > 0)
                    p = wilcoxon(np.log(r[ok]), alternative="greater").pvalue if ok.sum() >= 8 and not np.allclose(r[ok], 1.0) else 1.0
                    s = fmt_sp(sp) + lab
                    if sp >= 1.10 and p < 0.01:
                        s = f"\\textbf{{{s}}}"
                    cells.append(s)
            out.append(" & ".join([eq if first else "", SOLVER_NAMES[spec]] + cells) + " \\\\")
            first = False
        if eq == "Poisson":
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ----------------------------------------------------- tolerance sweeps
    for eq in EQS:
        out.append(f"\\newcommand{{\\cawctol{eq.lower().replace('diff','')}}}{{")
        out.append("\\begin{tabular}{llcccccc}\n\\toprule")
        out.append("Solver & Method & $\\varepsilon{=}10^{-2}$ & $\\varepsilon{=}10^{-3}$ & $\\varepsilon{=}h^2$ & "
                   "$\\varepsilon{=}10^{-5}$ & $\\varepsilon{=}10^{-6}$ & $\\varepsilon{=}10^{-8}$ \\\\ \\midrule")
        for spec in SOLVER_ORDER:
            keys = [k for k in R if k[0] == eq and k[2] == spec and not k[3]]
            if not keys:
                continue
            d, g = R[keys[0]]
            P = g["policies"]
            base = P["classical"]
            pols = ["classical", "hints25", "hints10", "hints5", "hints50", "greedy", "oracle", "router"]
            for pi, pol in enumerate(pols):
                if pol not in P:
                    continue
                row = [SOLVER_NAMES[spec] if pi == 0 else "", POL_NAMES[pol]]
                for tol in [1e-2, 1e-3, d["h2"], 1e-5, 1e-6, 1e-8]:
                    key = tkey(d, tol)
                    row.append(cell_time(P[pol], base, key))
                if pol == "oracle":
                    row = [row[0]] + [f"\\textit{{{c}}}" for c in row[1:]]
                out.append(" & ".join(row) + " \\\\")
            out.append("\\midrule" if spec != SOLVER_ORDER[-1] else "\\bottomrule")
        out.append("\\end{tabular}}")

    # ------------------------------------------- AUC / final error (paper style)
    T = None
    out.append("\\newcommand{\\caauc}{")
    out.append("\\begin{tabular}{lcccccc}\n\\toprule")
    out.append("& \\multicolumn{3}{c}{Poisson} & \\multicolumn{3}{c}{ConvDiff} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    out.append("Method & $\\|e^{(T)}_h\\|/\\|u_h\\|$ & AUC & $p$ & $\\|e^{(T)}_h\\|/\\|u_h\\|$ & AUC & $p$ \\\\ \\midrule")
    for spec in SOLVER_ORDER:
        have = [(eq, R[[k for k in R if k[0] == eq and k[2] == spec and not k[3]][0]])
                for eq in EQS if [k for k in R if k[0] == eq and k[2] == spec and not k[3]]]
        if not have:
            continue
        T = have[0][1][0]["args"]["T"]
        out.append(f"\\multicolumn{{7}}{{c}}{{{SOLVER_NAMES[spec]}-related solvers}} \\\\ \\midrule")
        for pol in ["classical", "hints25", "router", "oracle"]:
            row = [POL_NAMES[pol]]
            for eq in EQS:
                m = dict(have).get(eq)
                if m is None:
                    row += ["--", "--", "--"]
                    continue
                d, g = m
                P = g["policies"]
                if pol not in P:
                    row += ["--", "--", "--"]
                    continue
                err = np.array([r["err_T"] for r in P[pol]])
                auc = np.array([r["auc_T"] for r in P[pol]])
                auc_r = np.array([r["auc_T"] for r in P["router"]])
                def ms(x):
                    med = np.mean(x)
                    se = np.std(x, ddof=1) / math.sqrt(len(x))
                    return f"{med:.2e} ({se:.1e})"
                bold = pol == "router" and all(np.mean(auc) <= np.mean(np.array([r["auc_T"] for r in P[q]]))
                                                for q in ["classical", "hints25"])
                cells = [ms(err), ms(auc)]
                if pol in ("classical", "hints25"):
                    p = ttest_rel(auc, auc_r, alternative="greater").pvalue if not np.allclose(auc, auc_r) else 1.0
                    cells.append(pval_str(p))
                else:
                    cells.append("-")
                if bold:
                    cells = [f"\\textbf{{{c}}}" for c in cells[:2]] + cells[2:]
                if pol == "oracle":
                    cells = [f"\\textit{{{c}}}" for c in cells]
                row += cells
            out.append(" & ".join(row) + " \\\\")
        out.append("\\midrule" if spec != SOLVER_ORDER[-1] else "\\bottomrule")
    out.append("\\end{tabular}}")
    out.append(f"\\newcommand{{\\caT}}{{{T}}}")
    out.append(f"\\newcommand{{\\caN}}{{{N}}}")

    # ------------------------------------------------ usage / iteration counts
    out.append("\\newcommand{\\causage}{")
    out.append("\\begin{tabular}{llcccccc}\n\\toprule")
    out.append("& & \\multicolumn{2}{c}{HINTS ($\\tau{=}25$)} & \\multicolumn{2}{c}{Cost-aware oracle} & "
               "\\multicolumn{2}{c}{Learned router} \\\\")
    out.append("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
    out.append("Equation & Solver & iters & NO calls & iters & NO calls & iters & NO calls \\\\ \\midrule")
    for eq in EQS:
        first = True
        for spec in SOLVER_ORDER:
            keys = [k for k in R if k[0] == eq and k[2] == spec and not k[3]]
            if not keys:
                continue
            d, g = R[keys[0]]
            key = tkey(d, d["h2"])
            P = g["policies"]
            cells = []
            for pol in ["hints25", "oracle", "router"]:
                it = iters(P[pol], key)
                nno = np.array([np.nan if r["tol"][key]["no_calls"] is None else r["tol"][key]["no_calls"] for r in P[pol]])
                cells += [f"{np.median(it):.0f}", f"{np.nanmedian(nno):.0f}"]
            out.append(" & ".join([eq if first else "", SOLVER_NAMES[spec]] + cells) + " \\\\")
            first = False
        if eq == "Poisson":
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ------------------------------------------------------------- costs
    out.append("\\newcommand{\\cacosts}{")
    out.append("\\begin{tabular}{llcccc}\n\\toprule")
    out.append("Equation & Solver & classical iteration & corrector iteration & $m_{\\text{solver}}$ & $m_{\\text{NO}}$ \\\\ \\midrule")
    for eq in EQS:
        first = True
        for spec in SOLVER_ORDER:
            keys = [k for k in R if k[0] == eq and k[2] == spec and not k[3]]
            if not keys:
                continue
            d, g = R[keys[0]]
            c = g["costs"]
            out.append(" & ".join([eq if first else "", SOLVER_NAMES[spec], f"{c[spec]*1e6:.0f}\\,$\\mu$s",
                                   f"{c['no']*1e6:.0f}\\,$\\mu$s", str(g["m"][0]), str(g["m"][-1])]) + " \\\\")
            first = False
        if eq == "Poisson":
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ------------------------------------------------------------- ensembles
    ens_keys = [k for k in R if k[3]]
    if not ens_keys:
        out.append("\\newcommand{\\caens}{\\begin{tabular}{c}(ensemble results pending)\\end{tabular}}")
        out.append("\\newcommand{\\caensusage}{\\begin{tabular}{c}(ensemble results pending)\\end{tabular}}")
    if ens_keys:
        out.append("\\newcommand{\\caens}{")
        out.append("\\begin{tabular}{llcccc}\n\\toprule")
        out.append("Equation & $\\mathcal{W}$ & Best solver only & Best pairwise router & "
                   "Router$(\\mathrm{NO}\\cup\\mathcal{W})$ & Oracle$(\\mathrm{NO}\\cup\\mathcal{W})$ \\\\ \\midrule")
        for eq in EQS:
            first = True
            for k in sorted([k for k in ens_keys if k[0] == eq], key=lambda k: len(k[2].split("+"))):
                d, g = R[k]
                key = tkey(d, d["h2"])
                P = g["policies"]
                members = k[2].split("+")
                # best member solver alone, from the pairwise runs (same instances)
                cls = {}
                for s_ in members:
                    kk = [q for q in R if q[0] == eq and q[2] == s_ and not q[3]]
                    if kk:
                        cls[s_] = np.median(times(R[kk[0]][1]["policies"]["classical"],
                                                  tkey(R[kk[0]][0], d["h2"])))
                bc = min(cls, key=cls.get)
                bc_name = SOLVER_NAMES[bc]
                base = R[[q for q in R if q[0] == eq and q[2] == bc and not q[3]][0]][1]["policies"]["classical"]
                # best pairwise router (from pairwise runs)
                pw = {}
                for s in members:
                    kk = [q for q in R if q[0] == eq and q[2] == s and not q[3]]
                    if kk:
                        pw[s] = R[kk[0]][1]["policies"]["router"]
                bp = min(pw, key=lambda s: np.median(times(pw[s], tkey(R[[q for q in R if q[0]==eq and q[2]==s and not q[3]][0]][0], d["h2"]))))
                pw_rows = pw[bp]
                pw_key = tkey(R[[q for q in R if q[0] == eq and q[2] == bp and not q[3]][0]][0], d["h2"])
                t_pw = times(pw_rows, pw_key)
                t_ens = times(P["router"], key)
                sp_pw, _ = paired_speedup(t_pw, t_ens)
                wname = "\\{" + ", ".join(SOLVER_NAMES[s] for s in members) + "\\}"
                row = [eq if first else "", f"${wname}$",
                       f"{fmt_time(cls[bc])} ({bc_name})",
                       f"{fmt_time(np.median(t_pw))} ({SOLVER_NAMES[bp]})",
                       f"{fmt_time(np.median(t_ens))} ({fmt_sp(sp_pw)} vs pairwise)",
                       "\\textit{" + fmt_time(np.median(times(P['oracle'], key))) + "}"]
                out.append(" & ".join(row) + " \\\\")
                first = False
            if eq == "Poisson":
                out.append("\\midrule")
        out.append("\\bottomrule\n\\end{tabular}}")
        # ensemble usage
        out.append("\\newcommand{\\caensusage}{")
        out.append("\\begin{tabular}{llcccccc}\n\\toprule")
        out.append("Equation & $\\mathcal{W}$ & Jacobi & GS & SymGS & Jacobi (0.67) & SOR (1.5) & DeepONet \\\\ \\midrule")
        for eq in EQS:
            first = True
            for k in sorted([k for k in ens_keys if k[0] == eq], key=lambda k: len(k[2].split("+"))):
                d, g = R[k]
                members = k[2].split("+")
                # per-instance fraction of iterations spent in each operation
                # up to the h^2 crossing (stored by bench.py)
                fracs = np.array([r.get("op_frac", [np.nan] * (len(members) + 1)) for r in g["policies"]["router"]])
                cells = []
                for s in ["jacobi", "gs", "ssor", "jacobi_0.67", "sor_1.5"]:
                    if s in members:
                        j = members.index(s)
                        cells.append(f"{np.nanmean(fracs[:, j]):.3f} ({np.nanstd(fracs[:, j]):.3f})")
                    else:
                        cells.append("-")
                cells.append(f"{np.nanmean(fracs[:, -1]):.3f} ({np.nanstd(fracs[:, -1]):.3f})")
                wname = "\\{" + ", ".join(SOLVER_NAMES[s] for s in members) + "\\}"
                out.append(" & ".join([eq if first else "", f"${wname}$"] + cells) + " \\\\")
                first = False
            if eq == "Poisson":
                out.append("\\midrule")
        out.append("\\bottomrule\n\\end{tabular}}")

    # ------------------------------------------ summary macros for the text
    def rng_macro(name, vals):
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return
        lo, hi = min(vals), max(vals)
        out.append(f"\\newcommand{{\\{name}Min}}{{{fmt_sp(lo)}}}")
        out.append(f"\\newcommand{{\\{name}Max}}{{{fmt_sp(hi)}}}")

    summ = {"Solver": [], "Hints": [], "Best": [], "SolverDeep": [], "HintsDeep": [], "BestDeep": [],
            "OracleRatio": []}
    for eq in EQS:
        for spec in SOLVER_ORDER:
            keys = [k for k in R if k[0] == eq and k[2] == spec and not k[3]]
            if not keys:
                continue
            d, g = R[keys[0]]
            P = g["policies"]
            taus = [p for p in P if p.startswith("hints")]
            for tol, suf in [(d["h2"], ""), (1e-8, "Deep")]:
                key = tkey(d, tol)
                t_r = times(P["router"], key)
                summ["Solver" + suf].append(paired_speedup(times(P["classical"], key), t_r)[0])
                summ["Hints" + suf].append(paired_speedup(times(P["hints25"], key), t_r)[0])
                bt = min(taus, key=lambda p: np.median(times(P[p], key)))
                summ["Best" + suf].append(paired_speedup(times(P[bt], key), t_r)[0])
            key = tkey(d, d["h2"])
            summ["OracleRatio"].append(np.median(times(P["router"], key)) / np.median(times(P["oracle"], key)))
    for name, vals in summ.items():
        rng_macro("caSp" + name, vals)
    n_cells = len(summ["Solver"])
    out.append(f"\\newcommand{{\\caNumCells}}{{{n_cells}}}")
    out.append(f"\\newcommand{{\\caCellsRouterBeatsBest}}{{{sum(v >= 1.0 for v in summ['Best'])}}}")
    out.append(f"\\newcommand{{\\caCellsRouterBeatsHints}}{{{sum(v >= 1.0 for v in summ['Hints'])}}}")

    os.makedirs("paper", exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", OUT_TEX)


if __name__ == "__main__":
    main()
