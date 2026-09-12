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
import re

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

SOLVER_NAMES = {"jacobi": "Jacobi", "jacobi_0.67": "Jacobi (0.67)", "gs": "GS",
                "ssor": "SymGS", "sor_1.5": "SOR (1.5)"}
SOLVER_ORDER = ["jacobi", "jacobi_0.67", "gs", "ssor", "sor_1.5"]
POL_NAMES = {"classical": "Solver only", "hints25": "HINTS ($\\tau{=}25$)",
             "hints5": "HINTS ($\\tau{=}5$)", "hints10": "HINTS ($\\tau{=}10$)",
             "hints50": "HINTS ($\\tau{=}50$)", "greedy": "Greedy oracle (Alg.~1)",
             "oracle": "Cost-aware oracle", "router": "Learned router (ours)"}
EQS = ["Poisson", "ConvDiff", "AnisoDiff"]
EQ_NAMES = {"Poisson": "Poisson", "ConvDiff": "ConvDiff", "AnisoDiff": "AnisoDiff"}


RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
MAIN_N = int(os.environ.get("MAIN_N", "128"))  # grid of the main / per-tolerance / significance tables
OUT_TEX = os.environ.get("OUT_TEX", "paper/costaware_tables.tex")


def load(pattern=None):
    pattern = pattern or f"{RESULTS_DIR}/*.json"
    R = {}
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path))
        a = d["args"]
        if "ensemble" not in a:  # usage_*.json (decision traces), not benchmark output
            continue
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
        # majority censored: report the (median) time spent up to the iteration cap as a lower bound
        cap = np.median([r.get("t_total_live", np.nan) for r in rows])
        return f"$>${fmt_time(cap)}$^{{\\dagger {cens}}}$" if np.isfinite(cap) else "--"
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
            keys = [k for k in R if k[0] == eq and k[1] == MAIN_N and k[2] == spec and not k[3]]
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
        if eq != EQS[-1] and any(k[0] == EQS[EQS.index(eq)+1] for k in R):
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
            keys = [k for k in R if k[0] == eq and k[1] == MAIN_N and k[2] == spec and not k[3]]
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
        if eq != EQS[-1] and any(k[0] == EQS[EQS.index(eq)+1] for k in R):
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ----------------------------------------------------- tolerance sweeps
    for eq in EQS:
        out.append(f"\\newcommand{{\\cawctol{eq.lower().replace('diff','')}}}{{")
        out.append("\\begin{tabular}{llcccccc}\n\\toprule")
        out.append("Solver & Method & $\\varepsilon{=}10^{-2}$ & $\\varepsilon{=}10^{-3}$ & $\\varepsilon{=}h^2$ & "
                   "$\\varepsilon{=}10^{-5}$ & $\\varepsilon{=}10^{-6}$ & $\\varepsilon{=}10^{-8}$ \\\\ \\midrule")
        for spec in SOLVER_ORDER:
            keys = [k for k in R if k[0] == eq and k[1] == MAIN_N and k[2] == spec and not k[3]]
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
    eqs_present = [eq for eq in EQS if any(k[0] == eq and not k[3] for k in R)]
    out.append("\\newcommand{\\caauc}{")
    out.append("\\begin{tabular}{l" + "ccc" * len(eqs_present) + "}\n\\toprule")
    out.append("& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{EQ_NAMES[eq]}}}" for eq in eqs_present) + " \\\\ "
               + "".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(eqs_present))))
    out.append("Method & " + " & ".join("$\\|e^{(T)}_h\\|/\\|u_h\\|$ & AUC & $p$" for _ in eqs_present) + " \\\\ \\midrule")
    for spec in SOLVER_ORDER:
        have = [(eq, R[[k for k in R if k[0] == eq and k[1] == MAIN_N and k[2] == spec and not k[3]][0]])
                for eq in EQS if [k for k in R if k[0] == eq and k[1] == MAIN_N and k[2] == spec and not k[3]]]
        if not have:
            continue
        T = have[0][1][0]["args"]["T"]
        out.append(f"\\multicolumn{{{1+3*len(eqs_present)}}}{{c}}{{{SOLVER_NAMES[spec]}-related solvers}} \\\\ \\midrule")
        for pol in ["classical", "hints25", "router", "oracle"]:
            row = [POL_NAMES[pol]]
            for eq in eqs_present:
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
            keys = [k for k in R if k[0] == eq and k[1] == MAIN_N and k[2] == spec and not k[3]]
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
        if eq != EQS[-1] and any(k[0] == EQS[EQS.index(eq)+1] for k in R):
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ------------------------------------------------------------- costs
    out.append("\\newcommand{\\cacosts}{")
    out.append("\\begin{tabular}{llcccc}\n\\toprule")
    out.append("Equation & Solver & classical iteration & corrector iteration & $m_{\\text{solver}}$ & $m_{\\text{NO}}$ \\\\ \\midrule")
    for eq in EQS:
        first = True
        for spec in SOLVER_ORDER:
            keys = [k for k in R if k[0] == eq and k[1] == MAIN_N and k[2] == spec and not k[3]]
            if not keys:
                continue
            d, g = R[keys[0]]
            c = g["costs"]
            out.append(" & ".join([eq if first else "", SOLVER_NAMES[spec], f"{c[spec]*1e6:.0f}\\,$\\mu$s",
                                   f"{c['no']*1e6:.0f}\\,$\\mu$s", str(g["m"][0]), str(g["m"][-1])]) + " \\\\")
            first = False
        if eq != EQS[-1] and any(k[0] == EQS[EQS.index(eq)+1] for k in R):
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ------------------------------------------------------------- ensembles
    ens_keys = [k for k in R if k[3]]
    ens_ratios, ens_vs_solver = [], []
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
                    kk = [q for q in R if q[0] == eq and q[1] == MAIN_N and q[2] == s_ and not q[3]]
                    if kk:
                        cls[s_] = np.median(times(R[kk[0]][1]["policies"]["classical"],
                                                  tkey(R[kk[0]][0], d["h2"])))
                bc = min(cls, key=cls.get)
                bc_name = SOLVER_NAMES[bc]
                base = R[[q for q in R if q[0] == eq and q[1] == MAIN_N and q[2] == bc and not q[3]][0]][1]["policies"]["classical"]
                # best pairwise router (from pairwise runs)
                pw = {}
                for s in members:
                    kk = [q for q in R if q[0] == eq and q[2] == s and not q[3]]
                    if kk:
                        pw[s] = R[kk[0]][1]["policies"]["router"]
                bp = min(pw, key=lambda s: np.median(times(pw[s], tkey(R[[q for q in R if q[0]==eq and q[1]==MAIN_N and q[2]==s and not q[3]][0]][0], d["h2"]))))
                pw_rows = pw[bp]
                pw_key = tkey(R[[q for q in R if q[0] == eq and q[1] == MAIN_N and q[2] == bp and not q[3]][0]][0], d["h2"])
                t_pw = times(pw_rows, pw_key)
                t_ens = times(P["router"], key)
                sp_pw, _ = paired_speedup(t_pw, t_ens)
                ens_ratios.append(np.median(t_pw) / np.median(t_ens))
                ens_vs_solver.append(cls[bc] / np.median(t_ens))
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
            keys = [k for k in R if k[0] == eq and k[1] == MAIN_N and k[2] == spec and not k[3]]
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

    # ------------------------------------------------ strong classical baselines
    BASE_NAMES = {"fft": "FFT direct solve", "mg": "Multigrid V(2,2) alone", "cg": "CG",
                  "pcg_ssor": "PCG (SymGS)", "pcg_mg": "PCG (multigrid)", "bicgstab": "BiCGSTAB",
                  "bicgstab_mg": "BiCGSTAB (multigrid)", "gmres": "GMRES(20)"}
    Bf = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/baselines_*.json")):
        d = json.load(open(path))
        Bf[(d["args"]["equation"], d["args"]["N"])] = d

    def base_times(d, m, tol):
        key = tkey(d, tol)
        return np.array([np.inf if r["tol"][key]["t_live"] is None else r["tol"][key]["t_live"] for r in d["methods"][m]])

    def base_iters(d, m, tol):
        key = tkey(d, tol)
        return np.array([np.inf if r["tol"][key]["iters"] is None else r["tol"][key]["iters"] for r in d["methods"][m]])

    def med_str(ts, ref=None):
        m = np.median(ts)
        s_ = fmt_time(m)
        if ref is not None and np.isfinite(m):
            sp, _ = paired_speedup(ts, ref)
            s_ += f" ({fmt_sp(sp)})"
        cens = int((~np.isfinite(ts)).sum())
        if cens:
            s_ += f"$^{{\\dagger {cens}}}$"
        return s_

    for N in sorted(set(k[1] for k in Bf)):
        out.append(f"\\newcommand{{\\cabaselines{ {128: '', 256: 'B', 512: 'C'}.get(N, 'X') }}}{{")
        out.append("\\begin{tabular}{llccccc}\n\\toprule")
        out.append("Equation & Method & $\\varepsilon{=}10^{-3}$ & $\\varepsilon{=}h^2$ & $\\varepsilon{=}10^{-6}$ & $\\varepsilon{=}10^{-8}$ & iters to $h^2$ \\\\ \\midrule")
        for eq in EQS:
            if (eq, N) not in Bf:
                continue
            d = Bf[(eq, N)]
            tl = [1e-3, d["h2"], 1e-6, 1e-8]
            # router reference: best pairwise router at h^2 among stationary solvers, plus the MG pairing
            pw = {s_: R[k] for k in R for s_ in [k[2]] if k[0] == eq and k[1] == N and not k[3] and k[2] in SOLVER_ORDER}
            mgp = [R[k] for k in R if k[0] == eq and k[1] == N and not k[3] and k[2] == "mg"]
            best_s = min(pw, key=lambda s_: np.median(times(pw[s_][1]["policies"]["router"], tkey(pw[s_][0], d["h2"])))) if pw else None
            ref = times(pw[best_s][1]["policies"]["router"], tkey(pw[best_s][0], d["h2"])) if best_s else None
            first = True
            for m in d["methods"]:
                cells = [med_str(base_times(d, m, t), ref) for t in tl]
                its = base_iters(d, m, d["h2"])
                cells.append(f"{np.median(its):.0f}" if np.isfinite(np.median(its)) else "--")
                out.append(" & ".join([eq if first else "", BASE_NAMES.get(m, m)] + cells) + " \\\\")
                first = False
            if best_s:
                dd, g = pw[best_s]
                P = g["policies"]
                for pol, lab in [("hints25", f"HINTS ($\\tau{{=}}25$), {SOLVER_NAMES[best_s]}"),
                                 ("router", f"Learned router, $\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[best_s]}}}\\}}$")]:
                    cells = [med_str(times(P[pol], tkey(dd, t)), ref if pol != "router" else None) for t in tl]
                    cells.append(f"{np.median(iters(P[pol], tkey(dd, dd['h2']))):.0f}")
                    row = " & ".join(["", lab] + cells) + " \\\\"
                    out.append(f"\\textbf{{{row}}}" if False else row)
            if mgp:
                dd, g = mgp[0]
                P = g["policies"]
                for pol, lab in [("router", "Learned router, $\\{\\mathrm{NO}, \\text{Multigrid}\\}$"),
                                 ("oracle", "Cost-aware oracle, $\\{\\mathrm{NO}, \\text{Multigrid}\\}$")]:
                    cells = [med_str(times(P[pol], tkey(dd, t))) for t in tl]
                    cells.append(f"{np.median(iters(P[pol], tkey(dd, dd['h2']))):.0f}")
                    out.append(" & ".join(["", lab] + cells) + " \\\\")
            if eq == "Poisson":
                out.append("\\midrule")
        out.append("\\bottomrule\n\\end{tabular}}")

    # ------------------------------------------------------------- scaling
    Ns = sorted(set(k[1] for k in R if not k[3]))
    out.append("\\newcommand{\\cascaling}{")
    out.append("\\begin{tabular}{llcccccccc}\n\\toprule")
    out.append("Equation & $N$ & Solver only & HINTS ($\\tau{=}25$) & HINTS ($\\tau{=}5$) & Learned router & Cost-aware oracle & Multigrid & PCG/BiCGSTAB (MG) & FFT direct \\\\ \\midrule")
    for eq in EQS:
        for spec in ["jacobi", "gs"]:
            first = True
            for N in Ns:
                k = [q for q in R if q[0] == eq and q[1] == N and q[2] == spec and not q[3]]
                if not k:
                    continue
                d, g = R[k[0]]
                P = g["policies"]
                key = tkey(d, d["h2"])
                tr_ = times(P["router"], key)
                cells = [cell_time(P["classical"], P["classical"], key),
                         med_str(times(P["hints25"], key), tr_), med_str(times(P["hints5"], key), tr_),
                         med_str(tr_), "\\textit{" + med_str(times(P["oracle"], key)) + "}"]
                if (eq, N) in Bf:
                    db = Bf[(eq, N)]
                    kry = "pcg_mg" if eq == "Poisson" else "bicgstab_mg"
                    cells += [med_str(base_times(db, "mg", d["h2"]), tr_),
                              med_str(base_times(db, kry, d["h2"]), tr_) if kry in db["methods"] else "--",
                              med_str(base_times(db, "fft", d["h2"]), tr_)]
                else:
                    cells += ["--", "--", "--"]
                lab = f"{eq}, {SOLVER_NAMES[spec]}" if first else ""
                out.append(" & ".join([lab, f"${N}^2$"] + cells) + " \\\\")
                first = False
            out.append("\\midrule")
    out[-1] = "\\bottomrule\n\\end{tabular}}"

    # ------------------------------------------------------------- overheads
    op = f"{RESULTS_DIR}/overheads.json"
    if os.path.exists(op):
        O = json.load(open(op))
        out.append("\\newcommand{\\caoverheads}{")
        out.append("\\begin{tabular}{lccccc}\n\\toprule")
        Ns_o = sorted(int(k) for k in O["per_op"])
        out.append("Operation & " + " & ".join(f"$N={n}$" for n in Ns_o) + " \\\\ \\midrule")
        rows = [("Jacobi iteration", "jacobi"), ("GS iteration", "gs"), ("SymGS iteration", "ssor"),
                ("Multigrid V(2,2) cycle", "mg"), ("FFT direct solve", "fft"),
                ("DeepONet corrector call", "corrector"), ("Router decision (ours)", "router_decision"),
                ("LSTM router decision (\\Cref{sec:experiments})", "lstm_router_decision")]
        def fmt_us(v):
            if v is None or not isinstance(v, (int, float)):
                return "--"
            return f"{v*1e6:.0f}\\,$\\mu$s" if v < 1e-3 else f"{v*1e3:.2f}\\,ms"
        for lab, key in rows:
            cells = []
            for n in Ns_o:
                v = O["per_op"][str(n)].get(key)
                if key == "mg" and n < 64:
                    v = None
                cells.append(fmt_us(v))
            out.append(lab + " & " + " & ".join(cells) + " \\\\")
        out.append("\\bottomrule\n\\end{tabular}}")
        # amortisation: corrector training and router training times
        tr = O["training"]
        def get(k):
            return tr.get(k)
        out.append("\\newcommand{\\caamort}{")
        out.append("\\begin{tabular}{llccccc}\n\\toprule")
        out.append("Equation & $N$ & Corrector data + fit & Router training (GS pairing) & Saving per solve vs HINTS ($\\tau{=}25$) & Break-even solves (router) & Saving per solve vs solver only \\\\ \\midrule")
        for eq in EQS:
            first = True
            for N in Ns:
                c = get(f"corrector_{eq}_{N}")
                rt = get(f"router_{eq}_{N}_gs")
                k = [q for q in R if q[0] == eq and q[1] == N and q[2] == "gs" and not q[3]]
                if not k or c is None:
                    continue
                d, g = R[k[0]]
                P = g["policies"]
                key = tkey(d, d["h2"])
                t_r = np.median(times(P["router"], key)); t_h = np.median(times(P["hints25"], key)); t_c = np.median(times(P["classical"], key))
                sav_h = t_h - t_r; sav_c = t_c - t_r
                be = (rt["train_s"] / sav_h) if (rt and sav_h > 0) else np.inf
                out.append(" & ".join([eq if first else "", f"${N}^2$",
                                       f"{c['data_s'] + c['fit_s']:.0f}\\,s",
                                       f"{rt['train_s']:.0f}\\,s" if rt else "--",
                                       fmt_time(sav_h) if sav_h > 0 else "--",
                                       f"{be:.0f}" if np.isfinite(be) else "--",
                                       fmt_time(sav_c)]) + " \\\\")
                first = False
            if eq == "Poisson":
                out.append("\\midrule")
        out.append("\\bottomrule\n\\end{tabular}}")
        lst = O["per_op"]
        n128 = lst.get("128", {})
        if n128.get("lstm_router_decision") and n128.get("jacobi"):
            out.append(f"\\newcommand{{\\caLstmOverJacobi}}{{{n128['lstm_router_decision']/n128['jacobi']:.0f}}}")
            out.append(f"\\newcommand{{\\caLstmMs}}{{{n128['lstm_router_decision']*1e3:.1f}}}")

    # ------------------------------------------- significance + seed robustness
    def wilcoxon_p(base, ours):
        """one-sided paired Wilcoxon (alternative: ours faster), censoring-aware"""
        _, r = paired_speedup(base, ours)
        ok = np.isfinite(r) & (r > 0)
        if ok.sum() < 8 or np.allclose(r[ok], 1.0):
            return 1.0
        return float(wilcoxon(np.log(r[ok]), alternative="greater").pvalue)

    def ttest_p(base, ours):
        ok = np.isfinite(base) & np.isfinite(ours)
        if ok.sum() < 8 or np.allclose(base[ok], ours[ok]):
            return 1.0
        return float(ttest_rel(np.log(base[ok]), np.log(ours[ok]), alternative="greater").pvalue)

    def pstr(pv):
        if pv < 1e-10:
            return "$<10^{-10}$"
        if pv < 1e-3:
            return f"$10^{{{int(np.floor(np.log10(pv)))}}}$"
        return f"{pv:.3f}"

    Sf = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/seeds_*.json")):
        d = json.load(open(path))
        Sf[(d["args"]["equation"], d["args"]["N"])] = d

    out.append("\\newcommand{\\castats}{")
    out.append("\\begin{tabular}{llcccccccc}\n\\toprule")
    out.append("& & \\multicolumn{3}{c}{$\\varepsilon = h^2$: $p$ (Wilcoxon / $t$)} & \\multicolumn{2}{c}{$\\varepsilon = 10^{-8}$: $p$} & \\multicolumn{3}{c}{Router over 5 training seeds} \\\\")
    out.append("\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}")
    out.append("Equation & Solver & vs.\\ solver only & vs.\\ HINTS-25 & vs.\\ best $\\tau$ & vs.\\ HINTS-25 & vs.\\ best $\\tau$ & time to $h^2$ & speedup vs.\\ HINTS-25 & seeds with $p{<}0.01$ \\\\ \\midrule")
    for eq in EQS:
        first = True
        for spec in SOLVER_ORDER:
            keys = [k for k in R if k[0] == eq and k[1] == MAIN_N and k[2] == spec and not k[3]]
            if not keys:
                continue
            d, g = R[keys[0]]
            P = g["policies"]
            taus = [p_ for p_ in P if p_.startswith("hints")]
            cells = []
            for tol, with_solver in [(d["h2"], True), (1e-8, False)]:
                key = tkey(d, tol)
                t_r = times(P["router"], key)
                bt = min(taus, key=lambda p_: np.median(times(P[p_], key)))
                if with_solver:
                    t_c = times(P["classical"], key)
                    cells.append(f"{pstr(wilcoxon_p(t_c, t_r))} / {pstr(ttest_p(t_c, t_r))}")
                cells.append(f"{pstr(wilcoxon_p(times(P['hints25'], key), t_r))} / {pstr(ttest_p(times(P['hints25'], key), t_r))}")
                cells.append(f"{pstr(wilcoxon_p(times(P[bt], key), t_r))} / {pstr(ttest_p(times(P[bt], key), t_r))}")
            # seeds
            sd = Sf.get((eq, keys[0][1]))
            if sd and spec in sd["groups"] and sd["groups"][spec]:
                key = tkey(d, d["h2"])
                th = times(P["hints25"], key)
                meds, sps, nsig = [], [], 0
                for s_, blk in sd["groups"][spec].items():
                    tr_ = np.array([np.inf if r["tol"][key]["t_live"] is None else r["tol"][key]["t_live"] for r in blk["rows"]])
                    meds.append(np.median(tr_))
                    sps.append(paired_speedup(th, tr_)[0])
                    if wilcoxon_p(th, tr_) < 0.01 and wilcoxon_p(times(P[min(taus, key=lambda p_: np.median(times(P[p_], key)))], key), tr_) < 0.01:
                        nsig += 1
                cells += [f"{np.mean(meds)*1e3:.2f} $\\pm$ {np.std(meds)*1e3:.2f}\\,ms",
                          f"{min(sps):.2f}--{max(sps):.2f}$\\times$", f"{nsig}/{len(meds)}"]
            else:
                cells += ["--", "--", "--"]
            out.append(" & ".join([eq if first else "", SOLVER_NAMES[spec]] + cells) + " \\\\")
            first = False
        if eq != EQS[-1] and any(k[0] == EQS[EQS.index(eq)+1] for k in R):
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ensembles: p-values
    out.append("\\newcommand{\\castatsens}{")
    out.append("\\begin{tabular}{llcccc}\n\\toprule")
    out.append("Equation & $\\mathcal{W}$ & vs.\\ best solver only & vs.\\ best pairwise router (faster) & vs.\\ best pairwise router (slower) & vs.\\ oracle$(\\mathrm{NO}\\cup\\mathcal{W})$ (slower) \\\\ \\midrule")
    for eq in EQS:
        first = True
        for k in sorted([k for k in ens_keys if k[0] == eq], key=lambda k: len(k[2].split("+"))):
            d, g = R[k]
            key = tkey(d, d["h2"])
            members = k[2].split("+")
            t_e = times(g["policies"]["router"], key)
            t_o = times(g["policies"]["oracle"], key)
            cls = {}
            for s_ in members:
                kk = [q for q in R if q[0] == eq and q[1] == MAIN_N and q[2] == s_ and not q[3]]
                if kk:
                    cls[s_] = times(R[kk[0]][1]["policies"]["classical"], tkey(R[kk[0]][0], d["h2"]))
            bc = min(cls, key=lambda s_: np.median(cls[s_]))
            pw = {s_: times(R[[q for q in R if q[0] == eq and q[1] == MAIN_N and q[2] == s_ and not q[3]][0]][1]["policies"]["router"], key) for s_ in members if [q for q in R if q[0] == eq and q[1] == MAIN_N and q[2] == s_ and not q[3]]}
            bp = min(pw, key=lambda s_: np.median(pw[s_]))
            wname = "\\{" + ", ".join(SOLVER_NAMES[s_] for s_ in members) + "\\}"
            out.append(" & ".join([eq if first else "", f"${wname}$", pstr(wilcoxon_p(cls[bc], t_e)),
                                   pstr(wilcoxon_p(pw[bp], t_e)), pstr(wilcoxon_p(t_e, pw[bp])),
                                   pstr(wilcoxon_p(t_e, t_o))]) + " \\\\")
            first = False
        if eq != EQS[-1] and any(k[0] == EQS[EQS.index(eq)+1] for k in R):
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # strong baselines: p-values (router with best stationary pairing vs each baseline)
    out.append("\\newcommand{\\castatsbase}{")
    out.append("\\begin{tabular}{llcccc}\n\\toprule")
    out.append("Equation & Method & $\\varepsilon{=}10^{-3}$ & $\\varepsilon{=}h^2$ & $\\varepsilon{=}10^{-6}$ & $\\varepsilon{=}10^{-8}$ \\\\ \\midrule")
    for (eq, N) in sorted(Bf):
        d = Bf[(eq, N)]
        pw = {k[2]: R[k] for k in R if k[0] == eq and k[1] == N and not k[3] and k[2] in SOLVER_ORDER}
        if not pw:
            continue
        best_s = min(pw, key=lambda s_: np.median(times(pw[s_][1]["policies"]["router"], tkey(pw[s_][0], d["h2"]))))
        dd, g = pw[best_s]
        first = True
        for m in d["methods"]:
            cells = []
            for tol in [1e-3, d["h2"], 1e-6, 1e-8]:
                t_r = times(g["policies"]["router"], tkey(dd, tol))
                tb = base_times(d, m, tol)
                pv_f = wilcoxon_p(tb, t_r)   # router faster
                pv_s = wilcoxon_p(t_r, tb)   # router slower
                cells.append(pstr(pv_f) if np.median(tb) >= np.median(t_r) else f"slower: {pstr(pv_s)}")
            out.append(" & ".join([f"{eq} ${N}^2$" if first else "", BASE_NAMES.get(m, m)] + cells) + " \\\\")
            first = False
        out.append("\\midrule")
    out[-1] = "\\bottomrule\n\\end{tabular}}"

    # ---------------------------------------- macros for the main-text sentence
    vs_mg, vs_kry, fft_ratio, vs_mg_ens = [], [], [], []
    for (eq, N), d in Bf.items():
        if N != 128:
            continue
        pw = {k[2]: R[k] for k in R if k[0] == eq and k[1] == N and not k[3] and k[2] in SOLVER_ORDER}
        if not pw:
            continue
        best_s = min(pw, key=lambda s_: np.median(times(pw[s_][1]["policies"]["router"], tkey(pw[s_][0], d["h2"]))))
        dd, g = pw[best_s]
        key = tkey(dd, dd["h2"])
        t_r = times(g["policies"]["router"], key)
        if "mg" in d["methods"]:
            vs_mg.append(paired_speedup(base_times(d, "mg", dd["h2"]), t_r)[0])
        kry = "pcg_mg" if eq == "Poisson" else "bicgstab_mg"
        if kry in d["methods"]:
            vs_kry.append(paired_speedup(base_times(d, kry, dd["h2"]), t_r)[0])
        if "fft" in d["methods"]:
            fft_ratio.append(np.median(t_r) / np.median(base_times(d, "fft", dd["h2"])))
        mgp = [R[k] for k in R if k[0] == eq and k[1] == N and not k[3] and k[2] == "mg"]
        if mgp:
            vs_mg_ens.append(paired_speedup(times(mgp[0][1]["policies"]["classical"], tkey(mgp[0][0], dd["h2"])),
                                            times(mgp[0][1]["policies"]["router"], tkey(mgp[0][0], dd["h2"])))[0])
    for name, vals in [("caVsMg", vs_mg), ("caVsKrylov", vs_kry), ("caFftRatio", fft_ratio), ("caVsMgEns", vs_mg_ens)]:
        rng_macro(name, vals)

    # placeholders for macros whose data may not exist yet
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(out)))
    for name in ["cabaselines", "cabaselinesB", "cabaselinesC", "caoverheads", "caamort",
                 "castats", "castatsens", "castatsbase", "cascaling"]:
        if name not in defined:
            out.append(f"\\newcommand{{\\{name}}}{{\\begin{{tabular}}{{c}}(results pending)\\end{{tabular}}}}")
    for name, val in [("caLstmMs", "--"), ("caLstmOverJacobi", "--"), ("caEnsVsPairMin", "--"),
                      ("caVsMgMin", "--"), ("caVsMgMax", "--"), ("caVsKrylovMin", "--"), ("caVsKrylovMax", "--"),
                      ("caFftRatioMin", "--"), ("caFftRatioMax", "--"), ("caVsMgEnsMin", "--"), ("caVsMgEnsMax", "--"),
                      ("caEnsVsPairMax", "--"), ("caEnsVsSolverMin", "--"), ("caEnsVsSolverMax", "--"), ("caNumEns", "--")]:
        if name not in defined and not (ens_ratios and name.startswith("caEns")) and not (ens_ratios and name == "caNumEns"):
            out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    if ens_ratios:
        out.append(f"\\newcommand{{\\caEnsVsPairMin}}{{{fmt_sp(min(ens_ratios))}}}")
        out.append(f"\\newcommand{{\\caEnsVsPairMax}}{{{fmt_sp(max(ens_ratios))}}}")
        out.append(f"\\newcommand{{\\caEnsVsSolverMin}}{{{fmt_sp(min(ens_vs_solver))}}}")
        out.append(f"\\newcommand{{\\caEnsVsSolverMax}}{{{fmt_sp(max(ens_vs_solver))}}}")
        out.append(f"\\newcommand{{\\caNumEns}}{{{len(ens_ratios)}}}")

    os.makedirs("paper", exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", OUT_TEX)


if __name__ == "__main__":
    main()
