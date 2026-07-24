"""Generate LaTeX tables for the wall-clock evaluation from results_wallclock/*.json.

Writes paper/wallclock_tables.tex containing:
  \\wcmaintable      - main-text table: time to truncation-level tolerance (h^2),
                       jacobi pairing, both equations, N in {31, 63, 127}
  \\wctoltablepoisson / \\wctoltableconvdiff - appendix: full tolerance sweep
  \\wcsolvertable    - appendix: solver-strength axis (gs, damped jacobi, sor)
  \\wchintstable     - appendix: HINTS tau sweep at Poisson N=63
All entries: median over test instances; speedups are per-instance paired
medians vs the classical-only baseline; p-values are one-sided paired t-tests
on log times (alternative: hybrid faster).
"""

import glob
import json

import numpy as np
from scipy.stats import ttest_rel

H2 = {31: "1.04e-3", 63: "2.52e-4", 127: "6.2e-05"}
POL_NAMES = {"classical": "Solver only", "hints25": "HINTS ($\\tau{=}25$)",
             "router": "Learned router (ours)", "oracle_ca": "\\textit{Cost-aware oracle}"}


def load_all(pattern="results_wallclock/*.json"):
    out = {}
    for path in glob.glob(pattern):
        d = json.load(open(path))
        a = d["args"]
        for spec, polres in d["results"].items():
            out[(a["equation"], a["N"], spec)] = polres
    return out


def tol_key(rows, tol_float):
    """Find the stored key matching a float tolerance."""
    keys = rows[0]["err"].keys()
    for k in keys:
        if abs(float(k) - tol_float) < 1e-12 * max(1.0, tol_float):
            return k
    # nearest within 2% (h^2 entries stored as parsed floats)
    for k in keys:
        if abs(float(k) - tol_float) < 0.02 * tol_float:
            return k
    return None


def times(rows, tol_float, crit="err"):
    k = tol_key(rows, tol_float)
    if k is None:
        return None
    return np.array([np.inf if r[crit][k][0] == "inf" else r[crit][k][0] for r in rows])


def fmt_time(x):
    if not np.isfinite(x):
        return "--"
    if x < 1e-3:
        return f"{x*1e6:.0f}\\,$\\mu$s"
    if x < 1.0:
        return f"{x*1e3:.1f}\\,ms"
    return f"{x:.2f}\\,s"


def cell(rows, base_rows, tol, with_p=False):
    ts = times(rows, tol)
    tb = times(base_rows, tol)
    if ts is None or tb is None or not np.isfinite(np.median(ts)):
        return "--"
    sp = np.median(tb / ts)
    s = f"{fmt_time(np.median(ts))} ({sp:.1f}$\\times$)"
    if with_p and rows is not base_rows:
        ok = np.isfinite(ts) & np.isfinite(tb)
        if ok.sum() >= 8 and not np.allclose(ts[ok], tb[ok]):
            p = ttest_rel(np.log(tb[ok]), np.log(ts[ok]), alternative="greater").pvalue
            s += " [$p{<}10^{-3}$]" if p < 1e-3 else f" [$p{{=}}{p:.3f}$]"
    return s


def main():
    R = load_all()
    out = []

    # ---------------- main table: time to h^2 tolerance, jacobi pairing
    out.append("\\newcommand{\\wcmaintable}{")
    out.append("\\begin{tabular}{llcccc}\n\\toprule")
    out.append("Equation & $N$ & Solver only & HINTS ($\\tau{=}25$) & "
               "Learned router (ours) & \\textit{Cost-aware oracle} \\\\ \\midrule")
    for eq in ["Poisson", "ConvDiff"]:
        for N in [31, 63, 127]:
            key = (eq, N, "jacobi")
            if key not in R:
                continue
            pr = R[key]
            base = pr["classical"]
            tol = float(H2[N])
            row = [f"{eq} & ${N}\\times{N}$"]
            for pol in ["classical", "hints25", "router", "oracle_ca"]:
                row.append(cell(pr[pol], base, tol) if pol in pr else "--")
            out.append(" & ".join(row) + " \\\\")
        if eq == "Poisson":
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ---------------- appendix: tolerance sweeps
    for eq in ["Poisson", "ConvDiff"]:
        out.append(f"\\newcommand{{\\wctoltable{eq.lower().replace('diff','')}}}{{")
        out.append("\\begin{tabular}{llccccc}\n\\toprule")
        out.append("$N$ & Method & $\\varepsilon{=}10^{-3}$ & $\\varepsilon{=}h^2$ & "
                   "$\\varepsilon{=}10^{-5}$ & $\\varepsilon{=}10^{-6}$ & "
                   "$\\varepsilon{=}10^{-8}$ \\\\ \\midrule")
        for N in [31, 63, 127]:
            key = (eq, N, "jacobi")
            if key not in R:
                continue
            pr = R[key]
            base = pr["classical"]
            for pol in ["classical", "hints25", "router", "oracle_ca"]:
                if pol not in pr:
                    continue
                row = [f"${N}^2$" if pol == "classical" else "",
                       POL_NAMES[pol]]
                for tol in [1e-3, float(H2[N]), 1e-5, 1e-6, 1e-8]:
                    row.append(cell(pr[pol], base, tol))
                out.append(" & ".join(row) + " \\\\")
            out.append("\\midrule" if N != 127 else "\\bottomrule")
        out.append("\\end{tabular}}")

    # ---------------- appendix: solver-strength axis
    out.append("\\newcommand{\\wcsolvertable}{")
    out.append("\\begin{tabular}{llcccc}\n\\toprule")
    out.append("Setting & Solver & Solver only & HINTS ($\\tau{=}25$) & "
               "Learned router (ours) & \\textit{Cost-aware oracle} \\\\ \\midrule")
    solver_names = {"jacobi": "Jacobi", "jacobi_0.67": "Jacobi (0.67)",
                    "gs": "GS", "sor_1.5": "SOR (1.5)"}
    for (eq, N) in [("Poisson", 63), ("ConvDiff", 63), ("Poisson", 127)]:
        for spec in ["jacobi", "jacobi_0.67", "gs", "sor_1.5"]:
            key = (eq, N, spec)
            if key not in R:
                continue
            pr = R[key]
            base = pr["classical"]
            tol = float(H2[N])
            row = [f"{eq} ${N}^2$", solver_names[spec]]
            for pol in ["classical", "hints25", "router", "oracle_ca"]:
                row.append(cell(pr[pol], base, tol) if pol in pr else "--")
            out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ---------------- appendix: HINTS tau sweep (Poisson 63)
    out.append("\\newcommand{\\wchintstable}{(pending)}")
    key = ("Poisson", 63, "jacobi")
    if key in R and "hints5" in R[key]:
        out.pop()
        pr = R[key]
        base = pr["classical"]
        out.append("\\newcommand{\\wchintstable}{")
        out.append("\\begin{tabular}{lccccc}\n\\toprule")
        out.append("Method & $\\varepsilon{=}10^{-3}$ & $\\varepsilon{=}h^2$ & "
                   "$\\varepsilon{=}10^{-5}$ & $\\varepsilon{=}10^{-6}$ & "
                   "$\\varepsilon{=}10^{-8}$ \\\\ \\midrule")
        names = {"hints5": "HINTS ($\\tau{=}5$)", "hints10": "HINTS ($\\tau{=}10$)",
                 "hints25": "HINTS ($\\tau{=}25$)", "hints50": "HINTS ($\\tau{=}50$)",
                 "hints100": "HINTS ($\\tau{=}100$)",
                 "router": "Learned router (ours)"}
        for pol in ["hints5", "hints10", "hints25", "hints50", "hints100", "router"]:
            if pol not in pr:
                continue
            row = [names[pol]]
            for tol in [1e-3, 2.52e-4, 1e-5, 1e-6, 1e-8]:
                row.append(cell(pr[pol], base, tol))
            out.append(" & ".join(row) + " \\\\")
        out.append("\\bottomrule\n\\end{tabular}}")

    with open("paper/wallclock_tables.tex", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote paper/wallclock_tables.tex")


if __name__ == "__main__":
    main()
