"""Generate the 127^2 all-solver HINTS-comparison LaTeX table (WU accounting)
into paper/wallclock_hints127.tex as \\wchintsallsolvers."""
import glob, json
import numpy as np

def wu_ratios(path):
    dd = json.load(open(path)); spec = list(dd['results'].keys())[0]
    d = dd['results'][spec]; eq = dd['args']['equation']
    oc = d['op_costs']; cc, cn, cd = oc['classical'], oc['no'], oc.get('router_decide', 0.0)
    def wu(pol, tolf, dec=0.0):
        rows = d[pol]
        for k in rows[0]['err']:
            if abs(float(k)-tolf) < 0.03*tolf:
                out = []
                for r in rows:
                    i = r['err'][k][1]
                    if i == 'inf' or not np.isfinite(i): out.append(np.inf); continue
                    if pol == 'oracle': out.append(r['err'][k][0]); continue
                    nno = r.get('err_no_at', {}).get(k)
                    if nno is None or nno == 'inf':
                        nno = np.floor(i/25.0) if pol == 'hints25' else 0
                    out.append(i*cc + float(nno)*(cn-cc) + i*dec)
                return np.array(out)
    res = {}
    for tag, tolf in [("h2", 6.2e-5), ("h2_10", 6.2e-6)]:
        h = wu('hints25', tolf)
        res[tag] = (np.median(h/wu('oracle', tolf)),
                    np.median(h/wu('router', tolf, cd)))
    return eq, spec, res

NAMES = {"jacobi": "Jacobi", "jacobi_0.67": "Jacobi (0.67)", "gs": "GS",
         "ssor": "SymGS", "sor_1.5": "SOR (1.5)"}
cells = {}
for p in glob.glob('results_wallclock_phaseB/*.json'):
    eq, spec, r = wu_ratios(p)
    cells[(eq, spec)] = r

def fmt(x):
    s = f"{x:.2f}$\\times$"
    return f"\\textbf{{{s}}}" if x >= 1.10 else s

out = ["\\newcommand{\\wchintsallsolvers}{", "\\begin{tabular}{llcccc}", "\\toprule",
       "& & \\multicolumn{2}{c}{Oracle greedy vs HINTS} & "
       "\\multicolumn{2}{c}{Learned router vs HINTS} \\\\",
       "Equation & Solver & $\\varepsilon{=}h^2$ & $\\varepsilon{=}h^2/10$ & "
       "$\\varepsilon{=}h^2$ & $\\varepsilon{=}h^2/10$ \\\\ \\midrule"]
for eq in ["Poisson", "ConvDiff"]:
    for spec in ["jacobi", "jacobi_0.67", "gs", "ssor", "sor_1.5"]:
        r = cells[(eq, spec)]
        out.append(f"{eq} & {NAMES[spec]} & {fmt(r['h2'][0])} & {fmt(r['h2_10'][0])} & "
                   f"{fmt(r['h2'][1])} & {fmt(r['h2_10'][1])} \\\\")
    if eq == "Poisson":
        out.append("\\midrule")
out += ["\\bottomrule", "\\end{tabular}}"]
open('paper/wallclock_hints127.tex', 'w').write("\n".join(out) + "\n")
print("wrote paper/wallclock_hints127.tex")
