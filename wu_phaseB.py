"""WU-accounted Phase B analysis: router and hints recomputed from op counts
(err_no_at fields) x constants; oracle rows already constants-charged.
Router additionally pays its measured per-decision constant every iteration."""
import glob, json, sys
import numpy as np
pat = sys.argv[1]
TOLS = [("h2",6.2e-5),("h2/10",6.2e-6),("1e-6",1e-6),("1e-8",1e-8)]
for path in sorted(glob.glob(pat)):
    dd = json.load(open(path)); spec = list(dd['results'].keys())[0]
    d = dd['results'][spec]; eq = dd['args']['equation']
    oc = d['op_costs']; cc, cn, cd = oc['classical'], oc['no'], oc.get('router_decide', 0.0)
    def wu(pol, tolf, dec_cost=0.0):
        rows = d[pol]
        for k in rows[0]['err']:
            if abs(float(k)-tolf) < 0.03*tolf:
                out = []
                for r in rows:
                    i = r['err'][k][1]
                    if i == 'inf' or not np.isfinite(i): out.append(np.inf); continue
                    if pol == 'oracle':
                        out.append(r['err'][k][0])  # already constants-charged
                    else:
                        nno = r.get('err_no_at', {}).get(k, None)
                        if nno is None or nno == 'inf':
                            nno = np.floor(i/25.0) if pol == 'hints25' else 0
                        out.append(i*cc + float(nno)*(cn-cc) + i*dec_cost)
                return np.array(out)
    line_r, line_o = [], []
    for nm, tolf in TOLS:
        h = wu('hints25', tolf)
        r = wu('router', tolf, dec_cost=cd) if 'router' in d else None
        o = wu('oracle', tolf)
        line_o.append(f"{nm}:{np.median(h/o):.2f}")
        line_r.append(f"{nm}:{np.median(h/r):.2f}" if r is not None else f"{nm}:--")
    print(f"{eq:9s} {spec:12s} hints/ORACLE " + " ".join(line_o) +
          "  |  hints/ROUTER " + " ".join(line_r))
