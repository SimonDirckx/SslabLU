"""Reproducibility, discretization convergence, rank properties, performance and
accuracy-vs-cost tables (stdout). Run eda_load.py first."""
import os, pickle
import numpy as np
from eda_common import *

g = lambda v: "%8.1e" % v if v == v else "     -  "
print("=" * 100)
print("A. REPRODUCIBILITY (1013 vs 1018, same configuration)")
rbd = []
for r in find(rep="1013", kind="rb"):
    s = one(sweep=r["sweep"], rep="1018", npy=r["npy"], p=r["p"], N=r["N"], kind="rb")
    rbd.append(max(rel(fld(s, f), fld(r, f)) for f in ("eta", "u", "v")))
print("  rb: max rel diff over eta/u/v, %d pairs: median %.1e max %.1e" % (len(rbd), np.median(rbd), max(rbd)))
for k in ("smap", "solver", "both"):
    rat, dd, ee = [], [], []
    for r in find(rep="1013", kind=k):
        s = one(sweep=r["sweep"], rep="1018", npy=r["npy"], p=r["p"], N=r["N"], solver=r["solver"], rk=r["rk"], rbrk=r["rbrk"])
        b13, b18 = rbof(r), rbof(s)
        e13, e18 = rel(fld(r), fld(b13)), rel(fld(s), fld(b18))
        d = rel(fld(s), fld(r))
        rat.append(e18 / e13); dd.append(d / np.sqrt(e13 * e18)); ee.append(e13)
    rat, dd = np.array(rat), np.array(dd)
    print("  %-6s %2d pairs: err_1018/err_1013 median %.2f [10%%..90%%: %.2f..%.2f]; |X18-X13| / err: median %.2f [%.2f..%.2f]"
          % (k, len(rat), np.median(rat), *np.percentile(rat, [10, 90]), np.median(dd), *np.percentile(dd, [10, 90])))
# rb across sweeps (compare forms on/off, different processes)
xs = [rel(fld(one(sweep="rankiso", rep=rp, npy=32, p=p, N=N, kind="rb")), fld(one(sweep="solver", rep=rp, npy=32, p=p, N=N, kind="rb")))
      for rp in ("1013", "1018") for p in (8, 12, 16) for N in (8, 16, 32)]
print("  rb solver-sweep vs rankiso-sweep (npan_y=32): max %.1e" % max(xs))

print("=" * 100)
print("B. DISCRETIZATION CONVERGENCE (rb; reference p=16 npan_y=32 N=8, 1013)")
ref = one(sweep="solver", rep="1013", npy=32, p=16, N=8, kind="rb")
E = {}
for f in ("eta", "u", "v"):
    for t in ("mid", "final"):
        for p in (8, 12, 16):
            for npy in (8, 16, 32):
                r = one(sweep="solver", rep="1013", npy=npy, p=p, N=8, kind="rb")
                E[(f, t, p, npy)] = rel(fld(r, f, t), fld(ref, f, t))
for f in ("eta", "u", "v"):
    print("  %-3s final: " % f + "   ".join("p=%d: %s" % (p, " ".join(g(E[(f, 'final', p, n)]) for n in (8, 16, 32))) for p in (8, 12, 16)))
print("  observed h-order log2(e(npy)/e(2npy)), eta final:  " + "  ".join(
    "p=%d: %.1f, %s" % (p, np.log2(E[("eta", "final", p, 8)] / E[("eta", "final", p, 16)]),
                         "%.1f" % np.log2(E[("eta", "final", p, 16)] / E[("eta", "final", p, 32)]) if p < 16 else "-") for p in (8, 12, 16)))
print("  mid/final error ratio (eta): " + "  ".join("p%d/npy%d %.2f" % (p, n, E[("eta", "mid", p, n)] / E[("eta", "final", p, n)])
                                                  for p in (8, 12) for n in (8, 16, 32)))
# reference-free successive differences
print("  reference-free |X(p,npy) - X(p,2npy)| eta final: " + "  ".join(
    "p=%d: %s" % (p, " ".join(g(rel(fld(one(sweep="solver", rep="1013", npy=n, p=p, N=8, kind="rb")),
                                      fld(one(sweep="solver", rep="1013", npy=2 * n, p=p, N=8, kind="rb")))) for n in (8, 16)))
    for p in (8, 12, 16)))
print("  p-difference at npy=32: |X(12)-X(16)| %.1e, |X(8)-X(12)| %.1e" % (
    rel(fld(one(sweep="solver", rep="1013", npy=32, p=12, N=8, kind="rb")), fld(ref)),
    rel(fld(one(sweep="solver", rep="1013", npy=32, p=8, N=8, kind="rb")),
        fld(one(sweep="solver", rep="1013", npy=32, p=12, N=8, kind="rb")))))
DISC = {p: E[("eta", "final", p, 32)] for p in (8, 12)}
DISC[16] = E[("eta", "final", 12, 32)] * (E[("eta", "final", 16, 16)] / E[("eta", "final", 12, 16)])
print("  discretization error at npan_y=32 (p=16 extrapolated by the p12->p16 ratio at npy=16): "
      + " ".join("p%d %.1e" % (p, DISC[p]) for p in (8, 12, 16)))
print("  mass drift rel (rb, wall leakage / int|eta|): " + "  ".join(
    "p%d: %s" % (p, " ".join(g(one(sweep="solver", rep="1013", npy=n, p=p, N=8, kind="rb")["mass_rel"]) for n in (8, 16, 32))) for p in (8, 12, 16)))

print("=" * 100)
print("C. RANK PROPERTIES (rankiso, npan_y=32; error vs rb on the same layout, geometric mean of the two reps)")


def herr(k, p, N, rank, f="eta", t="final"):
    es = []
    for rp in ("1013", "1018"):
        if k == "both":
            x = one(sweep="solver", rep=rp, npy=32, p=p, N=N, kind="both", rk=rank) or \
                one(sweep="rankiso", rep=rp, npy=32, p=p, N=N, rk=rank, rbrk=rank)
        elif k == "smap":
            x = one(sweep="rankiso", rep=rp, npy=32, p=p, N=N, kind="smap", rk=rank)
        else:
            x = one(sweep="rankiso", rep=rp, npy=32, p=p, N=N, solver="rbhbs", rk=int(1.5 * p), rbrk=rank)
        if x is None:
            return np.nan
        es.append(rel(fld(x, f, t), fld(rbof(x), f, t)))
    return float(np.exp(np.mean(np.log(es))))


RANKS = (4, 6, 8, 12)
for p in (8, 12, 16):
    print("  p=%d (disc. err %.1e)" % (p, DISC[p]))
    for k in ("smap", "solver"):
        for N in (8, 16, 32):
            e = [herr(k, p, N, r) for r in RANKS]
            b = -np.polyfit(RANKS, np.log10(e), 1)[0]
            need = np.interp(np.log10(DISC[p]), -np.log10(e), RANKS) if min(e) < DISC[p] else np.nan
            print("    %-6s N=%2d: %s   decay %.2f decades/rank   rank for err=disc: %s"
                  % (k, N, " ".join(g(x) for x in e), b, "%.1f" % need if need == need else (">12" if min(e) > DISC[p] else "<4")))
print("  -- additivity: both(r) / max(smap(r), solver(r)) and both / (smap + solver)")
for p, rs in ((8, (4, 8, 12)), (12, (6, 12)), (16, (8,))):
    for r in rs:
        for N in (8, 16, 32):
            a, s, b = herr("smap", p, N, r), herr("solver", p, N, r), herr("both", p, N, r)
            print("    p=%2d r=%2d N=%2d: smap %s solver %s both %s   both/max %.2f  both/sum %.2f"
                  % (p, r, N, g(a), g(s), g(b), b / max(a, s), b / (a + s)))
print("  -- velocity / eta error ratio, and solver-check vs true error (solver variants)")
vr, cr, mr = [], [], []
for x in find(sweep="rankiso", kind="solver"):
    if x["rbrk"] == int(1.5 * x["p"]):
        continue
    rb = rbof(x)
    ee = rel(fld(x), fld(rb)); ve = rel(fld(x, "v"), fld(rb, "v")); me = rel(fld(x, "eta", "mid"), fld(rb, "eta", "mid"))
    vr.append(ve / ee); cr.append(ee / x["check"]); mr.append(me / ee)
print("    v/eta error: median %.1f [%.1f..%.1f];  eta error / check: median %.0f [%.0f..%.0f];  mid/final: median %.2f [%.2f..%.2f]"
      % (np.median(vr), *np.percentile(vr, [10, 90]), np.median(cr), *np.percentile(cr, [10, 90]), np.median(mr), *np.percentile(mr, [10, 90])))
for k in ("smap", "solver"):
    vr = [rel(fld(x, "v"), fld(rbof(x), "v")) / rel(fld(x), fld(rbof(x))) for x in find(sweep="rankiso", kind=k)]
    print("    %-6s v/eta error ratio: median %.1f [%.1f..%.1f]" % (k, np.median(vr), *np.percentile(vr, [10, 90])))
print("  -- mass drift excess |mass_rel - rb mass_rel| vs eta error (log-log slope, correlation)")
for k in ("smap", "solver"):
    xs, ys = [], []
    for x in find(sweep="rankiso", kind=k):
        xs.append(rel(fld(x), fld(rbof(x)))); ys.append(abs(x["mass_rel"] - rbof(x)["mass_rel"]))
    xs, ys = np.log10(xs), np.log10(np.maximum(ys, 1e-12))
    print("    %-6s slope %.2f corr %.2f; excess/err median %.0f" % (k, np.polyfit(xs, ys, 1)[0], np.corrcoef(xs, ys)[0, 1], np.median(10 ** (ys - xs))))

print("=" * 100)
print("D. PERFORMANCE")
step = lambda r: r["t_rhs"] + r["t_solve"] + r["t_recon"]
leaf = lambda r: r["t_rhs"] + r["t_recon"]
print("  -- timing reproducibility 1013 vs 1018 (ratio 1018/1013), all matched runs")
pairs = [(r, one(sweep=r["sweep"], rep="1018", npy=r["npy"], p=r["p"], N=r["N"], solver=r["solver"], rk=r["rk"], rbrk=r["rbrk"]))
         for r in find(rep="1013")]
for nm, fn in (("rhs+recon/step", leaf), ("solve/step", lambda r: r["t_solve"]), ("assemble", lambda r: r["asm"]), ("factor", lambda r: r["fac"])):
    q = np.array([fn(b) / fn(a) for a, b in pairs])
    print("    %-15s median %.2f [10%%..90%%: %.2f..%.2f]  max dev %.0f%%" % (nm, np.median(q), *np.percentile(q, [10, 90]), 100 * np.max(np.abs(q - 1))))
print("    within-run step-time variation (CV of t_rhs): median %.3f%%" % (100 * np.median([r["t_rhs_cv"] for r in rows])))
print("  -- share of conservative-form run time: setup vs 800 steps (rhs / solve / recon)")
for p in (8, 12, 16):
    for N in (8, 32):
        for k in ("rb", "both"):
            x = find(sweep="solver", rep="1013", npy=32, p=p, N=N, kind=k)
            x = x[0] if k == "rb" else [y for y in x if y["rk"] == p][0]
            tot = x["asm"] + x["fac"] + x["keep"] + 800 * step(x)
            print("    p=%2d N=%2d %-4s total(1 form) %6.0f s: assemble %4.1f%% factor %4.2f%% keep %4.1f%% | rhs %4.1f%% solve %4.2f%% recon %4.1f%%"
                  % (p, N, k if k == "rb" else "hbs" + str(p), tot, *(100 * v / tot for v in
                     (x["asm"], x["fac"], x["keep"], 800 * x["t_rhs"], 800 * x["t_solve"], 800 * x["t_recon"]))))
print("  -- chunk-sizing pathology: rhs+recon/step of rb vs min over compressed variants, same layout")
for npy, p, N in [(32, p, N) for p in (8, 12, 16) for N in (8, 16, 32)]:
    rr = [leaf(x) for x in find(npy=npy, p=p, N=N, kind="rb")]
    cc = [leaf(x) for x in find(npy=npy, p=p, N=N) if x["kind"] != "rb"]
    print("    p=%2d N=%2d: rb %.2f..%.2f   compressed min %.2f max %.2f   rb/min %.2f" % (p, N, min(rr), max(rr), min(cc), max(cc), min(rr) / min(cc)))
print("  -- per-step leaf cost vs DoF (min over all variants and reps per layout, N = 8)")
dofs, ts = [], []
for p in (8, 12, 16):
    for npy in (8, 16, 32):
        x = find(npy=npy, p=p, N=8)
        nx = 8 * (4 * npy // 8) // 2
        dof = nx * npy * p * p + p * nx * (2 * npy + 1)
        t = min(leaf(y) for y in x)
        dofs.append(dof); ts.append(t)
        print("    p=%2d npy=%2d DoF %7d: %.3f s/step  (%.2f us/DoF)" % (p, npy, dof, t, 1e6 * t / dof))
print("    fit t ~ DoF^%.2f" % np.polyfit(np.log(dofs), np.log(ts), 1)[0])
print("  -- layout at fixed DoF (npan_y=32): N = 8 / 16 / 32, min over variants & reps")
for p in (8, 12, 16):
    print("    p=%2d  rhs+recon %s | dense assemble %s | dense factor %s | rb solve/step %s | hbs factor (r=p) %s | hbs solve/step %s" % (
        p, "/".join("%.2f" % min(leaf(y) for y in find(npy=32, p=p, N=N)) for N in (8, 16, 32)),
        "/".join("%.1f" % min(y["asm"] for y in find(npy=32, p=p, N=N, kind="rb")) for N in (8, 16, 32)),
        "/".join("%.2f" % min(y["fac"] for y in find(npy=32, p=p, N=N, kind="rb")) for N in (8, 16, 32)),
        "/".join("%.4f" % min(y["t_solve"] for y in find(npy=32, p=p, N=N, kind="rb")) for N in (8, 16, 32)),
        "/".join("%.2f" % min(y["fac"] for y in find(npy=32, p=p, N=N, kind="both", rk=p)) for N in (8, 16, 32)),
        "/".join("%.3f" % min(y["t_solve"] for y in find(npy=32, p=p, N=N, kind="both", rk=p)) for N in (8, 16, 32))))
print("  -- S-map assembly time vs rank (smap variants, rankiso, mean of reps), dense in brackets")
for p in (8, 12, 16):
    for N in (8, 32):
        a = [np.mean([y["asm"] for y in find(sweep="rankiso", npy=32, p=p, N=N, kind="smap", rk=r)]) for r in RANKS]
        d = np.mean([y["asm"] for y in find(sweep="rankiso", npy=32, p=p, N=N, kind="rb")])
        print("    p=%2d N=%2d: %s  [dense %.1f]  nc=%d" % (p, N, " ".join("%5.1f" % v for v in a), d, 32 * p))
print("  -- HBS solver factor and solve/step vs RB_RK (solver variants), N=32")
for p in (8, 12, 16):
    xs = [(r, np.mean([y["fac"] for y in find(sweep="rankiso", npy=32, p=p, N=32, solver="rbhbs", rk=int(1.5 * p), rbrk=r)]),
           np.mean([y["t_solve"] for y in find(sweep="rankiso", npy=32, p=p, N=32, solver="rbhbs", rk=int(1.5 * p), rbrk=r)])) for r in RANKS]
    print("    p=%2d: " % p + "  ".join("r%d fac %.2f solve %.3f" % x for x in xs))

print("=" * 100)
print("E. ACCURACY vs COST (total eta error vs reference; cost = setup + 800 steps, conservative form, min over reps)")
pts = []
for x in find(rep="1013"):
    y = one(sweep=x["sweep"], rep="1018", npy=x["npy"], p=x["p"], N=x["N"], solver=x["solver"], rk=x["rk"], rbrk=x["rbrk"])
    if x is ref:
        continue
    e = rel(fld(x), fld(ref))
    c = min(z["asm"] + z["fac"] + z["keep"] + 800 * step(z) for z in (x, y))
    pts.append((c, e, x))
pts.sort(key=lambda t: t[0])
front, best = [], np.inf
for c, e, x in pts:
    if e < best:
        best = e; front.append((c, e, x))
print("  Pareto front (cheapest run that beats every cheaper one):")
for c, e, x in front:
    print("    %7.0f s  err %.1e   p=%2d npy=%2d N=%2d %s rk=%d rbrk=%d" % (c, e, x["p"], x["npy"], x["N"], x["kind"], x["rk"], x["rbrk"]))
pickle.dump(dict(DISC=DISC, E=E, pts=[(c, e, {k: x[k] for k in ("p", "npy", "N", "kind", "rk", "rbrk", "sweep")}) for c, e, x in pts],
                 front=[(c, e, {k: x[k] for k in ("p", "npy", "N", "kind", "rk", "rbrk")}) for c, e, x in front]),
            open(os.path.join(OUT, "eda_summary.pkl"), "wb"))
