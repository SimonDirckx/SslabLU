import numpy as np
"""Replicate noise floor, rank needed per tolerance, p-independence of rank, and
accuracy vs cost on a chunk-clean step cost. Run eda_load.py first."""
import os
from eda_common import *
leaf = lambda r: r["t_rhs"] + r["t_recon"]
twin = lambda r: one(sweep=r["sweep"], rep="1018", npy=r["npy"], p=r["p"], N=r["N"], solver=r["solver"], rk=r["rk"], rbrk=r["rbrk"])

print("1. replicate disagreement d = |X1018 - X1013| vs compression error e = |X - rb| (eta final)")
E, D = [], []
for r in find(rep="1013"):
    if r["kind"] == "rb":
        continue
    E.append(rel(fld(r), fld(rbof(r)))); D.append(rel(fld(twin(r)), fld(r)))
E, D = np.array(E), np.array(D)
for lo, hi in ((1e-3, 1), (1e-5, 1e-3), (1e-7, 1e-5), (1e-8, 1e-7), (1e-10, 1e-8)):
    m = (E >= lo) & (E < hi)
    if m.any():
        print("   e in [%.0e, %.0e): %3d runs, d/e median %.1e  max %.1e;  d median %.1e" % (lo, hi, m.sum(), np.median(D[m] / E[m]), np.max(D[m] / E[m]), np.median(D[m])))

print("2. rank needed for compression error <= discretization error (log-linear between ranks; geometric mean of reps)")
RANKS = (4, 6, 8, 12)
DISC = {8: 5.0e-05, 12: 8.2e-07, 16: 1.0e-08}


def herr(k, p, N, rank):
    es = []
    for rp in ("1013", "1018"):
        if k == "smap":
            x = one(sweep="rankiso", rep=rp, npy=32, p=p, N=N, kind="smap", rk=rank)
        else:
            x = one(sweep="rankiso", rep=rp, npy=32, p=p, N=N, solver="rbhbs", rk=int(1.5 * p), rbrk=rank)
        es.append(rel(fld(x), fld(rbof(x))))
    return float(np.exp(np.mean(np.log(es))))


def crossing(e, tol):
    for i in range(len(RANKS)):
        if e[i] <= tol:
            if i == 0:
                return "<=4"
            a, b = np.log10(e[i - 1]), np.log10(e[i])
            return "%.1f" % (RANKS[i - 1] + (RANKS[i] - RANKS[i - 1]) * (a - np.log10(tol)) / (a - b))
    return ">12"


for p in (8, 12, 16):
    for k in ("smap", "solver"):
        print("   p=%2d %-6s tol %.0e:  N=8 %5s  N=16 %5s  N=32 %5s   | tol 1e-6: %s" % (
            p, k, DISC[p], *(crossing([herr(k, p, N, r) for r in RANKS], DISC[p]) for N in (8, 16, 32)),
            " ".join(crossing([herr(k, p, N, r) for r in RANKS], 1e-6) for N in (8, 16, 32))))

print("3. does fixed absolute rank give a p-independent error? spread (max/min over p) at each rank, N")
for k in ("smap", "solver"):
    for N in (8, 16, 32):
        print("   %-6s N=%2d: " % (k, N) + "  ".join("r%d x%.1f" % (r, max(herr(k, p, N, r) for p in (8, 12, 16)) / min(herr(k, p, N, r) for p in (8, 12, 16))) for r in RANKS))

print("4. accuracy vs cost with chunk-clean step cost (min rhs+recon over every run of the layout)")
clean = {}
for x in rows:
    key = (x["npy"], x["p"], x["N"])
    clean[key] = min(clean.get(key, np.inf), leaf(x))
ref = one(sweep="solver", rep="1013", npy=32, p=16, N=8, kind="rb")
pts = []
for x in rows:
    if x is ref or x["rep"] != "1013":
        continue
    y = twin(x)
    cost = np.mean([z["asm"] + z["fac"] + z["keep"] + 800 * (clean[(z["npy"], z["p"], z["N"])] + z["t_solve"]) for z in (x, y)])
    disc = rel(fld(rbof(x)), fld(ref))                               # rb on this layout
    comp = np.sqrt(rel(fld(x), fld(rbof(x))) * rel(fld(y), fld(rbof(y)))) if x["kind"] != "rb" else 0.0
    pts.append((cost, max(disc, comp), disc, comp, x))
pts.sort(key=lambda t: t[0])
best = np.inf
print("   front on max(discretization error, compression error):")
for c, e, d, cm, x in pts:
    if e < best * 0.9:
        best = e
        print("   %6.0f s  err %.1e (disc %.1e, comp %.1e)  p=%2d npy=%2d N=%2d %-6s RK=%2d RB_RK=%2d"
              % (c, e, d, cm, x["p"], x["npy"], x["N"], x["kind"], x["rk"], x["rbrk"]))
print("   cost of rb vs best compressed variant at equal layout (clean step cost), npan_y=32:")
for p in (8, 12, 16):
    for N in (8, 32):
        rb_ = [t for t in pts if t[4]["kind"] == "rb" and t[4]["npy"] == 32 and t[4]["p"] == p and t[4]["N"] == N][0]
        ok = [t for t in pts if t[4]["kind"] != "rb" and t[4]["npy"] == 32 and t[4]["p"] == p and t[4]["N"] == N and t[3] <= rb_[2]]
        b = min(ok, key=lambda t: t[0]) if ok else None
        print("     p=%2d N=%2d: rb %6.0f s | cheapest compressed with comp <= disc: %s" % (
            p, N, rb_[0], ("%6.0f s (%s RK=%d RB_RK=%d)" % (b[0], b[4]["kind"], b[4]["rk"], b[4]["rbrk"])) if b else "none"))
import pickle
pickle.dump(dict(pts=[(c, e, d, cm, {k: x[k] for k in ("p", "npy", "N", "kind", "rk", "rbrk")}) for c, e, d, cm, x in pts], E=E, D=D),
            open(os.path.join(OUT, "eda_extra.pkl"), "wb"))
