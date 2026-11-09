"""Table and helpers for manuscript_figs.py (run manuscript_load.py first).
Each row is one run; `kind` is rb (dense) | smap (rb on HBS S-maps) |
solver (rbhbs, S-maps at RK = 1.5p) | both (rbhbs with RK = RB_RK)."""
import os
import pickle
import numpy as np
from eda_common import OUT

ROWS_PKL = os.path.join(OUT, "manuscript_rows.pkl")
if not os.path.exists(ROWS_PKL):
    raise SystemExit("%s not found: run manuscript_load.py first" % ROWS_PKL)
_D = pickle.load(open(ROWS_PKL, "rb"))
rows, ROOTS = _D["rows"], _D["roots"]


def kind(r):
    if r["solver"] == "rb":
        return "rb" if r["rk"] == 0 else "smap"
    return "both" if r["sweep"] == "solver" else ("solver" if r["rk"] == int(1.5 * r["p"]) else "both")


for r in rows:
    r["kind"] = kind(r)
    r["leaf"] = r["t_rhs"] + r["t_recon"]               # rhs + reconstruction phases, per step
    r["step"] = r["leaf"] + r["t_solve"]
    r["once"] = r["asm"] + r["fac"] + r["keep"]
    r["cost"] = r["once"] + 800 * r["step"]             # conservative form only
    nx = 2 * r["npy"]                                   # leaves are 1:2, 2 npan_y across x
    r["dof"] = nx * r["npy"] * r["p"] ** 2 + r["p"] * nx * (2 * r["npy"] + 1)


def fld(r, f="eta", t="final"):
    a = r["%s_%s_320" % (f, t)]
    return a - r["mean_" + t] if f == "eta" else a


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def find(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def one(**kw):
    x = find(**kw)
    return x[0] if x else None


def rbof(r):
    return one(sweep=r["sweep"], npy=r["npy"], p=r["p"], N=r["N"], kind="rb")


def rbany(npy, p, N=8):
    """Dense rb run of this layout, from the solver sweep if it has one."""
    return one(sweep="solver", npy=npy, p=p, N=N, kind="rb") or one(sweep="rankiso", npy=npy, p=p, N=N, kind="rb")


# self-convergence reference: the dense N = 8 run with the most DoF. The other
# dense N = 8 run closest to it bounds its error (REF_ERR = their difference,
# an upper estimate when that run is the second most accurate)
_fine = sorted({(r["dof"], r["npy"], r["p"]) for r in rows if r["kind"] == "rb" and r["N"] == 8})
REF = rbany(*_fine[-1][1:])
REF_ERR, REF2 = min(((rel(fld(rbany(n, p)), fld(REF)), rbany(n, p)) for _, n, p in _fine[:-1]), key=lambda t: t[0])
