"""
probe_psi: convergence of test_periodic_vanishing_edge runs as a function of
psi, from their saved files.

1. Self-convergence (probes_*.npz, --ref + positional runs).  Every run
   evaluates phi at the same probe points (same seed and --nprobe; arch radial
   probe coordinates are unstretched, so runs with any --arch-stretch are
   comparable).  psi is rebuilt from the probe coordinates: psi = (x^2 + y^2)/Rc^2
   does not depend on Rc, only on the core half-width s (torus.s = 0.55 in the
   test).  For each run, |phi_run - phi_ref| is binned in psi.

2. Residual estimator (eta_*.npz from --est-dump, --eta).  The eta samples are
   random points per leaf (so they differ between runs), but each sample carries
   its own psi; per psi bin we print max |eta|, rms |eta| and the bin's share of
   sum eta^2.  Bin statistics are comparable between runs.

    python probe_psi.py --ref probes_<finest>.npz probes_<a>.npz ... \\
                        --eta eta_<a>.npz eta_<b>.npz ...
Either part may be given alone.
"""
import argparse
import os

import numpy as np


def probe_psi(d, s):
    """psi at the probes of a probes_*.npz (patch 0 core, 1..4 arches rot 0..3)."""
    k, u = np.asarray(d["patch"]), np.asarray(d["u"])
    xy = np.empty((len(k), 2))
    c = k == 0
    xy[c] = -s + 2 * s * u[c]                       # core box [-s, s]^2 (units of Rc)
    a = ~c
    x, eta = -1 + 2 * u[a, 0], -1 + 2 * u[a, 1]     # x: unstretched radial blend parameter
    th = np.pi * eta / 4
    U = (1 - x) / 2 * s + (1 + x) / 2 * np.cos(th)
    V = (1 - x) / 2 * s * eta + (1 + x) / 2 * np.sin(th)
    ang = (k[a] - 1) * np.pi / 2
    xy[a, 0] = np.cos(ang) * U - np.sin(ang) * V
    xy[a, 1] = np.sin(ang) * U + np.cos(ang) * V
    return (xy ** 2).sum(axis=1)


def label(d, path):
    m = {k[5:]: d[k] for k in d.files if k.startswith("meta_")}
    if not m:
        return os.path.basename(path)
    p = "-".join(str(int(v)) for v in np.atleast_1d(m.get("p", [])))
    st = float(m["stretch"]) if "stretch" in m else 0.0
    return "n%d m%d p%s Ns%d nz%d st%.3g" % (int(m["n"]), int(m["m"]), p, int(m["Ns"]), int(m["nz"]), st)


def eta_label(path):
    """eta_Ns.._nz.._n.._m.._p.._Lam.._N[_st..].npz -> 'n.. m.. p.. st..'."""
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    keep = [q for q in parts if q[:1] in "nmp" and q[1:2].isdigit() or q.startswith("st")]
    return " ".join(keep) if keep else os.path.basename(path)


def binned(v, b, nbin, f):
    return " ".join("%9.2e" % f(v[b == i]) if np.any(b == i) else "%9s" % "-" for i in range(nbin))


def rms(v):
    return np.sqrt(np.mean(v ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="probe .npz files to compare with --ref")
    ap.add_argument("--ref", help="reference (most accurate) probe .npz")
    ap.add_argument("--eta", nargs="+", default=[], help="eta dump .npz files (--est-dump)")
    ap.add_argument("--s", type=float, default=0.55, help="core half-width / Rc (torus.s)")
    ap.add_argument("--bins", type=str, default="0,0.3,0.6,0.8,0.9,0.95,0.99,1.0",
                    help="psi bin edges")
    args = ap.parse_args()
    if args.runs and not args.ref:
        ap.error("probe runs need --ref")
    if not args.ref and not args.eta:
        ap.error("nothing to do: give --ref (+ runs) and/or --eta")

    edges = np.array([float(v) for v in args.bins.split(",")])
    edges[-1] += 1e-12
    nbin = len(edges) - 1
    head = " ".join("%9s" % ("%.2f-%.2f" % (edges[i], min(edges[i + 1], 1.0))) for i in range(nbin))

    if args.ref:
        ref = np.load(args.ref)
        psi = probe_psi(ref, args.s)
        if psi.max() > 1 + 1e-12:
            raise ValueError("probe psi > 1 (%.3g): wrong --s?" % psi.max())
        b = np.digitize(psi, edges) - 1
        vref = ref["values"]
        print("SELF-CONVERGENCE  |phi_run - phi_ref| at the probes")
        print("reference: %s  (%d probes, max |phi| %.3g)"
              % (label(ref, args.ref), len(psi), np.max(np.abs(vref))))
        print("%-34s %s %9s" % ("psi bin", head, "all"))
        print("%-34s %s %9d" % ("probes per bin", " ".join("%9d" % np.sum(b == i) for i in range(nbin)), len(psi)))
        for path in args.runs:
            d = np.load(path)
            if not (np.array_equal(d["patch"], ref["patch"]) and np.allclose(d["u"], ref["u"])
                    and np.allclose(d["t"], ref["t"])):
                print("%s: different probe points (other --nprobe?), skipped" % path)
                continue
            e = np.abs(d["values"] - vref)
            for name, f in (("max", np.max), ("rms", rms)):
                print("%-34s %s %9.2e" % ("%s  %s" % (label(d, path)[:29], name), binned(e, b, nbin, f), f(e)))
        print()

    if args.eta:
        print("RESIDUAL ESTIMATOR  |eta| at the random interior samples")
        print("%-34s %s %9s" % ("psi bin", head, "all"))
        for path in args.eta:
            d = np.load(path)
            e, pe = np.abs(np.asarray(d["eta"])), np.asarray(d["psi"])
            be = np.digitize(np.clip(pe, 0, 1), edges) - 1
            tot = np.sum(e ** 2)
            lab = eta_label(path)[:24]
            print("%-34s %s %9d" % ("%s  n" % lab, " ".join("%9d" % np.sum(be == i) for i in range(nbin)), len(e)))
            for name, f in (("max", np.max), ("rms", rms)):
                print("%-34s %s %9.2e" % ("%s  %s" % (lab, name), binned(e, be, nbin, f), f(e)))
            share = " ".join("%9.3f" % (np.sum(e[be == i] ** 2) / tot) if np.any(be == i) else "%9s" % "-"
                             for i in range(nbin))
            print("%-34s %s %9.3f" % ("%s  frac eta^2" % lab, share, 1.0))


if __name__ == "__main__":
    main()
