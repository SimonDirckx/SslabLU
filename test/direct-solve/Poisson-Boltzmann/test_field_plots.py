"""
test_field_plots: check the field-plotting pipeline (pb.fieldviz + plot_fields.py)
with PRESCRIBED potentials, no PDE solve.

A leaf grid over the full torus (pb.prolong.LeafGrid, same 5-patch layout and
maps as the solver) gets nodal values of a known phi; the pipeline then
evaluates the leaf polynomials on the cross-sections and differentiates them at
the wall, exactly as for a solver run.  Cases (--cases, default all):

  flux   phi = psi - 1   (a flux function, 0 on the wall)
         sections: every phi contour lies ON a dashed flux surface;
         wall: a E_n = -a |grad psi| < 0 everywhere (sequential map), largest
         in magnitude where the flux surfaces are packed closest to the wall.
  z      phi = Z / a      (linear in the vertical coordinate)
         sections: horizontal phi contours, equally spaced, same in every cut;
         wall: a E_n = -n_Z, -1 at the top, +1 at the bottom (diverging map).
  r      phi = (R - R00) / a   (linear in the major radius)
         sections: vertical phi contours; wall: a E_n = -n_R, -1 on the
         outboard side, +1 inboard.

Numerical checks against references that do NOT use the pipeline's route:
  - sections: phi at the section points vs the formula (interpolation error);
  - wall normal: from the cross product of finite-difference tangents of the
    forward map, oriented by the rho derivative (not J^{-T} grad rho);
  - E_n: central difference of the formula along that normal (z, r); for flux
    -2 / (n . dq/drho), dq/drho by finite differences of the forward map.
Expected: errors decrease with --p (spectral); at the defaults roughly
phi <~ 1e-6, normal <~ 1e-6, E_n (relative) <~ 1e-4.

    PYTHONPATH=../../.. python test_field_plots.py [--input wout_csx.nc] [--n 4 --m 1 --p 8 --nt 32]
Figures and .npz in --out (default results_test_field_plots/).
"""
import argparse
import os
import time

import numpy as np

import geometry.geom_3D.stellerator as st
from pb.prolong import LeafGrid
from pb import fieldviz
import plot_fields


def wall_reference(geom, W, h_rel=1e-6):
    """Outward unit normal at wall points W = (t, Rc cos th, Rc sin th), from
    central-difference tangents of geom.forward (independent of J^{-T})."""
    Rc, L = geom.Rc, geom.L
    t, th = W[:, 0], np.arctan2(W[:, 2], W[:, 1])
    f = lambda tt, aa, rr=1.0: np.stack(geom.forward(np.c_[tt, rr * Rc * np.cos(aa), rr * Rc * np.sin(aa)]), -1)
    ht, ha = h_rel * L, 1e-6
    qt = (f(t + ht, th) - f(t - ht, th)) / (2 * ht)
    qa = (f(t, th + ha) - f(t, th - ha)) / (2 * ha)
    n = np.cross(qt, qa)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    hr = 1e-6
    q_rho = (f(t, th, 1.0 + hr) - f(t, th, 1.0 - hr)) / (2 * hr)   # outward direction
    n *= np.sign(np.einsum("ni,ni->n", n, q_rho))[:, None]
    return f(t, th), n, q_rho


def cases(geom, a, h):
    """name -> (phi at torus reference points W (with q = forward(W)),
    reference E_n(q, n, q_rho) at the wall).  z, r: central difference of the
    formula along the normal.  flux: d psi/dn = 2 rho |grad rho| and, since
    grad rho . dq/drho = 1 with grad rho parallel to n, |grad rho| = 1/(n . dq/drho)
    (no inverse map: Newton on every point is ~3 ms/pt)."""
    Rc = geom.Rc
    psi_W = lambda W: (W[:, 1] ** 2 + W[:, 2] ** 2) / Rc ** 2
    fd = lambda f: (lambda q, n, q_rho: -(f(q + h * n) - f(q - h * n)) / (2 * h))
    z_q = lambda q: q[:, 2] / a
    r_q = lambda q: (np.hypot(q[:, 0], q[:, 1]) - geom.R00) / a
    return {
        "flux": (lambda W, q: psi_W(W) - 1.0,
                 lambda q, n, q_rho: -2.0 / np.einsum("ni,ni->n", n, q_rho)),
        "z": (lambda W, q: z_q(q), fd(z_q)),
        "r": (lambda W, q: r_q(q), fd(r_q)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="wout_csx.nc")
    ap.add_argument("--s", type=float, default=0.55, help="core half-width / Rc (as the PB runs)")
    ap.add_argument("--n", type=int, default=4, help="core leaves per side (arch tangential)")
    ap.add_argument("--m", type=int, default=1, help="arch radial leaves")
    ap.add_argument("--nt", type=int, default=32, help="axial leaves around the torus")
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--cases", default="flux,z,r")
    ap.add_argument("--ncut", type=int, default=6)
    ap.add_argument("--ns", type=int, default=41, help="section samples per patch direction")
    ap.add_argument("--wall-nt", type=int, default=200)
    ap.add_argument("--wall-neta", type=int, default=24)
    ap.add_argument("--patches", action="store_true")
    ap.add_argument("--out", default="results_test_field_plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    geom = st.configure(args.input)
    st.s = args.s
    a = getattr(geom, "aminor", None) or geom.Rc
    t0 = time.perf_counter()
    lg = LeafGrid([[0.0, -geom.Rc, -geom.Rc], [geom.L, geom.Rc, geom.Rc]], args.p, geom.Rc, args.s,
                  args.n, args.m, args.nt)
    W = lg.nodes
    q = np.stack(geom.forward(W), -1)
    print("%s: leaf grid n=%d m=%d nt=%d p=%d, %d nodes (%.1fs); a = %.4f, R00 = %.4f"
          % (args.input, args.n, args.m, args.nt, args.p, len(W), time.perf_counter() - t0, a, geom.R00))

    cs = cases(geom, a, h=1e-5 * a)
    names = [c.strip() for c in args.cases.split(",") if c.strip()]
    ok_all = True
    for name in names:
        phi_W, En_of = cs[name]
        t0 = time.perf_counter()
        samp = fieldviz.FieldSampler.from_leafgrid(lg, phi_W(W, q), geom)
        d = samp.collect(ncut=args.ncut, ns=args.ns, nt=args.wall_nt, neta=args.wall_neta,
                         label="test: %s" % name, aref=a)
        t_col = time.perf_counter() - t0
        # sections: interpolation error vs the formula
        Ws = d["sec_W"].reshape(-1, 3)
        e_sec = np.max(np.abs(d["sec_phi"].ravel() - phi_W(Ws, np.stack(geom.forward(Ws), -1))))
        # wall: normal and E_n vs the independent references
        Ww = d["wall_W"].reshape(-1, 3)
        qw, n_ref, q_rho = wall_reference(geom, Ww)
        e_q = np.max(np.abs(d["wall_q"].reshape(-1, 3) - qw))
        e_n = np.max(np.linalg.norm(d["wall_n"].reshape(-1, 3) - n_ref, axis=1))
        En_ref = En_of(qw, n_ref, q_rho)
        En = d["wall_En"].ravel()
        e_En = np.max(np.abs(En - En_ref)) / np.max(np.abs(En_ref))
        e_wphi = np.max(np.abs(d["wall_phi"].ravel() - phi_W(Ww, qw)))
        scale = np.max(np.abs(d["sec_phi"]))
        ok = e_sec < 1e-4 * scale and e_n < 1e-5 and e_En < 1e-3
        ok_all &= ok
        print("[%s] sections: max |phi - phi_exact| %.2e (|phi| <= %.2f); wall: |q - q_ref| %.1e, "
              "|n - n_ref| %.1e, phi %.1e, a E_n in [%.3f, %.3f], rel. err %.2e  [%.1fs]  %s"
              % (name, e_sec, scale, e_q, e_n, e_wphi, a * En.min(), a * En.max(), e_En, t_col,
                 "PASS" if ok else "CHECK"))
        npz = os.path.join(args.out, "fields_test_%s.npz" % name)
        fieldviz.save(npz, d)
        plot_fields.plot_all(dict(np.load(npz)), os.path.splitext(npz)[0], patches=args.patches)
    print("all numerical checks passed" if ok_all else
          "some checks above tolerance (see CHECK); rerun with a larger --p to see whether they decrease")
    print("visual checks: see the docstring of this file for what each figure must show")


if __name__ == "__main__":
    main()
