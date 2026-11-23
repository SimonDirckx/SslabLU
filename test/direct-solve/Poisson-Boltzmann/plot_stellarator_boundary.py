"""
plot_stellarator_boundary: inspect the stellarator geometry (default: W7-X from input.w7x)
(geometry/geom_3D/stellerator.py).

Figure 1: the boundary surface (rho = 1) over the full torus, coloured by the
          height Z; shape only (no axes, background or title), transparent PNG.
Figure 2: poloidal cross-sections at several toroidal angles over one field
          period (R, Z plane), labelled by zeta = n_fp x (toroidal angle) in
          [0, 2 pi) (phi is reserved for the potential): boundary, flux surfaces rho = 0.25 .. 0.75, the
          axis, and the 5-patch cross-section (core square and the four arch
          seams), i.e. exactly what the HPS leaves are mapped onto.  Crossing
          coordinate lines would mean a folded map.
Also prints the geometry checks (stellerator.check_param) unless --no-check.

Run from this folder with the repo root on PYTHONPATH:
    PYTHONPATH=../../.. python plot_stellarator_boundary.py [--show] [--out stellarator.png]
    ... --toy            the toy W7-X-like boundary;  --mmax/--nmax truncate the spectrum
    ... --twisted        same plots for the twisted elliptic torus (Fourier form)
"""
import argparse

import numpy as np


def surface(torus, rho, nt, nth):
    """Physical points (nt, nth, 3) of the surface rho = const."""
    t = np.linspace(0, torus.L, nt)
    th = np.linspace(0, 2 * np.pi, nth)
    T, TH = np.meshgrid(t, th, indexing="ij")
    p = np.stack([T, torus.Rc * rho * np.cos(TH), torus.Rc * rho * np.sin(TH)], -1)
    return np.stack(torus.forward(p), -1)


def cross_section(torus, ang, pts):
    """(R, Z) of reference cross-section points pts (n, 2) at toroidal angle ang
    (the cylindrical angle, in [0, 2 pi))."""
    t = torus.L * ang / (2 * np.pi)
    p = np.c_[np.full(len(pts), t), pts]
    R, Z = torus.fields(p, order=0)
    return R[""], Z[""]


def zeta_label(k, n):
    """LaTeX title for cut k of n over one field period: zeta = n_fp * (toroidal
    angle) = 2 pi k / n, written as a multiple of pi (zeta, not phi: phi is the
    potential)."""
    from fractions import Fraction
    f = Fraction(2 * k, n)
    if f == 0:
        return r"$\zeta = 0$"
    num = "" if f.numerator == 1 else str(f.numerator)
    return r"$\zeta = %s\pi$" % num if f.denominator == 1 else r"$\zeta = \frac{%s\pi}{%d}$" % (num, f.denominator)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stellarator_boundary.png", help="output file prefix (.png)")
    ap.add_argument("--show", action="store_true", help="open interactive windows")
    ap.add_argument("--nt", type=int, default=400, help="toroidal resolution of the surface")
    ap.add_argument("--nth", type=int, default=64, help="poloidal resolution of the surface")
    ap.add_argument("--ncut", type=int, default=6, help="cross-sections over one field period")
    ap.add_argument("--input", default=None, help="VMEC input file (default: input.w7x via stellerator.DEFAULT_INPUT)")
    ap.add_argument("--toy", action="store_true", help="plot the toy W7-X-like boundary instead")
    ap.add_argument("--mmax", type=int, default=None, help="truncate the boundary spectrum to m <= mmax")
    ap.add_argument("--nmax", type=int, default=None, help="truncate the boundary spectrum to |n| <= nmax")
    ap.add_argument("--twisted", action="store_true", help="plot the twisted elliptic torus instead")
    ap.add_argument("--no-check", action="store_true", help="skip stellerator.check_param")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    import geometry.geom_3D.stellerator as st
    if args.twisted:
        import geometry.geom_3D.full_twisted_torus as ftt
        torus = st.FourierTorus(st.twisted_torus_modes(ftt.R0, ftt.a, ftt.b, int(ftt.twist)), 1, ftt.Rc, ftt.s)
        name = "twisted elliptic torus"
    else:
        torus = st.build("toy" if args.toy else args.input, mmax=args.mmax, nmax=args.nmax)
        name = "%s (nfp = %d)" % ("toy stellarator" if args.toy else "W7-X" if args.input is None
                                  else args.input, torus.nfp)
    print("%s: R00 = %.3f, Rc = %.3f, L = %.3f, %d boundary modes (m > 0), axis terms (n, Rax, Rbd, Zax, Zbd):"
          % (name, torus.R00, torus.Rc, torus.L, len(torus.modes)))
    for a in torus.axis_terms:
        print("   ", a)
    if not args.no_check:
        st.check_param(torus=torus)

    # ---------------- figure 1: 3D boundary
    # shape only: no axes, panes, title or background; axes span the whole
    # figure, limits hug the data with true proportions
    q = surface(torus, 1.0, args.nt, args.nth)
    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    zc = (q[..., 2] - q[..., 2].min()) / max(np.ptp(q[..., 2]), 1e-12)
    ax.plot_surface(q[..., 0], q[..., 1], q[..., 2], facecolors=cm.viridis(zc), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=True)
    lo, hi = q.reshape(-1, 3).min(0), q.reshape(-1, 3).max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=35, azim=-60)
    try:
        ax.set_box_aspect(tuple(hi - lo), zoom=1.35)       # zoom needs matplotlib >= 3.6
    except TypeError:
        ax.set_box_aspect(tuple(hi - lo))
    ax.margins(0)
    out3d = args.out.replace(".png", "_3d.png")
    fig.savefig(out3d, dpi=200, bbox_inches="tight", pad_inches=0, transparent=True)
    print("saved", out3d)

    # ---------------- figure 2: cross-sections over one field period
    nfp = torus.nfp
    angs = np.linspace(0, 2 * np.pi / nfp, args.ncut, endpoint=False)   # toroidal angles
    th = np.linspace(0, 2 * np.pi, 400)
    Rc, s = torus.Rc, torus.s
    lin = np.linspace(-s * Rc, s * Rc, 50)
    core_edges = [np.c_[lin, np.full_like(lin, c)] for c in (-s * Rc, s * Rc)] + \
                 [np.c_[np.full_like(lin, c), lin] for c in (-s * Rc, s * Rc)]
    seams = []                                         # arch seams: core corner -> wall (blend map)
    for cx, cy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        x = np.linspace(-1, 1, 50)
        u = (1 - x) / 2 * s + (1 + x) / 2 * np.cos(np.pi / 4)
        seams.append(Rc * np.c_[cx * u, cy * u])
    ncol = min(3, args.ncut)
    nrow = int(np.ceil(args.ncut / ncol))
    # all curves first: every panel gets the same data span (centred on its own
    # cross-section) and equal aspect, so all panels have the same size and scale
    panels = []
    for ang in angs:
        curves = []
        for rho, sty in ((1.0, "k-"), (0.75, "C0-"), (0.5, "C0-"), (0.25, "C0-")):
            curves.append((cross_section(torus, ang, Rc * rho * np.c_[np.cos(th), np.sin(th)]),
                           dict(color=sty[:-1], lw=1.6 if rho == 1 else 0.7)))
        for j in range(16):                            # theta coordinate lines
            r = np.linspace(0, Rc, 30)
            curves.append((cross_section(torus, ang, np.c_[r * np.cos(j * np.pi / 8), r * np.sin(j * np.pi / 8)]),
                           dict(color="0.75", lw=0.5)))
        for e in core_edges + seams:
            curves.append((cross_section(torus, ang, e), dict(color="C3", lw=1.0)))
        panels.append((ang, curves, cross_section(torus, ang, np.zeros((1, 2)))))
    lo = np.array([[min(c[0][0].min() for c in cv), min(c[0][1].min() for c in cv)] for _, cv, _ in panels])
    hi = np.array([[max(c[0][0].max() for c in cv), max(c[0][1].max() for c in cv)] for _, cv, _ in panels])
    span = 1.06 * max((hi - lo).max(axis=0).max(), 1e-12)   # one square window for all panels
    fig2, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow), squeeze=False)
    for k, (ang, curves, ax_pt) in enumerate(panels):
        a2 = axs[k // ncol][k % ncol]
        for (R, Z), kw in curves:
            a2.plot(R, Z, "-", **kw)
        a2.plot(ax_pt[0], ax_pt[1], "r+", ms=10)
        c = (lo[k] + hi[k]) / 2
        a2.set_xlim(c[0] - span / 2, c[0] + span / 2)
        a2.set_ylim(c[1] - span / 2, c[1] + span / 2)
        a2.set_aspect("equal", adjustable="box")
        a2.set_title(zeta_label(k, args.ncut))
        a2.set_xlabel(r"$R$"); a2.set_ylabel(r"$Z$")
    for k in range(args.ncut, nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    fig2.tight_layout()
    out2d = args.out.replace(".png", "_cuts.png")
    fig2.savefig(out2d, dpi=150, bbox_inches="tight")
    print("saved", out2d)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
