"""
plot_wout_surfaces: VMEC's own interior flux surfaces from a wout_*.nc file, at
the same toroidal cuts as plot_stellarator_boundary.py (zeta = nfp phi over one
field period), compared with the rho^m fill of stellerator.FourierTorus.

Solid: VMEC surfaces s = rho^2 for rho = 0.25, 0.5, 0.75, 1 (coefficients
linearly interpolated in s between the ns surfaces), grey: VMEC theta lines,
red +: VMEC axis.  Dashed: the same rho surfaces of the current FourierTorus map
(rho^m fill of the boundary), i.e. what the HPS patches are mapped through.

    PYTHONPATH=../../.. python plot_wout_surfaces.py [--input wout_csx.nc] [--out csx_vmec_surfaces.png]
"""
import argparse

import numpy as np


def load(path):
    from scipy.io import netcdf_file
    with netcdf_file(path, "r", mmap=False) as f:
        v = f.variables
        return dict(nfp=int(v["nfp"].data), xm=np.asarray(v["xm"].data), xn=np.asarray(v["xn"].data),
                    rmnc=np.asarray(v["rmnc"].data), zmns=np.asarray(v["zmns"].data))


def coeffs_at(w, s):
    """rmnc, zmns at flux label s in [0, 1] (linear in s between surfaces)."""
    ns = w["rmnc"].shape[0]
    x = s * (ns - 1)
    j = min(int(np.floor(x)), ns - 2)
    a = x - j
    return ((1 - a) * w["rmnc"][j] + a * w["rmnc"][j + 1],
            (1 - a) * w["zmns"][j] + a * w["zmns"][j + 1])


def RZ(w, rc, zs, theta, phi):
    ang = np.outer(theta, w["xm"]) - w["xn"] * phi
    return np.cos(ang) @ rc, np.sin(ang) @ zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="wout_csx.nc")
    ap.add_argument("--out", default="csx_vmec_surfaces.png")
    ap.add_argument("--ncut", type=int, default=6)
    ap.add_argument("--no-map", action="store_true", help="do not overlay the FourierTorus surfaces")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_stellarator_boundary import zeta_label, cross_section

    w = load(args.input)
    nfp = w["nfp"]
    torus = None
    if not args.no_map:
        import geometry.geom_3D.stellerator as st
        torus = st.build(args.input)
    zetas = np.linspace(0, 2 * np.pi, args.ncut, endpoint=False)
    th = np.linspace(0, 2 * np.pi, 400)
    rhos = (0.25, 0.5, 0.75, 1.0)
    sgrid = np.linspace(0, 1, 60)

    panels = []
    for zeta in zetas:
        phi = zeta / nfp
        cv = []
        for k in range(16):                                    # VMEC theta lines
            tk = np.array([k * np.pi / 8])
            pts = np.array([RZ(w, *coeffs_at(w, s), tk, phi) for s in sgrid])[:, :, 0]
            cv.append(((pts[:, 0], pts[:, 1]), dict(color="0.75", lw=0.5, ls="-")))
        for rho in rhos:                                       # VMEC flux surfaces
            cv.append((RZ(w, *coeffs_at(w, rho ** 2), th, phi),
                       dict(color="k" if rho == 1 else "C0", lw=1.6 if rho == 1 else 0.9, ls="-")))
        if torus is not None:                                  # current rho^m map
            for rho in rhos[:-1]:
                cv.append((cross_section(torus, phi, torus.Rc * rho * np.c_[np.cos(th), np.sin(th)]),
                           dict(color="C1", lw=0.9, ls="--")))
        ax_pt = RZ(w, *coeffs_at(w, 0.0), np.zeros(1), phi)
        panels.append((cv, ax_pt))

    lo = np.array([[min(c[0][0].min() for c in cv), min(c[0][1].min() for c in cv)] for cv, _ in panels])
    hi = np.array([[max(c[0][0].max() for c in cv), max(c[0][1].max() for c in cv)] for cv, _ in panels])
    span = 1.06 * (hi - lo).max()
    ncol = min(3, args.ncut)
    nrow = int(np.ceil(args.ncut / ncol))
    tile = 5.0
    fig, axs = plt.subplots(nrow, ncol, figsize=(tile * ncol, tile * nrow), squeeze=False)
    for k, (cv, ax_pt) in enumerate(panels):
        a = axs[k // ncol][k % ncol]
        for (R, Z), kw in cv:
            a.plot(R, Z, **kw)
        a.plot(ax_pt[0], ax_pt[1], "r+", ms=10)
        c = (lo[k] + hi[k]) / 2
        a.set_xlim(c[0] - span / 2, c[0] + span / 2)
        a.set_ylim(c[1] - span / 2, c[1] + span / 2)
        a.set_aspect("equal", adjustable="box")
        a.set_title(zeta_label(k, args.ncut), fontsize=25)
        a.set_xlabel(r"$R$", fontsize=16); a.set_ylabel(r"$Z$", fontsize=16)
        a.tick_params(labelsize=13)
    for k in range(args.ncut, nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    fig.tight_layout(pad=0.6, w_pad=0.8, h_pad=0.8)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
