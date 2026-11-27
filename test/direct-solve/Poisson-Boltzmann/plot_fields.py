"""
plot_fields: figures of a potential phi on the stellarator from a fields .npz
written by pb.fieldviz (test_periodic_vanishing_edge.py --field-plots, or
test_field_plots.py).  numpy + matplotlib only.

  <prefix>_sections.png  phi on poloidal cross-sections over one field period
                         (filled, shared colour scale; thin black: phi contours;
                         dashed white: flux surfaces psi = 0.2 .. 0.8; thick black:
                         the wall).  phi contours ON the dashed lines = phi is a
                         flux function there; where they cross, phi is 3D.
  <prefix>_wall3d.png    a E_n = -a d phi / d n on the wall (a = minor radius, n the
                         outward normal), full torus.
  <prefix>_wallmap.png   the same field unrolled: toroidal angle (in field periods)
                         x VMEC poloidal angle theta; dashed: the section cuts.
Colour maps: diverging (centred at 0) when the field changes sign, else sequential.

    python plot_fields.py results_.../fields_<tag>.npz [--prefix NAME] [--patches]
"""
import argparse
import os

import numpy as np


def _tri(ns, npatch):
    i, j = np.meshgrid(np.arange(ns - 1), np.arange(ns - 1), indexing="ij")
    a = (i * ns + j).ravel()
    t = np.r_[np.c_[a, a + ns, a + ns + 1], np.c_[a, a + ns + 1, a + 1]]
    return np.vstack([t + k * ns * ns for k in range(npatch)])


def color_scale(v):
    """(cmap name, vmin, vmax): diverging and symmetric when v changes sign."""
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    big = max(abs(lo), abs(hi), 1e-300)
    if lo < -0.02 * big and hi > 0.02 * big:
        return "RdBu_r", -big, big
    return "viridis", lo, hi


def _zeta_label(k, n):
    try:
        from plot_stellarator_boundary import zeta_label
        return zeta_label(k, n)
    except ImportError:
        return r"$\zeta = 2\pi \cdot %d/%d$" % (k, n)


def plot_sections(d, out, patches=False, ncol=3):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    R, Z, phi, psi = d["sec_R"], d["sec_Z"], d["sec_phi"], d["sec_psi"]
    ncut, npatch, ns = R.shape[:3]
    tri = _tri(ns, npatch)
    cmap, vmin, vmax = color_scale(phi)
    levels = np.linspace(vmin, vmax, 25)
    lo = np.array([[R[k].min(), Z[k].min()] for k in range(ncut)])
    hi = np.array([[R[k].max(), Z[k].max()] for k in range(ncut)])
    span = 1.06 * (hi - lo).max()
    nrow = int(np.ceil(ncut / ncol))
    tile = 4.6
    fig, axs = plt.subplots(nrow, min(ncol, ncut), figsize=(tile * min(ncol, ncut) + 1.2, tile * nrow),
                            squeeze=False, constrained_layout=True)
    for k in range(ncut):
        a = axs[k // ncol][k % ncol]
        T = mtri.Triangulation(R[k].ravel(), Z[k].ravel(), tri)
        a.tricontourf(T, phi[k].ravel(), levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both")
        a.tricontour(T, phi[k].ravel(), levels=levels[1:-1:2], colors="k", linewidths=0.5)
        a.tricontour(T, psi[k].ravel(), levels=[0.2, 0.4, 0.6, 0.8], colors="w", linewidths=0.9,
                     linestyles="--")
        if patches:
            for j in range(npatch):
                for e in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
                    a.plot(R[k, j][e], Z[k, j][e], color="0.35", lw=0.5)
        wall = np.concatenate([np.c_[R[k, j, -1, :], Z[k, j, -1, :]] for j in range(1, npatch)])
        a.plot(np.r_[wall[:, 0], wall[:1, 0]], np.r_[wall[:, 1], wall[:1, 1]], "k-", lw=1.6)
        c = (lo[k] + hi[k]) / 2
        a.set_xlim(c[0] - span / 2, c[0] + span / 2)
        a.set_ylim(c[1] - span / 2, c[1] + span / 2)
        a.set_aspect("equal", adjustable="box")
        a.set_title(_zeta_label(k, ncut), fontsize=18)
        a.set_xlabel(r"$R$"); a.set_ylabel(r"$Z$")
    for k in range(ncut, nrow * axs.shape[1]):
        axs[k // ncol][k % ncol].axis("off")
    fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), ax=axs.ravel().tolist(), shrink=0.85, label=r"$\phi$")
    lab = str(d["label"]) if "label" in d else ""
    if lab:
        fig.suptitle(lab, fontsize=12)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def _wall_closed(d):
    q, aE = d["wall_q"], float(d["aref"]) * d["wall_En"]
    q = np.concatenate([q, q[:, :1]], axis=1)            # close in theta
    aE = np.concatenate([aE, aE[:, :1]], axis=1)
    q = np.concatenate([q, q[:1]], axis=0)               # close in t (periodic)
    aE = np.concatenate([aE, aE[:1]], axis=0)
    return q, aE


def plot_wall3d(d, out, elev=35, azim=-60):
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    q, aE = _wall_closed(d)
    cmap, vmin, vmax = color_scale(aE)
    norm = Normalize(vmin, vmax)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_axes([0, 0, 0.86, 1], projection="3d")
    ax.set_axis_off()
    ax.plot_surface(q[..., 0], q[..., 1], q[..., 2], facecolors=colormaps[cmap](norm(aE)),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    lo, hi = q.reshape(-1, 3).min(0), q.reshape(-1, 3).max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect(tuple(hi - lo), zoom=1.3)
    except TypeError:
        ax.set_box_aspect(tuple(hi - lo))
    cax = fig.add_axes([0.87, 0.25, 0.02, 0.5])
    fig.colorbar(ScalarMappable(norm, cmap), cax=cax, label=r"$a\,E_n = -a\,\partial\phi/\partial n$")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def plot_wall_map(d, out):
    import matplotlib.pyplot as plt
    nfp = int(d["nfp"])
    aE = float(d["aref"]) * d["wall_En"]
    x = d["wall_tor"] * nfp / (2 * np.pi)                 # field periods
    th = d["wall_theta"]
    cmap, vmin, vmax = color_scale(aE)
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    pc = ax.pcolormesh(x, th / np.pi, aE.T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    for a in np.asarray(d["angs"]) * nfp / (2 * np.pi):
        ax.axvline(a, color="k", lw=0.6, ls="--")
    ax.set_xlabel("toroidal angle (field periods)")
    ax.set_ylabel(r"VMEC poloidal angle $\theta / \pi$")
    fig.colorbar(pc, ax=ax, label=r"$a\,E_n$")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def plot_all(d, prefix, patches=False):
    import matplotlib
    matplotlib.use("Agg")
    plot_sections(d, prefix + "_sections.png", patches=patches)
    plot_wall3d(d, prefix + "_wall3d.png")
    plot_wall_map(d, prefix + "_wallmap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--prefix", default=None, help="output prefix (default: the npz path without .npz)")
    ap.add_argument("--patches", action="store_true", help="draw the patch outlines on the sections")
    args = ap.parse_args()
    d = dict(np.load(args.npz, allow_pickle=False))
    plot_all(d, args.prefix or os.path.splitext(args.npz)[0], patches=args.patches)


if __name__ == "__main__":
    main()
