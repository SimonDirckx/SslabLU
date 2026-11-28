"""
plot_fields: figures of a potential phi on the stellarator from a fields .npz
written by pb.fieldviz (test_periodic_vanishing_edge.py --field-plots, or
test_field_plots.py).  numpy + matplotlib only.

  <prefix>_sections.png  phi on poloidal cross-sections over one field period
                         (filled, shared colour scale; thin solid black: phi
                         contours; dashed white: flux surfaces psi = 0.2 .. 0.8;
                         thick black: the wall).  phi contours that never cross
                         the dashed lines = phi is a flux function there; where
                         they cross, phi is 3D.
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


PSI_LINES = (0.2, 0.4, 0.6, 0.8)


def plot_sections(d, out, patches=False, ncol=3, phi_lines=None, psi_lines=True, title=True):
    """phi_lines: phi values of the thin black contours (default: every other
    filled level, which in general do NOT coincide with the dashed psi lines)."""
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
        a.tricontour(T, phi[k].ravel(), levels=levels[1:-1:2] if phi_lines is None else sorted(phi_lines),
                     colors="k", linewidths=0.6, linestyles="solid")
        if psi_lines:
            a.tricontour(T, psi[k].ravel(), levels=list(PSI_LINES), colors="w", linewidths=1.2,
                         linestyles=(0, (4, 3)))
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
    if lab and title:
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


def _tex_rc():
    """Matplotlib rc for LaTeX-looking text: real LaTeX when a latex binary is on
    PATH, else mathtext with Computer Modern (same look, no TeX install needed)."""
    import shutil
    rc = {"font.family": "serif", "font.size": 13, "axes.formatter.use_mathtext": True}
    if shutil.which("latex") and shutil.which("dvipng"):
        rc.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"})
    else:
        rc.update({"mathtext.fontset": "cm", "font.serif": ["cmr10", "DejaVu Serif"]})
    return rc


def plot_wall3d(d, out, elev=35, azim=-60, title=None, transparent=True):
    """a E_n on the wall in 3D.  Transparent background; colour bar placed just
    right of the projected surface, title just above it."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d import proj3d
    q, aE = _wall_closed(d)
    cmap, vmin, vmax = color_scale(aE)
    norm = Normalize(vmin, vmax)
    if title is None:
        title = r"Normal electric field on the wall, $a E_n = -a\,\partial\phi/\partial n$"
    with plt.rc_context(_tex_rc()):
        fig = plt.figure(figsize=(12, 7))
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax.plot_surface(q[..., 0], q[..., 1], q[..., 2], facecolors=colormaps[cmap](norm(aE)),
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        lo, hi = q.reshape(-1, 3).min(0), q.reshape(-1, 3).max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect(tuple(hi - lo), zoom=1.3)
        except TypeError:
            ax.set_box_aspect(tuple(hi - lo))
        # bounding box of the projected surface, in figure coordinates
        fig.canvas.draw()
        P = q.reshape(-1, 3)
        xs, ys, _ = proj3d.proj_transform(P[:, 0], P[:, 1], P[:, 2], ax.get_proj())
        f = fig.transFigure.inverted().transform(ax.transData.transform(np.c_[xs, ys]))
        (x0, y0), (x1, y1) = f.min(0), f.max(0)
        h = 0.875 * (y1 - y0)
        cax = fig.add_axes([x1 + 0.06, (y0 + y1) / 2 - h / 2, 0.0225, h])
        cb = fig.colorbar(ScalarMappable(norm, cmap), cax=cax)         # ticks only, no label
        tt = fig.text((x0 + x1) / 2, y1 + 0.075, title, ha="center", va="bottom", fontsize=24)
        # crop to surface + title + colour bar (the hidden 3D axes box is much larger)
        from matplotlib.transforms import Bbox
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        surf = Bbox.from_extents(*fig.transFigure.transform((x0, y0)), *fig.transFigure.transform((x1, y1)))
        box = Bbox.union([surf, tt.get_window_extent(r), cb.ax.get_tightbbox(r)])
        box = box.transformed(fig.dpi_scale_trans.inverted()).padded(0.08)
        fig.savefig(out, dpi=200, bbox_inches=box, transparent=transparent)
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


def plot_all(d, prefix, patches=False, phi_lines=None, psi_lines=True, title=True):
    import matplotlib
    matplotlib.use("Agg")
    plot_sections(d, prefix + "_sections.png", patches=patches, phi_lines=phi_lines,
                  psi_lines=psi_lines, title=title)
    plot_wall3d(d, prefix + "_wall3d.png")
    plot_wall_map(d, prefix + "_wallmap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--prefix", default=None, help="output prefix (default: the npz path without .npz)")
    ap.add_argument("--patches", action="store_true", help="draw the patch outlines on the sections")
    ap.add_argument("--no-psi", action="store_true", help="omit the dashed white flux-surface lines")
    ap.add_argument("--no-title", action="store_true", help="omit the overall title of the sections figure")
    args = ap.parse_args()
    d = dict(np.load(args.npz, allow_pickle=False))
    plot_all(d, args.prefix or os.path.splitext(args.npz)[0], patches=args.patches,
             psi_lines=not args.no_psi, title=not args.no_title)


if __name__ == "__main__":
    main()
