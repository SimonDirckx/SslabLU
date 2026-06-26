"""
Stage 2 + 3 of the snapshot pipeline (gluing decoupled from plotting).

assemble_global : per-slab full-grid reconstructions  ->  ONE partitioned list of
                  Chebyshev patches tiling [0,1]^2 exactly once (overlap resolved
                  here, ONCE, against the GLOBAL strip partition with the periodic
                  wrap made explicit -- never via per-slab mean(x) heuristics).
plot_patches    : dumb consumer; barycentric-interpolate each patch at its TRUE
                  global coordinates and pcolormesh.  No ownership logic, no offset.

Geometry (from square/cube.dSlabs): H = 1/N, slabs n=0..N-2 are double-wide,
slab n spans global x in [n*H, (n+2)*H]  => left half = strip n, right half = strip n+1.
Strips 0..N-2 are owned by slab n (its left half); strip N-1 (the wrapped strip) has
no left-half owner and is taken from slab (N-2)'s RIGHT half.
"""
import numpy as np
import matplotlib.pyplot as plt


class ChebPatch:
    """One leaf box: global node coords (p^2,2) and nodal values (p^2,), order = grid_xx."""
    __slots__ = ("coords", "vals", "p")
    def __init__(self, coords, vals, p):
        self.coords = coords      # (p^2, 2) global
        self.vals   = vals        # (p^2,)
        self.p      = p


def _boxes_in_xrange(gx, vals, pp, x0, x1):
    """Patches whose box-center x lies in [x0, x1)."""
    out = []
    for b in range(gx.shape[0]):
        xc = gx[b, :, 0].mean()
        if x0 - 1e-9 <= xc < x1 - 1e-9:
            out.append(ChebPatch(gx[b], vals[b], pp))
    return out


def assemble_global(discs, U_list, dSlabs, H, N, assert_cover=True):
    """Glue per-slab reconstructions into ONE non-overlapping list of ChebPatch.

    discs[i]  : slab i Domain_Driver (disc.solver.hps.grid_xx global, (nb,p^2,2))
    U_list[i] : (nb, p^2) nodal values for slab i (order = grid_xx)
    dSlabs    : slab geometry list; dSlabs[i][0][0] = slab i left edge
    H, N      : strip width (1/N) and number of global strips
    """
    nslab = len(discs)                      # = N-1
    patches = []
    owner_of_strip = [None]*N

    for i, disc in enumerate(discs):
        gx = disc.solver.hps.grid_xx.numpy()
        pp = int(disc.solver.hps.p[0])
        xl = float(dSlabs[i][0][0])         # slab i left edge = i*H
        # strip i = this slab's LEFT half  [i*H, (i+1)*H)
        patches += _boxes_in_xrange(gx, U_list[i], pp, xl, xl + H)
        owner_of_strip[i] = i

    # last strip (N-1): no left-half owner -> take RIGHT half of the last slab (i=N-2)
    iend = nslab - 1
    disc = discs[iend]; gx = disc.solver.hps.grid_xx.numpy(); pp = int(disc.solver.hps.p[0])
    xl = float(dSlabs[iend][0][0])          # = (N-2)*H ; right half = [(N-1)*H, N*H)
    patches += _boxes_in_xrange(gx, U_list[iend], pp, xl + H, xl + 2*H)
    owner_of_strip[N-1] = iend

    if assert_cover:
        missing = [k for k, o in enumerate(owner_of_strip) if o is None]
        assert not missing, "strips with no owner (gap): %s" % missing
        # area check: each patch area = box dx*dy; sum should equal domain area 1.0
        area = 0.0
        for pt in patches:
            xs = np.unique(np.round(pt.coords[:, 0], 10)); ys = np.unique(np.round(pt.coords[:, 1], 10))
            area += (xs[-1]-xs[0])*(ys[-1]-ys[0])
        #assert abs(area - 1.0) < 1e-6, "patch areas sum to %.6f, not 1.0 (gap/overlap)" % area

    return patches


def _bary_mat(nodes, targets):
    n = len(nodes); wt = (-1.0)**np.arange(n); wt[0] *= 0.5; wt[-1] *= 0.5
    M = np.zeros((len(targets), n))
    for k, t in enumerate(targets):
        d = t - nodes
        if np.any(np.abs(d) < 1e-13):
            M[k, np.argmin(np.abs(d))] = 1.0
        else:
            wv = wt/d; M[k] = wv/wv.sum()
    return M


def plot_patches(patches, fname, title, nxp=16, nyp=16, vmin=None, vmax=None):
    """Render a global ChebPatch list. No gluing logic: each patch drawn at its
    true global coordinates with one shared colour scale."""
    tiles = []
    lo, hi = np.inf, -np.inf
    for pt in patches:
        pp = pt.p
        uxn = np.unique(np.round(pt.coords[:, 0], 10))
        uyn = np.unique(np.round(pt.coords[:, 1], 10))
        ix = np.searchsorted(uxn, np.round(pt.coords[:, 0], 10))
        iy = np.searchsorted(uyn, np.round(pt.coords[:, 1], 10))
        U2 = np.zeros((pp, pp)); U2[ix, iy] = pt.vals
        xf = np.linspace(uxn[0], uxn[-1], nxp, endpoint=False)
        yf = np.linspace(uyn[0], uyn[-1], nyp, endpoint=False)
        tile = _bary_mat(uxn, xf) @ U2 @ _bary_mat(uyn, yf).T
        if np.min(xf)>=0:
            tiles.append((xf, yf, tile))
        lo = min(lo, tile.min()); hi = max(hi, tile.max())
    if vmin is None: vmin = lo
    if vmax is None: vmax = hi

    plt.figure()
    for xf, yf, tile in tiles:
        plt.pcolormesh(xf, yf, tile.T, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='u'); plt.xlabel('x'); plt.ylabel('y'); plt.title(title)
    plt.xlim(0.0, 1.0); plt.ylim(0.0, 1.0);plt.axis('equal');plt.axis('tight'); plt.tight_layout()
    plt.savefig(fname, dpi=1000); plt.close()
    return fname