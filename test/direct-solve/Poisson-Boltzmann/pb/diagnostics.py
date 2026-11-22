"""
Diagnostics for runs WITHOUT an exact solution (pb.profiles.PBProfiles):

  probes         fixed points (patch index, normalized cross-section coords,
                 global t), identical for every resolution with the same
                 (Rc, s): the leaf polynomial of the slab whose midplane is
                 nearest is evaluated there.  Saved per run; comparing two runs
                 gives a self-convergence estimate.
  overlap        neighbouring slabs' converged local fields must agree on their
                 common single slab (both equal the global discrete solution):
                 max mismatch over matching leaf-interior nodes (~ iteration
                 tolerance; a direct check of the periodic wiring and of E).
  flux function  least-squares fit phi ~ F(psi) on psi <= psi_max; the max
                 residual is the departure of phi from a flux function.
"""
import numpy as np
from scipy.spatial import cKDTree

from pb.prolong import lagrange_matrix


# ---------------------------------------------------------------- probes
def make_probes(L, n_per_patch=200, npatch=5, seed=12345):
    """Normalized cross-section coordinates u in [0.02, 0.98]^2 per patch and
    a global axial coordinate t in [0, L)."""
    rng = np.random.default_rng(seed)
    k = np.repeat(np.arange(npatch), n_per_patch)
    u = rng.uniform(0.02, 0.98, size=(len(k), 2))
    t = rng.uniform(0.0, L, size=len(k))
    return dict(patch=k, u=u, t=t)


def _eval_patch(P, grid, off, phi, u, tloc):
    """Leaf polynomial of patch P (leaf grid `grid`, field phi[off:]) at the
    patch-reference points (box[:2] scaled by u, axial tloc)."""
    box = np.asarray(P.box, float)
    X = np.c_[box[0, 0] + u[:, 0] * (box[0, 1] - box[0, 0]),
              box[1, 0] + u[:, 1] * (box[1, 1] - box[1, 0]), tloc]
    nleaf = np.asarray(P.nleaf, int)
    ia = [np.clip(np.floor((X[:, a] - box[a, 0]) / (box[a, 1] - box[a, 0]) * nleaf[a]).astype(int),
                  0, nleaf[a] - 1) for a in range(3)]
    b = np.ravel_multi_index(tuple(ia), nleaf)            # box = (i n1 + j) n2 + k
    npl = grid.shape[1]
    lo, hi = grid.min(axis=1), grid.max(axis=1)           # leaf boxes (nodes include faces)
    rel0 = (grid[0] - lo[0]) / (hi[0] - lo[0])
    r = (X - lo[b]) / (hi[b] - lo[b])
    B = np.ones((len(X), npl))
    for a in range(3):
        nodes, idx = np.unique(np.round(rel0[:, a], 12), return_inverse=True)
        B *= lagrange_matrix(nodes, r[:, a])[:, idx]
    V = np.asarray(phi)[off:off + grid.shape[0] * npl].reshape(-1, npl)
    return np.einsum("kn,kn->k", B, V[b])


def eval_probes_slabs(probes, slabs, H, L):
    """slabs: list of (ops, phi) in OMS slab order (slab i = [iH, (i+2)H])."""
    Ns = len(slabs)
    t = probes["t"]
    owner = (np.rint(t / H).astype(int) - 1) % Ns         # nearest midplane (i+1)H
    out = np.full(len(t), np.nan)
    for i, (ops, phi) in enumerate(slabs):
        t0 = float(ops.zbounds[0])
        sel_i = owner == i
        if not sel_i.any():
            continue
        tloc = t0 + np.mod(t - t0, L)
        for k, (P, grid, off) in enumerate(ops.leaf_grids()):
            s = np.flatnonzero(sel_i & (probes["patch"] == k))
            if len(s):
                assert np.all((tloc[s] >= t0 - 1e-12) & (tloc[s] <= t0 + 2 * H + 1e-12))
                out[s] = _eval_patch(P, grid, off, phi, probes["u"][s], tloc[s])
    assert np.all(np.isfinite(out)), "probe(s) not owned by any slab"
    return out


def eval_probes_global(probes, ops, phi, L):
    """One periodic slab over [z0, z0 + L) (the global coarse problem)."""
    t0 = float(ops.zbounds[0])
    tloc = t0 + np.mod(probes["t"] - t0, L)
    out = np.full(len(tloc), np.nan)
    for k, (P, grid, off) in enumerate(ops.leaf_grids()):
        s = np.flatnonzero(probes["patch"] == k)
        out[s] = _eval_patch(P, grid, off, phi, probes["u"][s], tloc[s])
    return out


def compare_probes(path, probes, vals):
    """Max |difference| against a saved run with the same probes."""
    d = np.load(path)
    if not (np.array_equal(d["patch"], probes["patch"]) and np.allclose(d["u"], probes["u"])
            and np.allclose(d["t"], probes["t"])):
        raise ValueError("%s was made with different probe points" % path)
    diff = np.max(np.abs(vals - d["values"]))
    return diff, diff / np.max(np.abs(d["values"])), {k: d[k] for k in d.files if k.startswith("meta_")}


# ---------------------------------------------------------------- overlap
def overlap_mismatch(slabs_W, slabs_phi, slabs_I, zb, L, tol=1e-9):
    """max_i max |phi_i - phi_{i+1}| over leaf-interior nodes of their common
    single slab [z0_{i+1}, z1_i] (periodic wrap for the last pair).
    Returns (max mismatch, per-pair array, number of unmatched nodes)."""
    Ns = len(slabs_W)
    per, unmatched = np.zeros(Ns), 0
    for i in range(Ns):
        j = (i + 1) % Ns
        Wi, Wj = slabs_W[i], slabs_W[j].copy()
        zj0 = zb[j][0]
        if j < i:                                          # wrap: slab 0 seen from slab Ns-1
            Wj[:, 0] += L
            zj0 += L
        lo, hi = zj0, zb[i][1]
        mi = slabs_I[i] & (Wi[:, 0] > lo) & (Wi[:, 0] < hi)
        mj = slabs_I[j] & (Wj[:, 0] > lo) & (Wj[:, 0] < hi)
        dist, idx = cKDTree(Wi[mi]).query(Wj[mj])
        ok = dist < tol
        unmatched += int(np.sum(~ok)) + abs(int(mi.sum()) - int(mj.sum()))
        if ok.any():
            per[i] = np.max(np.abs(slabs_phi[j][mj][ok] - slabs_phi[i][mi][idx[ok]]))
    return per.max(), per, unmatched


# ---------------------------------------------------------------- flux function
def flux_function_fit(psi, phi, psi_max=0.8, deg=10, psi_out=None):
    """phi ~ F(psi) (Chebyshev, least squares) on psi <= psi_max.
    Returns (max |phi - F(psi)|, F at psi_out, coefficients)."""
    m = psi <= psi_max
    x = 2.0 * psi[m] / psi_max - 1.0
    coef = np.polynomial.chebyshev.chebfit(x, phi[m], deg)
    res = np.max(np.abs(phi[m] - np.polynomial.chebyshev.chebval(x, coef)))
    psi_out = np.linspace(0.0, psi_max, 9) if psi_out is None else np.asarray(psi_out)
    return res, np.polynomial.chebyshev.chebval(2.0 * psi_out / psi_max - 1.0, coef), coef
