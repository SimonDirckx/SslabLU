"""
Nested prolongation between two 5-patch slab discretizations with the same
patch maps (same geom, Rc, s), coarsened in h and/or p independently in the
three directions (tangential n/p_n, radial m/p_m, axial nz/p_z).

Every fine leaf lies inside one coarse leaf in patch reference space (uniform
leaves per patch, integer leaf ratios, the same arch radial stretch for both,
see arch_stretch.py), so P is: for each fine leaf, evaluate
the parent leaf's tensor Lagrange interpolant (on the coarse Chebyshev nodes)
at the fine leaf's nodes.  One small matrix per child position, applied in
batches; no inverse maps, no point search.
"""
import numpy as np
import torch

from solver.cylinder_slab import slab_patches, patch_order
from solver.hpsmultidomain.hpsmultidomain.hps_multidomain_disc import HPS_Multidomain
from solver.hpsmultidomain.hpsmultidomain.pdo import PDO_3d

_one = lambda xx: torch.ones(xx.shape[0], dtype=xx.dtype, device=xx.device)
# never evaluated: HPS_Multidomain.__init__ only builds grids.  c12.. given so
# the face-node choice (Gauss) matches the real build.
_DUMMY_PDO = PDO_3d(c11=_one, c22=_one, c33=_one, c12=_one, c13=_one, c23=_one)


class LeafGrid:
    """Leaf-grid nodes of a slab discretization WITHOUT building it, in the
    same order as CylinderSlab(..., keep_leaf=True).nodes()."""

    def __init__(self, geom, p, Rc, s, n, m, nz, stretch=0.0):
        self.spec = dict(p=p, n=n, m=m, nz=nz, stretch=stretch)
        self.patches = []
        for P in slab_patches(geom, Rc, s, n, m, nz, stretch):
            pv = np.array(patch_order(p, P.kind))
            hps = HPS_Multidomain(_DUMMY_PDO, torch.tensor(P.box), P.a, pv, 3)
            self.patches.append(dict(P=P, grid=hps.grid_xx.numpy(),
                                     nleaf=np.array(P.nleaf, dtype=int), pv=pv))
        Xc = np.vstack([d["P"].to_phys(d["grid"].reshape(-1, 3)) for d in self.patches])
        self.nodes = np.ascontiguousarray(Xc[:, [2, 0, 1]])          # OMS (t, x, y)
        sizes = [d["grid"].shape[0] * d["grid"].shape[1] for d in self.patches]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])


def _layout(grid0):
    """Sorted relative 1D nodes in [0, 1] per axis and, for every leaf node,
    its index into them (identical in every leaf of a patch)."""
    lo, hi = grid0.min(axis=0), grid0.max(axis=0)
    rel = (grid0 - lo) / (hi - lo)
    u, idx = [], []
    for a in range(3):
        ua, ia = np.unique(np.round(rel[:, a], 12), return_inverse=True)
        u.append(ua); idx.append(ia)
    return u, idx


def lagrange_matrix(u, x):
    """L[i, j] = l_j(x_i), Lagrange basis on nodes u (barycentric, exact at nodes)."""
    u, x = np.asarray(u, float), np.asarray(x, float)
    w = 1.0 / np.prod(u[:, None] - u[None, :] + np.eye(len(u)), axis=1)
    d = x[:, None] - u[None, :]
    hit = np.abs(d) < 1e-13
    d[hit] = 1.0
    t = w / d
    L = t / t.sum(axis=1, keepdims=True)
    rows = hit.any(axis=1)
    L[rows] = hit[rows].astype(float)
    return L


class Prolongation:
    """P: coarse leaf-grid values -> fine leaf-grid values.

    Default: coarse and fine discretize the same patches.  For a fine slab
    inside a longer (e.g. periodic, whole-torus) coarse grid along the axial
    patch axis 2: t_offset = index of the fine slab's first axial leaf in the
    GLOBAL fine axial leaf numbering, nt_global = total number of fine axial
    leaves of the coarse domain (wrap-around: periodic coarse grid)."""

    def __init__(self, coarse, fine, t_offset=0, nt_global=None):
        self.coarse, self.fine = coarse, fine
        self.ops = []
        for pc, pf in zip(coarse.patches, fine.patches):
            assert pc["P"].kind == pf["P"].kind and pc["P"].rot == pf["P"].rot
            assert np.allclose(pc["P"].box[:2], pf["P"].box[:2]), "cross-sections must coincide"
            if getattr(pc["P"], "stretch", 0.0) != getattr(pf["P"], "stretch", 0.0):
                raise ValueError("coarse and fine arch stretch differ (%g vs %g): leaves are "
                                 "only nested in xi for the same map"
                                 % (getattr(pc["P"], "stretch", 0.0), getattr(pf["P"], "stretch", 0.0)))
            nfg = pf["nleaf"].copy()
            if nt_global is not None:
                nfg[2] = int(nt_global)
            elif not np.allclose(pc["P"].box, pf["P"].box):
                raise ValueError("axial extents differ: pass t_offset / nt_global")
            r = nfg // pc["nleaf"]
            if np.any(r * pc["nleaf"] != nfg):
                raise ValueError("fine leaf counts %s must be multiples of coarse %s"
                                 % (nfg, pc["nleaf"]))
            uc, ic = _layout(pc["grid"][0])
            uf, jf = _layout(pf["grid"][0])
            # per axis and child position c: fine child nodes (c + uf) / r in the parent
            B = [[lagrange_matrix(uc[a], (c + uf[a]) / r[a]) for c in range(r[a])]
                 for a in range(3)]
            nchild = int(np.prod(r))
            M = np.empty((nchild, pf["grid"].shape[1], pc["grid"].shape[1]))
            for cp in range(nchild):
                c = np.unravel_index(cp, r)
                M[cp] = (B[0][c[0]][jf[0]][:, ic[0]] * B[1][c[1]][jf[1]][:, ic[1]]
                         * B[2][c[2]][jf[2]][:, ic[2]])
            # leaf numbering of HPS_Multidomain.get_grid: box = (i n1 + j) n2 + k
            nbf = int(np.prod(pf["nleaf"]))
            I = list(np.unravel_index(np.arange(nbf), pf["nleaf"]))
            I[2] = (I[2] + int(t_offset)) % nfg[2]               # global axial leaf index
            parent = np.ravel_multi_index(tuple(I[a] // r[a] for a in range(3)), pc["nleaf"])
            child = np.ravel_multi_index(tuple(I[a] % r[a] for a in range(3)), r)
            sel = [np.flatnonzero(child == cp) for cp in range(nchild)]
            self.ops.append(dict(M=M, parent=parent, sel=sel,
                                 npc=pc["grid"].shape[1], npf=pf["grid"].shape[1], nbf=nbf))

    def __call__(self, vc):
        vc = np.asarray(vc)
        out = np.empty(self.fine.offsets[-1], dtype=vc.dtype)
        for i, op in enumerate(self.ops):
            a, b = self.coarse.offsets[i], self.coarse.offsets[i + 1]
            Vc = vc[a:b].reshape(-1, op["npc"])
            Vf = np.empty((op["nbf"], op["npf"]), dtype=vc.dtype)
            for cp, s in enumerate(op["sel"]):
                Vf[s] = Vc[op["parent"][s]] @ op["M"][cp].T
            out[self.fine.offsets[i]:self.fine.offsets[i + 1]] = Vf.reshape(-1)
        return out


def check_prolongation(P, seed=0):
    """Max error of P on a random tensor polynomial of degree p_c - 1 per axis
    in each patch's reference coordinates (expect ~1e-13)."""
    rng = np.random.default_rng(seed)
    vc = np.empty(P.coarse.offsets[-1]); vf_ex = np.empty(P.fine.offsets[-1])
    for i, (pc, pf) in enumerate(zip(P.coarse.patches, P.fine.patches)):
        box = pc["P"].box
        coefs = [rng.standard_normal(pc["pv"][a]) for a in range(3)]   # degree pv-1

        def q(X):
            Z = 2 * (X - box[:, 0]) / (box[:, 1] - box[:, 0]) - 1          # patch coords in [-1, 1]
            return np.prod([np.polynomial.chebyshev.chebval(Z[:, a], coefs[a])
                            for a in range(3)], axis=0)
        vc[P.coarse.offsets[i]:P.coarse.offsets[i + 1]] = q(pc["grid"].reshape(-1, 3))
        vf_ex[P.fine.offsets[i]:P.fine.offsets[i + 1]] = q(pf["grid"].reshape(-1, 3))
    return np.max(np.abs(P(vc) - vf_ex)) / np.max(np.abs(vf_ex))
