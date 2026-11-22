"""
Components of the a-posteriori bound of pb_fixed_operator.tex (Verification, 2):

    |phi~ - phi|_inf <= C_P max_leaves |eta| + C_Gamma max_faces |[d_n phi~]|
                        + max_wall |phi~ - phi_b|,
    eta = -Lap phi~ + G[phi~] - f*,

evaluated at random OFF-NODE points of the reconstructed leaf polynomials
(collocation makes eta vanish at the interior nodes, and the flux / boundary
conditions hold at the Gauss face nodes only).  The constants depend on the
geometry and the leaf size only, so for p-convergence the components are
reported as they are.

  eta      : at random points inside the leaves, via the reference-coordinate
             coefficients of each patch (Patch.ref_coeffs on the geometry PDO,
             i.e. exactly the operator the leaves discretize);
  jumps    : physical normal derivative, outward from both sides, summed:
                 [d_n phi] = sum_sides s e_a . A~ grad_X phi / sqrt(e_a . A~ e_a)
             (A~ = reference metric, e_a the reference normal), on leaf faces
             inside a patch and on patch-patch faces (the other side found by
             Newton on its patch map);
  wall     : phi~ - phi_b at random points on the wall faces.

Each slab contributes only its own single slab around the midplane, t in
[t0 + H/2, t0 + 3H/2), so every point of the torus is counted once.

Sanity checks returned alongside (both should be small):
  eta_nodes  : eta at the leaf-interior collocation nodes (~ the local chord
               tolerance times c: checks the operator, signs and G);
  wall_gauss : phi~ - phi_b at the Gauss wall nodes (~ rounding: checks the
               leaf evaluation and the face geometry).
"""
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.spatial import cKDTree

from solver.cylinder_slab import pdo3d_to_cyl


def _axis_basis(u, x):
    """Lagrange basis on nodes u in [0, 1] and its first two derivatives at x
    (w.r.t. the relative coordinate), via Chebyshev coefficients."""
    p = len(u)
    Ci = np.linalg.inv(C.chebvander(2 * u - 1, p - 1))           # column j: coeffs of l_j
    z = 2 * np.asarray(x) - 1
    L0 = C.chebvander(z, p - 1) @ Ci
    L1 = C.chebvander(z, p - 2) @ (2 * C.chebder(Ci, 1, axis=0))
    L2 = C.chebvander(z, p - 3) @ (4 * C.chebder(Ci, 2, axis=0))
    return L0, L1, L2


class _PatchField:
    """Leaf polynomials of one patch of a slab field."""

    def __init__(self, P, grid, off, phi, imask):
        self.P, self.grid = P, grid
        self.box = np.asarray(P.box, float)
        self.nleaf = np.asarray(P.nleaf, int)
        nb, self.npl, _ = grid.shape
        self.lo, self.hi = grid.min(axis=1), grid.max(axis=1)     # leaf boxes
        rel0 = (grid[0] - self.lo[0]) / (self.hi[0] - self.lo[0])
        self.nodes, self.idx = [], []
        for a in range(3):
            u, inv = np.unique(np.round(rel0[:, a], 12), return_inverse=True)
            self.nodes.append(u); self.idx.append(inv)
        self.V = np.asarray(phi)[off:off + nb * self.npl].reshape(nb, self.npl)
        self.Jc = np.asarray(imask[off:off + self.npl], bool)      # same in every leaf

    def leaf_of(self, X):
        ia = [np.clip(np.floor((X[:, a] - self.box[a, 0]) / (self.box[a, 1] - self.box[a, 0])
                               * self.nleaf[a]).astype(int), 0, self.nleaf[a] - 1) for a in range(3)]
        return np.ravel_multi_index(tuple(ia), self.nleaf)

    def eval(self, X, b, chunk=4096):
        """value (N,), gradient (N,3), Hessian (N,3,3) in patch reference coords."""
        if len(X) <= chunk:
            return self._eval(X, b)
        parts = [self._eval(X[i:i + chunk], b[i:i + chunk]) for i in range(0, len(X), chunk)]
        return tuple(np.concatenate([q[j] for q in parts]) for j in range(3))

    def _eval(self, X, b):
        span = self.hi[b] - self.lo[b]
        r = (X - self.lo[b]) / span
        E = []
        for a in range(3):
            E.append([M[:, self.idx[a]] for M in _axis_basis(self.nodes[a], r[:, a])])
        Vb = self.V[b]
        f = lambda d: np.einsum("kn,kn->k", Vb, E[0][d[0]] * E[1][d[1]] * E[2][d[2]])
        val = f((0, 0, 0))
        grad = np.empty((len(X), 3))
        hess = np.empty((len(X), 3, 3))
        for a in range(3):
            d = [0, 0, 0]; d[a] = 1
            grad[:, a] = f(d) / span[:, a]
            for c in range(a, 3):
                d = [0, 0, 0]; d[a] += 1; d[c] += 1
                hess[:, a, c] = hess[:, c, a] = f(d) / (span[:, a] * span[:, c])
        return val, grad, hess


def _oms(P, X):
    """Patch reference points -> OMS coordinates (t, x, y)."""
    return P.to_phys(X)[:, [2, 0, 1]]


class ResidualEstimator:
    def __init__(self, pdo_geo, model, Rc, nsamp=3, wall_tol=1e-9):
        """pdo_geo: geometry PDO in OMS coordinates (the one given to OMS);
        model: PBProfiles / PBManufactured (g, fstar, phi_b or phistar)."""
        self.pde = pdo3d_to_cyl(pdo_geo)
        self.model, self.Rc, self.nsamp, self.wall_tol = model, Rc, nsamp, wall_tol
        self.phib = model.phi_b if hasattr(model, "phi_b") else model.phistar

    # ---------------------------------------------------------- pieces
    def _eta(self, F, X, b):
        val, grad, hess = F.eval(X, b)
        At, bt, c, f = F.P.ref_coeffs(X, self.pde)
        lap = -np.einsum("nkl,nkl->n", At, hess) + np.einsum("nk,nk->n", bt, grad) + c * val - f
        W = _oms(F.P, X)
        return lap + self.model.g(W, val) - self.model.fstar(W)

    def _flux_out(self, F, X, b, a, s):
        _, grad, _ = F.eval(X, b)
        At = F.P.ref_a(X, self.pde)
        return s * np.einsum("nk,nk->n", At[:, a, :], grad) / np.sqrt(At[:, a, a])

    def _is_wall(self, P, X):
        x = P.to_phys(X)
        return x[:, 0] ** 2 + x[:, 1] ** 2 >= self.Rc ** 2 * (1 - self.wall_tol)

    @staticmethod
    def _face_leaves(F, a, side):
        """Leaves of F touching its box face (axis a, side 0 = lo / 1 = hi)."""
        I = np.unravel_index(np.arange(int(np.prod(F.nleaf))), F.nleaf)
        return np.flatnonzero(I[a] == (F.nleaf[a] - 1 if side else 0))

    def _face_points(self, F, leaves, a, side, rng, gauss=None):
        """Points on face (a, side) of the given leaves: random tangential
        coordinates, or the Gauss-Legendre tensor nodes (gauss = (qt1, qt2))."""
        tang = [t for t in range(3) if t != a]
        if gauss is None:
            r = rng.uniform(size=(len(leaves), self.nsamp, 3))
        else:
            g = [(np.polynomial.legendre.leggauss(q)[0] + 1) / 2 for q in gauss]
            G0, G1 = np.meshgrid(g[0], g[1], indexing="ij")
            r = np.zeros((len(leaves), G0.size, 3))
            r[..., tang[0]], r[..., tang[1]] = G0.ravel(), G1.ravel()
        r[..., a] = float(side)
        lo, hi = F.lo[leaves][:, None, :], F.hi[leaves][:, None, :]
        X = (lo + r * (hi - lo)).reshape(-1, 3)
        b = np.repeat(leaves, r.shape[1])
        return X, b

    # ---------------------------------------------------------- one slab
    def slab(self, ops, phi, t_own, seed=0):
        """Components on slab `ops` (CylinderSlabOps) with leaf field phi,
        restricted to t in [t_own[0], t_own[1])."""
        rng = np.random.default_rng(seed)
        imask = ops.interior_mask()
        Fs = [_PatchField(P, grid, off, phi, imask) for P, grid, off in ops.leaf_grids()]
        own = lambda X: (X[:, 2] >= t_own[0]) & (X[:, 2] < t_own[1])
        out = dict(eta=0.0, jump_leaf=0.0, jump_patch=0.0, wall=0.0, eta_nodes=0.0,
                   wall_gauss=0.0, n_eta=0, n_jump_leaf=0, n_jump_patch=0, n_wall=0,
                   patch_newton_fail=0)
        upd = lambda k, v: out.__setitem__(k, max(out[k], float(np.max(np.abs(v))) if len(v) else 0.0))

        trees = []
        for k, F in enumerate(Fs):
            others = [j for j in range(len(Fs)) if j != k]
            pts = [Fs[j].P.to_phys(Fs[j].grid.reshape(-1, 3)) for j in others]
            lab = np.concatenate([np.full(len(q), j) for j, q in zip(others, pts)])
            ref = np.vstack([Fs[j].grid.reshape(-1, 3) for j in others])
            trees.append((cKDTree(np.vstack(pts)), lab, ref))

        for k, F in enumerate(Fs):
            nb = F.V.shape[0]
            # -- interior residual: random points, and the collocation nodes (check)
            r = rng.uniform(size=(nb, self.nsamp, 3))
            X = (F.lo[:, None, :] + r * (F.hi - F.lo)[:, None, :]).reshape(-1, 3)
            b = np.repeat(np.arange(nb), self.nsamp)
            m = own(X)
            if m.any():
                upd("eta", self._eta(F, X[m], b[m])); out["n_eta"] += int(m.sum())
            Xn = F.grid[:, F.Jc, :].reshape(-1, 3)
            bn = np.repeat(np.arange(nb), int(F.Jc.sum()))
            m = own(Xn)
            if m.any():
                upd("eta_nodes", self._eta(F, Xn[m], bn[m]))

            # -- leaf-leaf faces inside the patch
            I = np.unravel_index(np.arange(nb), F.nleaf)
            for a in range(3):
                sel = np.flatnonzero(I[a] < F.nleaf[a] - 1)
                if not len(sel):
                    continue
                J = [I[c][sel].copy() for c in range(3)]
                J[a] += 1
                nbr = np.ravel_multi_index(tuple(J), F.nleaf)
                X, b = self._face_points(F, sel, a, 1, rng)
                b2 = np.repeat(nbr, self.nsamp)
                m = own(X)
                if m.any():
                    j = self._flux_out(F, X[m], b[m], a, 1.0) + self._flux_out(F, X[m], b2[m], a, -1.0)
                    upd("jump_leaf", j); out["n_jump_leaf"] += int(m.sum())

            # -- box faces of the patch: wall or another patch
            for a in range(2):                                     # cross-section axes
                for side in (0, 1):
                    leaves = self._face_leaves(F, a, side)
                    X, b = self._face_points(F, leaves, a, side, rng)
                    m = own(X)
                    X, b = X[m], b[m]
                    if not len(X):
                        continue
                    wall = self._is_wall(F.P, X)
                    if wall.any():
                        Xw = X[wall]
                        v, _, _ = F.eval(Xw, b[wall])
                        upd("wall", v - self.phib(_oms(F.P, Xw))); out["n_wall"] += len(Xw)
                        tang = [t for t in range(3) if t != a]
                        Xg, bg = self._face_points(F, leaves, a, side, rng,
                                                   gauss=[len(F.nodes[t]) - 2 for t in tang])
                        mg = own(Xg) & self._is_wall(F.P, Xg)
                        if mg.any():
                            vg, _, _ = F.eval(Xg[mg], bg[mg])
                            upd("wall_gauss", vg - self.phib(_oms(F.P, Xg[mg])))
                    if (~wall).any():
                        j, nfail = self._patch_jump(Fs, trees[k], F, X[~wall], b[~wall], a, side)
                        upd("jump_patch", j); out["n_jump_patch"] += len(j)
                        out["patch_newton_fail"] += nfail
        return out

    def _patch_jump(self, Fs, tree, F, X, b, a, side, tol=1e-9):
        """Flux jump across a patch-patch face: locate the points in the other
        patch (nearest node, then Newton on its map), sum the outward fluxes."""
        x = F.P.to_phys(X)
        kd, lab, ref = tree
        _, nn = kd.query(x)
        fA = self._flux_out(F, X, b, a, 1.0 if side else -1.0)
        jumps, nfail = np.full(len(X), np.nan), 0
        for j in np.unique(lab[nn]):
            s = np.flatnonzero(lab[nn] == j)
            G = Fs[j]
            Y = ref[nn[s]].copy()
            for _ in range(50):
                res = G.P.to_phys(Y) - x[s]
                if np.max(np.abs(res)) < 1e-13:
                    break
                Y -= np.einsum("nkj,nj->nk", G.P.grad_ref(Y), res)
            ok = np.max(np.abs(G.P.to_phys(Y) - x[s]), axis=1) < 1e-10
            # which face of G: the cross-section coordinate at its box bound
            dlo = np.abs(Y[:, :2] - G.box[:2, 0]); dhi = np.abs(Y[:, :2] - G.box[:2, 1])
            dmin = np.minimum(dlo, dhi)
            aB = np.argmin(dmin, axis=1)
            ok &= dmin[np.arange(len(Y)), aB] < tol * max(1.0, np.abs(G.box).max())
            nfail += int(np.sum(~ok))
            for ax in (0, 1):
                t = np.flatnonzero(ok & (aB == ax))
                if not len(t):
                    continue
                sideB = dhi[t, ax] < dlo[t, ax]
                Yt = Y[t].copy()
                Yt[:, ax] = np.where(sideB, G.box[ax, 1], G.box[ax, 0])   # snap onto the face
                bB = G.leaf_of(Yt)
                fB = self._flux_out(G, Yt, bB, ax, np.where(sideB, 1.0, -1.0))
                jumps[s[t]] = fA[s[t]] + fB
        return jumps[np.isfinite(jumps)], nfail
