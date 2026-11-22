"""
Components of the a-posteriori bound of pb_fixed_operator.tex (Verification, 2),
completed with the value jumps:

    e = phi~ - phi solves -Lap e + c e = (continuous residual of phi~), c >= 0, with
      volume part   eta = -Lap phi~ + G[phi~] - f*   inside the leaves,
      face parts    [d_n phi~] (single layer) and [phi~] (double layer) on leaf
                    faces -- HPS matches values and fluxes at the Gauss face
                    nodes only, so both jump off-node,
      boundary part phi~ - phi_b on the wall;
    |e|_inf <= C_P max|eta| + C_Gamma max|[d_n phi~]| + C_D max|[phi~]| + max|phi~ - phi_b|.

All components are evaluated at random OFF-NODE points of the reconstructed
leaf polynomials.  The constants depend on the geometry and leaf size only, so
for p-convergence the components are reported as they are.

  eta          : random points inside the leaves, with the reference-coordinate
                 coefficients of each patch (Patch.ref_coeffs on the geometry
                 PDO: exactly the operator the leaves discretize);
  jump_flux_*  : physical normal derivative, outward from both sides, summed:
                     sum_sides s e_a . A~ grad_X phi / sqrt(e_a . A~ e_a)
                 (A~ reference metric, e_a the reference normal);
  jump_val_*   : phi_A - phi_B at the same points;
                 *_leaf on leaf faces inside a patch, *_patch on patch-patch
                 faces.  The patches are conforming: the gluing pairs every
                 Gauss slot of a patch-patch leaf face with one slot of the
                 other patch, and the tangential reference coordinates of the
                 two sides are related by an affine map, fitted from those
                 slots (residual checked) and verified at the random points in
                 physical space -- no map inversion, failure is an error;
  wall         : phi~ - phi_b at random points on the wall faces.

Every maximum is returned with its location (patch, psi, t, position inside the
leaf, face axis).  Each slab contributes only its own single slab around the
midplane, t in [t0 + H/2, t0 + 3H/2), so every point of the torus counts once.

Sanity checks (both should be small):
  eta_nodes  : eta at (a sample of) the leaf-interior collocation nodes
               (~ local chord tolerance times c: checks operator, signs, G);
  wall_gauss : phi~ - phi_b at the Gauss wall nodes (<< the off-node wall value).
"""
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.spatial import cKDTree

from solver.cylinder_slab import pdo3d_to_cyl

COMPONENTS = ("eta", "jump_flux_leaf", "jump_flux_patch", "jump_val_leaf", "jump_val_patch",
              "wall", "eta_nodes", "wall_gauss")
_AXIS_NAME = {"core": ("x", "y", "t"), "arch": ("radial", "tangential", "t")}


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

    def rel(self, X, b):
        return (X - self.lo[b]) / (self.hi[b] - self.lo[b])

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


def patch_label(P):
    return "core" if P.kind == "core" else "arch[rot %d]" % P.rot


def format_loc(loc):
    if not loc:
        return "-"
    s = "%s psi %.3f t %.3f leaf-rel (%.2f, %.2f, %.2f)" % (
        loc["patch"], loc["psi"], loc["t"], *loc["rel"])
    if loc.get("axis") is not None:
        s += ", face normal %s" % loc["axis"]
    return s


class ResidualEstimator:
    def __init__(self, pdo_geo, model, Rc, nsamp=3, node_samples=8, wall_tol=1e-9, dump=False):
        """pdo_geo: geometry PDO in OMS coordinates (the one given to OMS);
        model: PBProfiles / PBManufactured (g, fstar, psi, phi_b or phistar).
        nsamp: random points per leaf / leaf face; node_samples: collocation
        nodes per leaf for the eta_nodes check."""
        self.pde = pdo3d_to_cyl(pdo_geo)
        self.model, self.Rc, self.wall_tol = model, Rc, wall_tol
        self.nsamp, self.node_samples = nsamp, node_samples
        self.dump = dump               # keep every eta sample (coords, psi, patch, leaf position)
        self.phib = model.phi_b if hasattr(model, "phi_b") else model.phistar

    # ---------------------------------------------------------- pieces
    def _eta(self, F, X, b):
        val, grad, hess = F.eval(X, b)
        At, bt, c, f = F.P.ref_coeffs(X, self.pde)
        lap = -np.einsum("nkl,nkl->n", At, hess) + np.einsum("nk,nk->n", bt, grad) + c * val - f
        W = _oms(F.P, X)
        return lap + self.model.g(W, val) - self.model.fstar(W)

    def _side(self, F, X, b, a, s):
        """(value, outward physical normal derivative) on face axis a, sign s."""
        val, grad, _ = F.eval(X, b)
        At = F.P.ref_a(X, self.pde)
        return val, s * np.einsum("nk,nk->n", At[:, a, :], grad) / np.sqrt(At[:, a, a])

    def _is_wall(self, P, X):
        x = P.to_phys(X)
        return x[:, 0] ** 2 + x[:, 1] ** 2 >= self.Rc ** 2 * (1 - self.wall_tol)

    @staticmethod
    def _face_leaves(F, a, side):
        I = np.unravel_index(np.arange(int(np.prod(F.nleaf))), F.nleaf)
        return np.flatnonzero(I[a] == (F.nleaf[a] - 1 if side else 0))

    def _face_points(self, F, leaves, a, side, rng, gauss=None):
        """Points on face (a, side) of the given leaves: random tangential
        coordinates, or the Gauss-Legendre tensor nodes (gauss = (q1, q2))."""
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
        return (lo + r * (hi - lo)).reshape(-1, 3), np.repeat(leaves, r.shape[1])

    # ---------------------------------------------------------- bookkeeping
    def _record(self, out, key, vals, F, X, b, axis=None):
        if not len(vals):
            return
        a = np.abs(vals)
        i = int(np.argmax(a))
        if a[i] > out[key]:
            out[key] = float(a[i])
            W = _oms(F.P, X[i:i + 1])
            out["loc"][key] = dict(patch=patch_label(F.P), psi=float(self.model.psi(W)[0]),
                                   t=float(W[0, 0]), rel=tuple(F.rel(X[i:i + 1], b[i:i + 1])[0]),
                                   axis=None if axis is None else _AXIS_NAME[F.P.kind][axis])

    @staticmethod
    def merge(total, part):
        """Combine per-slab results: max (with location) / sums of counts."""
        if total is None:
            total = {k: (0 if k.startswith("n_") or k.endswith("fail") else 0.0) for k in part
                     if k not in ("loc", "dump")}
            total["loc"] = {}
        if "dump" in part:
            total.setdefault("dump", []).extend(part["dump"])
        for k, v in part.items():
            if k in ("loc", "dump"):
                continue
            if k.startswith("n_") or k.endswith("fail"):
                total[k] += v
            elif v > total[k]:
                total[k] = v
                if k in part["loc"]:
                    total["loc"][k] = part["loc"][k]
        return total

    # ---------------------------------------------------------- one slab
    def slab(self, ops, phi, t_own, seed=0):
        """Components on slab `ops` (CylinderSlabOps) with leaf field phi,
        restricted to t in [t_own[0], t_own[1])."""
        rng = np.random.default_rng(seed)
        imask = ops.interior_mask()
        Fs = [_PatchField(P, grid, off, phi, imask) for P, grid, off in ops.leaf_grids()]
        own = lambda X: (X[:, 2] >= t_own[0]) & (X[:, 2] < t_own[1])
        out = {k: 0.0 for k in COMPONENTS}
        out.update(n_eta=0, n_face_leaf=0, n_face_patch=0, n_wall=0, face_map_err=0.0, loc={})

        for k, F in enumerate(Fs):
            nb = F.V.shape[0]
            # -- interior residual at random points
            r = rng.uniform(size=(nb, self.nsamp, 3))
            X = (F.lo[:, None, :] + r * (F.hi - F.lo)[:, None, :]).reshape(-1, 3)
            b = np.repeat(np.arange(nb), self.nsamp)
            m = own(X)
            if m.any():
                eta = self._eta(F, X[m], b[m])
                self._record(out, "eta", eta, F, X[m], b[m])
                out["n_eta"] += int(m.sum())
                if self.dump:
                    W = _oms(F.P, X[m])
                    out.setdefault("dump", []).append(dict(
                        W=W, psi=self.model.psi(W), patch=np.full(len(W), k), rot=np.full(len(W), F.P.rot),
                        leaf=b[m], rel=F.rel(X[m], b[m]), eta=eta))
            # -- check: eta at a sample of the collocation nodes
            Jc = np.flatnonzero(F.Jc)
            pick = np.stack([rng.choice(Jc, size=min(self.node_samples, len(Jc)), replace=False)
                             for _ in range(nb)])
            Xn = F.grid[np.arange(nb)[:, None], pick].reshape(-1, 3)
            bn = np.repeat(np.arange(nb), pick.shape[1])
            m = own(Xn)
            if m.any():
                self._record(out, "eta_nodes", self._eta(F, Xn[m], bn[m]), F, Xn[m], bn[m])

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
                if not m.any():
                    continue
                X, b, b2 = X[m], b[m], b2[m]
                vA, fA = self._side(F, X, b, a, 1.0)
                vB, fB = self._side(F, X, b2, a, -1.0)
                self._record(out, "jump_flux_leaf", fA + fB, F, X, b, axis=a)
                self._record(out, "jump_val_leaf", vA - vB, F, X, b, axis=a)
                out["n_face_leaf"] += len(X)

            # -- wall faces of the patch (cross-section box faces on the wall)
            for a in range(2):
                for side in (0, 1):
                    leaves = self._face_leaves(F, a, side)
                    X, b = self._face_points(F, leaves, a, side, rng)
                    m = own(X) & self._is_wall(F.P, X)
                    if not m.any():
                        continue
                    Xw, bw = X[m], b[m]
                    v, _, _ = F.eval(Xw, bw)
                    self._record(out, "wall", v - self.phib(_oms(F.P, Xw)), F, Xw, bw, axis=a)
                    out["n_wall"] += len(Xw)
                    tang = [t for t in range(3) if t != a]
                    Xg, bg = self._face_points(F, leaves, a, side, rng,
                                               gauss=[len(F.nodes[t]) - 2 for t in tang])
                    mg = own(Xg) & self._is_wall(F.P, Xg)
                    if mg.any():
                        vg, _, _ = F.eval(Xg[mg], bg[mg])
                        self._record(out, "wall_gauss", vg - self.phib(_oms(F.P, Xg[mg])),
                                     F, Xg[mg], bg[mg], axis=a)

        # -- patch-patch faces, through the gluing correspondence
        for g in self._patch_faces(ops, Fs):
            FA, FB = Fs[g["kA"]], Fs[g["kB"]]
            n = self.nsamp
            rA = np.zeros((n, 3)); rA[:, g["tA"]] = rng.uniform(size=(n, 2)); rA[:, g["aA"]] = g["sA"]
            rB = np.zeros((n, 3)); rB[:, g["tB"]] = np.c_[rA[:, g["tA"]], np.ones(n)] @ g["M"]
            rB[:, g["aB"]] = g["sB"]
            bA, bB = np.full(n, g["bA"]), np.full(n, g["bB"])
            XA = FA.lo[bA] + rA * (FA.hi[bA] - FA.lo[bA])
            XB = FB.lo[bB] + rB * (FB.hi[bB] - FB.lo[bB])
            m = own(XA)
            if not m.any():
                continue
            XA, XB, bA, bB = XA[m], XB[m], bA[m], bB[m]
            err = float(np.max(np.abs(FA.P.to_phys(XA) - FB.P.to_phys(XB))))
            out["face_map_err"] = max(out["face_map_err"], err)
            if err > 1e-9:
                raise RuntimeError("patch face map %s leaf %d -> %s leaf %d: physical mismatch %.2e"
                                   % (patch_label(FA.P), g["bA"], patch_label(FB.P), g["bB"], err))
            vA, fA = self._side(FA, XA, bA, g["aA"], 1.0 if g["sA"] else -1.0)
            vB, fB = self._side(FB, XB, bB, g["aB"], 1.0 if g["sB"] else -1.0)
            self._record(out, "jump_flux_patch", fA + fB, FA, XA, bA, axis=g["aA"])
            self._record(out, "jump_val_patch", vA - vB, FA, XA, bA, axis=g["aA"])
            out["n_face_patch"] += len(XA)
        return out

    @staticmethod
    def _patch_faces(ops, Fs, tol=1e-9, fit_tol=1e-10):
        """Leaf-face pairs between different patches from the gluing
        (ops.dof_of_slot): per pair the leaves, face axes/sides and the affine
        map M (3x2) with  rB_tang = [rA_tang, 1] @ M  (relative leaf coords)."""
        pk, lf, ax, sd, rel = [], [], [], [], []
        for k, (F, d) in enumerate(zip(Fs, ops.patches)):
            nb, se = d["nb"], d["se"]
            Xs = np.asarray(d["Xref"]).reshape(nb, se, 3)
            r = (Xs - F.lo[:, None, :]) / (F.hi - F.lo)[:, None, :]
            r = r.reshape(-1, 3)
            d0, d1 = np.abs(r), np.abs(r - 1.0)
            on = np.minimum(d0, d1) < tol
            if not np.all(on.sum(axis=1) == 1):
                raise RuntimeError("Gauss slot not on exactly one leaf face (patch %s)" % patch_label(F.P))
            a = np.argmax(on, axis=1)
            rows = np.arange(len(r))
            pk.append(np.full(len(r), k)); lf.append(np.repeat(np.arange(nb), se))
            ax.append(a); sd.append((d1[rows, a] < d0[rows, a]).astype(int)); rel.append(r)
        pk, lf, ax, sd, rel = map(np.concatenate, (pk, lf, ax, sd, rel))
        dof = np.asarray(ops.dof_of_slot)
        if len(dof) != len(pk):
            raise RuntimeError("slot bookkeeping mismatch (%d slots vs %d dofs)" % (len(pk), len(dof)))
        o = np.argsort(dof, kind="stable")
        same = dof[o[1:]] == dof[o[:-1]]
        s1, s2 = o[:-1][same], o[1:][same]
        cross = pk[s1] != pk[s2]
        s1, s2 = s1[cross], s2[cross]
        swap = pk[s1] > pk[s2]
        s1[swap], s2[swap] = s2[swap], s1[swap].copy()
        key = np.stack([pk[s1], lf[s1], ax[s1], sd[s1], pk[s2], lf[s2], ax[s2], sd[s2]], axis=1)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        groups = []
        for gi, kk in enumerate(uk):
            idx = np.flatnonzero(inv == gi)
            kA, bA, aA, sA, kB, bB, aB, sB = (int(x) for x in kk)
            tA = [t for t in range(3) if t != aA]; tB = [t for t in range(3) if t != aB]
            RA, RB = rel[s1[idx]][:, tA], rel[s2[idx]][:, tB]
            if len(idx) < 3:
                raise RuntimeError("patch face pair with %d matched slots" % len(idx))
            A1 = np.c_[RA, np.ones(len(idx))]
            M = np.linalg.lstsq(A1, RB, rcond=None)[0]
            res = np.max(np.abs(A1 @ M - RB))
            if res > fit_tol:
                raise RuntimeError("patch face %s leaf %d / %s leaf %d: tangential map not affine "
                                   "(fit residual %.2e)" % (patch_label(Fs[kA].P), bA,
                                                            patch_label(Fs[kB].P), bB, res))
            groups.append(dict(kA=kA, bA=bA, aA=aA, sA=sA, tA=tA, kB=kB, bB=bB, aB=aB, sB=sB,
                               tB=tB, M=M))
        return groups


def save_dump(path, total):
    """All eta samples of a merged result (ResidualEstimator(dump=True)) to .npz:
    W (t, x, y), psi, patch (0 core, 1-4 arches), rot, leaf, rel (position in
    the leaf, patch-reference axis order), eta."""
    parts = total.get("dump", [])
    np.savez(path, **{k: np.concatenate([d[k] for d in parts]) for k in parts[0]})
