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
leaf polynomials.  Face components are evaluated FACE-WISE: the restriction of a
leaf polynomial (and of its normal derivative) to a leaf face is an exact
tangential polynomial, built once per face and interpolated to the points (no
volume evaluation, no Hessian):
  leaf faces  (both leaves in one patch): one map, so one A~ at each point; the
              jump traces [phi], [d_Xa phi] form one face polynomial, giving
              [phi] and the reference-gradient jump; conormal applied once;
  patch faces (leaves in two patches): per side the traces of its own leaf, at
              the matched points (affine tangential correspondence), conormal
              with that side's A~, outward fluxes summed;
  wall        value trace only.
check_points per batch are re-evaluated with the volume polynomial
(face_eval_err).  The constants depend on the geometry and leaf size only, so
for p-convergence the components are reported as they are.

  eta          : random points inside the leaves, with the reference-coordinate
                 coefficients of each patch (Patch.ref_coeffs on the geometry
                 PDO: exactly the operator the leaves discretize);
  jump_flux_*  : physical normal derivative, outward from both sides, summed:
                     sum_sides s e_a . A~ grad_X phi / sqrt(e_a . A~ e_a)
                 (A~ reference metric, e_a the reference normal);
  jump_val_*   : phi_A - phi_B at the same points;
                 All are leaf faces; the suffix says which kind:
                   *_leaf   within-patch faces: both leaves in one patch (one
                            reference map, points shared directly);
                   *_patch  between-patch faces: the leaves belong to two
                            patches (two maps, glued).
                 (Suffixes kept for CSV compatibility; printed as "within-patch"
                 / "between-patch".)  The patches are conforming: the gluing
                 pairs every Gauss slot of a between-patch face with one slot
                 of the other patch, and the tangential reference coordinates of the
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
import time

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.spatial import cKDTree

from solver.cylinder_slab import pdo3d_to_cyl

COMPONENTS = ("eta", "jump_flux_leaf", "jump_val_leaf", "jump_flux_patch", "jump_val_patch",
              "wall", "eta_nodes", "wall_gauss")
TIMERS = ("t_interior", "t_leaf_faces", "t_patch_faces", "t_wall")
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
        # tensor form T[leaf, i0, i1, i2] (axis nodes sorted) for face traces
        q = tuple(len(u) for u in self.nodes)
        if self.npl != int(np.prod(q)):
            raise RuntimeError("leaf grid is not a full tensor grid (%d nodes, %s per axis)" % (self.npl, q))
        for a in range(3):
            if abs(self.nodes[a][0]) > 1e-10 or abs(self.nodes[a][-1] - 1.0) > 1e-10:
                raise RuntimeError("leaf nodes do not include the leaf faces (axis %d)" % a)
        self.T = np.empty((nb,) + q)
        self.T[:, self.idx[0], self.idx[1], self.idx[2]] = self.V
        self.dend = [_axis_basis(u, np.array([0.0, 1.0]))[1] for u in self.nodes]   # d/dr at r = 0, 1

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

    def face_trace(self, leaves, a, side, deriv=True):
        """Traces of the leaf polynomials on face (axis a, side 0/1) of the given
        leaves, as tensors on the tangential nodes (tangential axes in increasing
        order): value (nl, q0, q1) and the reference normal derivative d/dX_a
        (nl, q0, q1), or None without deriv.  Exact: restriction of a tensor
        polynomial to a face, and of its derivative."""
        T = self.T[leaves]
        val = np.take(T, -1 if side else 0, axis=a + 1)
        if not deriv:
            return val, None
        dn = np.moveaxis(T, a + 1, -1) @ self.dend[a][1 if side else 0]
        return val, dn / (self.hi[leaves, a] - self.lo[leaves, a])[:, None, None]

    def face_eval(self, Tv, Tn, j, a, rt, span_t):
        """Face polynomials at N points of faces with normal axis a.
        Tv, Tn (nf, q0, q1): value and normal-derivative traces (Tn may be None);
        j (N,) face index; rt (N, 2) relative tangential coordinates; span_t (N, 2)
        tangential leaf widths.  Returns value (N,) and the reference gradient
        (N, 3) (normal component from Tn, tangential ones by differentiating Tv),
        or None without Tn."""
        tang = [t for t in range(3) if t != a]
        B = [_axis_basis(self.nodes[t], rt[:, c])[:2] for c, t in enumerate(tang)]
        V = Tv[j]
        val = np.einsum("nij,ni,nj->n", V, B[0][0], B[1][0])
        if Tn is None:
            return val, None
        g = np.empty((len(j), 3))
        g[:, a] = np.einsum("nij,ni,nj->n", Tn[j], B[0][0], B[1][0])
        g[:, tang[0]] = np.einsum("nij,ni,nj->n", V, B[0][1], B[1][0]) / span_t[:, 0]
        g[:, tang[1]] = np.einsum("nij,ni,nj->n", V, B[0][0], B[1][1]) / span_t[:, 1]
        return val, g

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
    def __init__(self, pdo_geo, model, Rc, nsamp=3, node_samples=8, wall_tol=1e-9, dump=False,
                 check_points=16):
        """pdo_geo: geometry PDO in OMS coordinates (the one given to OMS);
        model: PBProfiles / PBManufactured (g, fstar, psi, phi_b or phistar).
        nsamp: random points per leaf / leaf face; node_samples: collocation
        nodes per leaf for the eta_nodes check; check_points: per batch of face
        points, how many are re-evaluated with the full volume polynomial
        (face_eval_err, expect ~1e-13; 0 = off)."""
        self.check_points = int(check_points)
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

    def _conormal(self, P, X, a, g):
        """e_a . A~ g / sqrt(e_a . A~ e_a) at reference points X of patch P, for
        reference gradients (or gradient jumps) g: the physical normal
        derivative along +grad(X_a) (A~ = inverse metric of the patch coords)."""
        At = P.ref_a(X, self.pde)
        return np.einsum("nk,nk->n", At[:, a, :], g) / np.sqrt(At[:, a, a])

    @staticmethod
    def _check(out, ref, got):
        """face-wise vs volume evaluation, relative to 1 + |value|."""
        if len(ref):
            d = np.abs(np.asarray(ref) - np.asarray(got)) / (1.0 + np.abs(ref))
            out["face_eval_err"] = max(out["face_eval_err"], float(np.max(d)))

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
            if k.startswith(("n_", "t_")) or k.endswith("fail"):
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
        out = {k: 0.0 for k in COMPONENTS + TIMERS}
        out.update(n_eta=0, n_face_leaf=0, n_face_patch=0, n_wall=0, face_map_err=0.0,
                   face_eval_err=0.0, loc={})
        ncheck = slice(0, self.check_points)

        for k, F in enumerate(Fs):
            nb = F.V.shape[0]
            span = F.hi - F.lo
            t0 = time.perf_counter()
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

            out["t_interior"] += time.perf_counter() - t0

            # -- LEAF FACES (both leaves in this patch): one map, one A~, so the
            #    jump of the two leaf polynomials is itself a face polynomial:
            #    [phi] and [grad_X phi] from the jump traces, conormal applied once.
            t0 = time.perf_counter()
            I = np.unravel_index(np.arange(nb), F.nleaf)
            for a in range(3):
                tang = [t for t in range(3) if t != a]
                sel = np.flatnonzero(I[a] < F.nleaf[a] - 1)
                if not len(sel):
                    continue
                J = [I[c][sel].copy() for c in range(3)]
                J[a] += 1
                nbr = np.ravel_multi_index(tuple(J), F.nleaf)    # neighbour across the +a face
                X, b = self._face_points(F, sel, a, 1, rng)
                jf = np.repeat(np.arange(len(sel)), self.nsamp)
                m = own(X)
                if not m.any():
                    continue
                X, b, jf = X[m], b[m], jf[m]
                used, j = np.unique(jf, return_inverse=True)
                j = np.asarray(j).reshape(-1)
                vA, nA = F.face_trace(sel[used], a, 1)
                vB, nB = F.face_trace(nbr[used], a, 0)
                jv, jg = F.face_eval(vA - vB, nA - nB, j, a, F.rel(X, b)[:, tang], span[b][:, tang])
                self._record(out, "jump_flux_leaf", self._conormal(F.P, X, a, jg), F, X, b, axis=a)
                self._record(out, "jump_val_leaf", jv, F, X, b, axis=a)
                out["n_face_leaf"] += len(X)
                if self.check_points:
                    Xc, bc, b2 = X[ncheck], b[ncheck], nbr[jf[ncheck]]
                    VA, GA, _ = F.eval(Xc, bc)
                    VB, GB, _ = F.eval(Xc, b2)
                    self._check(out, VA - VB, jv[ncheck])
                    self._check(out, GA - GB, jg[ncheck])
            out["t_leaf_faces"] += time.perf_counter() - t0

            # -- wall faces of the patch (cross-section box faces on the wall)
            t0 = time.perf_counter()
            for a in range(2):
                tang = [t for t in range(3) if t != a]
                for side in (0, 1):
                    leaves = self._face_leaves(F, a, side)
                    X, b = self._face_points(F, leaves, a, side, rng)
                    jf = np.repeat(np.arange(len(leaves)), self.nsamp)
                    m = own(X) & self._is_wall(F.P, X)
                    if not m.any():
                        continue
                    Tv, _ = F.face_trace(leaves, a, side, deriv=False)
                    Xw, bw, jw = X[m], b[m], jf[m]
                    v, _ = F.face_eval(Tv, None, jw, a, F.rel(Xw, bw)[:, tang], span[bw][:, tang])
                    self._record(out, "wall", v - self.phib(_oms(F.P, Xw)), F, Xw, bw, axis=a)
                    out["n_wall"] += len(Xw)
                    if self.check_points:
                        self._check(out, F.eval(Xw[ncheck], bw[ncheck])[0], v[ncheck])
                    q = [len(F.nodes[t]) - 2 for t in tang]
                    Xg, bg = self._face_points(F, leaves, a, side, rng, gauss=q)
                    jg = np.repeat(np.arange(len(leaves)), q[0] * q[1])
                    mg = own(Xg) & self._is_wall(F.P, Xg)
                    if mg.any():
                        Xg, bg, jg = Xg[mg], bg[mg], jg[mg]
                        vg, _ = F.face_eval(Tv, None, jg, a, F.rel(Xg, bg)[:, tang], span[bg][:, tang])
                        self._record(out, "wall_gauss", vg - self.phib(_oms(F.P, Xg)),
                                     F, Xg, bg, axis=a)
            out["t_wall"] += time.perf_counter() - t0

        # -- PATCH FACES (the two leaves belong to different patches): two maps,
        #    two A~.  Per side: value and normal-derivative traces of its own leaf
        #    polynomial, evaluated at the matched points (the tangential
        #    correspondence is affine, so still a polynomial), conormal with that
        #    side's A~; the jump is the sum of the two outward fluxes.
        t0 = time.perf_counter()
        groups = self._patch_faces(ops, Fs)
        n = self.nsamp
        cols = {key: [] for key in ("kA", "bA", "aA", "sA", "rA", "kB", "bB", "aB", "sB", "rB")}
        for g in groups:
            rA = np.zeros((n, 3)); rA[:, g["tA"]] = rng.uniform(size=(n, 2)); rA[:, g["aA"]] = g["sA"]
            rB = np.zeros((n, 3)); rB[:, g["tB"]] = np.c_[rA[:, g["tA"]], np.ones(n)] @ g["M"]
            rB[:, g["aB"]] = g["sB"]
            for key in ("kA", "bA", "aA", "sA", "kB", "bB", "aB", "sB"):
                cols[key].append(np.full(n, g[key]))
            cols["rA"].append(rA); cols["rB"].append(rB)
        if groups:
            c = {key: np.concatenate(v) for key, v in cols.items()}
            XA = self._ref_points(Fs, c["kA"], c["bA"], c["rA"])
            m = own(XA)
            if m.any():
                c = {key: v[m] for key, v in c.items()}
                XA = XA[m]
                XB = self._ref_points(Fs, c["kB"], c["bB"], c["rB"])
                err = np.max(np.abs(self._phys(Fs, c["kA"], XA) - self._phys(Fs, c["kB"], XB)), axis=1)
                out["face_map_err"] = max(out["face_map_err"], float(err.max()))
                if err.max() > 1e-9:
                    i = int(np.argmax(err))
                    raise RuntimeError("patch face map %s leaf %d -> %s leaf %d: physical mismatch %.2e"
                                       % (patch_label(Fs[c["kA"][i]].P), c["bA"][i],
                                          patch_label(Fs[c["kB"][i]].P), c["bB"][i], err[i]))
                vA, fA, gA = self._side_eval(Fs, c["kA"], c["bA"], c["aA"], c["sA"], c["rA"], XA)
                vB, fB, gB = self._side_eval(Fs, c["kB"], c["bB"], c["aB"], c["sB"], c["rB"], XB)
                self._record_at(out, "jump_flux_patch", fA + fB, Fs, c["kA"], XA, c["bA"], c["aA"])
                self._record_at(out, "jump_val_patch", vA - vB, Fs, c["kA"], XA, c["bA"], c["aA"])
                out["n_face_patch"] += len(XA)
                for i in range(min(self.check_points, len(XA))):
                    for k_, X_, b_, v_, g_ in ((c["kA"], XA, c["bA"], vA, gA), (c["kB"], XB, c["bB"], vB, gB)):
                        V, G, _ = Fs[k_[i]].eval(X_[i:i + 1], b_[i:i + 1])
                        self._check(out, V, v_[i:i + 1])
                        self._check(out, G, g_[i:i + 1])
        out["t_patch_faces"] += time.perf_counter() - t0
        return out

    @staticmethod
    def _ref_points(Fs, k, b, r):
        """Reference points from (patch, leaf, relative position) per point."""
        X = np.empty_like(r)
        for kk in np.unique(k):
            s = k == kk
            F = Fs[kk]
            X[s] = F.lo[b[s]] + r[s] * (F.hi[b[s]] - F.lo[b[s]])
        return X

    @staticmethod
    def _phys(Fs, k, X):
        Y = np.empty_like(X)
        for kk in np.unique(k):
            s = k == kk
            Y[s] = Fs[kk].P.to_phys(X[s])
        return Y

    def _side_eval(self, Fs, k, b, a, side, r, X):
        """One side of patch-face points (patch k, leaf b, face axis a / side,
        relative position r, reference point X): value, OUTWARD conormal flux,
        reference gradient.  Face traces per (patch, axis, side), each leaf once."""
        N = len(k)
        val, flux, grad = np.empty(N), np.empty(N), np.empty((N, 3))
        for kk in np.unique(k):
            F = Fs[kk]
            span = F.hi - F.lo
            for aa in range(3):
                tang = [t for t in range(3) if t != aa]
                for ss in (0, 1):
                    sel = np.flatnonzero((k == kk) & (a == aa) & (side == ss))
                    if not len(sel):
                        continue
                    ul, j = np.unique(b[sel], return_inverse=True)
                    Tv, Tn = F.face_trace(ul, aa, ss)
                    v, g = F.face_eval(Tv, Tn, np.asarray(j).reshape(-1), aa,
                                       r[sel][:, tang], span[b[sel]][:, tang])
                    val[sel], grad[sel] = v, g
                    flux[sel] = (1.0 if ss else -1.0) * self._conormal(F.P, X[sel], aa, g)
        return val, flux, grad

    def _record_at(self, out, key, vals, Fs, k, X, b, axis):
        """_record for points of mixed patches: location from the maximizer's patch."""
        if not len(vals):
            return
        i = int(np.argmax(np.abs(vals)))
        self._record(out, key, vals[i:i + 1], Fs[k[i]], X[i:i + 1], b[i:i + 1], axis=int(axis[i]))

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
        order = np.argsort(inv, kind="stable")                  # slots grouped, ascending within
        ends = np.cumsum(np.bincount(inv, minlength=len(uk)))
        for gi, kk in enumerate(uk):
            idx = order[(ends[gi - 1] if gi else 0):ends[gi]]
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
