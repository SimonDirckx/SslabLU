"""
Local nonlinear solve on one double slab in statically condensed form
(pb_fixed_operator.tex, Sec. 5.3, Prop. 2 and eq. (lchord)):

    h_j       = f* - G[phi_j] + c_ref phi_j          (leaf-interior nodes)
    x_{j+1,I} = Aii^{-1} ( (Q h_j)_I - Aib b ),  x_{j+1,b} = b
    phi_{j+1} = leaf reconstruction from (x_{j+1}, h_j)

Only Aii(c_ref) is factored (once).  Q, the reconstruction and Aii all come
from the same CylinderSlab leaf operators (same pdo3d_to_cyl conversion and
gluing), so the iteration applies exactly M_i^{-1} = (-Lap + c_ref)^{-1}.
(Sign: CylinderSlab's glued system is A x = Q h, i.e. Q here is -Q of the tex.)
"""
import time
import numpy as np
import scipy.sparse.linalg as spla
import torch

import solver.solver as solverWrap
from solver.cylinder_slab import CylinderSlab, CylinderSlabOps


def _clock(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


class CondensedSlab:
    """Condensed slab solver for -Lap + c_ref with Aii factored.

    fast=False (reference): CylinderSlab(keep_leaf=True) with c_ref inside the
        PDO; every reduce/reconstruct call re-assembles and re-factors the leaf
        matrices through hpsmultidomain.
    fast=True : CylinderSlabOps -- geometry built once, c given as nodal values
        (c=..., in nodes() order) and changed with set_c(c) at the cost of
        batched leaf factorizations + a CSR scatter + a MUMPS refactorization
        (symbolic analysis reused when python-mumps allows it); each solve does
        ONE batched leaf solve shared by reduction and reconstruction.
        The PDO then carries the geometry only (pdo.c is added to c).

    solve(h, b) -> (phi on the leaf grid, skeleton x on all canonical dofs).
    """

    def __init__(self, geom, pdo, p, cyl, use_mumps=True, verbose=True, fast=False, c=None):
        cyl = dict(cyl)
        self.device = cyl.setdefault("device", torch.device("cpu"))
        self.fast, self.verbose = fast, verbose
        self.use_mumps = use_mumps and solverWrap.mumps is not None
        self._ctx = None
        self.timers = dict(reduce=0.0, skeleton=0.0, reconstruct=0.0)
        t0 = _clock(self.device)
        if fast:
            self.slab = CylinderSlabOps(geom, pdo, p, **cyl)
        else:
            self.slab = CylinderSlab(geom, pdo, p, keep_leaf=True, **cyl)
        self.t_build = _clock(self.device) - t0          # fast: geometry part only
        S = self.slab
        self.W = S.nodes()                         # leaf-grid nodes (OMS coords)
        self.interior = S.interior_mask()
        self.Wb = S.XX[S.Ib]                       # Dirichlet dofs
        self.Imid = S.midplane()
        self.t_set_c = 0.0
        self.factor = {}
        if not fast:
            self._factor()
        elif c is not None:
            self.set_c(c)
        if verbose:
            if self.factor:
                print("  slab: build %.2fs%s, Aii N=%d, %s factor %.2fs"
                      % (self.t_build, " (+ set_c %.2fs)" % self.t_set_c if fast else "",
                         S.Aii.shape[0], self.factor["backend"], self.factor["time_factor"]))
            else:
                print("  slab: geometry built in %.2fs (Aii N=%d), c not set yet"
                      % (self.t_build, len(S.Ii)))

    def set_c(self, c):
        """fast only: new zero-order coefficient at the leaf nodes, refactor."""
        if not self.fast:
            raise RuntimeError("set_c needs CondensedSlab(..., fast=True)")
        t0 = _clock(self.device)
        self.slab.set_c(c)
        self.t_set_c = _clock(self.device) - t0
        self._factor()
        if self.verbose:
            print("  slab: set_c %.2fs, %s factor %.2fs%s"
                  % (self.t_set_c, self.factor["backend"], self.factor["time_factor"],
                     " (analysis reused)" if self.factor.get("analysis_reused") else ""))

    def _factor(self):
        A = self.slab.Aii
        if self.use_mumps:
            reused = False
            if self._ctx is not None:
                # same sparsity pattern (CylinderSlabOps): try to keep the analysis
                ctx = self._ctx
                try:
                    ctx.set_matrix(A.tocsr(), symmetric=False)
                    t0 = time.perf_counter()
                    ctx.factor(reuse_analysis=True)
                    tf, ta, reused = time.perf_counter() - t0, 0.0, True
                    solverWrap._silence_mumps(ctx)
                except Exception:
                    ctx = None
            if not reused:
                ctx, ta, tf = solverWrap.setup_mumps(A.tocsr())
            self._ctx = ctx
            self._solve_ii = lambda r: np.asarray(solverWrap.mumps_solve(ctx, np.asfortranarray(r)))
            self.factor = solverWrap.mumps_factor_report(ctx, A, ta, tf)
            self.factor.update(backend="mumps", analysis_reused=reused)
        else:
            t0 = time.perf_counter()
            lu = spla.splu(A.tocsc())
            tf = time.perf_counter() - t0
            self._solve_ii = lu.solve
            self.factor = dict(backend="splu", N=A.shape[0], nnz_A=A.nnz,
                               nnz_factor=lu.L.nnz + lu.U.nnz, time_analysis=None,
                               time_factor=tf, mem_MB=None, analysis_reused=False)

    def solve(self, h, b):
        S, dev = self.slab, self.device
        t0 = _clock(dev)
        if self.fast:
            _, beta, z = S.reduce_body(h, b, return_z=True)
        else:
            _, beta = S.reduce_body(h, b)
        t1 = _clock(dev)
        x = np.empty(S.XX.shape[0])
        x[S.Ii] = self._solve_ii(beta)
        x[S.Ib] = b
        t2 = _clock(dev)
        phi = S.reconstruct(x, z=z) if self.fast else S.reconstruct(x, h)
        t3 = _clock(dev)
        self.timers["reduce"] += t1 - t0
        self.timers["skeleton"] += t2 - t1
        self.timers["reconstruct"] += t3 - t2
        return phi, x

    # ------------------------------------------------------------ off-grid
    def offgrid_points(self, npts_per_leaf=2, seed=0):
        """Random reference points inside every leaf (patch reference space)
        and the interpolation data to evaluate leaf-grid fields there.
        Returns OMS coordinates of the points; use self.interp(phi)."""
        rng = np.random.default_rng(seed)
        pts, ops = [], []
        for P, grid, off in self.slab.leaf_grids():
            nb, npl, _ = grid.shape
            lo, hi = grid.min(axis=1), grid.max(axis=1)            # (nb, 3) leaf boxes
            rel0 = (grid[0] - lo[0]) / (hi[0] - lo[0])             # same layout in every leaf
            nodes1d, idx = [], []
            for a in range(3):
                u, inv = np.unique(np.round(rel0[:, a], 12), return_inverse=True)
                nodes1d.append(u); idx.append(inv)
            r = rng.uniform(size=(nb, npts_per_leaf, 3))
            Xref = lo[:, None, :] + r * (hi - lo)[:, None, :]
            # barycentric Lagrange basis per axis: (nb, npts, p)
            Lw = []
            for a in range(3):
                u = nodes1d[a]
                w = 1.0 / np.prod(u[:, None] - u[None, :] + np.eye(len(u)), axis=1)
                d = r[..., a][..., None] - u
                d[np.abs(d) < 1e-14] = 1e-14
                t = w / d
                Lw.append(t / t.sum(-1, keepdims=True))
            B = Lw[0][..., idx[0]] * Lw[1][..., idx[1]] * Lw[2][..., idx[2]]   # (nb, npts, p^3)
            ops.append((off, nb, npl, B))
            Xc = P.to_phys(Xref.reshape(-1, 3))
            pts.append(Xc[:, [2, 0, 1]])
        self._interp = ops
        return np.vstack(pts)

    def interp(self, phi):
        out = []
        for off, nb, npl, B in self._interp:
            v = phi[off:off + nb * npl].reshape(nb, npl)
            out.append(np.einsum("lkn,ln->lk", B, v).reshape(-1))
        return np.concatenate(out)


# ---------------------------------------------------------------- chord
def chord(solve, phi0, h_of, tol=1e-12, maxit=200, monitor=None, interior=None):
    """Generic chord driver.

    solve(h)  -> (phi_new, x)      one application of M^{-1} with fixed Dirichlet data
    h_of(phi) -> h = f* - G[phi] + c_ref phi
    Stops when max|phi_{j+1} - phi_j| < tol.  monitor(j, phi, x, log) may add
    entries to the per-iteration log dict.

    The logged 'residual' of phi_j (j >= 1) is the leaf-interior collocation
    residual of the NONLINEAR system, |h_j - h_{j-1}|_inf: phi_j satisfies
    K phi_j + c_ref phi_j = h_{j-1} exactly, so K phi_j + G[phi_j] - f* = h_{j-1} - h_j.
    """
    phi, h_prev, logs = phi0, None, []
    for j in range(maxit):
        t0 = time.perf_counter()
        h = h_of(phi)
        t_g = time.perf_counter() - t0
        res = np.nan
        if h_prev is not None:
            d = h - h_prev
            res = np.max(np.abs(d if interior is None else d[interior]))
        phi_new, x = solve(h)
        inc = np.max(np.abs(phi_new - phi))
        log = dict(it=j + 1, increment=inc, residual_prev=res, t_G=t_g)
        if monitor is not None:
            monitor(j + 1, phi_new, x, log)
        logs.append(log)
        phi, h_prev = phi_new, h
        if inc < tol:
            break
    return phi, x, logs
