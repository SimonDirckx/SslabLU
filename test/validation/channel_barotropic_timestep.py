# =============================================================================
# channel_barotropic_timestep.py
#
# Time-stepped barotropic mode (implicit free surface) for the 2D re-entrant
# channel, in SslabLU. This is the time-dependent companion of
# channel_barotropic_sweep.py: same operator, same geometry, but now marching
# the linearized rotating shallow-water system
#
#     u_t - f v = -(g/L) eta_x                 (x nondimensionalized by L)
#     v_t + f u = -(g/L) eta_y
#     eta_t + (1/L) div( H(x) [u,v] ) = 0
#
# with a first-order IMEX (backward-Euler) implicit free surface, following the
# structure of test/rkprop/IMEXconvdiv.py and manuscript Sec. 6.1: gravity-wave
# terms implicit, Coriolis explicit. Substituting the velocity update into the
# continuity equation gives the per-step elliptic problem
#
#     -div( D(x) grad eta^{n+1} ) + eta^{n+1} = R^n            on the channel,
#      D(x) = g dt^2 H(x) / L^2,
#      R^n  = eta^n - (dt/L) div( H(x) [u*, v*] ),   [u*,v*] = explicit predictor,
#
# which is EXACTLY the operator of the sweep script (D = ell2 * H/H0 with
# ell2 = g H0 dt^2 / L^2, screening coefficient +1). Since dt is fixed, the
# operator is fixed: ONE S-map assembly and ONE cyclic block-Thomas
# factorization (with the SMW corner correction) serve every timestep; the
# per-step cost is a body-load rhs rebuild plus a block back-substitution.
# (Compare manuscript Sec. 6.1, which reused the S reduction but ran GMRES at
# every IMEX step; and Sec. 5.3, which deferred the direct solver.)
#
# BODY LOADS. The hpsalt skeleton system is A_CC u_C = -A_CX u_X + b_C where
# b_C = hps.reduce_body(...)[I_Ctot] is the statically-condensed body load
# (domain_driver.get_rhs). In the oms interface convention (Stot = I + S with
# S_l/S_r = +(Aii^{-1} Aib[:,I])[Ic]), the per-slab interface rhs is therefore
#
#     rhs = ( Aii^{-1} ( b_C - Aib[:,Igb] fgb ) )[Ic]
#
# (bc-only special case is oms.construct_Stot_helper's rhs = -(...)[Ic]).
# Reconstruction uses solve_dir_full(g, ff_body=fvec), which threads the body
# load through the leaf solves. Both the sign/indexing of this rhs and the leaf
# derivative matrices are verified by a manufactured-solution GATE (cubic
# u = x^3 + y^3, collocation-exact) before any time stepping happens --
# IMEXconvdiv.py's "gate" pattern ported to the oms stack.
#
# GRADIENTS OF THE NUMERICAL FIELD. R^n and the velocity update need grad(eta)
# and div(H u) from the numerical solution: applied per leaf box with the
# spectral differentiation matrices hps.H.Ds[3] (d/dx) and Ds[4] (d/dy), the
# same trick as IMEXconvdiv.py (einsum over the (nboxes, p^2) leaf arrays).
#
# BOUNDARY CONDITIONS. Periodic in x (cyclic slabs, seam slab on fictitious
# [-H, +H] -- all coefficient/IC/BC callables are 1-periodic in x). The y-walls
# carry Dirichlet SSH data (default 0: "clamped" walls, an open boundary to a
# reservoir at rest; set SSLABLU_WALL_AMP for a time-periodic "tidal" wall
# driver). NOTE: these are NOT solid walls -- no-normal-flow walls would need a
# Neumann/mixed problem_type through the oms path, which convdiv only did by
# hand in its own RefSlab. Consequence: total mass is conserved only up to the
# (physical) wall flux, so the conservation diagnostic below compares the
# divergence form against the non-conservative form, where the NON-conservative
# form adds a spurious volume term on top of the shared wall flux.
#
# SCENARIO (default): geostrophic adjustment. eta(0) is a periodic-in-x bump
# (von-Mises in x, Gaussian in y), u = v = 0. Gravity waves radiate around the
# channel (heavily damped by backward Euler once dt is large -- the point of an
# implicit free surface), scatter off the ridge, and leave a rotationally
# balanced residual. Diagnostics per step: mass drift |M - M0| for divergence
# vs non-conservative form (IMEXconvdiv's telescoping table), energy, max|eta|
# (stability check), and rhs/solve/reconstruction timings.
#
# Tests / outputs:
#   GATE    manufactured cubic on one all-Dirichlet double slab: derivative
#           matrices, skeleton body-rhs sign, body-load reconstruction.
#   RUN     NSTEPS of backward-Euler IMEX, both PDO forms (mass comparison).
#   OPTIONAL SSLABLU_DTCONV=1: dt-convergence ratios (~2.0 for backward Euler);
#           rebuilds the operator per dt, so this is slow and off by default.
#
#   channel_timestep_diag.csv        per-step diagnostics
#   channel_timestep_fields.png      eta snapshots at t = 0, T/2, T
#   channel_timestep_diagnostics.png mass drift / energy / max|eta| / timings
#
# Environment overrides (crystal-test style):
#   SSLABLU_N          slabs / interfaces              (default 8)
#   SSLABLU_P          polynomial order p, p_disc=p+2  (default 12)
#   SSLABLU_NPAN_X     x-panels per double slab, EVEN  (default 4)
#   SSLABLU_NPAN_Y     y-panels across the channel     (default 8)
#   SSLABLU_DT_H       timestep in hours               (default 0.25)
#   SSLABLU_NSTEPS     number of steps                 (default 48)
#   SSLABLU_RK         HBS rank for S-maps; 0 = dense  (default 0)
#   SSLABLU_COMPARE_FORMS  1 = also run non-conservative form   (default 1)
#   SSLABLU_DTCONV     1 = run dt-convergence study             (default 0)
#   SSLABLU_WALL_AMP   tidal wall SSH amplitude [m]             (default 0)
# =============================================================================

import io
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.double)

# --- oms packages (run from repo root; fallback mirrors the sweep script) ----
try:
    import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
except ImportError:
    _HPSMULTIDOMAIN_ROOT = Path(__file__).resolve().parent / "solver" / "hpsmultidomain"
    if str(_HPSMULTIDOMAIN_ROOT) not in sys.path:
        sys.path.insert(0, str(_HPSMULTIDOMAIN_ROOT))
    import hpsmultidomain.pdo as pdo

import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import multislab.omsdirectsolve as omsdirectsolve

CPU = torch.device('cpu')


################################################################
#
#   PHYSICAL SET-UP (identical to channel_barotropic_sweep.py)
#
################################################################

GRAV     =  9.81          # m/s^2
H0       =  4000.0        # reference depth [m]
LCHAN    =  1.0e6         # channel width Ly [m]; domain nondimensionalized by this
RIDGE_HR =  0.8           # ridge height as a fraction of H0
RIDGE_KB =  40.0          # von-Mises concentration: larger = narrower ridge
FCOR     = -1.0e-4        # Coriolis parameter [1/s] (explicit; needs |FCOR|*dt < 2)

# Meridional gap through the ridge (difference-of-tanh notch, cf. the Julia
# ridge_function). GAP_DEPTH = 0 disables it (y-independent ridge); 1 cuts the
# crest fully to full ocean depth inside [GAP_Y0, GAP_Y1]. Overridable below.
GAP_DEPTH = float(os.environ.get("SSLABLU_GAP_DEPTH", "1.0"))
GAP_Y0    = float(os.environ.get("SSLABLU_GAP_Y0", str(1.0 / 6.0)))
GAP_Y1    = float(os.environ.get("SSLABLU_GAP_Y1", "0.5"))
GAP_W     = float(os.environ.get("SSLABLU_GAP_W", "0.05"))   # tanh edge width

# initial SSH bump: 1-periodic von-Mises in x, Gaussian in y (seam-safe)
ETA0   = 1.0             # bump amplitude [m]
IC_KB  = 20.0            # x-concentration of the bump
IC_AY  = 50.0            # y-Gaussian decay
IC_CX  = 0.25            # bump center (off the ridge at x = 0.5)
IC_CY  = 0.5

GATE_TOL = 1.0e-8        # hard-fail threshold for the manufactured gate


def bump_x(x, lib):
    """1-periodic Gaussian-like ridge profile, centered at x = 0.5."""
    return lib.exp(RIDGE_KB * (lib.cos(2.0 * np.pi * (x - 0.5)) - 1.0))


def dbump_x(x, lib):
    """d/dx of bump_x (analytic)."""
    return (-2.0 * np.pi * RIDGE_KB * lib.sin(2.0 * np.pi * (x - 0.5))
            * bump_x(x, lib))


def gapfac(y, lib):
    """Meridional modulation of the ridge amplitude: ~1 outside the gap band,
    ~(1 - GAP_DEPTH) inside [GAP_Y0, GAP_Y1] (a difference-of-tanh notch, cf.
    the Julia ridge_function). Function of y ONLY, so bump_x's 1-periodicity in
    x -- and hence the seam-slab extension -- is untouched. GAP_DEPTH = 0 makes
    this identically 1 and recovers the y-independent ridge (c2 = Hpy = 0)."""
    if GAP_DEPTH == 0.0:
        return 1.0 + 0.0 * y
    return 1.0 - 0.5 * GAP_DEPTH * (lib.tanh((y - GAP_Y0) / GAP_W)
                                    - lib.tanh((y - GAP_Y1) / GAP_W))


def dgapfac(y, lib):
    """d/dy of gapfac (sech^2 = 1 - tanh^2)."""
    if GAP_DEPTH == 0.0:
        return 0.0 * y
    t0 = lib.tanh((y - GAP_Y0) / GAP_W)
    t1 = lib.tanh((y - GAP_Y1) / GAP_W)
    return -0.5 * GAP_DEPTH * ((1.0 - t0 ** 2) - (1.0 - t1 ** 2)) / GAP_W


def depth_frac(x, y, lib=np):
    """H(x,y)/H0 = 1 - hr*gap(y)*bump(x): the (nondimensional) bathymetry.
    The ridge crest (bump peak at x = 0.5) is cut down to (1 - GAP_DEPTH) of
    its height inside the meridional gap band, opening a deep channel there."""
    return 1.0 - RIDGE_HR * gapfac(y, lib) * bump_x(x, lib)


def ddepth_frac_dx(x, y, lib=np):
    """d/dx of depth_frac (analytic)."""
    return -RIDGE_HR * gapfac(y, lib) * dbump_x(x, lib)


def ddepth_frac_dy(x, y, lib=np):
    """d/dy of depth_frac (analytic; zero when GAP_DEPTH = 0)."""
    return -RIDGE_HR * dgapfac(y, lib) * bump_x(x, lib)


def make_pdo(ell2, conservative=True):
    """Screened-diffusion PDO, ell2 = g H0 dt^2 / L^2:

        -div(D grad eta) + eta,   D(x,y) = ell2 * depth_frac(x,y).

    conservative=True is the true divergence form: under the hpsalt convention
    A = -c11 u_xx - c22 u_yy + c1 u_x + c2 u_y + c u, that means c1 = -dD/dx AND
    c2 = -dD/dy. The gap makes D depend on y, so c2 is now nonzero and MUST be
    included -- dropping it would leave an operator that is not -div(D grad).
    conservative=False DROPS both first-order terms (i.e. -D Lap(eta) + eta):
    the non-conservative form used as the mass-drift comparison, cf. IMEXconvdiv.
    """
    def Dcoef(p):
        lib = torch if torch.is_tensor(p) else np
        return ell2 * depth_frac(p[:, 0], p[:, 1], lib)

    def c1(p):   # c1 = -dD/dx
        lib = torch if torch.is_tensor(p) else np
        return -ell2 * ddepth_frac_dx(p[:, 0], p[:, 1], lib)

    def c2(p):   # c2 = -dD/dy (nonzero only where the gap varies in y)
        lib = torch if torch.is_tensor(p) else np
        return -ell2 * ddepth_frac_dy(p[:, 0], p[:, 1], lib)

    if conservative:
        return pdo.PDO_2d(c11=Dcoef, c22=Dcoef, c1=c1, c2=c2,
                          c=pdo.const(c=1.0))
    return pdo.PDO_2d(c11=Dcoef, c22=Dcoef, c=pdo.const(c=1.0))


################################################################
#
#   GEOMETRY: flat unit square, x-periodic via cyclic slabs
#
################################################################

BNDS = [[0.0, 0.0], [1.0, 1.0]]


def channel_dSlabs(N):
    """N double-wide slabs; slab n is centered on interface x = n*H.
    Slab 0 straddles the seam: [-H, +H] (fictitious extension; every callable
    here is 1-periodic in x, so the extension is automatic)."""
    dSlabs = []
    H = (BNDS[1][0] - BNDS[0][0]) / N
    connectivity = []
    for n in range(N):
        c = BNDS[0][0] + n * H
        dSlabs += [[[c - H, BNDS[0][1]], [c + H, BNDS[1][1]]]]
        connectivity += [[(n - 1) % N, (n + 1) % N]]
    return dSlabs, connectivity, H


def gb(p):
    """Global boundary = the y-walls only. x has no boundary (periodic)."""
    lib = torch if torch.is_tensor(p) else np
    return ((lib.abs(p[:, 1] - BNDS[0][1]) < 1e-14) |
            (lib.abs(p[:, 1] - BNDS[1][1]) < 1e-14))


################################################################
#
#   KEPT PER-SLAB SOLVERS + QUADRATURE
#
################################################################

def bary_mat(nodes, targets):
    """Barycentric interpolation matrix from 1D nodes to target points
    (IMEXconvdiv.py's plot_field trick, with generic weights)."""
    n = len(nodes)
    bw = np.array([1.0 / np.prod(nodes[j] - np.delete(nodes, j))
                   for j in range(n)])
    M = np.zeros((len(targets), n))
    for k, t in enumerate(targets):
        d = t - nodes
        j0 = int(np.argmin(np.abs(d)))
        if abs(d[j0]) < 1e-13:
            M[k, j0] = 1.0
        else:
            w = bw / d
            M[k] = w / w.sum()
    return M


def cheb_quad_weights(nodes):
    """Exact quadrature weights for polynomial interpolation at the given
    (Chebyshev) nodes on [nodes[0], nodes[-1]]: w = V^{-T} m with the Chebyshev
    Vandermonde V[j,k] = T_k(t_j) and moments m_k = int_{-1}^{1} T_k."""
    a, b = nodes[0], nodes[-1]
    t = (2.0 * nodes - a - b) / (b - a)
    n = len(nodes)
    V = np.polynomial.chebyshev.chebvander(t, n - 1)
    m = np.zeros(n)
    ks = np.arange(0, n, 2)
    m[ks] = 2.0 / (1.0 - ks.astype(float) ** 2)
    return np.linalg.solve(V.T, m) * (b - a) / 2.0


class SlabSolve:
    """Everything needed per double slab to (a) rebuild the interface rhs from
    a body load and (b) reconstruct the full leaf field, every timestep.

    oms.construct_Stot_helper discards its slab solvers after assembling S
    (`del ... solver`), so each slab is discretized a second time here and
    KEPT. Wasteful but honest; unifying the two passes would mean extending
    oms itself to optionally retain its solvers.
    """

    def __init__(self, geom, diff_op, opts, gb_vec, own_split=None):
        self.sv = solverWrap.solverWrapper(opts)
        with redirect_stdout(io.StringIO()):
            self.sv.construct(np.array(geom), diff_op)
        sl = oms.slab(np.array(geom), gb_vec)
        (self.Il, self.Ir, self.Ic,
         self.Igb, self.XXi, self.XXb) = sl.compute_idxs_and_pts(self.sv)
        # the wrapper hands these back as torch tensors; everything downstream
        # here is numpy
        if torch.is_tensor(self.XXi):
            self.XXi = self.XXi.detach().numpy()
        if torch.is_tensor(self.XXb):
            self.XXb = self.XXb.detach().numpy()

        dd = self.sv.solver                       # hpsalt Domain_Driver
        self.dd = dd
        self.gx = dd.hps.grid_xx.detach().numpy() # (nboxes, p^2, 2), global coords
        self.nb, self.pp2 = self.gx.shape[0], self.gx.shape[1]
        self.D1 = dd.hps.H.Ds[3].detach().numpy() # leaf d/dx
        self.D2 = dd.hps.H.Ds[4].detach().numpy() # leaf d/dy

        # the leaf-grid flattening must match solve_dir_full's output ordering
        assert np.allclose(np.asarray(self.sv.XXfull), self.gx.reshape(-1, 2)), \
            "grid_xx flattening does not match XXfull ordering"

        # "own" boxes: the left half [c-H, c) of each double slab tiles the
        # channel exactly once (union over slabs = [-H, 1-H) == [0,1) mod 1)
        if own_split is not None:
            self.own = np.where(self.gx[:, :, 0].mean(axis=1) < own_split)[0]
        else:
            self.own = np.arange(self.nb)

        # per-box tensor-product quadrature weights (per-point, leaf ordering),
        # and the corner repair: leaf corners are NOT dofs in hpsalt
        # (dropped-corner HPS), so solve_dir_full fills them only approximately
        # (~1e-3 -- caught by the gate). Since Ds rows for edge points reference
        # corner columns, we overwrite each corner by barycentric interpolation
        # along its x-edge from the exact non-corner nodes after every solve.
        self.W = np.zeros((self.nb, self.pp2))
        self.cfix = []                            # per box: (corner_idx, row_idx, wts)
        self.box_meta = []                        # per box: (uxn, uyn, ix, iy)
        self._imats = {}                          # cache: (nx,ny) -> (Bx, By)
        for b in range(self.nb):
            uxn = np.unique(np.round(self.gx[b, :, 0], 12))
            uyn = np.unique(np.round(self.gx[b, :, 1], 12))
            ix = np.searchsorted(uxn, np.round(self.gx[b, :, 0], 12))
            iy = np.searchsorted(uyn, np.round(self.gx[b, :, 1], 12))
            self.W[b] = cheb_quad_weights(uxn)[ix] * cheb_quad_weights(uyn)[iy]
            self.box_meta.append((uxn, uyn, ix, iy))

            fixes = []
            nx, ny = len(uxn), len(uyn)
            for ci, cj in ((0, 0), (0, ny - 1), (nx - 1, 0), (nx - 1, ny - 1)):
                corner = np.where((ix == ci) & (iy == cj))[0]
                row = np.where((iy == cj) & (ix != 0) & (ix != nx - 1))[0]
                xs = self.gx[b, row, 0]
                bw = np.array([1.0 / np.prod(xs[j] - np.delete(xs, j))
                               for j in range(len(xs))])
                w = bw / (uxn[ci] - xs)
                fixes.append((corner[0], row, w / w.sum()))
            self.cfix.append(fixes)

    def gradx(self, F):
        return np.einsum('ij,bj->bi', self.D1, F)

    def grady(self, F):
        return np.einsum('ij,bj->bi', self.D2, F)

    def reduced_body(self, fvec):
        """b_C: statically-condensed body load on the interior skeleton dofs
        (rows of Aii), exactly as domain_driver.get_rhs forms it."""
        bC = self.dd.hps.reduce_body(CPU, None, fvec)[self.dd.I_Ctot]
        return bC.detach().cpu().numpy().real.ravel()

    def body_rhs(self, fvec, fgb):
        """Interface-system rhs of this slab (central-interface restriction):
        rhs = ( Aii^{-1} ( b_C - Aib[:,Igb] fgb ) )[Ic]."""
        w = self.sv.solver_ii @ (self.reduced_body(fvec)
                                 - self.sv.Aib[:, self.Igb] @ fgb)
        return np.asarray(w).ravel()[self.Ic]

    def reconstruct(self, ul, ur, fgb, fvec):
        """Full leaf field from solved neighbor traces + wall data + body load."""
        g = np.zeros(self.XXb.shape[0])
        g[self.Il] = ul
        g[self.Ir] = ur
        g[self.Igb] = fgb
        g = torch.from_numpy(g[:, np.newaxis])
        with redirect_stdout(io.StringIO()):   # mute per-solve residual prints
            uu = self.sv.solver.solve_dir_full(g, ff_body=fvec)
        uu = uu.detach().numpy() if torch.is_tensor(uu) else np.asarray(uu)
        uu = uu.real.reshape(self.nb, self.pp2)
        for b in range(self.nb):               # repair the non-dof leaf corners
            for corner, row, w in self.cfix[b]:
                uu[b, corner] = w @ uu[b, row]
        return uu

    def integrate_own(self, F):
        return float((self.W[self.own] * F[self.own]).sum())

    def interp_mats(self, nx, ny):
        """Cached barycentric leaf-to-uniform-subgrid matrices (all boxes are
        congruent, so one pair serves every box)."""
        if (nx, ny) not in self._imats:
            uxn, uyn, _, _ = self.box_meta[0]
            rx, ry = uxn - uxn[0], uyn - uyn[0]
            tx = (np.arange(nx) + 0.5) * rx[-1] / nx
            ty = (np.arange(ny) + 0.5) * ry[-1] / ny
            self._imats[(nx, ny)] = (bary_mat(rx, tx), bary_mat(ry, ty))
        return self._imats[(nx, ny)]


################################################################
#
#   GATE: manufactured cubic on ONE all-Dirichlet double slab
#
#   u_ex = x^3 + y^3 (collocation-exact for p_disc >= 4). Validates, in order:
#     (a) leaf derivative matrices Ds[3]/Ds[4] (orientation + physical scaling)
#     (b) sign/indexing of the skeleton-reduced body rhs
#     (c) body-load reconstruction through solve_dir_full
#
################################################################

def gate(ell2, geom, opts):
    diff_op = make_pdo(ell2, conservative=True)
    gb_all = lambda p: np.ones(p.shape[0], dtype=bool) if not torch.is_tensor(p) \
        else torch.ones(p.shape[0], dtype=torch.bool)
    ss = SlabSolve(geom, diff_op, opts, gb_all)

    D = lambda x, y: ell2 * depth_frac(x, y)
    Dx = lambda x, y: ell2 * ddepth_frac_dx(x, y)
    Dy = lambda x, y: ell2 * ddepth_frac_dy(x, y)
    u_ex = lambda P: P[..., 0] ** 3 + P[..., 1] ** 3
    # A u = -D(u_xx + u_yy) - D_x u_x - D_y u_y + u  (hpsalt signs, div form).
    # The D_y u_y term exercises the new c2 branch -- it is nonzero wherever the
    # gap band overlaps this slab, so the gate now validates c2 as well.
    f_ex = lambda x, y: (-D(x, y) * (6.0 * x + 6.0 * y)
                         - Dx(x, y) * 3.0 * x ** 2
                         - Dy(x, y) * 3.0 * y ** 2
                         + x ** 3 + y ** 3)

    xg, yg = ss.gx[:, :, 0], ss.gx[:, :, 1]
    Ue = u_ex(ss.gx)

    # (a) derivative matrices
    err_dx = np.max(np.abs(ss.gradx(Ue) - 3.0 * xg ** 2)) / np.max(3.0 * xg ** 2)
    err_dy = np.max(np.abs(ss.grady(Ue) - 3.0 * yg ** 2)) / np.max(3.0 * yg ** 2)

    fvec = torch.from_numpy(f_ex(xg, yg).reshape(-1, 1).copy())
    fgb = u_ex(ss.XXb[ss.Igb, :])

    # (b) skeleton body rhs: with all edges Dirichlet, Il = Ir = [] and the
    # interface identity reduces to u_i = Aii^{-1}(b_C - Aib fgb) on ALL of Ii
    ui = np.asarray(ss.sv.solver_ii @ (ss.reduced_body(fvec)
                                       - ss.sv.Aib[:, ss.Igb] @ fgb)).ravel()
    ue_i = u_ex(ss.XXi)
    err_skel = np.linalg.norm(ui - ue_i) / np.linalg.norm(ue_i)

    # (c) body-load reconstruction on the full leaf grids
    uu = ss.reconstruct(np.zeros(0), np.zeros(0), fgb, fvec)
    err_rec = np.linalg.norm(uu - Ue) / np.linalg.norm(Ue)

    print("=============GATE (manufactured cubic, one slab)=============")
    print("leaf d/dx matrix rel. err    = ", '%10.3E' % err_dx)
    print("leaf d/dy matrix rel. err    = ", '%10.3E' % err_dy)
    print("skeleton body-rhs rel. err   = ", '%10.3E' % err_skel)
    print("reconstruction rel. err      = ", '%10.3E' % err_rec)
    print("=============================================================")
    worst = max(err_dx, err_dy, err_skel, err_rec)
    if worst > GATE_TOL:
        raise RuntimeError("GATE FAILED: worst rel. err %.3E > %.1E -- "
                           "body-load sign/indexing or Ds scaling is wrong"
                           % (worst, GATE_TOL))


################################################################
#
#   THE TIME LOOP
#
################################################################

class ChannelModel:
    """Backward-Euler IMEX barotropic channel. Fixed dt -> the elliptic
    operator is fixed -> S assembly + cyclic Thomas factorization happen ONCE
    (in __init__); step() rebuilds only the body-load rhs and back-substitutes.
    State eta [m], u, v [m/s] live on the per-slab leaf grids (nboxes, p^2);
    overlapping slabs each carry their own consistent copy, convdiv-style."""

    def __init__(self, dt, conservative, assembler, dSlabs, connectivity, H,
                 opts, label=""):
        self.dt = dt
        self.label = label
        self.ell2 = GRAV * H0 * dt * dt / (LCHAN * LCHAN)
        self.diff_op = make_pdo(self.ell2, conservative)
        self.connectivity = connectivity
        self.N = len(dSlabs)

        zero_bc = lambda p: np.zeros(p.shape[0])

        tic = time.perf_counter()
        self.OMS = oms.oms(dSlabs, self.diff_op, gb, opts, connectivity)
        with redirect_stdout(io.StringIO()):
            S_list, rhs0, self.Ntot, self.nc = \
                self.OMS.construct_Stot_helper(zero_bc, assembler, dbg=0)
        self.t_asm = time.perf_counter() - tic

        tic = time.perf_counter()
        self.T, self.smw = omsdirectsolve.build_block_cyclic_tridiagonal_solver(
            self.OMS, S_list, rhs0, self.Ntot, self.nc)
        self.t_fac = time.perf_counter() - tic

        tic = time.perf_counter()
        self.sl = [SlabSolve(dSlabs[n], self.diff_op, opts, gb,
                             own_split=n * H) for n in range(self.N)]
        self.t_keep = time.perf_counter() - tic

        # initial condition: SSH bump at rest
        self.eta, self.u, self.v = [], [], []
        for s in self.sl:
            xg, yg = s.gx[:, :, 0], s.gx[:, :, 1]
            self.eta.append(ETA0
                            * np.exp(IC_KB * (np.cos(2.0 * np.pi * (xg - IC_CX)) - 1.0))
                            * np.exp(-IC_AY * (yg - IC_CY) ** 2))
            self.u.append(np.zeros_like(xg))
            self.v.append(np.zeros_like(xg))
        self.t = 0.0

    def wall_eta(self, pts, t):
        """Dirichlet SSH on the y-walls: 0 by default (clamped / open walls),
        or a zonal wavenumber-1 'tidal' driver if WALL_AMP is set."""
        if WALL_AMP == 0.0:
            return np.zeros(pts.shape[0])
        return (WALL_AMP * np.cos(2.0 * np.pi * pts[:, 0])
                * np.sin(2.0 * np.pi * t / WALL_PERIOD))

    def mass(self):
        return sum(s.integrate_own(self.eta[i]) for i, s in enumerate(self.sl))

    def energy(self):
        """Per-unit-density energy: int 1/2 g eta^2 + 1/2 H (u^2+v^2)."""
        tot = 0.0
        for i, s in enumerate(self.sl):
            Hp = H0 * depth_frac(s.gx[:, :, 0], s.gx[:, :, 1])
            e = (0.5 * GRAV * self.eta[i] ** 2
                 + 0.5 * Hp * (self.u[i] ** 2 + self.v[i] ** 2))
            tot += s.integrate_own(e)
        return tot

    def step(self):
        dt = self.dt
        gdtL = GRAV * dt / LCHAN
        tnew = self.t + dt

        # ---- explicit predictor + body load R^n, per slab -----------------
        tic = time.perf_counter()
        fgbs, fvecs, ustars, vstars = [], [], [], []
        rhstot = np.zeros(self.Ntot)
        for i, s in enumerate(self.sl):
            xg, yg = s.gx[:, :, 0], s.gx[:, :, 1]
            Hp = H0 * depth_frac(xg, yg)
            Hpx = H0 * ddepth_frac_dx(xg, yg)
            Hpy = H0 * ddepth_frac_dy(xg, yg)   # nonzero across the gap band

            us = self.u[i] + dt * (FCOR * self.v[i])
            vs = self.v[i] - dt * (FCOR * self.u[i])

            # div(H u*) = H (u*_x + v*_y) + H_x u* + H_y v*
            divHu = Hp * (s.gradx(us) + s.grady(vs)) + Hpx * us + Hpy * vs
            R = self.eta[i] - (dt / LCHAN) * divHu

            fgb = self.wall_eta(s.XXb[s.Igb, :], tnew)   # implicit wall data
            fvec = torch.from_numpy(R.reshape(-1, 1).copy())

            rhstot[i * self.nc:(i + 1) * self.nc] = s.body_rhs(fvec, fgb)
            fgbs.append(fgb); fvecs.append(fvec)
            ustars.append(us); vstars.append(vs)
        t_rhs = time.perf_counter() - tic

        # ---- one block back-substitution (factorization is reused) --------
        tic = time.perf_counter()
        uhat = omsdirectsolve.block_cyclic_tridiagonal_solve(
            self.OMS, self.T, self.smw, rhstot)
        t_slv = time.perf_counter() - tic

        # ---- reconstruction + velocity update -----------------------------
        tic = time.perf_counter()
        for i, s in enumerate(self.sl):
            kl, kr = self.connectivity[i]
            ul = uhat[kl * self.nc:(kl + 1) * self.nc]
            ur = uhat[kr * self.nc:(kr + 1) * self.nc]
            eta_new = s.reconstruct(ul, ur, fgbs[i], fvecs[i])
            self.u[i] = ustars[i] - gdtL * s.gradx(eta_new)
            self.v[i] = vstars[i] - gdtL * s.grady(eta_new)
            self.eta[i] = eta_new
        t_rec = time.perf_counter() - tic

        self.t = tnew
        maxeta = max(np.abs(e).max() for e in self.eta)
        return {"mass": self.mass(), "energy": self.energy(),
                "maxeta": maxeta, "t_rhs": t_rhs, "t_slv": t_slv,
                "t_rec": t_rec}

    def snapshot(self, nx=8, ny=8):
        """Global uniform image of eta: per-leaf barycentric resampling of the
        tiling ('own') boxes (IMEXconvdiv's plot_field pattern; smooth fields,
        no scatter banding from the Chebyshev point clustering)."""
        uxn0, uyn0, _, _ = self.sl[0].box_meta[0]
        bx, by = uxn0[-1] - uxn0[0], uyn0[-1] - uyn0[0]
        ncol, nrow = int(round(1.0 / bx)), int(round(1.0 / by))
        img = np.full((ncol * nx, nrow * ny), np.nan)
        for i, s in enumerate(self.sl):
            Bx, By = s.interp_mats(nx, ny)
            for b in s.own:
                uxn, uyn, ix, iy = s.box_meta[b]
                U2 = np.zeros((len(uxn), len(uyn)))
                U2[ix, iy] = self.eta[i][b]
                c = int(round(np.mod(uxn[0], 1.0) / bx)) % ncol
                r = int(round(uyn[0] / by))
                img[c * nx:(c + 1) * nx, r * ny:(r + 1) * ny] = Bx @ U2 @ By.T
        xc = (np.arange(ncol * nx) + 0.5) / (ncol * nx)
        yc = (np.arange(nrow * ny) + 0.5) / (nrow * ny)
        return xc, yc, img


################################################################
#
#   MAIN
#
################################################################

N        = int(os.environ.get("SSLABLU_N", "8"))
p        = int(os.environ.get("SSLABLU_P", "12"))
npan_x   = int(os.environ.get("SSLABLU_NPAN_X", "4"))   # keep EVEN
npan_y   = int(os.environ.get("SSLABLU_NPAN_Y", "8"))
dt_hours = float(os.environ.get("SSLABLU_DT_H", "0.25"))
NSTEPS   = int(os.environ.get("SSLABLU_NSTEPS", "48"))
RK       = int(os.environ.get("SSLABLU_RK", "0"))       # 0 = dense S-maps
CMP_FORM = os.environ.get("SSLABLU_COMPARE_FORMS", "1") != "0"
DO_DTCNV = os.environ.get("SSLABLU_DTCONV", "0") != "0"
WALL_AMP = float(os.environ.get("SSLABLU_WALL_AMP", "0.0"))
WALL_PERIOD = 12.42 * 3600.0                             # M2-ish tide [s]

p_disc    = p + 2
leaf_size = 2 * p
dSlabs, connectivity, H = channel_dSlabs(N)
a = np.array([H / npan_x, 0.5 / npan_y])
opts = solverWrap.solverOptions("hpsalt", [p_disc, p_disc], a)

dt   = 3600.0 * dt_hours
ell  = np.sqrt(GRAV * H0) * dt / LCHAN
ell2 = ell * ell


def make_assembler():
    if RK > 0:
        return mA.rkHMatAssembler(leaf_size, RK)
    return mA.denseMatAssembler()


print("=============CHANNEL TIMESTEP SETUP=============")
print("N slabs / interfaces     = ", N)
print("p_disc                   = ", p_disc)
print("panels (x per slab, y)   = ", npan_x, ",", npan_y)
print("dt                       = ", '%6.3f h' % dt_hours,
      " ell/L = %.3f  ell^2 = %.4f" % (ell, ell2))
print("steps / total time       = ", NSTEPS, "/ %.2f h" % (NSTEPS * dt_hours))
print("f*dt (explicit Coriolis) = ", '%6.3f' % (FCOR * dt))
print("S-map assembler          = ",
      ("HBS rk = %d" % RK) if RK > 0 else "dense")
print("wall forcing amplitude   = ", WALL_AMP, "m")
print("================================================")

# ---- GATE first: nothing runs unless signs and scalings check out ----------
gate(ell2, dSlabs[N // 2], opts)

# ---- build model(s): one factorization each, reused for every step ---------
tic = time.perf_counter()
modC = ChannelModel(dt, True, make_assembler(), dSlabs, connectivity, H, opts,
                    label="divergence form")
print("[divergence form]     assemble/factor/keep-slabs = "
      "%.2f / %.2f / %.2f s" % (modC.t_asm, modC.t_fac, modC.t_keep))
modN = None
if CMP_FORM:
    modN = ChannelModel(dt, False, make_assembler(), dSlabs, connectivity, H,
                        opts, label="non-conservative")
    print("[non-conservative]    assemble/factor/keep-slabs = "
          "%.2f / %.2f / %.2f s" % (modN.t_asm, modN.t_fac, modN.t_keep))
t_setup = time.perf_counter() - tic

M0C = modC.mass()
E0 = modC.energy()
M0N = modN.mass() if modN is not None else np.nan

snaps = {0: modC.snapshot()}
rows = []
hist = {"t": [0.0], "dMC": [0.0], "dMN": [0.0], "E": [E0],
        "maxeta": [ETA0], "t_slv": [], "t_rhs": [], "t_rec": []}

print("")
print(" step   t[h]    |M-M0| div-form   |M-M0| non-cons    E/E0     max|eta|"
      "   t_rhs   t_slv   t_rec")
for n in range(1, NSTEPS + 1):
    dC = modC.step()
    dN = modN.step() if modN is not None else None

    dMC = abs(dC["mass"] - M0C)
    dMN = abs(dN["mass"] - M0N) if dN is not None else np.nan
    print(" %4d  %6.2f     %10.3E       %10.3E     %7.4f   %8.4f"
          "   %5.2f   %5.3f   %5.2f"
          % (n, n * dt_hours, dMC, dMN, dC["energy"] / E0, dC["maxeta"],
             dC["t_rhs"], dC["t_slv"], dC["t_rec"]))

    rows.append([n, n * dt_hours, dC["mass"], dMC,
                 (dN["mass"] if dN else np.nan), dMN,
                 dC["energy"], dC["maxeta"],
                 dC["t_rhs"], dC["t_slv"], dC["t_rec"]])
    hist["t"].append(n * dt_hours)
    hist["dMC"].append(dMC); hist["dMN"].append(dMN)
    hist["E"].append(dC["energy"]); hist["maxeta"].append(dC["maxeta"])
    hist["t_rhs"].append(dC["t_rhs"]); hist["t_slv"].append(dC["t_slv"])
    hist["t_rec"].append(dC["t_rec"])

    if n in (NSTEPS // 2, NSTEPS):
        snaps[n] = modC.snapshot()

maxeta_run = max(hist["maxeta"])
stable = maxeta_run < 50.0 * ETA0

print("")
print("=============SUMMARY (%d steps, dt = %.2f h)=============" % (NSTEPS, dt_hours))
print("setup (both forms)       = ", '%8.2f s' % t_setup)
print("avg rhs / solve / recon  =  %6.3f / %6.4f / %6.3f s per step"
      % (np.mean(hist["t_rhs"]), np.mean(hist["t_slv"]), np.mean(hist["t_rec"])))
print("max|eta| over run        = ", '%8.4f m' % maxeta_run,
      " ->", "stable" if stable else "CHECK STABILITY")
print("final |M-M0| divergence  = ", '%10.3E' % hist["dMC"][-1])
if modN is not None:
    print("final |M-M0| non-cons    = ", '%10.3E' % hist["dMN"][-1])
    print("  (walls are clamped-SSH, not solid: both forms share the physical")
    print("   wall flux; the non-conservative excess is the spurious part)")
print("final E/E0               = ", '%8.4f' % (hist["E"][-1] / E0),
      " (backward Euler damps the radiated gravity waves)")
print("=========================================================")

# ---- CSV export -------------------------------------------------------------
rows = np.array(rows)
csv_name = "channel_timestep_diag.csv"
with open(csv_name, 'w') as f:
    f.write("step,t_hours,mass_div,dmass_div,mass_ncons,dmass_ncons,"
            "energy,max_eta,t_rhs,t_solve,t_recon\n")
    np.savetxt(f, rows, fmt='%.16e', delimiter=',')
print("Wrote %s  (%d rows)" % (csv_name, rows.shape[0]))

# ---- optional dt-convergence (rebuilds the operator per dt: slow) ----------
if DO_DTCNV:
    print("")
    print("=============DT-CONVERGENCE (divergence form)=============")
    nbase = max(4, NSTEPS // 8)
    Tfin = nbase * dt

    def run_final(dt_, nsteps_):
        m = ChannelModel(dt_, True, make_assembler(), dSlabs, connectivity, H,
                         opts)
        for _ in range(nsteps_):
            m.step()
        return np.concatenate([m.eta[i][m.sl[i].own].ravel()
                               for i in range(m.N)])

    eref = run_final(dt / 8.0, nbase * 8)
    print(" reference: dt/8, %d steps, T = %.3f h" % (nbase * 8, Tfin / 3600.0))
    print("     dt        N     rel-l2 vs ref     ratio   (O(dt) -> ~2.0)")
    prev = None
    for k in (1, 2, 4):
        e = run_final(dt / k, nbase * k)
        err = np.linalg.norm(e - eref) / np.linalg.norm(eref)
        r = ("%5.2f" % (prev / err)) if prev else "  -  "
        print("  %8.1f s  %4d     %.4e        %s" % (dt / k, nbase * k, err, r))
        prev = err

# ---- plots (Agg, PNGs; guarded so plotting never kills the results) --------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # snapshots
    keys = sorted(snaps.keys())
    figF, axF = plt.subplots(1, len(keys), figsize=(5.0 * len(keys), 4.4),
                             sharex=True, sharey=True, squeeze=False)
    axF = axF[0]
    for k, nstep in enumerate(keys):
        xc, yc, img = snaps[nstep]
        # per-panel symmetric scale: the late-time geostrophic residual is
        # orders of magnitude below the initial bump
        vmax = max(np.nanmax(np.abs(img)), 1e-12)
        pc = axF[k].pcolormesh(xc, yc, img.T, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax, shading='auto')
        figF.colorbar(pc, ax=axF[k], shrink=0.85)
        axF[k].set_title(r'$\eta$ at t = %.2f h  (max %.3g m)'
                         % (nstep * dt_hours, vmax))
        axF[k].set_xlabel('x / L')
        axF[k].set_xlim(0, 1); axF[k].set_ylim(0, 1)
        axF[k].set_aspect('equal')
    axF[0].set_ylabel('y / L')
    figF.suptitle('geostrophic adjustment over the ridge: one factorization, '
                  '%d back-substitutions' % NSTEPS, fontsize=11)
    figF.tight_layout(rect=[0, 0, 1, 0.93])
    figF.savefig('channel_timestep_fields.png', dpi=200)

    # diagnostics
    figD, axD = plt.subplots(2, 2, figsize=(11, 8))
    tt = np.array(hist["t"])
    axD[0, 0].semilogy(tt, np.maximum(hist["dMC"], 1e-18), 'o-',
                       label='divergence form')
    if modN is not None:
        axD[0, 0].semilogy(tt, np.maximum(hist["dMN"], 1e-18), 's--',
                           label='non-conservative')
    axD[0, 0].set_xlabel('t [h]'); axD[0, 0].set_ylabel(r'|M(t) - M$_0$|')
    axD[0, 0].set_title('mass drift (shared wall flux + spurious part)')
    axD[0, 0].grid(True, which='both', alpha=0.3); axD[0, 0].legend(fontsize=8)

    axD[0, 1].plot(tt, np.array(hist["E"]) / E0, 'o-')
    axD[0, 1].set_xlabel('t [h]'); axD[0, 1].set_ylabel(r'E(t) / E$_0$')
    axD[0, 1].set_title('energy (backward-Euler wave damping)')
    axD[0, 1].grid(True, alpha=0.3)

    axD[1, 0].plot(tt, hist["maxeta"], 'o-')
    axD[1, 0].set_xlabel('t [h]'); axD[1, 0].set_ylabel(r'max |$\eta$| [m]')
    axD[1, 0].set_title('stability check')
    axD[1, 0].grid(True, alpha=0.3)

    steps_ax = np.arange(1, NSTEPS + 1)
    axD[1, 1].plot(steps_ax, hist["t_rhs"], 'o-', label='body-load rhs')
    axD[1, 1].plot(steps_ax, hist["t_slv"], 's-', label='cyclic back-subst.')
    axD[1, 1].plot(steps_ax, hist["t_rec"], '^-', label='reconstruction')
    axD[1, 1].set_xlabel('step'); axD[1, 1].set_ylabel('time [s]')
    axD[1, 1].set_title('per-step cost (factorization amortized: %.2f s once)'
                        % modC.t_fac)
    axD[1, 1].grid(True, alpha=0.3); axD[1, 1].legend(fontsize=8)

    figD.suptitle('Channel barotropic timestepping: backward-Euler IMEX, '
                  'reused cyclic-Thomas factorization', fontsize=12)
    figD.tight_layout(rect=[0, 0, 1, 0.96])
    figD.savefig('channel_timestep_diagnostics.png', dpi=200)

    print("wrote channel_timestep_fields.png, channel_timestep_diagnostics.png")
except Exception as e:
    print("plotting skipped:", e)
