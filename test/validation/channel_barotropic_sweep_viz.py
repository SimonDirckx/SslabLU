# =============================================================================
# channel_barotropic_sweep.py
#
# Barotropic-mode (implicit free surface) solve for a 2D re-entrant channel,
# in SslabLU. The per-timestep elliptic problem is the variable-coefficient
# screened diffusion equation
#
#     -div( D(x,y) grad(eta) ) + eta = 0        on the channel,
#     eta periodic in x,  Dirichlet data on the y-walls,
#
# where D = g*H(x,y)*dt^2 / L^2 (nondimensionalized by the channel width L).
# The screening length is ell = sqrt(g*H)*dt / L: the distance an external
# gravity wave travels in one timestep. H(x,y) carries a periodic Gaussian-
# like ridge across the channel.
#
# Under the hpsalt convention  A = -c11 u_xx - c22 u_yy + c1 u_x + c2 u_y + c u,
# the divergence form expands as
#     c11 = c22 = D(x,y),   c1 = -dD/dx,   c2 = 0 (ridge is y-independent),
#     c   = +1   (coercive screening term; note the SIGN: the crystal tests
#                 use c = -kh^2*(...) which is the oscillatory Helmholtz case).
#
# Seam treatment: "flat with modular slabs". The domain is the flat unit
# square, slabs are cut in x with cyclic connectivity [(n-1)%N, (n+1)%N]
# (same as geometry/geom_3D/squareTorus.dSlabs), and the slab centered on the
# seam interface x = 0 == 1 simply extends to the fictitious range [-H, +H].
# This works because every coefficient / BC callable below is built from
# 1-periodic functions of x (Fourier modes and a von-Mises bump), so their
# smooth periodic extension to x < 0 and x > 1 is automatic -- no mod needed.
#
# Tests:
#   TEST 1 (per dt): reference solve. Dense assembler, cyclic block-Thomas
#           direct solve (with the Sherman-Morrison-Woodbury corner
#           correction), GMRES on the same operator as a conditioning probe,
#           and an overlap/seam consistency check on the reconstructed field.
#   TEST 2 (per dt): HBS rank sweep. rkHMatAssembler at increasing rk,
#           cyclic direct solve, error vs the dense reference, true residual,
#           timings, compression. Early stop once err < EARLY_STOP_TOL.
#
# Plots (Agg backend, written as PNGs; flip SHOW_PLOTS for windows):
#   channel_setup.png       bathymetry + slab decomposition + seam, and the
#                           D(x) profiles for every dt in the sweep
#   channel_fields.png      reconstructed SSH field eta(x,y) at up to three
#                           dt values (screening-length walk, wall-driven)
#   channel_diagnostics.png 2x2 rank-sweep diagnostics, one curve per dt:
#                           err vs rk, residual vs rk, cost vs rk,
#                           compression vs rk  (laplace_rank_sweep.py style)
#
# Placement: put this file in  test/2D/  of the SslabLU repo and run from the
# repo root (imports follow laplace_rank_sweep.py / periodicTest.py).
#
# Environment overrides (crystal-test style):
#   SSLABLU_N         number of slabs / interfaces        (default 8)
#   SSLABLU_P         polynomial order p (p_disc = p+2)   (default 12)
#   SSLABLU_NPAN_X    x-panels per double slab, EVEN      (default 4)
#   SSLABLU_NPAN_Y    y-panels across the channel         (default 8)
#   SSLABLU_CHECK_OVERLAP  1 = run seam/overlap check     (default 1)
# =============================================================================

import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from packaging.version import Version
from scipy.sparse.linalg import gmres

# --- oms packages (run from repo root; fallback mirrors crystalWaveGuideTest) -
try:
    import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
except ImportError:
    _HPSMULTIDOMAIN_ROOT = Path(__file__).resolve().parents[2] / "solver" / "hpsmultidomain"
    if str(_HPSMULTIDOMAIN_ROOT) not in sys.path:
        sys.path.insert(0, str(_HPSMULTIDOMAIN_ROOT))
    import hpsmultidomain.pdo as pdo

import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import multislab.omsdirectsolve as omsdirectsolve


class gmres_info(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
        self.resList = []

    def __call__(self, rk=None):
        self.niter += 1
        self.resList += [rk]
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


################################################################
#
#   SET-UP BVP:   screened diffusion on the re-entrant channel
#   - physical constants -> ell^2 = (sqrt(g*H0)*dt/L)^2 per dt
#   - bathymetry        (1-periodic von-Mises ridge)
#   - pdo               (divergence form expanded, hpsalt signs)
#   - BC                (prescribed SSH on the y-walls)
#
################################################################

GRAV     = 9.81          # m/s^2
H0       = 4000.0        # reference depth [m]
LCHAN    = 1.0e6         # channel width Ly [m]; domain nondimensionalized by this
RIDGE_HR = 0.8           # ridge height as a fraction of H0
RIDGE_KB = 40.0          # von-Mises concentration: larger = narrower ridge

# Outer sweep: barotropic timestep in hours. ell2 = (sqrt(g*H0)*dt/L)^2 walks
# the operator from nearly-decoupled interfaces (small dt) toward the Poisson
# limit (large dt).
DT_HOURS = [0.25, 0.5, 1.0, 2.0, 4.0]

# Inner sweep: HBS rank for the compressed S-maps.
RANKS          = [5, 10, 15, 20, 30, 40, 60]
EARLY_STOP_TOL = 1.0e-10

SHOW_PLOTS = False


def bump_x(x, lib):
    """1-periodic Gaussian-like ridge profile, centered at x = 0.5.

    b(x) = exp( kb * (cos(2 pi (x - 1/2)) - 1) ),  b(0.5) = 1, b(0) ~ exp(-2 kb).
    Smooth and exactly 1-periodic, so the fictitious extension of the seam slab
    to x in [-H, H] is automatically consistent.
    """
    return lib.exp(RIDGE_KB * (lib.cos(2.0 * np.pi * (x - 0.5)) - 1.0))


def dbump_x(x, lib):
    """d/dx of bump_x (analytic)."""
    return (-2.0 * np.pi * RIDGE_KB * lib.sin(2.0 * np.pi * (x - 0.5))
            * bump_x(x, lib))


def depth_frac(x, lib=np):
    """H(x)/H0 = 1 - hr*bump(x): the (nondimensional) bathymetry."""
    return 1.0 - RIDGE_HR * bump_x(x, lib)


def make_pdo(ell2):
    """Divergence-form screened diffusion PDO for ell^2 = (sqrt(g*H0)*dt/L)^2:

        -div(D grad eta) + eta,   D(x) = ell2 * depth_frac(x).
    """
    def Dcoef(p):
        lib = torch if torch.is_tensor(p) else np
        return ell2 * depth_frac(p[:, 0], lib)

    def c1(p):   # c1 = -dD/dx
        lib = torch if torch.is_tensor(p) else np
        return ell2 * RIDGE_HR * dbump_x(p[:, 0], lib)

    return pdo.PDO_2d(c11=Dcoef, c22=Dcoef, c1=c1, c=pdo.const(c=1.0))


def bc(p):
    """Prescribed SSH on the y-walls (the only true boundary): a zonal
    wavenumber-1 signal, stronger on the south wall. 1-periodic in x, so it
    extends smoothly across the seam slab's fictitious range.
    """
    lib = torch if torch.is_tensor(p) else np
    return lib.cos(2.0 * np.pi * p[:, 0]) * (1.0 - 1.5 * p[:, 1])


################################################################
#
#   GEOMETRY:  flat unit square, x-periodic via cyclic slabs
#   (2D analog of geometry/geom_3D/squareTorus.dSlabs)
#
################################################################

BNDS = [[0.0, 0.0], [1.0, 1.0]]


def channel_dSlabs(N):
    """N double-wide slabs; the i-th dSlab has the i-th interface (x = i*H)
    as its central interface. Slab 0 straddles the seam: [-H, +H]."""
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
#   SOLVE HELPERS
#
################################################################

def assemble_S(diff_op, dSlabs, connectivity, opts, assembler, dbg=0):
    OMS = oms.oms(dSlabs, diff_op, gb, opts, connectivity)
    tic = time.perf_counter()
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=dbg)
    t_asm = time.perf_counter() - tic
    Stot, rhstot = OMS.construct_Stot_and_rhstot_linearOperator(
        S_rk_list, rhs_list, Ntot, nc, dbg=0)
    return OMS, S_rk_list, rhs_list, Stot, rhstot, Ntot, nc, t_asm


def cyclic_direct_solve(OMS, S_rk_list, rhs_list, rhstot, Ntot, nc):
    tic = time.perf_counter()
    T, smw_block = omsdirectsolve.build_block_cyclic_tridiagonal_solver(
        OMS, S_rk_list, rhs_list, Ntot, nc)
    t_fac = time.perf_counter() - tic
    tic = time.perf_counter()
    uhat = omsdirectsolve.block_cyclic_tridiagonal_solve(OMS, T, smw_block, rhstot)
    t_slv = time.perf_counter() - tic
    return uhat, t_fac, t_slv


def gmres_probe(Stot, rhstot, tol):
    gInfo = gmres_info()
    if Version(scipy.__version__) >= Version("1.14"):
        u, info = gmres(Stot, rhstot, rtol=tol, callback=gInfo,
                        maxiter=300, restart=300)
    else:
        u, info = gmres(Stot, rhstot, tol=tol, callback=gInfo,
                        maxiter=300, restart=300)
    return u, info, gInfo.niter


def reconstruct_slab_fields(uhat, dSlabs, connectivity, opts, diff_op, nc, N):
    """Per-slab full-field reconstruction (periodicTest.py pattern): impose the
    solved neighbor-interface traces + wall data on each double slab and do one
    local solve. Returns [(XX, uu)] with x mapped back into [0,1) mod 1."""
    fields = []
    for slabInd in range(N):
        geom_i = np.array(dSlabs[slabInd])
        slab_i = oms.slab(geom_i, gb)
        solver = oms.solverWrap.solverWrapper(opts)
        solver.construct(geom_i, diff_op)
        Il, Ir, Ic, Igb, XXi, XXb = slab_i.compute_idxs_and_pts(solver)

        startL = connectivity[slabInd][0]
        startR = connectivity[slabInd][1]
        ul = uhat[startL * nc:(startL + 1) * nc]
        ur = uhat[startR * nc:(startR + 1) * nc]

        g = np.zeros(shape=(XXb.shape[0],))
        g[Il] = ul
        g[Ir] = ur
        g[Igb] = np.asarray(bc(XXb[Igb, :]))
        g = torch.from_numpy(g[:, np.newaxis])

        uu = solver.solver.solve_dir_full(g)
        uu = uu.detach().numpy().flatten() if torch.is_tensor(uu) \
            else np.asarray(uu).flatten()

        XX = np.array(solver.XXfull).copy()
        XX[:, 0] = np.mod(XX[:, 0], 1.0)     # fold the seam slab back into [0,1)
        fields.append((XX, uu))
    return fields


def overlap_mismatch(fields, N):
    """Max |u_i - u_j| over grid points shared by adjacent (overlapping) slabs,
    including the seam pair (N-1, 0). With NPAN_X even, adjacent slabs' panel
    grids coincide on the overlap, so shared points match exactly (up to
    rounding). This is the direct test that the modular-seam bookkeeping and
    the SMW corner blocks are globally consistent."""
    keyed = []
    for XX, uu in fields:
        d = {}
        for k in range(XX.shape[0]):
            key = (round(float(XX[k, 0]), 10), round(float(XX[k, 1]), 10))
            d[key] = uu[k]
        keyed.append(d)

    worst = 0.0
    for i in range(N):
        j = (i + 1) % N
        shared = set(keyed[i].keys()) & set(keyed[j].keys())
        if len(shared) == 0:
            print("WARNING: no shared points between slabs %d,%d "
                  "(is NPAN_X even?)" % (i, j))
            continue
        diff = max(abs(keyed[i][k] - keyed[j][k]) for k in shared)
        worst = max(worst, diff)
    return worst


################################################################
#
#   MAIN SWEEP
#
################################################################

N       = int(os.environ.get("SSLABLU_N", "8"))
p       = int(os.environ.get("SSLABLU_P", "12"))
npan_x  = int(os.environ.get("SSLABLU_NPAN_X", "4"))   # keep EVEN (overlap check)
npan_y  = int(os.environ.get("SSLABLU_NPAN_Y", "8"))
do_ovlp = os.environ.get("SSLABLU_CHECK_OVERLAP", "1") != "0"

p_disc    = p + 2            # hpsalt convention offset (cf. twistedTorus.py)
leaf_size = 2 * p            # HBS leaf size on the (1D line) interface
dSlabs, connectivity, H = channel_dSlabs(N)
a = np.array([H / npan_x, 0.5 / npan_y])   # per-dim panel half-widths:
                                           # x: 2H/(2*H/npan_x) = npan_x panels
                                           # y: 1/(2*0.5/npan_y) = npan_y panels
opts = solverWrap.solverOptions("hpsalt", [p_disc, p_disc], a)

print("=============CHANNEL SETUP=============")
print("N slabs / interfaces     = ", N)
print("H (slab half-pitch)      = ", '%10.3E' % H)
print("p_disc                   = ", p_disc)
print("panels (x per slab, y)   = ", npan_x, ",", npan_y)
print("HBS leaf size            = ", leaf_size)
print("ridge: hr, kb            = ", RIDGE_HR, ",", RIDGE_KB)
print("=======================================")

rows = []            # dt_hours, ell2, rk (-1 = dense ref), err_vs_ref, rel_res,
                     # t_asm, t_fac, t_slv, gmres_iters, overlap_err
sweep_by_dt = {}     # dt_hours -> dict of per-rank arrays (for the plots)
fields_by_dt = {}    # dt_hours -> (XYall, uall) reconstructed reference field
ncval = 0

for dt_hours in DT_HOURS:
    dt   = 3600.0 * dt_hours
    ell  = np.sqrt(GRAV * H0) * dt / LCHAN
    ell2 = ell * ell
    diff_op = make_pdo(ell2)

    print("")
    print("########################################################")
    print("##  dt = %.2f h   ell/L = %.3f   ell^2 = %.4f" % (dt_hours, ell, ell2))
    print("########################################################")

    # ------------------------------------------------------------------
    # TEST 1: dense reference + conditioning probe + seam check
    # ------------------------------------------------------------------
    assembler_ref = mA.denseMatAssembler()
    (OMS, S_ref, rhs_ref, Stot_ref, rhstot_ref,
     Ntot, nc, t_asm_ref) = assemble_S(diff_op, dSlabs, connectivity,
                                       opts, assembler_ref, dbg=1)
    ncval = nc

    uhat_ref, t_fac_ref, t_slv_ref = cyclic_direct_solve(
        OMS, S_ref, rhs_ref, rhstot_ref, Ntot, nc)

    res_ref = Stot_ref @ uhat_ref - rhstot_ref
    relres_ref = np.linalg.norm(res_ref) / np.linalg.norm(rhstot_ref)

    stol = 1e-10 * H * H
    _, info, niter = gmres_probe(Stot_ref, rhstot_ref, stol)

    ovlp = np.nan
    if do_ovlp:
        fields = reconstruct_slab_fields(uhat_ref, dSlabs, connectivity,
                                         opts, diff_op, nc, N)
        ovlp = overlap_mismatch(fields, N)
        XYall = np.concatenate([F[0] for F in fields], axis=0)
        uall  = np.concatenate([F[1] for F in fields], axis=0)
        fields_by_dt[dt_hours] = (XYall, uall)

    print("=============SUMMARY (dense reference)=============")
    print("Ntot, nc                 = ", Ntot, ",", nc)
    print("L2 rel. res              = ", '%10.3E' % relres_ref)
    print("GMRES iters (probe)      = ", niter, " (info =", info, ")")
    print("overlap/seam mismatch    = ", '%10.3E' % ovlp)
    print("t assemble/factor/solve  =  %8.3f / %8.3f / %8.3f s"
          % (t_asm_ref, t_fac_ref, t_slv_ref))
    print("===================================================")

    rows.append([dt_hours, ell2, -1, 0.0, relres_ref,
                 t_asm_ref, t_fac_ref, t_slv_ref, niter, ovlp])

    # ------------------------------------------------------------------
    # TEST 2: HBS rank sweep against the dense reference
    # ------------------------------------------------------------------
    sw = {"rk": [], "err": [], "res": [], "t": [], "compr": []}
    print("")
    print("  rk   err_vs_ref     rel_res      t_asm     t_fac     t_slv   compr")
    for rk in RANKS:
        assembler_rk = mA.rkHMatAssembler(leaf_size, rk)
        (OMSr, S_rk, rhs_rk, Stot_rk, rhstot_rk,
         _, _, t_asm) = assemble_S(diff_op, dSlabs, connectivity,
                                   opts, assembler_rk, dbg=0)

        uhat_rk, t_fac, t_slv = cyclic_direct_solve(
            OMSr, S_rk, rhs_rk, rhstot_rk, Ntot, nc)

        err = (np.linalg.norm(uhat_rk - uhat_ref)
               / np.linalg.norm(uhat_ref))
        # true residual: compressed solution against the DENSE operator
        res = Stot_ref @ uhat_rk - rhstot_ref
        relres = np.linalg.norm(res) / np.linalg.norm(rhstot_ref)
        compr = getattr(OMSr.stats, "compression", None)
        compr = np.nan if compr is None else compr

        print("%4d   %10.3E   %10.3E   %8.3f  %8.3f  %8.3f   %s"
              % (rk, err, relres, t_asm, t_fac, t_slv,
                 ("%6.3f" % compr) if np.isfinite(compr) else "   n/a"))

        rows.append([dt_hours, ell2, rk, err, relres,
                     t_asm, t_fac, t_slv, -1, np.nan])
        sw["rk"].append(rk);   sw["err"].append(err)
        sw["res"].append(relres)
        sw["t"].append(t_fac + t_slv)
        sw["compr"].append(compr)

        if err < EARLY_STOP_TOL:
            print("  -> early stop: err < %.1E at rk = %d" % (EARLY_STOP_TOL, rk))
            break
    sweep_by_dt[dt_hours] = {k: np.array(v) for k, v in sw.items()}

################################################################
#
#   CSV EXPORT
#
################################################################
rows = np.array(rows)
csv_name = "channel_barotropic_sweep.csv"
with open(csv_name, 'w') as f:
    f.write("dt_hours,ell2,rk,err_vs_ref,rel_res,"
            "t_assemble,t_factor,t_solve,gmres_iters,overlap_err\n")
    np.savetxt(f, rows, fmt='%.16e', delimiter=',')
print("")
print("Wrote %s  (%d rows)" % (csv_name, rows.shape[0]))


################################################################
#
#   PLOTS  (laplace_rank_sweep.py style: Agg backend, PNGs,
#           guarded by try/except so a headless/odd matplotlib
#           never kills the sweep results above)
#
################################################################
try:
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ---- Figure 1: the PDE / channel setup ------------------------------
    # (a) bathymetry map with the cyclic slab decomposition overlaid
    # (b) diffusion coefficient D(x) = ell^2 * H(x)/H0 per dt (semilogy),
    #     against the slab pitch H: D ~ H^2 marks the decoupled regime.
    figS, axS = plt.subplots(1, 2, figsize=(12.5, 4.4),
                             gridspec_kw={'width_ratios': [1.4, 1.0]})

    xg = np.linspace(0.0, 1.0, 600)
    yg = np.linspace(0.0, 1.0, 160)
    Hfield = np.tile(H0 * depth_frac(xg)[np.newaxis, :], (len(yg), 1))
    pc = axS[0].pcolormesh(xg, yg, Hfield, cmap='viridis', shading='auto')
    cb = figS.colorbar(pc, ax=axS[0], shrink=0.9)
    cb.set_label('depth H(x) [m]')
    for n in range(N):
        xi = n * H
        lc = 'r' if n == 0 else 'w'
        lw = 1.6 if n == 0 else 0.9
        axS[0].axvline(xi, color=lc, ls='--', lw=lw, alpha=0.85)
    axS[0].plot([], [], 'w--', lw=0.9, label='interfaces (x = nH)')
    axS[0].plot([], [], 'r--', lw=1.6, label='seam x = 0 == 1')
    axS[0].set_xlabel('x / L (periodic)'); axS[0].set_ylabel('y / L')
    axS[0].set_title('re-entrant channel: ridge bathymetry + %d cyclic slabs' % N)
    axS[0].legend(fontsize=8, loc='lower left', framealpha=0.7)

    for dt_hours in DT_HOURS:
        ell2_ = (np.sqrt(GRAV * H0) * 3600.0 * dt_hours / LCHAN) ** 2
        axS[1].semilogy(xg, ell2_ * depth_frac(xg),
                        label='dt = %.2f h' % dt_hours)
    axS[1].axhline(H * H, color='k', ls=':', lw=1.2,
                   label='slab pitch$^2$  H$^2$')
    axS[1].set_xlabel('x / L'); axS[1].set_ylabel(r'D(x) = $\ell^2\,$H(x)/H$_0$')
    axS[1].set_title('screened-diffusion coefficient per dt\n'
                     r'(-$\nabla\cdot$(D$\nabla\eta$) + $\eta$ = 0)')
    axS[1].grid(True, which='both', alpha=0.3); axS[1].legend(fontsize=8)

    figS.tight_layout()
    figS.savefig('channel_setup.png', dpi=200)

    # ---- Figure 2: reconstructed SSH fields (screening-length walk) ------
    if len(fields_by_dt) > 0:
        keys = sorted(fields_by_dt.keys())
        pick = [keys[0]]
        if len(keys) >= 3:
            pick = [keys[0], keys[len(keys) // 2], keys[-1]]
        elif len(keys) == 2:
            pick = keys
        figF, axF = plt.subplots(1, len(pick), figsize=(5.0 * len(pick), 4.4),
                                 sharex=True, sharey=True, squeeze=False)
        axF = axF[0]
        for k, dt_hours in enumerate(pick):
            XY, uu = fields_by_dt[dt_hours]
            vmax = np.max(np.abs(uu))
            ell_ = np.sqrt(GRAV * H0) * 3600.0 * dt_hours / LCHAN
            sc = axF[k].scatter(XY[:, 0], XY[:, 1], c=uu, s=3,
                                cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            figF.colorbar(sc, ax=axF[k], shrink=0.85)
            axF[k].set_title(r'$\eta$:  dt = %.2f h,  $\ell$/L = %.2f'
                             % (dt_hours, ell_))
            axF[k].set_xlabel('x / L')
            axF[k].set_xlim(0, 1); axF[k].set_ylim(0, 1)
            axF[k].set_aspect('equal')
        axF[0].set_ylabel('y / L')
        figF.suptitle('wall-driven SSH response: penetration depth follows '
                      r'the screening length $\ell=\sqrt{gH}\,$dt/L', fontsize=11)
        figF.tight_layout(rect=[0, 0, 1, 0.93])
        figF.savefig('channel_fields.png', dpi=200)

    # ---- Figure 3: 2x2 rank-sweep diagnostics, one curve per dt ----------
    figD, axD = plt.subplots(2, 2, figsize=(11, 8))
    for dt_hours in DT_HOURS:
        if dt_hours not in sweep_by_dt:
            continue
        sw = sweep_by_dt[dt_hours]
        lab = 'dt = %.2f h' % dt_hours
        axD[0, 0].semilogy(sw["rk"], sw["err"], 'o-', label=lab)
        axD[0, 1].semilogy(sw["rk"], sw["res"], 's--', label=lab)
        axD[1, 0].plot(sw["rk"], sw["t"], 'o-', label=lab)
        if np.any(np.isfinite(sw["compr"])):
            axD[1, 1].plot(sw["rk"], sw["compr"], 's-', label=lab)

    axD[0, 0].axhline(EARLY_STOP_TOL, color='k', ls=':', lw=1,
                      label='target %.0E' % EARLY_STOP_TOL)
    axD[0, 0].set_xlabel('HBS rank  rk')
    axD[0, 0].set_ylabel('rel. err vs dense reference')
    axD[0, 0].set_title('rank-sweep convergence (smaller dt -> lower rank)')
    axD[0, 0].grid(True, which='both', alpha=0.3); axD[0, 0].legend(fontsize=8)

    axD[0, 1].set_xlabel('HBS rank  rk')
    axD[0, 1].set_ylabel('rel. residual (dense operator)')
    axD[0, 1].set_title('true residual vs rank')
    axD[0, 1].grid(True, which='both', alpha=0.3); axD[0, 1].legend(fontsize=8)

    axD[1, 0].set_xlabel('HBS rank  rk')
    axD[1, 0].set_ylabel('factorize + solve  [s]')
    axD[1, 0].set_title('cost vs rank')
    axD[1, 0].grid(True, alpha=0.3); axD[1, 0].legend(fontsize=8)

    axD[1, 1].set_xlabel('HBS rank  rk')
    axD[1, 1].set_ylabel('S-map compression (HBS / dense)')
    axD[1, 1].set_title('compression vs rank')
    axD[1, 1].grid(True, alpha=0.3); axD[1, 1].legend(fontsize=8)

    figD.suptitle('Channel barotropic mode (screened diffusion), '
                  'HBS rank sweep across dt', fontsize=12)
    figD.tight_layout(rect=[0, 0, 1, 0.96])
    figD.savefig('channel_diagnostics.png', dpi=200)

    print("wrote channel_setup.png, channel_fields.png, channel_diagnostics.png")
    if SHOW_PLOTS:
        plt.show()
except Exception as e:
    print("plotting skipped:", e)
