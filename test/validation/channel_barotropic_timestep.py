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
# operator is fixed: ONE S-map assembly and ONE factorization of the cyclic
# block-tridiagonal interface system serve every timestep; the per-step cost
# is a body-load rhs rebuild plus a block solve. The factorization is dense
# cyclic red-black (cyclic reduction; the periodic corners are handled by the
# reduction itself) by default; cyclic red-black in HBS arithmetic
# (SSLABLU_SOLVER=rbhbs: every Schur complement and fill-in block compressed at
# rank SSLABLU_RB_RK); or cyclic block-Thomas with the SMW corner correction
# (SSLABLU_SOLVER=thomas).
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
# are set by SSLABLU_WALLS:
#   neumann   (default) solid, no-normal-flow walls. Every slab solver carries
#             the walls as Neumann faces (solverOptions bc_types), so the wall
#             values are unknowns of the local solves. With v^{n+1} = v* -
#             (g dt/L) d eta/dy, the wall data
#                 d eta/dn = n_y v* L / (g dt)        (outward normal, n_y = -/+1)
#             makes v^{n+1} vanish on the wall nodes, to round-off: the
#             reconstructed field's discrete d eta/dn equals the data, and the
#             velocity update uses the same leaf d/dy. The wall flux D d eta/dn
#             = (H dt/L) v*.n cancels the wall flux inside R^n, so mass changes
#             only by the steric source, to spectral accuracy. v* is NOT zeroed
#             at the walls: the explicit Coriolis term makes it nonzero there,
#             and the data accounts for it.
#   emulated  the earlier closed-wall emulation in the all-Dirichlet path
#             (zero-gradient wall data from the lagged adjacent eta, plus a taper
#             of v* to 0 at the walls). Leaks O(dt^2/delta) and needed the sponge
#             stabilizers below; kept for comparison with neumann, to be removed.
#   dirichlet Dirichlet SSH on the walls: steric-held (FORCED, WALL_STERIC),
#             clamped 0 (an open boundary to a reservoir at rest), or a
#             time-periodic "tidal" driver (SSLABLU_WALL_AMP). Not solid walls:
#             mass is conserved only up to the physical wall flux.
# In every mode the conservation diagnostics compare the divergence form with
# the non-conservative form, which adds a spurious volume term.
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
#   GATE    manufactured cubic on one double slab (x faces Dirichlet, y-walls
#           as configured): derivative matrices, skeleton body-rhs sign,
#           body-load reconstruction; with Neumann walls also the wall rows
#           and the discrete wall flux.
#   RUN     NSTEPS of backward-Euler IMEX, both PDO forms (mass comparison).
#   OPTIONAL SSLABLU_DTCONV=1: dt-convergence ratios (~2.0 for backward Euler);
#           rebuilds the operator per dt, so this is slow and off by default.
#   OPTIONAL SSLABLU_FLUXBUDGET=1: per-step volume budget of the divergence
#           form (ChannelModel.flux_budget): splits each step's volume change
#           exactly into the pointwise continuity residual by leaf-node class,
#           the product-rule div(H u) vs the collocation derivative of the nodal
#           product H u, the wall flux, and the normal-flux jumps across
#           internal leaf edges (each box carries its own copy of the edge
#           velocities). The interior residual and the wall flux / edge jumps at
#           non-corner nodes use no leaf-corner velocity; the other terms do,
#           and the discretization never defines one (corners are not dofs;
#           their u, v never reach eta), so only their SUM is meaningful. Writes
#           channel_timestep_fluxbudget.csv / .png and a VOLUME BUDGET summary
#           (with the edges carrying the largest jumps) into
#           run_sslablu_fluxbudget_... instead of run_sslablu_channel_... (same
#           solution, so the regular run directories are not overwritten or
#           picked up twice by the comparison scripts).
#
#   All outputs go to one directory per configuration (created as needed;
#   rerunning identical settings overwrites):
#     run_sslablu_channel_p<p>_N<N>_pan<npan_x>x<npan_y>[_rk<RK>][_rb|_rbhbs_rbrk<RB_RK>]_dt<dt>s_nsteps<NSTEPS>[_neumann][_ridgectr][_rng<SEED>]/
#   dt in seconds, %g-formatted exactly as reentrant_channel_sslablu.jl formats
#   its Δt, so paired runs share the dt/nsteps tokens; _rk<RK> marks HBS-
#   compressed S-maps, _rb the dense red-black solver and _rbhbs_rbrk<RB_RK>
#   the HBS one (no token = cyclic Thomas, as in runs made before red-black
#   existed), _neumann the true Neumann walls (no token = the emulated or
#   Dirichlet walls, as in runs made before Neumann walls existed), _ridgectr
#   marks SSLABLU_RIDGE_MIDPANEL=0, _rng<SEED> a non-zero SSLABLU_RNG_SEED
#   (randomized runs only, i.e. RK > 0).
#   channel_timestep_diag.csv        per-step diagnostics
#   channel_timestep_ssh.npz         eta/u/v samples for channel_ssh_compare.py
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
#   SSLABLU_SOLVER     rb | rbhbs (red-black; N must be a power of 2) | thomas
#                      (default rb). rbhbs needs SSLABLU_RK > 0.
#   SSLABLU_RB_RK      rbhbs SOLVER rank for pivots and fill-in (default p, half
#                      the HBS leaf size). Independent of SSLABLU_RK, the
#                      ASSEMBLER rank: two separate knobs.
#   SSLABLU_RNG_SEED   seed for the randomized HBS sketches     (default 0)
#   SSLABLU_COMPARE_FORMS  1 = also run non-conservative form   (default 1)
#   SSLABLU_DTCONV     1 = run dt-convergence study             (default 0)
#   SSLABLU_FLUXBUDGET 1 = per-step volume budget (see above)   (default 0)
#   SSLABLU_WALLS      neumann | emulated | dirichlet           (default neumann)
#   SSLABLU_WALL_AMP   tidal wall SSH amplitude [m] (dirichlet walls, bump scenario)
#   SSLABLU_SPONGE_W   wall band width [y/L] (sponge; emulated v* taper) (default 0.1)
#   SSLABLU_SPONGE_RATE  Rayleigh rate in the band [1/s]; 0 = off   (default 0)
#   SSLABLU_WALL_RELAX under-relaxation of the emulated wall copy   (default 1)
# =============================================================================

import io
import os
import sys
import time
import warnings
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
# NOT the same module as multislab.omsdirectsolve: this one has the solver classes
from direct_solve.omsdirectsolve import RedBlackSolver

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
# Ridge crest position x/L. With the crest at 0.5 it sits exactly on slab
# interface N/2 AND on a leaf-panel edge. SSLABLU_RIDGE_MIDPANEL=1 (default)
# shifts it by half a panel, 1/32, which is mid-panel for the default
# N * npan_x = 16 (panel edges at multiples of 2/(N*npan_x); checked below).
# SSLABLU_RIDGE_MIDPANEL=0 restores the centered ridge. The SAME env var is
# read by reentrant_channel_sslablu.jl, so one setting drives both models.
RIDGE_MIDPANEL = os.environ.get("SSLABLU_RIDGE_MIDPANEL", "0") != "0"
RIDGE_XC = 0.5 + (1.0 / 32.0 if RIDGE_MIDPANEL else 0.0)
FCOR     = -1.0e-4        # Coriolis parameter [1/s] (explicit; needs |FCOR|*dt < 2)
RHO0     =  1025.0        # reference seawater density [kg/m^3]

# ---- Forced re-entrant-channel scenario (wind + drag + steric buoyancy) -----
# Uniform SSH at rest, spun up by a steady zonal wind, equilibrated by linear
# bottom drag, and tilted meridionally by a prescribed steric height. The goal
# is an emerging north-high / south-low SSH with a "finger" steered through the
# ridge gap. FORCED = 0 recovers the original geostrophic-adjustment bump.
FORCED     = os.environ.get("SSLABLU_FORCED", "1") != "0"
TAU0       = float(os.environ.get("SSLABLU_TAU0",   "0.15"))   # wind stress amp [N/m^2]
RDRAG      = float(os.environ.get("SSLABLU_RDRAG",  "1.0e-5")) # linear bottom drag [1/s] default was 1e-5, successful (with dirichlet BC) was 5e-5
STERIC_AMP = float(os.environ.get("SSLABLU_STERIC", "0.0"))    # steric SSH half-range [m], default 0.5
GAMMA_S    = float(os.environ.get("SSLABLU_GAMMA_S","0.0")) # steric relaxation rate [1/s], default 1.0e-5
# y-wall treatment (see BOUNDARY CONDITIONS in the header):
#   neumann   true no-normal-flow walls: Neumann faces in every slab solver, with
#             data d eta/dn = n_y v* L/(g dt) (SlabSolve.wall_flux_data)
#   emulated  the earlier emulation in the all-Dirichlet path: (1) zero-gradient
#             wall data -- each wall node gets the adjacent inward eta^n, so
#             d eta/dn ~ 0; (2) v* tapered to 0 at the walls before div(H u*).
#             Kept for comparison with neumann; to be removed.
#   dirichlet Dirichlet SSH on the walls (wall_eta): open, not solid, walls
WALLS = os.environ.get("SSLABLU_WALLS", "neumann").lower() # want "neumann"
if WALLS not in ("neumann", "emulated", "dirichlet"):
    raise ValueError("SSLABLU_WALLS must be 'neumann', 'emulated' or 'dirichlet', got %r" % WALLS)
if "SSLABLU_WALL_NOFLUX" in os.environ:
    # replaced by SSLABLU_WALLS: 1 was the emulated closed walls, 0 the Dirichlet walls
    raise ValueError("SSLABLU_WALL_NOFLUX is replaced by SSLABLU_WALLS (neumann | emulated "
                     "| dirichlet); WALL_NOFLUX=1 was 'emulated', 0 was 'dirichlet'")
WALL_NOFLUX = WALLS != "dirichlet"     # closed walls: mass budget, wall |v| diagnostics
# Neumann wall faces for every slab solver (x faces: the slab interfaces)
NEUMANN_WALLS = {"x": "dirichlet", "y": "neumann"}
# y-wall Dirichlet data in FORCED (dirichlet walls only): 1 = hold walls at the
# steric height eta_s (tilt is BC-driven, establishes fast); 0 = zero walls, so
# the meridional tilt must emerge purely from the interior GAMMA_S buoyancy
# relaxation (slower, fully "inductive"). See the wall_eta method.
WALL_STERIC = os.environ.get("SSLABLU_WALL_STERIC", "1") != "0" # Default 1
# Wall-band stabilizers, confined to a band of width SPONGE_W next to the y-walls
# (dist = min(y, 1-y)). Built for the emulated walls, which are weakly UNSTABLE on
# long runs: the lagged zero-grad copy closes a feedback loop with |G| slightly
# > 1 for a wall-trapped, grid-scale mode, and a hard single-row v*=0 mask
# injects a Gibbs seed for it.
#   SPONGE_W    band width [y-units]: the sponge band and, for emulated walls, the
#               smooth taper of v* to 0 at the wall (0 => the hard v* mask)
#   SPONGE_RATE Rayleigh friction rate [1/s] in the band (absorbs the mode),
#               smoothly ramped to 0 at the band edge (a too-abrupt sponge
#               reflects). Applied to both u and v -> a frictional wall layer,
#               for closed walls (neumann, emulated). 0 = off (default); the
#               earlier emulated runs used 1e-3.
#   WALL_RELAX  under-relaxation of the emulated zero-grad wall copy (1 = full
#               copy, <1 lowers the feedback-loop gain).
# These do NOT touch the operator/gate; they only reshape the explicit predictor
# and wall data. For emulated walls they mitigate, not cure.
SPONGE_W    = float(os.environ.get("SSLABLU_SPONGE_W",    "0.1"))
SPONGE_RATE = float(os.environ.get("SSLABLU_SPONGE_RATE", "0.0")) #"0.0 default for neumann"
WALL_RELAX  = float(os.environ.get("SSLABLU_WALL_RELAX",  "1.0"))
# Diagnostic: seed a finite grid-scale wall perturbation in the IC (0 = off) so
# the wall-mode growth factor |G| can be read off a short run.
SEED_WALL   = float(os.environ.get("SSLABLU_SEED_WALL",   "0.0"))

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
    """1-periodic Gaussian-like ridge profile, centered at x = RIDGE_XC."""
    return lib.exp(RIDGE_KB * (lib.cos(2.0 * np.pi * (x - RIDGE_XC)) - 1.0))


def dbump_x(x, lib):
    """d/dx of bump_x (analytic)."""
    return (-2.0 * np.pi * RIDGE_KB * lib.sin(2.0 * np.pi * (x - RIDGE_XC))
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
    The ridge crest (bump peak at x = RIDGE_XC) is cut down to (1 - GAP_DEPTH) of
    its height inside the meridional gap band, opening a deep channel there."""
    return 1.0 - RIDGE_HR * gapfac(y, lib) * bump_x(x, lib)


def ddepth_frac_dx(x, y, lib=np):
    """d/dx of depth_frac (analytic)."""
    return -RIDGE_HR * gapfac(y, lib) * dbump_x(x, lib)


def ddepth_frac_dy(x, y, lib=np):
    """d/dy of depth_frac (analytic; zero when GAP_DEPTH = 0)."""
    return -RIDGE_HR * dgapfac(y, lib) * bump_x(x, lib)


def wind_stress(y, lib=np):
    """Steady zonal (eastward / 'westerly') wind stress tau^x(y) [N/m^2]:
    a single mid-channel jet, tau0*sin(pi*y), vanishing at both y-walls. Net
    eastward momentum input -> must be balanced by bottom drag + ridge form
    drag (the ACC / Southern-Ocean momentum budget). x-independent."""
    return TAU0 * lib.sin(np.pi * y)


def steric_height(y, lib=np):
    """Prescribed steric SSH target eta_s(y) [m]: linear meridional profile,
    high to the north (y = 1, +STERIC_AMP), low to the south (y = 0). The
    sea-surface expression of a meridional temperature/buoyancy gradient (warm
    equatorward). The free surface is relaxed toward this; the y-walls are held
    at it. x-independent."""
    return STERIC_AMP * (2.0 * y - 1.0)


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
    """Global boundary = the y-walls only. x has no boundary (periodic).
    With Neumann walls the wall points are unknowns of the slab solvers, not
    boundary points, so nothing matches and Igb is empty."""
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
    a body load and the wall data and (b) reconstruct the full leaf field, every
    timestep. With Neumann walls (opts.bc_types) the wall points are rows and
    columns of Aii: Ii is the interior skeleton followed by the Neumann points.

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

        # Neumann walls (SSLABLU_WALLS=neumann): the leaf grid node of each
        # Neumann point, in the solver's I_Ntot order, and the y-component of its
        # outward normal (-1 on y = 0, +1 on y = 1). This operator has no c12, so
        # the leaf faces are Chebyshev and every Neumann point is a grid node.
        self.neumann = dd.has_neumann
        if self.neumann:
            if dd.hps.interpolate:
                raise RuntimeError("Neumann walls here assume Chebyshev leaf faces "
                                   "(an operator without c12)")
            size_ext = len(dd.hps.H.JJ.Jx)
            Jx = np.asarray(dd.hps.H.JJ.Jx)
            single = dd.hps.I_single.detach().cpu().numpy()
            self.neu_box, self.neu_node = single // size_ext, Jx[single % size_ext]
            self.neu_ny = dd.normals_Ntot[:, 1].detach().cpu().numpy()
            assert np.allclose(self.gx[self.neu_box, self.neu_node],
                               dd.XX_active[dd.I_Ntot].detach().cpu().numpy()), \
                "Neumann points are not the leaf wall nodes"

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

        # --- closed-wall (no-flux) support -----------------------------------
        # leaf points on the y-walls (for the emulated v* mask and the IC seed)
        yy = self.gx[:, :, 1]
        self.wall_mask = ((np.abs(yy - BNDS[0][1]) < 1e-12)
                          | (np.abs(yy - BNDS[1][1]) < 1e-12))
        own_set = np.zeros(self.nb, dtype=bool)
        own_set[self.own] = True
        # wall-flux diagnostic: own boxes, leaf corners left out (they are not
        # dofs: their values are extrapolated, so their v is not constrained)
        corner = np.zeros_like(self.wall_mask)
        for b, (uxn, uyn, ix, iy) in enumerate(self.box_meta):
            corner[b] = (((ix == 0) | (ix == len(uxn) - 1))
                         & ((iy == 0) | (iy == len(uyn) - 1)))
        self.wall_mask_own = self.wall_mask & ~corner & own_set[:, None]

        # emulated walls: zero-gradient map: each y-wall boundary node (in Igb, indexing XXb) ->
        # its adjacent inward leaf node (same box + x-node, one y-node inward).
        # Setting eta_wall = eta there makes the discrete normal gradient ~ 0.
        coord2leaf = {}
        for b in range(self.nb):
            for j in range(self.pp2):
                coord2leaf[(round(float(self.gx[b, j, 0]), 9),
                            round(float(self.gx[b, j, 1]), 9))] = (b, j)
        inb, inj = [], []
        for idx in self.Igb:
            key = (round(float(self.XXb[idx, 0]), 9),
                   round(float(self.XXb[idx, 1]), 9))
            b, j = coord2leaf[key]
            uxn, uyn, ix, iy = self.box_meta[b]
            ix0, iy0 = ix[j], iy[j]
            iy_in = 1 if iy0 == 0 else (len(uyn) - 2)
            jin = int(np.where((ix == ix0) & (iy == iy_in))[0][0])
            inb.append(b); inj.append(jin)
        self.wall_in_b = np.array(inb, dtype=int)
        self.wall_in_j = np.array(inj, dtype=int)

        # smooth wall band (dist = min(y, 1-y)); Hermite smoothstep s in [0,1]
        # is 0 at the wall, 1 at/beyond the band edge, with zero slope at both
        # ends (so tapering v* by it injects no Gibbs kink).
        dist = np.minimum(yy - BNDS[0][1], BNDS[1][1] - yy)
        if SPONGE_W > 0.0:
            sfrac = np.clip(dist / SPONGE_W, 0.0, 1.0)
            smoothstep = sfrac * sfrac * (3.0 - 2.0 * sfrac)
        else:
            smoothstep = (~self.wall_mask).astype(float)   # hard mask fallback
        self.vtaper = smoothstep                    # multiplies v*  (0 at wall)
        self.sponge = SPONGE_RATE * (1.0 - smoothstep)   # Rayleigh rate (max at wall)

    def wall_zero_grad(self, eta_field):
        """Emulated walls: zero-gradient Dirichlet data on the y-walls: value at
        the adjacent inward leaf node (eta_wall = eta_first-interior => d eta/dn ~ 0)."""
        return eta_field[self.wall_in_b, self.wall_in_j]

    def wall_flux_data(self, vstar, gdtL):
        """Neumann walls: the data d eta/dn = n_y v* / gdtL at the Neumann points,
        (n_N, 1). The velocity update v = v* - gdtL d eta/dy then vanishes on the
        wall nodes, since the reconstructed eta's d/dy there is this data."""
        return (self.neu_ny * vstar[self.neu_box, self.neu_node] / gdtL)[:, None]

    def gradx(self, F):
        return np.einsum('ij,bj->bi', self.D1, F)

    def grady(self, F):
        return np.einsum('ij,bj->bi', self.D2, F)

    def skeleton_load(self, fvec, gN=None):
        """b: the statically-condensed body load on the rows of Aii, in the order
        Ii (domain_driver.get_rhs with zero Dirichlet data): b_C on the interior
        skeleton, then -- Neumann walls -- g_N + the single-copy load on the wall
        points."""
        zero_dir = torch.zeros(len(self.dd.I_Xtot), 1)
        b = self.dd.get_rhs(None, uu_dir_vec=zero_dir, ff_body_vec=fvec, uu_neu_vec=gN)
        return b.detach().cpu().numpy().real.ravel()

    def local_solve(self, fvec, fgb, gN=None):
        """u_i = Aii^{-1} ( b - Aib[:,Igb] fgb ) on all of Ii, with zero data on
        the slab interfaces (Il, Ir)."""
        w = self.sv.solver_ii @ (self.skeleton_load(fvec, gN)
                                 - self.sv.Aib[:, self.Igb] @ fgb)
        return np.asarray(w).ravel()

    def body_rhs(self, fvec, fgb, gN=None):
        """Interface-system rhs of this slab (central-interface restriction):
        rhs = ( Aii^{-1} ( b - Aib[:,Igb] fgb ) )[Ic]."""
        return self.local_solve(fvec, fgb, gN)[self.Ic]

    def reconstruct(self, ul, ur, fgb, fvec, gN=None):
        """Full leaf field from solved neighbor traces + wall data (Dirichlet fgb
        or Neumann gN) + body load."""
        g = np.zeros(self.XXb.shape[0])
        g[self.Il] = ul
        g[self.Ir] = ur
        g[self.Igb] = fgb
        g = torch.from_numpy(g[:, np.newaxis])
        with redirect_stdout(io.StringIO()):   # mute per-solve residual prints
            uu = self.sv.solver.solve_dir_full(g, ff_body=fvec, uu_neu=gN)
        uu = uu.detach().numpy() if torch.is_tensor(uu) else np.asarray(uu)
        uu = uu.real.reshape(self.nb, self.pp2)
        for b in range(self.nb):               # repair the non-dof leaf corners
            for corner, row, w in self.cfix[b]:
                uu[b, corner] = w @ uu[b, row]
        return uu

    def integrate_own(self, F):
        return float((self.W[self.own] * F[self.own]).sum())

    def budget_geom(self):
        """Bookkeeping for ChannelModel.flux_budget (built once). Per own box,
        its four edges as (key, node idx, 1D quadrature weights along the edge,
        normal axis, outward normal sign, leaf-corner mask); the key is the
        edge's position with x taken mod 1, so the two sides of an internal edge
        -- in this slab or the neighbor's own region -- meet under one key and a
        wall edge appears once. Plus a class per leaf node for the pointwise
        residual: 0 interior, 1 leaf edge, 2 wall (non-corner), 3 leaf corner."""
        if getattr(self, "_bg", None) is not None:
            return self._bg
        r9 = lambda v: round(float(v), 9)
        xm = lambda v: r9(r9(v) % 1.0)
        cls = np.zeros((self.nb, self.pp2), dtype=int)
        edges = []
        for b in self.own:
            uxn, uyn, ix, iy = self.box_meta[b]
            nx, ny = len(uxn), len(uyn)
            onx, ony = (ix == 0) | (ix == nx - 1), (iy == 0) | (iy == ny - 1)
            cls[b] = np.where(onx & ony, 3, np.where(self.wall_mask[b], 2,
                                                     np.where(onx | ony, 1, 0)))
            wx, wy = cheb_quad_weights(uxn), cheb_quad_weights(uyn)
            box = []
            for sel, axis, sgn, key in (
                    (ix == 0, 0, -1.0, ("v", xm(uxn[0]), r9(uyn[0]))),
                    (ix == nx - 1, 0, 1.0, ("v", xm(uxn[-1]), r9(uyn[0]))),
                    (iy == 0, 1, -1.0, ("h", r9(uyn[0]), xm(uxn[0]))),
                    (iy == ny - 1, 1, 1.0, ("h", r9(uyn[-1]), xm(uxn[0])))):
                idx = np.where(sel)[0]
                along, w1 = (iy[idx], wy) if axis == 0 else (ix[idx], wx)
                box.append((key, idx, w1[along], axis, sgn,
                            (along == 0) | (along == len(w1) - 1)))
            edges.append(box)
        self._bg = {"cls": cls, "edges": edges}
        return self._bg

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
#   GATE: manufactured cubic on ONE double slab
#
#   u_ex = x^3 + y^3 (collocation-exact for p_disc >= 4). The slab's x faces get
#   exact Dirichlet data, its y-walls exact data of the configured kind (values
#   for Dirichlet walls, the outward du/dn for Neumann walls). Validates, in
#   order:
#     (a) leaf derivative matrices Ds[3]/Ds[4] (orientation + physical scaling)
#     (b) sign/indexing of the skeleton-reduced body rhs (with Neumann walls,
#         also the wall rows and their data)
#     (c) body-load reconstruction through solve_dir_full
#     (d) Neumann walls: the reconstructed field's discrete du/dn on the wall
#         nodes equals the data -- what makes v = 0 on the walls exact
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
    fgb = u_ex(ss.XXb[ss.Igb, :])          # x faces (and Dirichlet walls)
    gN = None
    if ss.neumann:                         # outward du/dn = n_y u_y on the walls
        yN = ss.gx[ss.neu_box, ss.neu_node][:, 1]
        gN = (ss.neu_ny * 3.0 * yN ** 2)[:, None]

    # (b) skeleton body rhs: with every slab boundary point given exact data,
    # Il = Ir = [] and the interface identity reduces to
    # u_i = Aii^{-1}(b - Aib fgb) on ALL of Ii (wall points included if Neumann)
    ui = ss.local_solve(fvec, fgb, gN)
    ue_i = u_ex(ss.XXi)
    err_skel = np.linalg.norm(ui - ue_i) / np.linalg.norm(ue_i)

    # (c) body-load reconstruction on the full leaf grids
    uu = ss.reconstruct(np.zeros(0), np.zeros(0), fgb, fvec, gN)
    err_rec = np.linalg.norm(uu - Ue) / np.linalg.norm(Ue)

    # (d) Neumann walls: discrete du/dn of the reconstruction = the data
    err_flux = 0.0
    if ss.neumann:
        dudn = ss.neu_ny * ss.grady(uu)[ss.neu_box, ss.neu_node]
        err_flux = np.linalg.norm(dudn - gN[:, 0]) / np.linalg.norm(gN)

    print("=============GATE (manufactured cubic, one slab)=============")
    print("y-walls                      =  %s" % ("Neumann (du/dn data)" if ss.neumann
                                                  else "Dirichlet (values)"))
    print("leaf d/dx matrix rel. err    = ", '%10.3E' % err_dx)
    print("leaf d/dy matrix rel. err    = ", '%10.3E' % err_dy)
    print("skeleton body-rhs rel. err   = ", '%10.3E' % err_skel)
    print("reconstruction rel. err      = ", '%10.3E' % err_rec)
    if ss.neumann:
        print("wall du/dn vs data rel. err  = ", '%10.3E' % err_flux)
    print("=============================================================")
    worst = max(err_dx, err_dy, err_skel, err_rec, err_flux)
    if worst > GATE_TOL:
        raise RuntimeError("GATE FAILED: worst rel. err %.3E > %.1E -- "
                           "body-load sign/indexing, wall data or Ds scaling is wrong"
                           % (worst, GATE_TOL))


################################################################
#
#   THE TIME LOOP
#
################################################################

class ChannelModel:
    """Backward-Euler IMEX barotropic channel. Fixed dt -> the elliptic
    operator is fixed -> S assembly + cyclic red-black (dense or HBS) or
    Thomas factorization happen ONCE (in __init__); step() rebuilds only the
    body-load rhs and solves with the stored factors.
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

        # both solvers (and step()) use contiguous interface blocks i*nc:(i+1)*nc
        assert all(list(d) == list(range(i * self.nc, (i + 1) * self.nc))
                   for i, d in enumerate(self.OMS.glob_target_dofs)), \
            "interface dofs are not contiguous per slab"

        I_nc = np.eye(self.nc)
        tic = time.perf_counter()
        if SOLVER == "rb":
            # identity diagonal (Stot = I + S); cyclic=True: the wrap-around
            # couplings S_list[0][0] and S_list[-1][1] are reduced like any
            # other. HBS-compressed S-maps (SSLABLU_RK > 0) are densified
            # first, as the Thomas path does implicitly: RK compresses the
            # S-maps only, the factorization stays dense (RedBlackSolver can't
            # negate HBSMAT)
            S_rb = [[b if isinstance(b, np.ndarray) else np.asarray(b @ I_nc)
                     for b in blocks] for blocks in S_list]
            self.rb = RedBlackSolver(self.nc, cyclic=True)
            self.rb.factorize(S_rb, [I_nc] * self.N)
        elif SOLVER == "rbhbs":
            # HBS arithmetic throughout: the S-blocks stay compressed (rank
            # RK) and every Schur complement / fill-in block is compressed at
            # the solver's own rank RB_RK. One cluster tree serves every block,
            # which is valid because every interface carries the same y-points;
            # the tree structures are checked rather than assumed.
            tree0 = S_list[0][0].tree
            if not all(np.array_equal(b.tree.perm_leaf, tree0.perm_leaf)
                       and b.tree.nleaves == tree0.nleaves
                       for blocks in S_list for b in blocks):
                raise RuntimeError("HBS cluster trees differ between interfaces, so "
                                   "slab 0's tree cannot serve every block")
            self.rbhbs = omsdirectsolveHBS.RedBlackSolverHBS(
                self.nc, RB_RK, tree0, S_list[0][0].quad, cyclic=True, seed=RNG_SEED)
            self.rbhbs.factorize(S_list)     # T=None: identity-diagonal fast paths
        else:
            self.T, self.smw = omsdirectsolve.build_block_cyclic_tridiagonal_solver(
                self.OMS, S_list, rhs0, self.Ntot, self.nc)
        self.t_fac = time.perf_counter() - tic

        # per-run solver check on one random rhs (reference factors discarded):
        #   rb    -- against cyclic Thomas + SMW; both dense, expect ~1e-15
        #   rbhbs -- against dense cyclic red-black on the SAME S-blocks,
        #            densified: the solver's own compression error at rank
        #            RB_RK (the S-map compression at rank RK is common to both)
        self.solver_check, self.solver_check_label = np.nan, ""
        r = np.random.default_rng(0).standard_normal(self.Ntot)
        if SOLVER == "rb":
            T_, smw_ = omsdirectsolve.build_block_cyclic_tridiagonal_solver(
                self.OMS, S_list, rhs0, self.Ntot, self.nc)
            x_ref = omsdirectsolve.block_cyclic_tridiagonal_solve(self.OMS, T_, smw_, r)
            self.solver_check_label = "red-black vs Thomas"
        elif SOLVER == "rbhbs":
            ref_ = RedBlackSolver(self.nc, cyclic=True)
            ref_.factorize([[np.asarray(b @ I_nc) for b in blocks] for blocks in S_list])
            x_ref = np.asarray(ref_.solve(r)).ravel()
            self.solver_check_label = ("HBS red-black (rank %d) vs dense red-black"
                                       % RB_RK)
        if SOLVER in ("rb", "rbhbs"):
            self.solver_check = (np.linalg.norm(self.solve(r) - x_ref)
                                 / np.linalg.norm(x_ref))

        tic = time.perf_counter()
        self.sl = [SlabSolve(dSlabs[n], self.diff_op, opts, gb,
                             own_split=n * H) for n in range(self.N)]
        self.t_keep = time.perf_counter() - tic

        # steady forcing fields, precomputed once per slab (leaf grids)
        self.a_wind = []      # eastward wind acceleration tau^x/(rho0 H)
        self.eta_s = []       # steric relaxation target eta_s(y)
        for s in self.sl:
            xg, yg = s.gx[:, :, 0], s.gx[:, :, 1]
            Hp = H0 * depth_frac(xg, yg)
            self.a_wind.append(wind_stress(yg) / (RHO0 * Hp))
            self.eta_s.append(steric_height(yg))

        # initial condition
        self.eta, self.u, self.v = [], [], []
        for s in self.sl:
            xg, yg = s.gx[:, :, 0], s.gx[:, :, 1]
            if FORCED or SEED_WALL != 0.0:
                # uniform SSH at rest. FORCED: tilt induced by wind + steric.
                # SEED_WALL: a clean base for the wall-mode growth-rate probe.
                self.eta.append(np.zeros_like(xg))
            else:
                # geostrophic-adjustment bump (original scenario)
                self.eta.append(ETA0
                                * np.exp(IC_KB * (np.cos(2.0 * np.pi * (xg - IC_CX)) - 1.0))
                                * np.exp(-IC_AY * (yg - IC_CY) ** 2))
            self.u.append(np.zeros_like(xg))
            self.v.append(np.zeros_like(xg))

        # diagnostic seed: a finite-amplitude, x-grid-scale (checkerboard)
        # perturbation on the wall rows, to excite the wall-trapped numerical
        # mode directly so its growth factor |G| is measurable in a few hundred
        # steps instead of the ~1e4 it takes to emerge from round-off.
        if SEED_WALL != 0.0:
            for i, s in enumerate(self.sl):
                for b in range(s.nb):
                    _, _, ix, _ = s.box_meta[b]
                    wm = s.wall_mask[b]
                    self.eta[i][b, wm] += SEED_WALL * ((-1.0) ** ix[wm])
        # previous-step wall data, for the under-relaxed zero-grad copy
        self.fgb_prev = [np.zeros(len(s.Igb)) for s in self.sl]
        self.t = 0.0

    def solve(self, rhs):
        """Interface system (I + S) u = rhs with the stored factorization."""
        if SOLVER == "rb":
            with redirect_stdout(io.StringIO()):   # RedBlackSolver.solve prints
                return np.asarray(self.rb.solve(rhs)).ravel()
        if SOLVER == "rbhbs":
            return np.asarray(self.rbhbs.solve(rhs)).ravel()
        return omsdirectsolve.block_cyclic_tridiagonal_solve(
            self.OMS, self.T, self.smw, rhs)

    def wall_eta(self, pts, t):
        """Dirichlet SSH on the y-walls. In the FORCED scenario the walls are
        held at the steric target eta_s(y_wall) (so the interior tilt has
        consistent boundary data, not a clamped zero it must fight). Otherwise:
        0 (clamped / open walls), or a zonal wavenumber-1 'tidal' driver."""
        if FORCED:
            return steric_height(pts[:, 1]) if WALL_STERIC \
                else np.zeros(pts.shape[0])
        if WALL_AMP == 0.0:
            return np.zeros(pts.shape[0])
        return (WALL_AMP * np.cos(2.0 * np.pi * pts[:, 0])
                * np.sin(2.0 * np.pi * t / WALL_PERIOD))

    def band_mean_eta(self, ylo, yhi):
        """Area-weighted mean SSH over the meridional band ylo <= y < yhi
        (tiling 'own' boxes), using the leaf quadrature weights."""
        num = den = 0.0
        for i, s in enumerate(self.sl):
            W = s.W[s.own]
            m = ((s.gx[s.own, :, 1] >= ylo) & (s.gx[s.own, :, 1] < yhi))
            num += float((W * self.eta[i][s.own] * m).sum())
            den += float((W * m).sum())
        return num / den if den > 0 else np.nan

    def mer_tilt(self):
        """Meridional SSH tilt: mean(north third) - mean(south third) [m]."""
        return self.band_mean_eta(2.0 / 3.0, 1.0) - self.band_mean_eta(0.0, 1.0 / 3.0)

    def max_speed(self):
        mu = max(np.abs(u).max() for u in self.u)
        mv = max(np.abs(v).max() for v in self.v)
        return mu, mv

    def mass(self):
        return sum(s.integrate_own(self.eta[i]) for i, s in enumerate(self.sl))

    def abs_mass(self):
        """int |eta|: a scale for mass residuals when M0 = 0 (FORCED starts at rest)."""
        return sum(s.integrate_own(np.abs(self.eta[i])) for i, s in enumerate(self.sl))

    def relax_integral(self):
        """int (eta - eta_s) over the domain: the steric-relaxation mass source
        is -GAMMA_S times this, the ONLY term that should change total mass once
        the walls are closed (periodic x carries no net zonal flux)."""
        return sum(s.integrate_own(self.eta[i] - self.eta_s[i])
                   for i, s in enumerate(self.sl))

    def wall_vn(self):
        """Max and area-mean |v| (wall-normal velocity) on the y-walls, over the
        tiling boxes' wall nodes, leaf corners left out (not dofs). The closed-wall
        target is 0; this is the direct measure of residual transport through the
        walls. Neumann walls: 0 to round-off by construction."""
        vmax = num = den = 0.0
        for i, s in enumerate(self.sl):
            m = s.wall_mask_own
            if not m.any():
                continue
            av = np.abs(self.v[i][m])
            vmax = max(vmax, float(av.max()))
            num += float((s.W[m] * av).sum()); den += float(s.W[m].sum())
        return vmax, (num / den if den > 0 else 0.0)

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

        # expected mass change from the steric source this step (eta^n), used to
        # separate the physical relaxation source from spurious wall leakage
        relax_src = -dt * GAMMA_S * self.relax_integral() if FORCED else 0.0

        # ---- explicit predictor + body load R^n, per slab -----------------
        tic = time.perf_counter()
        fgbs, gNs, fvecs, ustars, vstars = [], [], [], [], []
        rhstot = np.zeros(self.Ntot)
        for i, s in enumerate(self.sl):
            xg, yg = s.gx[:, :, 0], s.gx[:, :, 1]
            Hp = H0 * depth_frac(xg, yg)
            Hpx = H0 * ddepth_frac_dx(xg, yg)
            Hpy = H0 * ddepth_frac_dy(xg, yg)   # nonzero across the gap band

            # explicit momentum predictor: rotation, and (if forced) eastward
            # wind stress tau/(rho0 H) and linear bottom drag -r u
            if FORCED:
                us = self.u[i] + dt * (FCOR * self.v[i]
                                       + self.a_wind[i] - RDRAG * self.u[i])
                vs = self.v[i] + dt * (-FCOR * self.u[i] - RDRAG * self.v[i])
            else:
                us = self.u[i] + dt * (FCOR * self.v[i])
                vs = self.v[i] - dt * (FCOR * self.u[i])

            if WALL_NOFLUX:
                # (i) sponge (closed walls; off unless SPONGE_RATE > 0): Rayleigh
                # friction absorbing layer near the walls, damping the
                # wall-trapped instability of the emulated walls (both components)
                if SPONGE_RATE > 0.0:
                    us = us - dt * s.sponge * self.u[i]
                    vs = vs - dt * s.sponge * self.v[i]
            if WALLS == "emulated":
                # (ii) smooth taper of the wall-normal velocity: 0 at the wall
                # (no transport into it) but ramped, so no Gibbs seed. Neumann
                # walls keep v* -- the wall data accounts for it.
                vs = vs * s.vtaper

            # div(H u*) = H (u*_x + v*_y) + H_x u* + H_y v*
            divHu = Hp * (s.gradx(us) + s.grady(vs)) + Hpx * us + Hpy * vs
            R = self.eta[i] - (dt / LCHAN) * divHu
            if FORCED:
                # steric buoyancy: explicit Newtonian relaxation of the free
                # surface toward eta_s(y). Explicit -> screening coefficient (and
                # the reused factorization / gate) unchanged; stable for
                # GAMMA_S*dt << 1.
                R = R - dt * GAMMA_S * (self.eta[i] - self.eta_s[i])

            gN = None
            if WALLS == "neumann":
                # no-normal-flow data d eta/dn = n_y v*/gdtL; the walls are
                # unknowns of the slab solves, so there is no Dirichlet wall data
                fgb = np.zeros(0)
                gN = s.wall_flux_data(vs, gdtL)
            elif WALLS == "emulated":
                # zero-gradient Dirichlet: eta_wall = adjacent inward eta^n so
                # d eta/dn ~ 0 (a discrete Neumann / no-flux wall), optionally
                # under-relaxed against last step's value to lower the loop gain
                fgb = s.wall_zero_grad(self.eta[i])
                if WALL_RELAX < 1.0:
                    fgb = ((1.0 - WALL_RELAX) * self.fgb_prev[i]
                           + WALL_RELAX * fgb)
                self.fgb_prev[i] = fgb
            else:
                fgb = self.wall_eta(s.XXb[s.Igb, :], tnew)   # Dirichlet wall data
            fvec = torch.from_numpy(R.reshape(-1, 1).copy())

            rhstot[i * self.nc:(i + 1) * self.nc] = s.body_rhs(fvec, fgb, gN)
            fgbs.append(fgb); gNs.append(gN); fvecs.append(fvec)
            ustars.append(us); vstars.append(vs)
        t_rhs = time.perf_counter() - tic

        # ---- one block solve (factorization is reused) --------------------
        tic = time.perf_counter()
        uhat = self.solve(rhstot)
        t_slv = time.perf_counter() - tic

        # ---- reconstruction + velocity update -----------------------------
        tic = time.perf_counter()
        for i, s in enumerate(self.sl):
            kl, kr = self.connectivity[i]
            ul = uhat[kl * self.nc:(kl + 1) * self.nc]
            ur = uhat[kr * self.nc:(kr + 1) * self.nc]
            eta_new = s.reconstruct(ul, ur, fgbs[i], fvecs[i], gNs[i])
            self.u[i] = ustars[i] - gdtL * s.gradx(eta_new)
            self.v[i] = vstars[i] - gdtL * s.grady(eta_new)
            self.eta[i] = eta_new
        t_rec = time.perf_counter() - tic

        self.t = tnew
        maxeta = max(np.abs(e).max() for e in self.eta)
        mu, mv = self.max_speed()
        wvmax, wvmean = self.wall_vn()
        return {"mass": self.mass(), "energy": self.energy(),
                "maxeta": maxeta, "tilt": self.mer_tilt(),
                "maxu": mu, "maxv": mv, "relax_src": relax_src,
                "wall_vn_max": wvmax, "wall_vn_mean": wvmean,
                "t_rhs": t_rhs, "t_slv": t_slv, "t_rec": t_rec}

    def flux_budget(self, eta_old):
        """Volume budget of the step just taken (SSLABLU_FLUXBUDGET), from
        eta^n = eta_old and the current eta, u, v. With k = dt/L, step()'s
        product-rule divergence div(H u) = H (u_x + v_y) + H_x u + H_y v, and
        div_c(H u) the collocation derivative of the nodal product H u,
            r = eta^{n+1} - eta^n + k div(H u^{n+1}) - steric source
        is the pointwise residual of the discrete continuity equation, and
            dM = steric + int r - k int [div - div_c](H u) - k sum_leaves oint H u.n
        exactly: the leaf quadrature integrates div_c of the nodal interpolant
        exactly (a discrete divergence theorem per leaf, checked as divthm_err).
        int r is split by node class -- interior (the PDE is collocated there,
        so ~0), leaf edges (HPS imposes flux continuity there instead), Neumann
        wall nodes, leaf corners (not dofs: interpolated) -- and the summed leaf
        boundary fluxes into the wall flux and the normal-flux jumps across
        internal leaf edges (within a slab or between the own regions of two
        slabs; x-normal and y-normal edges apart), each with its leaf-corner
        part. The identity holds for any velocity at the leaf corners; the
        model's own corner u, v never reach eta (corners are not dofs, and
        step() repairs only eta there), so they drift freely -- by up to ~2x
        near the ridge -- and would swamp the individual terms with pieces that
        cancel in the total. The budget therefore repairs them like eta's
        corners (cfix). That is a convention: it moves volume among res_edge,
        res_wall, res_corner, product_rule, wall_corner and jump_corner, whose
        SUM is convention-free; res_interior, wall_noncorner and the
        jump_noncorner terms use no corner value at all. The non-corner jumps
        are also accumulated per edge in self.jump_acc. Returns BUDGET_COLS: the
        terms as contributions to dM [m], the closure dM - sum(terms) and the
        worst per-leaf divergence-theorem error, both in m and both round-off
        when the bookkeeping is right."""
        k = self.dt / LCHAN
        res = np.zeros(4)
        dM = steric = prod = dthm = 0.0
        edge = {}                        # edge key -> [(slab, outward flux, corner part)]
        for i, s in enumerate(self.sl):
            bg, o = s.budget_geom(), s.own
            xg, yg = s.gx[:, :, 0], s.gx[:, :, 1]
            Hp = H0 * depth_frac(xg, yg)
            u, v = self.u[i].copy(), self.v[i].copy()
            for b in o:
                for corner, row, w in s.cfix[b]:
                    u[b, corner] = w @ u[b, row]
                    v[b, corner] = w @ v[b, row]
            div = (Hp * (s.gradx(u) + s.grady(v)) + H0 * ddepth_frac_dx(xg, yg) * u
                   + H0 * ddepth_frac_dy(xg, yg) * v)
            Fx, Fy = Hp * u, Hp * v
            divc = s.gradx(Fx) + s.grady(Fy)
            src = (-self.dt * GAMMA_S * (eta_old[i] - self.eta_s[i]) if FORCED
                   else np.zeros_like(u))
            deta = self.eta[i] - eta_old[i]
            r = deta + k * div - src
            W = s.W[o]
            dM += float((W * deta[o]).sum())
            steric += float((W * src[o]).sum())
            for c in range(4):
                res[c] += float((W * np.where(bg["cls"][o] == c, r[o], 0.0)).sum())
            prod -= k * float((W * (div[o] - divc[o])).sum())
            for b, box in zip(o, bg["edges"]):
                out = 0.0
                for key, idx, w1, axis, sgn, cm in box:
                    f = sgn * w1 * (Fx if axis == 0 else Fy)[b, idx]
                    edge.setdefault(key, []).append((i, f.sum(), f[cm].sum()))
                    out += f.sum()
                dthm = max(dthm, k * abs(float((s.W[b] * divc[b]).sum()) - out))
        # (total, corner part) of the summed outward fluxes: walls, and the
        # internal edges by orientation ("v": x-normal, "h": y-normal)
        flux = {"wall": np.zeros(2), "v": np.zeros(2), "h": np.zeros(2)}
        if not hasattr(self, "jump_acc"):
            self.jump_acc = {}           # edge key -> cumulative non-corner jump [m]
        for key, sides in edge.items():
            tot = np.array([[fs, fc] for _, fs, fc in sides]).sum(axis=0)
            if len(sides) == 1 and key[0] == "h" and key[1] in (0.0, 1.0):
                flux["wall"] += tot
            elif len(sides) == 2:
                flux[key[0]] += tot
                self.jump_acc[key] = self.jump_acc.get(key, 0.0) - k * (tot[0] - tot[1])
            else:
                raise RuntimeError("flux budget: edge %s has %d sides" % (key, len(sides)))
        nc = lambda g: -k * (flux[g][0] - flux[g][1])
        terms = [steric] + list(res) + [prod, nc("wall"), -k * flux["wall"][1],
                                         nc("v"), nc("h"), -k * (flux["v"][1] + flux["h"][1])]
        return [dM] + terms + [dM - sum(terms), dthm]

    def _resample(self, field, nx=8, ny=8):
        """Global uniform image of a per-slab leaf field: per-leaf barycentric
        resampling of the tiling ('own') boxes (IMEXconvdiv's plot_field
        pattern; smooth, no scatter banding from Chebyshev point clustering)."""
        uxn0, uyn0, _, _ = self.sl[0].box_meta[0]
        bx, by = uxn0[-1] - uxn0[0], uyn0[-1] - uyn0[0]
        ncol, nrow = int(round(1.0 / bx)), int(round(1.0 / by))
        img = np.full((ncol * nx, nrow * ny), np.nan)
        for i, s in enumerate(self.sl):
            Bx, By = s.interp_mats(nx, ny)
            for b in s.own:
                uxn, uyn, ix, iy = s.box_meta[b]
                U2 = np.zeros((len(uxn), len(uyn)))
                U2[ix, iy] = field[i][b]
                c = int(round(np.mod(uxn[0], 1.0) / bx)) % ncol
                r = int(round(uyn[0] / by))
                img[c * nx:(c + 1) * nx, r * ny:(r + 1) * ny] = Bx @ U2 @ By.T
        xc = (np.arange(ncol * nx) + 0.5) / (ncol * nx)
        yc = (np.arange(nrow * ny) + 0.5) / (nrow * ny)
        return xc, yc, img

    def snapshot(self, nx=8, ny=8):
        return self._resample(self.eta, nx, ny)

    def eval_at(self, fields, xq, yq):
        """Evaluate per-slab leaf fields at arbitrary points (x/L periodic,
        y/L in [0,1]) with the owning 'own' leaf box's tensor-product
        barycentric interpolant -- spectrally accurate, so this is the side
        that moves when comparing against a lower-order (FV) model. The
        seam slab's own boxes live at x < 0, hence the periodic shift into
        each box's frame. Returns one array per field in `fields`."""
        xq = np.mod(np.asarray(xq, dtype=float).ravel(), 1.0)
        yq = np.asarray(yq, dtype=float).ravel()
        outs = [np.full(xq.shape, np.nan) for _ in fields]
        todo = np.ones(xq.shape, dtype=bool)
        for i, s in enumerate(self.sl):
            for b in s.own:
                uxn, uyn, ix, iy = s.box_meta[b]
                xm = np.mod(xq - uxn[0], 1.0) + uxn[0]
                m = (todo & (xm <= uxn[-1])
                     & (yq >= uyn[0]) & (yq <= uyn[-1]))
                if not m.any():
                    continue
                Bx, By = bary_mat(uxn, xm[m]), bary_mat(uyn, yq[m])
                U2 = np.zeros((len(uxn), len(uyn)))
                for F, out in zip(fields, outs):
                    U2[ix, iy] = F[i][b]
                    out[m] = np.einsum('kj,jl,kl->k', Bx, U2, By)
                todo &= ~m
        assert not todo.any(), "eval_at: %d points not in any own box" % todo.sum()
        return outs

    def sample_centers(self, n):
        """eta, u, v at the cell centers of a uniform n x n grid on the unit
        square -- exactly the Oceananigans (Nx = Ny = n) tracer points."""
        c = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(c, c, indexing='ij')
        return [F.reshape(n, n) for F in
                self.eval_at([self.eta, self.u, self.v], X, Y)]


################################################################
#
#   MAIN
#
################################################################

N        = int(os.environ.get("SSLABLU_N", "8"))
p        = int(os.environ.get("SSLABLU_P", "8"))
npan_x   = int(os.environ.get("SSLABLU_NPAN_X", "4"))   # keep EVEN
npan_y   = int(os.environ.get("SSLABLU_NPAN_Y", "8"))
dt_hours = float(os.environ.get("SSLABLU_DT_H", "0.125"))
NSTEPS   = int(os.environ.get("SSLABLU_NSTEPS", "800")) # Default 48, longest was 19200
RK       = int(os.environ.get("SSLABLU_RK", "0"))       # ASSEMBLER rank; 0 = dense S-maps
SOLVER   = os.environ.get("SSLABLU_SOLVER", "rb").lower()  # rb | rbhbs | thomas
RB_RK    = int(os.environ.get("SSLABLU_RB_RK", str(p)))  # rbhbs SOLVER rank (not RK)
RNG_SEED = int(os.environ.get("SSLABLU_RNG_SEED", "0"))
if SOLVER not in ("rb", "rbhbs", "thomas"):
    raise ValueError("SSLABLU_SOLVER must be 'rb', 'rbhbs' or 'thomas', got %r" % SOLVER)
if SOLVER in ("rb", "rbhbs") and (N < 2 or N & (N - 1)):
    # fail here with the fix in the message, not deep inside factorize
    raise ValueError("red-black needs N = power of 2 slabs, got N = %d "
                     "(or set SSLABLU_SOLVER=thomas)" % N)
if SOLVER == "rbhbs" and RK <= 0:
    # RedBlackSolverHBS.factorize calls .to('cpu') on every S-block
    raise ValueError("SSLABLU_SOLVER=rbhbs needs HBS-compressed S-maps: set "
                     "SSLABLU_RK > 0 (dense numpy S-blocks are not accepted)")
if SOLVER == "rbhbs" and RB_RK <= 0:
    raise ValueError("SSLABLU_RB_RK must be positive, got %d" % RB_RK)
# seed the randomized sketches: rkHMatAssembler draws from the global numpy
# RNG, RedBlackSolverHBS takes RNG_SEED explicitly
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
if SOLVER == "rbhbs":
    # imported only when selected (it pulls in jax); a docstring in it holds an
    # invalid escape sequence, which warns whenever the module is recompiled
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        import direct_solve.omsdirectsolveHBS as omsdirectsolveHBS
CMP_FORM = os.environ.get("SSLABLU_COMPARE_FORMS", "1") != "0"
DO_DTCNV = os.environ.get("SSLABLU_DTCONV", "0") != "0"
FLUXBUDGET = os.environ.get("SSLABLU_FLUXBUDGET", "0") != "0"
# flux_budget() return order, as contributions to the step's volume change dM [m]
BUDGET_COLS = ["dM", "steric", "res_interior", "res_edge", "res_wall", "res_corner",
               "product_rule", "wall_noncorner", "wall_corner", "jump_noncorner_x",
               "jump_noncorner_y", "jump_corner", "closure", "divthm_err"]
WALL_AMP = float(os.environ.get("SSLABLU_WALL_AMP", "0.0"))
WALL_PERIOD = 12.42 * 3600.0                             # M2-ish tide [s]
# SSH comparison export (channel_ssh_compare.py): eta/u/v sampled at the
# Oceananigans cell centers of each n x n grid listed, at NSTEPS//2 and NSTEPS
SSH_NS  = [int(v) for v in os.environ.get("SSLABLU_SSH_NS", "80,160,320").split(",") if v]
SSH_OUT = os.environ.get("SSLABLU_SSH_OUT", "")   # default: <graph_directory>/channel_timestep_ssh.npz

p_disc    = p + 2
leaf_size = 2 * p
dSlabs, connectivity, H = channel_dSlabs(N)
a = np.array([H / npan_x, 0.5 / npan_y])

# where the ridge crest sits relative to the x leaf-panel edges (multiples of
# the panel width 2H/npan_x): 0 = on an edge, 0.5 = mid-panel
panel_w = 2.0 * H / npan_x
crest_frac = (RIDGE_XC / panel_w) % 1.0
if RIDGE_MIDPANEL and abs(crest_frac - 0.5) > 0.25:
    print("WARNING: SSLABLU_RIDGE_MIDPANEL=1 but the crest x = %.5f is at panel "
          "fraction %.2f for N = %d, npan_x = %d (not mid-panel)"
          % (RIDGE_XC, crest_frac, N, npan_x))
# Neumann walls are faces of every slab solver (S-map assembly, kept slabs, gate)
opts = solverWrap.solverOptions("hpsalt", [p_disc, p_disc], a,
                                bc_types=NEUMANN_WALLS if WALLS == "neumann" else None)

dt   = 3600.0 * dt_hours
ell  = np.sqrt(GRAV * H0) * dt / LCHAN
ell2 = ell * ell

# one output directory per configuration (see the header), so runs don't
# overwrite each other
SOLVER_TOKEN = {"rb": "_rb", "rbhbs": "_rbhbs_rbrk%d" % RB_RK, "thomas": ""}[SOLVER]
SOLVER_NAME  = {"rb": "red-black", "rbhbs": "HBS red-black", "thomas": "Thomas"}[SOLVER]
graph_directory = "run_sslablu_%s_p%d_N%d_pan%dx%d%s%s_dt%gs_nsteps%d%s%s%s" % (
    "fluxbudget" if FLUXBUDGET else "channel",
    p, N, npan_x, npan_y, ("_rk%d" % RK) if RK > 0 else "", SOLVER_TOKEN,
    dt, NSTEPS, "_neumann" if WALLS == "neumann" else "",
    "" if RIDGE_MIDPANEL else "_ridgectr",
    ("_rng%d" % RNG_SEED) if (RNG_SEED != 0 and RK > 0) else "")
SSH_OUT = SSH_OUT or os.path.join(graph_directory, "channel_timestep_ssh.npz")


def outpath(name):
    return os.path.join(graph_directory, name)


def make_assembler():
    if RK > 0:
        return mA.rkHMatAssembler(leaf_size, RK)
    return mA.denseMatAssembler()


print("=============CHANNEL TIMESTEP SETUP=============")
print("N slabs / interfaces     = ", N)
print("p_disc                   = ", p_disc)
print("panels (x per slab, y)   = ", npan_x, ",", npan_y)
print("ridge crest x/L          = ", '%.5f' % RIDGE_XC,
      " (panel fraction %.2f: %s)" % (crest_frac, "mid-panel" if RIDGE_MIDPANEL
                                      else "centered, on slab/panel edge"))
print("dt                       = ", '%6.3f h' % dt_hours,
      " ell/L = %.3f  ell^2 = %.4f" % (ell, ell2))
print("steps / total time       = ", NSTEPS, "/ %.2f h" % (NSTEPS * dt_hours))
print("f*dt (explicit Coriolis) = ", '%6.3f' % (FCOR * dt))
print("S-map assembler          = ",
      ("HBS rk = %d  (SSLABLU_RK)" % RK) if RK > 0 else "dense")
print("interface solver         = ", {
      "rb":     "cyclic red-black (dense)",
      "rbhbs":  "cyclic red-black, HBS rk = %d  (SSLABLU_RB_RK)" % RB_RK,
      "thomas": "cyclic block-Thomas + SMW (dense)"}[SOLVER])
if RK > 0:
    print("random-sketch seed       = ", RNG_SEED)
print("output directory         = ", graph_directory)
if WALLS == "neumann":
    wtxt = "closed / no-normal-flow: Neumann walls, d eta/dn = n_y v* L/(g dt)"
elif WALLS == "emulated":
    wtxt = "closed / no-flux, emulated (zero-grad + v* taper)"
elif FORCED:
    wtxt = ("steric-held Dirichlet" if WALL_STERIC
            else "zero Dirichlet (open reservoir)")
else:
    wtxt = (("tidal Dirichlet, amplitude %g m" % WALL_AMP) if WALL_AMP != 0.0
            else "zero Dirichlet (clamped)")
if FORCED:
    print("scenario                 =  FORCED (wind + drag + steric)")
    print("wind stress amp TAU0     = ", '%6.3f N/m^2' % TAU0)
    print("bottom drag RDRAG        = ", '%8.2E /s  (1/r = %5.1f h)'
          % (RDRAG, 1.0 / RDRAG / 3600.0))
    print("steric half-range        = ", '%6.3f m' % STERIC_AMP)
    print("steric relax GAMMA_S     = ", '%8.2E /s  (1/g = %5.1f h)'
          % (GAMMA_S, 0.0)) #1.0 / GAMMA_S / 3600.0))
else:
    print("scenario                 =  bump (geostrophic adjustment)")
print("y-walls                  = ", wtxt, " (SSLABLU_WALLS=%s)" % WALLS)
if WALL_NOFLUX:
    print("  sponge band / rate     =  %.3f / %.2E /s%s"
          % (SPONGE_W, SPONGE_RATE, "" if SPONGE_RATE > 0.0 else "  (off)"))
if WALLS == "emulated":
    print("  v* taper / wall relax  =  %s / %.2f"
          % ("smooth" if SPONGE_W > 0 else "hard", WALL_RELAX))
print("================================================")

# ---- GATE first: nothing runs unless signs and scalings check out ----------
gate(ell2, dSlabs[N // 2], opts)

# ---- build model(s): one factorization each, reused for every step ---------
tic = time.perf_counter()
modC = ChannelModel(dt, True, make_assembler(), dSlabs, connectivity, H, opts,
                    label="divergence form")
print("[divergence form]     assemble/factor/keep-slabs = "
      "%.2f / %.2f / %.2f s" % (modC.t_asm, modC.t_fac, modC.t_keep))
if SOLVER in ("rb", "rbhbs"):
    print("[divergence form]     %s, one random-rhs solve: rel. diff = %.3E"
          % (modC.solver_check_label, modC.solver_check))
modN = None
if CMP_FORM:
    modN = ChannelModel(dt, False, make_assembler(), dSlabs, connectivity, H,
                        opts, label="non-conservative")
    print("[non-conservative]    assemble/factor/keep-slabs = "
          "%.2f / %.2f / %.2f s" % (modN.t_asm, modN.t_fac, modN.t_keep))
t_setup = time.perf_counter() - tic

M0C = modC.mass()
E0 = modC.energy()
E0safe = E0 if E0 > 0 else 1.0    # FORCED starts at rest (E0 = 0): report abs E
M0N = modN.mass() if modN is not None else np.nan

snaps = {0: modC.snapshot()}
rows = []
hist = {"t": [0.0], "dMC": [0.0], "dMN": [0.0], "E": [E0],
        "maxeta": [modC.mer_tilt() * 0.0 + (0.0 if FORCED else ETA0)],
        "tilt": [modC.mer_tilt()], "maxu": [0.0], "maxv": [0.0],
        "mass": [M0C], "massexp": [M0C], "massres": [0.0], "wvmax": [0.0],
        "t_slv": [], "t_rhs": [], "t_rec": []}
cum_relax = 0.0     # running sum of the steric-relaxation mass source
ssh_export = {}     # samples for channel_ssh_compare.py
budget = []         # SSLABLU_FLUXBUDGET: per-step volume budget, divergence form

print("")
if FORCED:
    print(" step   t[h]    tilt[m]   max|u|   wall|v|    mass_res   E[J/rho0]"
          "   t_rhs t_slv t_rec")
else:
    print(" step   t[h]    |M-M0| div-form   |M-M0| non-cons    E/E0     max|eta|"
          "   t_rhs   t_slv   t_rec")
for n in range(1, NSTEPS + 1):
    eta_old = [e.copy() for e in modC.eta] if FLUXBUDGET else None
    dC = modC.step()
    if FLUXBUDGET:
        budget.append([n, n * dt_hours] + modC.flux_budget(eta_old))
    dN = modN.step() if modN is not None else None

    dMC = abs(dC["mass"] - M0C)
    dMN = abs(dN["mass"] - M0N) if dN is not None else np.nan
    # mass budget: expected = M0 + cumulative steric source; residual = the part
    # NOT explained by relaxation = spurious wall-flux leakage
    cum_relax += dC["relax_src"]
    mass_exp = M0C + cum_relax
    mass_res = dC["mass"] - mass_exp
    if FORCED:
        print(" %4d  %6.2f  %8.4f  %7.4f  %9.2E  %9.2E   %7.4f"
              "   %4.2f  %4.3f  %4.2f"
              % (n, n * dt_hours, dC["tilt"], dC["maxu"], dC["wall_vn_max"],
                 mass_res, dC["energy"], dC["t_rhs"], dC["t_slv"], dC["t_rec"]))
    else:
        print(" %4d  %6.2f     %10.3E       %10.3E     %7.4f   %8.4f"
              "   %5.2f   %5.3f   %5.2f"
              % (n, n * dt_hours, dMC, dMN, dC["energy"] / E0safe, dC["maxeta"],
                 dC["t_rhs"], dC["t_slv"], dC["t_rec"]))

    rows.append([n, n * dt_hours, dC["mass"], dC["tilt"],
                 dC["maxu"], dC["maxv"], dC["wall_vn_max"], mass_res,
                 dC["energy"], dC["maxeta"],
                 dC["t_rhs"], dC["t_slv"], dC["t_rec"]])
    hist["t"].append(n * dt_hours)
    hist["dMC"].append(dMC); hist["dMN"].append(dMN)
    hist["E"].append(dC["energy"]); hist["maxeta"].append(dC["maxeta"])
    hist["tilt"].append(dC["tilt"])
    hist["maxu"].append(dC["maxu"]); hist["maxv"].append(dC["maxv"])
    hist["mass"].append(dC["mass"]); hist["massexp"].append(mass_exp)
    hist["massres"].append(mass_res); hist["wvmax"].append(dC["wall_vn_max"])
    hist["t_rhs"].append(dC["t_rhs"]); hist["t_slv"].append(dC["t_slv"])
    hist["t_rec"].append(dC["t_rec"])

    if n in (NSTEPS // 2, NSTEPS):
        snaps[n] = modC.snapshot()
        tag = "final" if n == NSTEPS else "mid"
        ssh_export["t_" + tag] = n * dt
        ssh_export["eta_mean_" + tag] = modC.mass()   # unit-square area
        for ns in SSH_NS:
            e_, u_, v_ = modC.sample_centers(ns)
            ssh_export["eta_%s_%d" % (tag, ns)] = e_
            ssh_export["u_%s_%d" % (tag, ns)] = u_
            ssh_export["v_%s_%d" % (tag, ns)] = v_

maxeta_run = max(hist["maxeta"])
stable = maxeta_run < 50.0 * max(ETA0, STERIC_AMP)

print("")
print("=============SUMMARY (%d steps, dt = %.2f h)=============" % (NSTEPS, dt_hours))
print("setup (both forms)       = ", '%8.2f s' % t_setup)
print("avg rhs / solve / recon  =  %6.3f / %6.4f / %6.3f s per step"
      % (np.mean(hist["t_rhs"]), np.mean(hist["t_slv"]), np.mean(hist["t_rec"])))
print("max|eta| over run        = ", '%8.4f m' % maxeta_run,
      " ->", "stable" if stable else "CHECK STABILITY")
if FORCED:
    print("final meridional tilt    = ", '%8.4f m  (N third - S third)'
          % hist["tilt"][-1])
    print("  target (steric) tilt   = ", '%8.4f m  (2/3 of full range)'
          % (STERIC_AMP * 4.0 / 3.0))
    print("final max|u| / max|v|    =  %8.4f / %8.4f m/s  (zonal flow spin-up)"
          % (hist["maxu"][-1], hist["maxv"][-1]))
    print("  (not expected to be fully equilibrated: 1/r = %.0f h, run = %.0f h)"
          % (1.0 / RDRAG / 3600.0, NSTEPS * dt_hours))
    print("final energy             = ", '%10.3E J/rho0' % hist["E"][-1])
    # conservation / wall closure
    # relative to the larger of |M0|, the steric range and int |eta| (the FORCED
    # run starts at rest, M0 = 0, and STERIC_AMP may be 0 too)
    Mscale = max(abs(M0C), STERIC_AMP, modC.abs_mass(), 1e-30)
    print("wall closure (y-walls    = ", wtxt, ")")
    print("  final max|v| at walls  = ", '%10.3E m/s  (wall nodes, leaf corners excluded)'
          % hist["wvmax"][-1])
    print("  total mass drift       = ", '%10.3E  (%.2E rel)'
          % (hist["mass"][-1] - M0C, (hist["mass"][-1] - M0C) / Mscale))
    print("  drift from steric src  = ", '%10.3E  (expected, physical)'
          % (hist["massexp"][-1] - M0C))
    print("  residual (wall leakage)= ", '%10.3E  (%.2E rel) <- want ~0'
          % (hist["massres"][-1], hist["massres"][-1] / Mscale))
else:
    print("final |M-M0| divergence  = ", '%10.3E' % hist["dMC"][-1])
    if modN is not None:
        print("final |M-M0| non-cons    = ", '%10.3E' % hist["dMN"][-1])
        if WALLS == "dirichlet":
            print("  (walls are Dirichlet SSH, not solid: both forms share the physical")
            print("   wall flux; the non-conservative excess is the spurious part)")
        else:
            print("  (closed walls: only the non-conservative form's spurious volume")
            print("   term%s should change the mass)"
                  % ("" if WALLS == "neumann" else " and the emulated walls' leakage"))
    print("final E/E0               = ", '%8.4f' % (hist["E"][-1] / E0safe))
print("=========================================================")

# ---- CSV export -------------------------------------------------------------
rows = np.array(rows)
os.makedirs(graph_directory, exist_ok=True)
csv_name = outpath("channel_timestep_diag.csv")
with open(csv_name, 'w') as f:
    f.write("step,t_hours,mass,tilt_NmS,max_u,max_v,wall_vn_max,mass_resid,"
            "energy,max_eta,t_rhs,t_solve,t_recon\n")
    np.savetxt(f, rows, fmt='%.16e', delimiter=',')
print("Wrote %s  (%d rows)" % (csv_name, rows.shape[0]))

if FLUXBUDGET:
    budget = np.array(budget)
    bname = outpath("channel_timestep_fluxbudget.csv")
    with open(bname, 'w') as f:
        f.write("step,t_hours," + ",".join(BUDGET_COLS) + "\n")
        np.savetxt(f, budget, fmt='%.16e', delimiter=',')
    print("Wrote %s  (%d rows)" % (bname, budget.shape[0]))
    cum = budget[:, 2:-2].sum(axis=0)              # dM and the terms, summed over the run
    pct = lambda c: 100.0 * c / cum[0] if cum[0] != 0.0 else np.nan
    print("")
    print("=============VOLUME BUDGET (divergence form, %d steps)=============" % NSTEPS)
    print("sum of dM                       = %11.3E m   (M(T) - M0 = %.3E m)"
          % (cum[0], hist["mass"][-1] - M0C))
    # cum indices follow BUDGET_COLS[:-2]; first the groups that use no
    # leaf-corner velocity, then the split of the rest (which does)
    for name, cols in (("steric source", (1,)),
                       ("wall flux, non-corner nodes", (7,)),
                       ("interior PDE residual", (2,)),
                       ("edge-flux jumps, x-normal", (9,)),
                       ("edge-flux jumps, y-normal", (10,)),
                       ("leaf-bdry eqs + corners", (3, 4, 5, 6, 8, 11))):
        c = cum[list(cols)].sum()
        print("  %-28s  = %11.3E m   %7.1f %% of sum dM" % (name, c, pct(c)))
    print("  leaf-bdry eqs + corners, split (depends on the corner-velocity convention):")
    for col in (3, 4, 5, 6, 8, 11):
        print("    %-26s  = %11.3E m   %7.1f %%" % (BUDGET_COLS[col], cum[col], pct(cum[col])))
    print("  largest non-corner edge-flux jumps, cumulative per edge:")
    for key, c in sorted(modC.jump_acc.items(), key=lambda kv: -abs(kv[1]))[:6]:
        if key[0] == "v":
            where = "x-normal edge x = %.4f, y in [%.4f, %.4f]" % (key[1], key[2], key[2] + 1.0 / npan_y)
        else:
            where = "y-normal edge y = %.4f, x in [%.4f, %.4f]" % (key[1], key[2], key[2] + 2.0 * H / npan_x)
        print("    %-46s  %11.3E m   %7.1f %%" % (where, c, pct(c)))
    print("max |closure| per step          = %10.3E m  (dM - sum of terms: want round-off)"
          % np.abs(budget[:, -2]).max())
    print("max per-leaf div. theorem error = %10.3E m  (want round-off)" % budget[:, -1].max())
    print("=========================================================")

# ---- SSH comparison export (channel_ssh_compare.py) --------------------------
# wall_band: width of the near-wall band where this model's wall treatment
# differs from Oceananigans' closed walls by construction (the emulated walls'
# v* taper; the sponge, if on); channel_ssh_compare.py hatches it and leaves it
# out of the interior statistics
if WALLS == "emulated":
    wall_band = SPONGE_W
elif WALLS == "neumann" and SPONGE_RATE > 0.0:
    wall_band = SPONGE_W
else:
    wall_band = 0.0
np.savez(SSH_OUT, ns=np.array(SSH_NS), dt=dt, nsteps=NSTEPS, nmid=NSTEPS // 2,
         L=LCHAN, H0=H0, p=p, N=N, npan_x=npan_x, npan_y=npan_y, ridge_xc=RIDGE_XC,
         forced=FORCED, steric_amp=STERIC_AMP, gamma_s=GAMMA_S, tau0=TAU0,
         rdrag=RDRAG, walls=WALLS, wall_band=wall_band, wall_noflux=WALL_NOFLUX,
         sponge_w=SPONGE_W, sponge_rate=SPONGE_RATE, solver=SOLVER, rk=RK, rb_rk=RB_RK,
         rng_seed=RNG_SEED, solver_check=modC.solver_check, **ssh_export)
print("Wrote %s  (eta/u/v at %s-squared cell centers)" % (SSH_OUT, SSH_NS))

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

    # ---- SSH snapshots -------------------------------------------------------
    # FORCED: shared color scale across panels so the tilt GROWING is visible
    # (t=0 is uniform 0). Bump: per-panel scale (magnitudes span orders).
    keys = sorted(snaps.keys())
    figF, axF = plt.subplots(1, len(keys), figsize=(5.0 * len(keys), 4.4),
                             sharex=True, sharey=True, squeeze=False)
    axF = axF[0]
    shared_vmax = max(np.nanmax(np.abs(snaps[k][2])) for k in keys) if FORCED \
        else None
    for k, nstep in enumerate(keys):
        xc, yc, img = snaps[nstep]
        vmax = shared_vmax if FORCED else max(np.nanmax(np.abs(img)), 1e-12)
        vmax = max(vmax, 1e-12)
        pc = axF[k].pcolormesh(xc, yc, img.T, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax, shading='auto')
        figF.colorbar(pc, ax=axF[k], shrink=0.85)
        ttl = (r'$\eta$ at t = %.1f h' % (nstep * dt_hours)) if FORCED else \
              (r'$\eta$ at t = %.2f h  (max %.3g m)' % (nstep * dt_hours, vmax))
        axF[k].set_title(ttl)
        axF[k].set_xlabel('x / L')
        axF[k].set_xlim(0, 1); axF[k].set_ylim(0, 1)
        axF[k].set_aspect('equal')
    axF[0].set_ylabel('y / L')
    if FORCED:
        figF.suptitle('wind + steric forced channel: meridional SSH tilt '
                      '(N high / S low) + gap finger', fontsize=11)
    else:
        figF.suptitle('geostrophic adjustment over the ridge: one factorization, '
                      '%d back-substitutions' % NSTEPS, fontsize=11)
    figF.tight_layout(rect=[0, 0, 1, 0.93])
    figF.savefig(outpath('channel_timestep_fields.png'), dpi=200)

    # ---- diagnostics ---------------------------------------------------------
    figD, axD = plt.subplots(2, 2, figsize=(11, 8))
    tt = np.array(hist["t"])
    if FORCED:
        # zonal-mean SSH profiles <eta>_x(y) at the snapshot times
        for nstep in keys:
            xc, yc, img = snaps[nstep]
            axD[0, 0].plot(np.nanmean(img, axis=0), yc,
                           label='t = %.0f h' % (nstep * dt_hours))
        axD[0, 0].plot(steric_height(np.linspace(0, 1, 50)),
                       np.linspace(0, 1, 50), 'k:', label='steric target')
        axD[0, 0].set_xlabel(r'$\langle\eta\rangle_x$ [m]')
        axD[0, 0].set_ylabel('y / L')
        axD[0, 0].set_title('zonal-mean SSH profile: meridional tilt emerging')
        axD[0, 0].grid(True, alpha=0.3); axD[0, 0].legend(fontsize=8)

        axD[0, 1].plot(tt, hist["tilt"], 'o-')
        axD[0, 1].axhline(STERIC_AMP * 4.0 / 3.0, color='k', ls=':',
                          label='steric target')
        axD[0, 1].set_xlabel('t [h]'); axD[0, 1].set_ylabel(r'$\eta$ tilt N-S [m]')
        axD[0, 1].set_title('meridional tilt vs time (approach to steady state)')
        axD[0, 1].grid(True, alpha=0.3); axD[0, 1].legend(fontsize=8)

        axD[1, 0].plot(tt, hist["maxu"], 'o-', label='max|u| (zonal)')
        axD[1, 0].plot(tt, hist["maxv"], 's--', label='max|v| (meridional)')
        axD[1, 0].set_xlabel('t [h]'); axD[1, 0].set_ylabel('speed [m/s]')
        axD[1, 0].set_title('flow spin-up (wind in, bottom + form drag out)')
        axD[1, 0].grid(True, alpha=0.3); axD[1, 0].legend(fontsize=8)
    else:
        axD[0, 0].semilogy(tt, np.maximum(hist["dMC"], 1e-18), 'o-',
                           label='divergence form')
        if modN is not None:
            axD[0, 0].semilogy(tt, np.maximum(hist["dMN"], 1e-18), 's--',
                               label='non-conservative')
        axD[0, 0].set_xlabel('t [h]'); axD[0, 0].set_ylabel(r'|M(t) - M$_0$|')
        axD[0, 0].set_title('mass drift (%s)'
                            % ("shared wall flux + spurious part" if WALLS == "dirichlet"
                               else "closed walls: spurious part only"))
        axD[0, 0].grid(True, which='both', alpha=0.3); axD[0, 0].legend(fontsize=8)

        axD[0, 1].plot(tt, np.array(hist["E"]) / E0safe, 'o-')
        axD[0, 1].set_xlabel('t [h]'); axD[0, 1].set_ylabel(r'E(t) / E$_0$')
        axD[0, 1].set_title('energy (backward-Euler wave damping)')
        axD[0, 1].grid(True, alpha=0.3)

        axD[1, 0].plot(tt, hist["maxeta"], 'o-')
        axD[1, 0].set_xlabel('t [h]'); axD[1, 0].set_ylabel(r'max |$\eta$| [m]')
        axD[1, 0].set_title('stability check')
        axD[1, 0].grid(True, alpha=0.3)

    steps_ax = np.arange(1, NSTEPS + 1)
    axD[1, 1].plot(steps_ax, hist["t_rhs"], 'o-', label='body-load rhs')
    axD[1, 1].plot(steps_ax, hist["t_slv"], 's-', label='cyclic %s solve' % SOLVER_NAME)
    axD[1, 1].plot(steps_ax, hist["t_rec"], '^-', label='reconstruction')
    axD[1, 1].set_xlabel('step'); axD[1, 1].set_ylabel('time [s]')
    axD[1, 1].set_title('per-step cost (factorization amortized: %.2f s once)'
                        % modC.t_fac)
    axD[1, 1].grid(True, alpha=0.3); axD[1, 1].legend(fontsize=8)

    figD.suptitle('Channel barotropic timestepping: backward-Euler IMEX, '
                  'reused cyclic %s factorization' % SOLVER_NAME, fontsize=12)
    figD.tight_layout(rect=[0, 0, 1, 0.96])
    figD.savefig(outpath('channel_timestep_diagnostics.png'), dpi=200)

    outnames = "channel_timestep_fields.png, channel_timestep_diagnostics.png"

    # ---- forcing fields (FORCED only) ---------------------------------------
    if FORCED:
        yln = np.linspace(0.0, 1.0, 200)
        figW, axW = plt.subplots(1, 3, figsize=(15, 4.4))

        axW[0].plot(wind_stress(yln), yln, 'C0')
        axW[0].set_xlabel(r'$\tau^x(y)$ [N/m$^2$]'); axW[0].set_ylabel('y / L')
        axW[0].set_title('prescribed zonal wind stress\n(eastward / westerly)')
        axW[0].axvline(0, color='k', lw=0.6); axW[0].grid(True, alpha=0.3)

        axW[1].plot(steric_height(yln), yln, 'C3')
        axW[1].set_xlabel(r'$\eta_s(y)$ [m]'); axW[1].set_ylabel('y / L')
        axW[1].set_title('prescribed steric height target\n(N high / S low)')
        axW[1].axvline(0, color='k', lw=0.6); axW[1].grid(True, alpha=0.3)

        # bottom drag: the deceleration field -r*u at the final state (shows
        # x-structure, esp. the throughflow jet at the ridge gap)
        xc, yc, uimg = modC._resample(modC.u)
        drag = -RDRAG * uimg
        dmax = max(np.nanmax(np.abs(drag)), 1e-30)
        pc = axW[2].pcolormesh(xc, yc, drag.T, cmap='PuOr',
                               vmin=-dmax, vmax=dmax, shading='auto')
        figW.colorbar(pc, ax=axW[2], shrink=0.85)
        axW[2].set_xlabel('x / L'); axW[2].set_ylabel('y / L')
        axW[2].set_title(r'bottom drag $-r\,u$ [m/s$^2$] (final)'
                         '\n' r'$r$ = %.1e /s' % RDRAG)
        axW[2].set_aspect('equal')

        figW.suptitle('Forcing fields: steady wind stress, steric target, '
                      'and bottom drag', fontsize=12)
        figW.tight_layout(rect=[0, 0, 1, 0.93])
        figW.savefig(outpath('channel_timestep_forcings.png'), dpi=200)
        outnames += ", channel_timestep_forcings.png"

    # ---- wall closure / mass conservation (FORCED only) ---------------------
    if FORCED:
        figC, axC = plt.subplots(1, 2, figsize=(11, 4.4))
        tt = np.array(hist["t"])

        axC[0].plot(tt, np.array(hist["mass"]) - M0C, 'o-', ms=3,
                    label='total drift  M(t) - M$_0$')
        axC[0].plot(tt, np.array(hist["massexp"]) - M0C, 'k--',
                    label='steric source (expected)')
        axC[0].plot(tt, hist["massres"], 's-', ms=3,
                    label='residual = wall leakage (Neumann walls: ~0)')
        axC[0].set_xlabel('t [h]'); axC[0].set_ylabel(r'mass change [m$\cdot$area]')
        axC[0].set_title('mass budget: physical source vs spurious wall flux\n'
                         '(y-walls: %s)' % wtxt)
        axC[0].grid(True, alpha=0.3); axC[0].legend(fontsize=8)

        axC[1].semilogy(tt, np.maximum(np.abs(hist["wvmax"]), 1e-18), 'o-', ms=3)
        axC[1].set_xlabel('t [h]')
        axC[1].set_ylabel(r'max $|v|$ at y-wall nodes [m/s]')
        axC[1].set_title('wall-normal velocity (closed wall -> 0;\n'
                         'leaf corners excluded: not dofs)')
        axC[1].grid(True, which='both', alpha=0.3)

        figC.suptitle('Closed-wall diagnostics: is H u·n = 0 at the y-walls?',
                      fontsize=12)
        figC.tight_layout(rect=[0, 0, 1, 0.93])
        figC.savefig(outpath('channel_timestep_conservation.png'), dpi=200)
        outnames += ", channel_timestep_conservation.png"

    # ---- volume budget (SSLABLU_FLUXBUDGET) ----------------------------------
    if FLUXBUDGET:
        tt = budget[:, 1]
        cs = np.cumsum(budget[:, 2:-2], axis=0)      # columns follow BUDGET_COLS[:-2]
        figB, axB = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True)
        panels = (
            (axB[0], "terms that use no leaf-corner velocity",
             (((7,), "wall flux, non-corner nodes"),
              ((2,), "interior PDE residual (collocated)"),
              ((9,), "edge-flux jumps, x-normal edges"),
              ((10,), "edge-flux jumps, y-normal edges"),
              ((3, 4, 5, 6, 8, 11), "leaf-boundary equations + corners (sum)")) +
             ((((1,), "steric source"),) if FORCED and GAMMA_S != 0.0 else ())),
            (axB[1], "leaf-boundary equations + corners, split\n"
                     "(depends on the velocity assumed at leaf corners)",
             (((3,), "leaf-edge node residual (flux continuity, not the PDE)"),
              ((4,), "Neumann wall-node residual"),
              ((5,), "leaf-corner residual (interpolated)"),
              ((6,), r"product-rule div($Hu$) vs collocation"),
              ((8,), "wall flux, leaf corners"),
              ((11,), "edge-flux jumps, leaf corners"))))
        for a, ttl, series in panels:
            a.plot(tt, cs[:, 0], 'k-', lw=2.2, label=r'total $M(t) - M_0$')
            for cols, lab in series:
                a.plot(tt, cs[:, list(cols)].sum(axis=1), '-', lw=1.4, label=lab)
            a.axhline(0.0, color='0.6', lw=0.6)
            a.set_xlabel('t [h]'); a.set_title(ttl)
            a.grid(True, alpha=0.3)
            a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13),
                     ncol=2, frameon=False)      # below the panel, off the curves
        axB[0].set_ylabel('cumulative contribution to the volume [m]')
        figB.suptitle('Volume budget (divergence form): where M(t) - M$_0$ comes from'
                      '  (max |closure| %.1e m)' % np.abs(budget[:, -2]).max(), fontsize=12)
        figB.tight_layout(rect=[0, 0, 1, 0.94])
        figB.savefig(outpath('channel_timestep_fluxbudget.png'), dpi=200)
        outnames += ", channel_timestep_fluxbudget.png"

    print("wrote %s  (in %s/)" % (outnames, graph_directory))
except Exception as e:
    print("plotting skipped:", e)
