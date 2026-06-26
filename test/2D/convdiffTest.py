import numpy as np
import jax.numpy as jnp
import torch
import scipy
from packaging.version import Version

# oms packages
import solver.solver as solverWrap
import multislab.oms as oms
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
# validation & testing
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import geometry.geom_2D.square as square
import globalSolution as G

CPU = torch.device("cpu")

# ---------------------------------------------------------------------------
# Conservative (divergence-form) operator  (1/dt) - div(D grad)
#   c11=c22=D,  c1=-Dx,  c2=-Dy,  c=1/dt
# ---------------------------------------------------------------------------
def make_pdo_div_Dxy(Dbar, eps, sy, sx, dt):
    tp = 2*np.pi
    Dfun  = lambda xx: Dbar*(1.0 + eps*(sy*(2.0*xx[:,1]-1.0) + sx*torch.cos(tp*xx[:,0])))
    c1fun = lambda xx: tp*Dbar*eps*sx*torch.sin(tp*xx[:,0])      # = -Dx
    c2fun = lambda xx: -(2.0*Dbar*eps*sy) + 0.0*xx[:,0]          # = -Dy (const)
    return pdo.PDO_2d(Dfun, Dfun, c1=c1fun, c2=c2fun, c=pdo.const(1.0/dt))

def D_grad(Dbar, eps, sy, sx):
    D  = lambda x,y: Dbar*(1.0 + eps*(sy*(2*y-1) + sx*np.cos(2*np.pi*x)))
    Dx = lambda x,y: Dbar*eps*sx*(-2*np.pi*np.sin(2*np.pi*x))
    Dy = lambda x,y: Dbar*eps*sy*2.0 + 0.0*x
    return D, Dx, Dy


class gmres_info(object):
    def __init__(self, disp=False):
        self._disp = disp; self.niter = 0; self.resList = []
    def __call__(self, rk=None):
        self.niter += 1; self.resList += [rk]
        if self._disp: print('iter %3i\trk = %s' % (self.niter, str(rk)))


# ---------------------------------------------------------------------------
# CFL : advective stability on the spectral grid.
#   dt <= C * dx_min / max|v| ,   dx_min ~ (1/2) ax (1 - cos(pi/(p_disc-1)))
# uses the ACTUAL discretization order p_disc and the leaf x-extent ax.
# ---------------------------------------------------------------------------
def CFL(p_disc, ax, vmax, C=5.):
    dx_min = 0.5*ax*(1.0 - np.cos(np.pi/(p_disc-1)))
    return C*dx_min/vmax


# ===========================================================================
# Problem constants
# ===========================================================================
hpsalt   = True
jax_avail = False
torch_avail = True

m = False                              # 'manufactured': if True, run the manufactured-solution check

Tfinal      = 1.                    # integrate to t=1 when timestepping (m == False)
snap_times  = [0.01,0.5, Tfinal]              # save snapshots nearest these times (png)

p        = 12
p_disc   = p + 2                      # hps/hpsalt convention offset
a_exp    = 120.0                      # bump width (a_exp avoids clash with HPS 'a')
cx, cy   = 0.5, 0.5
Dbar, eps, sy, sx = 0.1, 0.3, 1.0, 0.5
Vmag     = 0.5                        # advection strength, max|v_x| = Vmag

# geometry  (define leaf scale BEFORE dt, since CFL needs ax)
N = 5
dSlabs, connectivity, H = square.dSlabs(N,periodic=True)

print("connectivity = ")
print(connectivity)
print("slabs:")
print(dSlabs)

a   = np.array([H/16, 1/16])            # leaf box scale (x, y)
ax  = 2*a[0]                          # leaf x-extent (a is half-width in HPS convention)

dt  = CFL(p_disc, ax, Vmag)
if not m:
    # reconcile dt so the grid lands exactly on Tfinal and t=0.5 (round Nsteps up to even)
    Nsteps = int(np.ceil(Tfinal/dt));  Nsteps += Nsteps % 2
    dt = Tfinal/Nsteps
    snap_steps = sorted({int(round(ts/dt)) for ts in snap_times})
op  = make_pdo_div_Dxy(Dbar, eps, sy, sx, dt)
D, Dx, Dy = D_grad(Dbar, eps, sy, sx)

print("dt (CFL) = %.5e" % dt)
if not m:
    print("Nsteps   = %d  (Tfinal=%.2f), snapshots at steps %s" % (Nsteps, Tfinal, snap_steps))

# ---------------------------------------------------------------------------
# Boundary data: homogeneous Neumann on physical walls
# ---------------------------------------------------------------------------
def bc(P):
    return np.zeros_like(P[:, 0])

# initial field U0 (tight Gaussian -> value & gradient ~0 at straight walls)
def U0(P):
    return np.exp(-a_exp*((P[:,0]-cx)**2 + (P[:,1]-cy)**2))

# steady Poiseuille velocity, right-to-left;  v = (v_x(y), 0)
def vfield(P):
    vx = -4.0*Vmag*P[:,1]*(1.0-P[:,1])          # max|v_x| = Vmag at y=1/2
    return np.stack([vx, np.zeros_like(vx)], axis=1)

# ---------------------------------------------------------------------------
# Manufactured solution (used when m == True).
#   u* = cos(2pi x) cos(pi y) :  periodic in x, d/dy = 0 at y=0,1  -> homogeneous
#   Neumann compatible.  Body forcing for the divergence operator:
#     f = (1/dt)u* - div(D grad u*) = (1/dt)u* - D Lap(u*) - (Dx u*_x + Dy u*_y)
#   This f is an analytic FULL-grid load (no numerical gradient needed), so it
#   exercises the reduce_body path directly and the recovered central traces
#   should match u* to spectral accuracy.
# ---------------------------------------------------------------------------
def ustar(P):
    return np.cos(2*np.pi*P[:,0])*np.cos(np.pi*P[:,1])

def manufactured_f(P):
    x, y = P[:,0], P[:,1]
    u   = np.cos(2*np.pi*x)*np.cos(np.pi*y)
    lap = -((2*np.pi)**2 + (np.pi)**2)*u
    uxg = -2*np.pi*np.sin(2*np.pi*x)*np.cos(np.pi*y)
    uyg = -np.pi*np.cos(2*np.pi*x)*np.sin(np.pi*y)
    return (1.0/dt)*u - D(x,y)*lap - (Dx(x,y)*uxg + Dy(x,y)*uyg)


# ===========================================================================
#   Body-load pipeline
#
#   full_load(disc, Uvec) -> f on the FULL leaf grid grid_xx
#       f = (1/dt) U^n  -  v . grad U^n               (advection lagged; Form 2)
#       grad U^n from the numerical field via leaf Chebyshev matrices Ds.
#
#   reduce_body_load(disc, f) -> (b_C, b_N)
#       per-leaf static condensation of the full-grid source (reduce_body),
#       returning the skeleton (C-space) and Neumann (N-space) reduced loads.
#
#   make_reduced_load(Ufunc) -> callable(disc) -> (b_C, b_N)
#       the COMPOSITION  reduce o f  fed to oms; applied per slab inside oms,
#       taking only the slab discretization disc (= the Domain_Driver), never
#       the coupled solve.  For step 0, Ufunc = U0 sampled on grid_xx.
# ===========================================================================
def _grid_and_Ds(disc):
    gx = disc.hps.grid_xx.numpy()                          # (nb, p^2, 2) full leaf grid
    D1 = disc.hps.H.Ds[3].numpy()                      # d/dx  (per-leaf, p^2 x p^2)
    D2 = disc.hps.H.Ds[4].numpy()                      # d/dy
    nb, pp2 = gx.shape[0], gx.shape[1]
    return gx, D1, D2, nb, pp2

def full_load(disc, Uvec):
    """f = (1/dt)U - v.gradU on the full leaf grid; Uvec is U^n at grid_xx (nb,p^2)."""
    gx, D1, D2, nb, pp2 = _grid_and_Ds(disc)
    Ui = Uvec.reshape(nb, pp2)
    ux = np.einsum('ij,bj->bi', D1, Ui)                # dU/dx per leaf
    uy = np.einsum('ij,bj->bi', D2, Ui)                # dU/dy per leaf
    P  = gx.reshape(nb*pp2, 2)
    vP = vfield(P).reshape(nb, pp2, 2)
    vdotgradU = vP[:,:,0]*ux + vP[:,:,1]*uy
    f = (1.0/dt)*Ui - vdotgradU                        # (1/dt)U - v.gradU
    return f.reshape(nb*pp2, 1)

def reduce_body_load(disc, fvec):
    """reduce_body: full-grid source -> (b_C on C-space, b_N on N-space)."""
    fvec_t = torch.as_tensor(fvec, dtype=torch.double)
    rb      = disc.hps.get_DtNs(CPU, mode='reduce_body', ff_body_vec=fvec_t).flatten(0,-2).numpy().real.ravel()
    Ic1     = disc.hps.I_copy1.numpy(); Ic2 = disc.hps.I_copy2.numpy()
    Iext    = disc.I_Xtot_in_unique.numpy()
    b_full_C = rb[Ic1] + rb[Ic2]                       # C-space (skeleton) reduced load
    b_full_X = rb[Iext]                                # X-space reduced load
    return b_full_C, b_full_X

def sample_on_grid(disc, Ufunc):
    """evaluate an analytic field Ufunc(P) on the full leaf grid grid_xx -> (nb*p^2,)."""
    gx = disc.hps.grid_xx.numpy()
    P  = gx.reshape(-1, 2)
    return Ufunc(P)

def reconstruct_on_grid(disc, uX_full, fvec):
    """Reconstruct U^{n+1} on the full leaf grid grid_xx.
    uX_full : exterior Dirichlet trace on I_Xtot (length nX) -- artificial faces carry the
              neighbour central traces, physical walls carry the SOLVED Neumann values from
              this slab's mixed solve (so the whole exterior is now known Dirichlet data).
    fvec    : the SAME full-grid body f^n used to form this step's rhs.
    Returns U on grid_xx as (nb, p^2).  Uses disc.solve_dir_full (full Dirichlet solve,
    interiors included)."""
    nb  = int(disc.hps.nboxes); pp2 = int(np.prod(disc.hps.p))
    uXt = torch.as_tensor(np.asarray(uX_full).reshape(-1, 1), dtype=torch.double)
    fvt = torch.as_tensor(np.asarray(fvec).reshape(-1, 1),  dtype=torch.double)
    flat = disc.solve_dir_full(uXt, ff_body=fvt)
    return flat.detach().cpu().numpy().reshape(nb, pp2)

def make_reduced_load(Ufunc):
    """Composition reduce o f as a per-slab callable disc -> (b_C, b_X).
    Returns the FULL X-space reduced load b_X; oms restricts it to the Neumann
    subset (solver.JN) since that index set lives at the wrapper level."""
    def reduced(disc):
        Uvec = sample_on_grid(disc, Ufunc)             # U^n on full grid (analytic for step 0)
        f    = full_load(disc, Uvec)                   # (1/dt)U - v.gradU on full grid
        b_C, b_X = reduce_body_load(disc, f)           # per-leaf condensation
        return b_C, b_X
    return reduced

def make_reduced_load_manufactured(ffunc):
    """Manufactured composition: f is the analytic full-grid load ffunc(P) (no numerical
    gradient), reduced via reduce_body. Used when m == True to validate the solver."""
    def reduced(disc):
        gx = disc.hps.grid_xx.numpy()
        f  = ffunc(gx.reshape(-1, 2)).reshape(-1, 1)
        b_C, b_X = reduce_body_load(disc, f)
        return b_C, b_X
    return reduced


# ---------------------------------------------------------------------------
# Snapshot plotting: smooth per-leaf (Chebyshev barycentric) interpolation of the
# full-grid field, one slab's left column per strip to avoid the overlap double-count.
# ---------------------------------------------------------------------------
def bary_mat(nodes, targets):
    """Barycentric interpolation matrix from Chebyshev-Lobatto `nodes` to `targets`.
    Row k maps nodal values -> value at targets[k].  Exact-node hits are handled
    so we never divide by zero."""
    n = len(nodes)
    wt = (-1.0)**np.arange(n)
    wt[0] *= 0.5
    wt[-1] *= 0.5
    M = np.zeros((len(targets), n))
    for k, t in enumerate(targets):
        d = t - nodes
        hit = np.abs(d) < 1e-13
        if np.any(hit):
            M[k, np.argmin(np.abs(d))] = 1.0
        else:
            wv = wt / d
            M[k] = wv / wv.sum()
    return M

def plot_snapshot(discs, Ufull, w, fname, title, dSlabs, nxp=16, nyp=16):
    """Smooth per-leaf (Chebyshev barycentric) render of the full-grid field over [0,1]^2.

    discs[i] : slab i's Domain_Driver (disc.hps.grid_xx is in GLOBAL coordinates,
               shape (nboxes, p^2, 2); slab i spans global x in [i*w, (i+2)*w]).
    Ufull[i] : (nboxes, p^2) nodal values for slab i, ordered like grid_xx.
    w        : strip width H (each slab is double-wide = 2*w).

    To avoid the overlap double-count, each strip is drawn from ONE slab's LEFT column
    (the boxes in that slab's own first half, x in [i*w, (i+1)*w]).  grid_xx is already
    global, so tiles are plotted at their true x with NO additional offset.
    """
    # ---- first pass: build all tiles, track global colour range ----
    tiles = []   # (xf, yf, tile2d)
    vmin, vmax = np.inf, -np.inf
    for i, disc in enumerate(discs):
        gx = disc.solver.hps.grid_xx.numpy()
        pp = int(disc.solver.hps.p[0])
        xl_i = float(dSlabs[i][0][0])                 # slab i left edge (global x)
        # left column = boxes in this slab's first half  [xl_i, xl_i + w]
        left = [b for b in range(gx.shape[0])
                if gx[b, :, 0].mean() < xl_i + w - 1e-9]
        left = sorted(left, key=lambda b: gx[b, :, 1].mean())
        for b in left:
            uxn = np.unique(np.round(gx[b, :, 0], 10))   # this box's x-nodes (global)
            uyn = np.unique(np.round(gx[b, :, 1], 10))   # this box's y-nodes
            ix = np.searchsorted(uxn, np.round(gx[b, :, 0], 10))
            iy = np.searchsorted(uyn, np.round(gx[b, :, 1], 10))
            U2 = np.zeros((pp, pp))
            U2[ix, iy] = Ufull[i][b]                     # nodal values -> (p,p) tensor
            # fine target grid inside this box (half-open so adjacent tiles don't overlap edges)
            xf = np.linspace(uxn[0], uxn[-1], nxp, endpoint=False)
            yf = np.linspace(uyn[0], uyn[-1], nyp, endpoint=False)
            tile = bary_mat(uxn, xf) @ U2 @ bary_mat(uyn, yf).T
            tiles.append((xf, yf, tile))
            vmin = min(vmin, tile.min())
            vmax = max(vmax, tile.max())

    # ---- second pass: draw with a single shared colour scale ----
    plt.figure(figsize=(6.4, 3.2))
    for xf, yf, tile in tiles:
        plt.pcolormesh(xf, yf, tile.T, shading='auto', cmap='viridis',
                       vmin=vmin, vmax=vmax)
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()
    return fname


# ===========================================================================
# Solution diagnostics on U^n  (per-slab full-grid nodal values, grid_xx order)
#   (1) conservation of mass : global integral of U over [0,1]^2
#   (2) Neumann test         : L2 and max of dU/dy on bottom (y=0) and top (y=1)
#   (3) periodicity test     : L2 and max of U(0,y) - U(1,y)
#
# Integrals use the leaf Chebyshev-Lobatto nodes with Clenshaw-Curtis weights.
# Overlap is resolved with the SAME ownership rule as the gluing/plotting:
# strip i (x in [i*H,(i+1)*H)) comes from slab i's LEFT half, and the wrapped
# strip N-1 from the last slab's RIGHT half -- so each physical point is counted
# exactly once.  (discs[i] are OMS.solvers wrappers; discs[i].solver is the
# Domain_Driver, discs[i].solver.hps.grid_xx is GLOBAL coordinates.)
# ===========================================================================
def _cc_weights(pp):
    """Clenshaw-Curtis weights on [-1,1] for pp Chebyshev-Lobatto nodes (ascending)."""
    n = pp - 1
    if n == 0:
        return np.array([0.0]), np.array([2.0])
    th = np.pi*np.arange(n+1)/n
    x  = np.cos(th)
    w  = np.zeros(n+1); ii = np.arange(1, n); v = np.ones(n-1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0/(n*n - 1)
        for k in range(1, n//2):
            v -= 2*np.cos(2*k*th[ii])/(4*k*k - 1)
        v -= np.cos(n*th[ii])/(n*n - 1)
    else:
        w[0] = w[n] = 1.0/(n*n)
        for k in range(1, (n-1)//2 + 1):
            v -= 2*np.cos(2*k*th[ii])/(4*k*k - 1)
    w[ii] = 2*v/n
    return x[::-1], w[::-1]                       # ascending nodes & matching weights

def _owned_xranges(i, nslab, w):
    """x-intervals slab i is responsible for (no double counting across the overlap)."""
    xl = i*w
    rngs = [(xl, xl + w)]                         # slab i's left half = strip i
    if i == nslab - 1:                            # last slab also owns the wrapped strip
        rngs.append((xl + w, xl + 2*w))
    return rngs

def _in_owned(xc, rngs):
    return any(r0 - 1e-9 <= xc < r1 - 1e-9 for (r0, r1) in rngs)


def diag_mass(discs, U, w):
    """(1) Global integral of U over [0,1]^2 (conservation of mass)."""
    nslab = len(discs); M = 0.0
    for i, disc in enumerate(discs):
        gx   = disc.solver.hps.grid_xx.numpy()
        rngs = _owned_xranges(i, nslab, w)
        for b in range(gx.shape[0]):
            xb = gx[b, :, 0]; yb = gx[b, :, 1]
            if not _in_owned(xb.mean(), rngs):
                continue
            uxn = np.unique(np.round(xb, 10)); uyn = np.unique(np.round(yb, 10))
            ix  = np.searchsorted(uxn, np.round(xb, 10))
            iy  = np.searchsorted(uyn, np.round(yb, 10))
            _, wx = _cc_weights(len(uxn)); _, wy = _cc_weights(len(uyn))
            dx = uxn[-1] - uxn[0]; dy = uyn[-1] - uyn[0]
            node_w = 0.25*dx*dy*wx[ix]*wy[iy]     # per-node 2-D quadrature weight
            M += float(np.sum(node_w*U[i][b]))
    return M


def run_diagnostics(discs, U, w, tag, mass_ref=None):
    """Print the three diagnostics for the field U on one line and return them."""
    M           = diag_mass(discs, U, w)
    drift = "" if mass_ref is None else "  (drift % .2e)" % (M - mass_ref)
    print("%s mass=% .8e%s "
          % (tag, M, drift))
    return dict(mass=M)


# ===========================================================================
# Build OMS (mixed = Neumann on physical walls, Dirichlet on artificial faces)
# ===========================================================================
formulation = "hpsalt"
opts = solverWrap.solverOptions(formulation, [p_disc, p_disc], a, 'mixed')
OMS  = oms.oms_lu(dSlabs, op,
                  lambda P: square.gb(P, jax_avail=jax_avail, torch_avail=torch_avail),
                  opts, connectivity)

if m:
    # manufactured: analytic full-grid load f = (1/dt)u* - div(D grad u*);
    # artificial vertical faces carry exact Dirichlet data u* (so the recovered
    # central traces can be compared to u*). Homogeneous Neumann on walls (bc=0).
    reduced_load = make_reduced_load_manufactured(manufactured_f)
    bc_use = lambda P: np.zeros_like(P[:,0])            # Dirichlet on artificial faces = u* ; walls Neumann handled separately
else:
    # transport step 0: U^n = U0 (analytic on grid); homogeneous Dirichlet coupling start.
    reduced_load = make_reduced_load(U0)
    bc_use = bc

S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc_use, reduced_load=reduced_load, dbg=0)
Stot, rhstot = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list, rhs_list, Ntot, nc, dbg=0)

gInfo = gmres_info()
stol  = 1e-10
if Version(scipy.__version__) >= Version("1.14"):
    uhat, info = gmres(Stot, rhstot, rtol=stol, callback=gInfo, maxiter=500, restart=500)
else:
    uhat, info = gmres(Stot, rhstot, tol=stol, callback=gInfo, maxiter=500, restart=500)

print("=============SUMMARY==============")
print("manufactured             = ", m)
print("H                        = ", '%10.3E' % H)
print("ord                      = ", p)
print("dt                       = ", '%10.3E' % dt)
print("nc                       = ", OMS.nc)
print("GMRES iters              = ", gInfo.niter)
print("GMRES info               = ", info)

# ---------------------------------------------------------------------------
# Manufactured-solution accuracy check: compare reduced unknowns uhat (the
# central interface traces, contiguous per slab) against u* at those points.
# ---------------------------------------------------------------------------
if m:
    err_tot = 0.0
    for slabInd in range(len(dSlabs)):
        geom   = np.array(dSlabs[slabInd])
        slab_i = oms.slab(geom, lambda P: square.gb(P, jax_avail, torch_avail))
        solver = solverWrap.solverWrapper(opts)
        solver.construct(geom, op, False, False)
        Il, Ir, Ic, Igb, XXi, XXb = slab_i.compute_idxs_and_pts(solver)
        XXc = XXi[Ic, :]
        XXc = XXc.detach().cpu().numpy() if hasattr(XXc, "detach") else np.asarray(XXc)
        uloc = uhat[slabInd*nc:(slabInd+1)*nc]
        uex  = ustar(XXc)
        e    = np.linalg.norm(uloc - uex)/np.linalg.norm(uex)
        err_tot = max(err_tot, e)
    print("manufactured rel-err     = ", '%10.3E' % err_tot)
print("==================================")


# ===========================================================================
# Timestepping (m == False): integrate U0 to Tfinal, snapshot near t=0.5, 1.0
# ===========================================================================
# This loop assumes two hooks on the OMS object (to be wired in oms.py):
#   OMS.discs            : list of per-slab Domain_Drivers (cached at first build)
#   OMS.uX_full(uhat, i, wallvals_i) : assemble slab i's exterior trace on I_Xtot
#                          (neighbour central traces on artificial faces via connectivity,
#                           solved Neumann wall values on the wall slots)
#   OMS.construct_rhstot(reduced_load) : rebuild ONLY rhstot against cached slabs
# Stot is time-independent and is assembled once, before the loop.
if not m:
    discs = OMS.solvers                      # per-slab Domain_Drivers (cached at first build)
    w     = H                              # strip width

    # initial state: U^0 sampled on each slab's full leaf grid
    U = [sample_on_grid(disc.solver, U0).reshape(int(disc.solver.hps.nboxes), int(np.prod(disc.solver.hps.p)))
         for disc in discs]

    # Stot is time-independent: assemble the coupling once, outside the loop
    Stot, _ = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list, rhs_list, Ntot, nc, dbg=0)

    # reference mass from the initial state (mass should stay ~constant under
    # divergence-form diffusion + divergence-free advection with no-flux walls)
    mass0 = diag_mass(discs, U, w)
    print("--- U^n diagnostics (mass / Neumann walls / x-periodicity) ---")
    run_diagnostics(discs, U, w, "[t=0.000 step   0]", mass_ref=mass0)

    snaps = []
    for n in range(Nsteps):
        # --- 1. per-slab body f^n and its reduction (computed ONCE, reused twice) ---
        fvecs  = [full_load(disc.solver, U[i]) for i, disc in enumerate(discs)]
        bredu  = [reduce_body_load(disc.solver, fvecs[i]) for i, disc in enumerate(discs)]  # [(b_C,b_X), ...]

        # --- 2. rebuild only the rhs from the precomputed reduced loads, then solve ---
        rhstot = OMS.construct_rhstot(bc_use, bredu)        # oms indexes bredu[slabInd]
        gI = gmres_info()
        if Version(scipy.__version__) >= Version("1.14"):
            uhat, _ = gmres(Stot, rhstot, rtol=stol, callback=gI, maxiter=500, restart=500)
        else:
            uhat, _ = gmres(Stot, rhstot, tol=stol, callback=gI, maxiter=500, restart=500)

        # --- 3. reconstruct U^{n+1} on each slab's full grid ---
        Unew = []
        for i, disc in enumerate(discs):
            b_C, b_X = bredu[i]
            uX_i = OMS.uX_full(uhat, i, b_C, b_X)           # exterior trace: faces + solved walls
            Unew.append(reconstruct_on_grid(disc.solver, uX_i, fvecs[i]))
        U = Unew

        # --- per-step diagnostics on U^{n+1} ---
        run_diagnostics(discs, U, w, "[t=%.3f step %3d]" % ((n + 1)*dt, n + 1),
                        mass_ref=mass0)

        # --- 4. snapshots near t = 0.5, 1.0 ---
        if (n + 1) in snap_steps:
            t  = (n + 1) * dt
            patches = G.assemble_global(discs, U, dSlabs, H, N)      # stage 2: glue once
            G.plot_patches(patches, "u_t%.1f.png" % t, "u(x,y) at t=%.2f" % t)  # stage 3: dumb pl
            print("snapshot t=%.2f  max|u|=%.3f"
                  % (t, max(np.abs(u).max() for u in U)) )
    print("timestepping done; snapshots:", snaps)