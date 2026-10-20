# =============================================================================
# channel_neumann_oms_regression.py
#
# Regression test: Neumann (no-flux) y-walls in the oms stack, the way
# channel_barotropic_timestep.py uses them (SSLABLU_WALLS=neumann). The slab
# solvers carry the walls as Neumann faces of hpsmultidomain's Domain_Driver
# (solverOptions(bc_types=...)), so oms.oms needs no wall handling of its own.
#
# WHAT IT CHECKS, on a periodic channel (N = 8 cyclic double slabs, the layout
# of the channel script) with a variable-coefficient divergence-form operator
#     -div(D grad u) + u = f,   D(x, y) x-periodic,
# and a manufactured, x-periodic u whose outward du/dn on the walls is NOT
# zero (so the wall data path is exercised, not just a homogeneous condition):
#   * index bookkeeping: with Neumann walls each slab's interior set Ii is the
#     interior skeleton followed by the wall points, its boundary set Ib holds
#     only the two slab interfaces, and no wall point is a global-boundary
#     Dirichlet point (Igb is empty);
#   * the interface solution against the exact u at the interface points, and
#     the reconstructed leaf fields (solve_dir_full with the wall data) against
#     the exact u, for
#       dense S-maps     + dense cyclic red-black,
#       HBS S-maps (RK)  + dense cyclic red-black,
#       HBS S-maps (RK)  + HBS cyclic red-black (solver rank RB_RK),
#     with a Dirichlet-wall control on dense S-maps;
#   * solverOptions refuses bc_types where it cannot apply.
#
# Tolerances: TOL_DENSE for dense S-maps (discretization + round-off; ~2e-13
# is typical), TOL_HBS for compressed ones (~3e-10 is typical at the default
# ranks). Exit status: 0 if everything passes, 1 otherwise.
#
# Run (repo root):  python test/validation/channel_neumann_oms_regression.py
# =============================================================================

import io
import sys
import time
import warnings
from contextlib import redirect_stdout

import numpy as np
import torch

torch.set_default_dtype(torch.double)

import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
import solver.solver as solverWrap
import multislab.oms as oms
import matAssembly.matAssembler as mA
from direct_solve.omsdirectsolve import RedBlackSolver
with warnings.catch_warnings():
    # a docstring in it holds an invalid escape sequence (warns on recompile)
    warnings.simplefilter("ignore", SyntaxWarning)
    import direct_solve.omsdirectsolveHBS as omsdirectsolveHBS

N, P, NPAN_X, NPAN_Y = 8, 16, 4, 8          # the channel script's defaults
P_DISC = P + 2
RK, RB_RK = 48, P                           # S-map (assembler) and solver ranks
TOL_DENSE, TOL_HBS = 1e-11, 1e-8
NEUMANN_WALLS = {"x": "dirichlet", "y": "neumann"}

H = 1.0 / N
A = np.array([H / NPAN_X, 0.5 / NPAN_Y])
D_SLABS = [[[n * H - H, 0.0], [n * H + H, 1.0]] for n in range(N)]
CONNECTIVITY = [[(n - 1) % N, (n + 1) % N] for n in range(N)]


def gb(P_):
    """Global boundary = the y-walls (Dirichlet in the control run only)."""
    L = torch if torch.is_tensor(P_) else np
    return (L.abs(P_[:, 1]) < 1e-14) | (L.abs(P_[:, 1] - 1.0) < 1e-14)


# ---- operator: -div(D grad u) + u in hpsalt form (c1 = -D_x, c2 = -D_y) ------
ELL2 = 0.05
def _lib(P_): return torch if torch.is_tensor(P_) else np
def D(x, y, L):  return ELL2 * (1.0 - 0.4 * L.cos(2 * np.pi * x) * (1.0 + 0.5 * y))
def Dx(x, y, L): return ELL2 * 0.4 * 2 * np.pi * L.sin(2 * np.pi * x) * (1.0 + 0.5 * y)
def Dy(x, y, L): return -ELL2 * 0.4 * L.cos(2 * np.pi * x) * 0.5


OPERATOR = pdo.PDO_2d(c11=lambda P_: D(P_[:, 0], P_[:, 1], _lib(P_)),
                      c22=lambda P_: D(P_[:, 0], P_[:, 1], _lib(P_)),
                      c1=lambda P_: -Dx(P_[:, 0], P_[:, 1], _lib(P_)),
                      c2=lambda P_: -Dy(P_[:, 0], P_[:, 1], _lib(P_)),
                      c=pdo.const(c=1.0))


# ---- manufactured solution (x-periodic; du/dn != 0 on the walls) ------------
def u(x, y, L):   return L.sin(2 * np.pi * x) * L.exp(y) + L.cos(2 * np.pi * x) * y ** 2
def ux(x, y, L):  return (2 * np.pi * L.cos(2 * np.pi * x) * L.exp(y)
                          - 2 * np.pi * L.sin(2 * np.pi * x) * y ** 2)
def uy(x, y, L):  return L.sin(2 * np.pi * x) * L.exp(y) + 2 * L.cos(2 * np.pi * x) * y
def lap(x, y, L): return ((1 - 4 * np.pi ** 2) * L.sin(2 * np.pi * x) * L.exp(y)
                          + L.cos(2 * np.pi * x) * (2 - 4 * np.pi ** 2 * y ** 2))
def f(x, y, L):   return (-D(x, y, L) * lap(x, y, L) - Dx(x, y, L) * ux(x, y, L)
                          - Dy(x, y, L) * uy(x, y, L) + u(x, y, L))


def kept_slab(opts, n):
    """One slab's solver, indices, body load, wall data and interface rhs."""
    sv = solverWrap.solverWrapper(opts)
    with redirect_stdout(io.StringIO()):
        sv.construct(np.array(D_SLABS[n]), OPERATOR)
    Il, Ir, Ic, Igb, XXi, XXb = oms.slab(np.array(D_SLABS[n]), gb).compute_idxs_and_pts(sv)
    XXi, XXb = np.asarray(XXi), np.asarray(XXb)
    dd = sv.solver
    X = dd.XXfull
    fvec = f(X[:, 0], X[:, 1], torch).unsqueeze(-1)
    gN = None
    if dd.has_neumann:                         # outward du/dn at the wall points
        Pn = dd.XX_active[dd.I_Ntot]
        grad = torch.stack((ux(Pn[:, 0], Pn[:, 1], torch), uy(Pn[:, 0], Pn[:, 1], torch)), 1)
        gN = (grad * dd.normals_Ntot).sum(1, keepdim=True)
    fgb = u(XXb[Igb, 0], XXb[Igb, 1], np)      # Dirichlet walls (control run)
    b = dd.get_rhs(None, uu_dir_vec=torch.zeros(len(dd.I_Xtot), 1), ff_body_vec=fvec,
                   uu_neu_vec=gN).numpy()[:, 0]
    rhs = np.asarray(sv.solver_ii @ (b - sv.Aib[:, Igb] @ fgb)).ravel()[Ic]
    return dict(sv=sv, dd=dd, Il=Il, Ir=Ir, Ic=Ic, Igb=Igb, XXi=XXi, fvec=fvec,
                gN=gN, fgb=fgb, rhs=rhs)


def assemble(opts, assembler):
    OMS = oms.oms(D_SLABS, OPERATOR, gb, opts, CONNECTIVITY)
    with redirect_stdout(io.StringIO()):
        S_list, _, Ntot, nc = OMS.construct_Stot_helper(lambda P_: np.zeros(P_.shape[0]), assembler)
    return S_list, Ntot, nc


def interface_solve(S_list, nc, rhs, which):
    I_nc = np.eye(nc)
    if which == "rbhbs":
        tree0 = S_list[0][0].tree
        rb = omsdirectsolveHBS.RedBlackSolverHBS(nc, RB_RK, tree0, S_list[0][0].quad,
                                                 cyclic=True, seed=0)
        rb.factorize(S_list)
        return np.asarray(rb.solve(rhs)).ravel()
    rb = RedBlackSolver(nc, cyclic=True)
    with redirect_stdout(io.StringIO()):
        rb.factorize([[b if isinstance(b, np.ndarray) else np.asarray(b @ I_nc) for b in blk]
                      for blk in S_list], [I_nc] * N)
        return np.asarray(rb.solve(rhs)).ravel()


def errors(slabs, uhat, nc):
    """Interface error against u at the interface points, and the reconstructed
    leaf fields (collocation nodes, leaf corners excluded) against u."""
    ei, ri, el, rl = [], [], [], []
    for n, s in enumerate(slabs):
        xi = s["XXi"][s["Ic"]]
        ei.append(uhat[n * nc:(n + 1) * nc] - u(xi[:, 0], xi[:, 1], np))
        ri.append(u(xi[:, 0], xi[:, 1], np))
        kl, kr = CONNECTIVITY[n]
        g = np.zeros(len(s["sv"].Ib))
        g[s["Il"]] = uhat[kl * nc:(kl + 1) * nc]
        g[s["Ir"]] = uhat[kr * nc:(kr + 1) * nc]
        g[s["Igb"]] = s["fgb"]
        with redirect_stdout(io.StringIO()):
            full = s["dd"].solve_dir_full(torch.from_numpy(g[:, None]), ff_body=s["fvec"],
                                          uu_neu=s["gN"])[:, 0]
        X = s["dd"].XXfull
        nb, P2 = int(s["dd"].hps.nboxes), int(np.prod(s["dd"].hps.p))
        J = torch.as_tensor(s["dd"].hps.H.JJ.Jtot)
        ex = u(X[:, 0], X[:, 1], torch).reshape(nb, P2)[:, J]
        el.append((full.reshape(nb, P2)[:, J] - ex).ravel())
        rl.append(ex.ravel())
    e_if = np.linalg.norm(np.concatenate(ei)) / np.linalg.norm(np.concatenate(ri))
    e_lf = (torch.linalg.norm(torch.cat(el)) / torch.linalg.norm(torch.cat(rl))).item()
    return e_if, e_lf


tic_all = time.perf_counter()
np.random.seed(0)
torch.manual_seed(0)
failures, summary = [], []


def check(ok, label):
    summary.append(("PASS  " if ok else "FAIL  ") + label)
    if not ok:
        failures.append(label)


print("=============CHANNEL NEUMANN-WALL OMS REGRESSION=============")
print("operator            =  -div(D grad u) + u, x-periodic unit square")
print("slabs / p / panels  = ", N, "/", P, "/", NPAN_X, "x", NPAN_Y)
print("ranks (S-map, solver) = ", RK, "/", RB_RK)
print("tolerances          =  %.0e (dense S-maps), %.0e (HBS S-maps)" % (TOL_DENSE, TOL_HBS))
print("=============================================================")

# ---- solverOptions refuses bc_types where it cannot apply ---------------------
for kw, label in ((dict(type="hps"), "bc_types with type='hps'"),
                  (dict(problem_type="mixed"), "bc_types with problem_type='mixed'")):
    args = dict(type="hpsalt", ord=[P_DISC, P_DISC], a=A, bc_types=NEUMANN_WALLS)
    args.update(kw)
    try:
        solverWrap.solverOptions(**args)
        check(False, "solverOptions refuses %s" % label)
    except ValueError:
        check(True, "solverOptions refuses %s" % label)

for walls in ("neumann", "dirichlet"):
    opts = solverWrap.solverOptions("hpsalt", [P_DISC, P_DISC], A,
                                    bc_types=NEUMANN_WALLS if walls == "neumann" else None)
    tic = time.perf_counter()
    slabs = [kept_slab(opts, n) for n in range(N)]
    t_keep = time.perf_counter() - tic

    s0 = slabs[0]
    nI, nC, nN = len(s0["sv"].Ii), len(s0["dd"].I_Ctot), len(s0["dd"].I_Ntot)
    nB, nIl, nIr, nIgb = len(s0["sv"].Ib), len(s0["Il"]), len(s0["Ir"]), len(s0["Igb"])
    print("[%s walls] slab 0: |Ii| = %d (I_Ctot %d + I_Ntot %d), |Ib| = %d, |Il| = %d, "
          "|Ir| = %d, |Igb| = %d   (kept slabs %.1f s)"
          % (walls, nI, nC, nN, nB, nIl, nIr, nIgb, t_keep))
    if walls == "neumann":
        check(nN > 0 and nI == nC + nN and nB == nIl + nIr and nIgb == 0,
              "neumann walls: wall points in Ii, interfaces only in Ib, Igb empty")
    else:
        check(nN == 0 and nI == nC and nIgb > 0, "dirichlet walls: walls in Igb (control)")

    configs = [("dense", "rb")] if walls == "dirichlet" else \
        [("dense", "rb"), ("hbs", "rb"), ("hbs", "rbhbs")]
    built = {}
    for asm, slv in configs:
        if asm not in built:
            tic = time.perf_counter()
            assembler = mA.rkHMatAssembler(2 * P, RK) if asm == "hbs" else mA.denseMatAssembler()
            built[asm] = assemble(opts, assembler) + (time.perf_counter() - tic,)
        S_list, Ntot, nc, t_asm = built[asm]
        rhs = np.concatenate([s["rhs"] for s in slabs])
        tic = time.perf_counter()
        uhat = interface_solve(S_list, nc, rhs, slv)
        t_slv = time.perf_counter() - tic
        e_if, e_lf = errors(slabs, uhat, nc)
        tol = TOL_DENSE if asm == "dense" else TOL_HBS
        label = "%s walls, %s S-maps + %s: interface %.2e, leaf fields %.2e (tol %.0e)" % (
            walls, asm, "HBS red-black" if slv == "rbhbs" else "dense red-black", e_if, e_lf, tol)
        print("  %s   [assemble %.1f s, solve %.1f s]" % (label, t_asm, t_slv))
        check(e_if < tol and e_lf < tol, label)

print("")
print("=============SUMMARY=============")
for line in summary:
    print(line)
print("total time          =  %.1f s" % (time.perf_counter() - tic_all))
if failures:
    print("%d FAILED" % len(failures))
else:
    print("ALL PASSED")
print("=================================")
sys.exit(1 if failures else 0)
