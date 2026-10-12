# =============================================================================
# cyclic_rb_hbs_regression.py
#
# Regression test: RedBlackSolverHBS (direct_solve.omsdirectsolveHBS and
# direct_solve.omsdirectsolveHBS_torch) against the dense RedBlackSolver on a
# PERIODIC slab system, in the regime where the coarsest-level cyclic fold
# matters.
#
# THE BUG THIS GUARDS. At the last reduction of a periodic system the single
# retained node's left and right neighbours are the same eliminated node, so
# the would-be off-diagonals A_0 and C_0 couple node 0 to itself and its
# diagonal is B_0 + A_0 + C_0. The HBS builders used to factorize B_0 alone.
# The dropped terms span the whole period, so the error grows with the
# screening length ell relative to the period L. Measured on the barotropic
# channel operator before the fix: ~7e-12 at ell/L = 0.04 (invisible),
# 2.8e-2 at ell/L = 0.71, 4.2e-2 at ell/L = 2.85 -- at every solver rank,
# including full rank. The per-block debug probes could not see it: they
# checked B_0 against the same unfolded formula. Only an end-to-end comparison
# against a dense solve can.
#
# TEST OPERATOR. Constant-coefficient screened Poisson  -ell^2 Lap u + u  on the
# unit square, periodic in x (N = 8 cyclic double slabs, the layout of
# test/validation/channel_barotropic_timestep.py), Dirichlet on the y-walls.
# The fold does not depend on the channel's ridge, so it is left out. The
# S-maps are HBS-compressed (rkHMatAssembler), which both HBS modules require.
#
# DESIGN, AND WHY:
#   * Compare IN-PROCESS against the dense RedBlackSolver on the SAME S-blocks
#     (densified). Assembled operators are not bit-reproducible run to run
#     (~5e-10 spread even with seeds), so stored answers cannot be used.
#   * Tolerances are compression-level: TOL_RK at solver rank RK_LOW (half the
#     HBS leaf size, so the leaf level really compresses -- at rank >= leaf
#     size it compresses nothing), TOL_FULL at rank nc (every block exact).
#     Both sit orders of magnitude below the ~2e-2 a missing fold produces.
#     The S-map compression rank does not enter: both solvers see the same
#     compressed blocks.
#   * FOLD-SENSITIVITY GUARD: on the dense factorization, measure how wrong the
#     coarsest solve would be without the fold, and fail if that is not far
#     above the tolerances -- i.e. if the chosen ell/L could not catch the bug.
#   * Matrix: numpy + torch modules x fused/unfused builders x compress_diag
#     True/False x cyclic/open x solver ranks {RK_LOW, nc}. The open
#     (non-periodic) cases guard against collateral changes.
#   * KNOWN ISSUE, reported but not failed: torch fused + compress_diag=False
#     crashes ("'numpy.ndarray' object has no attribute 'sub_'") whether or not
#     the system is periodic; it has nothing to do with the fold. It runs as an
#     expected failure, and the script says so if it starts passing.
#   * Torch debug probes (debug_blocks > 0) on a periodic factorization: the
#     folded B_0 must check at compression level, A_0 / C_0 must be recorded
#     as folded, and no zero-slot warning may fire.
#
# Exit status: 0 if everything passes, 1 otherwise.
#
# Run:  python test/direct-solve/cyclic_rb_hbs_regression.py        (~10 s, CPU)
# Env override:
#   RBHBS_ELL   comma-separated ell/L values to test   (default "1.0,4.0")
# =============================================================================

import io
import os
import sys
import time
import warnings
from contextlib import redirect_stdout

import numpy as np
import torch

torch.set_default_dtype(torch.double)

# omsdirectsolveHBS*.py carry docstrings with invalid escape sequences, which
# only produce compile-time SyntaxWarning noise on stderr
warnings.filterwarnings("ignore", category=SyntaxWarning)
# the full-rank case (rank nc >= leaf size) is deliberate: it makes every block
# exact, which is exactly what the torch module's rank-schedule check warns about
warnings.filterwarnings("ignore", message=r"rkStrat \(RedBlackSolverHBS\): rank reaches")

import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
from direct_solve.omsdirectsolve import RedBlackSolver
import direct_solve.omsdirectsolveHBS as hbs_np
import direct_solve.omsdirectsolveHBS_torch as hbs_torch


################################################################
#
#   SETTINGS
#
################################################################

P        = 12        # polynomial order, p_disc = p + 2 (channel default)
N        = 8         # cyclic double slabs; red-black needs a power of 2
NPAN_X   = 4         # x-panels per double slab
NPAN_Y   = 8         # y-panels
RK_S     = 12        # assembler (S-map) rank -- NOT the solver rank
ELLS     = [float(v) for v in os.environ.get("RBHBS_ELL", "1.0,4.0").split(",") if v]

LEAF     = 2 * P     # HBS leaf size used by rkHMatAssembler
RK_LOW   = LEAF // 2 # rank-limited solver case, below the leaf size

TOL_RK   = 1e-7      # solver rank RK_LOW (observed <= 1e-9)
TOL_FULL = 1e-12     # solver rank nc     (observed ~3e-15)
FOLD_MIN = 1e-4      # the missing fold must be this visible, >> TOL_RK
SEED     = 0


################################################################
#
#   OPERATOR: screened Poisson on the x-periodic unit square
#
################################################################

def cyclic_dSlabs(N):
    """N double slabs; slab n spans [n*H - H, n*H + H] and is centred on
    interface n.  Slab 0 straddles the seam at x = 0; the coefficients are
    constant, so the fictitious extension to x < 0 is automatic."""
    H = 1.0 / N
    dSlabs = [[[n * H - H, 0.0], [n * H + H, 1.0]] for n in range(N)]
    connectivity = [[(n - 1) % N, (n + 1) % N] for n in range(N)]
    return dSlabs, connectivity, H


def gb(p):
    """Global boundary = the y-walls only (x is periodic)."""
    lib = torch if torch.is_tensor(p) else np
    return (lib.abs(p[:, 1]) < 1e-14) | (lib.abs(p[:, 1] - 1.0) < 1e-14)


dSlabs, connectivity, H = cyclic_dSlabs(N)
opts = solverWrap.solverOptions("hpsalt", [P + 2, P + 2],
                                np.array([H / NPAN_X, 0.5 / NPAN_Y]))


def assemble(ell):
    """HBS-compressed S-blocks of  -ell^2 Lap u + u  (interface system I + S)."""
    diff_op = pdo.PDO_2d(c11=pdo.const(c=ell * ell), c22=pdo.const(c=ell * ell),
                         c=pdo.const(c=1.0))
    OMS = oms.oms(dSlabs, diff_op, gb, opts, connectivity)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    with redirect_stdout(io.StringIO()):
        S_list, _, Ntot, nc = OMS.construct_Stot_helper(
            lambda p: np.zeros(p.shape[0]), mA.rkHMatAssembler(LEAF, RK_S), dbg=0)
    return S_list, Ntot, nc


################################################################
#
#   SOLVERS
#
################################################################

def dense_solve(S_dense, cyclic, rhs):
    d = RedBlackSolver(S_dense[0][0].shape[0], cyclic=cyclic)
    d.factorize(S_dense)                         # T=None: identity diagonal
    return d, np.asarray(d.solve(rhs)).ravel()


def fold_sensitivity(d, nc):
    """Relative error the coarsest solve would make WITHOUT the fold, measured
    on the dense cyclic factorization:  ||(Bf - F)^-1 v - Bf^-1 v|| / ||Bf^-1 v||
    with Bf = B_0 + A_0 + C_0 (what the dense solver factorizes) and F = A_0 + C_0."""
    top = len(d.RB) - 1
    I = np.eye(nc)
    Bf = np.asarray(d.get_block_op(top, 0, 'diag').matmat(I))
    F = (np.asarray(d.get_block_op(top, 0, 'sub').matmat(I))
         + np.asarray(d.get_block_op(top, 0, 'super').matmat(I)))
    v = np.random.default_rng(SEED + 1).standard_normal(nc)
    x_fold = np.linalg.solve(Bf, v)
    return np.linalg.norm(np.linalg.solve(Bf - F, v) - x_fold) / np.linalg.norm(x_fold)


def hbs_solve(module, S_list, rhs, nc, rk, **kw):
    """Factorize with RedBlackSolverHBS from `module` and solve; mutes the
    torch module's per-level / per-node progress prints."""
    if module is hbs_torch:
        kw["device"] = "cpu"
    hb = module.RedBlackSolverHBS(nc, rk, S_list[0][0].tree, S_list[0][0].quad,
                                  seed=SEED, **kw)
    with redirect_stdout(io.StringIO()):
        hb.factorize(S_list)
        x = np.asarray(hb.solve(rhs)).ravel()
    return hb, x


def rel(x, ref):
    return np.linalg.norm(x - ref) / np.linalg.norm(ref)


################################################################
#
#   TEST MATRIX
#
################################################################

MODULES = [("numpy", hbs_np), ("torch", hbs_torch)]
failures = []          # (ell, description)
summary  = []          # per-ell summary lines
tic_all  = time.perf_counter()

print("=============CYCLIC RED-BLACK HBS REGRESSION=============")
print("operator            =  -ell^2 Lap u + u, x-periodic unit square")
print("slabs / p / panels  = ", N, "/", P, "/", NPAN_X, "x", NPAN_Y)
print("ell/L values        = ", ELLS)
print("solver ranks        =  %d (below leaf size %d), nc (exact)" % (RK_LOW, LEAF))
print("tolerances          =  %.0e (rank %d), %.0e (rank nc), fold guard >= %.0e"
      % (TOL_RK, RK_LOW, TOL_FULL, FOLD_MIN))
print("=========================================================")

for ell in ELLS:
    tic = time.perf_counter()
    S_list, Ntot, nc = assemble(ell)
    I_nc = np.eye(nc)
    S_dense = [[np.asarray(b @ I_nc) for b in blk] for blk in S_list]
    rhs = np.random.default_rng(SEED + 2).standard_normal(Ntot)

    ref, dense = {}, {}
    for cyclic in (True, False):
        dense[cyclic], ref[cyclic] = dense_solve(S_dense, cyclic, rhs)

    fs = fold_sensitivity(dense[True], nc)
    print("")
    print("ell/L = %g   nc = %d   Ntot = %d   assemble %.1f s" % (ell, nc, Ntot,
                                                                 time.perf_counter() - tic))
    print("  fold sensitivity (coarsest solve without the fold) = %.2e" % fs)
    if not fs >= FOLD_MIN:
        failures.append((ell, "fold guard: sensitivity %.2e < %.0e, so this ell/L "
                              "cannot detect a missing fold" % (fs, FOLD_MIN)))

    print("  %-6s %-8s %-6s %-6s %4s | %-9s %-7s | %s"
          % ("module", "builder", "diag", "BC", "rk", "rel err", "tol", "status"))
    worst = {}          # (BC, rank class) -> worst error over passing cases
    n_pass = n_xfail = 0
    for mname, module in MODULES:
        for fused in (True, False):
            for compress_diag in (True, False):
                known_issue = module is hbs_torch and fused and not compress_diag
                for cyclic in (True, False):
                    for rk in (RK_LOW, nc):
                        tol = TOL_FULL if rk >= nc else TOL_RK
                        desc = "%-6s %-8s %-6s %-6s %4d" % (
                            mname, "fused" if fused else "unfused",
                            "cmp" if compress_diag else "linop",
                            "cyclic" if cyclic else "open", rk)
                        try:
                            _, x = hbs_solve(module, S_list, rhs, nc, rk, cyclic=cyclic,
                                             fused=fused, compress_diag=compress_diag)
                        except Exception as e:
                            msg = "%s: %s" % (type(e).__name__, str(e)[:70])
                            if known_issue and "sub_" in str(e):
                                n_xfail += 1
                                print("  %s | %-9s %-7s | XFAIL (known issue)" % (desc, "-", "-"))
                            else:
                                failures.append((ell, desc + " raised " + msg))
                                print("  %s | %-9s %-7s | FAIL  %s" % (desc, "-", "-", msg))
                            continue
                        err = rel(x, ref[cyclic])
                        if err <= tol:
                            n_pass += 1
                            key = ("cyclic" if cyclic else "open",
                                   "full" if rk >= nc else "rk")
                            worst[key] = max(worst.get(key, 0.0), err)
                            status = "pass"
                            if known_issue:
                                status = ("XPASS -- the known torch fused + compress_diag="
                                          "False issue looks fixed; remove its known_issue "
                                          "exemption")
                        else:
                            failures.append((ell, "%s rel err %.2e > %.0e" % (desc, err, tol)))
                            status = "FAIL"
                        print("  %s | %.2e  %.0e   | %s" % (desc, err, tol, status))

    summary.append("ell/L = %-5g fold sens. %.2e | %d pass, %d xfail | worst cyclic "
                   "%.1e / %.1e, open %.1e / %.1e  (rank < nc / rank = nc)"
                   % (ell, fs, n_pass, n_xfail,
                      worst.get(("cyclic", "rk"), np.nan), worst.get(("cyclic", "full"), np.nan),
                      worst.get(("open", "rk"), np.nan), worst.get(("open", "full"), np.nan)))

    # ---- torch debug probes on a periodic factorization (first ell only) ----
    if ell == ELLS[0]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hb, _ = hbs_solve(hbs_torch, S_list, rhs, nc, RK_LOW, cyclic=True, fused=True,
                              debug_blocks=16, debug_true_inverse=True)
        top = [r for r in hb.blockErrors if r["nSlabs"] == 2]
        B0 = [r for r in top if r["kind"] == "B"]
        folded = sorted(r["kind"] for r in top
                        if r["kind"] in ("A", "C") and "folded" in (r.get("note") or ""))
        zero_warn = [w for w in caught if "stored as zero_op" in str(w.message)]
        B0_err = max((B0[0].get(k) or 0.0) for k in ("fwd", "adj", "inv_ref")) if B0 else np.inf
        ok = (len(B0) == 1 and B0_err <= TOL_RK and folded == ["A", "C"] and not zero_warn)
        print("  torch debug probes: B_0 worst of fwd/adj/inv_ref = %.2e, "
              "A_0/C_0 recorded as folded: %s, zero-slot warnings: %d  -> %s"
              % (B0_err, folded == ["A", "C"], len(zero_warn), "pass" if ok else "FAIL"))
        if not ok:
            failures.append((ell, "torch debug probes are not fold-aware"))


################################################################
#
#   SUMMARY
#
################################################################

print("")
print("=============SUMMARY=============")
for line in summary:
    print(line)
print("total time          =  %.1f s" % (time.perf_counter() - tic_all))
if failures:
    print("RESULT              =  FAIL (%d)" % len(failures))
    for ell, what in failures:
        print("  ell/L = %g: %s" % (ell, what))
else:
    print("RESULT              =  PASS")
print("=================================")
sys.exit(1 if failures else 0)
