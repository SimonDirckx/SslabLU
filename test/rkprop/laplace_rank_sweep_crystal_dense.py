# basic packages
import numpy as np
import torch
import time

# oms packages  (same conventions as twistedTorus.py / thinSlab3D.py)
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import matAssembly.HBS.slabTree as slabTree

# PyTorch-based HPS (hpsmultidomain): PDO container with torch coefficient callables
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo

# HBS direct solvers (ThomasSolverHBS / RedBlackSolverHBS)
try:
    import direct_solve.omsdirectsolveHBS as omsdirectHBS
except ImportError:
    import multislab.omsdirectsolveHBS as omsdirectHBS

from scipy.sparse.linalg import LinearOperator

# geometry (2D unit square, sliced along x)
try:
    import geometry.geom_2D.square as square
except ImportError:
    import geometry.square as square


################################################################
#
#   SET-UP BVP:         Laplace on the unit square
#   - pdo               (-u_xx - u_yy = 0)
#   - BC / exact sol.   log|x - x0|, harmonic source x0 OUTSIDE domain
#
################################################################

# Torch-based hps ("hpsalt"); jax-based path ("hps") not used here
jax_avail   = False
torch_avail = not jax_avail
hpsalt      = not jax_avail

from scipy.special import j0       # 2D Helmholtz radial solution (Bessel J0)

# --- Photonic-crystal Helmholtz via a MANUFACTURED (exact) solution ---------
# A homogeneous variable-coefficient equation cannot have BOTH a freely chosen
# coefficient AND a freely chosen analytic solution -- they are locked by
# c = Delta u / u.  To keep an exact solution to compare against (and leave the
# oms interface pipeline, which is homogeneous-Dirichlet only, untouched), we
# manufacture a coefficient that is consistent with an analytic solution:
#
#   S(x)   = sum_c J0(kh |x - x_c|)      over the crystal lattice sites x_c
#   u*(x)  = A + S(x)                    (A chosen so u* > 0 everywhere)
#   c(x)   = -kh^2 * S(x) / (A + S(x))   = -kh^2 (1 - A/u*)
#
# Since Delta J0(kh|x-x_c|) = -kh^2 J0(kh|x-x_c|), we have Delta S = -kh^2 S, so
#   -Delta u* + c u* = kh^2 S - kh^2 S = 0   (exactly homogeneous, verified).
# c(x) is spatially varying, indefinite, and lattice-structured -- an
# interference pattern of Bessel waves seeded on the crystal sites -- which is
# what drives the interesting HBS rank growth, while u* stays analytic.
#
# NOTE: this keeps the crystal GEOMETRY (same lattice as bfield_crystal) but not
# its exact Gaussian-bump amplitude formula; that formula has no analytic
# solution.  To use the exact Gaussian b(x) you would run a manufactured-
# FORCING test (f = A[u_exact]) through a body load -- hpsmultidomain supports
# ff_body in solve_dir_full, but oms.construct_Stot_helper would need to
# aggregate it into the interface RHS.  Ask if you want that variant.

kh           = 80.0                  # background wavenumber (40 is ~6.4 wavelengths across [0,1]);
                                     # higher kh -> more oscillatory S-maps -> higher off-diagonal
                                     # rank -> the sweep and per-level spectrum become informative.
                                     # (If a run "never converges", nudge kh by +/-2 to dodge a
                                     #  near-resonance of the indefinite Dirichlet problem.)
crystal_start = 0.2
crystal_end   = 0.8
dist          = 0.05                 # lattice spacing (same as bfield_crystal)

_sites = np.array([(x, y)
                   for x in np.arange(crystal_start, crystal_end + dist, dist)
                   for y in np.arange(crystal_start, crystal_end, dist)])

def _crystal_S(P):
    """S(x) = sum over lattice sites of J0(kh |x - x_c|), on numpy points P."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    out = np.zeros(P.shape[0])
    for (xc, yc) in _sites:
        out += j0(kh * np.sqrt((P[:, 0] - xc)**2 + (P[:, 1] - yc)**2))
    return out

# Offset A: fix once on a fine grid so that min(u*) = |min S| (this bounds the
# coefficient to |c| <= kh^2 and keeps u* safely away from zero).
_gx = np.linspace(0., 1., 150)
_GX, _GY = np.meshgrid(_gx, _gx)
_Sgrid = _crystal_S(np.column_stack([_GX.ravel(), _GY.ravel()]))
A_off = -2.0 * _Sgrid.min()          # A + min(S) = |min S| > 0

def bc(p):
    """Exact manufactured solution u*(x) = A + S(x); also the Dirichlet data."""
    return A_off + _crystal_S(np.array(p))     # np.array(): accepts torch or numpy

def _crystal_c(xx):
    """Zeroth-order coefficient c(x) = -kh^2 S/(A+S), returned as a torch tensor
    on xx's device/dtype (leaf assembly calls this with a torch xx)."""
    P = xx.detach().cpu().numpy()
    S = _crystal_S(P)
    cval = -kh * kh * S / (A_off + S)
    return torch.as_tensor(cval, dtype=xx.dtype, device=xx.device)

# hpsmultidomain convention (hps_parallel_leaf_ops.get_Aloc_2d):
#   A[u] = -c11 u_11 - c22 u_22 - c12 u_12 + c1 u_1 + c2 u_2 + c u
# Laplacian part c11 = c22 = 1; the crystal enters through the zeroth-order c(x).
# (Name pdo_lap kept so the rest of Test 1 is untouched; now a crystal operator.)
pdo_lap = pdo.PDO_2d(c11=pdo.const(1.), c22=pdo.const(1.), c=_crystal_c)

gb = lambda p: square.gb(p, jax_avail=jax_avail, torch_avail=torch_avail)


################################################################
#
#   TEST 1:  rank sweep  rk = 10, 20, 30, ...
#   rk is used in BOTH places it appears in the pipeline:
#     (1) rkHMatAssembler(leaf_size, rk)  -> compression of the
#         slab S-maps (off-diagonal blocks of the interface system)
#     (2) ThomasSolverHBS(nc, rk)         -> compression of the
#         Schur complements S' formed during the factorization
#
################################################################

N = 33                               # slices -> N-1 = 16 interfaces (power of 2, RB-compatible)
dSlabs, connectivity, H = square.dSlabs(N)

formulation = 'hpsalt'
p = 16                               # high order so discretization err << 1e-8
p_disc = p + 2                       # hpsalt convention: q = p-2 interface nodes per face,
                                     # so +2 to match the jax-hps convention (cf. twistedTorus.py)
# DEEP TREE: fine y-mesh so the interface carries ~256 dofs -> HBS tree has
# ~8 leaves / 4 levels, giving the hierarchy room to show per-level rank growth.
# (a[1]=1/32 -> 16 y-panels -> nc ~ 256.  Use 1/64 for ~512 dofs / 16 leaves if
#  the per-slab MUMPS builds are affordable; 1/16 -> nc~128 if too slow.)
a = np.array([H / 4., 1. / 64.])     # per-dim panel half-widths (hpsalt wants an array):
                                     # x: 2H/(2*H/4) = 4 panels across the double slab
                                     # y: 1/(2*1/32) = 16 panels across the unit height
opts = solverWrap.solverOptions(formulation, [p_disc, p_disc], a)

leaf_size = 32 #4*p                       # HBS leaf size on the (1D line) interface (nc/leaf ~ 8 leaves)
rkvec = np.arange(10, 90, 10, dtype=np.int64)
target = 1e-14

# ---- interface points / exact interface solution (computed once) ----------
# interface of slab `slabInd` = its center line x=(slabInd+1)H, points XXi[Ic].
# compute_inverse=False skips setup_solver_Aii (the MUMPS factorization of
# A_ii), which is not needed just to read off indices and points.
IFpts = []
for slabInd in range(len(dSlabs)):
    geom = np.array(dSlabs[slabInd])
    slab_i = oms.slab(geom, gb)
    solver = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom, pdo_lap, compute_inverse=False)
    Il, Ir, Ic, Igb, XXi, XXb = slab_i.compute_idxs_and_pts(solver)
    IFpts += [np.array(XXi[Ic, :])]          # torch -> numpy copy
u_true = np.concatenate([bc(P) for P in IFpts])
# The manufactured solution u* = A + S has a large constant (DC) mode A, which
# would dominate ||u_true|| and deflate the relative error. Measure error
# against the OSCILLATORY content instead, so the rank sweep is informative.
u_ac_norm = np.linalg.norm(u_true - u_true.mean())

err   = np.zeros(shape=(len(rkvec),))
res   = np.zeros(shape=(len(rkvec),))
tvec  = np.zeros(shape=(len(rkvec),))
cmpr  = np.zeros(shape=(len(rkvec),))     # S-map compression rate (HBS bytes / dense bytes)
ncval = 0
for ind in range(len(rkvec)):
    rk = int(rkvec[ind])

    assembler = mA.rkHMatAssembler(leaf_size, rk)                 # (1)
    OMS = oms.oms(dSlabs, pdo_lap, gb, opts, connectivity)
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=1)
    Stot, rhstot = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list, rhs_list, Ntot, nc, dbg=0)

    tic = time.time()
    ds = omsdirectHBS.ThomasSolverHBS(nc, rk)                     # (2)
    ds.factorize(S_rk_list)
    uhat = ds.solve(rhstot.copy(), OMS.glob_target_dofs)
    tsolve = time.time() - tic

    # ---- RedBlackSolverHBS variant (needs N-1 = power of 2) ----------
    # NOTE: inside RedBlackSolverHBS.factorize there is a commented-out
    #       per-level rank increase (`rk = rk #+ 20`); changing that to
    #       `rk = rk + 10` is the "increase rank by 10 at every level"
    #       reading of the experiment.
    #tree0 = S_rk_list[1][0].tree
    #ds = omsdirectHBS.RedBlackSolverHBS(nc, rk, tree0, False)
    #ds.factorize(S_rk_list)
    #uhat = ds.solve(rhstot.copy())

    res[ind]  = np.linalg.norm(Stot @ uhat - rhstot) / np.linalg.norm(rhstot)
    err[ind]  = np.linalg.norm(uhat - u_true) / u_ac_norm   # error vs oscillatory content
    tvec[ind] = tsolve
    cmpr[ind] = OMS.stats.compression
    ncval     = nc

    print("=============SUMMARY==============")
    print("rk                       = ", rk)
    print("m (dofs per interface)   = ", nc)
    print("L2 rel. res              = ", '%10.3E' % res[ind])
    print("L2 rel. err vs exact     = ", '%10.3E' % err[ind])
    print("factorize+solve time     = ", '%10.3E' % tsolve, "s")
    print("==================================")

    if err[ind] < target:
        print("target %.1E reached at rk = %d" % (target, rk))
        rkvec = rkvec[:ind + 1]
        err   = err[:ind + 1]
        res   = res[:ind + 1]
        tvec  = tvec[:ind + 1]
        cmpr  = cmpr[:ind + 1]
        break

# ------------------------- TEST 1 FINAL TABLE ------------------------------
print("")
print("================== TEST 1: LAPLACE RANK SWEEP ==================")
print("domain / discretization : unit square, photonic-crystal Helmholtz (hpsalt)")
print("order                   : p = %d (p_disc = %d), a = [%.4E, %.4E]"
      % (p, p_disc, a[0], a[1]))
print("# slices (slabs)        : %d   (# double slabs / interfaces = %d)" % (N, N - 1))
print("dofs per interface (m)  : %d   (total interface dofs = %d)" % (ncval, (N - 1) * ncval))
print("HBS leaf size           : %d" % leaf_size)
print("rank step               : +%d per run" % (rkvec[1] - rkvec[0] if len(rkvec) > 1 else 0))
print("target error            : %.1E" % target)
print("----------------------------------------------------------------")
print("%6s | %12s | %12s | %10s | %8s" %
      ("rk", "rel err(AC)", "rel res", "solve [s]", "compr."))
print("----------------------------------------------------------------")
for ind in range(len(rkvec)):
    flag = "  <-- target" if err[ind] < target else ""
    print("%6d | %12.3E | %12.3E | %10.2E | %8.3f%s" %
          (rkvec[ind], err[ind], res[ind], tvec[ind], cmpr[ind], flag))
print("================================================================")
if err[-1] < target:
    print("smallest rank reaching %.1E : rk = %d" % (target, rkvec[-1]))
else:
    print("target %.1E NOT reached up to rk = %d "
          "(error plateau -> check discretization: raise p or lower a)"
          % (target, rkvec[-1]))
print("")

fileName = 'laplaceRankSweep.csv'
errMat = np.zeros(shape=(len(rkvec), 5))
errMat[:, 0] = rkvec
errMat[:, 1] = err
errMat[:, 2] = res
errMat[:, 3] = tvec
errMat[:, 4] = cmpr
with open(fileName, 'w') as f:
    f.write('rk,err,res,solve_time,compression\n')
    np.savetxt(f, errMat, fmt='%.16e', delimiter=',')


# ------------------------- TEST 1 PLOTS ------------------------------------
# Four diagnostics (convergence, per-interface error, time, compression) in
# one figure, plus a triptych of the interface solution field (exact /
# computed / pointwise error).  `uhat` still holds the solution at the last
# rank actually run (the best one, since the loop breaks on reaching target).
# Uses the Agg backend so it runs headless on a compute node and just writes
# PNGs; flip SHOW_PLOTS to also open windows interactively.
SHOW_PLOTS = False
try:
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    uhat_last = uhat                                   # solution at rkvec[-1]
    XY        = np.concatenate(IFpts, axis=0)          # (Ntot,2), matches u_true order
    abserr    = np.abs(uhat_last - u_true)
    tiny      = 1e-16                                 # guard log10(0)

    # ---- Figure 1: 2x2 diagnostics -------------------------------------
    figD, axD = plt.subplots(2, 2, figsize=(11, 8))

    # (0,0) convergence of error and residual vs rank
    axD[0, 0].semilogy(rkvec, err, 'o-', label='rel. err vs exact')
    axD[0, 0].semilogy(rkvec, res, 's--', label='rel. residual')
    axD[0, 0].axhline(target, color='k', ls=':', lw=1, label='target %.0E' % target)
    axD[0, 0].set_xlabel('HBS rank  rk'); axD[0, 0].set_ylabel('relative L2')
    axD[0, 0].set_title('rank-sweep convergence')
    axD[0, 0].grid(True, which='both', alpha=0.3); axD[0, 0].legend(fontsize=8)

    # (0,1) pointwise error along the middle interface (vs y)
    mid = (len(IFpts) - 1) // 2
    sl  = slice(mid * ncval, (mid + 1) * ncval)
    ys  = IFpts[mid][:, 1]; order = np.argsort(ys)
    axD[0, 1].semilogy(ys[order], np.maximum(abserr[sl][order], tiny), '.-')
    axD[0, 1].set_xlabel('y'); axD[0, 1].set_ylabel('|u_hat - u_exact|')
    axD[0, 1].set_title('error along interface %d  (x = %.3f, rk = %d)'
                        % (mid, (mid + 1) * H, rkvec[-1]))
    axD[0, 1].grid(True, which='both', alpha=0.3)

    # (1,0) factorize+solve time vs rank
    axD[1, 0].plot(rkvec, tvec, 'o-', color='C3')
    axD[1, 0].set_xlabel('HBS rank  rk'); axD[1, 0].set_ylabel('factorize + solve  [s]')
    axD[1, 0].set_title('cost vs rank'); axD[1, 0].grid(True, alpha=0.3)

    # (1,1) S-map compression vs rank
    axD[1, 1].plot(rkvec, cmpr, 's-', color='C2')
    axD[1, 1].set_xlabel('HBS rank  rk'); axD[1, 1].set_ylabel('S-map compression (HBS / dense)')
    axD[1, 1].set_title('compression vs rank'); axD[1, 1].grid(True, alpha=0.3)

    figD.suptitle('Test 1: photonic-crystal Helmholtz (hpsalt), rank sweep', fontsize=12)
    figD.tight_layout(rect=[0, 0, 1, 0.97])
    figD.savefig('test1_diagnostics.png', dpi=200)

    # ---- Figure 2: interface solution field (exact / computed / error) --
    figF, axF = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    s0 = axF[0].scatter(XY[:, 0], XY[:, 1], c=u_true,   s=10, cmap='viridis')
    axF[0].set_title('exact  u* = A + sum J0'); figF.colorbar(s0, ax=axF[0], shrink=0.85)
    s1 = axF[1].scatter(XY[:, 0], XY[:, 1], c=uhat_last, s=10, cmap='viridis')
    axF[1].set_title('computed   u_hat (rk = %d)' % rkvec[-1]); figF.colorbar(s1, ax=axF[1], shrink=0.85)
    s2 = axF[2].scatter(XY[:, 0], XY[:, 1], c=np.log10(np.maximum(abserr, tiny)), s=10, cmap='magma')
    axF[2].set_title('log10 |u_hat - u_exact|'); figF.colorbar(s2, ax=axF[2], shrink=0.85)
    for ax in axF:
        ax.set_xlabel('x'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    axF[0].set_ylabel('y')
    figF.suptitle('Test 1: interface solution sampled on the %d interfaces' % (N - 1), fontsize=12)
    figF.tight_layout(rect=[0, 0, 1, 0.95])
    figF.savefig('test1_field.png', dpi=200)

    print("wrote test1_diagnostics.png, test1_field.png")
    if SHOW_PLOTS:
        plt.show()
except Exception as e:
    print("plotting skipped:", e)


# --------- TEST 1 DIAGNOSTIC: per-level off-diagonal S-block spectrum -------
# Weak-admissibility HBS keeps the leaf DIAGONAL blocks dense and compresses
# only the OFF-DIAGONAL sibling blocks, level by level. This diagnostic
# assembles ONE interface S-block with NO compression (dense assembler), walks
# the same slabTree the assembler uses, and SVDs the sibling off-diagonal block
# B[child1, child2] at every internal node. Plotting sigma_k/sigma_1 per level
# shows where the rank actually lives: for an oscillatory (high-kh) operator the
# COARSE levels (few big boxes, near the root) carry the highest rank, so a
# single rank cap under-resolves them first -- that is the rank growth to show.
# This is an operator property; it does NOT depend on the (smooth) RHS, which is
# why the Test-1 solve can look cheap while the coarse-level blocks are not.
try:
    import numpy as np, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    dense_asm = mA.denseMatAssembler()
    OMSd = oms.oms(dSlabs, pdo_lap, gb, opts, connectivity)
    S_dense, _, _, ncd = OMSd.construct_Stot_helper(bc, dense_asm, dbg=0)

    mid  = len(S_dense) // 2
    Sblk = np.asarray(S_dense[mid][0])              # nc x nc dense off-diagonal S-map
    XXI  = IFpts[mid]                               # interface points, same order as Sblk
    tree = slabTree.slabTree(XXI, False, leaf_size) # same tree the assembler builds

    # full-block numerical rank (context: how much the DENSE diagonals inflate it)
    sv_full = np.linalg.svd(Sblk, compute_uv=False)
    def numrank(sv, eps):
        return int(np.sum(sv / sv[0] > eps))

    def box_inds(node):
        if node.point_inds is not None:
            return node.point_inds
        return np.concatenate([box_inds(c) for c in node.children])

    # sibling off-diagonal spectra, grouped by (child) level
    sv_by_level = defaultdict(list)
    for node in tree._boxes.values():
        if node.is_leaf or len(node.children) != 2:
            continue
        i1 = box_inds(node.children[0])
        i2 = box_inds(node.children[1])
        if i1 is None or i2 is None or i1.size == 0 or i2.size == 0:
            continue
        B  = Sblk[np.ix_(i1, i2)]                    # target child1  <-  source child2
        sv = np.linalg.svd(B, compute_uv=False)
        sv_by_level[node.children[0].level].append(sv)

    # --- plot: one worst-case envelope per level ---
    figS, axS = plt.subplots(figsize=(7, 5))
    for lvl in sorted(sv_by_level):
        svs = sv_by_level[lvl]
        m   = min(s.size for s in svs)
        env = np.max(np.array([s[:m] / s[0] for s in svs]), axis=0)   # worst block at this level
        axS.semilogy(np.arange(1, m + 1), env, 'o-', ms=3,
                     label='level %d  (%d blocks, size %d)' % (lvl, len(svs), m))
    for rkv in rkvec:                                # swept ranks, for reference
        axS.axvline(rkv, color='0.7', ls=':', lw=0.7)
    axS.axhline(target, color='k', ls='--', lw=0.8, label='target %.0E' % target)
    axS.set_xlabel('singular index k'); axS.set_ylabel(r'$\sigma_k/\sigma_1$  (off-diagonal block)')
    axS.set_title('Test 1: per-level off-diagonal S-block spectrum\n'
                  '(nc = %d, kh = %.0f, leaf = %d)' % (ncd, kh, leaf_size))
    axS.grid(True, which='both', alpha=0.3)
    axS.legend(fontsize=8, title='coarse (near root) -> fine (near leaves)')
    figS.tight_layout(); figS.savefig('test1_offdiag_spectrum.png', dpi=200)

    print("wrote test1_offdiag_spectrum.png")
    print("full-block numerical rank : %d @1e-6, %d @1e-8, %d @1e-10  (nc=%d)"
          % (numrank(sv_full, 1e-6), numrank(sv_full, 1e-8),
             numrank(sv_full, 1e-10), ncd))
    print("per-level worst-case off-diagonal rank @1e-8:")
    for lvl in sorted(sv_by_level):
        worst = max(numrank(s, 1e-8) for s in sv_by_level[lvl])
        blk   = min(s.size for s in sv_by_level[lvl])
        print("   level %d (block size %3d): rank %d" % (lvl, blk, worst))
    if SHOW_PLOTS:
        plt.show()
except Exception as e:
    print("off-diagonal spectrum diagnostic skipped:", e)


################################################################
#
#   TEST 2:  off-diagonal blocks EXACTLY HBS  vs  dense solve
#   Blocks are built with nested telescoping bases at rank r_star
#   (weak admissibility), so their compression at rk >= r_star is
#   exact; the difference to the dense reference isolates the
#   error of compressing the Schur complements + the ULV solve.
#   (This test is discretization-free: it does not touch hpsalt.)
#
################################################################

def build_exact_weak_hbs(L, m, r, rng):
    """
    Dense (2^L * m) x (2^L * m) matrix that is EXACTLY HBS at rank r under
    weak admissibility: every sibling off-diagonal block, at every level of
    the dyadic partition, is U S V^T with nested bases; dense fill only on
    the leaf diagonal blocks.  (Adapted from HBSnew_strong._build_exact_hbs_1d.)
    """
    nleaf = 2 ** L
    orth = lambda a_, b_: np.linalg.qr(rng.standard_normal((a_, b_)))[0]

    Uf = {}; Vf = {}
    for lvl in range(L, 0, -1):
        for box in range(2 ** lvl):
            sz = m if lvl == L else 2 * r
            Uf[(lvl, box)] = orth(sz, r); Vf[(lvl, box)] = orth(sz, r)
    hU = {}; hV = {}
    for box in range(nleaf):
        hU[(L, box)] = Uf[(L, box)]; hV[(L, box)] = Vf[(L, box)]
    for lvl in range(L - 1, 0, -1):
        for box in range(2 ** lvl):
            c1, c2 = 2 * box, 2 * box + 1
            h1, h2 = hU[(lvl + 1, c1)], hU[(lvl + 1, c2)]
            bU = np.zeros((h1.shape[0] + h2.shape[0], 2 * r))
            bU[:h1.shape[0], :r] = h1; bU[h1.shape[0]:, r:] = h2
            hU[(lvl, box)] = bU @ Uf[(lvl, box)]
            k1, k2 = hV[(lvl + 1, c1)], hV[(lvl + 1, c2)]
            bV = np.zeros((k1.shape[0] + k2.shape[0], 2 * r))
            bV[:k1.shape[0], :r] = k1; bV[k1.shape[0]:, r:] = k2
            hV[(lvl, box)] = bV @ Vf[(lvl, box)]

    def lr(box, lvl):
        span = 2 ** (L - lvl); a0 = box * span
        return a0 * m, (a0 + span) * m

    A = np.zeros((nleaf * m, nleaf * m))
    for lvl in range(1, L + 1):
        for P in range(0, 2 ** lvl, 2):                      # sibling pairs (P, P+1)
            for (i, j) in ((P, P + 1), (P + 1, P)):
                S = rng.standard_normal((r, r))
                r0, r1 = lr(i, lvl); c0, c1 = lr(j, lvl)
                A[r0:r1, c0:c1] = hU[(lvl, i)] @ S @ hV[(lvl, j)].T
    for b in range(nleaf):
        A[b * m:(b + 1) * m, b * m:(b + 1) * m] = rng.standard_normal((m, m))
    return A


def wrap_with_tree(A, tree):
    """Dense block -> LinearOperator carrying .tree/.quad (as ThomasSolverHBS expects)."""
    n = A.shape[0]
    lo = LinearOperator(shape=(n, n), dtype=A.dtype,
                        matvec=lambda v: A @ v, rmatvec=lambda v: A.T @ v,
                        matmat=lambda V: A @ V, rmatmat=lambda V: A.T @ V)
    lo.dense = A
    lo.tree = tree
    lo.quad = False
    return lo


print("===============  TEST 2: exact-HBS blocks vs dense  ===============")

L2 = 4                 # tree depth
m2 = 16                # leaf size
r_star = 10            # exact HBS rank of the off-diagonal blocks
n_if = 8              # number of interface blocks (power of 2)
mtot = (2 ** L2) * m2
rng = np.random.default_rng(0)
block_norm = 0.25

# cluster tree on synthetic 1D "interface" points (line mode of slabTree)
XXsynth = np.column_stack([np.arange(mtot, dtype=float), np.zeros(mtot)])
tree = slabTree.slabTree(XXsynth, False, m2)

def make_block():
    B = build_exact_weak_hbs(L2, m2, r_star, rng)
    return block_norm * B / np.linalg.norm(B, 2)       # keep ||S||<1/2: 2nd-kind system

S_rk_list = []
dense_blocks = []
for i in range(n_if):
    if i == 0:
        Sr = make_block(); S_rk_list += [[wrap_with_tree(Sr, tree)]]
        dense_blocks += [(None, Sr)]
    elif i == n_if - 1:
        Sl = make_block(); S_rk_list += [[wrap_with_tree(Sl, tree)]]
        dense_blocks += [(Sl, None)]
    else:
        Sl, Sr = make_block(), make_block()
        S_rk_list += [[wrap_with_tree(Sl, tree), wrap_with_tree(Sr, tree)]]
        dense_blocks += [(Sl, Sr)]

# dense reference:  M = I + off-diagonals,  exact solve
Ntot2 = n_if * mtot
M = np.eye(Ntot2)
for i in range(n_if):
    Sl, Sr = dense_blocks[i]
    if Sl is not None:
        M[i*mtot:(i+1)*mtot, (i-1)*mtot:i*mtot] = Sl
    if Sr is not None:
        M[i*mtot:(i+1)*mtot, (i+1)*mtot:(i+2)*mtot] = Sr

x_true = rng.standard_normal(Ntot2)
rhs2 = M @ x_true
x_dense = np.linalg.solve(M, rhs2)
print("dense solve err vs x_true = ", '%10.3E' %
      (np.linalg.norm(x_dense - x_true) / np.linalg.norm(x_true)))

rkvec2 = np.arange(r_star, 61, 10, dtype=np.int64)
err2   = np.zeros(shape=(len(rkvec2),))
errx2  = np.zeros(shape=(len(rkvec2),))
tvec2  = np.zeros(shape=(len(rkvec2),))
for ind in range(len(rkvec2)):
    rk = int(rkvec2[ind])
    tic = time.time()
    ds = omsdirectHBS.ThomasSolverHBS(mtot, rk)
    ds.factorize(S_rk_list)
    x_hbs = ds.solve(rhs2.copy())
    tvec2[ind] = time.time() - tic
    err2[ind]  = np.linalg.norm(x_hbs - x_dense) / np.linalg.norm(x_dense)
    errx2[ind] = np.linalg.norm(x_hbs - x_true)  / np.linalg.norm(x_true)
    print("rk = %d done (%.2E s)" % (rk, tvec2[ind]))

# ------------------------- TEST 2 FINAL TABLE ------------------------------
# rk - r_star = "extra" rank beyond what makes the S-block compression exact;
# any error at rk >= r_star is factorization error (Schur compl. + ULV), not
# S-map compression error.
print("")
print("============ TEST 2: EXACT-HBS BLOCKS vs DENSE SOLVE ============")
print("# interface blocks      : %d" % n_if)
print("block size (mtot)       : %d   (total dofs = %d)" % (mtot, Ntot2))
print("HBS tree                : L = %d levels, leaf size = %d" % (L2, m2))
print("exact block rank r*     : %d   (S compression exact for rk >= r*)" % r_star)
print("rank step               : +%d per run" % (rkvec2[1] - rkvec2[0] if len(rkvec2) > 1 else 0))
print("-----------------------------------------------------------------")
print("%6s | %8s | %14s | %14s | %10s" %
      ("rk", "rk - r*", "err vs dense", "err vs x_true", "solve [s]"))
print("-----------------------------------------------------------------")
for ind in range(len(rkvec2)):
    print("%6d | %8d | %14.3E | %14.3E | %10.2E" %
          (rkvec2[ind], rkvec2[ind] - r_star, err2[ind], errx2[ind], tvec2[ind]))
print("=================================================================")
print("(err vs dense at rk >= r* isolates Schur-complement compression")
print(" + ULV error; if it keeps dropping as rk grows past r*, the")
print(" factorization needs more rank than the blocks themselves.)")

fileName2 = 'exactHBSvsDense.csv'
errMat2 = np.zeros(shape=(len(rkvec2), 4))
errMat2[:, 0] = rkvec2
errMat2[:, 1] = err2
errMat2[:, 2] = errx2
errMat2[:, 3] = tvec2
with open(fileName2, 'w') as f:
    f.write('rk,err_vs_dense,err_vs_true,solve_time\n')
    np.savetxt(f, errMat2, fmt='%.16e', delimiter=',')
