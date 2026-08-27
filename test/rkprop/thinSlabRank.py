"""HBS approximation of the interface maps of a SINGLE thin slab.

Two things changed with respect to the original thin-slab driver:

(1) The two discretizations are no longer two separate code paths.  Everything
    downstream only needs (Sii, Sib, XYtot, Ii, Ib) and a pair of quadrature
    weight vectors, so 'SOMS' and 'stencil' are now just two branches of the
    single assembly routine build_operator() and share the rest of the script.

(2) The balance operator, the GMRES solve and the timing study are gone.  We
    build one slab [0,2Lx] x [0,Ly] x [0,Lz] with the interface plane at
    x = Lx, form the two interface maps

        T_{l/r} = -( Sii^{-1} Sib[:, J_{l/r}] )   restricted to the plane x = Lx

    compress each of them into an HBS matrix, and report the relative error of
    the compression on a random test block, as a function of the HBS rank and
    the slab width Lx.
"""

import numpy as np

import SOMS3D_csr
import torch
import matAssembly.HBS.slabTree as slabTree
import matAssembly.HBS.HBStorch_strong as HBStorch_strong
import matAssembly.HBS.HBStorch as HBStorch
import solver.stencil.stencilSolver as stencil
import solver.stencil.geom as geom
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
import mumps
import scipy.sparse as sparse
import time
import os
import gc
import argparse
import re


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def bc_helmholtz(p, kh):
    """Free-space Green's function with source at (-0.5, -0.5, -0.5)."""
    r = np.sqrt((p[:, 0] + .5)**2 + (p[:, 1] + .5)**2 + (p[:, 2] + .5)**2)
    return np.real(np.exp(1j*kh*r)/(4*np.pi*r))


def match_rows(A, B, decimals=9):
    """For each row of A return the index of the matching row in B.

    Used to line up DOF orderings by coordinate: the physical x = 0 / x = 2*Lx
    faces against the interface plane (matched on (y,z) only), and the strictly
    interior interface points against their positions inside the full plane.
    A's rows must all occur in B (matched to `decimals` places)."""
    lut = {}
    for j in range(B.shape[0]):
        lut[tuple(np.round(B[j], decimals))] = j
    idx = np.empty(A.shape[0], dtype=np.int64)
    for i in range(A.shape[0]):
        idx[i] = lut[tuple(np.round(A[i], decimals))]
    return idx


# ICNTL(1) error stream, (2) diagnostic/warning, (3) global info, (4) verbosity
_ICNTL_OUT_STREAMS = (1, 2, 3, 4)

# ICNTL(9):  1 -> solve A x = b   (default)
#            0 -> solve A^T x = b against the same factors
_ICNTL_TRANSPOSE = 9
_ICNTL_SPARSE_RHS = 20      # ICNTL(20): 1 = sparse RHS
_ICNTL_BLOCK_SIZE = 27      # ICNTL(27): blocking size for multiple RHS
_ICNTL_BLR = 35             # ICNTL(35): block low-rank
_CNTL_BLR_TOL = 7      


def _silence_mumps(ctx):
    """Mute MUMPS' internal output streams (ICNTL(1..4))."""
    for i in _ICNTL_OUT_STREAMS:
        ctx.mumps_instance.icntl[i] = 0

def _enable_blr(ctx, blr_tol):
    """Block-low-rank factorization with dropping tolerance `blr_tol`."""
    ctx.mumps_instance.icntl[_ICNTL_BLR] = 1
    ctx.mumps_instance.cntl[_CNTL_BLR_TOL] = blr_tol
    # ICNTL(36) selects the BLR variant (UFSC); leave at default unless tuning.

def setup_mumps(A, ordering="metis", blr_tol=0.0, block_size=None, verbose=0):
    """Analyze + factorize A in one MUMPS context.

    Returns (ctx, time_analysis, time_factor).

    The context is left configured for forward solves; use `mumps_solve(ctx, b,
    transpose=True)` for A^{-T} applies against the same factors.

    Parameters
    ----------
    ordering : MUMPS ordering for the analysis ('metis', 'auto', 'amd', ...).
        Falls back to 'auto' if the requested ordering is not in this MUMPS
        build.
    blr_tol : if > 0, enable BLR compression with this dropping tolerance.
    block_size : ICNTL(27), the blocking size for multiple right-hand sides.
        Setting this to the number of columns you sample with gives one wide
        BLAS-3 block per chunk.
    verbose : >1 leaves MUMPS' own diagnostics on.
    """
    ctx = mumps.Context(verbose=bool(verbose > 1))
    # symmetric=False throughout: the symmetric/Cholesky path is deliberately
    # not used here.
    ctx.set_matrix(A, symmetric=False)

    tic = time.time()
    try:
        ctx.analyze(ordering=ordering)
    except Exception:
        if ordering == "auto":
            raise
        # e.g. METIS not compiled into this MUMPS build
        ctx.analyze(ordering="auto")
    time_analysis = time.time() - tic

    if verbose < 2:
        _silence_mumps(ctx)
    if blr_tol and blr_tol > 0:
        _enable_blr(ctx, blr_tol)

    tic = time.time()
    # reuse_analysis=True is the whole point of having called analyze():
    # factor() re-runs the analysis by default, so without this flag the
    # symbolic step is paid twice.
    ctx.factor(reuse_analysis=True)
    time_factor = time.time() - tic

    if verbose < 2:
        _silence_mumps(ctx)
    if block_size is not None:
        ctx.mumps_instance.icntl[_ICNTL_BLOCK_SIZE] = int(block_size)

    return ctx, time_analysis, time_factor

def setup_mumps_transpose(Sii, blr=False, blr_tol=1e-8, ordering="metis"):
    ctxT = mumps.Context()
    ctxT.analyze(Sii.T, ordering=ordering)
    if blr:
        ctxT.mumps_instance.icntl[35] = 1
        ctxT.mumps_instance.cntl[7] = blr_tol
    ctxT.analyze(Sii.T, ordering=ordering)
    ctxT.factor(Sii.T)
    return ctxT


def release_mumps(c):
    """Drop the native MUMPS memory: explicit teardown if the wrapper has one,
    otherwise release the handle so its finalizer frees the factors."""
    fin = getattr(c, "destroy", None) or getattr(c, "finalize", None)
    if callable(fin):
        try:
            fin()
        except Exception:
            pass
    else:
        try:
            c.mumps_instance = None
        except Exception:
            pass


def mumps_solve(ctx, b, transpose=False):
    """Solve A x = b, or A^T x = b, against an existing factorization.

    No context manager: ICNTL(9) is flipped, the solve runs, and the flag is
    restored in a finally block.  Reentrancy is the same as the original
    contextmanager version (i.e. do not interleave two transposed solves on
    one context from different threads).
    """
    if not transpose:
        return ctx.solve(b)

    inst = ctx.mumps_instance
    prev = inst.icntl[_ICNTL_TRANSPOSE]
    inst.icntl[_ICNTL_TRANSPOSE] = 0          # A^T x = b
    try:
        return ctx.solve(b)
    finally:
        inst.icntl[_ICNTL_TRANSPOSE] = prev   # restore A x = b

def mumps_solve_sparse(ctx, b, transpose=False):
    """As `mumps_solve` but for a sparse (csc) right-hand side.

    python-mumps' _solve_dense already resets ICNTL(20), so no manual reset is
    needed between sparse and dense solves.
    """
    if not transpose:
        return ctx._solve_sparse(b)

    inst = ctx.mumps_instance
    prev = inst.icntl[_ICNTL_TRANSPOSE]
    inst.icntl[_ICNTL_TRANSPOSE] = 0
    try:
        return ctx._solve_sparse(b)
    finally:
        inst.icntl[_ICNTL_TRANSPOSE] = prev


def _ints(tokens, n=None):
    """Parse ints given either space- or comma-separated."""
    flat = [t for t in re.split(r"[,\s]+", " ".join(tokens).strip()) if t]
    if n is not None and len(flat) != n:
        raise SystemExit(f"expected {n} values (got {flat})")
    return [int(t) for t in flat]


def _nums(tokens, n=None):
    """Parse floats; each entry may be a fraction like '1/16'."""
    flat = [t for t in re.split(r"[,\s]+", " ".join(tokens).strip()) if t]
    if n is not None and len(flat) != n:
        raise SystemExit(f"expected {n} values (got {flat})")
    out = []
    for t in flat:
        if "/" in t:
            a, b = t.split("/")
            out.append(float(a)/float(b))
        else:
            out.append(float(t))
    return out


# ---------------------------------------------------------------------------
#  command line
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="HBS approximation error of a single thin-slab interface map.")
_parser.add_argument("--type", choices=["SOMS", "stencil"], default="stencil",
                     help="discretization")
_parser.add_argument("--order", nargs="+", default=None,
                     help="(nx ny nz) for stencil [nx is overridden = int(2*ny*Lx)+1] "
                          "or (px py pz) for SOMS; space- or comma-separated")
_parser.add_argument("--widths", nargs="+", default=["1/16"],
                     help="slab half-widths Lx to sweep (fractions like 1/16 allowed); "
                          "the slab is [0,2Lx] with the interface at x = Lx")
_parser.add_argument("--shape", nargs="+", default=["1", "1"],
                     help="transverse extents Ly Lz")
_parser.add_argument("--rank", dest="rk", nargs="+", default=["20", "40", "60", "80", "100"],
                     help="HBS compression ranks to sweep")
_parser.add_argument("--admissibility", choices=["full", "partial", "weak"], default="weak",
                     help="HBS tree adjacency / admissibility")
_parser.add_argument("--nleaf", dest="nleaf", type=int, default=64,
                     help="number of DOFs per leaf")
_parser.add_argument("--ntest", dest="ntest", type=int, default=10,
                     help="number of random test vectors for the error estimate")
_parser.add_argument("--blr", dest="blr", type=float, default=0.,
                     help="BLR tolerance for the LU (0 = no BLR)")
_parser.add_argument("--nb", dest="nb", type=int, default=16,
                     help="SOMS number of blocks per unit length")
_parser.add_argument("--kh", dest="kh", type=float, default=0.,
                     help="wavenumber")
_parser.add_argument("--weighted", dest="weighted", action="store_true", default=False,
                     help="use the approximately L^2-weighted SOMS operator "
                          "(no effect on stencil)")
args = _parser.parse_args()

solve_method = args.type
order = _ints(args.order, 3) if args.order is not None else \
    ([9, 128, 128] if solve_method == "stencil" else [8, 8, 8])
widths = _nums(args.widths)
Ly, Lz = _nums(args.shape, 2)
ranks = _ints(args.rk)
admissibility = args.admissibility
nleaf = args.nleaf
ntest = args.ntest
blr_tol = args.blr
blr = blr_tol > 0
nb = args.nb
kh = args.kh
weighted = args.weighted

BLK = 32                     # solve block width (bounds the dense RHS memory)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
csv_path = "resultsHBSslab.csv"

# number of neighbour blocks a leaf can see -> how many samples per rank we need
if admissibility == 'weak':
    kmax = 1
elif admissibility == 'full':
    kmax = 9
else:
    kmax = 5


def c11(p):
    return np.ones_like(p[:, 0])


def c22(p):
    return np.ones_like(p[:, 0])


def c33(p):
    return np.ones_like(p[:, 0])


def c(p):
    return kh*kh*np.ones_like(p[:, 0])


HH = pdo.PDO_3d(c11=c11, c22=c22, c33=c33, c=c)
coeffs = {'c11': 1., 'c22': 1., 'c33': 1., 'c': kh**2}


# ---------------------------------------------------------------------------
#  THE ONLY PLACE WHERE THE TWO DISCRETIZATIONS DIFFER
# ---------------------------------------------------------------------------
def build_operator(Lx):
    """Assemble the slab [0,2Lx] x [0,Ly] x [0,Lz].

    Returns the interior matrix, the interior/boundary coupling, the full point
    list, the interior and boundary index sets, and the interior/boundary
    quadrature weights (all ones except for the weighted SOMS operator)."""
    if solve_method == 'SOMS':
        px, py, pz = order
        nbx = int(2*Lx*nb)              # even -> the interface x = Lx is a block face
        Sii, Sib, ftild, XYtot, Ii, Ib, wi, wb = SOMS3D_csr.SOMS_solver_sparse(
            px, py, pz, nbx, nb, nb, 2*Lx, Ly, Lz, coeffs, True, None,
            weighted=weighted)
        return Sii, Sib, XYtot, np.asarray(Ii), np.asarray(Ib), wi, wb

    nx = int(2*order[1]*Lx) + 1         # derive nx so the plane x = Lx lands on-grid
    solver = stencil.stencilSolver(
        HH, geom.BoxGeometry(np.array([[0, 0, 0], [2*Lx, Ly, Lz]])),
        [nx, order[1], order[2]])
    return (solver.Aii, solver.Aix, solver.XX,
            np.asarray(solver.Ji), np.asarray(solver.Jx),
            np.ones(solver.Aii.shape[0]), np.ones(solver.Aix.shape[1]))


# ---------------------------------------------------------------------------
#  sweep over the slab width
# ---------------------------------------------------------------------------
for Lx in widths:
    print("=========================================================")
    print(f"type={solve_method}  order={order}  Lx={Lx}  (Ly,Lz)=({Ly},{Lz})  "
          f"admissibility={admissibility}  nleaf={nleaf}")

    Sii, Sib, XYtot, Ii, Ib, wi, wb = build_operator(Lx)
    XXi = XYtot[Ii, :]
    XXb = XYtot[Ib, :]
    tol = 1e-9

    # index sets (identical logic for both discretizations)
    Jc = np.where(np.abs(XXi[:, 0] - Lx) < tol)[0]        # interior interface DOFs
    Jl = np.where(np.abs(XXb[:, 0] - 0.) < tol)[0]        # physical x = 0    face
    Jr = np.where(np.abs(XXb[:, 0] - 2*Lx) < tol)[0]      # physical x = 2*Lx face
    Jb = np.setdiff1d(np.arange(XXb.shape[0]),
                      np.concatenate([Jl, Jr])).astype(np.int64)
    Jplane = np.where(np.abs(XYtot[:, 0] - Lx) < tol)[0]  # full interface plane
    ndofs_if = len(Jplane)

    # Position of each interior interface DOF inside the interface block.  For
    # the stencil this is a strict subset (the plane carries a ring of physical
    # Dirichlet DOFs); for SOMS the two sets coincide and this is a permutation.
    Jc_inJc = match_rows(XXi[Jc], XYtot[Jplane])
    # order both faces exactly like the interface plane (match on (y,z))
    #Jl = Jl[match_rows(XYtot[Jplane][:, 1:], XXb[Jl][:, 1:])]
    #Jr = Jr[match_rows(XYtot[Jplane][:, 1:], XXb[Jr][:, 1:])]
    assert np.allclose(XXb[Jl][:, 1:], XYtot[Jplane][:, 1:]), "x=0 face not aligned to interface plane"
    assert np.allclose(XXb[Jr][:, 1:], XYtot[Jplane][:, 1:]), "x=2Lx face not aligned to interface plane"
    assert len(Jl) == ndofs_if and len(Jr) == ndofs_if, "face/plane size mismatch"

    print("|Jl| = ", len(Jl), " |Jr| = ", len(Jr),
          " |Jc| = ", len(Jc), " |plane| = ", ndofs_if)

    tic = time.time()
    ctx, _, _ = setup_mumps(Sii)
    print("LU decomposition total time = ", time.time()-tic)
    ctx.mumps_instance.icntl[27] = BLK        # one wide BLAS-3 block per chunk
    
    def smatmat(v, J, transpose=False):
        """Apply the interface map  -(Sii^{-1} Sib_J)  (or its transpose).

        Forward  : boundary data on face J (plane-ordered) -> interface block.
                   The interior solve fills the Jc_inJc rows; any remaining
                   (physical boundary) rows of the block stay 0.
        Transpose: interface block -> face J.  Only the Jc_inJc rows of the
                   input feed the transposed interior solve."""
        v_tmp = v[:, None] if v.ndim == 1 else v
        k = v_tmp.shape[1]
        out = np.zeros((ndofs_if, k))
        if not transpose:
            Sib_J = Sib[:, J].tocsc()
            for a in range(0, k, BLK):
                cc = slice(a, min(a + BLK, k))
                rhs = (Sib_J @ sparse.csc_matrix(v_tmp[:, cc])).tocsc()
                sol = mumps_solve_sparse(ctx,rhs)          # (len(Ii) x BLK) — bounded
                out[Jc_inJc, cc] = -sol[Jc, :]
                del rhs, sol
                ctx.mumps_instance.icntl[20] = 0
        else:
            Sib_J_T = Sib[:, J].T.tocsr()
            m = len(Jc)
            for a in range(0, k, BLK):
                cc = slice(a, min(a + BLK, k))
                w = v_tmp[Jc_inJc, cc]
                bw = w.shape[1]
                rhs = sparse.csc_matrix(
                    (w.ravel(order="F"), np.tile(Jc, bw), np.arange(0, m*bw+1, m)),
                    shape=(len(Ii), bw))
                sol = mumps_solve_sparse(ctx,rhs,transpose=True)
                out[:, cc] = -Sib_J_T @ sol
                del rhs, sol
                ctx.mumps_instance.icntl[20] = 0
        return out.flatten() if v.ndim == 1 else out

    # single-slab sanity check: the maps reproduce the trace at x = Lx
    ul = wb[Jl] * bc_helmholtz(XXb[Jl], kh)
    ur = wb[Jr] * bc_helmholtz(XXb[Jr], kh)
    ub = wb[Jb] * bc_helmholtz(XXb[Jb], kh)
    uc = wi[Jc] * bc_helmholtz(XXi[Jc], kh)
    pred = (smatmat(ul, Jl) + smatmat(ur, Jr))[Jc_inJc] \
        - (ctx.solve(Sib[:, Jb] @ ub))[Jc]
    print("slab map err = ", np.linalg.norm(uc - pred)/np.linalg.norm(uc))

    # -----------------------------------------------------------------------
    #  HBS tree + random sampling of both maps (this is what needs the LU)
    # -----------------------------------------------------------------------
    tree = slabTree.slabTree(XXb[Jl], False, nleaf) if admissibility == 'weak' \
        else slabTree.slabTree(XXb[Jl], False, nleaf, adjacency=admissibility)
    nl = len(tree.get_box_inds(tree.get_leaves()[0]))
    s = kmax*max(2*max(ranks), nl) + max(ranks) + max(20,max(ranks)//2)      # samples for the largest rank

    def sample_lr(Om_l, Om_r):
        """One stacked forward solve -> (Y_l, Y_r)."""
        Som = sparse.csc_matrix(np.hstack((Sib[:, Jl] @ Om_l, Sib[:, Jr] @ Om_r)))
        k = Om_l.shape[1] + Om_r.shape[1]
        out = np.zeros((ndofs_if, k))
        for a in range(0, k, BLK):
            cc = slice(a, min(a + BLK, k))
            sol = mumps_solve_sparse(ctx,Som[:, cc])
            out[Jc_inJc, cc] = -sol[Jc, :]
            del sol
            ctx.mumps_instance.icntl[20] = 0
        return out[:, :Om_l.shape[1]], out[:, Om_l.shape[1]:]

    def adjoint_sample_lr(Psi):
        """One shared adjoint solve (Sii^{-T} E_c^T Psi) -> (Z_l, Z_r)."""
        k = Psi.shape[1]
        m = len(Jc)
        Z_l = np.empty((len(Jl), k))
        Z_r = np.empty((len(Jr), k))
        SiblT = Sib[:, Jl].T.tocsr()
        SibrT = Sib[:, Jr].T.tocsr()
        for a in range(0, k, BLK):
            cc = slice(a, min(a + BLK, k))
            w = Psi[Jc_inJc, cc]
            bw = w.shape[1]
            rhs = sparse.csc_matrix(
                (w.ravel(order="F"), np.tile(Jc, bw), np.arange(0, m*bw+1, m)),
                shape=(len(Ii), bw))
            sol = mumps_solve_sparse(ctx,rhs,transpose=True)
            Z_l[:, cc] = -(SiblT @ sol)
            Z_r[:, cc] = -(SibrT @ sol)
            del sol
        return Z_l, Z_r

    tic = time.time()
    Om_l = np.random.standard_normal((len(Jl), s))
    Om_r = np.random.standard_normal((len(Jr), s))
    Psi = np.random.standard_normal((ndofs_if, s))      # shared corange test matrix
    Y_l, Y_r = sample_lr(Om_l, Om_r)
    Z_l, Z_r = adjoint_sample_lr(Psi)
    tSample = time.time()-tic
    print("sampling time (", s, " samples) = ", tSample)

    # reference blocks for the error study, computed while the factors are alive
    V = np.random.standard_normal((ndofs_if, ntest))
    ref_l = smatmat(V, Jl)
    ref_r = smatmat(V, Jr)

    release_mumps(ctx)
    del ctx, Sii, Sib
    gc.collect()

    # -----------------------------------------------------------------------
    #  rank sweep: compress and measure
    # -----------------------------------------------------------------------
    for rk in ranks:
        if admissibility == 'weak':
            SSl = HBStorch.HBSMAT(device=device, tree=tree)
            SSr = HBStorch.HBSMAT(device=device, tree=tree)
        else:
            SSl = HBStorch_strong.HBSMAT(device=device, tree=tree)
            SSr = HBStorch_strong.HBSMAT(device=device, tree=tree)

        tic = time.time()
        SSl.construct(rk, Om_l, Psi, Y_l, Z_l, fast=True)
        SSr.construct(rk, Om_r, Psi, Y_r, Z_r, fast=True)
        tHBS = time.time()-tic

        err_l = np.linalg.norm(SSl@V - ref_l)/np.linalg.norm(ref_l)
        err_r = np.linalg.norm(SSr@V - ref_r)/np.linalg.norm(ref_r)
        mem_MB = (SSl.nbytes + SSr.nbytes)/1e6

        print("Lx = %g  rank = %3d   err_l = %.3e   err_r = %.3e   "
              "mem = %.1f MB   t_HBS = %.2f s"
              % (Lx, rk, err_l, err_r, mem_MB, tHBS))

        header = ["type", "Lx", "ndofs_if", "nleaf", "admissibility", "kh",
                  "rank", "nsamples", "err_l", "err_r", "mem_MB",
                  "t_construct_s", "t_sample_s"]
        row = [solve_method, Lx, ndofs_if, nleaf, admissibility, kh,
               rk, s, err_l, err_r, mem_MB, tHBS, tSample]
        need_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if need_header:
                f.write(",".join(header) + "\n")
            f.write(",".join(str(x) for x in row) + "\n")

        del SSl, SSr
        gc.collect()

    del Om_l, Om_r, Psi, Y_l, Y_r, Z_l, Z_r, V, ref_l, ref_r, tree
    gc.collect()

print("appended results to", csv_path)