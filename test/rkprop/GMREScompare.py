import numpy as np
import matplotlib.pyplot as plt

import SOMS3D_csr
import torch
import matAssembly.HBS.slabTree as slabTree
import matAssembly.HBS.HBStorch_strong as HBStorch_strong
import matAssembly.HBS.HBStorch as HBStorch
from scipy.sparse.linalg import LinearOperator
import solver.stencil.stencilSolver as stencil
import solver.stencil.geom as geom
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
import solver.hpsmultidomain.hpsmultidomain.domain_driver as hps_solver
import mumps
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import time
import os
from scipy.sparse.linalg import gmres
torch.set_default_dtype(torch.float64)
def rss_gb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6   # kB -> GB
    return -1.0

def _enable_blr(ctx, blr_tol):
    ctx.mumps_instance.icntl[35] = 1
    ctx.mumps_instance.cntl[7] = blr_tol
    # inst.icntl[36] = 0      # BLR variant (UFSC); leave default unless tuning


def setup_mumps(Sii, blr=False, blr_tol=1e-8):
    ctx = mumps.Context()
    ctx.analyze(Sii)
    if blr:
        _enable_blr(ctx, blr_tol)   # set BEFORE analyze so estimates account for BLR
    ctx.analyze(Sii)                # symbolic factorization (sparsity pattern only)
    ctx.factor(Sii)                 # numeric factorization (BLR-compressed if enabled)
    return ctx


def setup_mumps_transpose(Sii, ctx=None, sym=False, blr=False, blr_tol=1e-8):
    ctxT = mumps.Context()
    ctxT.analyze(Sii.T)
    if blr:
        _enable_blr(ctxT, blr_tol)
    ctxT.analyze(Sii.T)
    ctxT.factor(Sii.T)
    return ctxT


def bc_laplace(p):
    """Free-space Green's function with source at (-0.5, -0.5, -0.5)."""
    r = np.sqrt((p[:,0]+.5)**2+(p[:,1]+.5)**2+(p[:,2]+.5)**2)
    return 1./(4*np.pi*r)
def bc_helmholtz(p,kh):
    """Free-space Green's function with source at (-0.5, -0.5, -0.5)."""
    r = np.sqrt((p[:,0]+.5)**2+(p[:,1]+.5)**2+(p[:,2]+.5)**2)
    return np.real(np.exp(1j*kh*r)/(4*np.pi*r))


def match_rows(A, B, decimals=9):
    """For each row of A return the index of the matching row in B (stencil path).

    Used to line up DOF orderings by coordinate, e.g. the physical x = 0 / x = Lx
    faces against the interface plane, or the strictly-interior interface points
    against their positions inside the full plane.  A and B must contain the same
    set of points (matched to `decimals` places)."""
    lut = {}
    for j in range(B.shape[0]):
        lut[tuple(np.round(B[j], decimals))] = j
    idx = np.empty(A.shape[0], dtype=np.int64)
    for i in range(A.shape[0]):
        idx[i] = lut[tuple(np.round(A[i], decimals))]
    return idx


class gmres_info(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
        self.resList=[]
    def __call__(self, rk=None):
        self.niter += 1
        self.resList+=[rk]
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


import argparse
import re

def _ints3(tokens):
    """Parse a triple of ints given either space- or comma-separated."""
    flat = [t for t in re.split(r"[,\s]+", " ".join(tokens).strip()) if t]
    if len(flat) != 3:
        raise SystemExit(f"order expects 3 values (got {flat})")
    return [int(t) for t in flat]

def _nums3(tokens):
    """Parse a triple of floats; each entry may be a fraction like '1/16'."""
    flat = [t for t in re.split(r"[,\s]+", " ".join(tokens).strip()) if t]
    if len(flat) != 3:
        raise SystemExit(f"shape expects 3 values (got {flat})")
    out = []
    for t in flat:
        if "/" in t:
            a, b = t.split("/")
            out.append(float(a) / float(b))
        else:
            out.append(float(t))
    return out

def _str2bool(s):
    """Parse a boolean given as a CLI string, so `--splitting True/False` works."""
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("1", "true", "t", "yes", "y")

def _solverdata(solver):
    Aii = solver.Aii
    Aib = solver.Aix.tocsc()        # column slicing  Aib[:, J0]
    Abi = solver.Axi.tocsr()        # row    slicing  Abi[J1]
    Abb = solver.Axx.tocsr()
    Ni  = Aii.shape[0]
    XXi = solver.XXi
    XXb = solver.XXb
    Ji  = solver.Ji
    Jb  = solver.Jx
    return Aii,Aib,Abi,Abb,Ni,XXi,XXb,Ji,Jb
def _cond(A,ctx,niter = 100):
    v = np.random.standard_normal((A.shape[0],))
    w = np.random.standard_normal((A.shape[0],))
    for j in range(niter):
        v = v/np.linalg.norm(v)
        w = w/np.linalg.norm(w)
        v = A@v
        w = ctx.solve(w)
    nrmest = (v.T@(A@v))/(v.T@v)
    inrmest = (w.T@(ctx.solve(w)))/(w.T@w)
    return nrmest*inrmest
    
def _cond_T(T,niter = 50):
    v = np.random.standard_normal((T.shape[0],))
    w = np.random.standard_normal((T.shape[0],))
    for j in range(niter):
        v = v/np.linalg.norm(v)
        w = w/np.linalg.norm(w)
        v = T@v
        w,_ = gmres(T,w,rtol = 1e-12,maxiter = 300)
    nrmest = v.T@(T@v)/(v.T@v)
    w1,_ = gmres(T,w,rtol = 1e-8)
    inrmest = (w.T@w1)/(w.T@w)
    return nrmest*inrmest


_parser = argparse.ArgumentParser(
    description="Thin-slab interface-map GMRES solve (LU and HBS-compressed).")
_parser.add_argument("--type", choices=["HPS", "stencil"], default="stencil",
                     help="discretization / solver type")
_parser.add_argument("--order", nargs="+", default=None,
                     help="(nx ny nz) for stencil [nx is overridden = int(ny*Lx)+1] "
                          "or (px py pz) for SOMS; space- or comma-separated")
_parser.add_argument("--shape", nargs="+", default=["1/16", "1", "1"],
                     help="domain extents Lx Ly Lz (fractions like 1/16 allowed)")
_parser.add_argument("--admissibility", choices=["full", "partial","weak"], default="full",
                     help="HBS tree adjacency / admissibility")
_parser.add_argument("--gmres-iters", dest="gmres_iters", type=int, default=100,
                     help="max GMRES iterations (sets maxiter & restart); 0 skips the GMRES solve")
_parser.add_argument("--rank", dest="rk", type=int, default=50,
                     help="rank of HBS approximation")
_parser.add_argument("--nb", dest="nb", type=int, default=8,
                     help="number of blocks in HPS")
_parser.add_argument("--kh", dest="kh", type=float, default=0.,
                     help="wavenumber")
_parser.add_argument("--nleaf", dest="nleaf", type=int, default=64,
                     help="number of DOFs per leaf")
_parser.add_argument("--splitting", dest="splitting", type=_str2bool, default=False,
                     help="sparse self-block extraction (HBS path only): keep the local "
                          "block Abb[Jx,Jx] exact and HBS-compress ONLY the smoothing "
                          "remainder -Abi Aii^{-1} Aib. Requires --admissibility weak "
                          "(must not use the HBS_strong interface).")
args = _parser.parse_args()

solve_method = args.type
# type-appropriate default order if none supplied
if args.order is None:
    order = [9, 128, 128] if solve_method == "stencil" else [6, 6, 6]
else:
    order = _ints3(args.order)
Lx, Ly, Lz = _nums3(args.shape)
admissibility = args.admissibility
gmres_iters = args.gmres_iters
rk = args.rk
nb = args.nb
splitting = args.splitting
if splitting and admissibility != 'weak':
    raise SystemExit(
        "--splitting requires --admissibility weak: the sparse self-block "
        "extraction uses the weak HBS interface (HBStorch) and must not coexist "
        "with the HBS_strong interface (full/partial admissibility).")





kh = args.kh

def  c11_np(p):
    return np.ones_like(p[:,0])
def  c22_np(p):
    return np.ones_like(p[:,0])
def  c33_np(p):
    return np.ones_like(p[:,0])
def  c_np(p):
    return kh*kh*np.ones_like(p[...,0])
HH_np = pdo.PDO_3d(c11=c11_np,c22=c22_np,c33=c33_np,c=c_np)


nx, ny, nz = order
print(  "============ GLOBAL SOLVER: ============" )
unitCube  = geom.BoxGeometry(np.array([[0,0,0],[1,1,1]]))
ord_ = [nx, ny, nz]

solver_glob = stencil.stencilSolver(HH_np, unitCube, ord_)



print("solver construction done")
# ---- four blocks (interior solve + conormal boundary rows) ----


Aii,Aib,Abi,Abb,Ni,XXi,XXb,Ji,Jb = _solverdata(solver_glob)
ctx = setup_mumps(Aii)

ui = bc_helmholtz(solver_glob.XXi,kh)
ub = bc_helmholtz(solver_glob.XXb,kh)
rhs = -Aib@ub


gInfo = gmres_info()
if gmres_iters > 0:
    tic = time.time()
    uhat, _ = gmres(Aii, rhs, rtol=1e-8, callback=gInfo,
                    maxiter=gmres_iters, restart=gmres_iters)
    solve_time_LU = time.time() - tic
    niter = gInfo.niter
    gmres_err = np.linalg.norm(uhat - ui) / np.linalg.norm(ui)
    print("time = ", solve_time_LU)
    print("niter = ", niter)
    print("u err = ", gmres_err)
else:
    solve_time_LU = float('nan'); niter = float('nan'); gmres_err = float('nan')
    print("GMRES solve skipped (gmres_iters = 0)")
print("condition number = ",_cond(Aii,ctx))

print(  "============   T SOLVER:   ============" )

nx = int(ny * Lx) + 1            # single-width slab; nx so x=Lx lands on-grid
print("stencil nx (derived from ny, Lx) = ", nx)
ord_ = [nx, ny, nz]

slabGeom    = geom.BoxGeometry(np.array([[0,0,0],[Lx,Ly,Lz]]))
solver_T = stencil.stencilSolver(HH_np, slabGeom, ord_)
Aii,Aib,Abi,Abb,Ni,XXi,XXb,Ji,Jb = _solverdata(solver_T)
tol = 1e-9


# Interior factorization: MUMPS (METIS nested dissection), NOT scipy splu.
    # The thin-slab interior is 3D-structured; COLAMD/splu produces catastrophic
    # fill (bandwidth ~ Ny*Nz) and makes the multi-RHS sampling solves crawl.
    # This mirrors the overlapping-stencil and HPS paths.
BLK = 32
tic  = time.time()
ctx  = setup_mumps(Aii, blr=False)
ctxT = setup_mumps_transpose(Aii, blr=False)
tLU  = time.time() - tic
print("MUMPS factorization time = ", tLU)
ctx.mumps_instance.icntl[27]  = BLK     # one wide BLAS-3 block per chunk
ctxT.mumps_instance.icntl[27] = BLK

def _mumps_solve(ctx_, B):
    """Apply Aii^-1 (ctx) or Aii^-T (ctxT) to a 1-D or 2-D rhs, BLK-chunked."""
    B = np.asarray(B)
    one_d = (B.ndim == 1)
    if one_d:
        B = B[:, None]
    out = np.empty((B.shape[0], B.shape[1]), dtype=np.float64)
    for c0 in range(0, B.shape[1], BLK):
        c = slice(c0, min(c0 + BLK, B.shape[1]))
        out[:, c] = ctx_._solve_sparse(sparse.csc_matrix(B[:, c]))
        ctx_.mumps_instance.icntl[20] = 0
    return out[:, 0] if one_d else out

solver_ii = LinearOperator((Ni, Ni), dtype=np.float64,
                            matvec  = lambda b: _mumps_solve(ctx,  b),
                            matmat  = lambda b: _mumps_solve(ctx,  b),
                            rmatvec = lambda b: _mumps_solve(ctxT, b),
                            rmatmat = lambda b: _mumps_solve(ctxT, b))


Jl = np.where(np.abs(XXb[:, 0] - 0.) < tol)[0]
Jr = np.where(np.abs(XXb[:, 0] - Lx) < tol)[0]

JJ  = np.array([i for i in range(len(Jb)) if i not in Jl and i not in Jr])
Jlc = np.array([i for i in range(len(Jb)) if i not in Jl])
Jrc = np.array([i for i in range(len(Jb)) if i not in Jr])

ndofs_if = len(Jl)
assert len(Jl) == len(Jr), "left/right interface face sizes differ"
XXl = XXb[Jl, :]
XXr = XXb[Jr, :]

ring = np.where((XXl[:, 1] < tol) | (XXl[:, 1] > Ly - tol) |
                    (XXl[:, 2] < tol) | (XXl[:, 2] > Lz - tol))[0]
inn  = np.where((XXl[:, 1] > tol) & (XXl[:, 1] < Ly - tol) &
                    (XXl[:, 2] > tol) & (XXl[:, 2] < Lz - tol))[0]




def make_T(J1, J0, with_local=True):
    Bbi = Abi[J1].tocsr()
    Bib = Aib[:, J0].tocsc()
    Bbb = Abb[J1][:, J0].tocsr() if with_local else None
    def fwd(v):
        v2 = v[:, None] if v.ndim == 1 else v
        out = -np.asarray(Bbi @ (solver_ii @ np.asarray(Bib @ v2)))
        if Bbb is not None:
            out = out + np.asarray(Bbb @ v2)
        out[ring, :] = 0.0                       # throw away ring conormal rows
        return out.ravel() if v.ndim == 1 else out
    def adj(v):
        v2 = (v[:, None] if v.ndim == 1 else v).copy()
        v2[ring, :] = 0.0                        # transpose of zeroing output rows
        out = -np.asarray(Bib.T @ (solver_ii.T @ np.asarray(Bbi.T @ v2)))
        if Bbb is not None:
            out = out + np.asarray(Bbb.T @ v2)
        return out.ravel() if v.ndim == 1 else out
    return fwd, adj

def _linop(shape, fa):
    f, a = fa
    return LinearOperator(shape, matvec=f, rmatvec=a, matmat=f, rmatmat=a,
                            dtype=np.float64)

nif = ndofs_if
LinOp_ll = _linop((nif, nif), make_T(Jl, Jl))
LinOp_rr = _linop((nif, nif), make_T(Jr, Jr))
LinOp_lr = _linop((nif, nif), make_T(Jl, Jr))
LinOp_rl = _linop((nif, nif), make_T(Jr, Jl))

nSlab = int(round(1. / Lx))
print("nSlab = ", nSlab)

# ---- block-tridiagonal balance operator (LU/exact maps) ----
# Interior-face rows carry the conormal flux balance; ring rows are identity
# (the discarded conormal is replaced by  u_ring = known trace).
def apply_balance(u):
    utmp = u[:, None] if u.ndim == 1 else u
    out = np.zeros_like(utmp)
    for j in range(nSlab - 1):
        b = j * nif
        blk = LinOp_ll @ utmp[b:b + nif] + LinOp_rr @ utmp[b:b + nif]
        if j > 0:
            blk = blk + LinOp_rl @ utmp[b - nif:b]
        if j < nSlab - 2:
            blk = blk + LinOp_lr @ utmp[b + nif:b + 2 * nif]
        blk[ring] = utmp[b:b + nif][ring]        # identity-pin the ring rows
        out[b:b + nif] = blk
    return out.ravel() if u.ndim == 1 else out
def apply_balance_T(u):
    utmp = u[:, None] if u.ndim == 1 else u
    out = np.zeros_like(utmp)
    for j in range(nSlab - 1):
        b = j * nif
        blk = LinOp_ll.T @ utmp[b:b + nif] + LinOp_rr.T @ utmp[b:b + nif]
        if j > 0:
            blk = blk + LinOp_lr.T @ utmp[b - nif:b]
        if j < nSlab - 2:
            blk = blk + LinOp_rl.T @ utmp[b + nif:b + 2 * nif]
        blk[ring] = utmp[b:b + nif][ring]        # identity-pin the ring rows
        out[b:b + nif] = blk
    return out.ravel() if u.ndim == 1 else out

A_balance = LinearOperator(shape=((nSlab - 1) * nif, (nSlab - 1) * nif),
                            matvec=apply_balance,rmatvec=apply_balance_T, dtype=np.float64)


def make_T(J1, J0, with_local=True):
    Bbi = Abi[J1].tocsr()
    Bib = Aib[:, J0].tocsc()
    Bbb = Abb[J1][:, J0].tocsr() if with_local else None
    def fwd(v):
        v2 = v[:, None] if v.ndim == 1 else v
        out = -np.asarray(Bbi @ (solver_ii @ np.asarray(Bib @ v2)))
        if Bbb is not None:
            out = out + np.asarray(Bbb @ v2)
        out[ring, :] = 0.0                       # throw away ring conormal rows
        return out.ravel() if v.ndim == 1 else out
    def adj(v):
        v2 = (v[:, None] if v.ndim == 1 else v).copy()
        v2[ring, :] = 0.0                        # transpose of zeroing output rows
        out = -np.asarray(Bib.T @ (solver_ii.T @ np.asarray(Bbi.T @ v2)))
        if Bbb is not None:
            out = out + np.asarray(Bbb.T @ v2)
        return out.ravel() if v.ndim == 1 else out
    return fwd, adj


# ---- rhs maps: interface-from-known-data ----
Trb,   _ = make_T(Jr, JJ)       # interior slab: right face from y/z faces
Tlb,   _ = make_T(Jl, JJ)       # interior slab: left  face from y/z faces
Trb_l, _ = make_T(Jr, Jrc)      # end slab 0:    right face from everything-but-Jr
Tlb_r, _ = make_T(Jl, Jlc)      # end slab N-1:  left  face from everything-but-Jl

# ---- rhs + manufactured interface trace ----
rhs   = np.zeros((nSlab - 1) * nif)
u_if  = np.zeros((nSlab - 1) * nif)
XXif  = np.zeros(((nSlab - 1) * nif, 3))
shiftL = np.array([Lx, 0., 0.])
for j in range(nSlab):
    XXbloc = XXb + j * shiftL
    if j == 0:
        ub_loc = bc_helmholtz(XXbloc[Jrc], kh)
        rhs[0:nif] -= Trb_l(ub_loc)
    elif j == nSlab - 1:
        ub_loc = bc_helmholtz(XXbloc[Jlc], kh)
        rhs[(j - 1) * nif:j * nif] -= Tlb_r(ub_loc)
    else:
        ub_loc = bc_helmholtz(XXbloc[JJ], kh)
        rhs[j * nif:(j + 1) * nif]       -= Trb(ub_loc)
        rhs[(j - 1) * nif:j * nif]       -= Tlb(ub_loc)
    if j < nSlab - 1:
        u_if[j * nif:(j + 1) * nif]  = bc_helmholtz(XXbloc[Jr], kh)
        XXif[j * nif:(j + 1) * nif, :] = XXbloc[Jr]

# ring rows are identity-pinned: set their rhs to the known Dirichlet trace
for j in range(nSlab - 1):
    rhs[j * nif + ring] = u_if[j * nif + ring]

gInfo = gmres_info()
if gmres_iters > 0:
    tic = time.time()
    uhat, _ = gmres(A_balance, rhs, rtol=1e-8, callback=gInfo,
                    maxiter=gmres_iters, restart=gmres_iters)
    solve_time_LU = time.time() - tic
    niter = gInfo.niter
    gmres_err = np.linalg.norm(uhat - u_if) / np.linalg.norm(u_if)
    print("time = ", solve_time_LU)
    print("niter = ", niter)
    print("u err = ", gmres_err)
else:
    solve_time_LU = float('nan'); niter = float('nan'); gmres_err = float('nan')
    print("GMRES solve skipped (gmres_iters = 0)")
Tdense = A_balance@np.identity(A_balance.shape[0])
print("condition number = ",np.linalg.cond(Tdense))


print(  "============   S SOLVER:   ============" )
cx = Lx
nx = int(2 * ny * Lx) + 1            # single-width slab; nx so x=Lx lands on-grid
print("stencil nx (derived from ny, Lx) = ", nx)
ord_ = [nx , ny, nz]

slabGeom    = geom.BoxGeometry(np.array([[0,0,0],[2*Lx,Ly,Lz]]))
solver_S = stencil.stencilSolver(HH_np, slabGeom, ord_)
Sii,Sib,Abi,Abb,Ni,XXi,XXb,Ji,Jb = _solverdata(solver_S)
XYtot = solver_S.XX
tol = 1e-9


tol = 1e-9

Jc = np.where(np.abs(XXi[:,0]-cx) < tol)[0]               # interior interface DOFs
Jl = np.where(np.abs(XXb[:,0]-0.) < tol)[0]               # physical x = 0  face
XXl = XXb[Jl,:]
Jr = np.where(np.abs(XXb[:,0]-2*Lx) < tol)[0]               # physical x = Lx face
Jb = np.setdiff1d(np.arange(XXb.shape[0]),
                    np.concatenate([Jl, Jr])).astype(np.int64)   # the four (y,z) faces

Jc_large = np.where(np.abs(XYtot[:,0]-cx) < tol)[0]       # full interface plane
Jc_inJc =  np.where((XYtot[Jc_large,1] > tol) &\
                (XYtot[Jc_large,1] < Ly-tol) &\
                (XYtot[Jc_large,2] > tol) &\
                (XYtot[Jc_large,2] < Lz-tol))[0]   # x = Lx/2
ndofs_if = len(Jc_large)                                  # size of one interface block

print("|Jl| = ", len(Jl))
print("|Jr| = ", len(Jr))
print("|Jc| = ", len(Jc))
print("|Jc_large| = ", ndofs_if)
print("|Jc_inJc|  = ", len(Jc_inJc))

assert len(Jl) == ndofs_if and len(Jr) == ndofs_if, "face/plane size mismatch"
assert np.allclose(XXb[Jr][:,1:3], XYtot[Jc_large][:,1:3]), "Jr not aligned to Jc_large"
assert np.allclose(XXb[Jl][:,1:3], XYtot[Jc_large][:,1:3]), "Jl not aligned to Jc_large"

def scatter(vec_Jc):
    """Scatter a length-len(Jc) interior result into a length-ndofs_if block."""
    out = np.zeros(ndofs_if)
    out[Jc_inJc] = vec_Jc
    return out

tic = time.time()
BLK = 32                                   # tune; see note below
ctx  = setup_mumps(Sii, blr=False)
ctxT = setup_mumps_transpose(Sii, blr=False)
tMUMPS = time.time()-tic
print("LU decomposition total time = ", tMUMPS)

ctx.mumps_instance.icntl[27]  = BLK        # one wide BLAS-3 block per chunk
ctxT.mumps_instance.icntl[27] = BLK

def smatmat(v, J, transpose=False):
    """Apply the interface map  -(Sii^{-1} Sib_J)  (or its transpose).

    Forward : boundary data on face J (Jc_large-ordered)  ->  interface block.
                The interior solve fills the Jc_inJc rows; the (y,z)-boundary
                rows of the block stay 0.
    Transpose: interface block -> face J.  Only the Jc_inJc rows of the input
                feed the (transposed) interior solve.
    """
    v_tmp = v[:, None] if v.ndim == 1 else v
    k = v_tmp.shape[1]

    if not transpose:
        Sib_J = Sib[:, J].tocsc()
        out = np.zeros((ndofs_if, k))
        for s in range(0, k, BLK):
            c = slice(s, min(s + BLK, k))
            rhs = (Sib_J @ sparse.csc_matrix(v_tmp[:, c])).tocsc()
            sol = ctx._solve_sparse(rhs)              # dense (len(Ii) x BLK) — bounded
            out[Jc_inJc, c] = -sol[Jc, :]
            del rhs, sol
            ctx.mumps_instance.icntl[20] = 0
    else:
        Sib_J_T = Sib[:, J].T.tocsr()
        out = np.zeros((ndofs_if, k))
        for s in range(0, k, BLK):
            c = slice(s, min(s + BLK, k))
            w  = v_tmp[Jc_inJc, c]; bw = w.shape[1]; m = len(Jc)
            rhs = sparse.csc_matrix(
                (w.ravel(order="F"), np.tile(Jc, bw), np.arange(0, m*bw+1, m)),
                shape=(len(Ii), bw))
            sol = ctxT._solve_sparse(rhs)
            out[:, c] = -Sib_J_T @ sol
            del rhs, sol
            ctx.mumps_instance.icntl[20] = 0
    return out.flatten() if v.ndim == 1 else out

LinOp_r = LinearOperator(shape=(ndofs_if,len(Jr)),\
    matvec = lambda v:smatmat(v,Jr), rmatvec = lambda v:smatmat(v,Jr,transpose=True),\
    matmat = lambda v:smatmat(v,Jr), rmatmat = lambda v:smatmat(v,Jr,transpose=True))
LinOp_l = LinearOperator(shape=(ndofs_if,len(Jl)),\
    matvec = lambda v:smatmat(v,Jl), rmatvec = lambda v:smatmat(v,Jl,transpose=True),\
    matmat = lambda v:smatmat(v,Jl), rmatmat = lambda v:smatmat(v,Jl,transpose=True))



ndslab = int(round(1./cx)) - 1
XXif = np.zeros((ndslab*ndofs_if,3))
rhs  = np.zeros((ndslab*ndofs_if,))
u_true = np.zeros((ndslab*ndofs_if,))
for i in range(ndslab):
    shift = np.array([i*cx, 0., 0.])

    XXif[i*ndofs_if:(i+1)*ndofs_if, :] = XYtot[Jc_large] + shift
    trace_full = bc_helmholtz(XYtot[Jc_large] + shift, kh)   # interior AND ring
    u_true[i*ndofs_if:(i+1)*ndofs_if]  = trace_full

    XXb_loc = XXb + shift
    uj = bc_helmholtz(XXb_loc, kh)

    if i == 0:                        # physical x=0 face known -> keep Jl, drop Jr
        Jb0 = np.setdiff1d(np.arange(XXb.shape[0]), Jr).astype(np.int64)
        blk = -ctx.solve(Sib[:, Jb0] @ uj[Jb0])[Jc]
        br = bc_helmholtz(XXb[Jr] + shift, kh)
        uc = bc_helmholtz(XXi[Jc] + shift, kh)
        ur = (LinOp_r @ br)[Jc_inJc]
    elif i == ndslab - 1:             # physical far face known -> keep Jr, drop Jl
        Jb0 = np.setdiff1d(np.arange(XXb.shape[0]), Jl).astype(np.int64)
        blk = -ctx.solve(Sib[:, Jb0] @ uj[Jb0])[Jc]
        bl = bc_helmholtz(XXb[Jl] + shift, kh)
        uc = bc_helmholtz(XXi[Jc] + shift, kh)
        ul = (LinOp_l @ bl)[Jc_inJc]
    else:                             # interior slab: only the (y,z) faces are known
        blk = -ctx.solve(Sib[:, Jb] @ uj[Jb])[Jc]
        bl = bc_helmholtz(XXb[Jl] + shift, kh)
        br = bc_helmholtz(XXb[Jr] + shift, kh)
        uc = bc_helmholtz(XXi[Jc] + shift, kh)
        ul = (LinOp_l @ bl)[Jc_inJc]
        ur = (LinOp_r @ br)[Jc_inJc]

    # interior rows -> centerline solve; ring rows -> known Dirichlet trace
    blk_full = trace_full.copy()
    blk_full[Jc_inJc] = blk
    rhs[i*ndofs_if:(i+1)*ndofs_if] = blk_full

def apply_balance(u):
    if u.ndim == 1:
        utmp = u[:,None]
    else:
        utmp = u
    out = utmp.copy()
    for j in range(ndslab):
        if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] -= LinOp_l@(utmp[(j-1)*ndofs_if:j*ndofs_if,:])
        if j < ndslab-1:   out[j*ndofs_if:(j+1)*ndofs_if,:] -= LinOp_r@(utmp[(j+1)*ndofs_if:(j+2)*ndofs_if,:])
    if u.ndim == 1:
        out = out.flatten()
    return out

A_balance = LinearOperator(shape=(ndslab*ndofs_if, ndslab*ndofs_if),
                            matvec=apply_balance, dtype=float)

N = A_balance.shape[0]
v = np.random.standard_normal((N,))
tic =time.time()
bb = A_balance@v
print("matvec time = ",time.time()-tic)
gInfo = gmres_info()
u = u_true
res = A_balance@u-rhs
print("res = ",np.linalg.norm(res))
if gmres_iters > 0:
    tic = time.time()
    uhat,_   = gmres(A_balance,rhs,rtol=1e-8,callback=gInfo,maxiter=gmres_iters,restart=gmres_iters)
    niter = gInfo.niter
    print("time = ",time.time()-tic)
    print("niter = ",niter)
    print("u err = ",np.linalg.norm(uhat-u)/np.linalg.norm(u))
else:
    print("GMRES solve skipped (gmres_iters = 0)")

Sdense = A_balance@np.identity(A_balance.shape[0])
print("condition number = ",np.linalg.cond(Sdense))