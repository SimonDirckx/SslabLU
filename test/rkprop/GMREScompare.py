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


######################################################
#                   HELPER FUNCTIONS
######################################################

def compute_Tmaps(slabGeom):
    solver_S = stencil.stencilSolver(op_pdo, slabGeom, ord_)
    Aii,Aib,Abi,Abb,Ni,XXi,XXb,Ji,Jb = _solverdata(solver_S)
    XYtot = solver_S.XX
    tol = 1e-9
    xl = slabGeom.bounds[0,0]
    xr = slabGeom.bounds[1,0]
    Jl = np.where(np.abs(XXb[:,0]-xl) < tol)[0]               # physical x = 0  face
    Jr = np.where(np.abs(XXb[:,0]-xr) < tol)[0]               # physical x = Lx face
    Jb = np.setdiff1d(np.arange(XXb.shape[0]),
                        np.concatenate([Jl, Jr])).astype(np.int64)   # the four (y,z) faces

    ndofs_if = len(Jl)                                  # size of one interface block

    # ring = interface-face nodes on the (y,z) physical boundary (their conormal
    # flux is ill-defined and is replaced by the known Dirichlet trace); inn =
    # the strictly-interior interface nodes that carry the flux balance.
    XXl = XXb[Jl, :]
    ring = np.where((XXl[:, 1] < tol) | (XXl[:, 1] > Ly - tol) |
                    (XXl[:, 2] < tol) | (XXl[:, 2] > Lz - tol))[0]
    inn  = np.where((XXl[:, 1] > tol) & (XXl[:, 1] < Ly - tol) &
                    (XXl[:, 2] > tol) & (XXl[:, 2] < Lz - tol))[0]

    tic = time.time()
    BLK = 32                                   # tune; see note below
    ctx  = setup_mumps(Aii, blr=False)
    ctxT = setup_mumps_transpose(Aii, blr=False)
    tMUMPS = time.time()-tic
    

    ctx.mumps_instance.icntl[27]  = BLK        # one wide BLAS-3 block per chunk
    ctxT.mumps_instance.icntl[27] = BLK

    def tmatmat(v, J1,J0, transpose=False):
        v_tmp = v[:, None] if v.ndim == 1 else v
        k = v_tmp.shape[1]

        if not transpose:
            Aib_J = Aib[:, J0].tocsc()
            out = np.zeros((ndofs_if, k))
            for s in range(0, k, BLK):
                c = slice(s, min(s + BLK, k))
                rhs = (Aib_J @ sparse.csc_matrix(v_tmp[:, c])).tocsc()
                sol = ctx._solve_sparse(rhs)
                out[:, c] =  Abb[J1][:,J0]@v_tmp[:, c] - Abi[J1]@sol  # conormal Schur block: +Abb - Abi Aii^{-1} Aib
                del rhs, sol
                ctx.mumps_instance.icntl[20] = 0
        else:
            Aib_J = Aib[:, J0].tocsc()
            out = np.zeros((ndofs_if, k))
            for s in range(0, k, BLK):
                c = slice(s, min(s + BLK, k))
                rhs = (Aib_J @ sparse.csc_matrix(v_tmp[:, c])).tocsc()
                sol = ctx._solve_sparse(rhs)
                out[:, c] =  Abb[J1][:,J0]@v_tmp[:, c] - Abi[J1]@sol  # conormal Schur block: +Abb - Abi Aii^{-1} Aib
                del rhs, sol
                ctx.mumps_instance.icntl[20] = 0
        return out.flatten() if v.ndim == 1 else out

    LinOp_rr = LinearOperator(shape=(len(Jr),len(Jr)),\
        matvec = lambda v:tmatmat(v,Jr,Jr), rmatvec = lambda v:tmatmat(v,Jr,Jr,transpose=True),\
        matmat = lambda v:tmatmat(v,Jr,Jr), rmatmat = lambda v:tmatmat(v,Jr,Jr,transpose=True))
    LinOp_rl = LinearOperator(shape=(len(Jr),len(Jl)),\
        matvec = lambda v:tmatmat(v,Jr,Jl), rmatvec = lambda v:tmatmat(v,Jr,Jl,transpose=True),\
        matmat = lambda v:tmatmat(v,Jr,Jl), rmatmat = lambda v:tmatmat(v,Jr,Jl,transpose=True))
    LinOp_lr = LinearOperator(shape=(len(Jl),len(Jr)),\
        matvec = lambda v:tmatmat(v,Jl,Jr), rmatvec = lambda v:tmatmat(v,Jl,Jr,transpose=True),\
        matmat = lambda v:tmatmat(v,Jl,Jr), rmatmat = lambda v:tmatmat(v,Jl,Jr,transpose=True))
    LinOp_ll = LinearOperator(shape=(len(Jl),len(Jl)),\
        matvec = lambda v:tmatmat(v,Jl,Jl), rmatvec = lambda v:tmatmat(v,Jl,Jl,transpose=True),\
        matmat = lambda v:tmatmat(v,Jl,Jl), rmatmat = lambda v:tmatmat(v,Jl,Jl,transpose=True))
    return LinOp_rr,LinOp_rl,LinOp_lr,LinOp_ll,Aib,Abi,Abb,Jl,Jr,Jb,ring,inn,XYtot,XXi,XXb,ctx,ctxT

def compute_Sl_and_Sr(slabGeom):
    solver_S = stencil.stencilSolver(op_pdo, slabGeom, ord_)
    Sii,Sib,Abi,Abb,Ni,XXi,XXb,Ji,Jb = _solverdata(solver_S)
    XYtot = solver_S.XX
    tol = 1e-9
    xl = slabGeom.bounds[0,0]
    xr = slabGeom.bounds[1,0]
    cx = (xr+xl)/2
    Jc = np.where(np.abs(XXi[:,0]-cx) < tol)[0]               # interior interface DOFs
    Jl = np.where(np.abs(XXb[:,0]-xl) < tol)[0]               # physical x = 0  face
    XXl = XXb[Jl,:]
    Jr = np.where(np.abs(XXb[:,0]-xr) < tol)[0]               # physical x = Lx face
    Jb = np.setdiff1d(np.arange(XXb.shape[0]),
                        np.concatenate([Jl, Jr])).astype(np.int64)   # the four (y,z) faces

    Jc_large = np.where(np.abs(XYtot[:,0]-cx) < tol)[0]       # full interface plane
    Jc_inJc =  np.where((XYtot[Jc_large,1] > tol) &\
                    (XYtot[Jc_large,1] < Ly-tol) &\
                    (XYtot[Jc_large,2] > tol) &\
                    (XYtot[Jc_large,2] < Lz-tol))[0]   # x = Lx/2
    ndofs_if = len(Jc_large)                                  # size of one interface block

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
                    shape=(len(Ji), bw))
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
    return LinOp_l,LinOp_r,Sib,scatter,Jc,Jl,Jr,Jb,Jc_large,Jc_inJc,XYtot,XXi,XXb,ctx,ctxT



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


######################################################
#                   MAIN BODY
######################################################



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
_parser.add_argument("--pde", choices=["greens", "convdiff", "vardiff"],
                     default="greens",
                     help="which PDE to test. greens (default): Laplace (kh=0) / "
                          "Helmholtz (kh>0) with a free-space Green's function as "
                          "the exact solution and zero interior forcing -- this is "
                          "the original behaviour and runs all three solvers. "
                          "convdiff: convection-diffusion eps*Lap u + beta.grad u "
                          "with a manufactured solution. vardiff: "
                          "variable-coefficient (non-divergence form) diffusion "
                          "c11(x)u_xx+c22(x)u_yy+c33(x)u_zz with a manufactured "
                          "solution. The manufactured PDEs carry a nonzero body "
                          "force and are exercised on the global solver only.")
_parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.1,
                     help="diffusion coefficient eps for --pde convdiff")
_parser.add_argument("--beta", dest="beta", nargs="+", default=["1", "0.5", "0.25"],
                     help="convection vector b1 b2 b3 for --pde convdiff "
                          "(space- or comma-separated; fractions like 1/2 allowed)")
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


beta = _nums3(args.beta)

class _PDO_local(object):
    def __init__(self, c11, c22, c33, c=None, c1=None, c2=None, c3=None):
        self.c11, self.c22, self.c33 = c11, c22, c33
        self.c, self.c1, self.c2, self.c3 = c, c1, c2, c3

# Separable manufactured solution, nonzero on the unit-cube boundary so the
# Dirichlet trace is nontrivial.  u* = sin(a x+pa) sin(b y+pb) sin(c z+pc).
_mA, _mB, _mC = 0.7*np.pi, 1.3*np.pi, 0.9*np.pi
_pA, _pB, _pC = 0.3, 0.7, 0.5
def _u_mms(p):
    return (np.sin(_mA*p[:,0]+_pA) * np.sin(_mB*p[:,1]+_pB) * np.sin(_mC*p[:,2]+_pC))
def _ux(p):  return _mA*np.cos(_mA*p[:,0]+_pA)*np.sin(_mB*p[:,1]+_pB)*np.sin(_mC*p[:,2]+_pC)
def _uy(p):  return _mB*np.sin(_mA*p[:,0]+_pA)*np.cos(_mB*p[:,1]+_pB)*np.sin(_mC*p[:,2]+_pC)
def _uz(p):  return _mC*np.sin(_mA*p[:,0]+_pA)*np.sin(_mB*p[:,1]+_pB)*np.cos(_mC*p[:,2]+_pC)
def _uxx(p): return -_mA**2 * _u_mms(p)
def _uyy(p): return -_mB**2 * _u_mms(p)
def _uzz(p): return -_mC**2 * _u_mms(p)

if args.pde == "greens":
    op_pdo     = HH_np
    u_exact_fn = lambda p: bc_helmholtz(p, kh)
    forcing_fn = lambda p: np.zeros(p.shape[0])
elif args.pde == "convdiff":
    eps = args.epsilon
    b1, b2, b3 = beta
    _ones = lambda p: np.ones_like(p[:, 0])
    op_pdo = _PDO_local(
        c11=lambda p: eps*_ones(p), c22=lambda p: eps*_ones(p),
        c33=lambda p: eps*_ones(p),
        c1=lambda p: b1*_ones(p), c2=lambda p: b2*_ones(p),
        c3=lambda p: b3*_ones(p))
    u_exact_fn = _u_mms
    forcing_fn = lambda p: (eps*(_uxx(p) + _uyy(p) + _uzz(p))
                            + b1*_ux(p) + b2*_uy(p) + b3*_uz(p))
    print("convection-diffusion: eps = %g, beta = (%g, %g, %g)"
          % (eps, b1, b2, b3))
elif args.pde == "vardiff":
    _a11 = lambda p: 1.0 + 0.5*np.cos(np.pi*p[:, 0])
    _a22 = lambda p: 1.0 + 0.5*np.cos(np.pi*p[:, 1])
    _a33 = lambda p: 1.0 + 0.5*np.cos(np.pi*p[:, 2])
    op_pdo     = _PDO_local(c11=_a11, c22=_a22, c33=_a33)
    u_exact_fn = _u_mms
    forcing_fn = lambda p: (_a11(p)*_uxx(p) + _a22(p)*_uyy(p) + _a33(p)*_uzz(p))
    print("variable-coefficient diffusion: c_ii(x) = 1 + 0.5*cos(pi*x_i)")
else:
    raise SystemExit("unknown --pde %s" % args.pde)

print("PDE = %s" % args.pde)


nx, ny, nz = order
print(  "============ GLOBAL SOLVER: ============" )
unitCube  = geom.BoxGeometry(np.array([[0,0,0],[1,1,1]]))
ord_ = [nx, ny, nz]

solver_glob = stencil.stencilSolver(op_pdo, unitCube, ord_)



print("solver construction done")
# ---- four blocks (interior solve + conormal boundary rows) ----


Aii,Aib,Abi,Abb,Ni,XXi,XXb,Ji,Jb = _solverdata(solver_glob)
ctx = setup_mumps(Aii)

ui  = u_exact_fn(solver_glob.XXi)
ub  = u_exact_fn(solver_glob.XXb)
fi  = forcing_fn(solver_glob.XXi)        # interior body force (0 for greens)
rhs = fi - Aib@ub


gInfo = gmres_info()
if gmres_iters > 0:
    tic = time.time()
    uhat, _ = gmres(Aii, rhs, rtol=1e-8, callback=gInfo,
                    maxiter=gmres_iters, restart=gmres_iters)
    solve_time_LU = time.time() - tic
    niter = gInfo.niter
    gmres_err = np.linalg.norm(uhat - ui) / np.linalg.norm(ui)
    gmres_res = np.linalg.norm(rhs - Aii @ uhat) / np.linalg.norm(rhs)
    print("time = ", solve_time_LU)
    print("niter = ", niter)
    print("gmres rel residual = ", gmres_res)
    print("u err = ", gmres_err)
else:
    solve_time_LU = float('nan'); niter = float('nan'); gmres_err = float('nan')
    print("GMRES solve skipped (gmres_iters = 0)")
print("condition number = ",_cond(Aii,ctx))

print(  "============   T SOLVER:   ============" )

nx = int(ny * Lx) + 1            # single-width slab; nx so x=Lx lands on-grid
print("stencil nx (derived from ny, Lx) = ", nx)
ord_ = [nx, ny, nz]

nslab = int(round(1./Lx))
ndofs_if = ny*nz

# Per-slab conormal-Schur maps.  Each slab [j*Lx,(j+1)*Lx] is assembled at its
# ABSOLUTE location (like the S solver) so variable coefficients are handled;
# for constant coefficients every slab is a translate of the reference slab.
Trr_l=[]; Trl_l=[]; Tlr_l=[]; Tll_l=[]
Aib_l=[]; Abi_l=[]; Abb_l=[]; ctx_l=[]
Jl_l=[]; Jr_l=[]; JJ_l=[]; XXb_l=[]; XXi_l=[]
tic = time.time()
for j in range(nslab):
    slabGeom = geom.BoxGeometry(np.array([[0,0,0],[Lx,Ly,Lz]]) +
                                np.array([j*Lx, 0., 0.]))
    (Trr,Trl,Tlr,Tll, Aib_j,Abi_j,Abb_j, Jl,Jr,JJ, ring,inn,
     XYtot,XXi,XXb, ctx,ctxT) = compute_Tmaps(slabGeom)
    Trr_l.append(Trr); Trl_l.append(Trl); Tlr_l.append(Tlr); Tll_l.append(Tll)
    Aib_l.append(Aib_j); Abi_l.append(Abi_j); Abb_l.append(Abb_j); ctx_l.append(ctx)
    Jl_l.append(Jl); Jr_l.append(Jr); JJ_l.append(JJ)
    XXb_l.append(XXb); XXi_l.append(XXi)
print("MUMPS factorization (all slabs) time = ", time.time()-tic)

nif     = len(Jl_l[0])              # interface block size  (= ny*nz)
nIF     = nslab - 1                 # number of interior interfaces
hx_slab = Lx/(nx-1)                 # normal spacing; conormal rows scale as -hx
ring    = ring                      # interface ring (same y/z order for every slab)

# ---- block-tridiagonal flux-balance operator (per-slab conormal Schur maps) ----
# Interface j sits between slab j (right face Jr) and slab j+1 (left face Jl).
# Diagonal:  Trr[j] + Tll[j+1];  sub-diag: Trl[j] (u_{j-1}); super: Tlr[j+1] (u_{j+1}).
# Ring rows are identity-pinned (their conormal flux is replaced by the known trace).
def apply_balance(u):
    utmp = u[:, None] if u.ndim == 1 else u
    out  = np.zeros_like(utmp)
    for j in range(nIF):
        b  = j*nif
        uj = utmp[b:b+nif]
        blk = Trr_l[j] @ uj + Tll_l[j+1] @ uj
        if j > 0:
            blk = blk + Trl_l[j]   @ utmp[b-nif:b]
        if j < nIF-1:
            blk = blk + Tlr_l[j+1] @ utmp[b+nif:b+2*nif]
        blk[ring] = uj[ring]
        out[b:b+nif] = blk
    return out.ravel() if u.ndim == 1 else out

A_balance = LinearOperator(shape=(nIF*nif, nIF*nif),
                           matvec=apply_balance, dtype=np.float64)

# ---- rhs: known-data flux + particular flux (interior body force) +
#           interface-node body force (-hx*f), ring identity-pinned. ----
def data_flux(j, J1, knownfaces, data):
    """Conormal flux on face J1 of slab j from known Dirichlet data on knownfaces:
       (Abb - Abi Aii^{-1} Aib)[J1, knownfaces] @ data."""
    Abi=Abi_l[j]; Aib=Aib_l[j]; Abb=Abb_l[j]; ctx=ctx_l[j]
    return Abb[J1][:,knownfaces]@data - Abi[J1]@ctx.solve(Aib[:,knownfaces]@data)

def part_flux(j, J1, f_int):
    """Particular conormal flux on face J1 of slab j from interior body force:
       Abi[J1] Aii^{-1} f_int   (zero for the homogeneous Green's-function case)."""
    return Abi_l[j][J1] @ ctx_l[j].solve(f_int)

rhs  = np.zeros(nIF*nif)
u_if = np.zeros(nIF*nif)
XXif = np.zeros((nIF*nif, 3))
for j in range(nIF):
    # left slab j -> flux on its Jr face (= interface j)
    JlL, JrL, JJL = Jl_l[j], Jr_l[j], JJ_l[j]
    knownL = np.concatenate([JJL, JlL]) if j == 0 else JJL
    dataL  = u_exact_fn(XXb_l[j][knownL])
    fL = (data_flux(j, JrL, knownL, dataL)
          + part_flux(j, JrL, forcing_fn(XXi_l[j])))

    # right slab j+1 -> flux on its Jl face (= interface j)
    JlR, JrR, JJR = Jl_l[j+1], Jr_l[j+1], JJ_l[j+1]
    knownR = np.concatenate([JJR, JrR]) if (j+1) == nslab-1 else JJR
    dataR  = u_exact_fn(XXb_l[j+1][knownR])
    fR = (data_flux(j+1, JlR, knownR, dataR)
          + part_flux(j+1, JlR, forcing_fn(XXi_l[j+1])))

    b = -(fL + fR)
    # interface nodes are BOUNDARY nodes of the non-overlapping slabs, so their
    # own body force is not contained in any slab interior -> add it explicitly.
    coords_if = XXb_l[j][JrL]
    b = b - hx_slab * forcing_fn(coords_if)

    uif = u_exact_fn(coords_if)
    b[ring] = uif[ring]
    rhs[j*nif:(j+1)*nif]      = b
    u_if[j*nif:(j+1)*nif]     = uif
    XXif[j*nif:(j+1)*nif, :]  = coords_if

# sanity: the exact interface trace should nearly satisfy the balance system
print("res (||A u_if - rhs||) = ", np.linalg.norm(A_balance@u_if - rhs))

gInfo = gmres_info()
if gmres_iters > 0:
    tic = time.time()
    uhat, _ = gmres(A_balance, rhs, rtol=1e-8, callback=gInfo,
                    maxiter=gmres_iters, restart=gmres_iters)
    solve_time_LU = time.time() - tic
    niter = gInfo.niter
    gmres_err = np.linalg.norm(uhat - u_if) / np.linalg.norm(u_if)
    gmres_res = np.linalg.norm(rhs - A_balance @ uhat) / np.linalg.norm(rhs)
    print("time = ", solve_time_LU)
    print("niter = ", niter)
    print("gmres rel residual = ", gmres_res)
    print("u err = ", gmres_err)
else:
    print("GMRES solve skipped (gmres_iters = 0)")

print(  "============   S SOLVER:   ============" )
cx = Lx
nx = int(2 * ny * Lx) + 1            # single-width slab; nx so x=Lx lands on-grid
print("stencil nx (derived from ny, Lx) = ", nx)
ord_ = [nx , ny, nz]


ndslab = int(round(1./cx)) - 1
ndofs_if = ny*nz
XXif = np.zeros((ndslab*ndofs_if,3))
rhs  = np.zeros((ndslab*ndofs_if,))
u_true = np.zeros((ndslab*ndofs_if,))
Llist = []
Rlist = []
for i in range(ndslab):

    shift = np.array([i*cx, 0., 0.])
    slabGeom = geom.BoxGeometry(np.array([[0,0,0],[2*Lx,Ly,Lz]])+shift)
    LinOp_l,LinOp_r,Sib,scatter,Jc,Jl,Jr,Jb,Jc_large,Jc_inJc,XYtot,XXi,XXb,ctx,ctxT = compute_Sl_and_Sr(slabGeom)
    Llist+=[LinOp_l]
    Rlist+=[LinOp_r]

    XXif[i*ndofs_if:(i+1)*ndofs_if, :] = XYtot[Jc_large]
    trace_full = u_exact_fn(XYtot[Jc_large])           # interior AND ring
    u_true[i*ndofs_if:(i+1)*ndofs_if]  = trace_full
    uj   = u_exact_fn(XXb)                              # Dirichlet trace on slab faces
    f_int = forcing_fn(XXi)                            # body force at slab interior pts

    if i == 0:                        # physical x=0 face known -> keep Jl, drop Jr
        Jb0 = np.setdiff1d(np.arange(XXb.shape[0]), Jr).astype(np.int64)
        br = u_exact_fn(XXb[Jr])
        ub_loc_rc = u_exact_fn(XXb)[Jb0]
        blk = ctx.solve(f_int - Sib[:, Jb0] @ ub_loc_rc)[Jc]        

    elif i == ndslab - 1:             # physical far face known -> keep Jr, drop Jl
        Jb0 = np.setdiff1d(np.arange(XXb.shape[0]), Jl).astype(np.int64)
        bl = u_exact_fn(XXb[Jl])      
        ub_loc_lc = u_exact_fn(XXb)[Jb0]
        blk = ctx.solve(f_int - Sib[:, Jb0] @ ub_loc_lc)[Jc]
    else:
        Jb0 = np.array([i for i in range(XXb.shape[0]) if not i in Jl and not i in Jr],dtype=np.int64)
        ub_loc_rlc = u_exact_fn(XXb)[Jb0]
        blk = ctx.solve(f_int - Sib[:, Jb0] @ ub_loc_rlc)[Jc]

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
        if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] -= Llist[j]@(utmp[(j-1)*ndofs_if:j*ndofs_if,:])
        if j < ndslab-1:   out[j*ndofs_if:(j+1)*ndofs_if,:] -= Rlist[j]@(utmp[(j+1)*ndofs_if:(j+2)*ndofs_if,:])
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

#Sdense = A_balance@np.identity(A_balance.shape[0])
#print("condition number = ",np.linalg.cond(Sdense))