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
import time
import os
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import splu
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
slabGeom = geom.BoxGeometry(np.array([[0,0,0],[Lx,Ly,Lz]]))


kh = args.kh

def  c11(p):
    return torch.ones_like(p[...,0])
def  c22(p):
    return torch.ones_like(p[...,0])
def  c33(p):
    return torch.ones_like(p[...,0])
def  c(p):
    return -kh*kh*torch.ones_like(p[...,0])
HH = pdo.PDO_3d(c11=c11,c22=c22,c33=c33,c=c)


print("============BUILDING SOLVER============")
print(f"type={solve_method}  order={order}  shape=({Lx},{Ly},{Lz})  "
      f"admissibility={admissibility}")

# ###########################################################################
# #############################   HPS SOLVER   ##############################
# ###########################################################################
if solve_method == 'HPS':
    px, py, pz = order            # polynomial order per block (CLI --order)
    nbx = int(Lx*nb)                       # 2 blocks in x -> interface at the centre x = cx
    nby = nb
    nbz = nb
    
    # HPS solver here
    a = np.array([Lx/2/nbx,Ly/2/nby,Lz/2/nbz])
    solver = hps_solver.Domain_Driver(slabGeom,HH,kh,a,np.array([px+2,py+2,pz+2],dtype=np.int64),d=3)
    tic = time.time()
    solver.build("reduced_cpu", "MUMPS")
    print("Solver done in ",time.time()-tic,"s")


    Ji = solver._Ji
    Jb = solver._Jx
    Aib = solver.Aix
    Abi = solver.Axi
    Abb = solver.Axx
    Aii = solver.Aii
    
    XYtot = solver.XX
    XXi = XYtot[Ji,:]
    XXb = XYtot[Jb,:]
    
    Jl = (torch.where(XXb[...,0]==0)[0]).detach().cpu().numpy()
    Jr = (torch.where(XXb[...,0]==Lx)[0]).detach().cpu().numpy()
    Jlc = np.array([i for i in range(len(Jb)) if not i in Jl])
    Jrc = np.array([i for i in range(len(Jb)) if not i in Jr])
    XXl = XXb[Jl,:]
    XXr = XXb[Jr,:]

    XY_shift = XYtot+torch.from_numpy(np.array([Lx,0,0]))
    XXb_shift = XXb +torch.from_numpy(np.array([Lx,0,0]))

    solver_ii = solver.solver_Aii
    
    print("|Jl| = ",len(Jl))
    print("|Jr| = ",len(Jr))


    def txx_matmat(v,J1,J0, transpose=False):
        v_tmp = v[..., None] if v.ndim == 1 else v

        if not transpose:
            out = Abb[J1,...][...,J0]@v_tmp - Abi[J1,...]@(solver_ii@(Aib[...,J0]@v_tmp))
        else:
            out = (Abb[J1,...][...,J0]).T@v_tmp - Aib[...,J0].T@(solver_ii.T@(Abi[J1,...].T@v_tmp))
        return out.flatten() if v.ndim == 1 else out

    def txx_matmat_rem(v,J1,J0, transpose=False):
        """Smoothing remainder of the interface map: T_{J1 J0} minus its local
        boundary block Abb[J1,J0].  Equals  -Abi[J1] Aii^{-1} Aib[:,J0], the
        purely interior-mediated (integral, asymptotically smooth) part.  This is
        what gets HBS-compressed for the self blocks when --splitting is on; the
        local Abb[Jx,Jx] block is kept exact and added back at apply time."""
        v_tmp = v[..., None] if v.ndim == 1 else v
        if not transpose:
            out = - Abi[J1,...]@(solver_ii@(Aib[...,J0]@v_tmp))
        else:
            out = - Aib[...,J0].T@(solver_ii.T@(Abi[J1,...].T@v_tmp))
        return out.flatten() if v.ndim == 1 else out

    # lr / rl means "left from right" / "right from left"

    LinOp_ll = LinearOperator(shape=(len(Jl),len(Jl)),\
        matvec = lambda v:txx_matmat(v,Jl,Jl), rmatvec = lambda v:txx_matmat(v,Jl,Jl,transpose=True),\
        matmat = lambda v:txx_matmat(v,Jl,Jl), rmatmat = lambda v:txx_matmat(v,Jl,Jl,transpose=True))
    LinOp_rr = LinearOperator(shape=(len(Jr),len(Jr)),\
        matvec = lambda v:txx_matmat(v,Jr,Jr), rmatvec = lambda v:txx_matmat(v,Jr,Jr,transpose=True),\
        matmat = lambda v:txx_matmat(v,Jr,Jr), rmatmat = lambda v:txx_matmat(v,Jr,Jr,transpose=True))
    LinOp_lr = LinearOperator(shape=(len(Jl),len(Jr)),\
        matvec = lambda v:txx_matmat(v,Jl,Jr), rmatvec = lambda v:txx_matmat(v,Jl,Jr,transpose=True),\
        matmat = lambda v:txx_matmat(v,Jl,Jr), rmatmat = lambda v:txx_matmat(v,Jl,Jr,transpose=True))
    
    LinOp_rl = LinearOperator(shape=(len(Jr),len(Jr)),\
        matvec = lambda v:txx_matmat(v,Jr,Jl), rmatvec = lambda v:txx_matmat(v,Jr,Jl,transpose=True),\
        matmat = lambda v:txx_matmat(v,Jr,Jl), rmatmat = lambda v:txx_matmat(v,Jr,Jl,transpose=True))

    # remainder (smoothing-only) operators for the SELF blocks; sampled by HBS
    # when --splitting is on so the construction never sees the local Abb stencil.
    LinOp_ll_rem = LinearOperator(shape=(len(Jl),len(Jl)),\
        matvec = lambda v:txx_matmat_rem(v,Jl,Jl), rmatvec = lambda v:txx_matmat_rem(v,Jl,Jl,transpose=True),\
        matmat = lambda v:txx_matmat_rem(v,Jl,Jl), rmatmat = lambda v:txx_matmat_rem(v,Jl,Jl,transpose=True))
    LinOp_rr_rem = LinearOperator(shape=(len(Jr),len(Jr)),\
        matvec = lambda v:txx_matmat_rem(v,Jr,Jr), rmatvec = lambda v:txx_matmat_rem(v,Jr,Jr,transpose=True),\
        matmat = lambda v:txx_matmat_rem(v,Jr,Jr), rmatmat = lambda v:txx_matmat_rem(v,Jr,Jr,transpose=True))
    
    
    
    
    u_tot_left = bc_helmholtz(XYtot,kh)
    u_tot_right = bc_helmholtz(XY_shift,kh)
    ub_l = bc_helmholtz(XXb[Jrc,...],kh)
    ur_l = bc_helmholtz(XXb[Jr,...],kh)
    ub_r = bc_helmholtz(XXb_shift[Jlc,...],kh)
    ul_r = bc_helmholtz(XXb_shift[Jl,...],kh)
    Trb_l = Abb[Jr,...][...,Jrc]@ub_l - Abi[Jr,...]@(solver_ii@(Aib[...,Jrc]@ub_l))
    Tlb_r = Abb[Jl,...][...,Jlc]@ub_r - Abi[Jl,...]@(solver_ii@(Aib[...,Jlc]@ub_r))

    Trr_l = LinOp_rr@ur_l
    Tll_r = LinOp_ll@ul_r
    def _np(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    XXb_np, XXi_np = _np(XXb), _np(XXi)
    ub = bc_helmholtz(XXb_np, kh)          # exact boundary trace
    ui_exact = bc_helmholtz(XXi_np, kh)    # exact interior values

    
    print("equilib1 = ",np.linalg.norm(Trr_l+Tll_r+Trb_l+Tlb_r,ord=np.inf))
    nSlab = int(1/Lx)
    print("nSlab = ",nSlab)
    ndofs_if = len(Jl)
    def apply_balance(u):
        if u.ndim == 1:
            utmp = u[:,None]
        else:
            utmp = u
        out = torch.zeros(utmp.shape)
        for j in range(nSlab-1):
            out[j*ndofs_if:(j+1)*ndofs_if,:] = torch.from_numpy( LinOp_ll@utmp[j*ndofs_if:(j+1)*ndofs_if,:]+LinOp_rr@utmp[j*ndofs_if:(j+1)*ndofs_if,:])
            if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] += torch.from_numpy(LinOp_rl@(utmp[(j-1)*ndofs_if:j*ndofs_if,:]))
            if j < nSlab-2:      out[j*ndofs_if:(j+1)*ndofs_if,:] += torch.from_numpy(LinOp_lr(utmp[(j+1)*ndofs_if:(j+2)*ndofs_if,:]))
        if u.ndim == 1:
            out = out.flatten()
        return out
    rhs = torch.zeros(((nSlab-1)*ndofs_if))
    u_if = torch.zeros(((nSlab-1)*ndofs_if))
    JJ = np.array([i for i in range(len(Jb)) if i not in Jl and i not in Jr])
    Jlc = np.array([i for i in range(len(Jb)) if i not in Jl])
    Jrc = np.array([i for i in range(len(Jb)) if i not in Jr])
    Trb = lambda u: Abb[Jr,...][...,JJ]@u - Abi[Jr,...]@(solver_ii@(Aib[...,JJ]@u))
    Tlb = lambda u: Abb[Jl,...][...,JJ]@u - Abi[Jl,...]@(solver_ii@(Aib[...,JJ]@u))

    Trb_l = lambda u: Abb[Jr,...][...,Jrc]@u - Abi[Jr,...]@(solver_ii@(Aib[...,Jrc]@u))
    Tlb_r = lambda u: Abb[Jl,...][...,Jlc]@u - Abi[Jl,...]@(solver_ii@(Aib[...,Jlc]@u))
    XXif = np.zeros(((nSlab-1)*ndofs_if,3))
    for j in range(nSlab):
        XXbloc = XXb+j*torch.from_numpy(np.array([Lx,0,0]))
        
        if j == 0 :
            ub_loc = bc_helmholtz(XXbloc[Jrc,:],kh) 
            rhs[j*ndofs_if:(j+1)*ndofs_if]-= Trb_l(ub_loc)
        elif j == nSlab-1:
            ub_loc = bc_helmholtz(XXbloc[Jlc,:],kh) 
            rhs[(j-1)*ndofs_if:j*ndofs_if]-= Tlb_r(ub_loc)
        else:
            ub_loc = bc_helmholtz(XXbloc[JJ,:],kh)
            rhs[j*ndofs_if:(j+1)*ndofs_if]-= Trb(ub_loc)
            rhs[(j-1)*ndofs_if:j*ndofs_if]-= Tlb(ub_loc)
        if j<nSlab-1:
            u_if[j*ndofs_if:(j+1)*ndofs_if] = bc_helmholtz(XXbloc[Jr,:],kh)
            XXif[j*ndofs_if:(j+1)*ndofs_if,:] = XXbloc[Jr,:]


    A_balance = LinearOperator(shape=((nSlab-1)*ndofs_if, (nSlab-1)*ndofs_if),
                               matvec=apply_balance, dtype=float)
    print("res = ",torch.linalg.norm(rhs-A_balance@u_if))


    if torch.cuda.is_available():
        device = 'cuda'          # or 'cuda:0' to pin a specific GPU
    else:
        device = 'cpu'
    if admissibility=='weak':
        tree = slabTree.slabTree(XXl,False,py*pz)
        TTrr = HBStorch.HBSMAT(device=device,tree=tree)
        TTll = HBStorch.HBSMAT(device=device,tree=tree)
        TTlr = HBStorch.HBSMAT(device=device,tree=tree)
        TTrl = HBStorch.HBSMAT(device=device,tree=tree)
        kmax = 1
    else:
        tree = slabTree.slabTree(XXl,False,py*pz,adjacency=admissibility)
        TTrr = HBStorch_strong.HBSMAT(device=device,tree=tree)
        TTll = HBStorch_strong.HBSMAT(device=device,tree=tree)
        TTlr = HBStorch_strong.HBSMAT(device=device,tree=tree)
        TTrl = HBStorch_strong.HBSMAT(device=device,tree=tree)
        if admissibility == 'strong': 
            kmax = 9 
        else: 
            kmax = 5
    
    nl = len(tree.get_box_inds(tree.get_leaves()[0]))

    s = kmax*max(2*rk,nl)+rk+10
    N = LinOp_rr.shape[0]
    
    Nb = N//nl
    if splitting:
        assert len(Jl) == len(Jr), \
            "splitting assumes |Jl| == |Jr| so D_ll and D_rr act on the same interface block"
        # exact local in-face blocks: plain slices of Abb (NOT Schur complements, no
        # Aii^{-1}); these carry the singular/differential part of the self maps and
        # are kept exact, added back at apply time.
        D_rr = Abb[Jr,...][...,Jr]
        D_ll = Abb[Jl,...][...,Jl]
        # construct the self-block HBS from the smoothing remainder -Abi Aii^{-1} Aib only
        src_rr, src_ll = LinOp_rr_rem, LinOp_ll_rem
    else:
        src_rr, src_ll = LinOp_rr, LinOp_ll
    tHBS = 0
    tSample = 0
    tic = time.time()
    Om = np.random.standard_normal((N,s))
    Psi = np.random.standard_normal((N,s))
    tSample+=time.time()-tic
    Y = src_rr@Om
    Z = src_rr.T@Psi
    
    tic=time.time()
    tic = time.time()
    TTrr.construct(rk,Om,Psi,Y,Z,fast=True)
    tHBS += time.time() - tic

    tic=time.time()
    Om = np.random.standard_normal((N,s))
    Psi = np.random.standard_normal((N,s))
    Y = src_ll@Om
    Z = src_ll.T@Psi
    tSample+=time.time()-tic
    tic = time.time()
    TTll.construct(rk,Om,Psi,Y,Z,fast=True)
    tHBS += time.time() - tic
    
    tic=time.time()
    Om = np.random.standard_normal((N,s))
    Psi = np.random.standard_normal((N,s))
    Y = LinOp_rl@Om
    Z = LinOp_rl.T@Psi
    tSample+=time.time()-tic
    tic = time.time()
    TTrl.construct(rk,Om,Psi,Y,Z,fast=True)
    tHBS += time.time() - tic

    tic=time.time()
    Om = np.random.standard_normal((N,s))
    Psi = np.random.standard_normal((N,s))
    Y = LinOp_lr@Om
    Z = LinOp_lr.T@Psi
    tSample+=time.time()-tic
    tic = time.time()
    TTlr.construct(rk,Om,Psi,Y,Z,fast=True)
    tHBS += time.time() - tic
    
    
    

    # Effective self maps. Without splitting these are just the HBS blocks; with
    # splitting the exact local block D_xx is kept and only the remainder is HBS,
    # so the self map is D_xx + TT.  err_rr/err_ll below compare these against the
    # TRUE self Schur maps LinOp_rr/LinOp_ll, so under splitting they report the
    # remainder-compression error (expected ~ the cross-block level).
    if splitting:
        def selfmat_rr(x): return _np(D_rr@x) + _np(TTrr@x)
        def selfmat_ll(x): return _np(D_ll@x) + _np(TTll@x)
    else:
        def selfmat_rr(x): return _np(TTrr@x)
        def selfmat_ll(x): return _np(TTll@x)

    v = np.random.standard_normal((N,10))
    err_rr = np.linalg.norm(selfmat_rr(v)-_np(LinOp_rr@v))/np.linalg.norm(v)
    err_ll = np.linalg.norm(selfmat_ll(v)-_np(LinOp_ll@v))/np.linalg.norm(v)
    err_lr = np.linalg.norm(TTlr@v-LinOp_lr@v)/np.linalg.norm(v)
    err_rl = np.linalg.norm(TTrl@v-LinOp_rl@v)/np.linalg.norm(v)
    print("err_rr = ",err_rr)
    print("err_ll = ",err_ll)
    print("err_lr = ",err_lr)
    print("err_rl = ",err_rl)

    def apply_balance_HBS(u):
        if u.ndim == 1:
            utmp = u[:,None]
        else:
            utmp = u
        out = torch.zeros(utmp.shape)
        for j in range(nSlab-1):
            out[j*ndofs_if:(j+1)*ndofs_if,:] = torch.from_numpy( selfmat_ll(utmp[j*ndofs_if:(j+1)*ndofs_if,:]) + selfmat_rr(utmp[j*ndofs_if:(j+1)*ndofs_if,:]) )
            if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] += torch.from_numpy(TTrl@(utmp[(j-1)*ndofs_if:j*ndofs_if,:]))
            if j < nSlab-2:      out[j*ndofs_if:(j+1)*ndofs_if,:] += torch.from_numpy(TTlr@(utmp[(j+1)*ndofs_if:(j+2)*ndofs_if,:]))
        if u.ndim == 1:
            out = out.flatten()
        return out
    
    A_balance_HBS = LinearOperator(shape=((nSlab-1)*ndofs_if, (nSlab-1)*ndofs_if),
                               matvec=apply_balance_HBS, dtype=float)
    
    gInfo = gmres_info()
    u = u_if#bc_helmholtz(XXif,kh)
    if gmres_iters > 0:
        tic = time.time()
        uhat,_   = gmres(A_balance_HBS,rhs,rtol=1e-8,callback=gInfo,maxiter=gmres_iters,restart=gmres_iters)
        solve_time_HBS = time.time()-tic
        niter = gInfo.niter
        gmres_err = np.linalg.norm(_np(uhat)-_np(u))/np.linalg.norm(_np(u))
        print("time = ",solve_time_HBS)
        print("niter = ",niter)
        print("u err = ",gmres_err)
    else:
        solve_time_HBS = float('nan')
        niter = float('nan')
        gmres_err = float('nan')
        print("GMRES solve skipped (gmres_iters = 0)")


    v = np.random.standard_normal(((nSlab-1)*ndofs_if,))
    tic = time.time()
    for i in range(20):
        v = A_balance_HBS@v
    tMV = (time.time()-tic)/20

    v = np.random.standard_normal(((nSlab-1)*ndofs_if,))
    tic = time.time()
    for i in range(3):
        v = A_balance@v
    tLUMV = (time.time()-tic)/3

    res_LU = np.linalg.norm(_np(A_balance@u)-_np(rhs))
    res_HBS = np.linalg.norm(_np(A_balance_HBS@u)-_np(rhs))
    v = np.random.standard_normal((A_balance.shape[0],))
    Av = _np(A_balance@v)
    errHBS = np.linalg.norm(A_balance_HBS@v-_np(Av))/np.linalg.norm(Av)


    
    if solver.MUMPS_mem<0:
        memLU = 2*abs(solver.MUMPS_mem)*(1e6)*8/1e9
    else:
        memLU = 2*solver.MUMPS_mem*8/1e9

    # ---- aggregate totals (edge-corrected) -----------------------------------
    # Each slab carries 4 maps; the 2 edge slabs are missing maps (a physical
    # boundary replaces one neighbour), so the naive *nSlab over-counts by 2 maps'
    # worth.  Subtract half the single-slab cost (= 2 of the 4 maps) from sampling
    # and construction.  (Previously the HBS line subtracted 2*tSample, too much.)
    total_LU_mem_GB   = nSlab*memLU
    total_LU_time_s   = nSlab*(solver.tMUMPS)
    total_HBS_mem_GB  = ((nSlab-2)*4+2)*(TTll.nbytes)/1e9
    sample_time_s     = tSample*nSlab - tSample/2
    total_HBS_time_s  = tHBS*nSlab - tHBS/2
    solve_time_LU_est = niter*tLUMV          # estimate: same iter count, LU matvec cost

    print("================ SUMMARY ====================")
    print("splitting                = ",splitting)
    print("N                        = ",Aii.shape[0])
    print("total LU mem             = ",total_LU_mem_GB,"GB")
    print("total LU time            = ",total_LU_time_s,"s")
    print("total HBS mem            = ",total_HBS_mem_GB,"GB")
    print("sample time              = ",sample_time_s,"s")
    print("total HBS time           = ",total_HBS_time_s,"s")
    print("HBS equilib. matvec time = ",tMV,'s')
    print("LU equilib. matvec time  = ",tLUMV,'s')
    print("err HBS                  = ",errHBS)
    print("res HBS                  = ",res_HBS)
    print("res LU                   = ",res_LU)
    print("=============================================")

    # ---- append results to resultsTform.csv (create with header if absent) ---
    _csv_path = "resultsTform.csv"
    _header = ["H", "gmres_err", "niter",
               "total_LU_time_s", "total_HBS_time_s", "sample_time_s",
               "HBS_matvec_time_s", "LU_matvec_time_s",
               "total_LU_mem_GB", "total_HBS_mem_GB",
               "solve_time_HBS_s", "solve_time_LU_est_s"]
    _row = [Lx, gmres_err, niter,
            total_LU_time_s, total_HBS_time_s, sample_time_s,
            tMV, tLUMV,
            total_LU_mem_GB, total_HBS_mem_GB,
            solve_time_HBS, solve_time_LU_est]
    _need_header = not os.path.exists(_csv_path)
    with open(_csv_path, "a") as _f:
        if _need_header:
            _f.write(",".join(_header) + "\n")
        _f.write(",".join(repr(x) for x in _row) + "\n")
    print(f"appended results row to {_csv_path}")

# ###########################################################################
# ###########################   STENCIL SOLVER   ############################
# ###########################################################################
# Non-overlapping stencil path.  The interface is a slab FACE (x = 0 / x = Lx),
# i.e. a boundary DOF, so the four-block conormal map from stencilSolver
# (Axx/Axi as the O(h^2) Dirichlet-to-Neumann / conormal derivative) is used
# directly -- no interior-plane scatter, no Jc_inJc framework.
#
# Interface unknowns = the strictly-interior x-face (ring excluded).  The ring
# (x-face met by a physical y/z boundary) is global Dirichlet data and folds
# into the known complement (JJ/Jlc/Jrc) exactly like the y/z faces, feeding the
# rhs only.  This keeps every interface DOF a node where the conormal is the
# clean O(h^2) map (full tangential stencil present), and keeps |Jl| == |Jr|.
elif solve_method == 'stencil':
    nx, ny, nz = order
    nx = int(ny * Lx) + 1            # single-width slab; nx so x=Lx lands on-grid
    print("stencil nx (derived from ny, Lx) = ", nx)
    ord_ = [nx, ny, nz]
    solver = stencil.stencilSolver(HH, slabGeom, ord_)

    # ---- four blocks (interior solve + conormal boundary rows) ----
    Aii = solver.Aii.tocsr()
    Aib = solver.Aix.tocsc()        # column slicing  Aib[:, J0]
    Abi = solver.Axi.tocsr()        # row    slicing  Abi[J1]
    Abb = solver.Axx.tocsr()
    tic = time.time()
    lu  = splu(Aii.tocsc())
    tLU = time.time() - tic
    print("LU (splu) factorization time = ", tLU)
    Ni  = Aii.shape[0]
    solver_ii = LinearOperator((Ni, Ni), dtype=np.float64,
                               matvec  = lu.solve,
                               matmat  = lu.solve,
                               rmatvec = lambda b: lu.solve(b, trans='T'),
                               rmatmat = lambda b: lu.solve(b, trans='T'))

    XYtot = solver.XX
    Ji = np.asarray(solver.Ji)
    Jb = np.asarray(solver.Jx)
    XXi = XYtot[Ji, :]
    XXb = XYtot[Jb, :]

    # ---- interface unknowns = strictly-interior x-faces (ring -> known) ----
    tol = 1e-9
    Jl = np.where((np.abs(XXb[:, 0] - 0.) < tol) &
                  (XXb[:, 1] > tol) & (XXb[:, 1] < Ly - tol) &
                  (XXb[:, 2] > tol) & (XXb[:, 2] < Lz - tol))[0]
    Jr = np.where((np.abs(XXb[:, 0] - Lx) < tol) &
                  (XXb[:, 1] > tol) & (XXb[:, 1] < Ly - tol) &
                  (XXb[:, 2] > tol) & (XXb[:, 2] < Lz - tol))[0]
    _sl, _sr = set(Jl.tolist()), set(Jr.tolist())
    JJ  = np.array([i for i in range(len(Jb)) if i not in _sl and i not in _sr])
    Jlc = np.array([i for i in range(len(Jb)) if i not in _sl])
    Jrc = np.array([i for i in range(len(Jb)) if i not in _sr])
    ndofs_if = len(Jl)
    assert len(Jl) == len(Jr), "left/right interface face sizes differ"
    XXl = XXb[Jl, :]
    XXr = XXb[Jr, :]
    print("|Jl| = ", len(Jl), "  |Jr| = ", len(Jr), "  |JJ| = ", len(JJ))

    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    # ---- local Schur (conormal) map  T_{J1,J0} = Abb[J1,J0] - Abi[J1] Aii^-1 Aib[:,J0]
    # Precompute the sliced sub-blocks once; with_local=False gives the smoothing
    # remainder  -Abi Aii^-1 Aib  used by the --splitting self blocks.
    def make_T(J1, J0, with_local=True):
        Bbi = Abi[J1].tocsr()
        Bib = Aib[:, J0].tocsc()
        Bbb = Abb[J1][:, J0].tocsr() if with_local else None
        def fwd(v):
            v2 = v[:, None] if v.ndim == 1 else v
            out = -np.asarray(Bbi @ (solver_ii @ np.asarray(Bib @ v2)))
            if Bbb is not None:
                out = out + np.asarray(Bbb @ v2)
            return out.ravel() if v.ndim == 1 else out
        def adj(v):
            v2 = v[:, None] if v.ndim == 1 else v
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
    # smoothing-remainder self maps (sampled by HBS when --splitting is on)
    LinOp_ll_rem = _linop((nif, nif), make_T(Jl, Jl, with_local=False))
    LinOp_rr_rem = _linop((nif, nif), make_T(Jr, Jr, with_local=False))

    # ---- rhs maps: interface-from-known-data ----
    Trb,   _ = make_T(Jr, JJ)       # interior slab: right face from y/z faces
    Tlb,   _ = make_T(Jl, JJ)       # interior slab: left  face from y/z faces
    Trb_l, _ = make_T(Jr, Jrc)      # end slab 0:    right face from everything-but-Jr
    Tlb_r, _ = make_T(Jl, Jlc)      # end slab N-1:  left  face from everything-but-Jl

    # ---- single-interface equilibrium check on the manufactured trace ----
    shiftL = np.array([Lx, 0., 0.])
    ur_l = bc_helmholtz(XXb[Jr], kh)
    ul_r = bc_helmholtz((XXb + shiftL)[Jl], kh)
    ub_l = bc_helmholtz(XXb[Jrc], kh)
    ub_r = bc_helmholtz((XXb + shiftL)[Jlc], kh)
    equilib1 = np.linalg.norm(
        LinOp_rr @ ur_l + LinOp_ll @ ul_r + Trb_l(ub_l) + Tlb_r(ub_r), ord=np.inf)
    print("equilib1 = ", equilib1, "  (O(h^2) for stencil, not ~1e-12)")

    nSlab = int(round(1. / Lx))
    print("nSlab = ", nSlab)

    # ---- block-tridiagonal balance operator (LU/exact maps) ----
    def apply_balance(u):
        utmp = u[:, None] if u.ndim == 1 else u
        out = np.zeros_like(utmp)
        for j in range(nSlab - 1):
            blk = j * nif
            out[blk:blk + nif] = (LinOp_ll @ utmp[blk:blk + nif]
                                  + LinOp_rr @ utmp[blk:blk + nif])
            if j > 0:
                out[blk:blk + nif] += LinOp_rl @ utmp[blk - nif:blk]
            if j < nSlab - 2:
                out[blk:blk + nif] += LinOp_lr @ utmp[blk + nif:blk + 2 * nif]
        return out.ravel() if u.ndim == 1 else out

    A_balance = LinearOperator(shape=((nSlab - 1) * nif, (nSlab - 1) * nif),
                               matvec=apply_balance, dtype=np.float64)

    # ---- rhs + manufactured interface trace ----
    rhs   = np.zeros((nSlab - 1) * nif)
    u_if  = np.zeros((nSlab - 1) * nif)
    XXif  = np.zeros(((nSlab - 1) * nif, 3))
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

    print("res = ", np.linalg.norm(rhs - A_balance @ u_if))

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

    # =====================  HBS-compressed balance  =======================
    print("===============  HBS version  ===============")
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    tree_leaf = 64
    if admissibility == 'weak':
        tree = slabTree.slabTree(XXl, False, tree_leaf)
        TTrr = HBStorch.HBSMAT(device=device, tree=tree)
        TTll = HBStorch.HBSMAT(device=device, tree=tree)
        TTlr = HBStorch.HBSMAT(device=device, tree=tree)
        TTrl = HBStorch.HBSMAT(device=device, tree=tree)
        kmax = 1
    else:
        tree = slabTree.slabTree(XXl, False, tree_leaf, adjacency=admissibility)
        TTrr = HBStorch_strong.HBSMAT(device=device, tree=tree)
        TTll = HBStorch_strong.HBSMAT(device=device, tree=tree)
        TTlr = HBStorch_strong.HBSMAT(device=device, tree=tree)
        TTrl = HBStorch_strong.HBSMAT(device=device, tree=tree)
        kmax = 9 if admissibility == 'strong' else 5

    nl = len(tree.get_box_inds(tree.get_leaves()[0]))
    s  = kmax * max(2 * rk, nl) + rk + 10
    N  = LinOp_rr.shape[0]

    if splitting:
        assert len(Jl) == len(Jr), \
            "splitting assumes |Jl| == |Jr| so D_ll and D_rr act on the same block"
        # exact local in-face blocks (NOT Schur complements): the singular/
        # differential part of the self maps, kept exact and added at apply time.
        D_rr = Abb[Jr][:, Jr].tocsr()
        D_ll = Abb[Jl][:, Jl].tocsr()
        src_rr, src_ll = LinOp_rr_rem, LinOp_ll_rem
    else:
        src_rr, src_ll = LinOp_rr, LinOp_ll

    # separate sampling per block (no simultaneous left/right sampling)
    tSample = 0.0
    tHBS = 0.0
    def _construct(TT, src):
        tic = time.time()
        Om  = np.random.standard_normal((N, s))
        Psi = np.random.standard_normal((N, s))
        Y = src @ Om
        Z = src.T @ Psi
        ts = time.time() - tic
        tic = time.time()
        TT.construct(rk, Om, Psi, Y, Z, fast=True)
        th = time.time() - tic
        return ts, th
    for TT, src in ((TTrr, src_rr), (TTll, src_ll), (TTrl, LinOp_rl), (TTlr, LinOp_lr)):
        ts, th = _construct(TT, src)
        tSample += ts; tHBS += th

    if splitting:
        def selfmat_rr(x): return np.asarray(D_rr @ x) + _np(TTrr @ x)
        def selfmat_ll(x): return np.asarray(D_ll @ x) + _np(TTll @ x)
    else:
        def selfmat_rr(x): return _np(TTrr @ x)
        def selfmat_ll(x): return _np(TTll @ x)

    v = np.random.standard_normal((N, 10))
    err_rr = np.linalg.norm(selfmat_rr(v) - _np(LinOp_rr @ v)) / np.linalg.norm(v)
    err_ll = np.linalg.norm(selfmat_ll(v) - _np(LinOp_ll @ v)) / np.linalg.norm(v)
    err_lr = np.linalg.norm(_np(TTlr @ v) - _np(LinOp_lr @ v)) / np.linalg.norm(v)
    err_rl = np.linalg.norm(_np(TTrl @ v) - _np(LinOp_rl @ v)) / np.linalg.norm(v)
    print("err_rr = ", err_rr)
    print("err_ll = ", err_ll)
    print("err_lr = ", err_lr)
    print("err_rl = ", err_rl)

    def apply_balance_HBS(u):
        utmp = u[:, None] if u.ndim == 1 else u
        out = np.zeros_like(utmp)
        for j in range(nSlab - 1):
            blk = j * nif
            out[blk:blk + nif] = (selfmat_ll(utmp[blk:blk + nif])
                                  + selfmat_rr(utmp[blk:blk + nif]))
            if j > 0:
                out[blk:blk + nif] += _np(TTrl @ utmp[blk - nif:blk])
            if j < nSlab - 2:
                out[blk:blk + nif] += _np(TTlr @ utmp[blk + nif:blk + 2 * nif])
        return out.ravel() if u.ndim == 1 else out

    A_balance_HBS = LinearOperator(shape=((nSlab - 1) * nif, (nSlab - 1) * nif),
                                   matvec=apply_balance_HBS, dtype=np.float64)

    gInfo = gmres_info()
    if gmres_iters > 0:
        tic = time.time()
        uhat, _ = gmres(A_balance_HBS, rhs, rtol=1e-8, callback=gInfo,
                        maxiter=gmres_iters, restart=gmres_iters)
        solve_time_HBS = time.time() - tic
        niter = gInfo.niter
        gmres_err = np.linalg.norm(uhat - u_if) / np.linalg.norm(u_if)
        print("time = ", solve_time_HBS)
        print("niter = ", niter)
        print("u err = ", gmres_err)
    else:
        solve_time_HBS = float('nan'); niter = float('nan'); gmres_err = float('nan')
        print("GMRES solve skipped (gmres_iters = 0)")

    # ---- matvec timings + memory ----
    v = np.random.standard_normal(((nSlab - 1) * nif,))
    tic = time.time()
    for _ in range(20):
        v = A_balance_HBS @ v
    tMV = (time.time() - tic) / 20
    v = np.random.standard_normal(((nSlab - 1) * nif,))
    tic = time.time()
    for _ in range(3):
        v = A_balance @ v
    tLUMV = (time.time() - tic) / 3

    memLU = (lu.L.nnz + lu.U.nnz) * 8 / 1e9
    res_LU  = np.linalg.norm(_np(A_balance @ u_if)     - _np(rhs))
    res_HBS = np.linalg.norm(_np(A_balance_HBS @ u_if) - _np(rhs))
    v  = np.random.standard_normal((A_balance.shape[0],))
    Av = _np(A_balance @ v)
    errHBS = np.linalg.norm(_np(A_balance_HBS @ v) - Av) / np.linalg.norm(Av)

    total_LU_mem_GB  = nSlab * memLU
    total_LU_time_s  = nSlab * tLU
    total_HBS_mem_GB = ((nSlab - 2) * 4 + 2) * (TTll.nbytes) / 1e9
    sample_time_s    = tSample * nSlab - tSample / 2
    total_HBS_time_s = tHBS * nSlab - tHBS / 2
    solve_time_LU_est = niter * tLUMV

    print("================ SUMMARY ====================")
    print("splitting                = ", splitting)
    print("N                        = ", Aii.shape[0])
    print("ndofs_if                 = ", nif)
    print("total LU mem             = ", total_LU_mem_GB, "GB")
    print("total LU time            = ", total_LU_time_s, "s")
    print("total HBS mem            = ", total_HBS_mem_GB, "GB")
    print("sample time              = ", sample_time_s, "s")
    print("total HBS time           = ", total_HBS_time_s, "s")
    print("HBS equilib. matvec time = ", tMV, "s")
    print("LU equilib. matvec time  = ", tLUMV, "s")
    print("err HBS                  = ", errHBS)
    print("res HBS                  = ", res_HBS)
    print("res LU                   = ", res_LU)
    print("=============================================")

    _csv_path = "resultsTform_stencil.csv"
    _header = ["H", "gmres_err", "niter",
               "total_LU_time_s", "total_HBS_time_s", "sample_time_s",
               "HBS_matvec_time_s", "LU_matvec_time_s",
               "total_LU_mem_GB", "total_HBS_mem_GB",
               "solve_time_HBS_s", "solve_time_LU_est_s"]
    _row = [Lx, gmres_err, niter,
            total_LU_time_s, total_HBS_time_s, sample_time_s,
            tMV, tLUMV,
            total_LU_mem_GB, total_HBS_mem_GB,
            solve_time_HBS, solve_time_LU_est]
    _need_header = not os.path.exists(_csv_path)
    with open(_csv_path, "a") as _f:
        if _need_header:
            _f.write(",".join(_header) + "\n")
        _f.write(",".join(repr(x) for x in _row) + "\n")
    print(f"appended results row to {_csv_path}")