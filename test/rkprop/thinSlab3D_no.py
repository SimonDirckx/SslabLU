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
    tHBS = 0
    tSample = 0
    tic = time.time()
    Om = np.random.standard_normal((N,s))
    Psi = np.random.standard_normal((N,s))
    tSample+=time.time()-tic
    Y = LinOp_rr@Om
    Z = LinOp_rr.T@Psi
    
    tic=time.time()
    tic = time.time()
    TTrr.construct(rk,Om,Psi,Y,Z,fast=True)
    tHBS += time.time() - tic

    tic=time.time()
    Om = np.random.standard_normal((N,s))
    Psi = np.random.standard_normal((N,s))
    Y = LinOp_ll@Om
    Z = LinOp_ll.T@Psi
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
    
    
    

    v = np.random.standard_normal((N,10))
    err_rr = np.linalg.norm(TTrr@v-LinOp_rr@v)/np.linalg.norm(v)
    err_ll = np.linalg.norm(TTll@v-LinOp_ll@v)/np.linalg.norm(v)
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
            out[j*ndofs_if:(j+1)*ndofs_if,:] = torch.from_numpy( TTll@utmp[j*ndofs_if:(j+1)*ndofs_if,:]+TTrr@utmp[j*ndofs_if:(j+1)*ndofs_if,:])
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
        niter = gInfo.niter
        print("time = ",time.time()-tic)
        print("niter = ",niter)
        print("u err = ",np.linalg.norm(_np(uhat)-_np(u))/np.linalg.norm(_np(u)))
    else:
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

    res_LU = np.linalg.norm(A_balance@_np(u)-_np(rhs))
    res_HBS = np.linalg.norm(A_balance_HBS@_np(u)-_np(rhs))


    res_LU = np.linalg.norm(A_balance@u-rhs)
    res_HBS = np.linalg.norm(A_balance_HBS@u-rhs)
    if solver.MUMPS_mem<0:
        memLU = 2*abs(solver.MUMPS_mem)*(1e6)*8/1e9
    else:
        memLU = 2*solver.MUMPS_mem*8/1e9


    print("================ SUMMARY ====================")
    print("N                        = ",Aii.shape[0])
    print("total LU mem             = ",nSlab*memLU,"GB")
    print("total LU time            = ",nSlab*(solver.tMUMPS),"s")
    print("total HBS mem            = ",((nSlab-2)*4+2)*(TTll.nbytes)/1e9,"GB")
    print("sample time              = ",tSample*((nSlab-2)),"s")
    print("total HBS time           = ",tHBS*((nSlab-2)),"s")
    print("HBS equilib. matvec time = ",tMV,'s')
    print("LU equilib. matvec time  = ",tLUMV,'s')
    print("res HBS                  = ",res_HBS)
    print("res LU                   = ",res_LU)
    print("=============================================")
    
    
    
    