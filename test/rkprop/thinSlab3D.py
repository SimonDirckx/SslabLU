import numpy as np
import matplotlib.pyplot as plt

import SOMS3D_csr
import torch
import matAssembly.HBS.slabTree as slabTree
import matAssembly.HBS.HBStorch as HBStorch
from scipy.sparse.linalg import LinearOperator
import solver.stencil.stencilSolver as stencil
import solver.stencil.geom as geom
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
import mumps
import scipy.sparse as sparse
import time
import os
from scipy.sparse.linalg import gmres

def rss_gb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6   # kB -> GB
    return -1.0

def setup_mumps(Sii):
    ctx = mumps.Context()
    ctx.analyze(Sii)             # symbolic factorization (uses sparsity pattern only)
    ctx.factor(Sii)              # numeric factorization
    return ctx
def setup_mumps_transpose(Sii):
    ctx = mumps.Context()
    ctx.analyze(Sii.T)
    ctx.factor(Sii.T)
    return ctx


def bc_laplace(p):
    """Free-space Green's function with source at (-0.5, -0.5, -0.5)."""
    r = np.sqrt((p[:,0]+.5)**2+(p[:,1]+.5)**2+(p[:,2]+.5)**2)
    return 1./(4*np.pi*r)
def bc_helmholtz(p,kh):
    """Free-space Green's function with source at (-0.5, -0.5, -0.5)."""
    r = np.sqrt((p[:,0]+.5)**2+(p[:,1]+.5)**2+(p[:,2]+.5)**2)
    return np.real(np.exp(1j*kh*r)/(4*np.pi*r))

Lx = 1./8.
Ly = 1.
Lz = 1.
cx = Lx/2
slabGeom = geom.BoxGeometry(np.array([[0,0,0],[Lx,Ly,Lz]]))


kh = 5.2

def  c11(p):
    return np.ones_like(p[:,0])
def  c22(p):
    return np.ones_like(p[:,0])
def  c33(p):
    return np.ones_like(p[:,0])
def  c(p):
    return kh*kh*np.ones_like(p[:,0])
HH = pdo.PDO_3d(c11=c11,c22=c22,c33=c33,c=c)


coeffs = {'c11': 1., 'c22': 1., 'c33': 1.,'c':kh**2}
print("============BUILDING SOLVER============")
solve_method = 'SOMS'
if solve_method == 'SOMS':
    nbx = 4
    nby = 16
    nbz = 16
    px = 6
    py = 6
    pz = 6
    Sii, Sib, ftild, XYtot, Ii, Ib, wi,wb = SOMS3D_csr.SOMS_solver_sparse(
         px, py, pz, nbx, nby, nbz, Lx, Ly, Lz,
         coeffs, True, None, weighted=False)
    
elif solve_method=='stencil':
    n = 128
    nx = int(Lx*n) + 1
    ny = n
    nz = n
    ord = [nx,ny,nz]
    solver = stencil.stencilSolver(HH,slabGeom,ord)
    Sii = solver.Aii
    Sib = solver.Aix
    XYtot = solver.XX
    Ii = solver.Ji
    Ib = solver.Jx
    wi = np.ones((Sii.shape[0],))
    wb = np.ones((Sib.shape[1],))
else:
    raise ValueError("solver type not recognized")

XXi = XYtot[Ii,:]
XXb = XYtot[Ib,:]
Jc = np.where(XXi[:,0]==cx)[0]
XXc = XXi[Jc,:]
Jl = np.where(XXb[:,0]==0)[0]
XXl = XXb[Jl,:]
Jr = np.where(XXb[:,0]==Lx)[0]

print("|Jl| = ",len(Jl))
print("|Jr| = ",len(Jr))
print("|Jc| = ",len(Jc))


XXr = XXb[Jr,:]
Jb = np.array([i for i in range(XXb.shape[0]) if i not in Jl and i not in Jr],dtype=np.int64)

uc = bc_helmholtz(XXc,kh)
ul = bc_helmholtz(XXl,kh)
ur = bc_helmholtz(XXr,kh)
ub = bc_helmholtz(XXb[Jb,:],kh)

ctx = setup_mumps(Sii)
ctxT = setup_mumps_transpose(Sii)


BLK = 32                                   # tune; see note below
ctx.mumps_instance.icntl[27]  = BLK        # one wide BLAS-3 block per chunk
ctxT.mumps_instance.icntl[27] = BLK

def smatmat(v,J , transpose=False):
    v_tmp = v[:, None] if v.ndim == 1 else v
    k = v_tmp.shape[1]
    
    
    if not transpose:
        Sib_J = Sib[:,J].tocsc()
        out = np.zeros((len(Jc), k))
        for s in range(0, k, BLK):
            c = slice(s, min(s + BLK, k))
            rhs = (Sib_J @ sparse.csc_matrix(v_tmp[:, c])).tocsc()
            sol = ctx._solve_sparse(rhs)              # dense (len(Ii) x BLK) — bounded
            out[:, c] = -sol[Jc, :]
            del rhs, sol
            ctx.mumps_instance.icntl[20]=0
    else:
        Sib_J_T = Sib[:,J].T.tocsr()
        out = np.zeros((len(Jr), k))

        for s in range(0, k, BLK):
            c = slice(s, min(s + BLK, k))
            w  = v_tmp[:, c]; bw = w.shape[1]; m = len(Jc)
            rhs = sparse.csc_matrix(
                (w.ravel(order="F"), np.tile(Jc, bw), np.arange(0, m*bw+1, m)),
                shape=(len(Ii), bw))
            sol = ctxT._solve_sparse(rhs)
            out[:, c] = -Sib_J_T @ sol
            del rhs, sol
            ctx.mumps_instance.icntl[20]=0
    return out.flatten() if v.ndim == 1 else out

LinOp_r = LinearOperator(shape=(len(Jc),len(Jr)),\
    matvec = lambda v:smatmat(v,Jr), rmatvec = lambda v:smatmat(v,Jr,transpose=True),\
    matmat = lambda v:smatmat(v,Jr), rmatmat = lambda v:smatmat(v,Jr,transpose=True))
LinOp_l = LinearOperator(shape=(len(Jc),len(Jl)),\
    matvec = lambda v:smatmat(v,Jl), rmatvec = lambda v:smatmat(v,Jl,transpose=True),\
    matmat = lambda v:smatmat(v,Jl), rmatmat = lambda v:smatmat(v,Jl,transpose=True))

err = uc-LinOp_l@ul-LinOp_r@ur-(ctx.solve(-(Sib[:,Jb]@ub)))[Jc]
print("err = ",np.linalg.norm(err))

ndofs_if = len(Jc)
ndslab = int(1/cx)-1
XXif = np.zeros((ndslab*ndofs_if,3))
rhs = np.zeros((ndslab*ndofs_if,))
for i in range(ndslab):
    XXc = XXi[Jc] + np.array([i*cx, 0., 0.])
    XXif[i*ndofs_if:(i+1)*ndofs_if, :] = XXc

    XXb_loc = XXb + np.array([i*cx, 0., 0.])
    XXl = XXb[Jl] + np.array([i*cx, 0., 0.])
    
    XXr = XXb[Jr] + np.array([i*cx, 0., 0.])
    uj = bc_helmholtz(XXb_loc, kh)
    if i == 0:
        XXr = XXb[Jr]
        XXc = XXi[Jc]
        
        br = bc_helmholtz(XXr,kh)
        uc = bc_helmholtz(XXc,kh)
        ur = LinOp_r@br
        Jb0 = np.array([k for k in range(XXb.shape[0]) if k not in Jr])
        blk = -ctx.solve(Sib[:, Jb0] @ uj[Jb0])[Jc]
        print("err slab ",i," = ",np.linalg.norm(uc-blk-ur))
    elif i == ndslab - 1:             # physical far face is known -> keep Jr, drop Jl
        Jb0 = np.array([k for k in range(XXb.shape[0]) if k not in Jl])
        blk = -ctx.solve(Sib[:, Jb0] @ uj[Jb0])[Jc]
        XXl = XXb[Jl] + np.array([i*cx, 0., 0.])
        XXc = XXi[Jc] + np.array([i*cx, 0., 0.])
        bl = bc_helmholtz(XXl,kh)
        uc = bc_helmholtz(XXc,kh)
        ul = LinOp_l@bl
        print("err slab ",i," = ",np.linalg.norm(uc-blk-ul))
    else:                             # interior slab: only the (y,z) faces are known
        blk = -ctx.solve(Sib[:, Jb] @ uj[Jb])[Jc]
        bl = bc_helmholtz(XXl,kh)
        br = bc_helmholtz(XXr,kh)
        uc = bc_helmholtz(XXc,kh)
        ul = LinOp_l@bl
        ur = LinOp_r@br
        print("err slab ",i," = ",np.linalg.norm(uc-blk-ul-ur))
    rhs[i*ndofs_if:(i+1)*ndofs_if] = blk

ui = bc_helmholtz(XXif,kh)

def apply_balance(u):
    if u.ndim == 1:
        utmp = u[:,None]
    else:
        utmp = u
    out = utmp.copy()
    for j in range(ndslab):
        if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] -= LinOp_l(utmp[(j-1)*ndofs_if:j*ndofs_if,:])
        if j < ndslab-1:      out[j*ndofs_if:(j+1)*ndofs_if,:] -= LinOp_r(utmp[(j+1)*ndofs_if:(j+2)*ndofs_if,:])
    if u.ndim == 1:
        out = out.flatten()
    return out

A_balance = LinearOperator(shape=(ndslab*ndofs_if, ndslab*ndofs_if),
                           matvec=apply_balance, dtype=float)

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
N = A_balance.shape[0]
v = np.random.standard_normal((N,))
tic =time.time()
bb = A_balance@v
print("matvec time = ",time.time()-tic)
gInfo = gmres_info()
u = bc_helmholtz(XXif,kh)
res = A_balance@u-rhs
print("res = ",np.linalg.norm(res))
tic = time.time()
uhat,_   = gmres(A_balance,rhs,rtol=1e-8,callback=gInfo,maxiter=75,restart=75)
niter = gInfo.niter
print("time = ",time.time()-tic)
print("niter = ",niter)
print("u err = ",np.linalg.norm(uhat-u)/np.linalg.norm(u))

print("===============  HBS version  ===============")

if torch.cuda.is_available():
    device = 'cuda'          # or 'cuda:0' to pin a specific GPU
else:
    device = 'cpu'

tree = slabTree.slabTree(XXc,False,py*pz,adjacency='full')

SSr = HBStorch.HBSMAT(device=device,tree=tree)
SSl = HBStorch.HBSMAT(device=device,tree=tree)

print("Sl shape = ",LinOp_l.shape)
print("Sr shape = ",LinOp_r.shape)
nl = len(tree.get_box_inds(tree.get_leaves()[0]))
rk = 150
s = 2*max(2*rk,nl)+rk+10
N = LinOp_r.shape[0]
Om = np.random.standard_normal((N,s))
Psi = np.random.standard_normal((N,s))
Nb = N//nl
Y = LinOp_r@Om
Z = LinOp_r.T@Psi
tic = time.time()
SSr.construct(rk,Om,Psi,Y,Z,fast=True)
print("HBS done in :",time.time()-tic)
Om = np.random.standard_normal((N,s))
Psi = np.random.standard_normal((N,s))
Nb = N//nl
Y = LinOp_l@Om
Z = LinOp_l.T@Psi
tic = time.time()
SSl.construct(rk,Om,Psi,Y,Z,fast=True)
print("HBS done in :",time.time()-tic)

def apply_balance_HBS(u):
    if u.ndim == 1:
        utmp = u[:,None]
    else:
        utmp = u
    out = utmp.copy()
    for j in range(ndslab):
        if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] -= SSl@(utmp[(j-1)*ndofs_if:j*ndofs_if,:])
        if j < ndslab-1:      out[j*ndofs_if:(j+1)*ndofs_if,:] -= SSr@(utmp[(j+1)*ndofs_if:(j+2)*ndofs_if,:])
    if u.ndim == 1:
        out = out.flatten()
    return out

A_balance_HBS = LinearOperator(shape=(ndslab*ndofs_if, ndslab*ndofs_if),
                           matvec=apply_balance_HBS, dtype=float)
N = A_balance.shape[0]
v = np.random.standard_normal((N,))
tic =time.time()
bb = A_balance_HBS@v
print("matvec time = ",time.time()-tic)
gInfo = gmres_info()
u = bc_helmholtz(XXif,kh)
res = A_balance_HBS@u-rhs
print("res = ",np.linalg.norm(res))
tic = time.time()
uhat,_   = gmres(A_balance_HBS,rhs,rtol=1e-8,callback=gInfo,maxiter=75,restart=75)
niter = gInfo.niter
print("time = ",time.time()-tic)
print("niter = ",niter)
print("u err = ",np.linalg.norm(uhat-u)/np.linalg.norm(u))


'''
print("============  SOLVER DONE  ============")
diff = Sii - Sii.T
print("max |Sii - Sii.T| =", abs(diff).max() if diff.nnz else 0.0)
print("=======================================")
print("============ FACTOR SOLVER ============")
tic = time.time()
ctx = SOMS3D_csr.setup_mumps(Sii)
if solve_method == 'SOMS':
    ctxT = SOMS3D_csr.setup_mumps_transpose(Sii)
else:
    ctxT = ctx
print("mem for Multislab with LU = ",ctx.mumps_instance.infog[17]/1e3/Lx,"GB")
print("LU time : ",(time.time()-tic)/Lx,"s")
print("============    LU DONE    ============")
print("MEM (GB) ctx = ",ctx.data.nbytes/1e9)
print("MEM (GB) ctx = ",ctxT.data.nbytes/1e9)
print("=======================================")

XXi = XYtot[Ii, :]
XXb = XYtot[Ib, :]

tol = 1e-10

# Interior evaluation plane
Jc  = np.where(np.abs(XXi[:,0] - cx) < tol)[0]   # x = Lx/2

Jc_large  = np.where(np.abs(XYtot[:,0] - cx) < tol)[0]   # x = Lx/2
Jc_inJc =  np.where((XYtot[Jc_large,1] > tol) &\
                    (XYtot[Jc_large,1] < Ly-tol) &\
                    (XYtot[Jc_large,2] > tol) &\
                    (XYtot[Jc_large,2] < Lz-tol))[0]   # x = Lx/2
print("|Jc| = ",len(Jc))
print("|Jc_large| = ",len(Jc_large))
print("|Jc_inJc| = ",len(Jc_inJc))
XXc = XXi[Jc,:]
wi = np.ones((len(Jc_inJc),))

# Boundary index sets (6 faces)
Jl  = np.where((np.abs(XXb[:,0] - 0. ) < tol))[0]
Jr  = np.where((np.abs(XXb[:,0] - Lx ) < tol))[0]
Jb = np.array([i for i in range(XXb.shape[0]) if i not in Jl and i not in Jr],dtype=np.int64)
XXr = XXb[Jr,:]
# Boundary and reference interior values
if kh == 0.:
    rhsS = bc_laplace(XXb)
    rhsS   = wb * rhsS
    uS   = bc_laplace(XXi[Jc, :])
else:
    rhsS = bc_helmholtz(XXb,kh)
    rhsS   = wb * rhsS
    uS   = bc_helmholtz(XXi[Jc, :],kh)

# Solution operator columns: SS* maps boundary data on face * -> centerline
# Sii @ u_i = -Sib @ u_b  =>  u_i = -Sii^{-1} Sib u_b
# Extract rows corresponding to Jc for each boundary face.
if torch.cuda.is_available():
    device = 'cuda'          # or 'cuda:0' to pin a specific GPU
else:
    device = 'cpu'

if solve_method=='SOMS':
    tree = slabTree.slabTree(XXc,False,py*pz,adjacency='full')
    BLK = 32                                   # tune; see note below
    ctx.mumps_instance.icntl[27]  = BLK        # one wide BLAS-3 block per chunk
    ctxT.mumps_instance.icntl[27] = BLK

    def smatmat(v,J , transpose=False):
        v_tmp = v[:, None] if v.ndim == 1 else v
        k = v_tmp.shape[1]
        
        
        if not transpose:
            Sib_J = Sib[:,J].tocsc()
            out = np.zeros((len(Jc), k))
            for s in range(0, k, BLK):
                c = slice(s, min(s + BLK, k))
                rhs = (Sib_J @ sparse.csc_matrix(v_tmp[:, c])).tocsc()
                sol = ctx._solve_sparse(rhs)              # dense (len(Ii) x BLK) — bounded
                out[:, c] = sol[Jc, :]
                del rhs, sol
                ctx.mumps_instance.icntl[20]=0
        else:
            Sib_J_T = Sib[:,J].T.tocsr()
            out = np.zeros((len(Jr), k))

            for s in range(0, k, BLK):
                c = slice(s, min(s + BLK, k))
                w  = v_tmp[:, c]; bw = w.shape[1]; m = len(Jc)
                rhs = sparse.csc_matrix(
                    (w.ravel(order="F"), np.tile(Jc, bw), np.arange(0, m*bw+1, m)),
                    shape=(len(Ii), bw))
                sol = ctxT._solve_sparse(rhs)
                out[:, c] = Sib_J_T @ sol
                del rhs, sol
                ctx.mumps_instance.icntl[20]=0
        return out.flatten() if v.ndim == 1 else out

    LinOp_r = LinearOperator(shape=(len(Jc),len(Jr)),\
        matvec = lambda v:smatmat(v,Jr), rmatvec = lambda v:smatmat(v,Jr,transpose=True),\
        matmat = lambda v:smatmat(v,Jr), rmatmat = lambda v:smatmat(v,Jr,transpose=True))
    LinOp_l = LinearOperator(shape=(len(Jc),len(Jr)),\
        matvec = lambda v:smatmat(v,Jl), rmatvec = lambda v:smatmat(v,Jl,transpose=True),\
        matmat = lambda v:smatmat(v,Jl), rmatmat = lambda v:smatmat(v,Jl,transpose=True))

elif solve_method=='stencil':    
    tree = slabTree.slabTree(XXr,False,8*8,adjacency='full')
    Sib_Jr_T = Sib[:,Jr].T.tocsr()
    Sib_Jr = Sib[:,Jr].tocsc()
    BLK = 32                                   # tune; see note below
    ctx.mumps_instance.icntl[27]  = BLK        # one wide BLAS-3 block per chunk
    ctxT.mumps_instance.icntl[27] = BLK

    def smatmat(v,J , transpose=False):
        v_tmp = v[:, None] if v.ndim == 1 else v
        k = v_tmp.shape[1]
        
        
        if not transpose:
            Sib_J = Sib[:,J].tocsc()
            out = np.zeros((len(Jc), k))
            for s in range(0, k, BLK):
                c = slice(s, min(s + BLK, k))
                rhs = (Sib_J @ sparse.csc_matrix(v_tmp[:, c])).tocsc()
                sol = ctx._solve_sparse(rhs)              # dense (len(Ii) x BLK) — bounded
                out[Jc_inJc, c] = sol[Jc, :]
                del rhs, sol
                ctx.mumps_instance.icntl[20]=0
        else:
            Sib_J_T = Sib[:,J].T.tocsr()
            out = np.zeros((len(Jr), k))

            for s in range(0, k, BLK):
                c = slice(s, min(s + BLK, k))
                w  = v_tmp[Jc_inJc, c]; bw = w.shape[1]; m = len(Jc)
                rhs = sparse.csc_matrix(
                    (w.ravel(order="F"), np.tile(Jc, bw), np.arange(0, m*bw+1, m)),
                    shape=(len(Ii), bw))
                sol = ctxT._solve_sparse(rhs)
                out[:, c] = Sib_J_T @ sol
                del rhs, sol
                ctx.mumps_instance.icntl[20]=0
        return out.flatten() if v.ndim == 1 else out

    LinOp_r = LinearOperator(shape=(len(Jc),len(Jr)),\
        matvec = lambda v:smatmat(v,Jr), rmatvec = lambda v:smatmat(v,Jr,transpose=True),\
        matmat = lambda v:smatmat(v,Jr), rmatmat = lambda v:smatmat(v,Jr,transpose=True))
    LinOp_l = LinearOperator(shape=(len(Jc),len(Jl)),\
        matvec = lambda v:smatmat(v,Jl), rmatvec = lambda v:smatmat(v,Jl,transpose=True),\
        matmat = lambda v:smatmat(v,Jl), rmatmat = lambda v:smatmat(v,Jl,transpose=True))
else:
    raise ValueError("solve method not recognized")
print("=========     PERM TEST:     =========")
print(np.allclose(np.sort(XXr[:,1]), np.sort(XYtot[Jc][:,1])))   # same y-set?
# row-by-row alignment in (y,z), in the orders actually used:
print(np.allclose(XXr[:,1:3], XYtot[Jc][:,1:3]))
assert tree.perm_leaf.shape[0] == XXr.shape[0]
assert np.unique(tree.perm_leaf).size == XXr.shape[0]
sizes = [len(tree.get_box_inds(l)) for l in tree.get_leaves()]
print(set(sizes))   # want a single value
print("=========  LINOP CONSTRUCTED  =========")
SSr = HBStorch.HBSMAT(device=device,tree=tree)
SSl = HBStorch.HBSMAT(device=device,tree=tree)

ub  = -(ctx.solve(Sib[:, Jb ]@rhsS[Jb ])[Jc])
uihat = ctx.solve(-Sib@rhsS)
print("============  COMPRESS HBS  ============")
print("Sl shape = ",LinOp_l.shape)
print("Sr shape = ",LinOp_r.shape)
nl = len(tree.get_box_inds(tree.get_leaves()[0]))
rk = 20
s = 9*max(2*rk,nl)+rk+10
N = LinOp_r.shape[0]
Om = np.random.standard_normal((N,s))
Psi = np.random.standard_normal((N,s))
Nb = N//nl
Y = LinOp_r@Om
Z = LinOp_r.T@Psi
SSr.construct(rk,Om,Psi,Y,Z,fast=True)

Om = np.random.standard_normal((N,s))
Psi = np.random.standard_normal((N,s))
Nb = N//nl
Y = LinOp_l@Om
Z = LinOp_l.T@Psi
SSl.construct(rk,Om,Psi,Y,Z,fast=True)


print("HBS time : ",(time.time()-tic)/Lx,"s")
print("============    HBS DONE    ============")
print("============  MATVEC TIMES  ============")

v = np.random.standard_normal((N,))
tic = time.time()
u = LinOp_r@v
print("LU time = ",time.time()-tic)
tic = time.time()
u = SSr@v
print("HBS time = ",time.time()-tic)

ur = -(SSr@rhsS[Jr])
ul = -(SSl@rhsS[Jl])
uhat_S = ul+ur+ub

err2 = np.linalg.norm(uhat_S - uS, ord=2) / np.linalg.norm(uS, ord=2)
print(f"err2 = {err2:.6e}")

ui = bc_helmholtz(XXi,kh)
err2 = np.linalg.norm(uihat - ui, ord=2) / np.linalg.norm(ui, ord=2)
print(f"err2 = {err2:.6e}")


from scipy.sparse.linalg import gmres

def blk_SSl(u):  return SSl @ u   # sub-diagonal block on an XXr-ordered vector
def blk_SSr(u):  return SSr @ u           # super-diagonal block







ndslab = int(1./Lx)-1
print("number of double slabs : ",ndslab)
ndofs_if = SSl.shape[0]
def apply_balance(u):
    if u.ndim == 1:
        utmp = u[:,None]
    else:
        utmp = u
    out = utmp.copy()
    for j in range(ndslab):
        if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] -= blk_SSl(utmp[(j-1)*ndofs_if:j*ndofs_if,:])
        if j < ndslab:      out[j*ndofs_if:(j+1)*ndofs_if,:] -= blk_SSr(utmp[j*ndofs_if:(j+1)*ndofs_if,:])
    if u.ndim == 1:
        out = out.flatten()
    return out

A_balance = LinearOperator(shape=(ndslab*ndofs_if, ndslab*ndofs_if),
                           matvec=apply_balance, dtype=float)

# ---------------------------------------------------------------------
# Right-hand side  b_j  =  contribution of all the KNOWN data on slab j:
#     * the four (y,z) faces (Jb)            -> every slab
#     * the physical x = 0 face (Jl)         -> slab 0      only
#     * the physical x = Lx_total face (Jr)  -> slab M-1    only
#     * (y,z)-boundary nodes pinned to the Dirichlet trace
# The known data is the manufactured solution (free-space Green's function),
# evaluated at the GLOBAL coordinates of each slab.
# ---------------------------------------------------------------------

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


#XXc = XXi[Jc,:]
#XXb = bdry
#Jl  = np.where((np.abs(XXb[:,0] - 0. ) < tol))[0]
#Jr  = np.where((np.abs(XXb[:,0] - Lx ) < tol))[0]
#Jb = np.array([i for i in range(XXb.shape[0]) if i not in Jl and i not in Jr],dtype=np.int64)
#XXr = XXb[Jr,:]


rhs  = np.zeros((A_balance.shape[0],))
XXif = np.zeros((A_balance.shape[0], 3))
for i in range(ndslab):
    # global interface (centerline) DOFs for slab i: the leftmost interface,
    # XYtot[Jc_large] (x = cx, already in the Jc_large / interface ordering),
    # translated by i*Lx in x.  Same shift used for the boundary data below.
    XXif[i*ndofs_if:(i+1)*ndofs_if, :] = XXi[Jc] + np.array([i*Lx, 0., 0.])

    XXb_loc = XXb + i*Lx
    uj = bc_helmholtz(XXb_loc, kh)

    # The centerline solve returns len(Jc) interior values, but each interface
    # block has len(Jc_large) = ndofs_if entries.  Scatter the solve result into
    # the Jc_inJc positions (the (y,z)-boundary entries of the block stay 0,
    # matching how SSl/SSr place their output via Jc_inJc).
    if i == 0:                        # physical x=0 face is known -> keep Jl, drop Jr
        XXr = XXi[Jc] + np.array([Lx, 0., 0.])
        XXc = XXi[Jc]
        
        br = bc_helmholtz(XXr,kh)
        uc = bc_helmholtz(XXc,kh)
        ur = LinOp_r@br
        Jb0 = np.array([k for k in range(XXb.shape[0]) if k not in Jr])
        blk = -ctx.solve(Sib[:, Jb0] @ uj[Jb0])[Jc]
        #print("err slab ",i," = ",np.linalg.norm(uc+blk-ur))
    elif i == ndslab - 1:             # physical far face is known -> keep Jr, drop Jl
        Jb0 = np.array([k for k in range(XXb.shape[0]) if k not in Jl])
        blk = -ctx.solve(Sib[:, Jb0] @ uj[Jb0])[Jc]
        XXl = XXi[Jc] + np.array([(i-1)*Lx, 0., 0.])
        XXc = XXi[Jc] + np.array([i*Lx, 0., 0.])
        bl = bc_helmholtz(XXl,kh)
        uc = bc_helmholtz(XXc,kh)
        ul = LinOp_l@bl
        #print("err slab ",i," = ",np.linalg.norm(uc+blk-ul))
    else:                             # interior slab: only the (y,z) faces are known
        blk = -ctx.solve(Sib[:, Jb] @ uj[Jb])[Jc]
        XXl = XXi[Jc] + np.array([(i-1)*Lx, 0., 0.])
        XXc = XXi[Jc] + np.array([i*Lx, 0., 0.])
        XXr = XXi[Jc] + np.array([(i+1)*Lx, 0., 0.])
        bl = bc_helmholtz(XXl,kh)
        br = bc_helmholtz(XXr,kh)
        uc = bc_helmholtz(XXc,kh)
        ul = LinOp_l@bl
        ur = LinOp_r@br
        print("err slab ",i," = ",np.linalg.norm(-uc-blk-ul-ur))



    rhs[i*ndofs_if:(i+1)*ndofs_if] = blk

gInfo = gmres_info()
u = bc_helmholtz(XXif,kh)
uhat,_   = gmres(A_balance,rhs,rtol=1e-8,callback=gInfo,maxiter=500,restart=500)
niter = gInfo.niter
print("niter = ",niter)
print("u err = ",np.linalg.norm(uhat-u)/np.linalg.norm(u))
#plt.figure(1)
#plt.plot(uhat-u)
#plt.show()
'''