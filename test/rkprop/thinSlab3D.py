import numpy as np
import matplotlib.pyplot as plt

import SOMS3D_csr
import torch
import matAssembly.HBS.slabTree as slabTree
import matAssembly.HBS.HBStorch_strong as HBStorch
from scipy.sparse.linalg import LinearOperator
import solver.stencil.stencilSolver as stencil
import solver.stencil.geom as geom
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
import mumps
import scipy.sparse as sparse
import time
import os

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


kh = 5.

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
solve_method = 'stencil'
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
print("LU factors done in : ",time.time()-tic,"s")
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
    tree = slabTree.slabTree(XXc,False,8*8) 
    def smatmat(v,transpose=False):                
                if (v.ndim == 1):
                    v_tmp = v[:,np.newaxis]
                else:
                    v_tmp = v

                if (not transpose):
                    result = (ctx.solve(Sib[:,Jr]@v_tmp))[I,:]
                else:
                    result      = np.zeros(shape=(len(Ii),v.shape[1]))
                    result[I,:] = v_tmp
                    result      = Sib[:,J].T @ (ctxT.solve(result))
                if (v.ndim == 1):
                    result = result.flatten()
                return result

    LinOp = LinearOperator(shape=(len(Jc),len(Jr)),\
        matvec = lambda v:smatmat(v,Jc,Jr), rmatvec = lambda v:smatmat(v,Jc,Jr,transpose=True),\
        matmat = lambda v:smatmat(v,Jc,Jr), rmatmat = lambda v:smatmat(v,Jc,Jr,transpose=True))

elif solve_method=='stencil':    
    tree = slabTree.slabTree(XXr,False,4*4,adjacency='full')
    Sib_Jr_T = Sib[:,Jr].T.tocsr()
    Sib_Jr = Sib[:,Jr].tocsr()
    def smatmat(v, transpose=False):
        if v.ndim == 1:
            v_tmp = v[:, np.newaxis]
        else:
            v_tmp = v

        if not transpose:
            # Forward:  L v = E_{Jc_inJc -> Jc_large} · P_{Jc ⊂ Ii} · A^{-1} · Sib[:,Jr] · v
            rhs = (Sib_Jr @ sparse.csc_matrix(v_tmp)).tocsc()
            result_tmp = (ctx._solve_sparse(rhs))[Jc, :]
            del rhs
            ctx.mumps_instance.icntl[20]=0
            result = np.zeros((len(Jc_large), v_tmp.shape[1]))
            result[Jc_inJc, :] = result_tmp
            del result_tmp

        else:
            # Transpose:  L^T w = Sib[:,Jr]^T · A^{-T} · P_{Jc}^T · E^T · w
            k = v_tmp.shape[1]
            w_Jc = v_tmp[Jc_inJc, :]            # (|Jc|, k)   -- Jc_inJc indexes Jc_large
            m = len(Jc)
            rhs = sparse.csc_matrix(
                (
                    w_Jc.ravel(order="F"),          # data, column-major
                    np.tile(Jc, k),                 # row indices
                    np.arange(0, m*k + 1, m)        # column pointers
                ),
                shape=(len(Ii), k),
            )
            sol = ctxT._solve_sparse(rhs)               # (|Ii|, k)
            del rhs                             # free 1.6 GB before the matmul
            ctxT.mumps_instance.icntl[20]=0
            result = Sib_Jr_T@sol
            # costa
            del sol
        if v.ndim == 1:
            result = result.flatten()
        return result

    LinOp = LinearOperator(shape=(len(Jc_large),len(Jr)),\
        matvec = lambda v:smatmat(v), rmatvec = lambda v:smatmat(v,transpose=True),\
        matmat = lambda v:smatmat(v), rmatmat = lambda v:smatmat(v,transpose=True))
else:
    raise ValueError("solve method not recognized")
print("=========     PERM TEST:     =========")
print(np.allclose(np.sort(XXr[:,1]), np.sort(XYtot[Jc_large][:,1])))   # same y-set?
# row-by-row alignment in (y,z), in the orders actually used:
print(np.allclose(XXr[:,1:3], XYtot[Jc_large][:,1:3]))
assert tree.perm_leaf.shape[0] == XXr.shape[0]
assert np.unique(tree.perm_leaf).size == XXr.shape[0]
sizes = [len(tree.get_box_inds(l)) for l in tree.get_leaves()]
print(set(sizes))   # want a single value
print("=========  LINOP CONSTRUCTED  =========")
SSr = HBStorch.HBSStrong(LinOp,device=device,tree=tree)
print("============  COMPRESS HBS  ============")
tic = time.time()
SSr.construct(20,fast = True)
print("HBS done in : ",time.time()-tic,"s")
print("============    HBS DONE    ============")

ul  = -(ctx.solve(Sib[:, Jl ]@rhsS[Jl])[Jc])
ub  = -(ctx.solve(Sib[:, Jb ]@rhsS[Jb ])[Jc])
ur = -(SSr@rhsS[Jr])[Jc_inJc]
uhat_S = ul+ur+ub

err2 = np.linalg.norm(uhat_S - uS, ord=2) / np.linalg.norm(uS, ord=2)
print(f"err2 = {err2:.6e}")
uihat = ctx.solve(-Sib@rhsS)
ui = bc_helmholtz(XXi,kh)
err2 = np.linalg.norm(uihat - ui, ord=2) / np.linalg.norm(ui, ord=2)
print(f"err2 = {err2:.6e}")