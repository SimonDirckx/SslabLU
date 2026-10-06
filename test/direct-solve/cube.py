import numpy as np
import direct_solve.omsdirectsolveHBS_torch as omsdirectHBS
from direct_solve.omsdirectsolveHBS_torch import rkStrat
import torch
import scipy
# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import time
import geometry.geom_3D.cube as cube
import matAssembly.HBS.HBStorch as HBStorch
import scipy.special as special
from packaging.version import Version
from scipy.sparse.linalg import gmres


def dense_to_linop(A):
    A = np.array(A)
    n = A.shape[0]
    lo = LinearOperator(
        shape=(n, n), dtype=A.dtype,
        matvec  = lambda v: A @ v,
        rmatvec = lambda v: A.T @ v,
        matmat  = lambda V: A @ V,
        rmatmat = lambda V: A.T @ V,
    )
    lo.solve = lambda v, mode='N': (
        np.linalg.solve(A, v) if mode == 'N' else np.linalg.solve(A.T, v)
    )
    lo.tree = lo.quad = None
    return lo

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

kh = 40.

def c11(p):
    return torch.ones_like(p[:,0])
def c22(p):
    return torch.ones_like(p[:,1])
def c33(p):
    return torch.ones_like(p[:,2])
Helm=pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33)

def bc(p):
    c0 = -1
    c1 = -1
    c2 = -1
    r = np.sqrt((p[:,0]-c0)**2+(p[:,1]-c1)**2+(p[:,2]-c2)**2)
    return np.cos(kh*r)/(4*np.pi*r)


N = 9
dSlabs,connectivity,H = cube.dSlabs(N)
p = 8
p_disc = p + 2 # To handle different conventions between hps and hpsalt
a = np.array([H/4,1/64,1/64])
rk0 = 30
assembler = mA.rkHMatAssembler(4*p,rk0,ndim=3)
opts = solverWrap.solverOptions('hpsalt',[p_disc,p_disc,p_disc],a,reduced_gpu=False)
tic_sys = time.time()
OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=False,torch_avail=True),opts,connectivity,stiff_mat_const=True,constructHBS=True,keepLU=True)
print("oms built")
Stot_lu,  rhstot_lu  = OMS.construct_Stot_and_rhstot(bc, assembler)               # LU operator + LU rhs
print("sys mats done in ",time.time()-tic_sys,"s")
Stot_HBS, rhstot_HBS = OMS.construct_Stot_and_rhstot(bc, assembler, rhsHBS=True)  # HBS operator + HBS rhs
print("sys mats done in ",time.time()-tic_sys,"s")
nc = OMS.nc
S_rk_list = OMS.hbs_blocks

step = 10
strat = rkStrat.linear(rk0+step,step,skip_first_level=False)
diagnostics = False
rb_solver = omsdirectHBS.RedBlackSolverHBS(nc,strat,tree = S_rk_list[0][0].tree,quad = False,compress_diag=True,fast=True,identity_diag=True,seed=None)
#rb_solver = omsdirectHBS.ThomasSolverHBS(nc,strat,diagnostics=diagnostics)

if diagnostics:
    S_dense_list = OMS.S_dense_list
    rb_solver.factorize(S_rk_list,S_exact=S_dense_list)
    print(rb_solver.report.table("comb",   rows="stages"))  # one level isolated
    print(rb_solver.report.table("ladder", rows="stages"))  # the solver's own path
    for r in rb_solver.report.comb():
        if r["block"] is not None and r["stage"] >= 3:
            norm = r["dense_abs"] / r["dense_rel"]
            print(f'{r["stage"]}  {r["block"]:22s}  norm {norm:.2e}  rel.err {r["dense_rel"]:.2e}')
            Sdense_lu = np.identity(nc*(N-1))
    uhat =rb_solver.solve(rhstot_lu)
    for i in range(len(S_dense_list)):
        if i>0:
            Sdense_lu[:,(i-1)*nc:i*nc][i*nc:(i+1)*nc,:] = S_dense_list[i][0]
        if i<len(S_dense_list)-1:
            Sdense_lu[:,(i+1)*nc:(i+2)*nc][i*nc:(i+1)*nc,:] = S_dense_list[i][-1]
    udense = np.linalg.solve(Sdense_lu,rhstot_lu)
    
else:
    tic = time.time()
    rb_solver.factorize(S_rk_list)
    print("factorization done in ",time.time()-tic)
    Ntot, nc = OMS.Ntot,OMS.nc
    def matvec_rb(v):
        return rb_solver.solve(v)
    Sinv_HBS_rb  = scipy.sparse.linalg.LinearOperator(shape=(Ntot,Ntot),matvec=matvec_rb,dtype=np.float64)
    gInfo = gmres_info()
    stol = 1e-7
    tic = time.time()
    if Version(scipy.__version__)>=Version("1.14"):
        uhat,info   = gmres(Stot_lu,rhstot_lu,rtol=stol,callback=gInfo,maxiter=20,restart=20,M=Sinv_HBS_rb)
    else:
        uhat,info   = gmres(Stot_lu,rhstot_lu,tol=stol,callback=gInfo,maxiter=20,restart=20,M=Sinv_HBS_rb)
    print("elapsed time gmres = ",time.time()-tic)
    print("pGMRES iters rb          = ", gInfo.niter)

for slabInd in range(len(dSlabs)):
    gc_HBS = uhat[slabInd*nc:(slabInd+1)*nc]
    geom    = np.array(dSlabs[slabInd])
    slab_i  = oms.slab(geom,lambda p : cube.gb(p,False,True))
    solver  = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom,Helm,False,False)
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
    XXc = XXi[Ic,:]
    gc = bc(XXc).detach().cpu().numpy()
    err = np.linalg.norm(gc_HBS-gc,ord=np.inf)
    print("===================LOCAL ERR===================")
    print("err = ",err)
    print("===============================================")