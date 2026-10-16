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

kh = 10*np.pi

def c11(p):
    return torch.ones_like(p[:,0])
def c22(p):
    return torch.ones_like(p[:,1])
def c33(p):
    return torch.ones_like(p[:,2])
def c(p):
    return -kh*kh*torch.ones_like(p[:,0])
Helm=pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33,c=c)

def bc(p):
    c0 = -1
    c1 = -1
    c2 = -1
    r = np.sqrt((p[:,0]-c0)**2+(p[:,1]-c1)**2+(p[:,2]-c2)**2)
    return np.cos(kh*r)/(4*np.pi*r)


N = 33
dSlabs,connectivity,H = cube.dSlabs(N)
p = 8
p_disc = p + 2 # To handle different conventions between hps and hpsalt
a = np.array([H/2,1/32,1/32])
opts = solverWrap.solverOptions('hpsalt',[p_disc,p_disc,p_disc],a,reduced_gpu=False)
rk0 = 200
assembler = mA.rkHMatAssembler(8*p*p,rk0,ndim=3)
opts = solverWrap.solverOptions('hpsalt',[p_disc,p_disc,p_disc],a,reduced_gpu=False)
tic_sys = time.time()
OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=False,torch_avail=True),opts,connectivity,stiff_mat_const=True,constructHBS=True,keepLU=True,keepDense=False)
print("oms built")
Stot_lu,  rhstot_lu  = OMS.construct_Stot_and_rhstot(bc, assembler,dbg = 0)               # LU operator + LU rhs
print("sys mats done in ",time.time()-tic_sys,"s")
Stot_HBS, rhstot_HBS = OMS.construct_Stot_and_rhstot(bc, assembler, rhsHBS=True,dbg = 0)  # HBS operator + HBS rhs
print("sys mats done in ",time.time()-tic_sys,"s")
S_rk_list = OMS.hbs_blocks
Ntot, nc = OMS.Ntot,OMS.nc
step = 30
strat = rkStrat.linear(rk0+step,step,skip_first_level=False)
rb_solver = omsdirectHBS.RedBlackSolverHBS(nc,strat,tree = S_rk_list[0][0].tree,quad = False,compress_diag=True,fast=True,identity_diag=True,compute_device='cpu',seed=None)
rb_solver.factorize(S_rk_list)


def set_solve_device(rb, dev):
    """Point the RB solver and every HBS block it uses at `dev`.
    Moves no data: the factors already live in the host masters."""
    dev = torch.device(dev)
    rb._drop_slot()
    rb.release_solve_cache()
    blocks = {id(b): b for b in rb._blocks}
    for SiM, T, T_hbs, SiP in rb.RB:                 # catch anything not registered
        for op in (*SiM, *T, *T_hbs, *SiP):
            blocks.setdefault(id(op), op)
    for b in blocks.values():
        if isinstance(b, HBStorch.HBSMAT):
            b.to(dev)                                # evicts any device copy
    rb.compute_device = dev

#set_solve_device(rb_solver, 'cpu')

def matvec_rb(v):
    return rb_solver.solve(v)
Sinv_HBS_rb  = scipy.sparse.linalg.LinearOperator(shape=(Ntot,Ntot),matvec=matvec_rb,dtype=np.float64)
gInfo = gmres_info()
stol = 1e-7
v = np.random.standard_normal(size=(Ntot,))

rb_solver.solve(v) #warmup


tic = time.time()
w = Sinv_HBS_rb@v
v = np.random.standard_normal(size=(Ntot,))
print("precond apply time = ",time.time()-tic)
tic = time.time()
w = Stot_lu@v
print("LU apply time = ",time.time()-tic)

tic=time.time()
if Version(scipy.__version__)>=Version("1.14"):
    uhat,info   = gmres(Stot_lu,rhstot_lu,rtol=stol,callback=gInfo,maxiter=50,restart=50,M=Sinv_HBS_rb)
else:
    uhat,info   = gmres(Stot_lu,rhstot_lu,tol=stol,callback=gInfo,maxiter=50,restart=50,M=Sinv_HBS_rb)
print("elapsed time gmres = ",time.time()-tic)
print("pGMRES iters rb          = ", gInfo.niter)
v = np.random.default_rng(0).standard_normal(Ntot)
y1 = Sinv_HBS_rb@v; y2 = Sinv_HBS_rb@v
print(np.linalg.norm(y1 - y2) / np.linalg.norm(y1))
y3 = Stot_HBS@v
print("precond vs HBS",np.linalg.norm(v-Sinv_HBS_rb@y3)/np.linalg.norm(v))
y4 = Stot_lu@v
y5 = Stot_HBS@v
print("precond vs LU",np.linalg.norm(y4-y5)/np.linalg.norm(y4))
print("precond vs LU (backward)",np.linalg.norm(v-Sinv_HBS_rb@(Stot_lu@v))/np.linalg.norm(v))
for slabInd in range(len(dSlabs)):
    gc_HBS = uhat[slabInd*nc:(slabInd+1)*nc]
    geom    = np.array(dSlabs[slabInd])
    slab_i  = oms.slab(geom,lambda p : cube.gb(p,False,True))
    solver  = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom,Helm,False,True)
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
    XXc = XXi[Ic,:]
    gc = bc(XXc).detach().cpu().numpy()
    g_from_solver = -solver.solver_ii@(solver.Aib@(bc(XXb).detach().cpu().numpy()))
    err = np.linalg.norm(gc_HBS-gc,ord=np.inf)
    err_ii = np.linalg.norm(gc-g_from_solver[Ic],ord=np.inf)
    err_HBS_vs_ii = np.linalg.norm(gc_HBS-g_from_solver[Ic],ord=np.inf)
    print("===================LOCAL ERR===================")
    print("err = ",err)
    print("err_ii = ",err_ii)
    print("err_HBS_vs_ii = ",err_HBS_vs_ii)
    print("===============================================")
