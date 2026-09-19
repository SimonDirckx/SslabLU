import numpy as np
import direct_solve.omsdirectsolveHBS as omsdirect
from direct_solve.omsdirectsolveHBS_torch import rkStrat
import jax.numpy as jnp
import torch
import scipy
from packaging.version import Version
import matplotlib.tri as tri

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import solver.spectralmultidomain.hps.pdo as pdo
# validation&testing
import time
from scipy.sparse.linalg import gmres
import solver.HPSInterp3D as interp
import matplotlib.pyplot as plt
import scipy.sparse.linalg as splinalg
import multislab.omsdirectsolve as omsdirect
#import multislab.omsdirectsolveHBS as omsdirectHBS
import direct_solve.omsdirectsolveHBS_torch as omsdirectHBS
import direct_solve.omsdirectsolve as omsdirect
import geometry.geom_3D.cube as cube
from scipy.sparse.linalg import LinearOperator
import matAssembly.HBS.HBStorch as HBStorch


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


#### TOGGLE FOR HPSMULTIDOMAIN (SEE KUMP ET AL.)
jax_avail   = False
torch_avail = not jax_avail
hpsalt      = torch_avail
kh = 100.
if jax_avail:
    def c11(p):
        return jnp.ones_like(p[...,0])
    def c22(p):
        return jnp.ones_like(p[...,0])
    def c33(p):
        return jnp.ones_like(p[...,0])
    def c(p):
        return -kh*kh*jnp.ones_like(p[...,0])
    Helm=pdo.PDO3d(c11=c11,c22=c22,c33=c33,c=c)


elif torch_avail:
    def c11(p):
        return torch.ones_like(p[:,0])
    def c22(p):
        return torch.ones_like(p[:,1])
    def c33(p):
        return torch.ones_like(p[:,2])
    def c(p):
        return -kh*kh*torch.ones_like(p[:,0])
    Helm=pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33,c=c)

else:
    def c11(p):
        return np.ones_like(p[:,0])
    def c22(p):
        return np.ones_like(p[:,0])
    def c33(p):
        return np.ones_like(p[:,0])
    def c(p):
        return -kh*kh*np.ones_like(p[:,0])
    Helm=pdo.PDO3d(c11=c11,c22=c22,c33=c33,c=c)
def bc(p):
    source_loc = np.array([-.5,-.2,1])
    rr = np.linalg.norm(p-source_loc.T,axis=1)
    return np.real(np.exp(1j*kh*rr)/(4*np.pi*rr))
    #return np.sin(kh*(p[:,0]+p[:,1]+p[:,2])/np.sqrt(3))


N = 33
dSlabs,connectivity,H = cube.dSlabs(N)
pvec = np.array([8],dtype = np.int64)
err=np.zeros(shape = (len(pvec),))
discr_time=np.zeros(shape = (len(pvec),))
sample_time = np.zeros(shape=(len(pvec),))
compr_time=np.zeros(shape = (len(pvec),))

solve_method = 'direct'
formulation = "hps"
tridiag = (solve_method=='direct')
for indp in range(len(pvec)):
    p = pvec[indp]
    p_disc = p
    if hpsalt:
        formulation = "hpsalt"
        p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt
    a = np.array([H/2,1/64,1/64])
    assembler = mA.rkHMatAssembler(512,300,ndim=3)
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a,reduced_gpu=True)
    OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity,stiff_mat_const=True)
    OMS_LU = oms.oms_lu(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity,stiff_mat_const=True)
    print("computing S blocks & rhs's...")
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=0)
    S_list_lu, rhs_list_lu, Ntot_lu, nc_lu = OMS_LU.construct_Stot_helper(bc, assembler)
    print("done")
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=0)
    Stot_lu,rhstot_lu  = OMS_LU.construct_Stot_and_rhstot_linearOperator(S_list_lu,rhs_list_lu,Ntot,nc,dbg=0)
    niter = 0
    print("type SrkList  = ",type(S_rk_list))
    print("len SrkList  = ",len(S_rk_list))
    print("type rhstot  = ",type(rhstot))
    print("type rhslist  = ",type(rhs_list))
    print("len rhs_list  = ",len(rhs_list))
    print("Ntot = ",Ntot)
    strat = rkStrat.constant(300)
    tic = time.time()
    rb_solver = omsdirectHBS.RedBlackSolverHBS(nc,strat,S_rk_list[0][0].tree,S_rk_list[0][0].quad,fast=True,device='cuda',debug_blocks=16,oversample=100)
    print("rb rk      =", rb_solver.rk)
    print("rb s       =", rb_solver._nsamples(rb_solver.rk))
    print("tree mls   =", rb_solver.tree._min_leaf_size)
    rb_solver.factorize(S_rk_list)
    rb_solver.print_timing()
    rb_solver.print_block_errors()
    print("RB solver factorized in ",time.time()-tic,"s")
    r = rb_solver.residency_report()
    print(f"{(r['GB_H2D']+r['GB_D2H'])/r['GB_total']:.1f}x")
    v = np.random.standard_normal(Ntot)
    rsv = rb_solver.solve(v)
    tic = time.time()
    vprime = Stot_lu@rsv
    print("Stot_lu apply time = ",time.time()-tic)
    print("delta =", np.linalg.norm(vprime - v)/np.linalg.norm(v))
    def matvec_rb(v):
        return rb_solver.solve(v)
    Sinv_HBS_rb  = scipy.sparse.linalg.LinearOperator(shape=(Ntot,Ntot),matvec=matvec_rb,dtype=np.float64)
    tic = time.time()
    v = np.random.standard_normal(Ntot)
    b = Stot_lu @ v
    u, info = gmres(Stot_lu, b, rtol=1e-6, M=Sinv_HBS_rb, maxiter=20, restart=20)
    print("kappa lower bound:", np.linalg.norm(u-v)/np.linalg.norm(v) / 1e-6)
    print("artificial solution error: ",np.linalg.norm(u-v)/np.linalg.norm(v))
    print("RB solver time = ",time.time()-tic)
    v = np.random.standard_normal(Ntot)
    tic = time.time()
    b = Sinv_HBS_rb@v
    print("precond apply time = ",time.time()-tic)
    tot = rb_solver.footprint()
    gInfo = gmres_info()
    stol = 1e-8
    tic = time.time()
    if Version(scipy.__version__)>=Version("1.14"):
        uhat,info   = gmres(Stot_lu,rhstot_lu,rtol=stol,callback=gInfo,maxiter=20,restart=20,M=Sinv_HBS_rb)
    else:
        uhat,info   = gmres(Stot_lu,rhstot_lu,tol=stol,callback=gInfo,maxiter=20,restart=20,M=Sinv_HBS_rb)
    print("elapsed time gmres = ",time.time()-tic)
    print("pGMRES iters rb          = ", gInfo.niter)
    nc = OMS.nc
    for slabInd in range(len(dSlabs)):
        geom    = np.array(dSlabs[slabInd])
        slab_i  = oms.slab(geom,lambda p : cube.gb(p,jax_avail,torch_avail))
        solver  = oms.solverWrap.solverWrapper(opts)
        solver.construct(geom,Helm,False,False)
        Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
        XXc = XXi[Ic,:]
        gc = bc(XXc)
        gc_hat = uhat[slabInd*nc:(slabInd+1)*nc]
        err_loc = np.linalg.norm(gc_hat-gc)/np.linalg.norm(gc)
        print("===================LOCAL ERR===================")
        print("err ghat = ",err_loc)
        print("===============================================")
