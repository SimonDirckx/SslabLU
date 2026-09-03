import numpy as np
import direct_solve.omsdirectsolveHBS as omsdirect

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
import direct_solve.omsdirectsolveHBS as omsdirectHBS
import direct_solve.omsdirectsolve as omsdirect
import geometry.geom_3D.cube as cube
from scipy.sparse.linalg import LinearOperator




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
kh = 50.
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


N = 9
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
    a = np.array([H/4,1/32,1/32])
    assembler = mA.rkHMatAssembler(p*p,256,ndim=3)
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a,reduced_gpu=False)
    OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity,stiff_mat_const=True)
    print("computing S blocks & rhs's...")
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=0)
    print("done")
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=0)
    niter = 0
    print("type SrkList  = ",type(S_rk_list))
    print("len SrkList  = ",len(S_rk_list))
    print("type rhstot  = ",type(rhstot))
    print("type rhslist  = ",type(rhs_list))
    print("len rhs_list  = ",len(rhs_list))
    print("Ntot = ",Ntot)
    tic = time.time()
    thomas_solver = omsdirectHBS.ThomasSolverHBS(nc,256)
    thomas_solver.factorize(S_rk_list)
    print("THOMAS solver factorized in ",time.time()-tic,"s")
    tic = time.time()
    def matvec_thomas(v):
        return thomas_solver.solve(v)
    Sinv_HBS_thomas  = scipy.sparse.linalg.LinearOperator(shape=(Ntot,Ntot),matvec=matvec_thomas,dtype=np.float64)
    tic = time.time()
    v = np.random.standard_normal(size=(Sinv_HBS_thomas.shape[0],))
    u = Sinv_HBS_thomas@v
    print("Thomas solver time = ",time.time()-tic)

    tic = time.time()
    rb_solver = omsdirectHBS.RedBlackSolverHBS(nc,256,S_rk_list[0][0].tree,S_rk_list[0][0].quad,fast=True,device='cpu')
    rb_solver.factorize(S_rk_list)
    print("RB solver factorized in ",time.time()-tic,"s")    
    def matvec_rb(v):
        return rb_solver.solve(v)
    Sinv_HBS_rb  = scipy.sparse.linalg.LinearOperator(shape=(Ntot,Ntot),matvec=matvec_rb,dtype=np.float64)
    tic = time.time()
    v = np.random.standard_normal(size=(Sinv_HBS_thomas.shape[0],))
    u = Sinv_HBS_rb@v
    print("RB solver time = ",time.time()-tic)

    ptgInfo = gmres_info()
    prbgInfo = gmres_info()
    
    stol = 1e-11*H*H
    if Version(scipy.__version__)>=Version("1.14"):
        uhat_thomas,_   = gmres(Stot,rhstot,rtol=stol,callback=ptgInfo,maxiter=100,restart=100,M=Sinv_HBS_thomas)
    else:
        uhat_thomas,_   = gmres(Stot,rhstot,tol=stol,callback=ptgInfo,maxiter=100,restart=100,M=Sinv_HBS_thomas)
    if Version(scipy.__version__)>=Version("1.14"):
        uhat_rb,_   = gmres(Stot,rhstot,rtol=stol,callback=prbgInfo,maxiter=100,restart=100,M=Sinv_HBS_rb)
    else:
        uhat_rb,_   = gmres(Stot,rhstot,tol=stol,callback=prbgInfo,maxiter=100,restart=100,M=Sinv_HBS_rb)
    res_thomas = Stot@uhat_thomas-rhstot
    res_rb = Stot@uhat_rb-rhstot
    print("=============SUMMARY==============")
    print("H                        = ",'%10.3E'%H)
    print("ord                      = ",p)
    print("npan_dim                 = ",(int)(H/a[0]),',',(int)(.5/a[1]))
    print("nc                       = ",OMS.nc)
    print("L2 rel. res thomas       = ", np.linalg.norm(res_thomas)/np.linalg.norm(rhstot))
    print("L2 rel. res rb           = ", np.linalg.norm(res_rb)/np.linalg.norm(rhstot))
    print("pGMRES iters thomas      = ", ptgInfo.niter)
    print("pGMRES iters rb          = ", prbgInfo.niter)
    print("==================================")
    nc = OMS.nc
    err_tot = 0
    uhat = uhat_rb
    for slabInd in range(len(dSlabs)):
        geom    = np.array(dSlabs[slabInd])
        slab_i  = oms.slab(geom,lambda p : cube.gb(p,jax_avail,torch_avail))
        solver  = oms.solverWrap.solverWrapper(opts)
        solver.construct(geom,Helm,False,False)
        Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
        startL = slabInd-1
        startR = slabInd+1
        g = np.zeros(shape=(XXb.shape[0],))
        g[Igb] = bc(XXb[Igb,:])
        if startL>-1:
            g[Il] = uhat[startL*nc:(startL+1)*nc]
        if startR<len(dSlabs):
            g[Ir] = uhat[startR*nc:(startR+1)*nc]
        ghat = bc(XXb)
        err_loc = np.linalg.norm(ghat-g)/np.linalg.norm(g)
        err_tot = np.max([err_loc,err_tot])
        print("===================LOCAL ERR===================")
        print("err ghat = ",err_loc)
        print("===============================================")
    
    print("===================GLOBAL ERR===================")
    print("err_tot = ",err_tot)
    print("===============================================")
    err[indp] = err_tot
    sample_time[indp] = OMS.stats.sampl_timing
    compr_time[indp] = OMS.stats.compr_timing
    discr_time[indp] = OMS.stats.discr_timing


fileName = 'cube.csv'
errMat = np.zeros(shape=(len(pvec),5))
errMat[:,0] = pvec
errMat[:,1] = err
errMat[:,2] = sample_time
errMat[:,3] = compr_time
errMat[:,4] = discr_time
with open(fileName,'w') as f:
    f.write('p,err,sample,compr,discr\n')
    np.savetxt(f,errMat,fmt='%.16e',delimiter=',')

plt.figure(0)
plt.semilogy(pvec,err)
plt.show()
