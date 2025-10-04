import numpy as np
import jax.numpy as jnp
import torch
import scipy
from packaging.version import Version
import matplotlib.tri as tri

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
#import solver.spectralmultidomain.hps.pdo as pdo
# validation&testing
import time
from scipy.sparse.linalg import gmres
import solver.HPSInterp3D as interp
import matplotlib.pyplot as plt
import scipy.sparse.linalg as splinalg
import multislab.omsdirectsolve as omsdirect

import geometry.geom_3D.cube as cube
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



jax_avail   = False
torch_avail = True
hpsalt      = True
kh = 5.25
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
        return torch.ones_like(p[:,0])
    def c33(p):
        return torch.ones_like(p[:,0])
    def c(p):
        return -kh*kh*torch.ones_like(p[:,0])
    Helm=pdo.PDO_3d(c11=c11,c22=c22,c33=c33,c=c)

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


N = 8
dSlabs,connectivity,H = cube.dSlabs(N)
pvec = np.array([4],dtype = np.int64)
err=np.zeros(shape = (len(pvec),))
discr_time=np.zeros(shape = (len(pvec),))
compr_time=np.zeros(shape = (len(pvec),))

#solve_method = 'iterative'
solve_method = 'direct'

tridiag = (solve_method=='direct')
for indp in range(len(pvec)):
    p = pvec[indp]
    p_disc = p
    if hpsalt:
        formulation = "hpsalt"
        p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt
    a = np.array([H/8,1/16,1/16])
    #assembler = mA.rkHMatAssembler(p*p,50)
    assembler = mA.denseMatAssembler() #ref sol & conv test for no HBS
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)
    OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
    print("computing S blocks & rhs's...")
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=2)
    print("done")
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)
    niter = 0
    if solve_method == 'iterative':
        Stot,rhstot  = OMS.construct_Stot_and_rhstot(S_rk_list,rhs_list,Ntot,nc,dbg=2)
        gInfo = gmres_info()
        stol = 1e-10*H*H

        if Version(scipy.__version__)>=Version("1.14"):
            uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=500,restart=500)
        else:
            uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=500,restart=500)
        niter = gInfo.niter
    elif solve_method == 'direct':
        rhstot = np.zeros(shape = (Ntot,))
        for i in range(len(rhs_list)):
            rhstot[i*nc:(i+1)*nc] = rhs_list[i]
        T = omsdirect.build_block_tridiagonal_solver(S_rk_list)
        uhat  = omsdirect.block_tridiagonal_solve(OMS, T, rhstot)
    
    res = Stot@uhat-rhstot

    
    print("=============SUMMARY==============")
    print("H                        = ",'%10.3E'%H)
    print("ord                      = ",p)
    print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
    print("GMRES iters              = ", niter)
    print("==================================")

    nc = OMS.nc
    err_tot = 0
    for slabInd in range(len(dSlabs)):
        geom    = np.array(dSlabs[slabInd])
        slab_i  = oms.slab(geom,lambda p : cube.gb(p,jax_avail,torch_avail))
        solver  = oms.solverWrap.solverWrapper(opts)
        solver.construct(geom,Helm,False)
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
    
    print("===================LOCAL ERR===================")
    print("err_tot = ",err_tot)
    print("===============================================")
    err[indp] = err_tot
    compr_time[indp] = OMS.stats.compr_timing
    discr_time[indp] = OMS.stats.discr_timing


fileName = 'cube.csv'
errMat = np.zeros(shape=(len(pvec),4))
errMat[:,0] = pvec
errMat[:,1] = err
errMat[:,2] = compr_time
errMat[:,3] = discr_time
with open(fileName,'w') as f:
    f.write('p,err,compr,discr\n')
    np.savetxt(f,errMat,fmt='%.16e',delimiter=',')

plt.figure(0)
plt.semilogy(pvec,err)
plt.show()
