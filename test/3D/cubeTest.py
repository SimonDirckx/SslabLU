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
kh = 50.25
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
pvec = np.array([2,7,8,9,10],dtype = np.int64)
err=np.zeros(shape = (len(pvec),))
discr_time=np.zeros(shape = (len(pvec),))
compr_time=np.zeros(shape = (len(pvec),))
for indp in range(len(pvec)):
    p = pvec[indp]
    p_disc = p
    if hpsalt:
        formulation = "hpsalt"
        p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt
    a = np.array([H/6,1/32,1/32])
    assembler = mA.rkHMatAssembler(p*p,50)
    #assembler = mA.denseMatAssembler() #ref sol & conv test for no HBS
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)
    OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
    print("computing Stot & rhstot...")
    Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler,2)
    print("done")
    
    gInfo = gmres_info()
    stol = 1e-10*H*H

    if Version(scipy.__version__)>=Version("1.14"):
        uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=500,restart=500)
    else:
        uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=500,restart=500)
    stop_solve = time.time()
    #Sdense = Stot@np.identity(Stot.shape[0])
    #uhat = np.linalg.solve(Sdense,rhstot)
    res = Stot@uhat-rhstot


    print("=============SUMMARY==============")
    print("H                        = ",'%10.3E'%H)
    print("ord                      = ",p)
    print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
    print("GMRES iters              = ", gInfo.niter)
    print("==================================")

    nc = OMS.nc

    


    nx=50
    ny=200
    nz=200

    xpts = np.linspace(0,1,nx)
    ypts = np.linspace(0,1,ny)
    zpts = np.linspace(0,1,nz)

    YY = np.zeros(shape=(nx*ny*nz,3))
    YY[:,0] = np.kron(np.kron(xpts,np.ones_like(ypts)),np.ones_like(zpts))
    YY[:,1] = np.kron(np.kron(np.ones_like(xpts),ypts),np.ones_like(zpts))
    YY[:,2] = np.kron(np.kron(np.ones_like(xpts),np.ones_like(ypts)),zpts)

    gYY = np.zeros(shape=(YY.shape[0],))
    err_tot = 0
    for slabInd in range(len(dSlabs)):
        geom    = np.array(dSlabs[slabInd])
        I0 = np.where(  (YY[:,0]>=geom[0,0]) & (YY[:,0]<=geom[1,0]) & (YY[:,1]>=geom[0,1]) & (YY[:,1]<=geom[1,1]))[0]
        YY0 = YY[I0,:]
        slab_i  = oms.slab(geom,lambda p : cube.gb(p,jax_avail,torch_avail))
        solver  = oms.solverWrap.solverWrapper(opts)
        solver.construct(geom,Helm)
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
        #g=g[:,np.newaxis]
        #uu00 = solver.solver.solve_dir_full(torch.tensor(g))
        #uu = np.array(uu00,dtype = np.float64,copy=True)
        #uu=uu.flatten()
        #ghat = solver.interp(YY0,uu)
        err_loc = np.linalg.norm(ghat-g,ord=np.inf)/np.linalg.norm(g,ord=np.inf)
        err_tot = np.max([err_loc,err_tot])
        print("err ghat = ",err_loc)
        #gYY[I0] = ghat

    #triang = tri.Triangulation(YY[:,0],YY[:,1])
    #tri0 = triang.triangles

    #gref = bc(YY)
    #np.save('ref_sol_waveguide.npy',gYY)
    
    print("err_tot = ",err_tot)
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
