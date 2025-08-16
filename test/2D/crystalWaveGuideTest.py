# basic packages
import numpy as np
import jax.numpy as jnp
import scipy
from packaging.version import Version
import matplotlib.tri as tri

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import hps.pdo as pdo

# validation&testing
import time
from scipy.sparse.linalg import gmres
#import solver.HPSInterp3D as interp
import matplotlib.pyplot as plt

import geometry.geom_2D.square as square
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


#nwaves = 24.623521102434587
nwaves = 24.673521102434584
kh = (nwaves+0.03)*2*np.pi+1.8

jax_avail=True
if jax_avail:
    def bfield(xx):
        
        mag   = 0.930655
        width = 2500; 
        
        b = jnp.zeros_like(xx[...,0])
        
        dist = 0.04
        x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
        y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
        
        # box of points [x0,x1] x [y0,y1]
        for x in np.arange(x0,x1,dist):
            for y in np.arange(y0,y1,dist):
                xx_sq_c = (xx[...,0] - x)**2 + (xx[...,1] - y)**2
                b += mag * jnp.exp(-width * xx_sq_c)

        # box of points [x0,x1] x [y0,y2]
        for x in np.arange(x2,x3,dist):
            for y in np.arange(y0,y2-0.5*dist,dist):
                xx_sq_c = (xx[...,0] - x)**2 + (xx[...,1] - y)**2
                b += mag * jnp.exp(-width * xx_sq_c)
                
        # box of points [x0,x3] x [y2,y3]
        for x in np.arange(x0,x3,dist):
            for y in np.arange(y2,y3,dist):
                xx_sq_c = (xx[...,0] - x)**2 + (xx[...,1] - y)**2
                b += mag * jnp.exp(-width * xx_sq_c)    
        
        kh_fun = -kh**2 * (1 - b)
        return kh_fun


    def c11(p):
        return jnp.ones_like(p[...,0])
    def c22(p):
        return jnp.ones_like(p[...,0])
    def c(p):
        return bfield(p)
    Lapl=pdo.PDO2d(c11,c22,None,None,None,c)

    
else:
    def bfield(xx):
        
        mag   = 0.930655
        width = 2500; 
        
        b = np.zeros(shape = (xx.shape[0],))
        
        dist = 0.04
        x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
        y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
        
        # box of points [x0,x1] x [y0,y1]
        for x in np.arange(x0,x1,dist):
            for y in np.arange(y0,y1,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * np.exp(-width * xx_sq_c)

        # box of points [x0,x1] x [y0,y2]
        for x in np.arange(x2,x3,dist):
            for y in np.arange(y0,y2-0.5*dist,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * np.exp(-width * xx_sq_c)
                
        # box of points [x0,x3] x [y2,y3]
        for x in np.arange(x0,x3,dist):
            for y in np.arange(y2,y3,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * np.exp(-width * xx_sq_c)    
        
        kh_fun = -kh**2 * (1 - b)
        return kh_fun


    def c11(p):
        return np.ones_like(p[:,0])
    def c22(p):
        return np.ones_like(p[:,0])
    def c(p):
        return bfield(p)
    Lapl=pdo.PDO2d(c11,c22,None,None,None,c)


def bc(p):
    return np.ones_like(p[:,0])

N = 8
dSlabs,connectivity,H = square.dSlabs(N)
print(connectivity)
pvec = np.array([8,10,12,14,16,18,20,22,24,26,28,30],dtype = np.int64)
err=np.zeros(shape = (len(pvec),))
discr_time=np.zeros(shape = (len(pvec),))
compr_time=np.zeros(shape = (len(pvec),))
for indp in range(len(pvec)):
    p = pvec[indp]
    a = [H/8,1/64]
    assembler = mA.rkHMatAssembler(p,25)
    #assembler = mA.denseMatAssembler()
    opts = solverWrap.solverOptions('hps',[p,p],a)
    OMS = oms.oms(dSlabs,Lapl,lambda p:square.gb(p,True),opts,connectivity)
    print("computing Stot & rhstot...")
    Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler,2)
    print("done")
    gInfo = gmres_info()
    stol = 1e-10*H*H

    if Version(scipy.__version__)>=Version("1.14"):
        uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=400,restart=400)
    else:
        uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=400,restart=400)

    stop_solve = time.time()
    res = Stot@uhat-rhstot


    print("=============SUMMARY==============")
    print("H                        = ",'%10.3E'%H)
    print("ord                      = ",p)
    print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
    print("GMRES iters              = ", gInfo.niter)
    print("==================================")

    nc = OMS.nc

    print("nc = ",nc)
    print("N = ",Stot.shape)


    nx=500
    ny=500

    xpts = np.linspace(0,1,nx)
    ypts = np.linspace(0,1,ny)

    YY = np.zeros(shape=(nx*ny,2))
    YY[:,0] = np.kron(xpts,np.ones_like(ypts))
    YY[:,1] = np.kron(np.ones_like(xpts),ypts)

    gYY = np.zeros(shape=(YY.shape[0],))

    for slabInd in range(len(dSlabs)):
        geom    = np.array(dSlabs[slabInd])
        I0 = np.where(  (YY[:,0]>=geom[0,0]) & (YY[:,0]<=geom[1,0]) & (YY[:,1]>=geom[0,1]) & (YY[:,1]<=geom[1,1]))[0]
        print("len I0 = ",len(I0))
        YY0 = YY[I0,:]
        slab_i  = oms.slab(geom,lambda p : square.gb(p,True))
        solver  = oms.solverWrap.solverWrapper(opts)
        solver.construct(geom,Lapl)
        Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
        startL = slabInd-1
        startR = slabInd+1
        g = np.zeros(shape=(XXb.shape[0],))
        g[Igb] = bc(XXb[Igb,:])
        if startL>-1:
            g[Il] = uhat[startL*nc:(startL+1)*nc]
        if startR<len(dSlabs):
            g[Ir] = uhat[startR*nc:(startR+1)*nc]
        g=g[:,np.newaxis]
        uu = solver.solver.solve_dir_full(g)
        uu=uu.flatten()
        ghat = solver.interp(YY0,uu)
        print("norm ghat = ",np.linalg.norm(ghat,ord=np.inf))
        gYY[I0] = ghat

    triang = tri.Triangulation(YY[:,0],YY[:,1])
    tri0 = triang.triangles

    gref = np.load('ref_sol_waveguide.npy')
    #np.save('ref_sol_waveguide.npy',gYY)
    print("err ref = ",np.linalg.norm(gref-gYY)/np.linalg.norm(gref))
    err[indp] = np.linalg.norm(gref-gYY)/np.linalg.norm(gref)
    compr_time[indp] = OMS.stats.compr_timing
    discr_time[indp] = OMS.stats.discr_timing


fileName = 'crystal_waveguide.csv'
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
