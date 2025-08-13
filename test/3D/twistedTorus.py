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
from hps.geom              import ParametrizedGeometry3D

# validation&testing
import time
from scipy.sparse.linalg import gmres
#import solver.HPSInterp3D as interp
import matplotlib.pyplot as plt

import geometry.geom_3D.twistedTorus as twisted

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


bnds = twisted.bnds

################################################################
#
#   SET-UP BVP:         Helmholtz on 3D Annulus
#   - wave number       (kh)
#   - bfield            (= kh*ones)
#   - pdo_mod           (pdo transformed to square)
#   - BC
#   - known exact sol.  (u_exact)
#
################################################################

nwaves = 15.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi
jax_avail = True
if jax_avail:
    def bfield(p,kh):
        return -kh*kh*jnp.ones_like(p[...,0])
else:
    def bfield(p,kh):
        return -kh*kh*np.ones(shape=(p.shape[0],))
param_geom=twisted.param_geom()
pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)

def bc(p):
    return np.ones_like(p[:,0])

def u_exact(p):
    z=twisted.z1(p)
    return np.sin(kh*z)

N = 16
dSlabs,connectivity,H = twisted.dSlabs(N)


p = 14
a = [H/8.,1/32,1/32]
assembler = mA.rkHMatAssembler(p*p,75)
opts = solverWrap.solverOptions('hps',[p,p,p],a)
OMS = oms.oms(dSlabs,pdo_mod,lambda p :twisted.gb(p,True),opts,connectivity)
Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler,2)

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

errInf = 0.
nc = OMS.nc

nx=500
ny=500

xpts = np.linspace(-4,4,nx)
ypts = np.linspace(-4,4,ny)

ZZ = np.zeros(shape=(nx*ny,3))
ZZ[:,0] = np.kron(xpts,np.ones_like(ypts))
ZZ[:,1] = np.kron(np.ones_like(xpts),ypts)

sliceYY = np.zeros(shape=ZZ.shape)
sliceYY[:,0] = twisted.y1(ZZ,False)
sliceYY[:,1] = twisted.y2(ZZ,False)
sliceYY[:,2] = twisted.y3(ZZ,False)


I = np.where( (sliceYY[:,0]>=twisted.bnds[0][0]) & (sliceYY[:,0]<=twisted.bnds[1][0]) & (sliceYY[:,1]>=twisted.bnds[0][1]) & (sliceYY[:,1]<=twisted.bnds[1][1]) & (sliceYY[:,2]>=twisted.bnds[0][2]) & (sliceYY[:,2]<=twisted.bnds[1][2]) )[0]


YY = sliceYY[I,:]
gYY = np.zeros(shape=(YY.shape[0],))
print("YY shape = ",YY.shape)
print("gYY shape = ",gYY.shape)


sliceZZ = np.zeros(shape=(len(I),3))
sliceZZ[:,0] = twisted.z1(YY,False)
sliceZZ[:,1] = twisted.z2(YY,False)
sliceZZ[:,2] = twisted.z3(YY,False)



for slabInd in range(len(connectivity)):
    geom    = np.array(dSlabs[slabInd])
    I0 = np.where(  (YY[:,0]>=geom[0,0]) & (YY[:,0]<=geom[1,0]) & (YY[:,1]>=geom[0,1]) & (YY[:,1]<=geom[1,1]) & (YY[:,2]>=geom[0,2]) & (YY[:,2]<=geom[1,2]) )[0]
    YY0 = YY[I0,:]
    slab_i  = oms.slab(geom,lambda p : twisted.gb(p,True))
    solver  = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom,pdo_mod)
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)

    startL = ((slabInd-1)%N)
    startR = ((slabInd+1)%N)
    ul = uhat[startL*nc:(startL+1)*nc]
    ur = uhat[startR*nc:(startR+1)*nc]
    g = np.zeros(shape=(XXb.shape[0],))
    g[Il]=ul
    g[Ir]=ur
    g[Igb] = bc(XXb[Igb,:])
    g=g[:,np.newaxis]
    uu = solver.solver.solve_dir_full(g)
    uu=uu.flatten()
    ghat = solver.interp(YY0,uu)
    gYY[I0] = ghat
    

triang = tri.Triangulation(sliceZZ[:,0],sliceZZ[:,1])
tri0 = triang.triangles

q1 = (sliceZZ[tri0[:,0],:]+sliceZZ[tri0[:,1],:])/2.
q2 = (sliceZZ[tri0[:,1],:]+sliceZZ[tri0[:,2],:])/2.
q3 = (sliceZZ[tri0[:,2],:]+sliceZZ[tri0[:,0],:])/2.


yy1 = np.zeros(shape = q1.shape)
yy2 = np.zeros(shape = q2.shape)
yy3 = np.zeros(shape = q3.shape)

yy1[:,0] = twisted.y1(q1,False)
yy1[:,1] = twisted.y2(q1,False)
yy1[:,2] = twisted.y3(q1,False)

yy2[:,0] = twisted.y1(q2,False)
yy2[:,1] = twisted.y2(q2,False)
yy2[:,2] = twisted.y3(q2,False)

yy3[:,0] = twisted.y1(q3,False)
yy3[:,1] = twisted.y2(q3,False)
yy3[:,2] = twisted.y3(q3,False)


b1 = (yy1[:,0]<twisted.bnds[0][0]) | (yy1[:,0]>twisted.bnds[1][0]) | (yy1[:,1]<twisted.bnds[0][1]) | (yy1[:,1]>twisted.bnds[1][1]) | (yy1[:,2]<twisted.bnds[0][2]) | (yy1[:,2]>twisted.bnds[1][2])
b2 = (yy2[:,0]<twisted.bnds[0][0]) | (yy2[:,0]>twisted.bnds[1][0]) | (yy2[:,1]<twisted.bnds[0][1]) | (yy2[:,1]>twisted.bnds[1][1]) | (yy2[:,2]<twisted.bnds[0][2]) | (yy2[:,2]>twisted.bnds[1][2])
b3 = (yy3[:,0]<twisted.bnds[0][0]) | (yy3[:,0]>twisted.bnds[1][0]) | (yy3[:,1]<twisted.bnds[0][1]) | (yy3[:,1]>twisted.bnds[1][1]) | (yy3[:,2]<twisted.bnds[0][2]) | (yy3[:,2]>twisted.bnds[1][2])


mask = (b1&b2)|(b1&b3)|(b2&b3)
triang.set_mask(mask)

plt.figure(5)
plt.tripcolor(triang, gYY, shading='flat')
plt.axis('equal')
plt.colorbar()
plt.savefig('twistedTorus.png',dpi=1000)
plt.show()