# basic packages
import numpy as np
import jax.numpy as jnp
import scipy
from packaging.version import Version


# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
from hps.geom              import ParametrizedGeometry3D

# validation&testing
import time
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import geometry.geom_3D.squareTorus as squareTorus

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


bnds = squareTorus.bnds

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

nwaves = 10.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi
jax_avail = True
if jax_avail:
    def bfield(p,kh):
        return -kh*kh*jnp.ones_like(p[...,0])
else:
    def bfield(p,kh):
        return -kh*kh*np.ones(shape=(p.shape[0],))
param_geom=squareTorus.param_geom()
pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)

def bc(p):
    z=squareTorus.z1(p)
    return np.sin(kh*z)

def u_exact(p):
    z=squareTorus.z1(p)
    return np.sin(kh*z)

N = 16
dSlabs,connectivity,H = squareTorus.dSlabs(N)


p = 8
a = [H/8.,1/16,1/16]
assembler = mA.rkHMatAssembler(p*p,50)
opts = solverWrap.solverOptions('hps',[p,p,p],a)
OMS = oms.oms(dSlabs,pdo_mod,lambda p :squareTorus.gb(p,True),opts,connectivity)
Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler,2)

gInfo = gmres_info()
stol = 1e-8*H*H

if Version(scipy.__version__)>=Version("1.14"):
    uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=300,restart=300)
else:
    uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=300,restart=300)

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
for slabInd in range(len(connectivity)):
    geom    = np.array(dSlabs[slabInd])
    slab_i  = oms.slab(geom,lambda p : squareTorus.gb(p,True))
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
    uu0 = bc(solver.XXfull)
    uu=uu.flatten()
    errI=np.linalg.norm(uu-uu0,ord=np.inf)
    errInf = np.max([errInf,errI])
    print(errI)
print("sup norm error = ",errInf)