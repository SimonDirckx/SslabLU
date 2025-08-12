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
#import solver.HPSInterp3D as interp
import matplotlib.pyplot as plt

import twistedTorusGeometry as twisted

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
param_geom=twisted.param_geom()
pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)

def bc(p):
    z=twisted.z1(p)
    return np.sin(kh*z)

def u_exact(p):
    z=twisted.z1(p)
    return np.sin(kh*z)

################################################################


##############################################################################################
#
#   SET-UP Slabs
#   - left-to-right convention  (!!!)
#   - single slabs              (slabs)
#   - slab connectivity         (connectivity, i.e. are two single slabs connected)
#   - interface connectivity    (if_connectivity, i.e. are two interfaces connected by a slab) 
#   - periodicity               (period, i.e. period in the x-dir)
#
##############################################################################################


N = 8
slabs,H = twisted.slabs(N)
connectivity,if_connectivity = twisted.connectivity(slabs)
period = 2.

##############################################################################################

#################################################################
#
#   Compute OMS (overlapping multislab)
#   - discretization options    (opts)
#   - off-diag block assembler  (assembler)
#   - Overlapping Multislab     (OMS)
#
#################################################################

tol = 1e-5
p = 10
a = [H/8.,1/8,1/8]
assembler = mA.rkHMatAssembler(p*p,75)
opts = solverWrap.solverOptions('hps',[p,p,p],a)
OMS = oms.oms(slabs,pdo_mod,lambda p: twisted.gb(p,True),opts,connectivity,if_connectivity,1.)
print("computing Stot & rhstot...")
Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler,2)
print("done")
#################################################################
#E = np.identity(Stot.shape[1])
#S00 = Stot@E
#S00T = Stot.T@E
#print("T err = ",np.linalg.norm(S00.T-S00T))
#Finally, solve

gInfo = gmres_info()
stol = 1e-10*H*H

if Version(scipy.__version__)>=Version("1.14"):
    uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=100,restart=100)
else:
    uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=100,restart=100)

stop_solve = time.time()
res = Stot@uhat-rhstot


print("=============SUMMARY==============")
print("H                        = ",'%10.3E'%H)
print("ord                      = ",p)
print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
print("GMRES iters              = ", gInfo.niter)
print("==================================")

uitot = np.zeros(shape=(0,))
XXtot = np.zeros(shape=(0,3))
dofs = 0
glob_target_dofs=OMS.glob_target_dofs
glob_source_dofs=OMS.glob_source_dofs
nc = OMS.nc
del OMS



# check err.
print("uhat shape = ",uhat.shape)
print("uhat type = ",type(uhat))
print("nc = ",nc)

fig = plt.figure(1)
N=len(connectivity)
errInf = 0.
for slabInd in range(len(connectivity)):
    geom    = np.array(oms.join_geom(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]],period))
    slab_i  = oms.slab(geom,gb)
    solver  = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom,pdo_mod)
    
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)

    startL = ((slabInd-1)%N)
    startR = ((slabInd+1)%N)
    ul = uhat[startL*nc:(startL+1)*nc]
    ur = uhat[startR*nc:(startR+1)*nc]
    u0l = bc(XXb[Il,:])
    u0r = bc(XXb[Ir,:])
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

