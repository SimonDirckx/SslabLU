import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib import cm
import matplotlib.tri as tri
import twistedTorusGeometry as twisted
import jax.numpy as jnp
import multislab.oms as oms
import matAssembly.matAssembler as mA
import scipy
from packaging.version import Version

# validation&testing
import time
from scipy.sparse.linalg import gmres
#import solver.HPSInterp3D as interp
import matplotlib.pyplot as plt

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

nwaves = 1.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi


def bfield(p,kh):
    return -kh*kh*jnp.ones_like(p[...,0])

def bc(p):
    z=twisted.z1(p,True)
    return np.sin(kh*z)



param_geom = twisted.param_geom(True)
pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)


H = 1./8.
N = (int)(1./H)
slabs = []
for n in range(N):
    bnds_n = [[n*H,0.,0.],[(n+1)*H,1.,1.]]
    slabs+=[bnds_n]

connectivity = [[N-1,0]]
for i in range(N-1):
    connectivity+=[[i,i+1]]
if_connectivity = []
for i in range(N):
    if_connectivity+=[[(i-1)%N,(i+1)%N]]

period = 1.
p = 8
a = [H/8.,1/16,1/16]
assembler = mA.rkHMatAssembler(p*p,200)
opts = solverWrap.solverOptions('hps',[p,p,p],a)

'''
OMS = oms.oms(slabs,pdo_mod,lambda p:twisted.gb(p,True),opts,connectivity,if_connectivity,1.)
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

'''
N=len(connectivity)
errInf = 0.
for slabInd in range(len(connectivity)):
    geom    = np.array(oms.join_geom(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]],period))
    slab_i  = oms.slab(geom,lambda p:twisted.gb(p,True))
    solver  = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom,pdo_mod)
    
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)

    startL = ((slabInd-1)%N)
    startR = ((slabInd+1)%N)
    #ul = uhat[startL*nc:(startL+1)*nc]
    #ur = uhat[startR*nc:(startR+1)*nc]
    u0l = bc(XXb[Il,:])
    u0r = bc(XXb[Ir,:])
    g = np.zeros(shape=(XXb.shape[0],))
    g[Il]=bc(XXb[Il,:])
    g[Ir]=bc(XXb[Ir,:])
    g[Igb] = bc(XXb[Igb,:])
    g=g[:,np.newaxis]
    uu = solver.solver.solve_dir_full(g)
    uu0 = bc(solver.XXfull)
    uu=uu.flatten()
    errI=np.linalg.norm(uu-uu0,ord=np.inf)
    errInf = np.max([errInf,errI])
    print(errI)
print("sup norm error = ",errInf)
