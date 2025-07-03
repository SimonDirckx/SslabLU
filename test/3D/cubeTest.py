# basic packages
import numpy as np
import jax.numpy as jnp
import scipy
from packaging.version import Version
import hps.pdo as pdo

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms


# validation&testing
import time
from scipy.sparse.linalg import gmres
import solver.HPSInterp3D as interp
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


jax_avail = True
if jax_avail:
    bnds = [[0.,0.,0.],[1.,1.,1.]]
    box_geom   = jnp.array(bnds)
else:
    bnds = [[0.,0.,0.],[1.,1.,1.]]
    box_geom   = np.array(bnds)


def gb_vec(P):
    # P is (N, 3)
    return (
        (np.abs(P[:, 0] - bnds[0][0]) < 1e-14) |
        (np.abs(P[:, 0] - bnds[1][0]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[0][1]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[1][1]) < 1e-14) | 
        (np.abs(P[:, 2] - bnds[0][2]) < 1e-14) | 
        (np.abs(P[:, 2] - bnds[1][2]) < 1e-14)
    )

#########################################################################################################


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

nwaves = 5.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi

def c11(p):
    return jnp.ones_like(p[...,0])
def c22(p):
    return jnp.ones_like(p[...,0])
def c33(p):
    return jnp.ones_like(p[...,0])

helmholtz = pdo.PDO3d(c11=c11,c22=c22,c33=c33,c=pdo.const(-kh*kh))

def bc(p):
    return np.sin(kh*p[:,0])

def u_exact(p):
    return np.sin(kh*p[:,0])

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


H = 1./8.
N = (int)(1./H)
slabs = []
for n in range(N):
    bnds_n = [[n*H,0.,0.],[(n+1)*H,1.,1.]]
    slabs+=[bnds_n]

connectivity = []
for i in range(N-1):
    connectivity+=[[i,i+1]]

if_connectivity = []
for i in range(N-1):
    if i==0:
        if_connectivity+=[[-1,(i+1)]]
    elif i==N-2:
        if_connectivity+=[[(i-1),-1]]
    else:
        if_connectivity+=[[(i-1),(i+1)]]

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
p = 8
a = [H/2.,1/16,1/16]
assembler = mA.denseMatAssembler()#mA.rkHMatAssembler(p*p,160)
opts = solverWrap.solverOptions('hps',[p,p,p],a)
OMS = oms.oms(slabs,helmholtz,gb_vec,opts,connectivity,if_connectivity)
print("computing Stot & rhstot...")
Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler,2)
print("done")
#################################################################
#E = np.identity(Stot.shape[1])
#S00 = Stot@E
#plt.spy(S00)
#plt.show()
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


slabInd = 0
geom    = np.array(oms.join_geom(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]]))
slab_i  = oms.slab(geom,gb)
solver  = oms.solverWrap.solverWrapper(opts)
solver.construct(geom,helmholtz)
Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
XXIc = np.array(XXi)[Ic,:]
u_known = np.sin(kh*XXIc[:,0])
ul = uhat[:nc]
ur = uhat[nc:2*nc]

print("err = ",np.linalg.norm(ul-u_known)/np.linalg.norm(u_known))

'''
for i in range(len(slabs)):
    slab = slabs[i]
    ul = uhat[glob_target_dofs[i]]
    ur = uhat[glob_source_dofs[i][1]]
    interp.check_err(slab,ul,ur,a,p,pdo_mod,gb,bc,u_exact)
'''
