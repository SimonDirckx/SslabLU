# basic packages
import numpy as np
import jax.numpy as jnp
import scipy
from packaging.version import Version

import torch


# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms

from hpsmultidomain.geom              import ParametrizedGeometry3D

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


#########################################################################################################
#
#   SET-UP GEOMETRY:        3D Annulus
#   - forward transform     (z1,z2,z3)
#   - backward transform    (y1,y2,y3)
#   - derivatives (up to 2nd) of backward
#   - Annulus boundary      (gb)
#
#########################################################################################################

jax_avail = False
torch_avail = True
if jax_avail:
    const_theta = 1./(2.*np.pi)
    r           = lambda zz: (zz[...,0]**2 + zz[...,1]**2)**0.5

    z1 = lambda zz: jnp.multiply( 1 + 1 * zz[...,1], jnp.cos(zz[...,0]/const_theta) )
    z2 = lambda zz: jnp.multiply( 1 + 1 * zz[...,1], jnp.sin(zz[...,0]/const_theta) )
    z3 = lambda zz: zz[...,2]


    y1 = lambda zz: const_theta*jnp.atan2(zz[...,1],zz[...,0])
    y2 = lambda zz: r(zz) - 1
    y3 = lambda zz: zz[...,2]

    y1_d1    = lambda zz: -const_theta     * jnp.divide(zz[...,1], r(zz)**2)
    y1_d2    = lambda zz: +const_theta     * jnp.divide(zz[...,0], r(zz)**2)
    y1_d1d1  = lambda zz: +2*const_theta   * jnp.divide(jnp.multiply(zz[...,0],zz[...,1]), r(zz)**4)
    y1_d2d2  = lambda zz: -2*const_theta   * jnp.divide(jnp.multiply(zz[...,0],zz[...,1]), r(zz)**4)
    y1_d1d1 = None; y1_d2d2 = None


    y2_d1    = lambda zz: jnp.divide(zz[...,0], r(zz))
    y2_d2    = lambda zz: jnp.divide(zz[...,1], r(zz))
    y2_d1d1  = lambda zz: jnp.divide(zz[...,1]**2, r(zz)**3)
    y2_d2d2  = lambda zz: jnp.divide(zz[...,0]**2, r(zz)**3)

    y3_d3    = lambda zz: jnp.ones(shape=zz[...,2].shape)

    bnds = [[0.,0.,0.],[1.,1.,1.]]
    box_geom   = jnp.array(bnds)
    param_geom = ParametrizedGeometry3D(box_geom,z1,z2,z3,y1,y2,y3,\
                        y1_d1=y1_d1, y1_d2=y1_d2,\
                        y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                        y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2,\
                        y3_d3=y3_d3)
elif torch_avail:
    const_theta = 1/(2.*torch.pi)
    r           = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5

    z1 = lambda zz: torch.mul( 1 + 1 * zz[:,1], torch.cos(zz[:,0]/const_theta) )
    z2 = lambda zz: torch.mul( 1 + 1 * zz[:,1], torch.sin(zz[:,0]/const_theta) )
    z3 = lambda zz: zz[:,2]


    y1 = lambda zz: const_theta* torch.atan2(zz[:,1],zz[:,0])
    y2 = lambda zz: r(zz) - 1
    y3 = lambda zz: zz[:,2]

    y1_d1    = lambda zz: -const_theta     * torch.div(zz[:,1], r(zz)**2)
    y1_d2    = lambda zz: +const_theta     * torch.div(zz[:,0], r(zz)**2)
    y1_d1d1  = lambda zz: +2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
    y1_d2d2  = lambda zz: -2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
    y1_d1d1 = None; y1_d2d2 = None


    y2_d1    = lambda zz: torch.div(zz[:,0], r(zz))
    y2_d2    = lambda zz: torch.div(zz[:,1], r(zz))
    y2_d1d1  = lambda zz: torch.div(zz[:,1]**2, r(zz)**3)
    y2_d2d2  = lambda zz: torch.div(zz[:,0]**2, r(zz)**3)

    y3_d3    = lambda zz: torch.ones(zz[:,2].shape)
    bnds = [[0.,0.,0.],[1.,1.,1.]]
    
    box_geom   = np.array(bnds)
    param_geom = ParametrizedGeometry3D(box_geom,z1,z2,z3,y1,y2,y3,\
                        y1_d1=y1_d1, y1_d2=y1_d2,\
                        y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                        y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2,\
                        y3_d3=y3_d3)
else:
    const_theta = 1/(2.*np.pi)
    r           = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5

    z1 = lambda zz: np.multiply( 1 + 1 * zz[:,1], np.cos(zz[:,0]/const_theta) )
    z2 = lambda zz: np.multiply( 1 + 1 * zz[:,1], np.sin(zz[:,0]/const_theta) )
    z3 = lambda zz: zz[:,2]


    y1 = lambda zz: const_theta* np.atan2(zz[:,1],zz[:,0])
    y2 = lambda zz: r(zz) - 1
    y3 = lambda zz: zz[:,2]

    y1_d1    = lambda zz: -const_theta     * np.divide(zz[:,1], r(zz)**2)
    y1_d2    = lambda zz: +const_theta     * np.divide(zz[:,0], r(zz)**2)
    y1_d1d1  = lambda zz: +2*const_theta   * np.divide(np.multiply(zz[:,0],zz[:,1]), r(zz)**4)
    y1_d2d2  = lambda zz: -2*const_theta   * np.divide(np.multiply(zz[:,0],zz[:,1]), r(zz)**4)
    y1_d1d1 = None; y1_d2d2 = None


    y2_d1    = lambda zz: np.divide(zz[:,0], r(zz))
    y2_d2    = lambda zz: np.divide(zz[:,1], r(zz))
    y2_d1d1  = lambda zz: np.divide(zz[:,1]**2, r(zz)**3)
    y2_d2d2  = lambda zz: np.divide(zz[:,0]**2, r(zz)**3)

    y3_d3    = lambda zz: np.ones(shape=zz[:,2].shape)
    bnds = [[0.,0.,0.],[1.,1.,1.]]
    
    box_geom   = np.array(bnds)
    param_geom = ParametrizedGeometry3D(box_geom,z1,z2,z3,y1,y2,y3,\
                        y1_d1=y1_d1, y1_d2=y1_d2,\
                        y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                        y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2,\
                        y3_d3=y3_d3)
    
#def gb(p):
#    return ((jnp.abs(p[...,1]-bnds[0][1]))<1e-14) | ((jnp.abs(p[...,1]-bnds[1][1]))<1e-14) | (jnp.abs(p[...,2]-bnds[0][2])<1e-14) | (jnp.abs(p[...,2]-bnds[1][2])<1e-14)
def gb(p):
    return (np.abs(p[:,1]-bnds[0][1])<1e-14) | (np.abs(p[:,1]-bnds[1][1])<1e-14) | (np.abs(p[:,2]-bnds[0][2])<1e-14) | (np.abs(p[:,2]-bnds[1][2])<1e-14)

#def gb_torch(p):
#    return torch.abs(p[:,1]-bnds[0][1])<1e-14 or torch.abs(p[:,1]-bnds[1][1])<1e-14 or torch.abs(p[:,2]-bnds[0][2])<1e-14 or torch.abs(p[:,2]-bnds[1][2])<1e-14

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

nwaves = 10.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi

if jax_avail:
    def bfield(p,kh):
        return -kh*kh*jnp.ones_like(p[...,0])
elif torch_avail:
    def bfield(p,kh):
        return -kh*kh*torch.ones(p.shape[0])
else:
    def bfield(p,kh):
        return -kh*kh*np.ones(shape=(p.shape[0],))

pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)

def bc(p):
    z=z1(p)
    return np.sin(kh*z)

def u_exact(p):
    z=z1(p)
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
p = 10 + 2 # There's a difference in standards between the hps implementations - for hpsmultidomain p is internal order and q is external, for spectralmultidomain it is opposite
a = np.array([H/8.,1/8,1/8])
assembler = mA.rkHMatAssembler(p*p,75)
opts = solverWrap.solverOptions('hps',[p,p,p],a)
OMS = oms.oms(slabs,pdo_mod,gb,opts,connectivity,if_connectivity,1.)
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
    g = torch.from_numpy(g)
    uu = solver.solver.solve_dir_full(g)
    uu0 = bc(solver.XXfull)
    uu=uu.flatten()
    errI=np.linalg.norm(uu-uu0,ord=np.inf)
    errInf = np.max([errInf,errI])
    print(errI)
print("sup norm error = ",errInf)

