import sys
sys.path.append('/home/simond/SslabLU')
import numpy as np
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

import time


def join_geom(slab1,slab2,period=None):
    
    xl1 = slab1[0][0]
    xr1 = slab1[1][0]
    yl1 = slab1[0][1]
    yr1 = slab1[1][1]
    zl1 = slab1[0][2]
    zr1 = slab1[1][2]

    xl2 = slab2[0][0]
    xr2 = slab2[1][0]
    yl2 = slab2[0][1]
    yr2 = slab2[1][1]
    zl2 = slab2[0][2]
    zr2 = slab2[1][2]
    if(np.abs(xr1-xl2)>1e-10):
        if period:
            xl1 -= period
            xr1 -= period
            return join_geom([[xl1,yl1,zl1],[xr1,yr1,zr1]],slab2)
        else:
            ValueError("slab shift did not work (is your period correct?)")
    else:
        totalSlab = [[xl1, yl1,zl1],[xr2,yr2,zr2]]
    return totalSlab
class slab:
    def __init__(self,geom,gb,transform=None):
        self.geom       =   geom
        self.transform  =   transform
        self.gb         =   gb

    def compute_idxs_and_pts(self,solver):
        XX = solver.XX
        XXb = XX[solver.Ib,:]
        XXi = XX[solver.Ii,:]
        xl = self.geom[0][0]
        xr = self.geom[1][0]
        xc=(xl+xr)/2.
        Il = [i for i in range(len(solver.Ib)) if np.abs(XXb[i,0]-xl)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
        Ir = [i for i in range(len(solver.Ib)) if np.abs(XXb[i,0]-xr)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
        Ic = [i for i in range(len(solver.Ii)) if np.abs(XXi[i,0]-xc)<1e-14]
        Igb = [i for i in range(len(solver.Ib)) if self.gb(XXb[i,:])]
        return Il,Ir,Ic,Igb,XXi,XXb
    
jax_avail = True

if jax_avail:
    const_theta = 1/(2.*np.pi)
    r           = lambda zz: (zz[...,0]**2 + zz[...,1]**2)**0.5

    z1 = lambda zz: jnp.multiply( 1 + 1 * zz[...,1], jnp.cos(zz[...,0]/const_theta) )
    z2 = lambda zz: jnp.multiply( 1 + 1 * zz[...,1], jnp.sin(zz[...,0]/const_theta) )
    z3 = lambda zz: zz[...,2]


    y1 = lambda zz: const_theta* jnp.atan2(zz[...,1],zz[...,0])
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
def gb(p):
    return np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14 or np.abs(p[2]-bnds[0][2])<1e-14 or np.abs(p[2]-bnds[1][2])<1e-14
nwaves = 2.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi

if jax_avail:
    def bfield(p,kh):
        return -kh*kh*jnp.ones_like(p[...,0])
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

H = 1./16.
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

p=8
a = [H/2.,1/16,1/16]

period = 1.
opts = solverWrap.solverOptions('hps',[p,p,p],a)

for slabInd in range(len(connectivity)):
    geom = np.array(join_geom(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]],period))
    slab_i = slab(geom,gb)
    print("construct HPS...")
    start = time.time()
    localSolver = solverWrap.solverWrapper(opts)
    localSolver.construct(geom,pdo_mod)
    stop = time.time()
    print("done in ",stop-start,"s")