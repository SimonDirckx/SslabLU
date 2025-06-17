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
import solver.HPSInterp3D as interp


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
p = 10
a = [H/4.,1/16,1/16]
assembler = mA.rkHMatAssembler((p+2)*(p+2),160)
opts = solverWrap.solverOptions('hps',[p,p,p],a)
OMS = oms.oms(slabs,pdo_mod,gb,opts,connectivity,if_connectivity,1.)
print("computing Stot & rhstot...")
Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler,2)
print("done")
#################################################################


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
#print("constuction time rk.     = ",trk)
#print("par. constuction time rk.= ",trk/(N-1))
#print("solve time               = ",(stop_solve-start_solve))
#print("par. solve time          = ",(stop_solve-start_solve)/(N-1))
#print("data (MB)                = ",data/1e6)
#print("data orig (MB)           = ",(8*Ntot+8*(nc*nc)*2.*(N-1))/1e6)
print("==================================")

uitot = np.zeros(shape=(0,))
XXtot = np.zeros(shape=(0,3))
dofs = 0
glob_target_dofs=OMS.glob_target_dofs
glob_source_dofs=OMS.glob_source_dofs

del OMS

# check err.
for i in range(len(slabs)):
    slab = slabs[i]
    ul = uhat[glob_target_dofs[i]]
    ur = uhat[glob_source_dofs[i][1]]
    interp.check_err(slab,ul,ur,a,p,pdo_mod,gb,bc,u_exact)


'''
XXtot,I=np.unique(XXtot,axis=0,return_index=True)
ui_exact = u_exact(XXtot)
uitot=uitot[I]
print('total u err inf = ',np.linalg.norm(ui_exact-uitot,ord=np.inf))
ZZ = np.zeros(shape = XXtot.shape)
ZZ[:,0] = z1(XXtot)
ZZ[:,1] = z2(XXtot)
ZZ[:,2] = z3(XXtot)
tri = Delaunay(XXtot[:,0:2])
plt.figure(0)
plt.tripcolor(ZZ[:,0],ZZ[:,1],uitot,triangles = tri.simplices.copy(),cmap='jet',shading='gouraud',antialiased=False,linewidth=0)
plt.colorbar()
plt.axis('equal')
plt.show()
'''