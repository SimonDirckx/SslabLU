import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
import time
import hps.hps_multidomain as HPS
import hps.geom as hpsGeom
from scipy.sparse        import block_diag
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.sparse.linalg   import LinearOperator
from scipy.sparse.linalg import gmres
from solver.solver import stMap
import matAssembly.matAssembler as mA
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay 
from hps.geom              import BoxGeometry, ParametrizedGeometry2D,ParametrizedGeometry3D
import scipy.special as special
from matplotlib.patches import Polygon
from hps.geom              import BoxGeometry, ParametrizedGeometry2D,ParametrizedGeometry3D
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import solver.HPSInterp3D as interp
import multislab.oms as oms
import jax.numpy as jnp
#import jax
#jax.config.update('jax_platform_name', 'cpu')


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
def mesh(slab):
    xpts = np.linspace(slab[0][0],slab[1][0],20)
    ypts = np.linspace(slab[0][1],slab[1][1],20)
    nx = len(xpts)
    ny = len(ypts)
    XY = np.zeros(shape = (nx*ny,2))
    for j in range(ny):
        for i in range(nx):
            XY[j+i*ny] = [xpts[i],ypts[j]]
    return XY


# description: slab1<slab2 assumed!
def join(slab1,slab2,period=None):
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
            return join([[xl1,yl1,zl1],[xr1,yr1,zr1]],slab2)
        else:
            ValueError("slab shift did not work (is your period correct?)")
    else:
        totalSlab = [[xl1, yl1,zl1],[xr2,yr2,zr2]]
    return totalSlab
# the final diameter of the domain is 4, so kh = (nwaves/4)*2pi
nwaves = 2.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi
def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return -kh*kh*np.ones(shape=(p.shape[0],))




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


    def bfield(p,kh):
        return -kh*kh*jnp.ones_like(p[...,0])

    bnds = [[0.,0.,0.],[1.,1.,1.]]
    box_geom   = jnp.array(bnds)
    param_geom = ParametrizedGeometry3D(box_geom,z1,z2,z3,y1,y2,y3,\
                        y1_d1=y1_d1, y1_d2=y1_d2,\
                        y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                        y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2,\
                        y3_d3=y3_d3)
    pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)
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
    def bfield(p,kh):
        return -kh*kh*np.ones(shape=(p.shape[0],))
    box_geom   = np.array(bnds)
    Om=stdGeom.Box(bnds)
    param_geom = ParametrizedGeometry3D(box_geom,z1,z2,z3,y1,y2,y3,\
                        y1_d1=y1_d1, y1_d2=y1_d2,\
                        y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                        y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2,\
                        y3_d3=y3_d3)
    pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)


def bc(p):
    #r = np.sqrt((z1(p)+3)**2+(z2(p))**2)
    z=z1(p)
    return np.sin(kh*z)#special.yn(0, kh*r)/4.
def u_exact(p):
    #r = np.sqrt((z1(p)+3)**2+(z2(p))**2)
    z=z1(p)
    return np.sin(kh*z)#special.yn(0, kh*r)/4

# periodic: x=0 and x=1 NOT part of gb
def gb(p):
    return np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14 or np.abs(p[2]-bnds[0][2])<1e-14 or np.abs(p[2]-bnds[1][2])<1e-14


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

period = 1.
tol = 1e-5

data = 0
p = 6
a = [H/2.,1/4,1/4]
assembler = mA.rkHMatAssembler((p+2)*(p+2),60)
opts = solverWrap.solverOptions('hps',[p,p,p],a)
OMS = oms.oms(slabs,pdo_mod,gb,opts,connectivity,if_connectivity)
print("computing Stot & rhstot...")
Stot_LO,rhstot0 = OMS.construct_Stot_and_rhstot(bc,assembler)
print("done")
gInfo = gmres_info()
stol = 1e-10*H*H
uhat,info   = gmres(Stot_LO,rhstot0,tol=stol,callback=gInfo,maxiter=100,restart=100)
stop_solve = time.time()
res = Stot_LO@uhat-rhstot0
nc = len(OMS.glob_target_dofs[0])
print('wavelength = ',wavelength)
print('ppw = ',wavelength*nc)

print("=============SUMMARY==============")
print("H                        = ",'%10.3E'%H)
print("ord                      = ",p)
print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot0))
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




for i in range(len(slabs)):
    slab = slabs[i]
    ul = uhat[OMS.glob_target_dofs[i]]
    ur = uhat[OMS.glob_source_dofs[i][1]]
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