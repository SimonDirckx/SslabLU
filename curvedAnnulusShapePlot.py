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
from matplotlib.patches import Polygon
from hps.geom              import BoxGeometry, ParametrizedGeometry2D,ParametrizedGeometry3D
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
import pyvista as pv

kh = 20.247

def bfield(xx,kh):
    
    b = np.ones(shape = (xx.shape[0],))
    
    kh_fun = -kh**2 * b
    return kh_fun


def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c33(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return bfield(p)
Lapl=pdo.PDO3d(c11,c22,c33,None,None,None,None,None,None,c)

def bc(p):
    return 1.

bnds = [[0.,0.,0],[3.,1.,.5]]
box_geom   = np.array([[0,0,0],[3.0,1.0,.5]])
Om=stdGeom.Box(bnds)

const_theta = 1/(np.pi/3)
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

y3_d3    = lambda zz: np.ones(shape=(zz.shape[0],))

param_geom = ParametrizedGeometry3D(box_geom,z1,z2,z3,y1,y2,y3,\
                    y1_d1=y1_d1, y1_d2=y1_d2,\
                    y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                    y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2,y3_d3=y3_d3)
pdo_mod = param_geom.transform_helmholtz_pdo(bfield, kh)

H = 1./4.
N = (int)(bnds[1][0]/H)
p = 22
a = H/2
def gb(p):
    return np.abs(p[0]-bnds[0][0])<1e-14 or np.abs(p[0]-bnds[1][0])<1e-14 or np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14 or np.abs(p[2]-bnds[0][2])<1e-14 or np.abs(p[2]-bnds[1][2])<1e-14

resolution = 50
min_x = bnds[0][0]
max_x = bnds[1][0]
min_y = bnds[0][1]
max_y = bnds[1][1]
min_z = bnds[0][2]
max_z = bnds[1][2]
xpts = np.linspace(min_x,max_x,resolution)
ypts = np.linspace(min_y,max_y,resolution)
zpts = np.linspace(min_z,max_z,resolution)


gXY = np.zeros(shape=(resolution*resolution*resolution,3))
gsqXY = np.zeros(shape=(resolution*resolution*resolution,3))
print("start loop")
for i in range(resolution):
    for j in range(resolution):
        for k in range(resolution):
            p=np.zeros(shape=(1,3))
            p[0,:] = np.array([xpts[i],ypts[j],zpts[k]])
            zp1 = z1(p)[0]
            zp2 = z2(p)[0]
            zp3 = z3(p)[0]
            gXY[i+j*resolution+k*resolution*resolution,:] = [zp1,zp2,zp3]
            gsqXY[i+j*resolution+k*resolution*resolution,:] = [xpts[i],ypts[j],zpts[k]]
gB = np.zeros(shape=(0,3))
for i in range(gXY.shape[0]):
    if gb(gsqXY[i,:]):
        p=np.reshape(gXY[i,:],newshape=(1,3))
        gB= np.append(gB,p,axis=0)
print("boundary done")
cloud = pv.PolyData(gB)
volume = cloud.delaunay_3d(alpha=.3)
shell = volume.extract_geometry()
shell.plot()