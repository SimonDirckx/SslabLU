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
p = 20
a = H/4


uitot = np.zeros(shape=(0,1))
btot = np.zeros(shape=(0,))
XXtot = np.zeros(shape=(0,2))
dofs = 0

uhat = np.load('uhat_3dAnnulus.npy')

for i in range(N):
    xl = i*H
    xr = (i+1)*H
    z00 = 0.
    z01 = .5
    geom = hpsGeom.BoxGeometry(np.array([[xl,0.,z00],[xr,1.,z01]]))
    zmid = (z00+z01)/2
    disc = HPS.HPSMultidomain(pdo_mod, geom, a, p)
    print("hps done")
    XX = disc._XX
    print("XX done")
    XXb = XX[disc.Jx,:]
    print("XXi done")
    XXi = XX[disc.Ji,:]
    print("XXb done")
    Ir = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xr)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10]
    Il = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xl)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10]
    nc = len(Ir)
    print("Is done")
    bvec = np.ones(shape=(len(disc.Jx),1))
    if i>0:
        bvec[Il,0] = uhat[(i-1)*nc:i*nc]
    if i<N-1:
        bvec[Ir,0] = uhat[i*nc:(i+1)*nc]
    print("b done")
    dofs+=bvec.shape[0]-nc
    ui = disc.solve_dir_full(bvec)
    print("u done")
    dofs+=ui.shape[0]
    zztot = disc._XXfull#np.zeros(shape=(disc._XXfull.shape[0],3))
    #zztot[:,0] = z1(disc._XXfull)
    #zztot[:,1] = z2(disc._XXfull)
    #zztot[:,2] = z3(disc._XXfull)
    print("zztot done")
    idxz0 = [i for i in range(zztot.shape[0]) if np.abs(zztot[i,2]-zmid)<1e-10]
    print("len(idxz0) = ",len(idxz0))
    zz0 = zztot[idxz0,0:2]
    XXtot=np.append(XXtot,zz0,axis=0)
    uitot=np.append(uitot,ui[idxz0,:],axis=0)
    Inr = [i for i in range(XXb.shape[0]) if not i in Ir]
    XXbnr = XXb[Inr,:]
    zzb = XXb[Inr,:]#np.zeros(shape=(XXbnr.shape[0],3))
    #zzb[:,0] = z1(XXbnr)
    #zzb[:,1] = z2(XXbnr)
    #zzb[:,2] = z3(XXbnr)
    idxz00 = [i for i in range(zzb.shape[0]) if np.abs(zzb[i,2]-zmid)<1e-10]
    zzb0=zzb[idxz00,0:2]
    XXtot=np.append(XXtot,zzb0,axis=0)
    uitot=np.append(uitot,bvec[idxz00],axis=0)
    print("XX data (MB) = ",XXtot.data.nbytes/1e6)
    print("u data  (MB) = ",uitot.data.nbytes/1e6)
    del geom,disc,zztot,XX,XXi,XXb,ui,bvec,zz0,Ir,Il


print('u shape = ',uitot.shape)
print('XX shape = ',XXtot.shape)
print('total dofs = ',dofs)

resolution = 1000
min_x = np.min(XXtot[:,0])#bnds[0][0]
max_x = np.max(XXtot[:,0])#bnds[1][0]
min_y = np.min(XXtot[:,1])#bnds[0][1]
max_y = np.max(XXtot[:,1])#bnds[1][1]
xpts = np.linspace(min_x,max_x,resolution)
ypts = np.linspace(min_y,max_y,resolution)
grid_x, grid_y    = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]

grid_solution           = griddata(XXtot, uitot[:,0], (grid_x, grid_y), method='cubic').T
print("grid_solution shape = ",grid_solution.shape)

print("grid_x shape = ",grid_x.shape)

gXY = np.zeros(shape=(grid_solution.shape[0]*grid_solution.shape[0],2))
gsqXY = np.zeros(shape=(grid_solution.shape[0]*grid_solution.shape[0],2))
for i in range(resolution):
    for j in range(resolution):
        p=np.zeros(shape=(1,3))
        p[0,:] = np.array([xpts[i],ypts[j],.25])
        zp1 = z1(p)[0]
        zp2 = z2(p)[0]
        gXY[i+j*resolution,:] = [zp1,zp2]
        gsqXY[i+j*resolution,:] = [xpts[i],ypts[j]]
tri = Delaunay(gsqXY)
plt.figure(0)
gsol = grid_solution.flatten()

plt.tripcolor(gXY[:,0],gXY[:,1],gsol,triangles = tri.simplices.copy(),cmap='jet',shading='gouraud')
plt.colorbar()
plt.axis('equal')
plt.savefig('annulus.png', transparent=True,format='png',bbox_inches='tight')
plt.show()
