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
import matplotlib.cm as cm
from scipy.spatial import Delaunay 

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


nwaves = 24.623521102434587
#nwaves = 24.673521102434584
#kh = (nwaves+0.03)*2*np.pi+1.8
#kh=157.02
#print("kh = ",kh)
#kapp = 11.1
#nwaves = 24.673521102434584
kh = (nwaves+0.03)*2*np.pi+1.8

def bfield(xx):
    
    mag   = 0.930655
    width = 2500; 
    
    b = np.zeros(shape = (xx.shape[0],))
    
    dist = 0.04
    x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
    y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
    
    # box of points [x0,x1] x [y0,y1]
    for x in np.arange(x0,x1,dist):
        for y in np.arange(y0,y1,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * np.exp(-width * xx_sq_c)

    # box of points [x0,x1] x [y0,y2]
    for x in np.arange(x2,x3,dist):
        for y in np.arange(y0,y2-0.5*dist,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * np.exp(-width * xx_sq_c)
            
    # box of points [x0,x3] x [y2,y3]
    for x in np.arange(x0,x3,dist):
        for y in np.arange(y2,y3,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * np.exp(-width * xx_sq_c)    
    
    return (1 - b)

def bfieldMESH(X,Y):
    
    mag   = 0.930655
    width = 2500; 
    
    b = np.zeros(shape = X.shape)
    
    dist = 0.04
    x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
    y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
    
    # box of points [x0,x1] x [y0,y1]
    for x in np.arange(x0,x1,dist):
        for y in np.arange(y0,y1,dist):
            xx_sq_c = (X - x)**2 + (Y - y)**2
            b += mag * np.exp(-width * xx_sq_c)

    # box of points [x0,x1] x [y0,y2]
    for x in np.arange(x2,x3,dist):
        for y in np.arange(y0,y2-0.5*dist,dist):
            xx_sq_c = (X - x)**2 + (Y - y)**2
            b += mag * np.exp(-width * xx_sq_c)
            
    # box of points [x0,x3] x [y2,y3]
    for x in np.arange(x0,x3,dist):
        for y in np.arange(y2,y3,dist):
            xx_sq_c = (X - x)**2 + (Y - y)**2
            b += mag * np.exp(-width * xx_sq_c)    
    
    #kh_fun = -kh**2 * (1 - b)
    return 1-b


def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return bfield(p)
Lapl=pdo.PDO2d(c11,c22,None,None,None,c)

def bc(p):
    return 1.

bnds = [[0.,0.],[1.,1.]]
Om=stdGeom.Box(bnds)
def gb(p):
    return np.abs(p[0]-bnds[0][0])<1e-10 or np.abs(p[0]-bnds[1][0])<1e-14 or np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14

H = 1./4.
N = (int)(1./H)
p = 10
a = 1./32.

uitot = np.zeros(shape=(0,1))
btot = np.zeros(shape=(0,))
XXtot = np.zeros(shape=(0,2))
dofs = 0

for i in range(N):
    xl = i*H
    xr = (i+1)*H
    geom = hpsGeom.BoxGeometry(np.array([[xl,0.],[xr,1.]]))
    disc = HPS.HPSMultidomain(Lapl, geom, a, p)
    XX = disc._XXfull
    b = bfield(XX)
    #if i<N-1:
    #    I = [i for i in range(XX.shape[0]) if XX[i,0]<xr]
    #else:
    #    I=range(XX.shape[0])
    btot=np.append(btot,b,axis=0)
    XXtot=np.append(XXtot,XX,axis=0)


XXtot,I = np.unique(XXtot,axis=0,return_index=True)
btot = btot[I]

Ix = [i for i in range(XXtot.shape[0]) if np.abs(XXtot[i,0])<1e-10]
Iy = [i for i in range(XXtot.shape[0]) if np.abs(XXtot[i,1])<1e-10]

xpts = XXtot[Ix,1]
ypts = XXtot[Iy,0]

X,Y = np.meshgrid(xpts,ypts)

B = bfieldMESH(X,Y)


resolution = 1000
min_x = np.min(XXtot[:,0])
max_x = np.max(XXtot[:,0])
min_y = np.min(XXtot[:,1])
max_y = np.max(XXtot[:,1])
xpts = np.linspace(min_x,max_x,resolution)
ypts = np.linspace(min_y,max_y,resolution)
grid_x, grid_y    = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]
grid_b           = griddata(XXtot, btot, (grid_x, grid_y), method='cubic').T

gsqXY = np.zeros(shape=(resolution*resolution,2))
for i in range(resolution):
    for j in range(resolution):
        gsqXY[i+j*resolution,:] = [xpts[i],ypts[j]]
tri = Delaunay(gsqXY)

plt.figure(0)
bvec = grid_b.flatten()

plt.tripcolor(gsqXY[:,0],gsqXY[:,1],bvec,triangles = tri.simplices.copy(),cmap='jet',shading='gouraud',antialiased=False,linewidth=0)
plt.colorbar()
bfieldstr = 'bfield_uniform_grid_tripcolor_'+str(p)+'_'+str(a)+'.png'
plt.axis('equal')
plt.savefig(bfieldstr, transparent=True,format='png',bbox_inches='tight')


plot_pad = .1
min_b = 0
max_b = 1

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
im = ax.imshow(grid_b, extent=(min_x-plot_pad,max_x+plot_pad,\
                                       min_y-plot_pad,max_y+plot_pad),\
                   vmin=min_b, vmax=max_b,\
                   origin='lower',cmap='jet',interpolation='bicubic')
plt.axis('equal')
bfieldstr = 'bfield_uniform_grid_imcolor_'+str(p)+'_'+str(a)+'.png'
plt.savefig(bfieldstr, transparent=True,format='png',bbox_inches='tight')
plt.figure(2)
plt.contourf(X, Y, B, 100, origin = 'lower',cmap=cm.jet)
plt.axis('equal')
plt.colorbar()
bfieldstr = 'bfield_cheb_grid_contourf_'+str(p)+'_'+str(a)+'.png'
plt.savefig(bfieldstr, transparent=True,format='png',bbox_inches='tight')
tri = Delaunay(XXtot)

plt.figure(3)

plt.tripcolor(XXtot[:,0],XXtot[:,1],btot,triangles = tri.simplices.copy(),cmap='jet',shading='gouraud',antialiased=False,linewidth=0)
plt.colorbar()
plt.axis('equal')
bfieldstr = 'bfield_cheb_grid_tripcolor_'+str(p)+'_'+str(a)+'.png'
plt.savefig(bfieldstr, transparent=True,format='png',bbox_inches='tight')
plt.show()
