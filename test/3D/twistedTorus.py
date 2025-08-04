import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
#import multiSlab as MS
#import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
#import geometry.standardGeometries as stdGeom
#import geometry.skeleton as skelTon
#import time
#import hps.hps_multidomain as HPS
#import hps.geom as hpsGeom
#from scipy.sparse        import block_diag
#import scipy.sparse as sparse
#import scipy.sparse.linalg as splinalg
#from scipy import interpolate
#from scipy.interpolate import griddata
#from scipy.sparse.linalg   import LinearOperator
#from scipy.sparse.linalg import gmres
#from solver.solver import stMap
#import matAssembly.matAssembler as mA
#from matplotlib.patches import Polygon
#from hps.geom              import BoxGeometry, ParametrizedGeometry2D,ParametrizedGeometry3D
#import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
from matplotlib import cm
import matplotlib.tri as tri

'''
def z1(p):
    c=np.cos(np.pi*p[1])
    s=np.sin(np.pi*p[1])
    rot = np.array([[c,-s],[s,c]])
    q = rot@np.array([p[0],p[2]])
    return c*(R+1+q[0])

def z2(p):
    c=np.cos(np.pi*p[1])
    s=np.sin(np.pi*p[1])
    rot = np.array([[c,-s],[s,c]])
    q = rot@np.array([p[0],p[2]])
    return s*(R+1+q[0])
def z3(p):
    c=np.cos(np.pi*p[1])
    s=np.sin(np.pi*p[1])
    rot = np.array([[c,-s],[s,c]])
    q = rot@np.array([p[0],p[2]])
    return q[1]
'''

def z1(p):
    c=np.cos(np.pi*p[:,1])
    s=np.sin(np.pi*p[:,1])
    c2 = np.multiply(c,c)
    cs = np.multiply(c,s)
    q = np.multiply(c2,p[:,0])-np.multiply(cs,p[:,2])+c*(R+1)
    return q

def z2(p):
    c=np.cos(np.pi*p[:,1])
    s=np.sin(np.pi*p[:,1])
    s2 = np.multiply(s,s)
    cs = np.multiply(c,s)
    q = np.multiply(cs,p[:,0])-np.multiply(s2,p[:,2])+s*(R+1)
    return q
def z3(p):
    c=np.cos(np.pi*p[:,1])
    s=np.sin(np.pi*p[:,1])
    q = np.multiply(s,p[:,0])+np.multiply(c,p[:,2])
    return q


def y1(p):
    # p is a vector of points, Nx3
    th = np.arctan2(p[:,1],p[:,0])
    c=np.cos(th)
    s=np.sin(th)
    c2 = np.multiply(c,c)
    cs = np.multiply(c,s)
    q = np.multiply(p[:,0],c2)+np.multiply(p[:,1],cs)-(R+1)*c + np.multiply(s,p[:,2])
    return q

def y2(p):
    th = np.arctan2(p[:,1],p[:,0])
    return th/np.pi

def y3(p):
    th = np.arctan2(p[:,1],p[:,0])
    c=np.cos(th)
    s=np.sin(th)
    s2 = np.multiply(s,s)
    cs = np.multiply(c,s)
    q = -np.multiply(p[:,0],cs)-np.multiply(p[:,1],s2)+(R+1)*s+np.multiply(c,p[:,2])
    return q

def y1_d1(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] - c2t*p[:,1]*p[:,1] - (R+1)*s*p[:,1] - p[:,2]*p[:,1]*c)
    return (c2t+1)/2. + A/r2

def y1_d2(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = ( c2t*p[:,0]*p[:,1] - s2t*p[:,0]*p[:,0] + (R+1)*s*p[:,0] + p[:,2]*p[:,0]*c)
    return s2t/2. +A/r2

def y1_d3(p):
    th = np.arctan2(p[:,1],p[:,0])
    return np.sin(th)


def y2_d1(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return -(p[:,1]/r2)/np.pi

def y2_d2(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return (p[:,0]/r2)/np.pi


def y3_d1(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (c2t*p[:,0]*p[:,1] + s2t*p[:,0]*p[:,0] - (R+1)*c*p[:,1] + p[:,2]*p[:,1]*s)
    return -s2t/2. + A/r2

def y3_d2(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] + c2t*p[:,0]*p[:,0] - (R+1)*c*p[:,0] + p[:,2]*p[:,0]*s)
    return (c2t-1)/2. - A/r2

def y3_d3(p):
    th = np.arctan2(p[:,1],p[:,0])
    return np.cos(th)




n=100
R = 1.5
xpts = np.linspace(-1,1,n)
ypts = np.linspace(-1,1,4*n)
zpts = np.linspace(-1,1,n)
nx = len(xpts)
ny = len(ypts)
nz = len(zpts)

def gb(p):
    return np.abs(p[0]+1)<1e-10 or np.abs(p[0]-1)<1e-10 or np.abs(p[2]+1)<1e-10 or np.abs(p[2]-1)<1e-10
def gbu(p):
    return np.abs(p[2]-1)<1e-10
def gbd(p):
    return np.abs(p[2]+1)<1e-10
def gbl(p):
    return np.abs(p[0]+1)<1e-10
def gbr(p):
    return np.abs(p[0]-1)<1e-10

XXu= np.zeros(shape=(0,3))
XXd= np.zeros(shape=(0,3))
XXl= np.zeros(shape=(0,3))
XXr= np.zeros(shape=(0,3))

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            p=np.array([xpts[i],ypts[j],zpts[k]])
            if gbu(p):
                XXu=np.append(XXu,np.reshape(p,newshape=(1,3)),axis=0)
            if gbd(p):
                XXd=np.append(XXd,np.reshape(p,newshape=(1,3)),axis=0)
            if gbl(p):
                XXl=np.append(XXl,np.reshape(p,newshape=(1,3)),axis=0)
            if gbr(p):
                XXr=np.append(XXr,np.reshape(p,newshape=(1,3)),axis=0)

triu = Delaunay(XXu[:,0:2]).simplices
trid = Delaunay(XXd[:,0:2]).simplices
tril = Delaunay(XXl[:,1:3]).simplices
trir = Delaunay(XXr[:,1:3]).simplices


tritot = triu
tritot = np.append(tritot,XXu.shape[0]+trid,axis=0)
tritot = np.append(tritot,XXu.shape[0]+XXd.shape[0]+tril,axis=0)
tritot = np.append(tritot,XXu.shape[0]+XXd.shape[0]+XXl.shape[0]+trir,axis=0)

XXtot = np.append(XXu,XXd,axis=0)
XXtot = np.append(XXtot,XXl,axis=0)
XXtot = np.append(XXtot,XXr,axis=0)

print('XXtot shape = ',XXtot.shape)

ZZtotx = z1(XXtot)
ZZtoty = z2(XXtot)
ZZtotz = z3(XXtot)

ZZtot = np.zeros(shape=(ZZtotx.shape[0],3))
ZZtot[:,0]=ZZtotx
ZZtot[:,1]=ZZtoty
ZZtot[:,2]=ZZtotz

n=800
xcross = np.linspace(-(R+2),(R+2),n)
ycross = np.linspace(-(R+2),(R+2),n)
h = 4*(2*(R+2)/n)*np.sqrt(2)
cross = np.zeros(shape=(0,2))
xy = np.zeros(shape = (len(xcross)*len(ycross),3))
for i in range(len(xcross)):
    for j in range(len(ycross)):
        xy[i+j*len(xcross),:] = [xcross[i],ycross[j],0.]

YYcross = np.zeros(shape=xy.shape)
YYcross[:,0] = y1(xy)
YYcross[:,1] = y2(xy)
YYcross[:,2] = y3(xy)

for ij in range(YYcross.shape[0]):
    yy = YYcross[ij,:]
    if yy[0]>-1 and yy[0]<1 and yy[1]>-1 and yy[1]<1 and yy[2]>-1 and yy[2]<1:
        cross = np.append(cross,np.reshape(xy[ij,0:2],newshape=(1,2)),axis=0)


tricross = tri.Triangulation(cross[:,0],cross[:,1])
max_radius = h
triangles = tricross.triangles

# Mask off unwanted triangles.
xtri = cross[triangles,0] - np.roll(cross[triangles,0], 1, axis=1)
ytri = cross[triangles,1] - np.roll(cross[triangles,1], 1, axis=1)
maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
tricross.set_mask(maxi > max_radius)

YYtotx = y1(ZZtot)
YYtoty = y2(ZZtot)
YYtotz = y3(ZZtot)


fig = plt.figure(0)
ax = fig.add_subplot(projection='3d')
ax.plot_trisurf(
    ZZtot[:,0], ZZtot[:,1], ZZtot[:,2],
    triangles=tritot,cmap = cm.viridis,edgecolor='None',antialiased=False,linewidth=0
)
plt.axis('equal')
plt.axis('off')
plt.savefig('twistedTorus.png',transparent=True,dpi=300)
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.scatter(YYtotx, YYtoty, YYtotz)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.axis('equal')

fig = plt.figure(2)
ax = fig.add_subplot()
#ax.tripcolor(tricross, np.ones(shape=(cross.shape[0],)), shading='gouraud')
ax.tripcolor(tricross, np.cos(3*np.arctan2(cross[:,1],cross[:,0])),shading='gouraud')
plt.axis('equal')
plt.axis('off')
plt.savefig('twistedTorusSection.png',transparent=True,dpi=300)
plt.show()