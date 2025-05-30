import numpy as np
import matplotlib.pyplot as plt
import hps.hps_multidomain as HPS
import hps.geom as hpsGeom
import pdo.pdo as pdo
import solver.HPSInterp3D as HPSInterp3D
import time
import tensorly.decomposition as decomp
import tensorly as tl
import tensorly.tenalg as tenalg



# set-up of functions

def f(xpts,ypts,zpts):
    X,Y,Z=np.meshgrid(xpts,ypts,zpts,indexing='ij')
    return np.cos(2.1*np.pi*Y)+np.sin(2.1*np.pi*X)*Z

def fvec(xx):
    return np.cos(2.1*np.pi*xx[:,1])+np.sin(2.1*np.pi*xx[:,0])*xx[:,2]


#################################
#   test for single box
#################################

p=20
xpts = np.cos(np.arange(p+2) * np.pi / (p + 1))
ypts = np.cos(np.arange(p+2) * np.pi / (p + 1))
zpts = np.cos(np.arange(p+2) * np.pi / (p + 1))

xpts = xpts[::-1]
ypts = ypts[::-1]
zpts = zpts[::-1]


res = 40
pad = 0.
xpts0=np.linspace(xpts[0]+pad,xpts[-1]-pad,res)
ypts0=np.linspace(ypts[0]+pad,ypts[-1]-pad,res)
zpts0=np.linspace(zpts[0]+pad,zpts[-1]-pad,res)

XY = np.zeros(shape = (res**3,3))
XYZ = np.zeros(shape = ((p+2)**3,3))
E=np.ones(shape=(res,))
Ep=np.ones(shape=(p+2,))
XY[:,0] = np.kron(np.kron(xpts0,E),E)
XY[:,1] = np.kron(np.kron(E,ypts0),E)
XY[:,2] = np.kron(np.kron(E,E),zpts0)

XYZ[:,0] = np.kron(np.kron(xpts,Ep),Ep)
XYZ[:,1] = np.kron(np.kron(Ep,ypts),Ep)
XYZ[:,2] = np.kron(np.kron(Ep,Ep),zpts)
F = fvec(XYZ)
F_exact = fvec(XY)
F_approx = HPSInterp3D.chebInterpFromSamples3D_XX(XYZ,p,F,XY)
print("sizes = ",F_exact.shape,"//",F_approx.shape)
errInf = np.linalg.norm(F_exact-F_approx,ord=np.inf)
print("errInf = ",errInf)


#########################################
# HPS TEST
#########################################
def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c33(p):
    return np.ones(shape=(p.shape[0],))

Lapl=pdo.PDO3d(c11,c22,c33)
geom = hpsGeom.BoxGeometry(np.array([[0.,0.,0.],[.5,1.,1.]]))
p=10
a=.125
disc = HPS.HPSMultidomain(Lapl, geom, a, p)
tic = time.time()
res = 31
x_eval = np.linspace(disc._box_geom[0][0],disc._box_geom[1][0],res)
y_eval = np.linspace(disc._box_geom[0][1],disc._box_geom[1][1],res)
z_eval = np.linspace(disc._box_geom[0][0],disc._box_geom[1][0],res)

#eval at z=.6

XY = np.zeros(shape=(res*res,3))
XY[:,0] = np.kron(x_eval,np.ones(shape=y_eval.shape))
XY[:,1] = np.kron(np.ones(shape=x_eval.shape),y_eval)
XY[:,2] = .6*np.ones(shape = (res*res,))

vals = fvec(disc._XXfull)
F_approx,XYlist = HPSInterp3D.interpHPS(disc,vals,XY)
F_exact = np.zeros(shape=(0,1))
ndofs = (p+2)*(p+2)*(p+2)
for i in range(len(XYlist)):
    F_exact= np.append(F_exact,fvec(XYlist[i]))
errInf = np.linalg.norm(F_exact-F_approx,ord=np.inf)
print(errInf)