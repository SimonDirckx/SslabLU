import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
from solver.pde_solver import AbstractPDESolver
import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
import geometry.slabGeometry as slabGeom
import itertools
import scipy.linalg as splinalg
from scipy.sparse.linalg import gmres
import time
import matplotlib.tri as tri
import clenshawCurtis
try:
	from petsc4py import PETSc
	petsc_imported = True
except:
	petsc_imported = False

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

Om=stdGeom.unitSquare()
kapp = 5.1
#set up pde
def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return -kapp*kapp*np.ones(shape=(p.shape[0],))
Lapl=pdo.PDO2d(c11,c22)#,None,None,None,c)


N=3
H=1./(N+1)


def l2g(p):
     return p

slabOm = slabGeom.boxSlab(l2g,Om.bnds,Om)

ord=[18,18]
k = 3
aa = 2**k
N=aa//2-1
NoB = 2**(k-1)
H = 1/(N+1)
print("N = ",N)
print("H = ",H)
print("NoB = ",NoB)


#####
# form 1D HPS pts
#####
xpts_orig = np.cos(np.pi*np.linspace(0,ord[0]-1,ord[0])/(ord[0]-1))
xpts_orig=xpts_orig[::-1]
xpts = .5*(1+xpts_orig)
nx=len(xpts)

NHPS = NoB*(nx-1)+1
HPSpts = np.zeros(shape=(NHPS,))
scXpts = xpts/(1.*NoB)
for i in range(NoB):
     HPSpts[i*(nx-1):(i+1)*(nx-1)]=scXpts[0:nx-1]+(1.*i)/(1.*NoB)
HPSpts[NHPS-1]=1     

wvec=np.zeros(shape=(NHPS,))
for i in range(NHPS-1):
    i_orig = i%(nx-1)
    xi = xpts_orig[i_orig] 
    wvec[i] = np.sqrt(1-xi*xi)
wvec[NHPS-1]=0.
C=(2*(nx-1)/np.pi)*NoB
wvecMod = wvec/C
wvecMod[wvec==0.]=1.

def tensWeight(w,x,y):
    i = np.argwhere(np.abs(HPSpts-x)<1e-14)[0][0]
    j = np.argwhere(np.abs(HPSpts-y)<1e-14)[0][0]
    return (w[i]*w[j])
x=HPSpts[2]
y=HPSpts[5]
print("w = ",tensWeight(wvecMod,x,y))
print("type(w) = ",type(tensWeight(wvecMod,x,y)))


a=1./aa
opts = solverWrap.solverOptions('hps',ord,a)
solver = solverWrap.solverWrapper(opts)
solver.construct(slabOm,Lapl)
Xi = solver.XXi
Xb = solver.XXb
n = solver.solver_ii.shape[0]
E=np.identity(n)
Aii = solver.solver_ii@E
Ii = solver.Ii
Is = []

for i in range(len(Ii)):
     x= Xi[i,0]
     xH = x/H
     xHr = (int)(np.round(xH))
     if np.abs(xH-xHr)<1e-10:
          Is+=[i]

[e,V] = np.linalg.eig(Aii)
imax = np.argmax(np.abs(e))
emax = e[imax]
vmax = V[Is,imax]
Xs = Xi[Is,:]
print("len(Is)//N = ",len(Is)//N)
m = len(Is)//N-1
Isub = []
for j in range(len(Is)):
    x=Xs[j,0]
    if np.abs(x-H)<1e-15:
         Isub+=[j]
W=np.zeros(shape=(len(Ii),len(Ii)))
for i in range(len(Ii)):
    [x,y] = Xi[i,:]
    w=tensWeight(wvecMod,x,y)
    W[i,i]=w

g = np.array([Xi[i,0]*Xi[i,1] for i in range(len(Ii))])
I0 = 28*(H*H)/3.
Ig = g.T@W@g
print("\|[g]\|^2 = ",Ig)
print("I0 = ",I0)
print("err = ",np.abs(Ig-I0)/np.abs(I0))
print("C = ",C)
L = np.sqrt(W)
AW = np.linalg.inv(L)@(Aii@L)
[e,V] = np.linalg.eig(AW)
imax = np.argmax(np.abs(e))
emax = e[imax]
vmax = V[:,imax]
vcheck = .1*np.array([np.sin(np.pi*Xi[i,0])*np.sin(np.pi*Xi[i,1]) for i in range(len(Ii))])
#plt.figure(1)
#plt.scatter(Xi[:,0],Xi[:,1],label='Xi')
#plt.scatter(Xb[:,0],Xb[:,1],label='Xb')
#plt.legend()
#plt.axis('equal')
#plt.figure(2)
#plt.scatter(Xi[:,0],Xi[:,1],label='Xi')
#plt.scatter(Xi[Is,0],Xi[Is,1],label='slabs')
#plt.legend()
#plt.axis('equal')
plt.figure(3)
plt.plot(vmax)
plt.plot(vcheck)
plt.show()
