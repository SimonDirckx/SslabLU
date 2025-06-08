import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
from solver.pde_solver import AbstractPDESolver
import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
import itertools
import scipy.linalg as splinalg
from scipy.sparse.linalg import gmres
import time
import matplotlib.tri as tri
from matAssembly.HBS.simpleoctree.simpletree import BalancedTree as tree
import matAssembly.HBS.HBSTree as HBS
try:
	from petsc4py import PETSc
	petsc_imported = True
except:
	petsc_imported = False
print("PETSC = ",petsc_imported)
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
start = time.time()
Om=stdGeom.unitSquare()
kapp = 5.12
#set up pde
def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return -kapp*kapp*np.ones(shape=(p.shape[0],))
Lapl=pdo.PDO2d(c11,c22,None,None,None,c)


N=3
H=1./(N+1)
print('ceil = ',np.ceil(2./H))
print('floor = ',np.floor(2./H))

skel = skelTon.standardBoxSkeleton(Om,N)
ord=[18,18]
a=H/4.
print("nyz = ",(int)(np.round((ord[0]-2)*(.5/a))))
overlapping = True
opts = solverWrap.solverOptions('hps',ord,a)
skel.setGlobalIdxs(skelTon.computeUniformGlobalIdxs(skel,opts))
slabList = skelTon.buildSlabs(skel,Lapl,opts,overlapping)
assembler = mA.denseMatAssembler()
assemblerList = [assembler for slab in slabList]
MultiSlab = MS.multiSlab(slabList,assemblerList)
MultiSlab.constructMats()

slab0:MS.Slab = slabList[1]
XXl = slab0.solverWrap.XXb[slab0.sourceIdxs[0],:]
XXr = slab0.solverWrap.XXb[slab0.sourceIdxs[1],:]
YY = slab0.solverWrap.XXi[slab0.targetIdxs[0],:]

XXiGlob = slab0.geom.l2g(slab0.solverWrap.XXi)
XXbGlob = slab0.geom.l2g(slab0.solverWrap.XXb)
t = tree(XXl,8)
L = t.nlevels
print("levels = ",L)
S = slab0.maps[0].A
print("S shape = ",S.shape)
rk = 4
s  = 5*rk
Om = np.random.standard_normal(size=(S.shape[0],s))
Psi = np.random.standard_normal(size=(S.shape[1],s))
print('Om//S : ',Om.shape,"//",S.shape)
Y = S@Om
Z = S.T@Psi
start = time.time()
SH = HBS.random_compression_HBS(t,Om,Psi,Y,Z,rk,s)
stop = time.time()
print('timing HBS = ',stop-start)
u = np.random.standard_normal(size=(S.shape[0],10))
u = u/np.linalg.norm(u)
Su = S.T@u
SHu = HBS.apply_HBS(SH,u,True)
print("Su err. = ",np.linalg.norm(Su-SHu))
print("Su relerr. = ",np.linalg.norm(Su-SHu)/np.linalg.norm(Su))
print("Data : ",SH.total_bytes())
N=S.shape[0]
n = 8

dataPred = rk*rk*(2**(L+2)-12)+2*rk*N
dataPred+= 8*N
print("Predicted data : ",dataPred)
print("Data/N : ",SH.total_bytes()/N)
print("Compression factor : ",SH.total_bytes()/(S.data.nbytes))


plt.figure(1)
plt.scatter(XXl[:,0],XXl[:,1])
plt.scatter(XXr[:,0],XXr[:,1])
plt.scatter(YY[:,0],YY[:,1])
plt.legend(['XXl','XXr','YY'])


plt.figure(2)
plt.scatter(XXiGlob[:,0],XXiGlob[:,1])
plt.scatter(XXbGlob[:,0],XXbGlob[:,1])
plt.legend(['i','b'])
plt.axis('equal')

plt.figure(3)
plt.plot(Su)
plt.plot(SHu)
plt.legend(['Su','SHu'])
plt.show()