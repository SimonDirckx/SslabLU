import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
import itertools
import scipy.linalg as splinalg
from scipy.sparse.linalg import gmres
import time


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

# set-up global geometry
start = time.time()
Om=stdGeom.unitSquare()

#set up pde
def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
Lapl=pdo.PDO2d(c11,c22)

########################
#   Set up skeleton
########################

#   Explanation:
#   this implementation is meant to allow for the general case of:
#           - unordered interfaces (e.g. hierarchical domain splitting)
#           - nonuniform interface discretizations (e.g. disc. galerkin)
#   (non-uniform here means that the discretization varies from one slab to the next)
#
#   in these cases, the interface connectivity ('skel.C')
#   and the global Idxs ('skel.globIdxs') are deferred to seperate methods.
#   The uniform, standardly ordered case is provided below

# set-up constants
# N             : number of interfaces in skeleton
# [ordx,ordy]   : stencil order in x-and-y direction

N=20
H=1/(N+1)
skel = skelTon.standardBoxSkeleton(Om,N)
ny = 150
nx = int(np.round(ny/(N+1)))
ord=[nx,ny]
hx = H/(ord[0]-1)
overlapping = True
if overlapping:
    ord[0]=2*ord[0]-1 #for fair conparison
    hx=2*H/(ord[0]-1)

# solver wrapper
opts = solverWrap.solverOptions('stencil',ord)

# a skeleton is a collection of interfaces each carrying their own indices
# referred to as 'global idxs'
# uniformGlobalIdxs is a special method that assumes each will be discretized
# in the same way
# a slablist is a list of overlapping/non-overlapping slabs
skel.setGlobalIdxs(skelTon.computeUniformGlobalIdxs(skel,opts))
slabList = skelTon.buildSlabs(skel,Lapl,opts,overlapping)



########################
#   Test assembly
########################

# known solution
def f(xy):
    if xy.ndim==1:
        return np.sin(np.pi*xy[1])*np.sinh(np.pi*xy[0])
    else:
        return np.multiply(np.sin(np.pi*xy[:,1]),np.sinh(np.pi*xy[:,0]))

#default dense assembler
assembler = mA.denseMatAssembler()
assemblerList = [assembler for slab in slabList]
MultiSlab = MS.multiSlab(slabList,assemblerList)
MultiSlab.constructMats()
rhs = MultiSlab.RHS(f)

u=np.zeros(shape=(MultiSlab.N,))
n0=0
hy=1./(ord[1]-1)
step=ord[1]-2

for i in range(N):
    slabi:MS.Slab=slabList[i]
    if overlapping:
        J=slabi.Ji
    else:
        J=slabi.Jb
    u[range(n0,n0+step)] = slabi.eval_global_func( f,[J[i] for i in slabi.targetIdxs[0]] )
    n0+=step


Linop       = MultiSlab.getLinOp()
gInfo = gmres_info()
uhat,info   = gmres(Linop,rhs,callback=gInfo)
res = MultiSlab.apply(uhat)-rhs
stop = time.time()
print("=============SUMMARY==============")
print("H            = ",'%10.3E'%H)
print("(hx//hy)     = ("'%10.3E'%hx,"//",'%10.3E'%hy,")")
print("L2 rel. res  = ", np.linalg.norm(res)/np.linalg.norm(rhs))
print("L2 rel. err  = ", np.linalg.norm(u-uhat)/np.linalg.norm(u))
print("GMRES iters  = ", gInfo.niter)
print("elapsed time = ",stop-start)
print("==================================")
plt.figure(1)
plt.plot(uhat)
plt.plot(u)
plt.legend(['uhat','u'])
plt.figure(2)
plt.semilogy(gInfo.resList)
plt.show()
