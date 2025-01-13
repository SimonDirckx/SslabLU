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

# set-up global geometry
start = time.time()
Om=stdGeom.unitSquare()
kapp = 20.5
#set up pde
def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return kapp*kapp*np.ones(shape=(p.shape[0],))
Lapl=pdo.PDO2d(c11,c22,None,None,None,c)



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

N=80
H=1./(N+1)
skel = skelTon.standardBoxSkeleton(Om,N)
ny = 80
nx = 5#int(np.round(ny/(N+1)))
ord=[nx,ny]
hx = H/(ord[0]-1)
overlapping = True
if overlapping:
    ord[0]=2*ord[0]-1 #for fair comparison
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
        return np.sin(kapp*xy[0])
    else:
        return np.sin(kapp*xy[:,0])

#default dense assembler
assembler = mA.denseMatAssembler()
assemblerList = [assembler for slab in slabList]
MultiSlab = MS.multiSlab(slabList,assemblerList)
MultiSlab.constructMats()
rhs = MultiSlab.RHS(f)

u=np.zeros(shape=(MultiSlab.N,))
n0=0
hy=1./(ord[1]-1)
step=(ord[1]-2)

for i in range(N):
    slabi:MS.Slab=slabList[i]
    if overlapping:
        J=slabi.Ii
    else:
        J=slabi.Ib
    u[range(n0,n0+step)] = slabi.eval_global_func( f,[J[i] for i in slabi.targetIdxs[0]] )
    n0+=step

Linop       = MultiSlab.getLinOp()
gInfo = gmres_info()
if petsc_imported == True:
    uhat,info   = gmres(Linop,rhs,rtol=1e-5,callback=gInfo,maxiter=25000,restart=200)
else:
    uhat,info   = gmres(Linop,rhs,tol=1e-5,callback=gInfo,maxiter=25000,restart=200)
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

########################
#   recover global sol
########################
'''
# test how to get interior solution from interface solution
XXList = np.zeros(shape=(0,2))
u=np.zeros(shape=(0,))
for i in range(len(slabList)):
    slab    = slabList[i]    
    XXi     = slab.solverWrap.XXi
    XXb     = slab.solverWrap.XXb
    XX      = slab.solverWrap.solver.XX
    b       = np.zeros(shape=(XXb.shape[0],))
    for Idx,IdxG in zip(slab.sourceIdxs,slab.globSourceIdxs):
        b[Idx] = uhat[IdxG]
    b[slab.idxsGB] = f(slab.geom.l2g(slab.solverWrap.XXb[slab.idxsGB,:]))
    u0      = -slab.solverWrap.solver.solver_ii@(slab.solverWrap.solver.Aib@b)
    Ji      = slab.solverWrap.solver.Ji
    Jb      = slab.solverWrap.solver.Jb
    utest   = np.zeros(shape=(slab.solverWrap.solver.XX.shape[0],))
    utest[Ji] = u0
    utest[Jb] = b
    if i==0:
        u00 = utest
        XX0 = XX
    else:
        u00 = np.array([utest[i] for i in range(len(utest)) if XX[i,0]>0])
        XX0 = np.array([XX[i,:] for i in range(XX.shape[0]) if XX[i,0]>0])
    u       = np.append(u,u00)
    XXList  = np.append(XXList,slab.geom.l2g(XX0),axis=0)


triang = tri.Triangulation(XXList[:,0], XXList[:,1])

plt.figure(1)
plt.tricontourf(triang, u,100)
plt.colorbar()
plt.axis('equal')
plt.show()
'''
