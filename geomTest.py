import numpy as np
import geometry.slabGeometry as slab
import pdo.pdo as pdo
from solver.stencil.stencilSolver import stencilSolver
import solver.solver as solverWrap
import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
# set-up global geometry
Om=stdGeom.unitCube()

#set up pde
def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c33(p):
    return np.ones(shape=(p.shape[0],))
Lapl=pdo.PDO3d(c11,c22,c33)

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

N=3
skel = skelTon.standardBoxSkeleton(Om,N)
ord=[10,5,5]
opts = solverWrap.solverOptions('stencil',ord)
overlapping = False #toggle
if overlapping:
    ord[0]=2*ord[0]-1
globIdxs = skelTon.computeUniformGlobalIdxs(skel,opts)
skel.setGlobalIdxs(globIdxs)
slablist = skelTon.buildSlabs(skel,Lapl,opts,overlapping)

# tests
for slabi in slablist:
    print("idxs = ",slabi.globIdxs)
    print("===============================")


XXiGlob = slablist[0].geom.l2g(slablist[0].solverWrap.XXi)
XXbGlob = slablist[0].geom.l2g(slablist[0].solverWrap.XXb)
N=len(slablist)
for i in range(1,N):
    slabi=slablist[i]
    XXiGlob=np.vstack((XXiGlob,slabi.geom.l2g(slabi.solverWrap.XXi)))
    XXbGlob=np.vstack((XXbGlob,slabi.geom.l2g(slabi.solverWrap.XXb)))

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.scatter(XXiGlob[:,0],XXiGlob[:,1],XXiGlob[:,2])
ax.scatter(XXbGlob[:,0],XXbGlob[:,1],XXbGlob[:,2])
plt.show()
