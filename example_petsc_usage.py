from petsc4py import PETSc
import numpy as np
import scipy.sparse as sp
from simple_fd_disc import FD_disc
from time import time

n = 100; buf = 15; kh = 0; # slab discretization
h = 1/n;
box_geom = np.array([[0,buf*h],[0,1],[0,1]])

fd = FD_disc(box_geom,h,kh)
fd.check_disc()

A = fd.A[fd.I_C][:,fd.I_C].tocsr()
pA = PETSc.Mat().createAIJ(A.shape, csr=(A.indptr,A.indices,A.data))

b  = np.random.rand(A.shape[0],)
pb = PETSc.Vec().createWithArray(b)
px = PETSc.Vec().createWithArray(np.ones(b.shape[0]))

ksp = PETSc.KSP().create()
ksp.setOperators(pA)
ksp.setType('preonly')
ksp.setConvergenceHistory()
ksp.getPC().setType('lu')
ksp.getPC().setFactorSolverType('mumps')

tic = time()
ksp.solve(pb, px)
toc_solve1 = time() - tic

tic = time()
ksp.solve(pb, px)
toc_solve2 = time() - tic

residual = pA * px - pb
print("\t The solve time with PETSc is: %5.2f,%5.2f"%(toc_solve1,toc_solve2))
print("\t The relative residual Ax=b is %5.2e"% (residual.norm() / pb.norm()))
