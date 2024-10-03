
import numpy        as     np
from   time         import time
from   matplotlib   import pyplot as plt

from   src_utils.petsc_solver import PETScSolver
from   src_utils.fd_disc      import FDDiscretization,PDO3d,pdo_const

##############################################################
n = 100; buf = 15; kh = 2*np.pi; # slab discretization
h = 1/n;
box_geom = np.array([[0,buf*h],[0,1],[0,1]])

pdo_ones = lambda xx: pdo_const(xx)
pdo_kh   = lambda xx: pdo_const(xx,-kh**2)

op = PDO3d(pdo_ones,pdo_ones,pdo_ones,c=pdo_kh)
fd = FDDiscretization(box_geom,h,op,kh=kh)

# setup a solver for A_CC using petsc
A_CC = fd.A[fd.I_C][:,fd.I_C].tocsr()

tic       = time()
solver_CC = PETScSolver(A_CC) 
toc_setup = time() - tic

# this just assigns an internal pointer to the petsc solver
fd.setup_solver_CC(solver_CC.solve_op)

# check the discretization, compared to known PDE solution
tic = time()
relerr = fd.check_discretization()
toc_solve = time() - tic

print("For N=%5.2e, the slab has %d x %d x %d points for a Poisson problem" % (np.prod(fd.ns),fd.ns[0],fd.ns[1],fd.ns[2]))
print("\t (setup time, solve time)=(%5.2f,%5.2f) s, discretization relative accuracy %5.2e" % (toc_setup,toc_solve,relerr))


####################################################################
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(fd.XX[fd.I_C,0]  ,fd.XX[fd.I_C,1]  ,fd.XX[fd.I_C,2],label='I_C') # internal nodes
ax.scatter(fd.XX[fd.I_L,0]  ,fd.XX[fd.I_L,1]  ,fd.XX[fd.I_L,2],label='I_L') # left boundary
ax.scatter(fd.XX[fd.I_R,0]  ,fd.XX[fd.I_R,1]  ,fd.XX[fd.I_R,2],label='I_R') # right boundary

ax.legend()
plt.show()