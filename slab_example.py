import numpy as np

from hps.pdo               import PDO2d,PDO3d,const
from hps.geom              import BoxGeometry
from slab_subdomain        import SlabSubdomain

from time import time
from matplotlib import pyplot as plt

a = 1/8 # mesh parameter for hps discretization
p = 10  # polynomial order for hps
kh = 8  # kh parameter for PDO

pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
box         = np.array([[0,0],[0.75,1.0]])
geom        = BoxGeometry(box)

# setting p = 2 will use FD discretization
# this code can also handle 3D discretizations, using a 3D geometry
# see usage example in hps package
slab      = SlabSubdomain(pdo,geom,a,p)

# if you are using a constant coefficient PDE, you can verify discretization
relerr    = slab.solver.verify_discretization(kh)

print ("\t kh = %5.2f, relerr= %5.2e" % (kh,relerr))

###########################################################################################

fig = plt.figure()
ax  = fig.add_subplot()

XX  = slab.solver.XX

ax.scatter(XX[slab.Jx,0],XX[slab.Jx,1],label='Jx')
ax.scatter(XX[slab.Ji,0],XX[slab.Ji,1],label='Ji')
ax.scatter(XX[slab.Jl,0],XX[slab.Jl,1],label='Jl')
ax.scatter(XX[slab.Jr,0],XX[slab.Jr,1],label='Jr')

ax.set_aspect('equal','box')

plt.legend()
plt.show()

###########################################################################################

#### in order to assemble the T system, there are properties
#### T_LL, T_LR, T_RL, T_RR which return linear operators

# here I am sampling its action
rank  = 10

Omega = np.random.rand(slab.Jr.shape[0],rank) 
Y_LR  = slab.T_LR(Omega)

Q     = np.linalg.qr(Y_LR)[0]
B     = slab.T_LR.rmatmat(Q).T

# as expected, there is slow decay in the spectrum of T_LR

T_LR_full =  slab.T_LR(np.eye(slab.Jr.shape[0]))
relerr = np.linalg.norm(Q @ B - T_LR_full,ord=2) / np.linalg.norm(T_LR_full,ord=2)

print("\t Low rank approximation of T_LR of rank=%d has relerr=%5.2e" % (rank,relerr))