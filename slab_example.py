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
box         = np.array([[0,0],[0.5,1.0]])
geom        = BoxGeometry(box)

# setting p = 2 will use FD discretization
# this code can also handle 3D discretizations, using a 3D geometry
# see usage example in hps package
slab      = SlabSubdomain(pdo,geom,a,p)

# if you are using a constant coefficient PDE, you can verify discretization
relerr    = slab.solver.verify_discretization(kh)

print ("\t kh = %5.2f, relerr= %5.2e" % (kh,relerr))

fig = plt.figure()
ax  = fig.add_subplot()

XX  = slab.solver.XX
I_C = slab.solver.I_C
I_X = slab.solver.I_X

ax.scatter(XX[I_X,0],XX[I_X,1])
ax.scatter(XX[I_C,0],XX[I_C,1])
ax.set_aspect('equal','box')

plt.show()

#### in order to assemble the T system, there is a method T_XX_op
