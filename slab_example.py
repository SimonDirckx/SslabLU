import numpy as np
import spectralmultidomain.hps as hps
from spectralmultidomain.hps.pdo               import PDO2d,PDO3d,const
from spectralmultidomain.hps.geom              import BoxGeometry
from slab_subdomain        import SlabSubdomain

from time import time
from matplotlib import pyplot as plt

a = 1/8  # mesh parameter for hps discretization
p = 10   # polynomial order for hps
kh = 8   # kh parameter for PDO

ndim = 2 # 3d capable -- I recommend p < 8 in 3d.

if (ndim == 2):

	pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
	box         = np.array([[0,0],[2,1.0]])
	geom        = BoxGeometry(box)
else:
	pdo         = PDO3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2))
	box         = np.array([[0,0,0],[2,1.0,1.0]])
	geom        = BoxGeometry(box)

# setting p = 2 will use FD discretization
slab      = SlabSubdomain(pdo,geom,a,p)

# if you are using a constant coefficient PDE, you can verify discretization
relerr    = slab.solver.verify_discretization(kh)

print ("\t kh = %5.2f, relerr= %5.2e" % (kh,relerr))

###########################################################################################

fig = plt.figure()
XX  = slab.solver.XX

if (slab.solver.ndim == 2):
	ax  = fig.add_subplot()
	ax.scatter(XX[slab.Jx,0],XX[slab.Jx,1],label='Jx')
	ax.scatter(XX[slab.Ji,0],XX[slab.Ji,1],label='Ji')
	ax.scatter(XX[slab.Jl,0],XX[slab.Jl,1],label='Jl')
	ax.scatter(XX[slab.Jr,0],XX[slab.Jr,1],label='Jr')
	ax.scatter(XX[slab.Jc,0],XX[slab.Jc,1],label='Jc')

	ax.set_aspect('equal','box')
else:
	ax  = fig.add_subplot(projection='3d')
	ax.scatter(XX[slab.Jx,0],XX[slab.Jx,1],XX[slab.Jx,2],label='Jx')
	ax.scatter(XX[slab.Ji,0],XX[slab.Ji,1],XX[slab.Ji,2],label='Ji')
	ax.scatter(XX[slab.Jl,0],XX[slab.Jl,1],XX[slab.Jl,2],label='Jl')
	ax.scatter(XX[slab.Jr,0],XX[slab.Jr,1],XX[slab.Jr,2],label='Jr')
	ax.scatter(XX[slab.Jc,0],XX[slab.Jc,1],XX[slab.Jc,2],label='Jc')

	ax.set_box_aspect([1,1,1])

plt.legend()
plt.show()

###########################################################################################

def randQB(Aop,rank):

	Omega = np.random.rand(Aop.shape[1],rank) 
	Y     = Aop(Omega)

	Q     = np.linalg.qr(Y)[0]
	B     = Aop.rmatmat(Q).T
	return Q,B

###########################################################################################

#### in order to assemble the T system, there are properties
#### Tll, Tlr, Trl, Trr which return linear operators

# here I am sampling the action of Tlr and confirming that it is low rank
rank  = 10
Q,B   = randQB(slab.Tlr,rank)

Ifull  = np.eye(slab.Jl.shape[0])

Tlr    =  slab.Tlr(Ifull)
relerr = np.linalg.norm(Q @ B - Tlr,ord=2) / np.linalg.norm(Tlr,ord=2)
print("\t Low rank approximation of Tlr of rank=%d has relerr=%5.2e" % (rank,relerr))


# here I am printing information about the condition number of the T sub-matrix
Ifull  = np.eye(slab.Jl.shape[0])
Tll    = slab.Tll(Ifull)
Tlr    = slab.Tlr(Ifull)
Trl    = slab.Trl(Ifull)
Trr    = slab.Trr(Ifull)

Tsubmat= np.block([[Tll,Tlr],[Trl,Trr]])
print("\t Condition number of Tsubmatrix %5.2f" % np.linalg.cond(Tsubmat))

print("")
###########################################################################################

#### in order to assemble the S system, there are properties
#### Scl, Scr which return linear operators
#### NOTE that these operators "include" the negative sign

# here I am sampling the action of Scr and confirming that it is low rank
rank  = 10
Q,B   = randQB(slab.Scr,rank)

Scr_full =  slab.Scr(np.eye(slab.Jr.shape[0]))
relerr   = np.linalg.norm(Q @ B - Scr_full,ord=2) / np.linalg.norm(Scr_full,ord=2)
print("\t Low rank approximation of Scr of rank=%d has relerr=%5.2e" % (rank,relerr))

# here I am printing information about the condition number of the S sub-matrix
Ifull   = np.eye(slab.Jl.shape[0])
Scl     = slab.Scl(Ifull)
Scr     = slab.Scr(Ifull)

Zfull   = np.zeros(Ifull.shape)

Ssubmat = np.block(([[Ifull,Zfull,Zfull],[Scl,Ifull,Scr],[Zfull,Zfull,Ifull]]))
print("\t Condition number of Ssubmatrix %5.2f" % np.linalg.cond(Ssubmat))