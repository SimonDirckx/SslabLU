import numpy as np

import solver.spectralmultidomain.hps as hps

from hps.pdo               import PDO2d,PDO3d,const
from hps.geom              import BoxGeometry
from slab_subdomain        import SlabSubdomain

from time import time
from matplotlib import pyplot as plt

import argparse
import pickle

parser = argparse.ArgumentParser(description="Slab rank behavior.")
parser.add_argument('--H', type=float, required=True, help='Slab width is H')
parser.add_argument('--p', type=int, required=True, help='Discretization order p')
parser.add_argument('--ainv',type=int,required=True)
parser.add_argument('--ndim', type=int, required=True, help='Number of dimensions ndim.')
parser.add_argument('--kh', type=float, default=0,required=False, help='Wavenumber kh.')
parser.add_argument('--pickle_loc', type=str, required=False, help='Path to pickle file')

args = parser.parse_args()

H    = args.H
p    = args.p
a    = 1.0/args.ainv
kh   = args.kh

ndim = args.ndim
if (ndim == 2):

	pdo         = PDO2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
	box         = np.array([[0,0],[2.0*H,1.0]])
	geom        = BoxGeometry(box)
else:
	pdo         = PDO3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2))
	box         = np.array([[0,0,0],[2.0*H,1.0,1.0]])
	geom        = BoxGeometry(box)

print(box)

# setting p = 2 will use FD discretization
slab      = SlabSubdomain(pdo,geom,a,p)

# if you are using a constant coefficient PDE, you can verify discretization
relerr    = slab.solver.verify_discretization(kh)

print ("\t kh = %5.2f, relerr= %5.2e" % (kh,relerr))

###########################################################################################

fig = plt.figure()
XX  = slab.solver.XX

if (XX.shape[0] < 1e5 and args.pickle_loc is None):

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

if (slab.Jl.shape[0] < 1e3):
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

if (slab.Jl.shape[0] < 1e3):
	# here I am printing information about the condition number of the S sub-matrix
	nL, nC, nR = slab.Jl.shape[0], slab.Jc.shape[0], slab.Jr.shape[0]

	Il, Ic, Ir = np.eye(nL), np.eye(nC), np.eye(nR)

	Scl = slab.Scl(Il)  # Should be (nC, nL)
	Scr = slab.Scr(Ir)  # Should be (nC, nR)

	Ssubmat = np.block([
	    [Il,           np.zeros((nL, nC)), np.zeros((nL, nR))],
	    [Scl,          Ic,                 Scr],
	    [np.zeros((nR, nL)), np.zeros((nR, nC)), Ir]
	])
	print("\t Condition number of Ssubmatrix %5.2f" % np.linalg.cond(Ssubmat))

# Example: partition the identity matrix into blocks of size 10
n = slab.Jl.shape[0]
block_size = 10

# Initialize the final result
Scl_blocks = []

# Loop over chunks
for i in range(0, n, block_size):
    # Take a block of the identity matrix
    I_block = np.eye(n)[:, i:i+block_size]
    
    # Apply slab.Scl to the block
    Scl_block = slab.Scl(I_block)
    
    # Store the result
    Scl_blocks.append(Scl_block)

# Concatenate results along the second axis (columns)
Scl = np.hstack(Scl_blocks)

s   = np.linalg.svd(Scl,compute_uv=False)

if (args.pickle_loc is None):
	plt.semilogy(s)
	plt.show()
else:
	my_data = {'H':args.H,'a':a,'p':args.p,'relerr':relerr,\
	'kh':kh, 'svd_Scl':s}
	with open(args.pickle_loc, 'wb') as f:
	    pickle.dump(my_data, f)

	print(f'Data saved to {args.pickle_loc}')