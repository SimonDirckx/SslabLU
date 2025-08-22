import numpy as np

import solver.spectralmultidomain.hps as hps
from   hps.sparse_utils import SparseSolver
from   scipy.sparse     import block_diag

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
parser.add_argument('--ntiles_unit',type=int,required=True)
parser.add_argument('--ndim', type=int, required=True, help='Number of dimensions ndim.')
parser.add_argument('--kh', type=float, default=0,required=False, help='Wavenumber kh.')
parser.add_argument('--pickle_loc', type=str, required=False, help='Path to pickle file')

args = parser.parse_args()

H    = args.H
p    = args.p
a    = 1.0/(2 * args.ntiles_unit)
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

# Get sparse sytem that arises from HPS discretization (only on patch interfaces)

Aii       = slab.solver.Aii
tic       = time()
Aii_inv   = SparseSolver(Aii)
toc_f1    = time() - tic

Omega     = np.random.rand(Aii.shape[0],100)
tic       = time()
Aii_inv.solve_op.matmat(Omega)
toc_ts1   = time() - tic

# Get number of double slabs
nd_slabs  = int(np.floor((1/H)))

print("One         double slab  ---- factor time %5.2e, solve time (100rhs) %5.2e" \
	% (toc_f1,toc_ts1))
print("Serial   %2.0d double slabs ---- factor time %5.2e, solve time (100rhs) %5.2e" \
	% (nd_slabs,nd_slabs*toc_f1,nd_slabs*toc_ts1))

# Create a block diagonal system of double slabs
tuple_of_copies = tuple(Aii.copy() for _ in range(nd_slabs))
block_diag_Aii  = block_diag(tuple_of_copies,format='csc')

tic             = time()
blocked_Aii_inv = SparseSolver(block_diag_Aii)
toc_fp          = time() - tic

Omega           = np.random.rand(block_diag_Aii.shape[0],100)
tic             = time()
blocked_Aii_inv.solve_op.matmat(Omega)
toc_tsp         = time() - tic

print("Parallel %2.0d double slabs ---- factor time %5.2e, solve time (100rhs) %5.2e" \
	% (nd_slabs,toc_fp,toc_tsp))

print("\nParallel speedup: %5.2e, %5.2e" % (nd_slabs*toc_f1/toc_fp,nd_slabs*toc_ts1/toc_tsp))

if (args.pickle_loc is None):
	print("Data not saved to pickle.")
else:
	my_data = {'H':args.H,'a':a,'p':args.p,'relerr':relerr,\
	'kh':kh}
	with open(args.pickle_loc, 'wb') as f:
	    pickle.dump(my_data, f)

	print(f'Data saved to {args.pickle_loc}')