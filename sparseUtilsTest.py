import scipy.sparse as sparse
from solver import sparse_utils
import numpy as np
import scipy.sparse.linalg as splinalg

# Data
nx  = 10
ny  = 10
A   = np.random.standard_normal(size=(nx,nx))
B   = np.random.standard_normal(size=(ny,ny))

# Validation Data 
Lnp = np.kron(A,B)
L   = sparse.kron(A,B)

# Lop from sparse_utils
Lop     = sparse_utils.CSRBuilder(L.shape[0],L.shape[1],nx*nx*ny*ny+1)
Lop.add_data(L)
Lop     = Lop.tocsr()

# L inv op from sparse_utils
print(Lop.shape)
sparseSolver=sparse_utils.SparseSolver(Lop)
LinvOp = sparseSolver.solve_op

# Validate
v=np.random.standard_normal(size=(Lnp.shape[0],))
w0=np.linalg.solve(Lnp,v)
w1=LinvOp@v
print(np.linalg.norm(w0-w1)/np.linalg.norm(w0))