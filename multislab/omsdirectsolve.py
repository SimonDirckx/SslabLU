import numpy as np
import solver.solver as solverWrap
from solver.solver import stMap

import numpy as np
import solver.solver as solverWrap
from scipy.sparse.linalg   import LinearOperator
from scipy.linalg   import lu_factor, lu_solve
from solver.solver import stMap
import time
import sys
import jax.numpy as jnp
import matplotlib.pyplot as plt
#import gc

import multislab.oms as oms

def build_block_tridiagonal_solver(S_rk_list):
    """
    
    [ B0 ] [ C0 ] [ 0  ] [ 0  ]
    [ A1 ] [ B1 ] [ C1 ] [ 0  ]
    [ 0  ] [ A2 ] [ B2 ] [ C2 ]
    [ 0  ] [ 0  ] [ A3 ] [ B3 ]

    Using linear operators corresponding to the slabs of a slab solver, we will construct a block tridiagonal direct solver

    This is based off of the Thomas algorithm, and used as a comparison point for red-black and nested dissection solvers.
    """

    #
    # For the block-tridiagonal system T. Assuming a structure like above:
    # A_i = S_rk_list[i][0]
    # B_i = Identity (Perk of the S formulation)
    # C_i = S_rk_list[i][1]
    #
    # For solving Tx = d partitioned into blocks,
    # the Thomas Algorithm applies first foward elimination for i = 1 to n:
    # A_i' = A_i (B_i-1)^-1,   B_i' = B_i - G_i C_i-1,   d_i = d_i - A_i' d_i-1
    #
    # Then backward substitution for i = n-1 to 0:
    # x_n = (B_n')^-1 d_n,   x_i = B_i' \ (d_i - C_i x_i+1)
    #
    # Parts we can precompute: A_i', B_i', C_i. We'll denote A_i' and B_i' as just A_i, B_i in the code.

    m = S_rk_list[0][0].shape[0]
    n = len(S_rk_list) - 1 # Accounts for E and F in periodic case
    I = np.eye(m, dtype=S_rk_list[0][0].dtype)

    # Thus we need three lists of block matrices: A, B, and C:
    A = []
    B = [I] # Set initial B_i to identity matrix LU factor (can specialize this to be just identity later)
    C = [S_rk_list[_][-1] for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)

    for i in range(1, n+1):
        if i==1:
            A_i = S_rk_list[i][0] @ B[-1]
        else:
            A_i = S_rk_list[i][0] @ lu_solve(B[-1], I)
            
        #B_i = I - C[i-1].rmatmat(A_i.T).T
        B_i = I - (C[i-1].T@(A_i.T)).T
        # Factorize B_i since it's always used in a solve. This means B_i will now be a tuple:
        B_i = lu_factor(B_i, overwrite_a=True)

        A.append(A_i)
        B.append(B_i)

    return A, B, C

def block_tridiagonal_solve(OMS, T, rhs):
    """
    Given precomputed factors for T and a RHS d, this solves Tx = d
    Note that d can be a matrix (i.e. this can handle multiple RHS)
    """
    A, B, C = T
    n       = len(A)
    indices = OMS.glob_target_dofs

    d = rhs.copy()

    #
    # For i = 1 to n, we have d_i = d_i - A_i d_i-1
    #
    for i in range(1, n+1):
        d[indices[i]] = d[indices[i]] - A[i-1] @ d[indices[i-1]]

    #
    # Then for i = n-1 to 0, we have x_n = B_n \ d_n,   x_i = B_i \ (d_i - C_i x_i+1)
    #
    x             = np.zeros(d.shape)
    x[indices[n]] = lu_solve(B[n], d[indices[n]])

    for i in range(n-1, 0, -1):
        x[indices[i]] = lu_solve(B[i], d[indices[i]] - C[i] @ x[indices[i+1]])

    # Since B[0] is the identity matrix, we can avoid a solve there:
    x[indices[0]] = d[indices[0]] - C[0] @ x[indices[1]]

    return x


def build_block_cyclic_tridiagonal_solver(OMS, S_rk_list, rhs_list, Ntot, nc):
    """
    
    [ B0 ] [ C0 ] [ 0  ] [ E  ]
    [ A1 ] [ B1 ] [ C1 ] [ 0  ]
    [ 0  ] [ A2 ] [ B2 ] [ C2 ]
    [ F  ] [ 0  ] [ A3 ] [ B3 ]

    Using linear operators corresponding to the slabs of a slab solver, we will construct a block cyclic tridiagonal direct solver

    This is based off of the Thomas algorithm, and used as a comparison point for red-black and nested dissection solvers.

    TODO: check if periodic or not, will determine if solver is strictly block-tridiagonal or SMW is needed
    """

    #
    # First we need to build matrices of the necessary subblocks. Start with the SMW formula:
    # Write our matrix as T + UV', where T is block-tridiagonal and UV' is the rank-2m modificatioon from the corner (E and F)
    #
    E = S_rk_list[0][0]
    F = S_rk_list[-1][1]
    m = E.shape[0] # Assuming all blocks are same size for now
    n = len(S_rk_list)

    I = np.eye(m, dtype=E.dtype)
    U = np.zeros((n*m, 2*m), dtype=E.dtype)
    V = np.zeros((n*m, 2*m), dtype=E.dtype)

    U[0:m,         0:m]   = E @ I
    U[(n-1)*m:n*m, m:2*m] = F @ I

    V[(n-1)*m:n*m, 0:m]   = I
    V[0:m,         m:2*m] = I

    A, B, C = build_block_tridiagonal_solver(S_rk_list)

    #
    # Now that we have the components of the tridiagonal solver and U,V, we can apply the SMW formula
    #
    Z = block_tridiagonal_solve(OMS, (A, B, C), U)

    smw_block = Z @ np.linalg.solve(np.eye(2*m, dtype=E.dtype) + V.T @ Z, V.T)

    return (A, B, C), smw_block

def block_cyclic_tridiagonal_solve(OMS, T, smw_block, d):
    """
    Given a block cyclic tridiagonal system already factorized into T = (A, B, C) and the Sherman-Morrison-Woodbury correction smw_block,
    this produces x using

    y = T \ d, x = y - smw_block @ y

    Note that d can be a matrix (i.e. this can handle multiple RHS)
    """

    y = block_tridiagonal_solve(OMS, T, d)
    x = y - smw_block @ y

    return x

# Construct a solver for the red-black scheme. This one is not periodic (aka not cyclical)
def build_block_RB_solver(OMS, S_rk_list, rhs_list, Ntot, nc, cyclic=False):
    """
    
    [  I   ] [ S_12 ] [  0   ] [  E?  ]
    [ S_21 ] [  I   ] [ S_23 ] [  0   ]
    [  0   ] [ S_32 ] [  I   ] [ S_34 ]
    [  F?  ] [  0   ] [ S_43 ] [  I   ]

    Using linear operators corresponding to the slabs of a slab solver, we will construct a block tridiagonal direct solver

    This is based off of the red-black algorithm. ASSUME POWER OF 2 FOR SLABCOUNT ONLY

    We need to store 4 objects:
    1. Original S_rk_list, all entries are needed for either even solves or RHS of odd solves
    2. For i odd, the factorized systems B_i' = (I - S_{i,i-1} S_{i-1,i} - S_{i,i+1} S_{i+1,i})^-1 that makes up the "main diagonal"
    3. For i odd, the factorized system  A_i' = (B_i') \ S_{i,i-1} S_{i-1,i-2}
    4. For i odd, the factorized system  C_i' = (B_i') \ S_{i,i+1} S_{i+1,i+2}
    """
    m      = S_rk_list[0][0].shape[0]
    nSlabs = len(S_rk_list)

    if not ((nSlabs & (nSlabs-1) == 0) and nSlabs != 0):
        ValueError("ERROR! Number of slabs is not a power of 2.")

    SiM = [-_[0] for _ in S_rk_list]
    SiP = [-_[-1] for _ in S_rk_list]

    if not cyclic:
        def zero_operator(m, dtype=float):
            """
            Return a SciPy LinearOperator acting on R^m → R^m that always returns zero.
            Supports matvec, rmatvec, matmat, rmatmat.
            """
            def matvec(x):
                # x is shape (m,)
                return np.zeros_like(x)

            def rmatvec(x):
                # x is shape (m,)
                return np.zeros_like(x)

            def matmat(X):
                # X is shape (m, k)
                return np.zeros_like(X)

            def rmatmat(X):
                # X is shape (m, k)
                return np.zeros_like(X)

            return LinearOperator(
                shape=(m, m),
                matvec=matvec,
                rmatvec=rmatvec,
                matmat=matmat,
                rmatmat=rmatmat,
                dtype=dtype,
            )
        SiM[0] = zero_operator
        SiP[-1] = zero_operator

    RB = build_block_RB_solver_level(m, nSlabs, SiM, SiP)

    return RB, SiM, SiP


# Construct a solver for the red-black scheme. This one is not periodic (aka not cyclical)
def build_block_RB_solver_level(m, nSlabs, SiM, SiP):
    """
    
    [  I   ] [ S_12 ] [  0   ] [  E?  ]
    [ S_21 ] [  I   ] [ S_23 ] [  0   ]
    [  0   ] [ S_32 ] [  I   ] [ S_34 ]
    [  F?  ] [  0   ] [ S_43 ] [  I   ]

    Using linear operators corresponding to the slabs of a slab solver, we will construct a block tridiagonal direct solver

    This is based off of the red-black algorithm.

    We need to store 4 objects:
    1. Original S_rk_list, all entries are needed for either even solves or RHS of odd solves
    2. For i odd, the factorized systems B_i' = (I - S_{i,i-1} S_{i-1,i} - S_{i,i+1} S_{i+1,i})^-1 that makes up the "main diagonal"
    3. For i odd, the factorized system  A_i' = (B_i') \ S_{i,i-1} S_{i-1,i-2}
    4. For i odd, the factorized system  C_i' = (B_i') \ S_{i,i+1} S_{i+1,i+2}
    """

    # First let's build 2, B_i:
    I   = np.eye(m, dtype=SiM[0].dtype)
    B_i = [I for _ in range(0, nSlabs, 2)]
    for i in range(0, nSlabs, 2):
        j = int(i/2)
        B_i[j] = B_i[j] - SiM[i] @ SiP[(i-1) % nSlabs] @ I - SiP[i] @ SiM[(i+1) % nSlabs] @ I

    # Now we factorize every B_i:
    B_i = [lu_factor(_, overwrite_a=True) for _ in B_i]

    # Next let's build 3, A_i:
    A_i = [lu_solve(B_i[int(_ / 2)], SiM[_] @ SiM[(_ - 1) % nSlabs] @ I) for _ in range(0, nSlabs, 2)]
    # And finally build 4, C_i:
    C_i = [lu_solve(B_i[int(_ / 2)], SiP[_] @ SiP[(_ + 1) % nSlabs] @ I) for _ in range(0, nSlabs, 2)]

    return (A_i, B_i, C_i)


def block_RB_solve(OMS, RB, SiM, SiP, v):
    """
    [  I   ] [ S_12 ] [  0   ] [  E?  ]
    [ S_21 ] [  I   ] [ S_23 ] [  0   ]
    [  0   ] [ S_32 ] [  I   ] [ S_34 ]
    [  F?  ] [  0   ] [ S_43 ] [  I   ]

    Solves using the Red-Black factorization. Assumes we have RB = (A_i, B_i, C_i) and v of size m * nSlabs
    """
    (A_i, B_i, C_i) = RB

    m        = A_i[0].shape[0]
    nReduced = len(B_i)
    nSlabs   = 2 * nReduced

    # Set up S:
    S = np.eye(m*nReduced, dtype=A_i[0].dtype)
    for i in range(nReduced):
        prev = (i - 1) % nReduced
        next = (i + 1) % nReduced
        S[i*m:(i+1)*m, prev*m:(prev+1)*m] = -A_i[i]
        S[i*m:(i+1)*m, next*m:(next+1)*m] = -C_i[i]
    
    S = lu_factor(S, overwrite_a=True)

    # Now build the RHS:
    vPrime = np.zeros(m*nReduced)
    for j in range(nReduced):
        i    = 2 * j
        prev = (i - 1) % nSlabs
        next = (i + 1) % nSlabs
        vPrime[j*m:(j+1)*m] = v[i*m:(i+1)*m] + SiM[i] @ v[prev*m:(prev+1)*m] + SiP[i] @ v[next*m:(next+1)*m]
        vPrime[j*m:(j+1)*m] = lu_solve(B_i[j], vPrime[j*m:(j+1)*m])

    vPrime = lu_solve(S, vPrime)
    u      = np.zeros(v.shape)

    for j in range(nReduced):
        i = 2 * j
        # We fill in the odd segments of u
        u[i*m:(i+1)*m] = vPrime[j*m:(j+1)*m]
        
        # Here we compute the even segments of u:
        next = (j + 1) % nReduced
        u[(i+1)*m:(i+2)*m] = SiM[i] @ vPrime[j*m:(j+1)*m] + v[(i+1)*m:(i+2)*m] + SiP[i] @ vPrime[next*m:(next+1)*m]

    return u