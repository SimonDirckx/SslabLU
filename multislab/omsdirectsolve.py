import numpy as np
import solver.solver as solverWrap
from solver.solver import stMap

import numpy as np
import solver.solver as solverWrap
from scipy.sparse.linalg   import LinearOperator
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
    B = [I] # Set initial B_i to identity matrix
    C = [S_rk_list[_][-1] for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)

    for i in range(1, n):
        A_i = S_rk_list[i][0] @ B[-1]
        B_i = I - A_i @ (C[i-1] @ I)

        A.append(A_i)
        B.append(B_i)

    return A, B, C

def block_tridiagonal_solve(OMS, A, B, C, rhs):
    """
    Given precomputed factors for T and a RHS d, this solves Tx = d
    Note that d can be a matrix (i.e. this can handle multiple RHS)
    """
    n       = len(A)
    indices = OMS.glob_target_dofs

    d = rhs.copy()

    #
    # For i = 1 to n, we have d_i = d_i - A_i d_i-1
    #
    for i in range(1, n):
        d[indices[i]] = d[indices[i]] - A[i-1] @ d[indices[i-1]]

    #
    # Then for i = n-1 to 0, we have x_n = B_n \ d_n,   x_i = B_i \ (d_i - C_i x_i+1)
    #
    x             = np.zeros(d.shape)
    x[indices[n]] = np.linalg.solve(B[-1], d[indices[n]])

    for i in range(n-1, -1, -1):
        x[indices[i]] = np.linalg.solve(B[i], d[indices[i]] - C[i] @ x[indices[i+1]])

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
    # Write our matrix as T + UV', where T is block-tridiagonal and UV' is the rank-2m modificatioon from the conerner (E and F)
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
    Z = block_tridiagonal_solve(OMS, A, B, C, U)

    smw_block = Z @ np.linalg.solve(np.eye(2*m, dtype=E.dtype) + V.T @ Z, V.T)

    return (A, B, C), smw_block

def block_cyclic_tridiagonal_solve(OMS, T, smw_block, d):
    """
    Given a block cyclic tridiagonal system already factorized into T = (A, B, C) and the Sherman-Morrison-Woodbury correction smw_block,
    this produces x using

    y = T \ d, x = y - smw_block @ y

    Note that d can be a matrix (i.e. this can handle multiple RHS)
    """
    A, B, C = T

    y = block_tridiagonal_solve(OMS, A, B, C, d)
    x = y - smw_block @ y

    return x