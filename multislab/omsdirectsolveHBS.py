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
import matAssembly.HBS.HBSnew as HBSnew
#import gc

import multislab.oms as oms


def Sprime_Linop(Sl,Sprime_prev,Sr,id=False):
    if id:
        def smatmat(v,transpose=False):        
            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):
                result = v_tmp-Sl@(Sr@v_tmp)
            else:
                result = v_tmp-Sr.T@(Sl.T@v_tmp)
            if (v.ndim == 1):
                result = result.flatten()
            return result

    else:
        def smatmat(v,transpose=False):        
            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):
                result = v_tmp-Sl@Sprime_prev.solve(Sr@v_tmp)
            else:
                result = v_tmp-Sr.T@Sprime_prev.solve(Sl.T@v_tmp,mode='T')
            if (v.ndim == 1):
                result = result.flatten()
            return result
    Sprime = LinearOperator(shape=(Sl.shape[0],Sr.shape[1]),\
        matvec = lambda v:smatmat(v), rmatvec = lambda v:smatmat(v,transpose=True),\
        matmat = lambda v:smatmat(v), rmatmat = lambda v:smatmat(v,transpose=True))
    return Sprime





def build_block_tridiagonal_solver(S_rk_list,tree,quad,rk):
    """
    
    [ I ] [S12] [ 0 ] [ 0 ]
    [S21] [ I ] [S23] [ 0 ]
    [ 0 ] [S32] [ I ] [S34]
    [ 0 ] [ 0 ] [S43] [ I ]

    Using linear operators corresponding to the slabs of a slab solver, we will construct a block tridiagonal direct solver

    This is based off of the Thomas algorithm, and used as a comparison point for red-black and nested dissection solvers.

    The recurrence (can be derived)
    ---------------------------------------------------------------------------
    S'_1 = I
    b'_1 = b_1
    
    and 
    
    S'_{i+1} = I-S_{i+1,i}S'_{i}\S_{i,i+1}
    b'_{i+1} = b_{i+1}-S_{i+1,i}S'_{i}\(b'_i)
    ---------------------------------------------------------------------------
    NOTE: different sign convention on S is possible, in this case, recurrence changes slightly

    """

    m = S_rk_list[0][0].shape[0]
    n = len(S_rk_list) - 1 # Accounts for E and F in periodic case
    I = np.eye(m, dtype=S_rk_list.dtype)

    # Thus we need three lists of block matrices: A, B, and C:
    Sl = [S_rk_list[_][0] for _ in range(n)]
    Sprime = [I] # Set initial Sprime
    Sr = [S_rk_list[_][-1] for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)
    for i in range(1, n+1):
        if i==1:
            Sprime_i = HBSnew.HBSMAT(Sprime_Linop(Sl[0],I,Sr[0],id=True))
            Sprime_i.construct(rk)
        else:
            Sprime[i-1].compute_ULV()
            Sprime_i = HBSnew.HBSMAT(Sprime_Linop(Sl[i-1],Sprime[i-1],Sr[i-1]),tree,quad)
            Sprime_i.construct(rk)
        Sprime.append(Sprime_i)
    return Sl, Sprime, Sr

def block_tridiagonal_solve(OMS, T, rhs):
    """
    Given precomputed factors for T and a RHS d, this solves Tx = d
    Note that d can be a matrix (i.e. this can handle multiple RHS)
    """
    Sl, Sprime, Sr = T
    n       = len(Sl)
    indices = OMS.glob_target_dofs

    d = rhs.copy()

    #
    # For i = 1 to n, we have d_i = d_i - A_i d_i-1
    #
    for i in range(1, n+1):
        d[indices[i]] = d[indices[i]] - Sl[i-1]@Sprime[i-1].solve(d[indices[i-1]])

    #
    # Then for i = n-1 to 0, we have x_n = B_n \ d_n,   x_i = B_i \ (d_i - C_i x_i+1)
    #
    x             = np.zeros(d.shape)
    x[indices[n]] = Sprime[n].solve( d[indices[n]] )

    for i in range(n-1, 0, -1):
        x[indices[i]] = Sprime[i].solve( d[indices[i]] - Sr[i] @ x[indices[i+1]] )

    # Since B[0] is the identity matrix, we can avoid a solve there:
    x[indices[0]] = d[indices[0]] - Sr[0] @ x[indices[1]]
    return x


