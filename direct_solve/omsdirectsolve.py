import numpy as np
from scipy.sparse.linalg   import LinearOperator
from scipy.linalg   import lu_factor, lu_solve, block_diag
import time
import pickle

from abc import ABC, abstractmethod

import os
np.set_printoptions(linewidth=os.get_terminal_size().columns)
np.set_printoptions(threshold=10000)

class DirectSolver(ABC):

    def __init__(self, m, cyclic=False):
        self.m = m
        self.cyclic = cyclic

    @abstractmethod
    def factorize(self, S_rk_list, T):
        """Need to implement"""
        pass

    @abstractmethod
    def solve(self, rhs):
        """Need to implement"""
        pass



def reconstruct_from_lu_factor(Tinv):
    lu, piv = Tinv
    n = lu.shape[0]

    # Extract L and U from the compact representation
    L = np.tril(lu, k=-1) + np.eye(n)  # Lower triangular with 1s on diagonal
    U = np.triu(lu)                     # Upper triangular

    # Reconstruct the permutation matrix from pivot indices
    P = np.eye(n)
    for i, p in enumerate(piv):
        P[[i, p]] = P[[p, i]]  # Apply row swaps in order

    # T = P^T @ L @ U  (since scipy stores PA = LU, so A = P^T @ L @ U)
    T = P.T @ L @ U
    return T

class BlockTridiagonalSolver(DirectSolver):

    def factorize_helper(self, S_rk_list, T):
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
        # A_i' = A_i (B_i-1)'^-1,   B_i' = B_i - G_i C_i-1,   d_i = d_i - A_i' d_i-1
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
        B = [lu_factor(T[0])] # Set initial B_i to identity matrix LU factor (can specialize this to be just identity later)
        C = [S_rk_list[_][-1] for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)

        for i in range(1, n+1):
            if i==1:
                A_i = S_rk_list[i][0] @ lu_solve(B[-1], I)
            else:
                A_i = S_rk_list[i][0] @ lu_solve(B[-1], I)
                
            #B_i = I - C[i-1].rmatmat(A_i.T).T
            B_i = T[i] - (C[i-1].T@(A_i.T)).T
            # Factorize B_i since it's always used in a solve. This means B_i will now be a tuple:
            B_i = lu_factor(B_i, overwrite_a=True)

            A.append(A_i)
            B.append(B_i)
        
        self.A = A
        self.B = B
        self.C = C

    def solve_helper(self, rhs, glob_target_dofs=None):
        """
        Given precomputed factors for T and a RHS d, this solves Tx = d
        Note that d can be a matrix (i.e. this can handle multiple RHS)
        """
        m = self.m
        A = self.A
        B = self.B
        C = self.C
        n = len(A)
        d = rhs.copy()

        if glob_target_dofs is None:
            indices = [range(l*m, (l+1)*m) for l in range(len(B))]
        else:
            indices = glob_target_dofs

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
        # But not this time:
        x[indices[0]] = lu_solve(B[0], d[indices[0]] - C[0] @ x[indices[1]])

        return x

    def factorize(self, S_rk_list, T, glob_target_dofs=None):
        self.factorize_helper(S_rk_list, T)

        self.smw_block = None

        if self.cyclic:
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

            Z = self.solve_helper(U, glob_target_dofs)

            self.smw_block = Z @ np.linalg.solve(np.eye(2*m, dtype=E.dtype) + V.T @ Z, V.T)


    def solve(self, rhs, glob_target_dofs=None):
        x = self.solve_helper(rhs, glob_target_dofs)

        if self.cyclic:
            x = x - self.smw_block @ x

        return x

class RedBlackSolver(DirectSolver):

    # Construct a solver for the red-black scheme. This one is not periodic (aka not cyclical)
    def factorize(self, S_rk_list, T):
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
        3. For i odd, the factorized system  A_i' = (B_i') \\ S_{i,i-1} S_{i-1,i-2}
        4. For i odd, the factorized system  C_i' = (B_i') \\ S_{i,i+1} S_{i+1,i+2}
        """
        m      = S_rk_list[0][0].shape[0]
        nSlabs = len(S_rk_list)

        if not ((nSlabs & (nSlabs-1) == 0) and nSlabs != 0):
            ValueError("ERROR! Number of slabs is not a power of 2.")

        SiM = [_[0]  for _ in S_rk_list]
        SiP = [_[-1] for _ in S_rk_list]

        if not self.cyclic:
            SiM[0] = np.zeros((m,m))
            SiP[-1] = np.zeros((m,m))

        RB = [(SiM, T, [lu_factor(_) for _ in T], SiP)]

        l = nSlabs
        while l > 1:
            RB.append(self.build_block_RB_solver_level(m, l, RB[-1], cyclic=self.cyclic))
            l = int(l / 2)

        self.nSlabs = nSlabs
        self.RB = RB

    def solve(self, rhs):
        """
        [  I   ] [ S_12 ] [  0   ] [  E?  ]
        [ S_21 ] [  I   ] [ S_23 ] [  0   ]
        [  0   ] [ S_32 ] [  I   ] [ S_34 ]
        [  F?  ] [  0   ] [ S_43 ] [  I   ]

        Solves using the Red-Black factorization. Assumes we have RB = (A_i, B_i, C_i) and v of size m * nSlabs
        """
        m  = self.m
        RB = self.RB

        m = RB[0][0][0].shape[0]
        # Building the RHS:
        vPrimes = [rhs.copy()]

        # Now build the RHS:
        for l in range(len(RB) - 1):

            (SiM, _, T_inv, SiP) = RB[l]

            nSlabs   = len(SiM)
            nReduced = int(nSlabs / 2)
            vPrime   = np.zeros(m*nReduced)
            vPrev    = vPrimes[-1]

            for j in range(nReduced):
                i    = 2 * j
                prev = (i - 1) % nSlabs
                next = (i + 1) % nSlabs
                vPrime[j*m:(j+1)*m] = vPrev[i*m:(i+1)*m] - SiM[i] @ lu_solve(T_inv[prev], vPrev[prev*m:(prev+1)*m]) - SiP[i] @ lu_solve(T_inv[next], vPrev[next*m:(next+1)*m])

            vPrimes.append(vPrime)

        # Now get u:
        vPrimes[-1] = lu_solve(RB[-1][2][0], vPrimes[-1])
        
        for l in range(len(RB) - 1, 0, -1):
            (SiM, _, Tinv, SiP)   = RB[l-1]
            nReduced = int(len(SiM) / 2)
            for j in range(nReduced):
                i = 2 * j
                # We fill in the odd segments of u
                vPrimes[l-1][i*m:(i+1)*m] = vPrimes[l][j*m:(j+1)*m]
                
                # Here we compute the even segments of u:
                next = (j + 1) % nReduced
                vPrimes[l-1][(i+1)*m:(i+2)*m] -= SiM[i+1] @ vPrimes[l][j*m:(j+1)*m] + SiP[i+1] @ vPrimes[l][next*m:(next+1)*m]
                vPrimes[l-1][(i+1)*m:(i+2)*m] = lu_solve(Tinv[i+1], vPrimes[l-1][(i+1)*m:(i+2)*m])

        return vPrimes[0]


    # Construct a solver for the red-black scheme. This one is not periodic (aka not cyclical)
    def build_block_RB_solver_level(self, m, nSlabs, RB_level, cyclic=False):
        """
        
        [  T_1 ] [ S_12 ] [  0   ] [  E?  ]
        [ S_21 ] [  T_2 ] [ S_23 ] [  0   ]
        [  0   ] [ S_32 ] [  T_3 ] [ S_34 ]
        [  F?  ] [  0   ] [ S_43 ] [  T_4 ]

        Using linear operators corresponding to the slabs of a slab solver, we will construct a block tridiagonal direct solver

        This is based off of the red-black algorithm.

        We need to store 4 objects:
        1. Original S_rk_list, all entries are needed for either even solves or RHS of odd solves
        2. For i odd, the factorized systems B_i' = (T_i - S_{i,i-1} (T_i-1)^-1 S_{i-1,i} - S_{i,i+1} (T_i+1)^-1 S_{i+1,i})^-1 that makes up the "main diagonal"
        3. For i odd, the factorized system  A_i' = S_{i,i-1} (T_i-1)^-1 S_{i-1,i-2}
        4. For i odd, the factorized system  C_i' = S_{i,i+1} (T_i+1)^-1 S_{i+1,i+2}

        These become the new T_i, S_{i,i-1}, and S_{i,i+1} respectively.
        """

        SiM = RB_level[0]
        SiP = RB_level[3]

        T     = RB_level[1]
        T_inv = RB_level[2]

        # First let's build 2, B_i':
        I = np.eye(m, dtype=SiM[1].dtype)
        B_i = []
        for i in range(0, nSlabs, 2):
            B = reconstruct_from_lu_factor(T_inv[i])
            if (i > 0) or cyclic:
                B -= SiM[i] @ lu_solve(T_inv[i-1], (SiP[(i-1) % nSlabs] @ I))
            if (i < nSlabs - 1) or cyclic:
                B -= SiP[i] @ lu_solve(T_inv[i+1], (SiM[(i+1) % nSlabs] @ I))
            B_i.append(B)

        # Next let's build 3, A_i:
        A_i = [-SiM[_] @ lu_solve(T_inv[_-1], (SiM[(_ - 1) % nSlabs] @ I)) for _ in range(0, nSlabs, 2)]
        # And finally build 4, C_i:
        C_i = [-SiP[_] @ lu_solve(T_inv[_+1], (SiP[(_ + 1) % nSlabs] @ I)) for _ in range(0, nSlabs, 2)]

        # We need to account for cyclic effects if this is the topmost layer:
        if len(B_i) == 1 and cyclic:
            B_i[0] += A_i[0] + C_i[0]

        # Now we factorize every B_i:
        #B_i = [lu_factor(_, overwrite_a=True) for _ in B_i]

        return (A_i, B_i, [lu_factor(_, overwrite_a=True) for _ in B_i], C_i)

class NestedDissectionSolver:
    """
         [ L    B_l    0 ]    [ L     0    B_l] [u_left ]   [v_left]
    from [C_l    M    C_r] to [ 0     R    B_r] [u_right] = [v_right]
         [ 0    B_r    R ]    [C_l   C_r    M ] [u_sep  ]   [v_sep  ]
    """

    def __init__(self, L, R, C_l, C_r, B_l, B_r, M):
        self.L   = L
        self.R   = R
        self.C_l = C_l
        self.C_r = C_r
        self.B_l = B_l
        self.B_r = B_r
        self.M   = M


def build_nested_dissection(S_rk_list, T, cyclic=False):
    """
    Builds a factorized and sorted nested dissection solver.
    """
    m   = T[0].shape[0]
    SiM = [_[0]  for _ in S_rk_list]
    SiP = [_[-1] for _ in S_rk_list]

    nSlabs = len(T)

    if nSlabs > 2 and not cyclic:
        SiM[0] = np.zeros((m,m))
        SiP[-1] = np.zeros((m,m))

    build_nested_dissection_level(SiM, SiP, T, cyclic=cyclic)

    return 1

def build_nested_dissection_level(SiM, SiP, T, cyclic=False):

    m      = T[0].shape[0]
    nSlabs = len(T)

    # Special cases for nSlabs = 1 or 2
    if nSlabs == 1:
        return NestedDissectionSolver(None, None, None, None, None, None, lu_factor(T[0]))
    elif nSlabs == 2:
        return build_nested_dissection_2x2()
    else:
        # Denote the blocks that belong to left, right, and separator (merge)
        sep = nSlabs // 2

        T_l = T[:sep]
        T_r = T[(sep+1):]
        T_sep = T[sep]

        # TODO: might need to fix these indices for cyclic case
        if cyclic:
            print("might need to fix sub-block indices for cyclic case")

        SiM_l = SiM[:(len(T_l)-1)]
        SiM_r = SiM[-(len(T_r)-1):]

        SiP_l = SiP[:(len(T_l)-1)]
        SiP_r = SiP[-(len(T_r)-1):]

        # First set left and right blocks
        L = build_nested_dissection_level(SiM_l, SiP_l, T_l)
        R = build_nested_dissection_level(SiM_r, SiP_r, T_r)

        # TODO: might need to fix these indices for cyclic case
        # we also might need to pad these with zeros... or maybe not
        C_l = SiM[sep-1]
        C_r = SiP[sep]

        B_l = SiP[sep-1]
        B_r = SiM[sep]

        M = 1 # Build this out properly

        return NestedDissectionSolver(L, R, C_l, C_r, B_l, B_r, M)

def build_nested_dissection_2x2(SiM, SiP, T):
    """
    Special case when there are only two blocks of the same size, so there isn't really a separator.
    """

    S_12 = SiM[0]
    S_21 = SiP[0]

    T_1 = T[0]
    T_2 = T[1]

    T_1inv = lu_factor(T_1)

    # TODO: finish

    L = T_1inv
    C_l = S_21
    B_l = S_12

    M = lu_factor(T_2 - S_21 @ lu_solve(T_1inv, S_12))

    return NestedDissectionSolver(L, None, C_l, None, B_l, None, M)
    

# TODO: test 2x2 case. Decide best way to factorize everything...