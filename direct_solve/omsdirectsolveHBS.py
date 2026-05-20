import numpy as np
from scipy.sparse.linalg   import LinearOperator
import jax.numpy as jnp
import matAssembly.HBS.HBSnew as HBSnew
from abc import ABC, abstractmethod
from direct_solve.omsdirectsolve import DirectSolver

# ---------------------------------------------------------------------------
# Linear operator helpers
# ---------------------------------------------------------------------------

class id_op(LinearOperator):
    """Identity operator."""
    def __init__(self, n,dtype=np.float64):
        super().__init__(shape=(n, n), dtype=dtype)
    def _matvec(self, v):         return v.copy()
    def _matmat(self, v):         return v.copy()
    def _rmatvec(self, v):        return v.copy()
    def _rmatmat(self, v):        return v.copy()
    def solve(self, v, mode='N'): return v.copy()

def dense_to_linop(A):
    A = np.array(A)
    n = A.shape[0]
    lo = LinearOperator(
        shape=(n, n), dtype=A.dtype,
        matvec  = lambda v: A @ v,
        rmatvec = lambda v: A.T @ v,
        matmat  = lambda V: A @ V,
        rmatmat = lambda V: A.T @ V,
    )
    lo.solve = lambda v, mode='N': (
        np.linalg.solve(A, v) if mode == 'N' else np.linalg.solve(A.T, v)
    )
    lo.tree = lo.quad = None
    return lo

def _linop_from_mat(A):
    """Wrap a dense numpy matrix as a LinearOperator with .solve and .tree/.quad."""
    n  = A.shape[0]
    lo = LinearOperator(
        shape   = (n, n),
        dtype   = A.dtype,
        matvec  = lambda v: A @ v,
        rmatvec = lambda v: A.T @ v,
        matmat  = lambda V: A @ V,
        rmatmat = lambda V: A.T @ V,
    )
    lo.solve = lambda v, mode='N': (
        np.linalg.solve(A, v) if mode == 'N' else np.linalg.solve(A.T, v)
    )
    lo.tree = None
    lo.quad = None
    return lo


def STS_linop(Sl, T, Sr):
    """Returns the LinearOperator  -Sl @ T^{-1} @ Sr."""
    def sts_matmat(v, transpose=False):
        v_tmp = v[:, np.newaxis] if v.ndim == 1 else v
        if not transpose:
            result = -Sl.matmat(T.solve(Sr.matmat(v_tmp)))
        else:
            result = -Sr.rmatmat(T.solve(Sl.rmatmat(v_tmp), mode='T'))
        return result.flatten() if v.ndim == 1 else result

    return LinearOperator(
        shape   = (T.shape[0], T.shape[1]),
        dtype   = np.float64,
        matvec  = lambda v: sts_matmat(v),
        rmatvec = lambda v: sts_matmat(v, transpose=True),
        matmat  = lambda v: sts_matmat(v),
        rmatmat = lambda v: sts_matmat(v, transpose=True),
    )


def RB_linop(Ti, tm, tp, SiPi, SiMi, smp, spm):
    """
    Returns the LinearOperator for the Schur complement diagonal:
        Ti - SiPi @ tp^{-1} @ smp - SiMi @ tm^{-1} @ spm
    tm, tp, smp, spm may be None (boundary).
    """
    def smatmat(v, transpose=False):
        v_tmp = v[:, np.newaxis] if v.ndim == 1 else v
        if not transpose:
            result = Ti.matmat(v_tmp)
            if tp is not None:
                result -= SiPi.matmat(tp.solve(smp.matmat(v_tmp)))
            if tm is not None:
                result -= SiMi.matmat(tm.solve(spm.matmat(v_tmp)))
        else:
            result = Ti.rmatmat(v_tmp)
            if tp is not None:
                result -= smp.rmatmat(tp.solve(SiPi.rmatmat(v_tmp), mode='T'))
            if tm is not None:
                result -= spm.rmatmat(tm.solve(SiMi.rmatmat(v_tmp), mode='T'))
        return result.flatten() if v.ndim == 1 else result

    return LinearOperator(
        shape   = (Ti.shape[0], Ti.shape[1]),
        dtype   = np.float64,
        matvec  = lambda v: smatmat(v),
        rmatvec = lambda v: smatmat(v, transpose=True),
        matmat  = lambda v: smatmat(v),
        rmatmat = lambda v: smatmat(v, transpose=True),
    )

def Sprime_Linop(Sl,Sprime_prev,Sr,id=False):
    if id:
        def smatmat(v,transpose=False):        
            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):
                result = v_tmp-Sl.matmat(Sr.matmat(v_tmp))
            else:
                result = v_tmp-Sr.rmatmat(Sl.rmatmat(v_tmp))
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
                result = v_tmp-Sl.matmat(Sprime_prev.solve(Sr.matmat(v_tmp)))
            else:
                result = v_tmp-Sr.rmatmat(Sprime_prev.solve(Sl.rmatmat(v_tmp),mode='T'))
            if (v.ndim == 1):
                result = result.flatten()
            return result
    Sprime = LinearOperator(shape=(Sl.shape[0],Sr.shape[1]),\
        matvec = lambda v:smatmat(v), rmatvec = lambda v:smatmat(v,transpose=True),\
        matmat = lambda v:smatmat(v), rmatmat = lambda v:smatmat(v,transpose=True))
    return Sprime

def Dprime_Linop(D,A,B,Dprev):
    def dmatmat(v,transpose=False):        
        if (v.ndim == 1):
            v_tmp = v[:,np.newaxis]
        else:
            v_tmp = v

        if (not transpose):
            result = D.matmat(v_tmp)-A.matmat(Dprev.solve(B.matmat(v_tmp)))
        else:
            result = D.rmatmat(v_tmp)-B.rmatmat(Dprev.solve(A.rmatmat(v_tmp),mode='T'))
        if (v.ndim == 1):
            result = result.flatten()
        return result
    Dprime = LinearOperator(shape=(D.shape[0],D.shape[1]),\
        matvec = lambda v:dmatmat(v), rmatvec = lambda v:dmatmat(v,transpose=True),\
        matmat = lambda v:dmatmat(v), rmatmat = lambda v:dmatmat(v,transpose=True))
    return Dprime
# returns -Sl@T\Sr
def STS_linop(Sl,T,Sr):
    def sts_matmat(v,transpose=False):        
        if (v.ndim == 1):
            v_tmp = v[:,np.newaxis]
        else:
            v_tmp = v

        if (not transpose):
            result = -Sl.matmat(T.solve(Sr.matmat(v_tmp)))
        else:
            result = -Sr.rmatmat(T.solve(Sl.rmatmat(v_tmp),mode='T'))
        if (v.ndim == 1):
            result = result.flatten()
        return result
    STS = LinearOperator(shape=(T.shape[0],T.shape[1]),\
        matvec = lambda v:sts_matmat(v), rmatvec = lambda v:sts_matmat(v,transpose=True),\
        matmat = lambda v:sts_matmat(v), rmatmat = lambda v:sts_matmat(v,transpose=True))
    return STS


'''

Fredholm second kind Block Tridiagonal (BTD) solver using HBS acceleration
Uses that the diagonal is identity

'''

class ThomasSolverHBS(DirectSolver):

    def __init__(self,m,rk,cyclic=False):
        super().__init__(m,cyclic)
        self.rk = rk
        self.solve_method = None
    def factorize_helper(self, S_rk_list, diagList=None):
        if diagList==None:
            self.factorize_id_diag(S_rk_list)
            self.solve_method = 'id_diag'
        else:
            self.factorize_with_diag(S_rk_list, diagList)
            self.solve_method = 'diag'
    def factorize_id_diag(self, S_rk_list):
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
        
        S'_{i+1} = I-S_{i+1,i}S'_{i}\\S_{i,i+1}
        b'_{i+1} = b_{i+1}-S_{i+1,i}\\S'_{i}\(b'_i)
        ---------------------------------------------------------------------------
        NOTE: different sign convention on S is possible, in this case, recurrence changes slightly

        """
        rk = self.rk
        m = S_rk_list[0][0].shape[0]
        n = len(S_rk_list) - 1
        I = id_op(m)
        Sl = [S_rk_list[_][0] for _ in range(1,n+1)]
        Sprime = [I]
        Sr = [S_rk_list[_][-1] for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)
        for i in range(1, n+1):
            if i==1:
                Sprime_i = HBSnew.HBSMAT(Sprime_Linop(Sl[0],I,Sr[0],id=True),Sl[0].tree,Sl[0].quad)
                Sprime_i.construct(rk,compute_ULV=True)
                
            else:
                Sprime_i = HBSnew.HBSMAT(Sprime_Linop(Sl[i-1],Sprime[i-1],Sr[i-1]),Sl[i-1].tree,Sl[i-1].quad)
                Sprime_i.construct(rk,compute_ULV=True)
            Sprime.append(Sprime_i)
        self.A = Sl
        self.B = Sprime
        self.C = Sr
    
    def factorize_with_diag(self, AB_list,D_list):
        """
    
        [ D0 ] [ B0 ] [ 00 ] [ 00 ]
        [ A0 ] [ D1 ] [ B1 ] [ 00 ]
        [ 00 ] [ A1 ] [ D2 ] [ B2 ]
        [ 00 ] [ 00 ] [ A2 ] [ D3 ]


        """
        rk = self.rk
        m = D_list[0].shape[0]
        n = len(D_list) - 1

        # Thus we need three lists of block matrices: A, B, and C:
        A = [AB_list[_][0] for _ in range(n)]
        B = [D_list[0]] # Set initial B_i to identity matrix LU factor (can specialize this to be just identity later)
        C = [AB_list[_][-1] for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)

        for i in range(1, n+1):
            B_i = HBSnew.HBSMAT(Dprime_Linop(D_list[i],A[i-1],C[i-1],B[-1],D_list[i].tree,D_list[i-1].quad))
            B_i.construct(self.rk,compute_ULV=True)    
            B.append(B_i)
        
        self.A = A
        self.B = B
        self.C = C
    
    def solve_helper(self,rhs,glob_target_dofs=None):
        if self.solve_method=='id_diag':
            return self.solve_id_diag(rhs,glob_target_dofs)
        elif self.solve_method == 'diag':
            return self.solve_with_diag(rhs,glob_target_dofs)
        else:
            raise ValueError('Factorization not set')
    
    def solve_id_diag(self, rhs,glob_target_dofs = None):
        
        m       = self.m
        Sl      = self.A
        Sprime  = self.B
        Sr      = self.C
        n       = len(Sl)
        d       = rhs.copy()

        if rhs.ndim==1:
            d = d[:,np.newaxis]

        if glob_target_dofs is None:
            indices = [range(l*m, (l+1)*m) for l in range(len(Sprime))]
        else:
            indices = glob_target_dofs
        
        for i in range(1, n+1):
            if i==1:
                d[indices[i],:] = d[indices[i],:] - Sl[i-1]@d[indices[i-1],:]
            else:
                d[indices[i],:] = d[indices[i],:] - Sl[i-1]@(Sprime[i-1].solve(d[indices[i-1],:]))

        x             = np.zeros(d.shape)
        x[indices[n],:] = Sprime[n].solve(d[indices[n],:] )
        for i in range(n-1, 0, -1):
            x[indices[i],:] = Sprime[i].solve(d[indices[i],:] - Sr[i] @ x[indices[i+1],:] )

        x[indices[0],:] = d[indices[0],:] - Sr[0] @ x[indices[1],:]
        if rhs.ndim==1:
            x = x.flatten()
        return x
    
    def solve_with_diag(self, rhs,glob_target_dofs = None):
        
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
        
        for i in range(1, n+1):
            d[indices[i]] = d[indices[i]] - A[i-1] @ B[i].solve( d[indices[i-1]])

        x             = np.zeros(d.shape)
        x[indices[n]] = B[n].solve( d[indices[n]])

        for i in range(n-1, 0, -1):
            x[indices[i]] = B[i].solve( d[indices[i]] - C[i] @ x[indices[i+1]])

        x[indices[0]] = B[0].solve( d[indices[0]] - C[0] @ x[indices[1]])

        return x

    def factorize(self, S_rk_list, T=None):
        self.factorize_helper(S_rk_list, T)

    def solve(self, rhs, glob_target_dofs=None):
        x = self.solve_helper(rhs, glob_target_dofs)
        if self.cyclic:
            x = x - self.smw_block @ x

        return x



# ---------------------------------------------------------------------------
# HBS Red-Black solver
# ---------------------------------------------------------------------------

class RedBlackSolverHBS(DirectSolver):
    """
    Block-tridiagonal solver using cyclic reduction (red-black),
    replacing dense LU factorizations with HBS-compressed operators.

    RB level structure mirrors the dense RedBlackSolver exactly:
      RB[l] = (SiM, T, T_hbs, SiP)  -- all four lists of length nSlabs_at_level
      SiM[i]   : left  off-diagonal at node i
      T[i]     : diagonal LinearOperator at node i
      T_hbs[i] : HBS factorization of T[i]  (replaces lu_factor)
      SiP[i]   : right off-diagonal at node i

    lu_solve(T_inv[k], v)  -->  T_hbs[k].solve(v)
    lu_factor(T[i])        -->  HBSnew.HBSMAT(...).construct(rk, compute_ULV=True)
    """

    def __init__(self, m, rk, tree, quad, cyclic=False):
        super().__init__(m, cyclic)
        self.rk   = rk
        self.tree = tree
        self.quad = quad

    # ------------------------------------------------------------------

    def _hbs(self, linop,rk = None):
        if rk is None:
            rkloc = self.rk
        else:
            rkloc = rk
        """Compress a LinearOperator into an HBS matrix and factorize it."""
        h = HBSnew.HBSMAT(linop, self.tree, self.quad)
        h.construct(rkloc, compute_ULV=True)
        return h

    # ------------------------------------------------------------------
    # factorize  -- mirrors RedBlackSolver.factorize
    # ------------------------------------------------------------------

    def factorize(self, S_rk_list, T=None):
        m      = S_rk_list[0][0].shape[0]
        nSlabs = len(S_rk_list)

        if not ((nSlabs & (nSlabs - 1) == 0) and nSlabs != 0):
            raise ValueError("Number of slabs must be a power of 2.")

        SiM = [_[0]  for _ in S_rk_list]
        SiP = [_[-1] for _ in S_rk_list]

        # Boundary zeros -- kept as zero LinearOperators (like np.zeros in dense version)
        # so that indexing is uniform throughout.
        if not self.cyclic:
            SiM[0]  = _linop_from_mat(np.zeros((m, m)))
            SiP[-1] = _linop_from_mat(np.zeros((m, m)))

        if T is None:
            #T = [dense_to_linop(np.eye(m)) for _ in range(nSlabs)]
            T = [id_op(m,S_rk_list[0][0].dtype) for _ in range(nSlabs)]
        # At level 0, T operators are used directly without HBS compression.
        # Deeper levels produce HBS objects via _build_level.
        T_hbs = T

        RB = [(SiM, T, T_hbs, SiP)]

        l = nSlabs
        rk = self.rk
        while l > 1:
            RB.append(self._build_level(m, l, RB[-1],rk))
            rk = rk #+ 20
            l //= 2

        self.nSlabs = nSlabs
        self.RB     = RB

    # ------------------------------------------------------------------
    # _build_level  -- mirrors build_block_RB_solver_level
    # ------------------------------------------------------------------

    def _build_level(self, m, nSlabs, RB_level,rk):
        """
        Eliminate all odd-indexed nodes and return a reduced level of length nSlabs//2.

        Dense counterpart formulas:
          B_i  = T[i] - SiM[i] @ T[i-1]^{-1} @ SiP[i-1]
                       - SiP[i] @ T[i+1]^{-1} @ SiM[i+1]   (new diagonal, factorized)
          A_i  = -SiM[i] @ T[i-1]^{-1} @ SiM[i-1]          (new left  off-diag)
          C_i  = -SiP[i] @ T[i+1]^{-1} @ SiP[i+1]          (new right off-diag)
        for even i = 0, 2, ..., nSlabs-2.

        Returns (A_i, B_i, T_hbs_new, C_i) matching the dense (A_i, B_i, T_inv, C_i) layout.
        """
        SiM   = RB_level[0]
        T     = RB_level[1]
        T_hbs = RB_level[2]
        SiP   = RB_level[3]

        cyclic = self.cyclic

        B_i      = []
        T_hbs_new = []

        for i in range(0, nSlabs, 2):
            spm = SiP[(i - 1) % nSlabs] if ((i > 0) or cyclic) else None
            smp = SiM[(i + 1) % nSlabs] if ((i < nSlabs - 1) or cyclic) else None
            tm  = T_hbs[(i - 1) % nSlabs] if spm is not None else None
            tp  = T_hbs[(i + 1) % nSlabs] if smp is not None else None

            # B_i linop = T[i] - SiM[i]@T[i-1]^{-1}@SiP[i-1] - SiP[i]@T[i+1]^{-1}@SiM[i+1]
            B_linop = RB_linop(T[i], tm, tp, SiP[i], SiM[i], smp, spm)
            B_hbs   = self._hbs(B_linop,rk)
            B_i.append(B_linop)       # keep as LinearOperator (mirrors dense T list)
            T_hbs_new.append(B_hbs)   # factorized (mirrors lu_factor)

        # A_i = -SiM[i] @ T[i-1]^{-1} @ SiM[i-1]  for even i = 0,2,...
        A_i = [
            self._hbs(STS_linop(SiM[i], T_hbs[(i - 1) % nSlabs], SiM[(i - 1) % nSlabs]),rk)
            for i in range(0, nSlabs, 2)
        ]

        # C_i = -SiP[i] @ T[i+1]^{-1} @ SiP[i+1]  for even i = 0,2,...
        C_i = [
            self._hbs(STS_linop(SiP[i], T_hbs[(i + 1) % nSlabs], SiP[(i + 1) % nSlabs]),rk)
            for i in range(0, nSlabs, 2)
        ]

        return (A_i, B_i, T_hbs_new, C_i)

    # ------------------------------------------------------------------
    # solve  -- mirrors RedBlackSolver.solve
    # ------------------------------------------------------------------

    def solve(self, rhs):
        m  = self.m
        RB = self.RB

        # ---- forward reduction ----------------------------------------
        # Mirrors dense forward loop exactly; lu_solve(T_inv[k], v) -> T_hbs[k].solve(v)
        vPrimes = [rhs.copy()]

        for l in range(len(RB) - 1):
            SiM, _, T_hbs, SiP = RB[l]

            nSlabs   = len(SiM)
            nReduced = nSlabs // 2
            vPrev    = vPrimes[-1]
            vPrime   = np.zeros(m * nReduced)

            for j in range(nReduced):
                i = 2 * j

                prev = (i - 1) % nSlabs if (self.cyclic or i > 0)          else None
                next = (i + 1) % nSlabs if (self.cyclic or i < nSlabs - 1) else None

                contrib = vPrev[i*m:(i+1)*m].copy()
                if prev is not None:
                    contrib -= SiM[i].matmat(
                        T_hbs[prev].solve(vPrev[prev*m:(prev+1)*m, np.newaxis])
                    )[:, 0]
                if next is not None:
                    contrib -= SiP[i].matmat(
                        T_hbs[next].solve(vPrev[next*m:(next+1)*m, np.newaxis])
                    )[:, 0]

                vPrime[j*m:(j+1)*m] = contrib

            vPrimes.append(vPrime)

        # ---- coarsest solve -------------------------------------------
        # Mirrors:  vPrimes[-1] = lu_solve(RB[-1][2][0], vPrimes[-1])
        vPrimes[-1] = RB[-1][2][0].solve(vPrimes[-1])

        # ---- back substitution ----------------------------------------
        # Mirrors dense back-sub exactly; lu_solve(Tinv[i+1], v) -> T_hbs[i+1].solve(v)
        for l in range(len(RB) - 1, 0, -1):
            SiM, _, T_hbs, SiP = RB[l - 1]

            nSlabs   = len(SiM)
            nReduced = nSlabs // 2

            for j in range(nReduced):
                i = 2 * j

                # Copy even node solution down from coarser level
                vPrimes[l-1][i*m:(i+1)*m] = vPrimes[l][j*m:(j+1)*m]

                # Recover odd node i+1
                next_j = (j + 1) % nReduced
                contrib = SiM[i+1].matmat(
                    vPrimes[l][j*m:(j+1)*m, np.newaxis]
                )[:, 0]
                if self.cyclic or j + 1 < nReduced:
                    contrib += SiP[i+1].matmat(
                        vPrimes[l][next_j*m:(next_j+1)*m, np.newaxis]
                    )[:, 0]

                vPrimes[l-1][(i+1)*m:(i+2)*m] -= contrib
                vPrimes[l-1][(i+1)*m:(i+2)*m] = T_hbs[i+1].solve(
                    vPrimes[l-1][(i+1)*m:(i+2)*m]
                )

        return vPrimes[0]