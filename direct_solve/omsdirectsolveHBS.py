import numpy as np
from scipy.sparse.linalg   import LinearOperator
import jax.numpy as jnp
import matAssembly.HBS.HBStorch as HBSnew
from abc import ABC, abstractmethod
from direct_solve.omsdirectsolve import DirectSolver
import torch

# ---------------------------------------------------------------------------
# Linear operator helpers
# ---------------------------------------------------------------------------

def _rdtype(*ops):
    """Promoted dtype of a set of operators; float64 if none carry one."""
    dts = [getattr(o, "dtype", None) for o in ops]
    dts = [d for d in dts if d is not None]
    return np.result_type(*dts) if dts else np.float64


class id_op(LinearOperator):
    """Identity operator."""
    def __init__(self, n,dtype=np.float64):
        super().__init__(shape=(n, n), dtype=dtype)
        self.tree = None
        self.quad = None
    def _matvec(self, v):         return v.copy()
    def _matmat(self, v):         return v.copy()
    def _rmatvec(self, v):        return v.copy()
    def _rmatmat(self, v):        return v.copy()
    def solve(self, v, mode='N'): return v.copy()


class zero_op(LinearOperator):
    """Zero operator; replaces a materialized dense zero block."""
    def __init__(self, n, dtype=np.float64, m=None):
        m = n if m is None else m
        super().__init__(shape=(m, n), dtype=dtype)
        self.tree = None
        self.quad = None
    def _matvec(self, v):
        return np.zeros(self.shape[0], dtype=self.dtype)
    def _matmat(self, V):
        return np.zeros((self.shape[0], V.shape[1]), dtype=self.dtype)
    def _rmatvec(self, v):
        return np.zeros(self.shape[1], dtype=self.dtype)
    def _rmatmat(self, V):
        return np.zeros((self.shape[1], V.shape[1]), dtype=self.dtype)
    def solve(self, v, mode='N'):
        raise NotImplementedError(
            "zero_op is singular: a boundary off-diagonal block was solved "
            "with, which means a boundary guard is missing upstream."
        )


class dead_op:
    """Placeholder for a diagonal slot that provably has no consumer.

    Occupies its position in T_hbs so that indexing stays uniform, and raises
    on every operation so that a consumer missed by the analysis in
    RedBlackSolverHBS surfaces as an exception rather than as a wrong answer.
    """
    def __init__(self, why=""):
        self.why  = why
        self.tree = None
        self.quad = None

    def _die(self, *args, **kwargs):
        raise RuntimeError(
            f"dead_op used ({self.why}): this diagonal block was never built "
            "because RedBlackSolverHBS determined that nothing reads it. "
            "Pass skip_unused_ulv=False to restore unconditional construction."
        )

    matmat = rmatmat = matvec = rmatvec = solve = _die


def _is_id(op):
    """True if `op` is a structural identity, i.e. an id_op instance.

    Detection is deliberately structural rather than semantic: the fast paths
    below skip work only when the object is known to be the identity, so a
    dense np.eye wrapped in a LinearOperator would take the slow path.
    RedBlackSolverHBS._normalize_diag exists to convert such diagonals up
    front.
    """
    return isinstance(op, id_op)


def _linop_from_mat(A):
    """Wrap a dense numpy matrix as a LinearOperator with .solve and .tree/.quad."""
    A  = np.asarray(A)
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


dense_to_linop = _linop_from_mat


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
        shape   = (Sl.shape[0], Sr.shape[1]),
        dtype   = _rdtype(Sl, T, Sr),
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
                result = result - SiPi.matmat(tp.solve(smp.matmat(v_tmp)))
            if tm is not None:
                result = result - SiMi.matmat(tm.solve(spm.matmat(v_tmp)))
        else:
            result = Ti.rmatmat(v_tmp)
            if tp is not None:
                result = result - smp.rmatmat(tp.solve(SiPi.rmatmat(v_tmp), mode='T'))
            if tm is not None:
                result = result - spm.rmatmat(tm.solve(SiMi.rmatmat(v_tmp), mode='T'))
        return result.flatten() if v.ndim == 1 else result

    return LinearOperator(
        shape   = (Ti.shape[0], Ti.shape[1]),
        dtype   = _rdtype(Ti, tm, tp, SiPi, SiMi, smp, spm),
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
    Sprime = LinearOperator(shape=(Sl.shape[0],Sr.shape[1]),
        dtype  = _rdtype(Sl, Sprime_prev, Sr),
        matvec = lambda v:smatmat(v), rmatvec = lambda v:smatmat(v,transpose=True),
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
    Dprime = LinearOperator(shape=(D.shape[0],D.shape[1]),
        dtype  = _rdtype(D, A, B, Dprev),
        matvec = lambda v:dmatmat(v), rmatvec = lambda v:dmatmat(v,transpose=True),
        matmat = lambda v:dmatmat(v), rmatmat = lambda v:dmatmat(v,transpose=True))
    return Dprime


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
        if  torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
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
        I = id_op(m, S_rk_list[0][0].dtype)
        Sl = [S_rk_list[_][0] for _ in range(1,n+1)]
        Sprime = [I]
        Sr = [S_rk_list[_][-1] for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)
        for i in range(1, n+1):
            if i==1:
                Sprime_i = HBSnew.HBSMAT(Sprime_Linop(Sl[0],I,Sr[0],id=True),device=device,tree = Sl[0].tree,quad = Sl[0].quad)
                Sprime_i.construct(rk,compute_ULV=True,fast=True)
                
            else:
                Sprime_i = HBSnew.HBSMAT(Sprime_Linop(Sl[i-1],Sprime[i-1],Sr[i-1]),device=device,tree = Sl[i-1].tree,quad = Sl[i-1].quad)
                Sprime_i.construct(rk,compute_ULV=True,fast=True)
            Sprime.append(Sprime_i.to('cpu'))
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
            B_i = HBSnew.HBSMAT(Dprime_Linop(D_list[i], A[i-1], C[i-1], B[-1]),
                                tree=D_list[i].tree, quad=D_list[i].quad)
            B_i.construct(self.rk,compute_ULV=True,fast=True)    
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

        x             = np.zeros(d.shape, dtype=d.dtype)
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
            d[indices[i]] = d[indices[i]] - A[i-1] @ B[i-1].solve( d[indices[i-1]])

        x             = np.zeros(d.shape, dtype=d.dtype)
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

    ---------------------------------------------------------------------
    FUSED CONSTRUCTION (fused=True, default)
    ---------------------------------------------------------------------
    The three operators produced at each retained node,

        B_i = T_i - S^+_i T_{i+1}^{-1} S^-_{i+1} - S^-_i T_{i-1}^{-1} S^+_{i-1}
        A_i =     - S^-_i T_{i-1}^{-1} S^-_{i-1}
        C_i =     - S^+_i T_{i+1}^{-1} S^+_{i+1}

    (S^- = SiM, S^+ = SiP) are compressed by black-box sampling.  Driving
    each one through its own `construct` re-solves the same eliminated
    diagonals against the same right-hand sides.  Writing out which
    constructions consume the eliminated node k:

        B_{k-1} : T_k^{-1} S^-_k Om        A_{k+1} : T_k^{-1} S^-_k Om
        C_{k-1} : T_k^{-1} S^+_k Om        B_{k+1} : T_k^{-1} S^+_k Om

    -- four solves, two distinct right-hand sides.  This class instead shares
    one Omega / Psi pair across the whole level, caches

        Xm_k = T_k^{-1} S^-_k Om,     Xp_k = T_k^{-1} S^+_k Om

    once per eliminated node (a single fused solve with 2s columns), and then
    forms all three sample blocks at each retained node from them.  The
    adjoint side shares differently but just as well: B_i^T and C_i^T both
    need t^+_i = T_{i+1}^{-T} (S^+_i)^T Psi, and B_i^T and A_i^T both need
    t^-_i = T_{i-1}^{-T} (S^-_i)^T Psi.

    Per (retained, eliminated) pair this takes the forward+adjoint sampling
    from 13 elementary applies/solves to 9, and the call count from 13 to 6.
    The compressions themselves are unchanged in number and rank.

    Requires HBSMAT.construct(rk, Om, Psi, Y, Z, ...), i.e. the
    externally-sampled path.  Set fused=False for the original one-operator-
    at-a-time construction.

    ---------------------------------------------------------------------
    IDENTITY DIAGONAL
    ---------------------------------------------------------------------
    When T_i = I the level-0 work collapses: T_k^{-1} is a no-op, so pass 1
    needs no solve and no fused right-hand side, and T_i Om / T_i^T Psi are
    Om / Psi themselves.  Level 0 holds half of all eliminated nodes, so this
    is the bulk of the identity saving.  The structure does not survive the
    reduction -- B_i = I - E_i is not the identity -- so there is nothing
    further to exploit from level 1 on.

    These fast paths key off `isinstance(op, id_op)`.  A caller supplying
    T = [dense_to_linop(np.eye(m))] * nSlabs would silently take the slow
    path, so `factorize` routes any supplied T through `_normalize_diag`
    first; see `identity_diag`.

    ---------------------------------------------------------------------
    ULV FACTORIZATION COUNT  (skip_unused_ulv=True, default)
    ---------------------------------------------------------------------
    Compressing an operator and factorizing it are separate costs.  Only a
    diagonal that is *eliminated* is ever inverted, so most of the operators
    built here need the compressed form for applies but no ULV at all:

      * A_i and C_i become SiM / SiP one level down and are only ever fed to
        matmat / rmatmat -- never solved.  `zero_op.solve` raising is the
        same invariant stated defensively.  Their ULV is pure waste.

      * A retained node i becomes child j = i // 2 at the next level, where
        only odd j are eliminated.  So B_i needs a ULV iff j is odd, i.e.
        i % 4 == 2, plus the coarsest node (nSlabs == 2, where `solve`
        inverts the single survivor directly).

      * When compress_diag=False the next level's T[i] is the uncompressed
        RB_linop, so an even B_hbs has no consumer at all -- neither its ULV
        nor its compression.  Those slots get a `dead_op`.

    With an identity level-0 diagonal (no factorizations there at all) the
    resulting count is

        sum_{l=1}^{L-1} N/2^(l+1)  +  1  =  N/2

    against the 3(N-1) - 2*log2(N) unconditional ULVs of the previous
    version, and against 2N-1 for a textbook cyclic reduction that also
    factorizes the N identity diagonals at level 0.  `nULV`, `nULVSkipped`
    and `nDeadSkipped` record this directly.

    Operators built without a ULV get their `solve` replaced by a raising
    stub, so a consumer missed by the analysis above fails loudly.
    """

    def __init__(self, m, rk, tree, quad, cyclic=False,
                 compress_diag=True, fused=True, device='cpu', fast=False,
                 seed=0, identity_diag=None, skip_unused_ulv=True):
        super().__init__(m, cyclic)
        self.rk   = rk
        self.tree = tree
        self.quad = quad
        self.compress_diag = compress_diag
        self.fused  = fused
        self.device = device
        self.fast   = fast
        self.identity_diag = identity_diag
        self.skip_unused_ulv = skip_unused_ulv
        self._dtype = np.float64
        self._rng   = np.random.default_rng(seed)
        self.nConstruct = 0
        self.nSolve     = 0     
        self.nApply     = 0     
        self.nIdSkipped = 0     
        self.nULV         = 0   
        self.nULVSkipped  = 0   
        self.nDeadSkipped = 0   

    # ------------------------------------------------------------------

    def _nsamples(self, rk):
        """Sample count, matching HBSMAT.construct's internal choice.

        Every operator sharing an Omega must use the same s.  All nodes share
        self.tree, so one value per level is consistent by construction.
        """
        mls = getattr(self.tree, "_min_leaf_size", rk)
        return 2 * max(rk, mls) + rk + 10

    def _want_ulv(self, compute_ULV):
        """Resolve a requested compute_ULV against the opt-out flag."""
        return True if not self.skip_unused_ulv else bool(compute_ULV)

    def _guard_no_ulv(self, h, label):
        """Replace `solve` on an unfactorized operator with a raising stub.

        The savings here rest on an analysis of which diagonals are inverted.
        If that analysis is wrong somewhere, the failure mode without this
        guard depends entirely on what HBSMAT does when asked to solve
        without a ULV factorization -- possibly a wrong answer with no
        warning.  With it, the mistake is an exception naming the block.
        """
        def _no_ulv_solve(*args, **kwargs):
            raise RuntimeError(
                f"solve() called on {label}, which was built with "
                "compute_ULV=False because RedBlackSolverHBS determined it "
                "is only ever applied, never inverted. Pass "
                "skip_unused_ulv=False to restore unconditional "
                "factorization."
            )
        try:
            h.solve = _no_ulv_solve
        except Exception:
            # Attribute is not settable on this HBSMAT build; the operator is
            # still correct for applies, we just lose the tripwire.
            pass
        return h

    def _dead_diag(self, label):
        """Placeholder for a diagonal with no consumer at all."""
        if not self.skip_unused_ulv:
            raise AssertionError("_dead_diag reached with skip_unused_ulv=False")
        self.nDeadSkipped += 1
        return dead_op(label)

    def _hbs(self, linop, rk=None, device=None, compute_ULV=True, label=None):
        """Compress a LinearOperator into an HBS matrix.

        compute_ULV=False compresses for applies only and skips the
        factorization; the result is fitted with a raising `solve`.
        """
        rkloc  = self.rk if rk is None else rk
        dev    = self.device if device is None else device
        ulv    = self._want_ulv(compute_ULV)
        h = HBSnew.HBSMAT(linop, device=dev, tree=self.tree, quad=self.quad)
        h.construct(rkloc, compute_ULV=ulv, fast=self.fast)
        self.nConstruct += 1
        if ulv:
            self.nULV += 1
        else:
            self.nULVSkipped += 1
            h = self._guard_no_ulv(h, label or "an HBS block")
        return h

    def _hbs_from_samples(self, rk, Om, Psi, Y, Z, compute_ULV=True, label=None):
        """Compress from externally supplied samples Y = M Om, Z = M^T Psi.

        Om/Psi/Y/Z must be numpy: constructHBS calls torch.from_numpy on all
        four.  compute_ULV=False skips the factorization; see `_hbs`.
        """
        ulv = self._want_ulv(compute_ULV)
        h = HBSnew.HBSMAT(device=self.device, tree=self.tree, quad=self.quad)
        h.construct(rk, Om=np.ascontiguousarray(Om), Psi=np.ascontiguousarray(Psi),
                    Y=np.ascontiguousarray(Y), Z=np.ascontiguousarray(Z),
                    compute_ULV=ulv, fast=self.fast)
        h.to('cpu')
        self.nConstruct += 1
        if ulv:
            self.nULV += 1
        else:
            self.nULVSkipped += 1
            h = self._guard_no_ulv(h, label or "an HBS block")
        return h

    # ------------------------------------------------------------------

    @staticmethod
    def _needs_ulv(i, nSlabs):
        """Does retained node `i` of a level of size `nSlabs` need a ULV?

        Node i becomes child j = i // 2 one level down, and only the odd
        children are eliminated there, so a ULV is needed iff j is odd, i.e.
        i % 4 == 2.  The exception is the coarsest level: when nSlabs == 2
        the single child j = 0 is inverted directly by `solve`.
        """
        return (i % 4 == 2) or (nSlabs == 2)

    # -- small counted wrappers ---------------------------------------- #

    def _ap(self, op, X):
        self.nApply += 1
        return np.asarray(op.matmat(X))

    def _apT(self, op, X):
        self.nApply += 1
        return np.asarray(op.rmatmat(X))

    def _sv(self, op, X, mode='N'):
        self.nSolve += 1
        return np.asarray(op.solve(X, mode=mode))

    def _normalize_diag(self, T, m):
        """Swap identity diagonals for id_op so the fast paths engage.

        The level-0 savings below are triggered by `isinstance(op, id_op)`, not
        by the operator being mathematically the identity.  A caller passing
        `[dense_to_linop(np.eye(m))] * nSlabs` would otherwise run a real m x m
        GEMM per apply and np.linalg.solve on a dense identity per eliminated
        node -- same answer, far more expensive, no warning.

        The probe is exact-arithmetic reliable: if T - I is nonzero then
        (T - I) X = 0 for a Gaussian X has probability zero, so two columns
        suffice.
        """
        if self.identity_diag is False:
            return T

        out, replaced = [], 0
        for op in T:
            if _is_id(op):
                out.append(op)
                continue
            if self.identity_diag is True:
                out.append(id_op(m, self._dtype))
                replaced += 1
                continue
            X = self._rng.standard_normal(size=(m, 2))
            try:
                Y = np.asarray(op.matmat(X))
            except Exception:
                out.append(op)
                continue
            nrm = np.linalg.norm(X)
            if nrm > 0 and np.linalg.norm(Y - X) <= 1e-12 * nrm:
                out.append(id_op(m, self._dtype))
                replaced += 1
            else:
                out.append(op)
        self.nIdNormalized = replaced
        return out

    # ------------------------------------------------------------------
    # factorize
    # ------------------------------------------------------------------

    def factorize(self, S_rk_list, T=None):
        m      = S_rk_list[0][0].shape[0]
        nSlabs = len(S_rk_list)

        if not ((nSlabs & (nSlabs - 1) == 0) and nSlabs != 0):
            raise ValueError("Number of slabs must be a power of 2.")

        self._dtype = S_rk_list[0][0].dtype

        SiM = [_[0].to('cpu')  for _ in S_rk_list]
        SiP = [_[-1].to('cpu') for _ in S_rk_list]

        # Boundary zeros -- kept as zero LinearOperators so indexing is uniform.
        if not self.cyclic:
            SiM[0]  = zero_op(m, self._dtype)
            SiP[-1] = zero_op(m, self._dtype)

        if T is None:
            T = [id_op(m, self._dtype) for _ in range(nSlabs)]
        else:
            T = self._normalize_diag(list(T), m)
        # At level 0, T operators are used directly without HBS compression.
        T_hbs = T

        RB = [(SiM, T, T_hbs, SiP)]

        l = nSlabs
        rk = self.rk
        while l > 1:
            builder = self._build_level_fused if self.fused else self._build_level
            RB.append(builder(m, l, RB[-1], rk))
            rk = rk  # + 20
            l //= 2

        self.nSlabs = nSlabs
        self.RB     = RB

    # ------------------------------------------------------------------
    # _build_level_fused  -- tier-2 shared solves
    # ------------------------------------------------------------------

    def _build_level_fused(self, m, nSlabs, RB_level, rk):
        SiM   = RB_level[0]
        T     = RB_level[1]
        T_hbs = RB_level[2]
        SiP   = RB_level[3]

        cyclic = self.cyclic
        dtype  = self._dtype

        s   = self._nsamples(rk)
        Om  = self._rng.standard_normal(size=(m, s))
        Psi = self._rng.standard_normal(size=(m, s))

        # ---------------------------------------------------------------
        # pass 1 -- eliminated (odd) nodes: one fused solve each
        #
        #   Xm_k = T_k^{-1} S^-_k Om   feeds B_{k-1} and A_{k+1}
        #   Xp_k = T_k^{-1} S^+_k Om   feeds C_{k-1} and B_{k+1}
        #
        # Xp is skipped for the final odd node in the non-cyclic case: its two
        # consumers are C_{nSlabs-2} (structurally zero, since S^+_{nSlabs-1}
        # = 0) and B_{nSlabs}, which does not exist.
        # ---------------------------------------------------------------
        Xm, Xp = {}, {}
        for k in range(1, nSlabs, 2):
            need_p = cyclic or (k != nSlabs - 1)

            if _is_id(T_hbs[k]):
                # T_k^{-1} is a no-op, so there is no solve to fuse.  Going
                # through the general path would allocate an m x 2s block,
                # copy it inside id_op.solve, and slice it straight back
                # apart -- pure overhead at level 0, which holds half of all
                # eliminated nodes.
                Xm[k] = self._ap(SiM[k], Om)
                if need_p:
                    Xp[k] = self._ap(SiP[k], Om)
                self.nIdSkipped += 1
                continue

            cols = [self._ap(SiM[k], Om)]
            if need_p:
                cols.append(self._ap(SiP[k], Om))
            RHS = cols[0] if len(cols) == 1 else np.concatenate(cols, axis=1)

            X = self._sv(T_hbs[k], RHS)        # one solve, up to 2s columns
            Xm[k] = X[:, :s]
            if need_p:
                Xp[k] = X[:, s:]

        # ---------------------------------------------------------------
        # pass 2 -- retained (even) nodes
        # ---------------------------------------------------------------
        B_i, T_hbs_new, A_i, C_i = [], [], [], []

        for i in range(0, nSlabs, 2):
            has_left  = cyclic or i > 0
            has_right = cyclic or i < nSlabs - 1
            kL = (i - 1) % nSlabs
            kR = (i + 1) % nSlabs

            # A_0 is zero exactly when there is no left neighbour.
            # C_{nSlabs-2} is zero because S^+_{nSlabs-1} = 0, even though the
            # right neighbour exists -- the asymmetry is because the zeroed
            # SiM sits at an even index and the zeroed SiP at an odd one.
            A_is_zero = (not cyclic) and i == 0
            C_is_zero = (not cyclic) and i == nSlabs - 2

            # T_i Om and T_i^T Psi are Om and Psi themselves when T_i = I.
            # The updates below are out-of-place, so no copy is needed here.
            if _is_id(T[i]):
                Y_B, Z_B = Om, Psi
                self.nIdSkipped += 1
            else:
                Y_B = self._ap(T[i], Om)
                Z_B = self._apT(T[i], Psi)
            Y_A = Y_C = Z_A = Z_C = None

            if has_right:
                # forward: one apply of S^+_i covering both B and C
                cols = [Xm[kR]] if C_is_zero else [Xm[kR], Xp[kR]]
                W = self._ap(SiP[i], cols[0] if len(cols) == 1
                             else np.concatenate(cols, axis=1))
                Y_B = Y_B - W[:, :s]
                if not C_is_zero:
                    Y_C = -W[:, s:]

                # adjoint: t^+ = T_{i+1}^{-T} (S^+_i)^T Psi serves B and C
                rhs_p = self._apT(SiP[i], Psi)
                if _is_id(T_hbs[kR]):
                    tp = rhs_p
                    self.nIdSkipped += 1
                else:
                    tp = self._sv(T_hbs[kR], rhs_p, mode='T')
                Z_B = Z_B - self._apT(SiM[kR], tp)
                if not C_is_zero:
                    Z_C = -self._apT(SiP[kR], tp)

            if has_left:
                # Xp first (B term), Xm second (A term)
                W = self._ap(SiM[i], np.concatenate([Xp[kL], Xm[kL]], axis=1))
                Y_B = Y_B - W[:, :s]
                Y_A = -W[:, s:]

                # adjoint: t^- = T_{i-1}^{-T} (S^-_i)^T Psi serves B and A
                rhs_m = self._apT(SiM[i], Psi)
                if _is_id(T_hbs[kL]):
                    tm = rhs_m
                    self.nIdSkipped += 1
                else:
                    tm = self._sv(T_hbs[kL], rhs_m, mode='T')
                Z_B = Z_B - self._apT(SiP[kL], tm)
                Z_A = -self._apT(SiM[kL], tm)

            # Guard the degenerate case where neither branch ran: Y_B/Z_B
            # would still alias Om/Psi, which construct would then receive as
            # both the test matrix and its own samples.
            if Y_B is Om:
                Y_B = Om.copy()
            if Z_B is Psi:
                Z_B = Psi.copy()

            # ---- compress from the shared samples ----------------------
            need_ULV = self._needs_ulv(i, nSlabs)

            if need_ULV or self.compress_diag or not self.skip_unused_ulv:
                B_hbs = self._hbs_from_samples(rk, Om, Psi, Y_B, Z_B,
                                               compute_ULV=need_ULV,
                                               label=f"B[{i}] (nSlabs={nSlabs})")
            else:
                # compress_diag=False hands the uncompressed RB_linop to the
                # next level as T[i], so this slot is read by nobody: skip
                # the compression itself, not just the factorization.
                B_hbs = self._dead_diag(f"B[{i}] (nSlabs={nSlabs})")
            T_hbs_new.append(B_hbs)

            if self.compress_diag:
                B_i.append(B_hbs)
            else:
                spm = SiP[kL] if has_left  else None
                smp = SiM[kR] if has_right else None
                tmo = T_hbs[kL] if has_left  else None
                tpo = T_hbs[kR] if has_right else None
                B_i.append(RB_linop(T[i], tmo, tpo, SiP[i], SiM[i], smp, spm))

            # A_i and C_i become SiM / SiP one level down and are only ever
            # applied, never solved with -- no ULV, unconditionally.
            A_i.append(zero_op(m, dtype) if A_is_zero
                       else self._hbs_from_samples(rk, Om, Psi, Y_A, Z_A,
                                                   compute_ULV=False,
                                                   label=f"A[{i}] (nSlabs={nSlabs})"))
            C_i.append(zero_op(m, dtype) if C_is_zero
                       else self._hbs_from_samples(rk, Om, Psi, Y_C, Z_C,
                                                   compute_ULV=False,
                                                   label=f"C[{i}] (nSlabs={nSlabs})"))

        return (A_i, B_i, T_hbs_new, C_i)

    # ------------------------------------------------------------------
    # _build_level  -- original one-operator-at-a-time path (fused=False)
    # ------------------------------------------------------------------

    def _build_level(self, m, nSlabs, RB_level, rk):
        SiM   = RB_level[0]
        T     = RB_level[1]
        T_hbs = RB_level[2]
        SiP   = RB_level[3]

        cyclic = self.cyclic
        dtype  = self._dtype

        B_i       = []
        T_hbs_new = []

        for i in range(0, nSlabs, 2):
            spm = SiP[(i - 1) % nSlabs] if ((i > 0) or cyclic) else None
            smp = SiM[(i + 1) % nSlabs] if ((i < nSlabs - 1) or cyclic) else None
            tm  = T_hbs[(i - 1) % nSlabs] if spm is not None else None
            tp  = T_hbs[(i + 1) % nSlabs] if smp is not None else None

            need_ULV = self._needs_ulv(i, nSlabs)
            B_linop  = RB_linop(T[i], tm, tp, SiP[i], SiM[i], smp, spm)

            if need_ULV or self.compress_diag or not self.skip_unused_ulv:
                B_hbs = self._hbs(B_linop, rk, compute_ULV=need_ULV,
                                  label=f"B[{i}] (nSlabs={nSlabs})")
            else:
                B_hbs = self._dead_diag(f"B[{i}] (nSlabs={nSlabs})")

            B_i.append(B_hbs if self.compress_diag else B_linop)
            T_hbs_new.append(B_hbs)

        A_i = []
        for i in range(0, nSlabs, 2):
            if (not cyclic) and i == 0:
                A_i.append(zero_op(m, dtype))
            else:
                A_i.append(self._hbs(
                    STS_linop(SiM[i], T_hbs[(i - 1) % nSlabs],
                              SiM[(i - 1) % nSlabs]), rk,
                    compute_ULV=False, label=f"A[{i}] (nSlabs={nSlabs})"))

        C_i = []
        for i in range(0, nSlabs, 2):
            if (not cyclic) and i == nSlabs - 2:
                C_i.append(zero_op(m, dtype))
            else:
                C_i.append(self._hbs(
                    STS_linop(SiP[i], T_hbs[(i + 1) % nSlabs],
                              SiP[(i + 1) % nSlabs]), rk,
                    compute_ULV=False, label=f"C[{i}] (nSlabs={nSlabs})"))

        return (A_i, B_i, T_hbs_new, C_i)

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(self, rhs):
        m  = self.m
        RB = self.RB

        # ---- forward reduction ----------------------------------------
        vPrimes = [rhs.copy()]
        dtype   = np.result_type(np.asarray(rhs).dtype, self._dtype)

        for l in range(len(RB) - 1):
            SiM, _, T_hbs, SiP = RB[l]

            nSlabs   = len(SiM)
            nReduced = nSlabs // 2
            vPrev    = vPrimes[-1]
            vPrime   = np.zeros(m * nReduced, dtype=dtype)

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
        vPrimes[-1] = RB[-1][2][0].solve(vPrimes[-1])

        # ---- back substitution ----------------------------------------
        for l in range(len(RB) - 1, 0, -1):
            SiM, _, T_hbs, SiP = RB[l - 1]

            nSlabs   = len(SiM)
            nReduced = nSlabs // 2

            for j in range(nReduced):
                i = 2 * j

                vPrimes[l-1][i*m:(i+1)*m] = vPrimes[l][j*m:(j+1)*m]

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
