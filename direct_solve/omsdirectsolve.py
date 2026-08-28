import numpy as np
from scipy.sparse.linalg   import LinearOperator
from scipy.linalg   import lu_factor, lu_solve, block_diag
import time
import pickle

from abc import ABC, abstractmethod
from dataclasses import dataclass

import os
np.set_printoptions(linewidth=200)
np.set_printoptions(threshold=10000)

"""
Dense (uncompressed) counterpart of omsdirectsolveHBS.py.

The two modules are kept structurally identical so a driver can swap one solver
for the other: same class layout, method names, recurrences and index
conventions.  The only difference is what plays the role of a factorized
diagonal block -- HBSnew.HBSMAT(...).construct(rk, compute_ULV=True) there,
lu_op(...) here.  Both expose .solve(v, mode='N'|'T') and the LinearOperator
apply interface, so the call sites read the same.

The HBS module does `from direct_solve.omsdirectsolve import DirectSolver`, so
nothing here may import from it; the shared operator helpers are therefore
duplicated in both files rather than imported.
"""


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
    def __init__(self, n, dtype=np.float64):
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


def _is_id(op):
    """True if `op` is a structural identity, i.e. an id_op instance.

    Detection is structural rather than semantic, matching the HBS module.
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


class lu_op(LinearOperator):
    """Dense LU factorization of an operator -- the dense stand-in for HBSMAT.

    Accepts a dense array or any LinearOperator; a LinearOperator is
    materialized with one matmat against the identity, which is the same m
    applies the old `lu_solve(B[-1], I)` / `reconstruct_from_lu_factor` path
    cost.  `solve` carries the mode='N' | 'T' convention used by HBSMAT.solve,
    so call sites here and in the HBS module are textually the same.
    """

    def __init__(self, op, dtype=None):
        if isinstance(op, np.ndarray):
            A = np.array(op, copy=True)
        else:
            dt = getattr(op, "dtype", np.float64) if dtype is None else dtype
            A  = np.asarray(op.matmat(np.eye(op.shape[1], dtype=dt)))

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"lu_op requires a square operator, got shape {A.shape}.")

        super().__init__(shape=A.shape, dtype=A.dtype)
        self._A   = A
        self._lu  = lu_factor(A, overwrite_a=False)
        self.tree = getattr(op, "tree", None)
        self.quad = getattr(op, "quad", None)

    def _matvec(self, v):  return self._A @ v
    def _matmat(self, V):  return self._A @ V
    def _rmatvec(self, v): return self._A.T @ v
    def _rmatmat(self, V): return self._A.T @ V

    def solve(self, v, mode='N'):
        return lu_solve(self._lu, v, trans=0 if mode == 'N' else 1)

    def construct(self, *args, **kwargs):
        """No-op, for API parity with HBSMAT.construct; the LU is already built."""
        return self

    @property
    def dense(self):
        return self._A


def _densify_block(op):
    """Materialize an operator as a dense array, cheaply when possible."""
    if op is None:
        return None
    if isinstance(op, np.ndarray):
        return np.array(op, copy=True)
    if hasattr(op, "dense"):            # lu_op keeps its matrix already
        return np.array(op.dense, copy=True)
    I = np.eye(op.shape[1], dtype=op.dtype)
    return np.asarray(op.matmat(I))


def _as_op(x):
    """Coerce a raw numpy block into a LinearOperator, leave operators alone.

    Existing dense call sites hand these routines plain m x m arrays while the
    HBS call sites hand them HBS operators; ingesting through this keeps both
    working without branching further down.
    """
    return lu_op(x) if isinstance(x, np.ndarray) else x


def _as_solvable(op):
    """Return something exposing .solve, factorizing densely if it does not."""
    return op if hasattr(op, "solve") else lu_op(op)


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


def Sprime_Linop(Sl, Sprime_prev, Sr, id=False):
    if id:
        def smatmat(v, transpose=False):
            if (v.ndim == 1):
                v_tmp = v[:, np.newaxis]
            else:
                v_tmp = v

            if (not transpose):
                result = v_tmp - Sl.matmat(Sr.matmat(v_tmp))
            else:
                result = v_tmp - Sr.rmatmat(Sl.rmatmat(v_tmp))
            if (v.ndim == 1):
                result = result.flatten()
            return result

    else:
        def smatmat(v, transpose=False):
            if (v.ndim == 1):
                v_tmp = v[:, np.newaxis]
            else:
                v_tmp = v

            if (not transpose):
                result = v_tmp - Sl.matmat(Sprime_prev.solve(Sr.matmat(v_tmp)))
            else:
                result = v_tmp - Sr.rmatmat(Sprime_prev.solve(Sl.rmatmat(v_tmp), mode='T'))
            if (v.ndim == 1):
                result = result.flatten()
            return result
    Sprime = LinearOperator(shape=(Sl.shape[0], Sr.shape[1]),
        dtype  = _rdtype(Sl, Sprime_prev, Sr),
        matvec = lambda v: smatmat(v), rmatvec = lambda v: smatmat(v, transpose=True),
        matmat = lambda v: smatmat(v), rmatmat = lambda v: smatmat(v, transpose=True))
    return Sprime


def Dprime_Linop(D, A, B, Dprev):
    def dmatmat(v, transpose=False):
        if (v.ndim == 1):
            v_tmp = v[:, np.newaxis]
        else:
            v_tmp = v

        if (not transpose):
            result = D.matmat(v_tmp) - A.matmat(Dprev.solve(B.matmat(v_tmp)))
        else:
            result = D.rmatmat(v_tmp) - B.rmatmat(Dprev.solve(A.rmatmat(v_tmp), mode='T'))
        if (v.ndim == 1):
            result = result.flatten()
        return result
    Dprime = LinearOperator(shape=(D.shape[0], D.shape[1]),
        dtype  = _rdtype(D, A, B, Dprev),
        matvec = lambda v: dmatmat(v), rmatvec = lambda v: dmatmat(v, transpose=True),
        matmat = lambda v: dmatmat(v), rmatmat = lambda v: dmatmat(v, transpose=True))
    return Dprime


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """One intermediate block of a factorization, kept for offline study.

    kind separates blocks worth compressing ('general') from the structural
    identities and zeros, which are recorded without a dense copy.
    """
    level: int
    node:  int
    role:  str            # 'sub' | 'diag' | 'super'
    kind:  str            # 'general' | 'identity' | 'zero'
    shape: tuple
    dense: object = None

    @property
    def key(self):
        """Identity of the slot this block occupies."""
        return (self.level, self.node, self.role)


class DirectSolver(ABC):

    def __init__(self, m, cyclic=False, save_levels=False):
        self.m = m
        self.cyclic = cyclic
        # Set here rather than in factorize: ThomasSolverHBS.solve reads this
        # whenever cyclic=True but its factorize never writes it, so the
        # attribute has to exist from construction and be filled by build_smw.
        self.smw_block = None
        # Keep a dense copy of every intermediate block produced during the
        # factorization, for offline studies (e.g. HBS rank growth).  Off by
        # default: the log costs one m x m array per block per level.
        self.save_levels = save_levels
        self._blocks     = None
        # Operator dtype, set during factorize; used to promote a right-hand
        # side so a real rhs against complex operators does not silently cast.
        self._dtype      = np.float64

    # ------------------------------------------------------------------
    # right-hand side handling
    # ------------------------------------------------------------------

    @staticmethod
    def _as_2d(rhs):
        """Present any rhs as (N, nrhs), reporting whether it arrived flat."""
        r = np.asarray(rhs)
        if r.ndim == 1:
            return r[:, np.newaxis], True
        if r.ndim != 2:
            raise ValueError(f"rhs must be 1-D or 2-D, got shape {r.shape}.")
        return r, False

    def _work_dtype(self, rhs):
        return np.result_type(np.asarray(rhs).dtype, self._dtype)

    @abstractmethod
    def factorize(self, S_rk_list, T=None):
        """Need to implement"""
        pass

    @abstractmethod
    def solve(self, rhs, glob_target_dofs=None):
        """Need to implement"""
        pass

    # ------------------------------------------------------------------
    # intermediate block log
    # ------------------------------------------------------------------

    def _init_block_log(self):
        self._blocks = []

    def _log_block(self, level, node, role, op):
        """Record one intermediate block.

        role is 'sub' | 'diag' | 'super'.
        """
        if not self.save_levels or op is None:
            return
        if isinstance(op, zero_op):
            kind = 'zero'
        elif _is_id(op):
            kind = 'identity'
        else:
            kind = 'general'
        self._blocks.append(Block(
            level = level,
            node  = node,
            role  = role,
            kind  = kind,
            shape = tuple(op.shape),
            dense = _densify_block(op) if kind == 'general' else None,
        ))

    def get_blocks(self, level=None, node=None, role=None, kind='general'):
        """Filtered view of the intermediate block log.

        Defaults to the 'general' blocks, i.e. those an HBS study would need to
        compress.  Pass kind=None to see the structural identities and zeros too.
        """
        if self._blocks is None:
            raise RuntimeError(
                "No block log: construct the solver with save_levels=True.")
        out = self._blocks
        if level is not None:
            out = [b for b in out if b.level == level]
        if node is not None:
            out = [b for b in out if b.node == node]
        if role is not None:
            out = [b for b in out if b.role == role]
        if kind is not None:
            out = [b for b in out if b.kind == kind]
        return out

    @property
    def n_levels(self):
        if self._blocks is None:
            raise RuntimeError(
                "No block log: construct the solver with save_levels=True.")
        return 1 + max(b.level for b in self._blocks)

    def get_block_op(self, level, node, role):
        """Live operator currently occupying a slot (see set_block_op)."""
        raise NotImplementedError

    def set_block_op(self, level, node, role, op):
        """Swap the live operator in a slot, returning the previous occupant.

        Used to substitute a compressed block into an otherwise exact solver and
        measure the effect on the final solution.  Only the slots the solve
        actually reads are touched, so the factorization is NOT rebuilt: this
        measures a block's direct contribution to the solve, not the error it
        would have propagated into later levels had it been compressed during
        the factorization.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # cyclic (periodic) correction
    # ------------------------------------------------------------------

    def build_smw(self, S_rk_list, glob_target_dofs=None):
        """
        [ B0 ] [ C0 ] [ 0  ] [ E  ]
        [ A1 ] [ B1 ] [ C1 ] [ 0  ]
        [ 0  ] [ A2 ] [ B2 ] [ C2 ]
        [ F  ] [ 0  ] [ A3 ] [ B3 ]

        Writes the cyclic system as T + U V', with T block-tridiagonal and U V'
        the rank-2m corner modification (E and F).  Given the tridiagonal
        factorization already stored on self, this forms

            Z         = T \\ U
            smw_block = Z (I + V' Z)^{-1} V'

        so that solve can finish with  x = y - smw_block @ y,  y = T \\ d.

        Lives on the base class so both the dense and HBS Thomas solvers reach
        it identically; the red-black reduction handles periodicity natively
        (its level indexing is already modular) and does not use this.

        Call this after factorize whenever cyclic=True.
        """
        m = self.m
        E = _as_op(S_rk_list[0][0])
        F = _as_op(S_rk_list[-1][-1])
        n = len(S_rk_list)

        dtype = _rdtype(E, F)
        I = np.eye(m, dtype=dtype)
        U = np.zeros((n*m, 2*m), dtype=dtype)
        V = np.zeros((n*m, 2*m), dtype=dtype)

        if glob_target_dofs is None:
            indices = [np.arange(l*m, (l+1)*m) for l in range(n)]
        else:
            indices = [np.asarray(idx) for idx in glob_target_dofs]

        first, last = indices[0], indices[n-1]
        lo, hi      = np.arange(0, m), np.arange(m, 2*m)

        U[np.ix_(first, lo)] = np.asarray(E.matmat(I))
        U[np.ix_(last,  hi)] = np.asarray(F.matmat(I))

        V[np.ix_(last,  lo)] = I
        V[np.ix_(first, hi)] = I

        Z = self.solve_helper(U, glob_target_dofs)

        self.smw_block = Z @ np.linalg.solve(np.eye(2*m, dtype=dtype) + V.T @ Z, V.T)

        return self.smw_block


def reconstruct_from_lu_factor(Tinv):
    """Unused since lu_op keeps its dense matrix; kept for existing call sites."""
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


'''

Fredholm second kind Block Tridiagonal (BTD) solver, dense LU
Uses that the diagonal is identity

'''

class ThomasSolver(DirectSolver):

    def __init__(self, m, cyclic=False, save_levels=False):
        super().__init__(m, cyclic, save_levels)
        self.solve_method = None

    def factorize_helper(self, S_rk_list, diagList=None):
        if diagList == None:
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
        b'_{i+1} = b_{i+1}-S_{i+1,i}\\S'_{i}\\(b'_i)
        ---------------------------------------------------------------------------
        NOTE: different sign convention on S is possible, in this case, recurrence changes slightly

        Note the change from the old A_i' = A_i (B_{i-1}')^-1 formulation: the
        off-diagonals are left untouched here and (S'_{i-1})^-1 is applied
        during the forward sweep instead, matching ThomasSolverHBS.
        """
        m = S_rk_list[0][0].shape[0]
        n = len(S_rk_list) - 1 # Accounts for E and F in periodic case
        I = id_op(m, S_rk_list[0][0].dtype)
        self._dtype = S_rk_list[0][0].dtype
        Sl = [_as_op(S_rk_list[_][0])  for _ in range(1, n+1)]
        Sprime = [I]
        Sr = [_as_op(S_rk_list[_][-1]) for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)
        for i in range(1, n+1):
            if i == 1:
                Sprime_i = lu_op(Sprime_Linop(Sl[0], I, Sr[0], id=True))
            else:
                Sprime_i = lu_op(Sprime_Linop(Sl[i-1], Sprime[i-1], Sr[i-1]))
            Sprime.append(Sprime_i)
        self.A = Sl
        self.B = Sprime
        self.C = Sr

        if self.save_levels:
            self._init_block_log()
            self._log_block_thomas()

    def factorize_with_diag(self, AB_list, D_list):
        """

        [ D0 ] [ B0 ] [ 00 ] [ 00 ]
        [ A0 ] [ D1 ] [ B1 ] [ 00 ]
        [ 00 ] [ A1 ] [ D2 ] [ B2 ]
        [ 00 ] [ 00 ] [ A2 ] [ D3 ]

        NOTE ON INDEXING: A is read as AB_list[0..n-1][0], i.e. the sub-diagonal
        of row i is AB_list[i-1][0].  The identity-diagonal path above instead
        reads AB_list[1..n][0].  This mirrors ThomasSolverHBS exactly -- the two
        paths there disagree by one index -- so the same S_rk_list cannot be fed
        to both without shifting.  Flagged, not silently reconciled.
        """
        D_list = [_as_op(_) for _ in D_list]
        m = D_list[0].shape[0]
        n = len(D_list) - 1
        self._dtype = _rdtype(*D_list)

        # Thus we need three lists of block matrices: A, B, and C:
        A = [_as_op(AB_list[_][0])  for _ in range(n)]
        # D_0 is only ever used through .solve, so it is factorized up front
        # unless the caller already handed over something that can solve.
        B = [_as_solvable(D_list[0])]
        C = [_as_op(AB_list[_][-1]) for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)

        for i in range(1, n+1):
            B_i = lu_op(Dprime_Linop(D_list[i], A[i-1], C[i-1], B[-1]))
            B.append(B_i)

        self.A = A
        self.B = B
        self.C = C

        if self.save_levels:
            self._init_block_log()
            self._log_block_thomas()

    # ------------------------------------------------------------------
    # intermediate block access
    # ------------------------------------------------------------------

    def _log_block_thomas(self):
        """Log the sweep.

        `level` is the elimination step: the diagonal S'_i (or D'_i) produced at
        step i is logged as level i, node i.  The off-diagonals are inputs the
        sweep never modifies, so they are logged once at level 0, node i being
        the row they sit in (A[i-1] is the sub-diagonal of row i, C[i] the
        super-diagonal).
        """
        for i, op in enumerate(self.B):
            self._log_block(i, i, 'diag', op)
        for i, op in enumerate(self.A):
            self._log_block(0, i + 1, 'sub', op)
        for i, op in enumerate(self.C):
            self._log_block(0, i, 'super', op)

    def get_block_op(self, level, node, role):
        if role == 'diag':
            return self.B[level]
        if role == 'sub':
            return self.A[node - 1]
        if role == 'super':
            return self.C[node]
        raise ValueError(f"unknown role {role!r}")

    def set_block_op(self, level, node, role, op):
        prev = self.get_block_op(level, node, role)
        if role == 'diag':
            self.B[level] = op
        elif role == 'sub':
            self.A[node - 1] = op
        elif role == 'super':
            self.C[node] = op
        else:
            raise ValueError(f"unknown role {role!r}")
        return prev

    def solve_helper(self, rhs, glob_target_dofs=None):
        if self.solve_method == 'id_diag':
            return self.solve_id_diag(rhs, glob_target_dofs)
        elif self.solve_method == 'diag':
            return self.solve_with_diag(rhs, glob_target_dofs)
        else:
            raise ValueError('Factorization not set')

    def solve_id_diag(self, rhs, glob_target_dofs=None):
        """
        Given precomputed factors and a RHS d, this solves Tx = d.

        d may be a vector or an (N, nrhs) matrix; the shape it arrived in is the
        shape it comes back in.
        """
        m       = self.m
        Sl      = self.A
        Sprime  = self.B
        Sr      = self.C
        n       = len(Sl)

        d, was_1d = self._as_2d(rhs)
        d         = d.astype(self._work_dtype(d), copy=True)

        if glob_target_dofs is None:
            indices = [range(l*m, (l+1)*m) for l in range(len(Sprime))]
        else:
            indices = glob_target_dofs

        #
        # For i = 1 to n, we have d_i = d_i - S_{i,i-1} (S'_{i-1} \ d_i-1)
        #
        for i in range(1, n+1):
            if i == 1:
                d[indices[i], :] = d[indices[i], :] - Sl[i-1] @ d[indices[i-1], :]
            else:
                d[indices[i], :] = d[indices[i], :] - Sl[i-1] @ (Sprime[i-1].solve(d[indices[i-1], :]))

        #
        # Then for i = n-1 to 0, we have x_n = S'_n \ d_n,   x_i = S'_i \ (d_i - S_{i,i+1} x_i+1)
        #
        x                = np.zeros(d.shape, dtype=d.dtype)
        x[indices[n], :] = Sprime[n].solve(d[indices[n], :])
        for i in range(n-1, 0, -1):
            x[indices[i], :] = Sprime[i].solve(d[indices[i], :] - Sr[i] @ x[indices[i+1], :])

        # Since S'_0 is the identity, we can avoid a solve there:
        x[indices[0], :] = d[indices[0], :] - Sr[0] @ x[indices[1], :]
        if was_1d:
            x = x.flatten()
        return x

    def solve_with_diag(self, rhs, glob_target_dofs=None):
        """
        As solve_id_diag, but for a general (non-identity) diagonal.

        d may be a vector or an (N, nrhs) matrix.
        """
        m = self.m
        A = self.A
        B = self.B
        C = self.C
        n = len(A)

        d, was_1d = self._as_2d(rhs)
        d         = d.astype(self._work_dtype(d), copy=True)

        if glob_target_dofs is None:
            indices = [range(l*m, (l+1)*m) for l in range(len(B))]
        else:
            indices = glob_target_dofs

        for i in range(1, n+1):
            d[indices[i], :] = d[indices[i], :] - A[i-1] @ B[i-1].solve(d[indices[i-1], :])

        x                = np.zeros(d.shape, dtype=d.dtype)
        x[indices[n], :] = B[n].solve(d[indices[n], :])

        for i in range(n-1, 0, -1):
            x[indices[i], :] = B[i].solve(d[indices[i], :] - C[i] @ x[indices[i+1], :])

        # But not this time:
        x[indices[0], :] = B[0].solve(d[indices[0], :] - C[0] @ x[indices[1], :])

        if was_1d:
            x = x.flatten()
        return x

    def factorize(self, S_rk_list, T=None):
        self.factorize_helper(S_rk_list, T)

    def solve(self, rhs, glob_target_dofs=None):
        x = self.solve_helper(rhs, glob_target_dofs)
        if self.cyclic:
            x = x - self.smw_block @ x

        return x


# Old name, kept so existing call sites keep importing.
BlockTridiagonalSolver = ThomasSolver


# ---------------------------------------------------------------------------
# Dense Red-Black solver
# ---------------------------------------------------------------------------

class RedBlackSolver(DirectSolver):
    """
    Block-tridiagonal solver using cyclic reduction (red-black), dense LU.

    Level structure matches RedBlackSolverHBS position for position:
      RB[l] = (SiM, T, T_fac, SiP)  -- all four lists of length nSlabs_at_level
      SiM[i]   : left  off-diagonal at node i
      T[i]     : diagonal LinearOperator at node i
      T_fac[i] : LU factorization of T[i]  (where HBS keeps T_hbs[i])
      SiP[i]   : right off-diagonal at node i

    The three operators produced at each retained node,

        B_i = T_i - S^+_i T_{i+1}^{-1} S^-_{i+1} - S^-_i T_{i-1}^{-1} S^+_{i-1}
        A_i =     - S^-_i T_{i-1}^{-1} S^-_{i-1}
        C_i =     - S^+_i T_{i+1}^{-1} S^+_{i+1}

    are formed through the same RB_linop / STS_linop wrappers the HBS solver
    samples, then materialized column-by-column against the identity instead of
    being compressed.  There is no dense analogue of _build_level_fused: the
    sharing it does is a sampling optimization, and exact materialization has
    nothing to share.

    Boundary blocks are zero_op rather than np.zeros((m,m)) so indexing stays
    uniform, and an omitted diagonal defaults to id_op, both as in the HBS
    version.
    """

    def __init__(self, m, cyclic=False, compress_diag=True, seed=0,
                 identity_diag=None, save_levels=False):
        super().__init__(m, cyclic, save_levels)
        # Densely, compress_diag selects whether the Schur diagonal carried to
        # the next level is the materialized LU (True) or the lazy RB_linop
        # (False).  Name kept for parity with RedBlackSolverHBS.
        self.compress_diag = compress_diag
        # How to treat a user-supplied diagonal list T:
        #   None  -- probe each entry and swap exact identities for id_op
        #   True  -- assert every entry is the identity and swap unconditionally
        #   False -- leave T exactly as given
        self.identity_diag = identity_diag
        self._dtype = np.float64
        self._rng   = np.random.default_rng(seed)
        # bookkeeping, same names as the HBS solver
        self.nConstruct = 0
        self.nSolve     = 0
        self.nApply     = 0
        self.nIdSkipped = 0

    # ------------------------------------------------------------------

    def _factor(self, A):
        """Dense LU of a materialized block; the counterpart of _hbs."""
        self.nConstruct += 1
        return lu_op(A)

    def _densify(self, op):
        """Materialize an operator against the identity."""
        self.nApply += 1
        I = np.eye(op.shape[1], dtype=op.dtype)
        return np.asarray(op.matmat(I))

    def _normalize_diag(self, T, m):
        """Swap identity diagonals for id_op so the fast paths engage.

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
        """

        [  I   ] [ S_12 ] [  0   ] [  E?  ]
        [ S_21 ] [  I   ] [ S_23 ] [  0   ]
        [  0   ] [ S_32 ] [  I   ] [ S_34 ]
        [  F?  ] [  0   ] [ S_43 ] [  I   ]

        Red-black (cyclic reduction) factorization. ASSUME POWER OF 2 FOR SLABCOUNT ONLY
        """
        m      = S_rk_list[0][0].shape[0]
        nSlabs = len(S_rk_list)

        if not ((nSlabs & (nSlabs-1) == 0) and nSlabs != 0):
            raise ValueError("Number of slabs must be a power of 2.")

        self._dtype = S_rk_list[0][0].dtype

        SiM = [_as_op(_[0])  for _ in S_rk_list]
        SiP = [_as_op(_[-1]) for _ in S_rk_list]

        # Boundary zeros -- kept as zero LinearOperators so indexing is uniform.
        if not self.cyclic:
            SiM[0]  = zero_op(m, self._dtype)
            SiP[-1] = zero_op(m, self._dtype)

        if T is None:
            T = [id_op(m, self._dtype) for _ in range(nSlabs)]
        else:
            T = self._normalize_diag([_as_op(_) for _ in T], m)
        # At level 0 the diagonals are used as given; _as_solvable only has work
        # to do if a caller passed a bare LinearOperator with no .solve.
        T_fac = [_as_solvable(_) for _ in T]

        RB = [(SiM, T, T_fac, SiP)]

        l = nSlabs
        while l > 1:
            RB.append(self._build_level(m, l, RB[-1]))
            l //= 2

        self.nSlabs = nSlabs
        self.RB     = RB

        if self.save_levels:
            self._init_block_log()
            for l, (SiM_l, _, T_fac_l, SiP_l) in enumerate(RB):
                for i in range(len(SiM_l)):
                    self._log_block(l, i, 'sub',   SiM_l[i])
                    self._log_block(l, i, 'diag',  T_fac_l[i])
                    self._log_block(l, i, 'super', SiP_l[i])

    # ------------------------------------------------------------------
    # intermediate block access
    # ------------------------------------------------------------------

    # Which slot of RB[l] = (SiM, T, T_fac, SiP) the solve actually reads.
    # The diagonal is read through T_fac; slot 1 (T) is consumed during
    # factorization only and is deliberately left alone.
    _ROLE_SLOT = {'sub': 0, 'diag': 2, 'super': 3}

    def get_block_op(self, level, node, role):
        return self.RB[level][self._ROLE_SLOT[role]][node]

    def set_block_op(self, level, node, role, op):
        slot = self._ROLE_SLOT[role]
        prev = self.RB[level][slot][node]
        self.RB[level][slot][node] = op
        return prev

    # ------------------------------------------------------------------
    # _build_level
    # ------------------------------------------------------------------

    def _build_level(self, m, nSlabs, RB_level):
        """
        [  T_1 ] [ S_12 ] [  0   ] [  E?  ]
        [ S_21 ] [  T_2 ] [ S_23 ] [  0   ]
        [  0   ] [ S_32 ] [  T_3 ] [ S_34 ]
        [  F?  ] [  0   ] [ S_43 ] [  T_4 ]

        Eliminates the odd nodes and returns (A_i, B_i, T_fac_new, C_i) for the
        retained even ones, which become the next level's off-diagonals and
        diagonal.
        """
        SiM   = RB_level[0]
        T     = RB_level[1]
        T_fac = RB_level[2]
        SiP   = RB_level[3]

        cyclic = self.cyclic
        dtype  = self._dtype

        B_linops = []
        B_dense  = []

        for i in range(0, nSlabs, 2):
            spm = SiP[(i-1) % nSlabs] if ((i > 0) or cyclic) else None
            smp = SiM[(i+1) % nSlabs] if ((i < nSlabs-1) or cyclic) else None
            tm  = T_fac[(i-1) % nSlabs] if spm is not None else None
            tp  = T_fac[(i+1) % nSlabs] if smp is not None else None

            B_linop = RB_linop(T[i], tm, tp, SiP[i], SiM[i], smp, spm)
            B_linops.append(B_linop)
            B_dense.append(self._densify(B_linop))

        A_dense = []
        for i in range(0, nSlabs, 2):
            if (not cyclic) and i == 0:
                A_dense.append(None)
            else:
                A_dense.append(self._densify(
                    STS_linop(SiM[i], T_fac[(i-1) % nSlabs],
                              SiM[(i-1) % nSlabs])))

        C_dense = []
        for i in range(0, nSlabs, 2):
            if (not cyclic) and i == nSlabs - 2:
                C_dense.append(None)
            else:
                C_dense.append(self._densify(
                    STS_linop(SiP[i], T_fac[(i+1) % nSlabs],
                              SiP[(i+1) % nSlabs])))

        # We need to account for cyclic effects if this is the topmost layer:
        # with one node left its two wrap-around neighbours are itself, so A and
        # C fold into the diagonal.  Applied before the LU below.
        #
        # NOTE: RedBlackSolverHBS._build_level and _build_level_fused have no
        # counterpart to this, so the two solvers will disagree at the coarsest
        # level whenever cyclic=True.  Kept here because dropping it makes the
        # periodic solve wrong; see the accompanying notes.
        if len(B_dense) == 1 and cyclic:
            B_dense[0] = B_dense[0] + A_dense[0] + C_dense[0]

        T_fac_new = [self._factor(_) for _ in B_dense]
        B_i       = T_fac_new if self.compress_diag else B_linops

        A_i = [zero_op(m, dtype) if _ is None else dense_to_linop(_) for _ in A_dense]
        C_i = [zero_op(m, dtype) if _ is None else dense_to_linop(_) for _ in C_dense]

        return (A_i, B_i, T_fac_new, C_i)

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(self, rhs, glob_target_dofs=None):
        """
        Solves using the Red-Black factorization, rhs of length m * nSlabs.

        rhs may be a vector or an (m*nSlabs, nrhs) matrix; the shape it arrived
        in is the shape it comes back in.  All right-hand sides ride through the
        reduction together, so each diagonal solve and each off-diagonal apply
        happens once for the whole batch rather than once per column.
        """
        if glob_target_dofs is not None:
            raise NotImplementedError(
                "RedBlackSolver assumes contiguous block ordering; "
                "glob_target_dofs is only supported by ThomasSolver.")

        m  = self.m
        RB = self.RB

        V, was_1d = self._as_2d(rhs)
        dtype     = self._work_dtype(V)
        nrhs      = V.shape[1]

        # ---- forward reduction ----------------------------------------
        vPrimes = [V.astype(dtype, copy=True)]

        for l in range(len(RB) - 1):
            SiM, _, T_fac, SiP = RB[l]

            nSlabs   = len(SiM)
            nReduced = nSlabs // 2
            vPrev    = vPrimes[-1]
            vPrime   = np.zeros((m * nReduced, nrhs), dtype=dtype)

            for j in range(nReduced):
                i = 2 * j

                prev = (i - 1) % nSlabs if (self.cyclic or i > 0)          else None
                next = (i + 1) % nSlabs if (self.cyclic or i < nSlabs - 1) else None

                contrib = vPrev[i*m:(i+1)*m, :].copy()
                if prev is not None:
                    contrib -= SiM[i].matmat(
                        T_fac[prev].solve(vPrev[prev*m:(prev+1)*m, :]))
                if next is not None:
                    contrib -= SiP[i].matmat(
                        T_fac[next].solve(vPrev[next*m:(next+1)*m, :]))

                vPrime[j*m:(j+1)*m, :] = contrib

            vPrimes.append(vPrime)

        # ---- coarsest solve -------------------------------------------
        vPrimes[-1] = np.asarray(RB[-1][2][0].solve(vPrimes[-1]))

        # ---- back substitution ----------------------------------------
        for l in range(len(RB) - 1, 0, -1):
            SiM, _, T_fac, SiP = RB[l - 1]

            nSlabs   = len(SiM)
            nReduced = nSlabs // 2

            for j in range(nReduced):
                i = 2 * j

                vPrimes[l-1][i*m:(i+1)*m, :] = vPrimes[l][j*m:(j+1)*m, :]

                next_j  = (j + 1) % nReduced
                contrib = SiM[i+1].matmat(vPrimes[l][j*m:(j+1)*m, :])
                if self.cyclic or j + 1 < nReduced:
                    contrib = contrib + SiP[i+1].matmat(
                        vPrimes[l][next_j*m:(next_j+1)*m, :])

                vPrimes[l-1][(i+1)*m:(i+2)*m, :] -= contrib
                vPrimes[l-1][(i+1)*m:(i+2)*m, :] = T_fac[i+1].solve(
                    vPrimes[l-1][(i+1)*m:(i+2)*m, :])

        x = vPrimes[0]
        return x.flatten() if was_1d else x


# ---------------------------------------------------------------------------
# Nested dissection -- no HBS counterpart yet, left as-is
# ---------------------------------------------------------------------------

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