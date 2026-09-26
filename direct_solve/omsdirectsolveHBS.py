import numpy as np
from scipy.sparse.linalg   import LinearOperator
import jax.numpy as jnp
import matAssembly.HBS.HBStorch as HBSnew
from abc import ABC, abstractmethod
from direct_solve.omsdirectsolve import DirectSolver
import torch
import math
import warnings


# ---------------------------------------------------------------------------
# Per-stage compression rank (ported verbatim from the torch solver)
# ---------------------------------------------------------------------------

class rkStrat:
    """Per-stage HBS compression rank.

    A "stage" is one step of a factorization that compresses freshly formed
    operators from samples drawn through the operators of the previous stage.
    For RedBlackSolverHBS that is one builder call: stage j consumes RB[j] and
    produces RB[j+1], so rank(j) is the rank of everything in RB[j+1].  For a
    Thomas-style sweep the stage is the step index of the recurrence.  The
    class knows nothing about either; it maps a non-negative integer to a rank.

    Why a schedule at all
    ---------------------
    The forward operators compress to near machine accuracy at a modest rank,
    but each stage samples through the *inverse* of the previous stage's
    diagonal.  The Schur complements fill in, their off-diagonal blocks get
    less compressible, and a rank that was ample at the leaves is not ample at
    the root -- which is also the block that gets inverted directly and whose
    error lands straight in the residual.  Growing the rank with the stage
    puts the resolution where the loss happens.  It is close to free: the
    number of operators per stage halves while the rank grows by a factor g,
    so per-stage sampling cost scales like g/2 and compression like g^2/2.

    It cannot repair error already baked into the samples, though.  Stage j's
    right-hand sides are generated with stage j-1's *compressed* operators, so
    if stage j-1 was under-resolved, a higher rank at stage j only fits a
    wrong operator more precisely.  Growth should start early rather than jump
    at the end.

    Modes
    -----
      'const'   r(e) = rk0
      'linear'  r(e) = rk0 + rate*e        (rate = additive increment / stage)
      'geom'    r(e) = rk0 * rate**e       (rate = multiplicative factor)

    where e is the effective stage index (see skip_first_level).  Evaluation
    is always from this closed form, never by iterating r *= rate.  Iterated
    rounding compounds the rounding bias at every stage (upward here, since
    the rule below rounds up), so the schedule drifts off the intended curve:
    rk0=33, g=1.05 gives 33 35 36 38 40 42 44 46 from the closed form but
    33 35 37 39 41 43 45 47 when iterated.

    skip_first_level
    ----------------
    When the stage-0 diagonal is the identity there is no inversion at that
    stage, so its samples carry only the input operators' own compression
    error and no growth is warranted yet.  With skip_first_level=True
    (default) the growth index is shifted, e = max(stage-1, 0), so stages 0
    and 1 share the base rank and growth starts at stage 2.  This is only
    sound when stage 0 really is inversion-free; RedBlackSolverHBS checks
    that its level-0 diagonal is all id_op and copies the strategy with the
    flag off when it is not.  (In a Thomas sweep the analogue is the i == 1
    step, where S'_1 = I.)

    Rounding
    --------
    Half-up to an integer, then up to the next multiple of round_to.  Rounding
    up rather than to nearest means a schedule never silently gives back the
    resolution it was asked for; round_to > 1 keeps the GEMM shapes aligned
    and makes the schedule insensitive to small changes in rate.

    Bounds
    ------
    rk_min is enforced.  rk_max only warns -- exceeding it is a decision for
    the caller, not an error.  The binding practical limit is the HBS leaf
    size nl: once rk >= nl the leaf blocks have k = min(rk, nl) = n, the null
    complement W1 is empty and the leaf level compresses nothing.  Pass nl to
    validate() to get that warning for a whole schedule up front.
    """

    _MODES = ('const', 'linear', 'geom')

    def __init__(self, rk0, mode='const', rate=None, skip_first_level=True,
                 rk_max=None, rk_min=1, round_to=1, monotone=False):
        mode = str(mode).lower()
        aliases = {'constant': 'const', 'c': 'const',
                   'lin': 'linear', 'l': 'linear',
                   'geometric': 'geom', 'g': 'geom', 'exp': 'geom'}
        mode = aliases.get(mode, mode)
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}, got {mode!r}")

        rk0 = int(rk0)
        if rk0 < 1:
            raise ValueError(f"rk0 must be >= 1, got {rk0}")
        round_to = int(round_to)
        if round_to < 1:
            raise ValueError(f"round_to must be >= 1, got {round_to}")
        rk_min = int(rk_min)
        if rk_min < 1:
            raise ValueError(f"rk_min must be >= 1, got {rk_min}")
        if rk_max is not None:
            rk_max = int(rk_max)
            if rk_max < rk_min:
                raise ValueError(f"rk_max ({rk_max}) < rk_min ({rk_min})")

        if mode == 'const':
            if rate not in (None, 0):
                warnings.warn(f"rkStrat: mode='const' ignores rate={rate!r}",
                              UserWarning, stacklevel=2)
            rate = 0.0
        else:
            if rate is None:
                raise ValueError(f"mode={mode!r} requires a rate "
                                 "(increment per stage for 'linear', "
                                 "multiplicative factor for 'geom')")
            rate = float(rate)
            if mode == 'geom':
                if rate <= 0:
                    raise ValueError(f"geometric factor must be > 0, got {rate}")
                if rate < 1.0:
                    warnings.warn(
                        f"rkStrat: geometric factor {rate} < 1 gives a "
                        "DECREASING rank schedule. Rank normally has to grow "
                        "with the stage, because each stage samples through "
                        "the previous stage's inverse. Proceeding as asked.",
                        UserWarning, stacklevel=2)
            elif rate < 0:
                warnings.warn(
                    f"rkStrat: linear increment {rate} < 0 gives a DECREASING "
                    "rank schedule. Rank normally has to grow with the stage. "
                    "Proceeding as asked.", UserWarning, stacklevel=2)

        self.rk0   = rk0
        self.mode  = mode
        self.rate  = rate
        self.rk_max   = rk_max
        self.rk_min   = rk_min
        self.round_to = round_to
        self.monotone = bool(monotone)
        self._skip_first_level = bool(skip_first_level)
        self._cache  = {}
        self._warned = set()

    # -- constructors ---------------------------------------------------

    @classmethod
    def constant(cls, rk0, **kw):
        return cls(rk0, mode='const', **kw)

    @classmethod
    def linear(cls, rk0, inc, **kw):
        return cls(rk0, mode='linear', rate=inc, **kw)

    @classmethod
    def geometric(cls, rk0, factor, **kw):
        return cls(rk0, mode='geom', rate=factor, **kw)

    @classmethod
    def coerce(cls, rk):
        """Accept an rkStrat, or wrap a plain int as a constant schedule."""
        if isinstance(rk, cls):
            return rk
        return cls.constant(int(rk))

    def copy(self, **overrides):
        """Shallow copy with fields overridden; warning state is not carried."""
        kw = dict(rk0=self.rk0, mode=self.mode, rate=self.rate,
                  skip_first_level=self._skip_first_level,
                  rk_max=self.rk_max, rk_min=self.rk_min,
                  round_to=self.round_to, monotone=self.monotone)
        if self.mode == 'const':
            kw['rate'] = None
        kw.update(overrides)
        return rkStrat(**kw)

    # -- the flag, readable under either spelling ------------------------

    @property
    def skip_first_level(self):
        return self._skip_first_level

    @skip_first_level.setter
    def skip_first_level(self, v):
        if bool(v) != self._skip_first_level:
            self._skip_first_level = bool(v)
            self._cache.clear()

    # -- evaluation ------------------------------------------------------

    def _raw(self, stage):
        e = stage - 1 if self._skip_first_level else stage
        if e < 0:
            e = 0
        if self.mode == 'const':
            return float(self.rk0)
        if self.mode == 'linear':
            return self.rk0 + self.rate * e
        return self.rk0 * (self.rate ** e)

    def rank(self, stage):
        """Compression rank for `stage` (0-based)."""
        stage = int(stage)
        if stage < 0:
            raise ValueError(f"stage must be >= 0, got {stage}")
        if stage in self._cache:
            return self._cache[stage]

        r = int(math.floor(self._raw(stage) + 0.5))        # half-up
        if self.round_to > 1:                              # up to a multiple
            r = -(-r // self.round_to) * self.round_to
        if r < self.rk_min:
            r = self.rk_min
        if self.monotone and stage > 0:
            r = max(r, self.rank(stage - 1))

        if self.rk_max is not None and r > self.rk_max and stage not in self._warned:
            self._warned.add(stage)
            warnings.warn(
                f"rkStrat: rank {r} at stage {stage} exceeds rk_max="
                f"{self.rk_max}. Not clamped -- using {r}.",
                UserWarning, stacklevel=2)

        self._cache[stage] = r
        return r

    __call__ = rank

    def ranks(self, nstages):
        return [self.rank(j) for j in range(int(nstages))]

    # -- diagnostics -----------------------------------------------------

    def validate(self, nstages, nl=None, label=''):
        """Warn about a whole schedule up front. Returns the ranks."""
        rs = self.ranks(nstages)
        where = f" ({label})" if label else ''
        if nl is not None:
            nl = int(nl)
            bad = [(j, r) for j, r in enumerate(rs) if r >= nl]
            if bad:
                warnings.warn(
                    f"rkStrat{where}: rank reaches {bad[0][1]} at stage "
                    f"{bad[0][0]} (and at {len(bad)} stage(s) in total), which "
                    f"is >= the HBS leaf size nl={nl}. At the leaf level "
                    "k = min(rk, nl) = n, so the null complement is empty and "
                    "the leaf level compresses nothing; those blocks degrade "
                    "toward dense. Lower rk0 or the rate, or use a larger "
                    "leaf size.", UserWarning, stacklevel=2)
            elif any(2 * r > nl for r in rs):
                warnings.warn(
                    f"rkStrat{where}: rank reaches {max(rs)} against leaf size "
                    f"nl={nl}; above nl/2 the leaf compression saves little.",
                    UserWarning, stacklevel=2)
        return rs

    def describe(self, nstages=None):
        if self.mode == 'const':
            body = f"rk = {self.rk0}"
        elif self.mode == 'linear':
            body = f"rk = {self.rk0} + {self.rate:g}*e"
        else:
            body = f"rk = {self.rk0} * {self.rate:g}^e"
        bits = [body, f"e = stage{'-1' if self._skip_first_level else ''}"]
        if self.round_to > 1:
            bits.append(f"->mult of {self.round_to}")
        if self.rk_max is not None:
            bits.append(f"rk_max {self.rk_max} (warn only)")
        if self.monotone:
            bits.append("monotone")
        s = f"rkStrat[{self.mode}]: " + ", ".join(bits)
        if nstages:
            s += "  ->  " + " ".join(str(r) for r in self.ranks(nstages))
        return s

    def __repr__(self):
        return (f"rkStrat(rk0={self.rk0}, mode={self.mode!r}, rate={self.rate!r}, "
                f"skip_first_level={self._skip_first_level}, rk_max={self.rk_max}, "
                f"rk_min={self.rk_min}, round_to={self.round_to}, "
                f"monotone={self.monotone})")


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


def _leaf_size(tree):
    """Leaf size HBSMAT actually uses -- not tree._min_leaf_size.  None when
    the tree does not expose it (the schedule's leaf-size check is skipped)."""
    try:
        return len(tree.perm_leaf) // tree.nleaves
    except (AttributeError, TypeError, ZeroDivisionError):
        return None


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


def _load_diagnostics():
    """Imported lazily: hbs_diagnostics imports this module."""
    try:
        from . import hbs_diagnostics as diag
    except ImportError:
        import hbs_diagnostics as diag
    return diag


def _sum_linop(*ops):
    """LinearOperator for the sum of same-shape operators."""
    def mm(V):
        return sum(np.asarray(o.matmat(V)) for o in ops)
    def rmm(V):
        return sum(np.asarray(o.rmatmat(V)) for o in ops)
    return LinearOperator(
        shape   = ops[0].shape,
        dtype   = _rdtype(*ops),
        matvec  = lambda v: mm(v.reshape(-1, 1)).ravel(),
        rmatvec = lambda v: rmm(v.reshape(-1, 1)).ravel(),
        matmat  = mm,
        rmatmat = rmm,
    )


'''

Fredholm second kind Block Tridiagonal (BTD) solver using HBS acceleration
Uses that the diagonal is identity

'''

class ThomasSolverHBS(DirectSolver):

    def __init__(self,m,rk,cyclic=False,diagnostics=False,diagnostics_opts=None):
        """diagnostics=True runs hbs_diagnostics.diagnose_thomas after every
        factorize and stores the result in self.report.  diagnostics_opts is
        passed through to it (metrics, rhs, nprobe, seed, ...).  Off by
        default; when off, nothing extra is computed."""
        super().__init__(m,cyclic)
        if diagnostics and cyclic:
            raise ValueError("diagnostics do not cover cyclic ThomasSolverHBS")
        # rk may be an int (constant schedule) or an rkStrat.  A stage is one
        # step of the recurrence: stage j consumes S'_j and produces S'_{j+1},
        # so strat.rank(j) is the rank of S'_{j+1} -- the same convention as
        # RedBlackSolverHBS, with one block per stage.  self.rk stays the
        # stage-0 rank; the per-stage ranks are in self.rkSchedule after
        # factorize().
        self.rkStrat = rkStrat.coerce(rk)
        self.rk = self.rkStrat.rank(0)
        self.rkSchedule = None
        self.solve_method = None
        self.diagnostics = diagnostics
        self.diagnostics_opts = dict(diagnostics_opts or {})
        self.report = None

    def _rank_schedule(self, nstages, first_diag, tree):
        """Per-stage ranks for one factorization (stage j -> S'_{j+1}).

        skip_first_level is only sound when stage 0 carries no inversion,
        i.e. the first diagonal is the identity (always so for id_diag,
        where S'_0 = I).  Otherwise the flag is turned off for this
        factorization rather than silently under-resolving stage 1."""
        strat = self.rkStrat
        if (strat.skip_first_level and strat.mode != 'const'
                and not _is_id(first_diag)):
            warnings.warn(
                "ThomasSolverHBS: rkStrat has skip_first_level=True but the "
                "first diagonal block is not the identity, so stage 0 does "
                "invert. Disabling the skip for this factorization.", UserWarning)
            strat = strat.copy(skip_first_level=False)
        self.rkSchedule = strat.validate(nstages, nl=_leaf_size(tree),
                                         label='ThomasSolverHBS')
        print(" " + strat.describe(nstages))
        return self.rkSchedule

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
        m = S_rk_list[0][0].shape[0]
        n = len(S_rk_list) - 1
        I = id_op(m, S_rk_list[0][0].dtype)
        Sl = [S_rk_list[_][0] for _ in range(1,n+1)]
        Sprime = [I]
        Sr = [S_rk_list[_][-1] for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)
        ranks = self._rank_schedule(n, I, Sl[0].tree if n > 0 else None)
        for i in range(1, n+1):
            rk = ranks[i-1]                    # stage i-1 produces S'_i
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
        m = D_list[0].shape[0]
        n = len(D_list) - 1

        # Thus we need three lists of block matrices: A, B, and C:
        A = [AB_list[_][0] for _ in range(n)]
        B = [D_list[0]] # Set initial B_i to identity matrix LU factor (can specialize this to be just identity later)
        C = [AB_list[_][-1] for _ in range(n)] # C is easy, unmodified from original matrix (last entry is F)

        ranks = self._rank_schedule(n, D_list[0], getattr(D_list[0], 'tree', None))
        for i in range(1, n+1):
            B_i = HBSnew.HBSMAT(Dprime_Linop(D_list[i], A[i-1], C[i-1], B[-1]),
                                tree=D_list[i].tree, quad=D_list[i].quad)
            B_i.construct(ranks[i-1],compute_ULV=True,fast=True)    # stage i-1 -> B_i
            B.append(B_i)
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D_list   # kept for diagnostics (exact Dprime rebuild)
    
    def solve_helper(self,rhs,glob_target_dofs=None):
        if self.solve_method=='id_diag':
            return self.solve_id_diag(rhs,glob_target_dofs)
        elif self.solve_method == 'diag':
            return self.solve_with_diag(rhs,glob_target_dofs)
        else:
            raise ValueError('Factorization not set')
    
    def solve_id_diag(self, rhs,glob_target_dofs = None, Sprime=None):
        # Sprime: optional override of the factored diagonals (diagnostics
        # use this to splice exact blocks in after a given stage).
        m       = self.m
        Sl      = self.A
        Sprime  = self.B if Sprime is None else Sprime
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
    
    def solve_with_diag(self, rhs,glob_target_dofs = None, B=None):
        # B: optional override of the factored diagonals (see solve_id_diag).
        m = self.m
        A = self.A
        B = self.B if B is None else B
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

    def factorize(self, S_rk_list, T=None, S_exact=None, D_exact=None):
        """S_exact / D_exact are only read when diagnostics=True."""
        self.factorize_helper(S_rk_list, T)
        self.report = None
        if self.diagnostics:
            self.report = _load_diagnostics().diagnose_thomas(
                self, S_exact=S_exact, D_exact=D_exact, **self.diagnostics_opts)

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

    ---------------------------------------------------------------------
    CYCLIC COARSEST STEP
    ---------------------------------------------------------------------
    In the cyclic case, reducing nSlabs = 2 -> 1 leaves node 0 with node 1 as
    both its left and right neighbour, so after elimination the survivor
    couples to itself through A_0 and C_0 as well as through B_0.  The
    coarsest diagonal is therefore

        B_0 + A_0 + C_0 = T_0 - (S^-_0 + S^+_0) T_1^{-1} (S^-_1 + S^+_1),

    and that sum is what gets compressed and factorized there.  A_0 and C_0
    are stored as zero_op at that level, since nothing else reads them.
    Cyclic problems need nSlabs >= 2.
    """

    def __init__(self, m, rk, tree, quad, cyclic=False,
                 compress_diag=True, fused=True, device='cpu', fast=False,
                 seed=0, identity_diag=None, skip_unused_ulv=True,
                 diagnostics=False, diagnostics_opts=None):
        """diagnostics=True runs hbs_diagnostics.diagnose_redblack after every
        factorize and stores the result in self.report.  diagnostics_opts is
        passed through to it (metrics, rhs, nprobe, seed, ...).  Off by
        default; when off, nothing extra is computed.  The diagnostics use
        their own random generator and do not touch the counters below."""
        super().__init__(m, cyclic)
        self.diagnostics = diagnostics
        self.diagnostics_opts = dict(diagnostics_opts or {})
        self.report = None
        # rk may be an int (constant schedule) or an rkStrat.  self.rk stays
        # the stage-0 rank so existing callers that read or print it keep
        # meaning what they meant; the per-stage ranks are in self.rkSchedule
        # after factorize().
        self.rkStrat = rkStrat.coerce(rk)
        self.rk   = self.rkStrat.rank(0)
        self.rkSchedule = None
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

    @property
    def nl(self):
        """Leaf size HBSMAT actually uses -- not tree._min_leaf_size.
        None when the tree does not expose it (the schedule check is then
        skipped)."""
        return _leaf_size(self.tree)

    def _nsamples(self, rk):
        """Sample count, matching HBSMAT.construct's internal choice.

        Every operator sharing an Omega must use the same s.  All nodes share
        self.tree, so one value per level is consistent by construction.
        """
        mls = getattr(self.tree, "_min_leaf_size", rk)
        n_max = max(mls, 2*rk)
        p = 10
        s = max(n_max + rk + p , (int)(np.ceil(1.5 * n_max)) )
        return s

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

    def factorize(self, S_rk_list, T=None, S_exact=None, T_exact=None):
        """S_exact / T_exact are only read when diagnostics=True."""
        m      = S_rk_list[0][0].shape[0]
        nSlabs = len(S_rk_list)

        if not ((nSlabs & (nSlabs - 1) == 0) and nSlabs != 0):
            raise ValueError("Number of slabs must be a power of 2.")
        if self.cyclic and nSlabs < 2:
            raise ValueError("Cyclic RedBlackSolverHBS needs at least 2 slabs.")

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

        # ---- rank schedule ------------------------------------------------
        # Stage j is one builder call: it consumes RB[j] and produces RB[j+1],
        # so strat.rank(j) is the rank of every operator in RB[j+1].
        #
        # skip_first_level is only sound when stage 0 carries no inversion,
        # i.e. the level-0 diagonal is the identity.  A caller-supplied T that
        # is not all id_op breaks that, so the flag is turned off for this
        # factorization rather than silently under-resolving stage 1.
        strat   = self.rkStrat
        nstages = nSlabs.bit_length() - 1              # log2(nSlabs)
        if (strat.skip_first_level and strat.mode != 'const'
                and not all(_is_id(op) for op in T)):
            warnings.warn(
                "RedBlackSolverHBS: rkStrat has skip_first_level=True but the "
                "level-0 diagonal is not the identity, so stage 0 does invert. "
                "Disabling the skip for this factorization.", UserWarning)
            strat = strat.copy(skip_first_level=False)
        self.rkSchedule = strat.validate(nstages, nl=self.nl,
                                         label='RedBlackSolverHBS')
        print(" " + strat.describe(nstages))

        l = nSlabs
        j = 0
        while l > 1:
            rk = self.rkSchedule[j]
            builder = self._build_level_fused if self.fused else self._build_level
            RB.append(builder(m, l, RB[-1], rk))
            j += 1
            l //= 2

        self.nSlabs = nSlabs
        self.RB     = RB

        self.report = None
        if self.diagnostics:
            self.report = _load_diagnostics().diagnose_redblack(
                self, S_exact=S_exact, T_exact=T_exact, **self.diagnostics_opts)

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
            # Cyclic 2 -> 1 step: A_0 and C_0 are self-couplings of the lone
            # survivor and are folded into its diagonal (see class docstring).
            fold = cyclic and nSlabs == 2

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

            if fold:
                # (B+A+C) Om and (B+A+C)^T Psi from samples already in hand.
                Y_B = Y_B + Y_A + Y_C
                Z_B = Z_B + Z_A + Z_C

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
                B_lin = RB_linop(T[i], tmo, tpo, SiP[i], SiM[i], smp, spm)
                if fold:
                    B_lin = _sum_linop(B_lin,
                                       STS_linop(SiM[i], T_hbs[kL], SiM[kL]),
                                       STS_linop(SiP[i], T_hbs[kR], SiP[kR]))
                B_i.append(B_lin)

            # A_i and C_i become SiM / SiP one level down and are only ever
            # applied, never solved with -- no ULV, unconditionally.
            A_i.append(zero_op(m, dtype) if (A_is_zero or fold)
                       else self._hbs_from_samples(rk, Om, Psi, Y_A, Z_A,
                                                   compute_ULV=False,
                                                   label=f"A[{i}] (nSlabs={nSlabs})"))
            C_i.append(zero_op(m, dtype) if (C_is_zero or fold)
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
        # Cyclic 2 -> 1 step: fold A_0 and C_0 into B_0 (see class docstring).
        fold = cyclic and nSlabs == 2

        for i in range(0, nSlabs, 2):
            spm = SiP[(i - 1) % nSlabs] if ((i > 0) or cyclic) else None
            smp = SiM[(i + 1) % nSlabs] if ((i < nSlabs - 1) or cyclic) else None
            tm  = T_hbs[(i - 1) % nSlabs] if spm is not None else None
            tp  = T_hbs[(i + 1) % nSlabs] if smp is not None else None

            need_ULV = self._needs_ulv(i, nSlabs)
            B_linop  = RB_linop(T[i], tm, tp, SiP[i], SiM[i], smp, spm)
            if fold:
                B_linop = _sum_linop(
                    B_linop,
                    STS_linop(SiM[i], T_hbs[(i - 1) % nSlabs], SiM[(i - 1) % nSlabs]),
                    STS_linop(SiP[i], T_hbs[(i + 1) % nSlabs], SiP[(i + 1) % nSlabs]))

            if need_ULV or self.compress_diag or not self.skip_unused_ulv:
                B_hbs = self._hbs(B_linop, rk, compute_ULV=need_ULV,
                                  label=f"B[{i}] (nSlabs={nSlabs})")
            else:
                B_hbs = self._dead_diag(f"B[{i}] (nSlabs={nSlabs})")

            B_i.append(B_hbs if self.compress_diag else B_linop)
            T_hbs_new.append(B_hbs)

        A_i = []
        for i in range(0, nSlabs, 2):
            if ((not cyclic) and i == 0) or fold:
                A_i.append(zero_op(m, dtype))
            else:
                A_i.append(self._hbs(
                    STS_linop(SiM[i], T_hbs[(i - 1) % nSlabs],
                              SiM[(i - 1) % nSlabs]), rk,
                    compute_ULV=False, label=f"A[{i}] (nSlabs={nSlabs})"))

        C_i = []
        for i in range(0, nSlabs, 2):
            if ((not cyclic) and i == nSlabs - 2) or fold:
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

    def _forward_reduce(self, rhs, nlev=None):
        """Forward reduction through the first `nlev` levels (all by default).

        Returns the list of reduced right-hand sides, one per level visited.
        Split out of `solve` so diagnostics can stop the reduction early.
        """
        m  = self.m
        RB = self.RB
        if nlev is None:
            nlev = len(RB) - 1

        vPrimes = [rhs.copy()]
        dtype   = np.result_type(np.asarray(rhs).dtype, self._dtype)

        for l in range(nlev):
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
        return vPrimes

    def _back_substitute(self, vPrimes):
        """Back substitution from the coarsest entry of `vPrimes`, which must
        already hold that level's solution.  Works in place; returns level 0."""
        m  = self.m
        RB = self.RB

        for l in range(len(vPrimes) - 1, 0, -1):
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

    def solve(self, rhs):
        vPrimes = self._forward_reduce(rhs)
        # ---- coarsest solve -------------------------------------------
        vPrimes[-1] = self.RB[-1][2][0].solve(vPrimes[-1])
        return self._back_substitute(vPrimes)