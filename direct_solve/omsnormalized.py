from direct_solve.sample_reuse import SampleGraph
import numpy as np
from scipy.sparse.linalg   import LinearOperator
import matAssembly.HBS.HBStorch_new as HBSnew
from abc import ABC, abstractmethod
from direct_solve.omsdirectsolve import DirectSolver
import torch
import time
import gc
import math
import warnings
import re
import collections
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


_CENSUS = [True]
def census(tag, topn=12):
    if not _CENSUS[0]:
        return
    seen, big, total = set(), [], 0
    for o in gc.get_objects():
        try:
            if not torch.is_tensor(o) or not o.is_cuda:
                continue
            st = o.untyped_storage()
            p = st.data_ptr()
            if p in seen:
                continue
            seen.add(p)
            mb = st.nbytes() / 2**20
            total += mb
            if mb > 32:
                big.append((mb, tuple(o.shape)))
        except Exception:
            pass
    big.sort(reverse=True)
    print(f"[{tag}] alloc {torch.cuda.memory_allocated()/2**30:5.2f} GB "
          f"peak {torch.cuda.max_memory_allocated()/2**30:5.2f} GB "
          f"reachable {total/2**10:5.2f} GB "
          f"in {len(big)} blocks >32MB")
    for mb, sh in big[:topn]:
        print(f"        {mb:8.1f} MB  {sh}")

def _resolve_device(spec):
    if spec is None or spec == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(spec)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            f"compute_device={spec!r} requested but torch.cuda.is_available() "
            "is False. Pass 'cpu' or 'auto' if a host fallback is acceptable."
        )
    if dev.type == 'cuda' and dev.index is None:
        dev = torch.device('cuda', torch.cuda.current_device())
    return dev

# ---------------------------------------------------------------------------
# Linear operator helpers
# ---------------------------------------------------------------------------

class _NoResidency:
    """Residency no-ops for operators that own no device memory."""
    strict = False
    def prefetch(self, *a, **kw):  return self
    def evict(self, *a, **kw):     return self
    def release_ulv(self):         return self
    def device_nbytes(self, include_ulv=True): return 0
    @property
    def is_resident(self): return True


def _rdtype(*ops):
    """Promoted dtype of a set of operators; float64 if none carry one."""
    dts = [getattr(o, "dtype", None) for o in ops]
    dts = [d for d in dts if d is not None]
    return np.result_type(*dts) if dts else np.float64


class _TorchDirect:
    """Bypass scipy's numpy coercion for torch operands.

    scipy's LinearOperator.matmat/rmatmat run np.asanyarray(X) before
    dispatching to _matmat/_rmatmat, which raises on a CUDA tensor
    ("can't convert cuda:0 device type tensor to numpy").  The _matmat
    implementations in id_op/zero_op are already torch-aware, so route
    tensors straight to them and leave the numpy path to scipy.

    Must precede LinearOperator in the bases so these win the MRO.
    """
    def matmat(self, X):
        return self._matmat(X) if torch.is_tensor(X) else super().matmat(X)

    def rmatmat(self, X):
        return self._rmatmat(X) if torch.is_tensor(X) else super().rmatmat(X)

    def matvec(self, x):
        if not torch.is_tensor(x):
            return super().matvec(x)
        y = self._matmat(x[:, None] if x.ndim == 1 else x)
        return y[:, 0] if x.ndim == 1 else y

    def rmatvec(self, x):
        if not torch.is_tensor(x):
            return super().rmatvec(x)
        y = self._rmatmat(x[:, None] if x.ndim == 1 else x)
        return y[:, 0] if x.ndim == 1 else y


class id_op(_TorchDirect,_NoResidency,LinearOperator):
    """Identity operator."""

    def __init__(self, n, dtype=np.float64):
        super().__init__(shape=(n, n), dtype=dtype)
        self.tree = None
        self.quad = None
    def _matvec(self, v):         return v.clone() if torch.is_tensor(v) else v.copy()
    _matmat = _rmatvec = _rmatmat = _matvec
    def solve(self,v,mode='N'):
        return v.clone() if torch.is_tensor(v) else v.copy()
def _zeros_like_input(V,rows,dtype):
    cols = V.shape[1] if V.ndim == 2 else 1
    if torch.is_tensor(V):
        return torch.zeros(rows,cols,dtype=V.dtype,device=V.device)
    return np.zeros((rows,cols),dtype=dtype)

def _fmt(x):
    return "    -    " if x is None else f"{x:9.2e}"


class zero_op(_TorchDirect,_NoResidency,LinearOperator):
    """Zero operator; replaces a materialized dense zero block."""
    def __init__(self, n, dtype=np.float64, m=None):
        m = n if m is None else m
        super().__init__(shape=(m, n), dtype=dtype)
        self.tree = None
        self.quad = None
    def _matmat(self, V):   return _zeros_like_input(V,self.shape[0],self.dtype)
    def _rmatmat(self, v):  return _zeros_like_input(v,self.shape[1],self.dtype)
    _matvec, _rmatvec = _matmat, _rmatmat


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


def _linop_from_mat(A,device=None):
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
        # rkStrat is not wired into this solver yet: the stages here are the
        # steps of the recurrence (with i == 1, S'_1 = I, as the analogue of
        # skip_first_level).  Accept one so the call signature matches
        # RedBlackSolverHBS, but say plainly that only stage 0 is used.
        self.rkStrat = rkStrat.coerce(rk)
        if self.rkStrat.mode != 'const':
            warnings.warn(
                "ThomasSolverHBS does not implement a rank schedule yet; "
                f"using the stage-0 rank {self.rkStrat.rank(0)} throughout.",
                UserWarning, stacklevel=2)
        self.rk = self.rkStrat.rank(0)
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


class _BlockRedBlackSolverHBS(DirectSolver):
    """Normalized red-black solver for real I + offdiag(S) systems.

    Fixed random tests and operator samples live on the host (pinned for
    CUDA). random_buffer_bytes caps the reusable GPU RNG buffer;
    sample_panel_cols bounds composite sampling work per application.
    HBS kernels retain their existing independent memory-budget controls.
    Each next-level coupling is a standalone rank-k HBS compression of
    -D^{-1} C. Only D is compressed at rank 2k and ULV-factorized; its
    factors are retained for RHS reduction, not wrapped into the couplings.
    The SVD compression algorithm is unchanged. No Gram matrices are formed.
    block_diagnostics enables independent four-column random-probe checks
    during factorization, including global inverse checks after construction.
    Results are printed separately from dbg timings and retained in diagnostics.
    Diagnostic time is excluded from levelTimes, but included in external
    wall-clock measurements of factorize(). No diagnostic work occurs in solve().

    constant_blocks=True asserts that all interior left blocks represent the
    same operator, and likewise all interior right blocks (L need not equal R).
    One supplied representative per direction is used; independently compressed
    copies may differ slightly. Non-cyclic end rows remain untouched. For a
    cyclic system the assertion covers all rows. Equality is not inferred from
    PDE coefficients or checked numerically. Blocks must use the same ordering.
    Repeated constructions are shared within each level; boundary-dependent
    operators remain distinct. No product wrappers are passed to the next level.
    Intermediate samples are shared per level, on the host. sample_cache_bytes
    defaults to None (retain all reusable intermediates until level completion);
    a byte limit enables LRU eviction, and 0 disables intermediate caching.
    Counts and peak cache memory are available in sampling_stats and dbg output.
    """
    def __init__(self, m, rk, tree, quad, cyclic=False, seed=None,
                device='cpu', compute_device=None, work_device='cpu',
                oversample=None, dbg=0, sample_panel_cols=64,
                random_buffer_bytes=32 << 20, block_diagnostics=False,
                constant_blocks=False, sample_cache_bytes=None):
        super().__init__(m, cyclic)
        self.m, self.cyclic = m, cyclic
        self.sample_cache_bytes = sample_cache_bytes
        self.sampling_stats = []
        self._sample_graph = None
        self.rkStrat = rkStrat.coerce(rk)
        self.rk = self.rkStrat.rank(0)
        self.tree, self.quad = tree, quad
        self.fac = 4 if quad else 2
        self.compute_device = _resolve_device(
            compute_device if compute_device is not None else device)
        self.work_device = torch.device(work_device)
        if self.work_device.type != 'cpu':
            raise ValueError("RedBlackSolverHBS requires host work_device='cpu'")
        if self.compute_device.type not in ('cpu', 'cuda'):
            raise ValueError("compute_device must be CPU or CUDA")
        self.sample_panel_cols = int(sample_panel_cols)
        self.random_buffer_bytes = int(random_buffer_bytes)
        if self.sample_panel_cols < 1 or self.random_buffer_bytes < 8:
            raise ValueError("sample_panel_cols and random_buffer_bytes must be positive")
        self._pin = self.compute_device.type == 'cuda'
        self._random_buffer = None
        self._sample_cache = None
        self.levels = []
        self.oversample = oversample
        self.dbg = dbg
        self.block_diagnostics = bool(block_diagnostics)
        self.constant_blocks = bool(constant_blocks)
        self.constant_block_stats = []
        self.diagnostics = {'levels': [], 'global': None}
        self.dtype = torch.float64
        pl = tree.perm_leaf
        self.perm = (pl.detach().to('cpu', torch.int64) if torch.is_tensor(pl)
                     else torch.as_tensor(np.asarray(pl), dtype=torch.int64))
        self._gen = torch.Generator(device=self.compute_device)
        if seed is not None:
            self._gen.manual_seed(seed)
        self.RB = None
        self.blockTimes = []
        self.levelTimes = []

    # -- sizes ----------------------------------------------------------------

    @property
    def nl(self):
        """Leaf size HBSMAT actually uses."""
        return len(self.perm) // self.tree.nleaves

    def oversampling(self, rk):
        """p for rank rk: None -> rk; int -> p; float -> p * rk; callable."""
        p = self.oversample
        if p is None:
            return int(rk)
        if callable(p):
            return int(p(rk))
        if isinstance(p, int) and not isinstance(p, bool):
            return int(p)
        return int(round(float(p) * rk))

    def _nsamples(self, rk):
        return max(self.fac * rk, self.nl) + rk + 100

    # -- the only two operations on blocks ------------------------------------

    def _ap(self, op, X, out=None, alpha=1.0, beta=0.0, T=False):
        """out = alpha op X + beta out  (op^T with T); None is zero."""
        if out is None:                     # results stay on work_device
            out, beta = self._new(X.shape[1]), 0.0
        if op is None:
            return out.zero_() if beta == 0.0 else out.mul_(beta)
        f = op.rmatmat if T else op.matmat
        return f(X, out=out, alpha=alpha, beta=beta, io_perm=self.perm)

    def _sv(self, op, X, mode='N'):
        """X <- op^{-1} X in place (op^{-T} with mode 'T'); None is I."""
        if op is not None:
            op.solve(X, mode=mode, out=X, io_perm=self.perm)
        return X

    def _new(self, ncol):
        return torch.empty(self.m, ncol, dtype=self.dtype, device='cpu',
                           pin_memory=self._pin)

    def _clock(self):
        if self.dbg and self.compute_device.type == 'cuda':
            torch.cuda.synchronize(self.compute_device)
        return time.perf_counter()

    def _compress(self, rk, Om, Psi, Y, Z, ulv):
        h = HBSnew.HBSMAT(device=self.compute_device, tree=self.tree,
                          quad=self.quad)
        h.construct(rk, Om, Psi, Y, Z, compute_ULV=ulv)
        return h

    # -- sampling: fixed host arrays; only bounded random chunks on GPU ----

    def _fill_random(self, target):
        if not self._pin:
            target.normal_(generator=self._gen)
            return
        capacity = min(target.numel(), self.random_buffer_bytes // target.element_size())
        if self._random_buffer is None or self._random_buffer.numel() < capacity:
            self._random_buffer = torch.empty(capacity, dtype=self.dtype,
                                              device=self.compute_device)
        flat = target.view(-1)
        stream = torch.cuda.current_stream(self.compute_device)
        for a in range(0, flat.numel(), capacity):
            count = min(capacity, flat.numel() - a)
            buf = self._random_buffer[:count]
            buf.normal_(generator=self._gen)
            flat[a:a+count].copy_(buf, non_blocking=True)
            # Generation and D2H share a stream, so reuse is ordered after
            # the preceding copy. Synchronize once before host consumption.
        stream.synchronize()

    def _samples(self, rk):
        s = self._nsamples(rk)
        Om, Psi = self._sample_graph.roots['Om'], self._sample_graph.roots['Psi']
        if self._sample_cache is None or self._sample_cache[0].shape[1] != s:
            self._sample_cache = self._new(s), self._new(s)
        Y, Z = self._sample_cache
        return Om[:, :s], Psi[:, :s], Y, Z

    def _sample_compress(self, action, rk, ulv, kind, index):
        t0 = self._clock()
        Om, Psi, Y, Z = self._samples(rk)
        # HBS apply/solve stream rows internally. Column panels also bound
        # the upper-level vector workspaces of composite applications.
        for a in range(0, Om.shape[1], self.sample_panel_cols):
            b = min(a + self.sample_panel_cols, Om.shape[1])
            self._sample_span = (a, b)
            action(Om[:, a:b], Y[:, a:b], False)
            action(Psi[:, a:b], Z[:, a:b], True)
        t1 = self._clock()
        h = self._compress(rk, Om, Psi, Y, Z, ulv)
        self.blockTimes[-1].append((kind, index, t1-t0, self._clock()-t1))
        return h

    def _edge(self, SiM, SiP, i, j):
        """Terms of the actual block (i,j); add both directions when N=2."""
        N = len(SiM)
        terms = []
        if (self.cyclic or i > 0) and (i-1) % N == j and SiM[i] is not None:
            terms.append(SiM[i])
        if (self.cyclic or i+1 < N) and (i+1) % N == j and SiP[i] is not None:
            terms.append(SiP[i])
        return terms

    def _edge_apply(self, terms, X, T=False):
        result = self._new(X.shape[1]).zero_()
        for op in terms:
            self._ap(op, X, out=result, beta=1., T=T)
        return result

    def _constant_input_blocks(self, SiM, SiP):
        """Use representatives only where the caller asserts constant blocks.

        Copies the lists, not factors. Retain distinct end-row maps, since
        physical boundary conditions can change them even in homogeneous media.
        """
        SiM, SiP = list(SiM), list(SiP)
        N = len(SiM)
        interior = list(range(N)) if self.cyclic else list(range(1, N-1))
        for blocks in (SiM, SiP):
            if not interior:
                continue
            present = [blocks[i] is not None for i in interior]
            if any(present) and not all(present):
                raise ValueError("constant_blocks requires a consistent interior sparsity pattern")
            representative = blocks[interior[0]]
            for i in interior:
                blocks[i] = representative
        return SiM, SiP

    @staticmethod
    def _edge_key(terms):
        # An edge can be a sum at the cyclic N=2 terminal stage. Sorting
        # canonicalizes that sum; multiplicities are retained.
        return tuple(sorted(id(op) for op in terms))

    @torch.no_grad()
    def factorize(self, S_rk_list, T=None):
        """Factor I + offdiag(S); samples and composite temporaries stay on host."""
        if T is not None:
            raise NotImplementedError("RedBlackSolverHBS supports identity diagonals only")
        nSlabs = len(S_rk_list)
        if nSlabs < 2 or nSlabs & (nSlabs - 1):
            raise ValueError("Number of slabs must be a power of 2 (>= 2).")
        if len(self.perm) != self.m:
            raise ValueError("m must equal the number of DOFs in each block")
        SiM = [S[0] for S in S_rk_list]
        SiP = [S[-1] for S in S_rk_list]
        if not self.cyclic:
            SiM[0] = SiP[-1] = None
        for op in SiM + SiP:
            if op is not None:
                if tuple(op.shape) != (self.m, self.m):
                    raise ValueError("All blocks must have shape (m,m)")
                if op.perm is None or not torch.equal(op.perm.cpu(), self.perm):
                    raise ValueError("All blocks must use the same tree permutation")
                op.compute_device = self.compute_device
        if self.constant_blocks:
            SiM, SiP = self._constant_input_blocks(SiM, SiP)
        ranks = self.rkStrat.validate(nSlabs.bit_length()-1, nl=self.nl,
                                      label='RedBlackSolverHBS')
        self.RB = None
        self.levels = []
        RB = [(SiM, SiP)]
        self.blockTimes, self.levelTimes = [], []
        self.constant_block_stats = []
        self.diagnostics = {'levels': [], 'global': None}
        self.sampling_stats = []
        # A private CPU generator never advances the compression RNG.
        diagnostic_gen = None
        if self.block_diagnostics:
            diagnostic_gen = torch.Generator(device='cpu')
            diagnostic_gen.manual_seed(1729)
            print("[block diagnostics] Independent random probes; Frobenius-norm ratios.")
            print("  D: compression vs incoming-level operator; ULV vs stored D;")
            print("     inverse vs incoming-level D; ulv_T checks transpose solve.")
            print("  S: compression vs sampled -Dtilde^{-1}C; equilibrium = ||Dop(SV)+CV||/||CV||.")
            print("  adjoint_*: bilinear forward/transpose consistency (scaled by norm products).")
            print("  Timings below exclude diagnostics; external factorize timing includes them.")
        try:
            for rk in ranks:
                t0 = self._clock()
                self.blockTimes.append([])
                RB.append(self._build_level(*RB[-1], rk))
                self.levelTimes.append((len(RB[-2][0]), self._clock()-t0, rk))
                if self.dbg >= 1:
                    self._report_level(len(self.levelTimes)-1)
                if self.block_diagnostics:
                    self._diagnose_level(len(self.levelTimes)-1, RB[-1], diagnostic_gen)
            self.RB = RB
            if self.block_diagnostics:
                self._diagnose_global(diagnostic_gen)
        finally:
            self._sample_graph = None
            self._sample_cache = None
            self._random_buffer = None

    def _build_level(self, SiM, SiP, rk):
        N = len(SiM)
        retained = list(range(0, N, 2))
        diagonals = []
        # Caches are local to this level/factorization: rank and input factors
        # cannot become stale. Keys preserve multiplication order.
        diagonal_cache, coupling_cache = {}, {}
        stats = {'diagonal_built': 0, 'diagonal_reused': 0,
                 'coupling_built': 0, 'coupling_reused': 0}
        # One Gaussian family per level. Couplings use prefixes of the same
        # probes used for diagonals, enabling reuse across their different ranks.
        self._sample_cache = None
        width = max(self._nsamples(rk), self._nsamples(2*rk))
        Om, Psi = self._new(width), self._new(width)
        self._fill_random(Om)
        self._fill_random(Psi)
        graph = self._sample_graph = SampleGraph(
            self._edge_apply, self._sv, {'Om': Om, 'Psi': Psi}, self.sample_cache_bytes)
        def value(expr):
            return graph.value(expr, *self._sample_span)
        for i in retained:
            neighbors = sorted({j for j in ((i-1) % N, (i+1) % N)
                                if self._edge(SiM, SiP, i, j)})
            pairs = [(self._edge(SiM, SiP, i, j),
                      self._edge(SiM, SiP, j, i)) for j in neighbors]
            key = (2*rk, tuple(sorted((self._edge_key(left), self._edge_key(right))
                                     for left, right in pairs)))
            if self.constant_blocks and key in diagonal_cache:
                diagonals.append(diagonal_cache[key])
                stats['diagonal_reused'] += 1
                continue
            def diagonal(X, out, transpose, pairs=pairs):
                out.copy_(X)
                for left, right in pairs:
                    first, second = (left, right) if transpose else (right, left)
                    tmp = graph.apply(first, graph.root(transpose), transpose)
                    out.sub_(value(graph.apply(second, tmp, transpose)))
            D = self._sample_compress(diagonal, 2*rk, True, 'D', i)
            diagonals.append(D)
            stats['diagonal_built'] += 1
            if self.constant_blocks:
                diagonal_cache[key] = D

        newM, newP = [None]*(N//2), [None]*(N//2)
        # At N=2, both cyclic directions were aggregated in D. The remaining
        # equation is normalized to I; there are no self-couplings to retain.
        if N > 2:
            for q, i in enumerate(retained):
                for step, target, kind in ((-1, newM, 'S-'), (1, newP, 'S+')):
                    j, dest = i+step, i+2*step
                    if not self.cyclic and not (0 <= j < N and 0 <= dest < N):
                        continue
                    j, dest = j % N, dest % N
                    left = self._edge(SiM, SiP, i, j)
                    right = self._edge(SiM, SiP, j, dest)
                    if not left or not right:
                        continue
                    key = (rk, id(diagonals[q]), self._edge_key(left), self._edge_key(right))
                    if self.constant_blocks and key in coupling_cache:
                        target[q] = coupling_cache[key]
                        stats['coupling_reused'] += 1
                        continue
                    def normalized(X, out, transpose, left=left,
                                   right=right, D=diagonals[q]):
                        # C = left @ right. Sample the complete normalized
                        # coupling; never separately compress C or store a
                        # product/solve wrapper in the next level.
                        tmp = graph.root(transpose)
                        if transpose:
                            tmp = graph.solve(D, tmp, True)
                            tmp = graph.apply(left, tmp, True)
                            tmp = graph.apply(right, tmp, True)
                        else:
                            tmp = graph.apply(right, tmp)
                            tmp = graph.apply(left, tmp)
                            tmp = graph.solve(D, tmp)
                        out.copy_(value(tmp)).mul_(-1.)
                    target[q] = self._sample_compress(
                        normalized, rk, False, kind, i)
                    stats['coupling_built'] += 1
                    if self.constant_blocks:
                        coupling_cache[key] = target[q]
        self.sampling_stats.append(dict(graph.stats))
        self._sample_graph = None
        self.constant_block_stats.append(stats)
        self.levels.append((SiM, SiP, diagonals))
        self._sample_cache = None
        return newM, newP

    def _report_level(self, j):
        N, dt, rk = self.levelTimes[j]
        ts = sum(t[2] for t in self.blockTimes[j])
        tc = sum(t[3] for t in self.blockTimes[j])
        print(f"[timing] level {j}: nSlabs = {N:5d}  rk = {rk:4d}  {dt:8.2f} s  "
              f"(sample {ts:.2f} s, compress {tc:.2f} s)")
        for kind, i, a, b in self.blockTimes[j]:
            print(f"   {kind}[{i:4d}]  sample {a:8.3f} s  compress {b:8.3f} s")
        ss = self.sampling_stats[j]
        print(f"[sample reuse] level {j}: hits={ss['hits']}, "
              f"apply columns={ss['apply_columns']}, solve columns={ss['solve_columns']}, "
              f"peak host MiB={ss['peak_bytes']/2**20:.1f}, evictions={ss['evictions']}")
        if self.constant_blocks:
            stats = self.constant_block_stats[j]
            print(f"[reuse] level {j}: D built={stats['diagonal_built']}, "
                  f"reused={stats['diagonal_reused']}; "
                  f"S built={stats['coupling_built']}, reused={stats['coupling_reused']}")

    # -- optional independent diagnostics -----------------------------------

    @staticmethod
    def _diagnostic_norm(X):
        return float(torch.linalg.vector_norm(X).item())

    @staticmethod
    def _diagnostic_ratio(numerator, denominator):
        if denominator == 0.:
            return 0. if numerator == 0. else float('inf')
        return numerator / denominator

    def _diagnostic_error(self, actual, reference):
        norm = self._diagnostic_norm
        return self._diagnostic_ratio(norm(actual-reference), norm(reference))

    def _diagnostic_probes(self, generator):
        # Keep diagnostic vectors on host and respect a smaller sampling panel.
        q = min(4, self.sample_panel_cols)
        X, Y = self._new(q), self._new(q)
        X.normal_(generator=generator)
        Y.normal_(generator=generator)
        return X, Y

    def _diagnostic_adjoint(self, X, Y, AX, ATY):
        norm = self._diagnostic_norm
        defect = abs(float(torch.sum(AX*Y).item()) - float(torch.sum(X*ATY).item()))
        return self._diagnostic_ratio(defect, norm(AX)*norm(Y)+norm(X)*norm(ATY))

    def _diagnostic_diagonal(self, SiM, SiP, i, X, transpose=False):
        """Uncompressed Schur diagonal action from the incoming stored blocks."""
        N = len(SiM)
        out = self._new(X.shape[1])
        out.copy_(X)
        for j in sorted({(i-1) % N, (i+1) % N}):
            left = self._edge(SiM, SiP, i, j)
            right = self._edge(SiM, SiP, j, i)
            if not left or not right:
                continue
            first, second = (left, right) if transpose else (right, left)
            tmp = self._edge_apply(first, X, T=transpose)
            out.sub_(self._edge_apply(second, tmp, T=transpose))
        return out

    def _diagnose_level(self, level, next_blocks, generator):
        t0 = time.perf_counter()
        SiM, SiP, diagonals = self.levels[level]
        newM, newP = next_blocks
        N = len(SiM)
        record = {'level': level, 'diagonals': [], 'couplings': []}
        print(f"[block diagnostics] level {level}: incoming blocks={N}, "
              f"probes={min(4, self.sample_panel_cols)}")
        for q, i in enumerate(range(0, N, 2)):
            D = diagonals[q]
            X, Y = self._diagnostic_probes(generator)
            DX = self._diagnostic_diagonal(SiM, SiP, i, X)
            DTY = self._diagnostic_diagonal(SiM, SiP, i, Y, True)
            stored_DX = self._ap(D, X)
            stored_DTY = self._ap(D, Y, T=True)
            inverse_X = self._new(X.shape[1])
            inverse_X.copy_(X)
            self._sv(D, inverse_X)
            inverse_TY = self._new(Y.shape[1])
            inverse_TY.copy_(Y)
            self._sv(D, inverse_TY, mode='T')
            metrics = {
                'compression': self._diagnostic_error(stored_DX, DX),
                'ulv': self._diagnostic_error(self._ap(D, inverse_X), X),
                'inverse': self._diagnostic_error(
                    self._diagnostic_diagonal(SiM, SiP, i, inverse_X), X),
                'ulv_T': self._diagnostic_error(self._ap(D, inverse_TY, T=True), Y),
                'adjoint_target': self._diagnostic_adjoint(X, Y, DX, DTY),
                'adjoint_hbs': self._diagnostic_adjoint(X, Y, stored_DX, stored_DTY),
            }
            record['diagonals'].append({'index': i, **metrics})
            print(f"  D[{i}] " + '  '.join(f"{key}={value:.3e}" for key, value in metrics.items()))
            for step, blocks, kind in ((-1, newM, 'S-'), (1, newP, 'S+')):
                G = blocks[q]
                if G is None:
                    continue
                j, dest = (i+step) % N, (i+2*step) % N
                left = self._edge(SiM, SiP, i, j)
                right = self._edge(SiM, SiP, j, dest)
                # Directly evaluate the operator used for normalized sampling.
                CX = self._edge_apply(left, self._edge_apply(right, X))
                exact_GX = self._new(X.shape[1])
                exact_GX.copy_(CX)
                self._sv(D, exact_GX)
                exact_GX.mul_(-1.)
                exact_GTY = self._edge_apply(
                    right, self._edge_apply(left, inverse_TY, T=True), T=True)
                exact_GTY.mul_(-1.)
                GX, GTY = self._ap(G, X), self._ap(G, Y, T=True)
                equilibrium = self._diagnostic_diagonal(SiM, SiP, i, GX)
                equilibrium.add_(CX)
                metrics = {
                    'compression': self._diagnostic_error(GX, exact_GX),
                    'compression_T': self._diagnostic_error(GTY, exact_GTY),
                    'equilibrium': self._diagnostic_ratio(
                        self._diagnostic_norm(equilibrium), self._diagnostic_norm(CX)),
                    'adjoint_samples': self._diagnostic_adjoint(X, Y, exact_GX, exact_GTY),
                    'adjoint_hbs': self._diagnostic_adjoint(X, Y, GX, GTY),
                }
                record['couplings'].append({'index': i, 'kind': kind, **metrics})
                print(f"  {kind}[{i}] " + '  '.join(f"{key}={value:.3e}" for key, value in metrics.items()))
        record['seconds'] = time.perf_counter()-t0
        self.diagnostics['levels'].append(record)
        print(f"[block diagnostics] level {level} checks: {record['seconds']:.3f} s")

    def _diagnose_global(self, generator):
        """Left/right inverse residuals against the original stored HBS system.

        The LU/reference PDE operator and pre-conversion blocks are not owned
        by this solver, so these checks do not test either of those objects.
        """
        t0 = time.perf_counter()
        SiM, SiP = self.RB[0]
        N, q = len(SiM), min(4, self.sample_panel_cols)
        V = torch.empty((N, self.m, q), dtype=self.dtype,
                        device='cpu', pin_memory=self._pin)
        V.normal_(generator=generator)
        def apply_system(X):
            out = torch.empty((N, self.m, q), dtype=self.dtype,
                              device='cpu', pin_memory=self._pin)
            out.copy_(X)
            for i in range(N):
                for j in sorted({(i-1) % N, (i+1) % N}):
                    for op in self._edge(SiM, SiP, i, j):
                        self._ap(op, X[j], out=out[i], beta=1.)
            return out
        def solve_internal(X):
            # Public solve expects original order; convert in both directions.
            external = torch.empty((N, self.m, q), dtype=self.dtype,
                                   device='cpu', pin_memory=self._pin)
            external[:, self.perm, :] = X
            result = self.solve(external.reshape(N*self.m, q)).reshape(N, self.m, q)
            return result[:, self.perm, :]
        left = self._diagnostic_error(solve_internal(apply_system(V)), V)
        right = self._diagnostic_error(apply_system(solve_internal(V)), V)
        metrics = {'left_inverse': left, 'right_inverse': right,
                   'seconds': time.perf_counter()-t0, 'probes': q}
        self.diagnostics['global'] = metrics
        print("[block diagnostics] global, against input HBS system:")
        print(f"  ||P A V - V||/||V||={left:.3e}  ||A P V - V||/||V||={right:.3e}")
        print(f"[block diagnostics] global checks: {metrics['seconds']:.3f} s")

    # -- solve -------------------------------------------------------------------

    @torch.no_grad()
    def solve(self, rhs):
        """Solve vector or multi-RHS input; preserve input family/device/shape.

        NumPy and Torch real inputs are computed in float64. The caller's
        storage is never modified. Only the entry/exit use the permutation.
        """
        if self.RB is None:
            raise RuntimeError("factorize must complete before solve")
        is_tensor = torch.is_tensor(rhs)
        source = rhs if is_tensor else torch.as_tensor(np.asarray(rhs))
        if source.is_complex():
            raise TypeError("This HBS implementation supports real-valued inputs only")
        N = len(self.RB[0][0])
        if source.ndim not in (1, 2) or source.shape[0] != N*self.m:
            raise ValueError(f"rhs must have shape ({N*self.m},) or ({N*self.m}, nrhs)")
        vector = source.ndim == 1
        nrhs = 1 if vector else source.shape[1]
        host = source.detach().to(device='cpu', dtype=self.dtype).reshape(N, self.m, nrhs)
        def allocate(n):
            return torch.empty((n, self.m, nrhs), dtype=self.dtype,
                               device='cpu', pin_memory=self._pin)
        current = allocate(N)
        current.copy_(host[:, self.perm, :])
        saved = []
        if nrhs:
            for SiM, SiP, diagonals in self.levels:
                n = len(SiM)
                saved.append(current)
                reduced = allocate(n//2)
                for q, i in enumerate(range(0, n, 2)):
                    reduced[q].copy_(current[i])
                    for j in sorted({(i-1) % n, (i+1) % n}):
                        terms = self._edge(SiM, SiP, i, j)
                        for op in terms:
                            self._ap(op, current[j], out=reduced[q], alpha=-1., beta=1.)
                    self._sv(diagonals[q], reduced[q])
                current = reduced
            for (SiM, SiP, _), old_rhs in zip(reversed(self.levels), reversed(saved)):
                n = len(SiM)
                expanded = allocate(n)
                expanded[::2].copy_(current)
                for i in range(1, n, 2):
                    expanded[i].copy_(old_rhs[i])
                    for j in sorted({(i-1) % n, (i+1) % n}):
                        for op in self._edge(SiM, SiP, i, j):
                            # Every neighbor j is retained (even). Read its
                            # value from the separate coarse solution, avoiding
                            # HBS's shared-storage guard on expanded's views.
                            self._ap(op, current[j//2], out=expanded[i], alpha=-1., beta=1.)
                current = expanded
        result = allocate(N)
        result[:, self.perm, :] = current
        result = result.reshape(N*self.m, nrhs)
        if vector:
            result = result[:, 0]
        return result.to(source.device) if is_tensor else result.numpy()


from direct_solve.oms_input import OMSInputMixin


class RedBlackSolverHBS(OMSInputMixin, _BlockRedBlackSolverHBS):
    """OMS-based preconditioner. Construct with (rank_strategy, oms, **options),
    then call factorize() and solve(rhs). Input operators are never recompressed.
    operator=None selects LU when retained, otherwise HBS; operator='hbs' or
    'lu' explicitly selects a source. See OMSInputMixin for tree options.
    """
    _rank_strategy_type = rkStrat
