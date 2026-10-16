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



# ---------------------------------------------------------------------------
# HBS Red-Black solver
# ---------------------------------------------------------------------------
#
# Device memory policy
# --------------------
# HBSMAT keeps its factors on the host and uploads them only for the duration
# of one operation (see HBStorch).  The solver therefore does no residency
# management of its blocks.  The one refinement is a single "slot": the block
# used most recently stays mirrored until a different block is used, so the
# runs of consecutive calls on one block (S^+_i three times in a row, say)
# upload it once.  At most ONE block is mirrored by the solver at any time,
# and the slot is emptied before every compression and at the end of every
# factorize/solve.
#
# What the solver still owns is its m x s working tensors:
#   Om, Psi, Y_B, Z_B          on the device (read by every step of a node)
#   Y_A, Z_A, Y_C, Z_C         on the host from the moment they are formed;
#                              HBSMAT streams them to the device in chunks
#   Xm_k, Xp_k (cached solves) on the device between producing them and their
#                              first use, on the host between their first and
#                              second use (three compressions apart)
# so the device peak of a factorization is ~ 6 m x s tensors + one block +
# HBSMAT's chunk working set (HBStorch._CHUNK_BYTES), independent of nSlabs.
# ---------------------------------------------------------------------------


class _NoReport:
    """Stand-in for the dense-reference report of an older version, so that
    scripts calling it get a message instead of an AttributeError."""
    _MSG = ("RedBlackSolverHBS.report (dense reference diagnostics) is not "
            "part of this version; use debug_blocks=... and "
            "print_block_errors() for per-block checks.")

    def table(self, *args, **kwargs):
        warnings.warn(self._MSG, UserWarning, stacklevel=2)
        return self._MSG

    def comb(self, *args, **kwargs):
        warnings.warn(self._MSG, UserWarning, stacklevel=2)
        return []

    ladder = comb


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

    def __init__(self, m, rk, tree, quad, cyclic=False,seed=None,
                 compress_diag=True, fused=True, device='cpu', fast=True, identity_diag=True, skip_unused_ulv=True,compute_device=None,strict_residency=False,
                 oversample=None,
                 debug_blocks=0, debug_inverse=True, debug_true_inverse=False,
                 debug_seed=1234,
                 solve_cache_bytes=None, solve_keep_cache=False,
                 solve_reserve_bytes=None, preflight='raise'):
        super().__init__(m, cyclic)
        # --- solve residency ------------------------------------------------
        # solve_keep_cache=False (default): every block is uploaded for the
        #     call that uses it and dropped afterwards (see the memory policy
        #     note above the class).  Nothing stays on the device between
        #     solves.
        # solve_keep_cache=True: blocks used by solve() stay on the device
        #     until release_solve_cache() or the next factorize(), so repeated
        #     solves (GMRES) upload the factorization once.  Only for a
        #     factorization that fits on the device: nothing caps it.
        # solve_cache_bytes / solve_reserve_bytes belonged to the removed
        #     LRU cache; accepted and ignored.
        self.solve_keep_cache    = bool(solve_keep_cache)
        self.solve_cache_bytes   = solve_cache_bytes
        self.solve_reserve_bytes = solve_reserve_bytes
        if solve_cache_bytes is not None or solve_reserve_bytes is not None:
            warnings.warn(
                "RedBlackSolverHBS: solve_cache_bytes / solve_reserve_bytes "
                "are ignored; blocks are staged per use (or kept with "
                "solve_keep_cache=True).", UserWarning, stacklevel=2)
        self.solveCacheStats = {}
        # factorize() first predicts its device and host memory per stage
        # (see preflight()):  'raise' stops before any work if a stage will
        # not fit, 'warn' only reports, 'off' skips the check.
        if preflight not in ('raise', 'warn', 'off', None, False):
            raise ValueError(f"preflight must be 'raise', 'warn' or 'off', got {preflight!r}")
        self.preflight_mode = preflight or 'off'
        self._slot     = None     # the one block the solver keeps mirrored
        self._kept     = {}       # id -> block, for solve_keep_cache
        self._keep_all = False
        # rk may be an int (constant schedule) or an rkStrat.  self.rk stays
        # the stage-0 rank so existing callers that read or print it, and
        # _nsamples(self.rk), keep meaning what they meant.
        self.rkStrat = rkStrat.coerce(rk)
        self.rk   = self.rkStrat.rank(0)
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
        self.compute_device = _resolve_device(
            compute_device if compute_device is not None else device)
        # Not forwarded to the blocks: under the per-use staging policy,
        # HBSMAT.strict would turn every staged call into an error.
        self.strict_residency = strict_residency
        if strict_residency:
            warnings.warn(
                "RedBlackSolverHBS: strict_residency is ignored; blocks are "
                "staged per use by design.", UserWarning, stacklevel=2)
        # Oversampling above the hard floor; see _nsamples.  None -> p = rk.
        self.oversample = oversample
        if oversample is not None and not callable(oversample):
            _p0 = self.oversampling(int(rk) if isinstance(rk, int)
                                    else rk.rank(0))
            if _p0 < 10:
                warnings.warn(
                    f"RedBlackSolverHBS: oversampling p={_p0} is very small; "
                    "the randomized range finder needs a margin over the "
                    "rank-rk floor and will be unreliable here.", UserWarning)
        # --- per-block compression diagnostics (off by default) ----------
        # debug_blocks = number of probe columns; 0 disables.  8-16 is plenty:
        # the estimate is a Frobenius ratio over t columns, so its own
        # relative noise is ~1/sqrt(t), which is far finer than the orders of
        # magnitude this is meant to separate.  Cost is t/s of the sampling
        # work (~1% at t=16, s=1600) plus one extra block held resident.
        self.debug_blocks  = int(debug_blocks)
        self.debug_inverse = bool(debug_inverse)
        # inv_ref: ||B (B_hbs)^-1 W - W|| / ||W|| with B the UNCOMPRESSED
        # reference chain -- a per-block analogue of the global residual, and
        # unlike fwd it is sensitive to the small singular directions that
        # the inverse depends on.  Costs memory: the neighbour operators must
        # stay staged until after the B block is built and solved, instead of
        # being retired before compression.
        self.debug_true_inverse = bool(debug_true_inverse) and self.debug_blocks > 0
        if self.debug_true_inverse:
            warnings.warn(
                "RedBlackSolverHBS: debug_true_inverse re-applies the "
                "uncompressed neighbour chain for every B block; expect extra "
                "uploads and time, not extra device memory.", UserWarning)
        self.blockErrors   = []
        self._dbg_stage    = 0
        self._pgen = torch.Generator(device=self.compute_device)
        self._pgen.manual_seed(int(debug_seed))
        self._blocks = []
        self._block_ids = set()      # id() of every op in _blocks; see _adopt
        self._tdtype = torch.float64
        self._tgen   = torch.Generator(device=self.compute_device)
        if seed is not None:
            self._tgen.manual_seed(seed)

    
    # ------------------------------------------------------------------

    @property
    def nl(self):
        """Leaf size HBSMAT actually uses -- not tree._min_leaf_size."""
        return len(self.tree.perm_leaf) // self.tree.nleaves

    def oversampling(self, rk):
        """Oversampling p for rank rk: the surplus over the hard floor.

        See `oversample` in __init__ for the accepted forms.
        """
        p = self.oversample
        if p is None:
            return int(rk)                       # default: p = rk
        if callable(p):
            return int(p(rk))
        if isinstance(p, int) and not isinstance(p, bool):
            return int(p)                        # absolute count
        return int(round(float(p) * rk))         # multiple of rk

    def _nsamples(self, rk):
        """Sample count s = max(fac*rk, nl) + rk + p.

        Every operator sharing an Omega must use the same s.  All nodes share
        self.tree and one rank per stage, so one value per stage is consistent
        by construction.

        max(fac*rk, nl) + rk is the HARD FLOOR -- the point at which the null
        space of Omega is just large enough to hold a rank-rk range at the
        worst level (leaf: n = nl, k = min(rk, nl); interior: n = fac*rk,
        k = rk).  Everything above it is oversampling, and p is now set
        explicitly rather than falling out of the algebra:

          p = rk (default)  reproduces max(fac*rk, nl) + 2*rk, the rule that
                            gave s = 1600 and the best delta so far at
                            rk = 400, kh = 100.
          p = <int>         absolute, for holding p fixed across a rank sweep
                            so the two effects can be separated.
          p = <float>       a multiple of rk.
          p = <callable>    p(rk).

        Both earlier rules coupled p to something it should not depend on.
        max(fac*rk,nl)+2*rk makes p track rank; 2*max(rk,nl)+rk+20 gives
        p = 2*(nl - rk) + 20 at the interior levels, which COLLAPSES as rk
        approaches nl (244 at rk=400, 20 at rk=512 for nl=512) -- exactly
        over the range of ranks worth trying.

        Uses the leaf size HBSMAT actually uses, not tree._min_leaf_size.
        """
        fac = 4 if self.quad else 2
        return max(fac * rk, self.nl) + rk + self.oversampling(rk)

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

    def _register(self, h):
            h.compute_device = self.compute_device
            h._built_by_rb = True
            if id(h) not in self._block_ids:
                self._block_ids.add(id(h))
                self._blocks.append(h)
            return h
    def _adopt(self, op):
        """Take ownership of an input operator: compute on this solver's
        device, and drop any device mirror it was handed over with (one an
        assembler prefetched would otherwise sit on the device all run)."""
        if hasattr(op, 'evict'):
            # S_rk_list commonly repeats the same operator object for every
            # slab (translation-invariant slabs).  Register it once, so the
            # reports count it once.
            if id(op) in self._block_ids:
                return op
            op.evict()
            op.compute_device = self.compute_device
            self._block_ids.add(id(op))
            self._blocks.append(op)
        return op

    # -- the one-block slot (see the memory policy note above the class) --
    def _use(self, op, groups):
        """Stage `op` for the next call.  HBS blocks only; anything else
        (id_op, zero_op, RB_linop, ...) owns no device memory."""
        if not isinstance(op, HBSnew.HBSMAT):
            return
        if self._keep_all:                     # solve_keep_cache
            op.prefetch(groups=groups)
            self._kept[id(op)] = op
            return
        if self._slot is not op:
            self._drop_slot()
            self._slot = op
        op.prefetch(groups=groups)

    def _drop_slot(self):
        op, self._slot = self._slot, None
        if op is not None and id(op) not in self._kept:
            op.evict()

    def _on_dev(self, Y):
        """Results of the solver's own calls live on compute_device.  HBS
        blocks already return there; operators that echo their input's device
        (id_op, zero_op) are moved when given a host-parked input."""
        if torch.is_tensor(Y) and not HBSnew._dev_eq(Y.device, self.compute_device):
            return Y.to(self.compute_device)
        return Y

    def _finish(self, h, ulv, label):
        # Recorded before _guard_no_ulv so traffic() can group blocks by
        # kind / level; set via try in case the attribute is not settable.
        try:
            h._label = label
        except Exception:
            pass
        self.nConstruct += 1
        if ulv:
            self.nULV += 1
        else:
            self.nULVSkipped += 1
            h = self._guard_no_ulv(h, label or "an HBS block")
        self._register(h)
        return h

    def _hbs(self, linop, rk=None, device=None, compute_ULV=True, label=None):
        """Compress a LinearOperator into an HBS matrix.

        compute_ULV=False compresses for applies only and skips the
        factorization; the result is fitted with a raising `solve`.
        """
        rkloc = self.rk if rk is None else rk
        dev   = self.compute_device if device is None else torch.device(device)
        ulv   = self._want_ulv(compute_ULV)

        self._drop_slot()           # no solver-held block during compression
        h = HBSnew.HBSMAT(linop, device=dev, tree=self.tree, quad=self.quad)
        h.construct(rkloc, compute_ULV=ulv, fast=self.fast)

        return self._finish(h, ulv, label)

    def _hbs_from_samples(self, rk, Om, Psi, Y, Z, compute_ULV=True, label=None,
                          io_perm=None):
        """Compress from externally supplied samples Y = M Om, Z = M^T Psi.

        The samples are passed through as they are -- device or host, torch
        or numpy, boxed (a one-element list, handing over the only reference
        so HBSMAT frees it after the leaf level) or not.  HBSMAT streams them
        to the device chunk by chunk, so a host-parked sample is never
        uploaded whole.  compute_ULV=False skips the factorization; see `_hbs`.
        """
        self._drop_slot()           # no solver-held block during compression
        ulv = self._want_ulv(compute_ULV)
        h = HBSnew.HBSMAT(device=self.compute_device, tree=self.tree,
                          quad=self.quad)
        h.construct(rk, Om=Om, Psi=Psi, Y=Y, Z=Z,
                    compute_ULV=ulv, fast=self.fast, io_perm=io_perm)
        return self._finish(h, ulv, label)
    # ------------------------------------------------------------------
    # block-level compression diagnostics
    # ------------------------------------------------------------------
    def _dbg_probe(self, t):
        """t fresh probe columns, drawn from a generator of their own.

        Deliberately NOT self._tgen: pulling from the sampling stream would
        shift every subsequent Omega/Psi, so a debug run would no longer be
        bit-comparable to a non-debug run at the same seed.  With a separate
        generator the factorization is unchanged and only the measurement is
        added.
        """
        return torch.randn(self.m, t, generator=self._pgen,
                           device=self.compute_device, dtype=self._tdtype)

    @staticmethod
    def _relerr(approx, ref):
        den = torch.linalg.norm(ref)
        if den == 0:
            return float(torch.linalg.norm(approx))      # absolute if ref is 0
        return float(torch.linalg.norm(approx - ref) / den)

    def _check_block(self, h, kind, nSlabs, node, rk, W, Q, ref_f, ref_a,
                     apply_ref=None):
        """Compare a freshly built block against the operator it approximates.

        W/Q are probe
        columns independent of the Omega/Psi the block was compressed from, so
        this is an OUT-OF-SAMPLE test: construct_D fits D to the sampled
        columns by least squares, so reusing those columns would report a
        fitting residual rather than an approximation error.
        """
        rec = dict(stage=self._dbg_stage, nSlabs=nSlabs, node=node, kind=kind,
                   rk=rk, fwd=None, adj=None, inv=None, inv_ref=None,
                   scale=None, invscale=None, cond=None)

        if ref_f is not None:
            rec['scale'] = float(torch.linalg.norm(ref_f) /
                                 torch.linalg.norm(W))          # ||M|| proxy

        if isinstance(h, dead_op):
            rec['note'] = 'dead (no consumer)'
            self.blockErrors.append(rec)
            return rec

        if isinstance(h, zero_op):
            # A zero slot has nothing to compress, so 'fwd' is reported as the
            # ABSOLUTE norm of the reference action -- it should be ~0.
            #
            # Two different situations used to print an identical 0.00e+00:
            #   * no reference exists (no neighbour on that side), nothing was
            #     measured;
            #   * a reference exists and was measured to be zero.
            # Only the second is evidence.  The first now reports None, which
            # prints as '-', and says so in the note.
            if ref_f is None:
                rec['note'] = 'zero (no neighbour, not checked)'
                self.blockErrors.append(rec)
                return rec
            rec['note'] = 'zero (checked)'
            rec['fwd']  = float(torch.linalg.norm(ref_f))
            if ref_a is not None:
                rec['adj'] = float(torch.linalg.norm(ref_a))
            # scale for a zero slot is an absolute norm too, not a ratio
            tol = 1e-10 * float(torch.linalg.norm(W))
            if rec['fwd'] > tol:
                warnings.warn(
                    f"{kind}[{node}] (nSlabs={nSlabs}) is stored as zero_op but "
                    f"its reference action has norm {rec['fwd']:.3e} "
                    f"(tol {tol:.3e}). The zero-slot analysis or the neighbour "
                    "indexing is wrong; no rank will fix this.", UserWarning)
            self.blockErrors.append(rec)
            return rec

        if ref_f is not None:
            rec['fwd'] = self._relerr(h.matmat(W), ref_f)
        if ref_a is not None:
            rec['adj'] = self._relerr(h.rmatmat(Q), ref_a)
        if self.debug_inverse and getattr(h, 'Qlist', None):
            # One solve serves two purposes.
            #   inv      -- consistency of the ULV factors with their own HBS
            #               matrix; independent of compression accuracy.
            #   invscale -- ||M^-1 W|| / ||W||, which with scale = ||M W|| /
            #               ||W|| gives cond = scale * invscale.
            #
            # cond is a LOWER BOUND on the true condition number: a Gaussian
            # probe of t columns underestimates both operator norms, more so
            # for ||M^-1||, whose largest direction is a single singular
            # vector the probe is unlikely to hit squarely.  Useful for
            # orders of magnitude, not for a sharp number.
            Xs = h.solve(W)
            rec['invscale'] = float(torch.linalg.norm(Xs) /
                                    torch.linalg.norm(W))
            if rec['scale'] is not None:
                rec['cond'] = rec['scale'] * rec['invscale']
            rec['inv'] = self._relerr(h.matmat(Xs), W)
            if apply_ref is not None:
                # The same solve measured against the true operator instead
                # of the compressed one.  inv stays near machine precision
                # whatever the rank (the ULV factors are consistent with
                # their own matrix); inv_ref is the number that should move
                # with rank and track delta.
                rec['inv_ref'] = self._relerr(apply_ref(Xs), W)
            del Xs

        self.blockErrors.append(rec)
        return rec

    def print_block_errors(self, per_block=False):
        """Summary of the per-block compression check."""
        if not self.blockErrors:
            print(" no block diagnostics recorded (debug_blocks=0)")
            return
        print(f"\n HBS block compression check "
              f"({self.debug_blocks} probe columns, out of sample)")
        print(f"   {'stage':>5} {'nSlabs':>6} {'kind':>4} {'rk':>5} {'n':>4}"
              f" {'fwd med':>9} {'fwd max':>9} {'adj max':>9} {'inv max':>9}"
              f" {'inv_ref':>9} {'|M|':>9} {'|M^-1|':>9} {'cond':>9}  worst")
        # Zero slots get their own row: grouping them with real blocks made
        # the group inherit the first member's note, so a stage-0 'A' row of
        # seven real blocks was labelled 'not checked' because A[0] is a zero
        # slot -- while the C zero checks, whose group starts with a real
        # block, were invisible.
        key = lambda r: (r['stage'], r['kind'], r.get('note') is not None)
        # sorted, not first-appearance: splitting zero slots into their own
        # group otherwise interleaves them with the real A/C rows
        order = {'B': 0, 'A': 1, 'C': 2}
        seen = sorted({key(r) for r in self.blockErrors},
                      key=lambda k: (k[0], order.get(k[1], 9), k[2]))
        for k in seen:
            grp  = [r for r in self.blockErrors if key(r) == k]
            head = (f"   {k[0]:>5} {grp[0]['nSlabs']:>6} {k[1]:>4}"
                    f" {grp[0]['rk']:>5} {len(grp):>4}")
            col  = lambda n: [r[n] for r in grp if r.get(n) is not None]
            f, a, iv, ivr = col('fwd'), col('adj'), col('inv'), col('inv_ref')
            sc, isc, cd = col('scale'), col('invscale'), col('cond')
            note = grp[0].get('note')
            if not f:
                print(f"{head}   {note or '-'}")
                continue
            med   = sorted(f)[len(f) // 2]
            worst = max(grp, key=lambda r: r.get('fwd') or -1.0)
            # '-' rather than nan for an empty group: A/C blocks are built
            # with compute_ULV=False, so they have no inverse to measure.
            # Printing nan there is indistinguishable from a real numerical
            # failure, which is the one thing this table must not hide.
            mx = lambda v: _fmt(max(v) if v else None)
            print(f"{head}"
                  f" {med:9.2e} {max(f):9.2e}"
                  f" {mx(a)} {mx(iv)} {mx(ivr)} {mx(sc)} {mx(isc)} {mx(cd)}"
                  f"  {worst['kind']}[{worst['node']}]"
                  + (f"  {note}" if note else ""))

        if per_block:
            print("   ---- per block ----")
            for r in self.blockErrors:
                print(f"   nSlabs={r['nSlabs']:3d} {r['kind']}[{r['node']:3d}]"
                      f" rk={r['rk']:4d}"
                      f" fwd={_fmt(r['fwd'])} adj={_fmt(r['adj'])}"
                      f" inv={_fmt(r['inv'])} inv_ref={_fmt(r.get('inv_ref'))}"
                      f" |M|={_fmt(r['scale'])} |M^-1|={_fmt(r.get('invscale'))}"
                      f" cond={_fmt(r.get('cond'))}"
                      + (f"  {r['note']}" if r.get('note') else ""))

    def residency_report(self):
        b = self._blocks
        return dict(
                nBlocks = len(b),
                nFill   = sum(getattr(x,'nFill',0) for x in b),
                nSpill  = sum(getattr(x,'nSpill',0) for x in b),
                GB_H2D  = sum(getattr(x,'bytesH2D',0)for x in b)/10**9,
                GB_D2H  = sum(getattr(x,'bytesD2H',0)for x in b)/10**9,
                GB_total= sum(x.device_nbytes() for x in b)/10**9
                )

    @staticmethod
    def _block_size(h):
        """Full size of a block from its tensor lists, independent of
        whether it is currently resident (unlike device_nbytes())."""
        return sum(t.nbytes
                   for names in getattr(h, '_GROUPS', {}).values()
                   for nm in names
                   for t in (getattr(h, nm, None) or [])
                   if torch.is_tensor(t))

    def traffic(self, verbose=True):
        """Host<->device traffic per (nSlabs, kind, parity) group.

        Ratios are relative to the full block size (_block_size), not to
        device_nbytes(), so they do not depend on what happens to be
        resident when this is called.  Counters are cumulative over the
        blocks' lifetime: call after factorize and again after each solve
        and difference the results to get per-phase cost.

        'par' is the parity of the block's index one level down, where it is
        consumed: odd-indexed operators are staged by both neighbours in the
        current fused builder.  Blocks without a label (the adopted level-0
        inputs) are grouped as 'input'.

        _spill() round trips of raw sample tensors are not counted here.
        """
        # key -> [nblocks, size, H2D, D2H, fills, spills]
        agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
        for h in self._blocks:
            lab = getattr(h, '_label', None)
            mo  = re.match(r'(\w)\[(\d+)\] \(nSlabs=(\d+)\)', lab) if lab else None
            if mo:
                kind, i, ns = mo.group(1), int(mo.group(2)), int(mo.group(3))
                key = (ns, kind, 'odd' if (i // 2) % 2 else 'even')
            else:
                key = (0, 'input', '-')
            a = agg[key]
            a[0] += 1
            a[1] += self._block_size(h)
            a[2] += getattr(h, 'bytesH2D', 0)
            a[3] += getattr(h, 'bytesD2H', 0)
            a[4] += getattr(h, 'nFill', 0)
            a[5] += getattr(h, 'nSpill', 0)

        tot = sum(a[1] for a in agg.values())
        h2d = sum(a[2] for a in agg.values())
        d2h = sum(a[3] for a in agg.values())

        if verbose:
            print(f"{'nSlabs':>6} {'kind':>5} {'par':>4} {'n':>4} {'GB':>7} "
                  f"{'H2D x':>6} {'D2H x':>6} {'fill/blk':>8} {'spill/blk':>9}")
            for k in sorted(agg):
                n, sz, bh, bd, f, s = agg[k]
                print(f"{k[0]:>6} {k[1]:>5} {k[2]:>4} {n:>4} {sz/1e9:7.2f} "
                      f"{bh/max(sz,1):6.2f} {bd/max(sz,1):6.2f} "
                      f"{f/n:8.2f} {s/n:9.2f}")
            print(f"total {tot/1e9:.2f} GB   H2D {h2d/max(tot,1):.2f}x   "
                  f"D2H {d2h/max(tot,1):.2f}x")

        return {'groups': {k: dict(n=v[0], bytes=v[1], H2D=v[2], D2H=v[3],
                                   nFill=v[4], nSpill=v[5])
                           for k, v in agg.items()},
                'total_bytes': tot, 'H2D': h2d, 'D2H': d2h}

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
        self._use(op, ('core',))
        return self._on_dev(op.matmat(X))

    def _apT(self, op, X):
        self.nApply += 1
        self._use(op, ('core',))
        return self._on_dev(op.rmatmat(X))

    def _sv(self, op, X, mode='N'):
        self.nSolve += 1
        self._use(op, ('core', 'ulv'))
        return self._on_dev(op.solve(X, mode=mode))

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

    # ------------------------------------------------------------------
    # pre-flight memory check
    # ------------------------------------------------------------------
    def _block_bytes(self, rk):
        """(core bytes, ULV bytes, apply bytes/col, solve bytes/col) of an HBS
        block this solver builds at rank rk -- from the same level shapes
        HBSMAT allocates its host master from, so these are exact."""
        fac = 4 if self.quad else 2
        Nb, nl, L = self.tree.nleaves, self.nl, self.tree.nlevels
        lv, root = HBSnew._level_shapes(Nb, nl, L, rk, fac)
        nb = lambda lay: 8 * sum(math.prod(sh) for v in lay.values() for sh in v)
        core = nb(HBSnew._core_layout(lv, root))
        ulv = nb(HBSnew._ulv_layout(lv, root))
        R = sum(n_b * n for n_b, n, _ in lv) + root[0] * root[1]
        m = self.m
        return core, ulv, 8 * (2 * R + 4 * m), 8 * (3 * R + 6 * m)

    @staticmethod
    def _host_available():
        """Bytes this process can still allocate on the host: MemAvailable,
        capped by a cgroup (job) limit when there is one."""
        avail = None
        try:
            for line in open('/proc/meminfo'):
                if line.startswith('MemAvailable:'):
                    avail = int(line.split()[1]) * 1024
        except Exception:
            pass
        try:
            lim = open('/sys/fs/cgroup/memory.max').read().strip()
            if lim != 'max':
                used = int(open('/sys/fs/cgroup/memory.current').read())
                room = int(lim) - used
                avail = room if avail is None else min(avail, room)
        except Exception:
            pass
        return avail

    def preflight(self, S_rk_list, T=None, ranks=None, mode='warn', verbose=True):
        """Predict device and host memory for each stage of factorize(), and
        stop before any work if a stage will not fit (mode='raise').

        Device, per stage j (consuming RB[j]):
          sampling   the largest block used at this stage (with its ULV if it
                     is solved with) + one streamed column chunk
                     (_STREAM_BYTES, or one column if that is larger)
          draw       one m x s test matrix while it is drawn and copied down
          construct  one construction chunk (HBStorch._CHUNK_BYTES) + the ULV
                     carries; the next level's samples are placed on the
                     device only when they fit the budget, else on the host
        compared with HBStorch.device_budget (85% of the device by default)
        minus what is already allocated.

        Host: every block built from here on (exact sizes), plus the
        solver's work buffers (13 m x s at most: Om, Psi, the six B/A/C
        samples, two cached solves of 2s columns, one adjoint rhs; 15 when
        cyclic, with a third cached solve) plus, in
        the worst case, construction's spilled samples (8 m x s), against
        the memory available to this process.

        Returns a dict with the per-stage table and the totals.
        """
        GiB = float(2 ** 30)
        m, nSlabs = self.m, len(S_rk_list)
        nstages = nSlabs.bit_length() - 1
        if ranks is None:
            strat = self.rkStrat
            if strat.skip_first_level and T is not None and not all(_is_id(o) for o in T):
                strat = strat.copy(skip_first_level=False)
            ranks = strat.ranks(nstages)
        dev = self.compute_device
        on_gpu = dev.type == 'cuda'
        budget = HBSnew.device_budget(dev) if on_gpu else None
        already = torch.cuda.memory_allocated(dev) if on_gpu else 0
        total_dev = (torch.cuda.get_device_properties(dev).total_memory
                     if on_gpu else None)
        stream = HBSnew._STREAM_BYTES[0] or (1 << 62)
        chunk = HBSnew._CHUNK_BYTES[0] or (1 << 62)

        # stage-0 inputs: their actual sizes
        seen, in_apply, in_solve = set(), 0, 0
        for ops in S_rk_list:
            for op in (ops[0], ops[-1]):
                if id(op) in seen or not isinstance(op, HBSnew.HBSMAT):
                    continue
                seen.add(id(op))
                if not op.Dmats:
                    continue
                core = sum(t.nbytes for nm in op._GROUPS['core'] for t in getattr(op, nm))
                in_apply = max(in_apply, core + max(stream, op._ws_per_col('apply')))
        for op in (T or []):
            if isinstance(op, HBSnew.HBSMAT) and op.Dmats:
                in_solve = max(in_solve, self._block_size(op)
                               + max(stream, op._ws_per_col('solve')))

        rows, host_new = [], 0
        for j, rk in enumerate(ranks):
            nS = nSlabs >> j
            s = self._nsamples(rk)
            unit = m * s * 8
            # blocks used while sampling stage j
            if j == 0:
                samp = max(in_apply, in_solve)
                big = samp
            else:
                rp = ranks[j - 1]
                core, ulv, apc, svc = self._block_bytes(rp)
                use_apply = core + max(stream, apc)
                use_solve = core + ulv + max(stream, svc)
                samp = max(use_apply, use_solve)
                big = core + ulv
            draw = unit + (256 << 20)
            # construction at stage j, rank rk
            core, ulv, _, _ = self._block_bytes(rk)
            fac = 4 if self.quad else 2
            lv, _ = HBSnew._level_shapes(self.tree.nleaves, self.nl,
                                          self.tree.nlevels, rk, fac)
            carries = max(2 * 8 * n_b * k * k for n_b, n, k in lv)
            per_blk = max(8 * (8 * n * s + 8 * n * n) for n_b, n, k in lv)
            cons = max(chunk, per_blk) + 2 * carries
            dev_peak = max(samp, draw, cons)
            # blocks built at stage j
            nodes = nS // 2
            built = 0
            for i in range(0, nS, 2):
                need = self._needs_ulv(i, nS)
                if need or self.compress_diag or not self.skip_unused_ulv:
                    built += core + (ulv if self._want_ulv(need) else 0)
            nAC = nodes - (0 if self.cyclic else 1)
            built += 2 * nAC * (core + (ulv if self._want_ulv(False) else 0))
            host_new += built
            work = (15 if self.cyclic else 13) * unit
            rows.append(dict(stage=j, nSlabs=nS, rk=rk, s=s, unit=unit,
                             largest_block=big, sampling=samp, draw=draw,
                             construct=cons, device_peak=dev_peak,
                             work=work, spill=8 * unit, built=built,
                             cum_built=host_new))

        work_peak = max(r['work'] for r in rows) if rows else 0
        spill_peak = max(r['spill'] for r in rows) if rows else 0
        host_need = host_new + work_peak + (0 if on_gpu else 0)
        host_avail = self._host_available()
        lock = HBSnew.memlock_limit()

        problems = []
        if on_gpu:
            for r in rows:
                if already + r['device_peak'] > budget:
                    what = max(('sampling', r['sampling']), ('draw', r['draw']),
                               ('construct', r['construct']), key=lambda x: x[1])
                    problems.append(
                        f"stage {r['stage']} (nSlabs={r['nSlabs']}, rk={r['rk']}): "
                        f"device peak {(already + r['device_peak'])/GiB:.1f} GiB "
                        f"({what[0]}) exceeds the budget {budget/GiB:.1f} GiB")
        if host_avail is not None and host_need > 0.95 * host_avail:
            problems.append(
                f"host: {host_need/GiB:.0f} GiB needed (new factor storage "
                f"{host_new/GiB:.0f} + work buffers {work_peak/GiB:.0f}), "
                f"{host_avail/GiB:.0f} GiB available")

        if verbose:
            hdr = (f" pre-flight: device {dev}"
                   + (f" budget {budget/GiB:.1f} of {total_dev/GiB:.1f} GiB,"
                      f" {already/GiB:.1f} GiB already allocated" if on_gpu else "")
                   + (f"; host available {host_avail/GiB:.0f} GiB" if host_avail else "")
                   + (f"; lockable {lock/GiB:.0f} GiB" if lock is not None else
                      "; lockable unlimited"))
            print(hdr)
            print(f"   {'stage':>5} {'nSlabs':>6} {'rk':>5} {'s':>6} {'m x s':>7}"
                  f" {'block':>7} {'sample':>7} {'constr':>7} {'device':>7}"
                  f" {'host work':>9} {'new blocks':>10}   (GiB)")
            for r in rows:
                print(f"   {r['stage']:>5} {r['nSlabs']:>6} {r['rk']:>5} {r['s']:>6}"
                      f" {r['unit']/GiB:7.2f} {r['largest_block']/GiB:7.2f}"
                      f" {r['sampling']/GiB:7.2f} {r['construct']/GiB:7.2f}"
                      f" {r['device_peak']/GiB:7.2f} {r['work']/GiB:9.1f}"
                      f" {r['built']/GiB:10.1f}")
            print(f"   new factor storage {host_new/GiB:.0f} GiB; host peak about "
                  f"{host_need/GiB:.0f} GiB (+ up to {spill_peak/GiB:.0f} GiB if "
                  f"construction has to spill samples)")
            if on_gpu:
                print(f"   page-locked first: {work_peak/GiB:.0f} GiB of work buffers "
                      f"(reserved before any block is built); block storage "
                      f"page-locks as far as the driver allows, the rest is "
                      f"ordinary memory (slower to upload, same results)")
            print("   pre-flight: " + ("OK" if not problems else "WILL NOT FIT"))

        rep = dict(stages=rows, host_new=host_new, host_need=host_need,
                   host_available=host_avail, device_budget=budget,
                   device_allocated=already, memlock=lock, problems=problems)
        if problems:
            msg = ("RedBlackSolverHBS pre-flight: this factorization will not fit:\n  "
                   + "\n  ".join(problems)
                   + "\n  Levers: HBStorch._STREAM_BYTES / _CHUNK_BYTES (device "
                     "working sets), the rank schedule or oversampling (block and "
                     "sample sizes).  Pass preflight='warn' to run anyway.")
            if mode == 'raise':
                raise MemoryError(msg)
            warnings.warn(msg, UserWarning)
        self.preflightReport = rep
        return rep

    @property
    def report(self):
        """Placeholder for the dense-reference report of an older version."""
        return _NoReport()

    def factorize(self, S_rk_list, T=None, S_exact=None):
        if S_exact is not None:
            warnings.warn(
                "RedBlackSolverHBS.factorize: S_exact (dense reference "
                "diagnostics) is not part of this version and is ignored.",
                UserWarning, stacklevel=2)
        m      = S_rk_list[0][0].shape[0]
        nSlabs = len(S_rk_list)

        if not ((nSlabs & (nSlabs - 1) == 0) and nSlabs != 0):
            raise ValueError("Number of slabs must be a power of 2.")
        # Blocks kept for a previous solve() would sit in this
        # factorization's memory, and refer to blocks about to be superseded.
        self.release_solve_cache()
        self._drop_slot()
        HBSnew.uv_timers_reset()
        HBSnew._UV_SYNC[0] = (self.compute_device.type == 'cuda')
        # Row order of every host work tensor of the fused builder: the
        # tree's leaf permutation, which every block built here shares, so
        # their streamed operations and compressions need no row gather.
        pl = self.tree.perm_leaf
        self._io = (pl.detach().to('cpu', torch.int64) if torch.is_tensor(pl)
                    else torch.as_tensor(np.asarray(pl), dtype=torch.int64))
        self._dtype = S_rk_list[0][0].dtype

        SiM = [self._adopt(_[0]) for _ in S_rk_list]
        SiP = [self._adopt(_[-1])for _ in S_rk_list]

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
        if strat.skip_first_level and not all(_is_id(op) for op in T):
            warnings.warn(
                "RedBlackSolverHBS: rkStrat has skip_first_level=True but the "
                "level-0 diagonal is not the identity, so stage 0 does invert. "
                "Disabling the skip for this factorization.", UserWarning)
            strat = strat.copy(skip_first_level=False)
        self.rkSchedule = strat.validate(nstages, nl=self.nl,
                                         label='RedBlackSolverHBS')
        print(" " + strat.describe(nstages))
        if self.preflight_mode != 'off':
            self.preflight(S_rk_list, T=T, ranks=self.rkSchedule,
                           mode=self.preflight_mode)
        # The fused builder's host work buffers, sized for the largest stage
        # and page-locked now, before the blocks built below can take the
        # lockable memory: they are what every streamed operation moves.
        # 9 single-width buffers and 2 cached solves of width 2s (3 when
        # cyclic: X for the last odd node is held from node 0 to the end).
        if self.fused and self.rkSchedule:
            s_max = max(self._nsamples(r) for r in self.rkSchedule)
            n1 = m * s_max
            nX = 3 if self.cyclic else 2
            pin = self.compute_device.type == 'cuda'
            nlock = HBSnew.host_pool.reserve([2 * n1] * nX + [n1] * 9, pinned=pin)
            if pin and nlock < 9 + nX:
                warnings.warn(
                    f"RedBlackSolverHBS: only {nlock} of {9 + nX} host work buffers "
                    "could be page-locked; transfers from the others are "
                    "slower.  Check `ulimit -l` / HBStorch._PIN_HOST.",
                    UserWarning)

        l = nSlabs
        j = 0
        self.levelTimes=[]
        self.blockErrors = []
        while l > 1:
            rk = self.rkSchedule[j]
            self._dbg_stage = j
            builder = self._build_level_fused if self.fused else self._build_level
            if self.compute_device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            RB.append(builder(m, l, RB[-1], rk))
            if self.compute_device.type == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t0
            self.levelTimes.append((l,dt,rk))
            print(f" level nSlabs = {l:5d} rk = {rk:4d} {dt:7.2f} s "
                  f"({dt/(l//2):.4f}s/node)")
            j += 1
            l //= 2

        self.nSlabs = nSlabs
        self.RB     = RB
        self._drop_slot()
        HBSnew.host_pool.clear()      # release the work buffers
        # Warn once if any host master could not be page-locked (uploads from
        # those are synchronous and slower).  Reporting only: never raises.
        try:
            self.pinned_report(verbose=False, warn=True)
        except Exception as e:                        # pragma: no cover
            warnings.warn(f"pinned_report failed: {e!r}", UserWarning)

    # ------------------------------------------------------------------
    # _build_level_fused  -- tier-2 shared solves
    # ------------------------------------------------------------------
    def _sync(self):
        if self.compute_device.type=='cuda':
            torch.cuda.synchronize()
    def timing_report(self,per_level=False):
        blocks = [b for b in self._blocks if getattr(b,'_built_by_rb',False)]
        def tot(attr):
            return sum(getattr(b,attr,0) for b in blocks)
        rep = dict(
                nBlocks     = len(blocks),
                setup       = tot('setupTime'),
                null        = tot('nullTime'),
                D           = tot('DTime'),
                ULV         = tot('tULV'),
                blockSolve  = tot('blockSolveTime'),
                compress    = tot('tCompress')
                )
        uv = HBSnew.uv_timers()
        rep['null_qr']    = uv['qr']
        rep['null_basis'] = uv['basis']
        rep['null_setup'] = uv['setup']
        rep['null_calls'] = uv['ncall']
        rep['residual'] = rep['compress']-(rep['null']+rep['D']+rep['ULV']+rep['blockSolve'])
        if per_level and hasattr(self, 'levelTimes'):
            rep['levels'] = list(self.levelTimes)
        return rep
    def print_timing(self):
        r = self.timing_report()
        tot = r['compress'] or 1.0
        print(f"  {r['nBlocks']} blocks, {r['compress']:.2f}s in compression")
        for k in ('null', 'null_qr', 'null_basis', 'null_setup',
                  'D', 'ULV', 'blockSolve', 'setup', 'residual'):
            print(f"    {k:<11s} {r[k]:7.2f}s  {100*r[k]/tot:5.1f}%")
        print(f"    ({r['null_calls']} compute_UV calls, "
              f"{r['null']/max(r['null_calls'],1)*1000:.1f} ms each)")
    # -- host-resident work tensors (see the memory policy note) ---------
    def _hnew(self, ncol):
        """(m, ncol) column-major host buffer in tree order, page-locked when
        computing on a GPU.  Return it with _hfree."""
        return HBSnew.host_empty(self.m, ncol,
                                 pinned=self.compute_device.type == 'cuda')

    @staticmethod
    def _hfree(*Xs):
        for X in Xs:
            if X is not None:
                HBSnew.host_pool.put(X)

    def _to_orig(self, X):
        """Tree-ordered host tensor -> original row order (fallback paths)."""
        Xo = torch.empty((X.shape[0], X.shape[1]), dtype=X.dtype)
        Xo[self._io] = X
        return Xo

    def _from_orig(self, R):
        """Result in original order (any device, numpy) -> tree order, host."""
        R = torch.as_tensor(np.asarray(R) if not torch.is_tensor(R) else R)
        R = R.to('cpu')
        return (R[:, None] if R.ndim == 1 else R)[self._io]

    def _mm(self, op, X, out, alpha=1.0, beta=0.0, C=None, T=False):
        """out <- alpha * op(X) + beta * C   (op^T with T=True; C defaults to
        out).  X, out, C: host tensors in tree order.  HBS blocks run it
        streamed, combining on the device chunk by chunk; the few other
        operator kinds (identity, zero, uncompressed linops) fall back to host
        arithmetic."""
        if isinstance(op, HBSnew.HBSMAT):
            self.nApply += 1
            self._use(op, ('core',))
            f = op.rmatmat if T else op.matmat
            f(X, out=out, alpha=alpha, beta=beta, C=C, io_perm=self._io)
            return out
        if _is_id(op):
            self.nIdSkipped += 1
            R = X
        elif isinstance(op, zero_op):
            R = None
        else:
            self.nApply += 1
            Xo = self._to_orig(X)
            R = self._from_orig(op.rmatmat(Xo) if T else op.matmat(Xo))
            del Xo
        Cc = out if C is None else C
        if beta == 0.0:
            if R is None:
                out.zero_()
            else:
                out.copy_(R)
                if alpha != 1.0:
                    out.mul_(alpha)
        else:
            if Cc is not out:
                out.copy_(Cc)
            if beta != 1.0:
                out.mul_(beta)
            if R is not None:
                out.add_(R, alpha=alpha)
        return out

    def _solve_into(self, op, X, mode='N'):
        """X <- op^{-1} X (op^{-T} with mode='T'), in place, X host in tree
        order.  A no-op for the identity."""
        if _is_id(op):
            self.nIdSkipped += 1
            return X
        self.nSolve += 1
        if isinstance(op, HBSnew.HBSMAT):
            self._use(op, ('core', 'ulv'))
            op.solve(X, mode=mode, out=X, io_perm=self._io)
            return X
        Xo = self._to_orig(X)
        X.copy_(self._from_orig(op.solve(Xo, mode=mode)))
        return X

    def _draw(self, s):
        """An m x s Gaussian test matrix, as a host buffer in tree order.

        Drawn on the compute device from self._tgen in one call, so the
        values are those of torch.randn(m, s) -- identical to earlier
        versions at the same seed -- then permuted and copied down in
        column chunks; the device holds one m x s plus one chunk."""
        dev = self.compute_device
        G = torch.randn(self.m, s, generator=self._tgen, device=dev,
                        dtype=self._tdtype)
        H = self._hnew(s)
        io = self._io.to(dev)
        c = max(1, (256 << 20) // (8 * self.m))
        pending = False
        for j0 in range(0, s, c):
            pending |= HBSnew._from_dev(H[:, j0:j0 + c], G[io, j0:j0 + c])
        if pending:
            torch.cuda.current_stream(dev).synchronize()
        del G
        return H

    def _build_level_fused(self, m, nSlabs, RB_level, rk):
        SiM   = RB_level[0]
        T     = RB_level[1]
        T_hbs = RB_level[2]
        SiP   = RB_level[3]

        cyclic = self.cyclic
        dtype  = self._dtype
        dev    = self.compute_device
        is_cuda = (dev.type == 'cuda')

        s   = self._nsamples(rk)
        # Every m x s tensor below lives on the host, in tree order, and is
        # touched only through streamed block operations (_mm, _solve_into)
        # and the compressions, which stream it too.  The device holds one
        # block plus one column chunk at a time.
        Om  = self._draw(s)
        Psi = self._draw(s)

        # ---------------------------------------------------------------
        # eliminated (odd) nodes: one fused solve each
        #
        #   Xm_k = T_k^{-1} S^-_k Om   feeds B_{k-1} and A_{k+1}
        #   Xp_k = T_k^{-1} S^+_k Om   feeds C_{k-1} and B_{k+1}
        #
        # Xp is skipped for the final odd node in the non-cyclic case: its two
        # consumers are C_{nSlabs-2} (structurally zero, since S^+_{nSlabs-1}
        # = 0) and B_{nSlabs}, which does not exist.
        #
        # Filled lazily by the first retained node that needs it, and freed
        # after its last consumer (node k-1's right branch and node k+1's left
        # branch).  X_k is one (m, 2s) buffer: [Xm_k | Xp_k].
        # ---------------------------------------------------------------
        Xs, Xuses = {}, {}

        def _ensure_X(k):
            if k in Xs:
                return
            need_p = cyclic or (k != nSlabs - 1)
            Xuses[k] = 2 if need_p else 1
            X = self._hnew(2 * s if need_p else s)
            self._mm(SiM[k], Om, out=X[:, :s])
            if need_p:
                self._mm(SiP[k], Om, out=X[:, s:])
            self._solve_into(T_hbs[k], X)        # one solve, up to 2s columns
            Xs[k] = X

        Xm = lambda k: Xs[k][:, :s]
        Xp = lambda k: Xs[k][:, s:]

        def _done_X(k):
            Xuses[k] -= 1
            if Xuses[k] <= 0:
                self._hfree(Xs.pop(k))
                del Xuses[k]

        # Accumulators Y_B, Z_B.  Their first term T_i Om (T_i^T Psi) is
        # Om (Psi) itself when T_i = I; it is then folded into the first
        # update as its C operand instead of being copied.
        def _acc(Ti, src, transpose):
            buf = self._hnew(s)
            if _is_id(Ti):
                self.nIdSkipped += 1
                return [buf, src]
            self._mm(Ti, src, out=buf, T=transpose)
            return [buf, None]

        def _acc_sub(acc, op, X, transpose=False):      # acc -= op X
            self._mm(op, X, out=acc[0], alpha=-1.0, beta=1.0, C=acc[1],
                     T=transpose)
            acc[1] = None

        def _acc_done(acc):
            if acc[1] is not None:                      # no update ran
                acc[0].copy_(acc[1])
            return acc[0]

        def _neg(op, X, transpose=False):               # -op X, fresh buffer
            out = self._hnew(s)
            return self._mm(op, X, out=out, alpha=-1.0, T=transpose)

        # ---------------------------------------------------------------
        # debug probes.  W/Q are independent of Om/Psi, so the checks below
        # are out of sample.  They are t columns wide, so they stay on the
        # device in original order and use the plain _ap/_sv wrappers.
        # ---------------------------------------------------------------
        dbg = self.debug_blocks > 0
        W = self._dbg_probe(self.debug_blocks) if dbg else None
        Q = self._dbg_probe(self.debug_blocks) if dbg else None
        Pm, Pp = {}, {}

        def _ensure_P(k):
            # Unlike _ensure_X this builds Pp for EVERY k, including the last
            # odd node where Xp has no consumer: that makes the C_is_zero slot
            # checkable.
            if k in Pm:
                return
            if _is_id(T_hbs[k]):
                Pm[k] = self._ap(SiM[k], W)
                Pp[k] = self._ap(SiP[k], W)
            else:
                Pm[k] = self._sv(T_hbs[k], self._ap(SiM[k], W))
                Pp[k] = self._sv(T_hbs[k], self._ap(SiP[k], W))

        def _refs_init(i):
            return {'B': [W.clone() if _is_id(T[i]) else self._ap(T[i], W),
                          Q.clone() if _is_id(T[i]) else self._apT(T[i], Q)],
                    'A': [None, None],
                    'C': [None, None]}

        dbg_true = self.debug_true_inverse and dbg

        def _apply_B(i, X, has_left, has_right, kL, kR):
            """B_i X with the UNCOMPRESSED chain (inv_ref only, t columns)."""
            y = X.clone() if _is_id(T[i]) else self._ap(T[i], X)
            if has_right:
                t = self._ap(SiM[kR], X)
                if not _is_id(T_hbs[kR]):
                    t = self._sv(T_hbs[kR], t)
                y = y - self._ap(SiP[i], t)
            if has_left:
                t = self._ap(SiP[kL], X)
                if not _is_id(T_hbs[kL]):
                    t = self._sv(T_hbs[kL], t)
                y = y - self._ap(SiM[i], t)
            return y

        # ---------------------------------------------------------------
        # retained (even) nodes
        # ---------------------------------------------------------------
        B_i, T_hbs_new, A_i, C_i = [], [], [], []

        for i in range(0, nSlabs, 2):
            has_left  = cyclic or i > 0
            has_right = cyclic or i < nSlabs - 1
            kL = (i - 1) % nSlabs
            kR = (i + 1) % nSlabs
            self._sync(); t0 = time.time()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(dev)
            # A_0 is zero exactly when there is no left neighbour.
            # C_{nSlabs-2} is zero because S^+_{nSlabs-1} = 0, even though the
            # right neighbour exists.
            A_is_zero = (not cyclic) and i == 0
            C_is_zero = (not cyclic) and i == nSlabs - 2

            YB = _acc(T[i], Om, False)
            ZB = _acc(T[i], Psi, True)
            Y_A = Y_C = Z_A = Z_C = None
            refs = _refs_init(i) if dbg else None

            if has_right:
                _ensure_X(kR)

                # forward: S^+_i Xm (B term) and S^+_i Xp (C term)
                _acc_sub(YB, SiP[i], Xm(kR))
                if not C_is_zero:
                    Y_C = _neg(SiP[i], Xp(kR))
                _done_X(kR)

                # adjoint: t^+ = T_{i+1}^{-T} (S^+_i)^T Psi serves B and C
                tp = self._hnew(s)
                self._mm(SiP[i], Psi, out=tp, T=True)
                self._solve_into(T_hbs[kR], tp, mode='T')

                if dbg:
                    # fC/aC are computed even when C_is_zero: that is the check
                    # on the zero-slot analysis.
                    _ensure_P(kR)
                    refs['B'][0] = refs['B'][0] - self._ap(SiP[i], Pm[kR])
                    refs['C'][0] = -self._ap(SiP[i], Pp[kR])
                    rp = self._apT(SiP[i], Q)
                    tpr = rp if _is_id(T_hbs[kR]) else self._sv(T_hbs[kR], rp,
                                                                mode='T')
                    refs['B'][1] = refs['B'][1] - self._apT(SiM[kR], tpr)
                    refs['C'][1] = -self._apT(SiP[kR], tpr)
                    del rp, tpr

                _acc_sub(ZB, SiM[kR], tp, transpose=True)
                if not C_is_zero:
                    Z_C = _neg(SiP[kR], tp, transpose=True)
                self._hfree(tp)

            if has_left:
                _ensure_X(kL)

                # forward: S^-_i Xp (B term) and S^-_i Xm (A term)
                _acc_sub(YB, SiM[i], Xp(kL))
                Y_A = _neg(SiM[i], Xm(kL))
                _done_X(kL)

                # adjoint: t^- = T_{i-1}^{-T} (S^-_i)^T Psi serves B and A
                tm = self._hnew(s)
                self._mm(SiM[i], Psi, out=tm, T=True)
                self._solve_into(T_hbs[kL], tm, mode='T')

                if dbg:
                    _ensure_P(kL)
                    refs['B'][0] = refs['B'][0] - self._ap(SiM[i], Pp[kL])
                    refs['A'][0] = -self._ap(SiM[i], Pm[kL])
                    rm = self._apT(SiM[i], Q)
                    tmr = rm if _is_id(T_hbs[kL]) else self._sv(T_hbs[kL], rm,
                                                                mode='T')
                    refs['B'][1] = refs['B'][1] - self._apT(SiP[kL], tmr)
                    refs['A'][1] = -self._apT(SiM[kL], tmr)
                    del rm, tmr

                _acc_sub(ZB, SiP[kL], tm, transpose=True)
                Z_A = _neg(SiM[kL], tm, transpose=True)
                self._hfree(tm)

            Y_B, Z_B = _acc_done(YB), _acc_done(ZB)
            del YB, ZB

            self._sync(); t1 = time.time()
            peak_sample = (torch.cuda.max_memory_allocated(dev)
                           if is_cuda else 0)
            # ---- compress from the shared samples ----------------------
            need_ULV = self._needs_ulv(i, nSlabs)

            if need_ULV or self.compress_diag or not self.skip_unused_ulv:
                B_hbs = self._hbs_from_samples(rk, Om, Psi, Y_B, Z_B,
                                               compute_ULV=need_ULV,
                                               label=f"B[{i}] (nSlabs={nSlabs})",
                                               io_perm=self._io)
            else:
                # compress_diag=False hands the uncompressed RB_linop to the
                # next level as T[i], so this slot is read by nobody: skip
                # the compression itself, not just the factorization.
                B_hbs = self._dead_diag(f"B[{i}] (nSlabs={nSlabs})")
            self._hfree(Y_B, Z_B)
            if dbg:
                self._check_block(
                    B_hbs, 'B', nSlabs, i, rk, W, Q, *refs['B'],
                    apply_ref=(lambda X: _apply_B(i, X, has_left, has_right,
                                                  kL, kR)) if dbg_true else None)
                self._drop_slot()
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
                                                   label=f"A[{i}] (nSlabs={nSlabs})",
                                                   io_perm=self._io))
            self._hfree(Y_A, Z_A)
            if dbg:
                self._check_block(A_i[-1], 'A', nSlabs, i, rk, W, Q, *refs['A'])
                self._drop_slot()
            C_i.append(zero_op(m, dtype) if C_is_zero
                       else self._hbs_from_samples(rk, Om, Psi, Y_C, Z_C,
                                                   compute_ULV=False,
                                                   label=f"C[{i}] (nSlabs={nSlabs})",
                                                   io_perm=self._io))
            self._hfree(Y_C, Z_C)
            if dbg:
                self._check_block(C_i[-1], 'C', nSlabs, i, rk, W, Q, *refs['C'])
                self._drop_slot()
                del refs

            self._sync(); t2 = time.time()
            lent = HBSnew.host_pool.nbytes[0]
            if is_cuda:
                print(f"node {i:3d}: sample {t1-t0:6.2f}s"
                      f" construct {t2-t1:6.2f}s (B+A+C)"
                      f" | device alloc {torch.cuda.memory_allocated(dev)/2**30:5.2f}"
                      f" peak sample {peak_sample/2**30:5.2f}"
                      f" node {torch.cuda.max_memory_allocated(dev)/2**30:5.2f} GB"
                      f" | host buffers {lent/2**30:6.2f} GB")
            else:
                print(f"node {i:3d}: sample {t1-t0:6.2f}s"
                      f" construct {t2-t1:6.2f}s (B+A+C)")

        self._drop_slot()
        if Xs:
            # Every X_k is freed by its last consumer; anything left here is
            # a bookkeeping slip.  Free it rather than stop the run.
            warnings.warn(f"_build_level_fused: cached solves left over for "
                          f"k={sorted(Xs)} at nSlabs={nSlabs}; freed.",
                          UserWarning)
            for X in Xs.values():
                self._hfree(X)
            Xs.clear()
        self._hfree(Om, Psi)
        Pm.clear(); Pp.clear()
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
        dev = self.compute_device
        input_is_numpy = isinstance(rhs,np.ndarray)
        was_vector = (np.asarray(rhs).ndim == 1 if input_is_numpy else rhs.ndim ==1 )
        # ---- forward reduction ----------------------------------------
        v0 = torch.as_tensor(rhs, dtype=self._tdtype, device=dev)
        if was_vector:
            v0 = v0[:,None]
        nrhs = v0.shape[1]
        vPrimes = [v0.clone()]

        # Blocks are staged per use through _ap/_sv (one block at a time), or
        # kept on the device across solves with solve_keep_cache=True.
        self._keep_all = self.solve_keep_cache
        ap, sv = self._ap, self._sv

        try:
            for l in range(len(RB) - 1):
                SiM, _, T_hbs, SiP = RB[l]

                nSlabs   = len(SiM)
                nReduced = nSlabs // 2
                vPrev    = vPrimes[-1]
                # every block is written below, so no zero fill
                vPrime   = torch.empty(m * nReduced, nrhs,dtype=self._tdtype,device=dev)

                # T_k^{-1} vPrev_k for odd k is read by node k-1 (as `next`) and
                # by node k+1 (as `prev`) -- the same solve on the same rhs.
                # Solve once and drop the result after its last reader.  In the
                # non-cyclic case the last odd node has a single reader.
                Tinv, uses = {}, {}

                def _Tinv(k):
                    if k not in Tinv:
                        blk = vPrev[k*m:(k+1)*m, :]
                        if _is_id(T_hbs[k]):
                            Tinv[k] = blk           # read-only use: no clone needed
                            self.nIdSkipped += 1
                        else:
                            Tinv[k] = sv(T_hbs[k], blk)
                        uses[k] = 2 if (self.cyclic or k != nSlabs - 1) else 1
                    x = Tinv[k]
                    uses[k] -= 1
                    if uses[k] == 0:
                        del Tinv[k], uses[k]
                    return x

                for j in range(nReduced):
                    i = 2 * j

                    prev = (i - 1) % nSlabs if (self.cyclic or i > 0)          else None
                    next = (i + 1) % nSlabs if (self.cyclic or i < nSlabs - 1) else None

                    # out-of-place updates below; the slice assignment copies
                    contrib = vPrev[i*m:(i+1)*m, :]
                    if prev is not None:
                        contrib = contrib - ap(SiM[i], _Tinv(prev))
                    if next is not None:
                        contrib = contrib - ap(SiP[i], _Tinv(next))

                    vPrime[j*m:(j+1)*m,:] = contrib

                assert not Tinv, "forward reduction left cached solves unconsumed"
                vPrimes.append(vPrime)

            # ---- coarsest solve ---------------------------------------
            vPrimes[-1] = sv(RB[-1][2][0], vPrimes[-1])

            # ---- back substitution ------------------------------------
            for l in range(len(RB) - 1, 0, -1):
                SiM, _, T_hbs, SiP = RB[l - 1]

                nSlabs   = len(SiM)
                nReduced = nSlabs // 2

                for j in range(nReduced):
                    i = 2 * j

                    vPrimes[l-1][i*m:(i+1)*m] = vPrimes[l][j*m:(j+1)*m]

                    next_j = (j + 1) % nReduced
                    contrib = ap(SiM[i+1], vPrimes[l][j*m:(j+1)*m,:])
                    if self.cyclic or j + 1 < nReduced:
                        contrib = contrib + ap(SiP[i+1], vPrimes[l][next_j*m:(next_j+1)*m, :])

                    blk = vPrimes[l-1][(i+1)*m:(i+2)*m] - contrib
                    vPrimes[l-1][(i+1)*m:(i+2)*m,:] = sv(T_hbs[i+1], blk)
        finally:
            # Runs on error too, so a failed solve never leaves a block on
            # the device unless solve_keep_cache asked for it.
            self._keep_all = False
            self._drop_slot()

        out = vPrimes[0]
        if was_vector:
            out = out[:,0]
        if input_is_numpy:
            return out.detach().cpu().numpy()
        return out

    def release_solve_cache(self):
        """Drop every block kept on the device by solve_keep_cache=True.
        A no-op otherwise."""
        kept, self._kept = self._kept, {}
        for op in kept.values():
            op.evict()

    # ------------------------------------------------------------------
    # host-master page-locking check
    # ------------------------------------------------------------------
    def pinned_report(self, verbose=True, warn=False):
        """How much of the factorization's host master memory is page-locked.

        HBStorch page-locks each block's host master with cudaHostRegister
        (_HostBuffer).  If registration fails -- _PIN_HOST off, driver
        refusal, not enough lockable memory -- it silently falls back to
        pageable memory, and every re-stage of that block becomes a slower,
        synchronous copy.  Every block has its master from construction on.
        """
        n_pin = n_page = 0
        b_pin = b_page = 0
        for h in self._blocks:
            host = getattr(h, '_host', None)
            if not isinstance(host, dict):
                continue
            for rec in host.values():
                if not rec:
                    continue
                for buf in rec.get('bufs', {}).values():
                    t = getattr(buf, 't', None)
                    if t is None or t.numel() == 0:
                        continue
                    nb = t.numel() * t.element_size()
                    if getattr(buf, 'registered', False):
                        n_pin += 1
                        b_pin += nb
                    else:
                        n_page += 1
                        b_page += nb
        rep = dict(pin_host_flag=bool(getattr(HBSnew, '_PIN_HOST', [None])[0]),
                   n_pinned=n_pin, n_pageable=n_page,
                   GB_pinned=b_pin / 1e9, GB_pageable=b_page / 1e9)
        if verbose:
            print(f"  host masters: {rep['GB_pinned']:.2f} GB page-locked in "
                  f"{n_pin} buffers, {rep['GB_pageable']:.2f} GB pageable in "
                  f"{n_page} buffers (_PIN_HOST={rep['pin_host_flag']})")
        if warn and n_page:
            warnings.warn(
                f"RedBlackSolverHBS: {rep['GB_pageable']:.2f} GB of host "
                f"masters in {n_page} buffers are NOT page-locked "
                f"(_PIN_HOST={rep['pin_host_flag']}).  Re-staging those "
                "blocks is synchronous and roughly half as fast.  Usual "
                "causes: HBSnew._PIN_HOST[0] set False, or the OS lock limit "
                "(ulimit -l) / available RAM too low for cudaHostRegister.",
                UserWarning)
        return rep

    def footprint(self, verbose=True):
        """Size of the factorization, by group and by tensor list, and how
        much of it is currently resident on the compute device.

        The per-list breakdown decides which storage lever is worth pulling:
        Dmats dominating points at the native-to-sibling-B conversion (each
        parent's 2rk x 2rk discrepancy block becomes two rk x rk sibling
        blocks), Umats/Vmats dominating points at per-level rank truncation."""
        tot = {'core': 0, 'ulv': 0}
        res = {'core': 0, 'ulv': 0}
        sub = {}
        nblk = 0

        for h in self._blocks:
            if not getattr(h, '_built_by_rb', False) or not hasattr(h, '_GROUPS'):
                continue
            nblk += 1
            for grp, names in h._GROUPS.items():
                gtot = 0
                for nm in names:
                    lst = getattr(h, nm, None) or []
                    n = sum(t.nbytes for t in lst if torch.is_tensor(t))
                    sub[(grp, nm)] = sub.get((grp, nm), 0) + n
                    gtot += n
                tot[grp] = tot.get(grp, 0) + gtot
                if h._resident.get(grp) is not None:
                    res[grp] = res.get(grp, 0) + gtot

        if verbose:
            print(f"  {nblk} solver-owned blocks")
            for grp in tot:
                print(f"    {grp:5s} total {tot[grp]/2**30:6.2f} GB   "
                      f"resident {res[grp]/2**30:6.2f} GB")
                for (g, nm), n in sorted(sub.items(), key=lambda kv: -kv[1]):
                    if g == grp and n:
                        print(f"      {nm:8s} {n/2**30:6.2f} GB  "
                              f"{100*n/max(tot[grp],1):5.1f}%")
            allt = sum(tot.values())
            print(f"    {'all':5s} total {allt/2**30:6.2f} GB   "
                  f"resident {sum(res.values())/2**30:6.2f} GB")
            if self.compute_device.type == 'cuda':
                free, cap = torch.cuda.mem_get_info(self.compute_device)
                print(f"    device capacity {cap/2**30:6.2f} GB, "
                      f"free {free/2**30:6.2f} GB")

        return {'total': tot, 'resident': res, 'by_list': sub}
