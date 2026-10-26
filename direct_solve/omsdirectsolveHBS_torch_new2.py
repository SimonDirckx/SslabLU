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
# Memory: the solver manages none.  Every block is an HBSMAT, which keeps its
# factors in host memory and streams them through the compute device for the
# duration of each construct / matmat / rmatmat / solve.  What the solver
# itself allocates are its work tensors (test matrices, samples, cached
# solves, reduced right-hand sides), all m x (columns) and all on
# `work_device`; they are freed by ordinary Python scoping.
#
# Row order: every block shares one tree, so every work tensor is kept in
# tree order and every HBSMAT call gets io_perm=perm, i.e. no row gather.
# Only solve() permutes, once on the way in and once on the way out.
# ---------------------------------------------------------------------------


class RedBlackSolverHBS(DirectSolver):
    """Block-tridiagonal solver by cyclic reduction (red-black) on HBS blocks.

    System, with the identity on the diagonal and one block row per slab:

        S^-_i u_{i-1} + u_i + S^+_i u_{i+1} = f_i,    i = 0 .. nSlabs-1,

    cyclic: indices mod nSlabs; otherwise S^-_0 = S^+_{nSlabs-1} = 0.  A zero
    block is None, and so is the identity diagonal T of level 0.

    Layout (which parity is retained at every level):

      'symmetric'  non-cyclic, nSlabs = 2^L - 1 (i.e. 2^L strips between the
                   two physical walls).  The even nodes are eliminated and the
                   odd ones kept, so both end nodes go first, every retained
                   node has a neighbour on each side at every level, all
                   coarse slabs are symmetric, and the last node standing is
                   the central interface (nSlabs-1)//2.
      'legacy'     nSlabs = 2^L (required for cyclic).  The odd nodes are
                   eliminated; non-cyclic, node 0 survives, so coarse slabs at
                   the left end are lopsided.
      'auto'       (default) symmetric if non-cyclic and nSlabs + 1 is a power
                   of 2, otherwise legacy.

    self.level_nodes[l] lists the original indices of the nodes present at
    level l; self.level_nodes[-1] == [(nSlabs-1)//2] for 'symmetric'.

    Each level eliminates the nodes of the other parity.  Retained node i
    becomes

        B_i = T_i - S^+_i T_{i+1}^{-1} S^-_{i+1} - S^-_i T_{i-1}^{-1} S^+_{i-1}
        A_i =     - S^-_i T_{i-1}^{-1} S^-_{i-1}          (next level's S^-)
        C_i =     - S^+_i T_{i+1}^{-1} S^+_{i+1}          (next level's S^+)

    compressed from samples that share one Om, Psi per level (fused):
    Xm_k = T_k^{-1} S^-_k Om and Xp_k = T_k^{-1} S^+_k Om are one solve per
    eliminated node k, used by both neighbours; the adjoint samples share
    t+ = T_{i+1}^{-T} (S^+_i)^T Psi and t- = T_{i-1}^{-T} (S^-_i)^T Psi.

    A ULV factorization is computed only for the B_i that get inverted: those
    that are eliminated one level down (i % 4 == 2 legacy, i % 4 == 1
    symmetric) and the coarsest one.
    Cyclic, at two nodes: the surviving node's A_0 and C_0 couple it to
    itself, so they are folded into B_0 (sampled together, one compression).

    normalized=True keeps the identity on the diagonal at EVERY level, so
    each level is again "identity + transfer maps" and the coarse maps are
    the transfer maps of the wider slabs.  With level-l maps G (G = S at
    level 0), retained node i gets

        M_i     = I - G^+_i G^-_{i+1} - G^-_i G^+_{i-1}     (HBS + ULV)
        G'^-_i  = - M_i^{-1} G^-_i G^-_{i-1}                (next level's S^-)
        G'^+_i  = - M_i^{-1} G^+_i G^+_{i+1}                (next level's S^+)

    Sampling needs no solves for M_i (the diagonal is I); G'^-_i and G'^+_i
    share one M_i^{-1} solve on [G^-_i Xm | G^+_i Xp] and one M_i^{-T} solve
    on Psi.  RB[l+1] holds (G'^-, M, G'^+): the middle slot is the block
    whose inverse forms level l+1's rhs, not a diagonal.  The solve:

        forward:  g'_j = M_j^{-1} (g_i - G^+_i g_{i+1} - G^-_i g_{i-1})
        back:     u_k  = g_k - G^-_k u_{k-1} - G^+_k u_{k+1}

    i.e. one ULV solve per retained node, none in the back pass.  The rhs
    uses exactly the M~^{-1} that the G' samples went through.

    store_inverse=True (normalized only) additionally compresses
    Minv_i = M_i^{-1} into its own HBS from the shared solve (the Om block is
    appended to it) and keeps that instead of M_i, so the solve is applies
    only.  It costs one extra compression per retained node and adds one
    approximation layer (Minv is a compression of M~^{-1}); worth it only if
    applying Minv is much cheaper than a ULV solve with M.

    Every stage, stage 0 included, samples through the inverse of a block it
    has just compressed, so for a like-for-like rank schedule against the
    unnormalized variant pass an rkStrat with skip_first_level=False.

    rk: int or rkStrat (rank per stage).  device: compute device of the
    blocks built here (and of the input blocks, which are moved to it).
    work_device: where the work tensors live.  dbg >= 1 records, per level,
    [(kind, node, t_sample, t_compress), ...] in self.blockTimes and prints
    it after each level.
    """

    def __init__(self, m, rk, tree, quad, cyclic=False, seed=None,
                 device='cpu', compute_device=None, work_device='cpu',
                 oversample=None, dbg=0, layout='auto', normalized=False,
                 store_inverse=False):
        super().__init__(m, cyclic)
        self.m, self.cyclic = m, cyclic
        self.normalized = bool(normalized)
        if store_inverse and not self.normalized:
            raise ValueError("store_inverse applies to normalized=True only")
        self.store_inverse = bool(store_inverse)
        if layout not in ('auto', 'symmetric', 'legacy'):
            raise ValueError("layout must be 'auto', 'symmetric' or 'legacy', "
                             f"got {layout!r}")
        if layout == 'symmetric' and cyclic:
            raise ValueError("layout='symmetric' needs physical walls; the "
                             "cyclic problem has no centre (use 'legacy')")
        self.layout = layout
        self._o = 0                       # retained parity, set in factorize
        self.level_nodes = []
        self.rkStrat = rkStrat.coerce(rk)
        self.rk = self.rkStrat.rank(0)
        self.tree, self.quad = tree, quad
        self.fac = 4 if quad else 2
        self.compute_device = _resolve_device(
            compute_device if compute_device is not None else device)
        self.work_device = torch.device(work_device)
        self.oversample = oversample
        self.dbg = dbg
        self.dtype = torch.float64
        pl = tree.perm_leaf
        self.perm = (pl.detach().to('cpu', torch.int64) if torch.is_tensor(pl)
                     else torch.as_tensor(np.asarray(pl), dtype=torch.int64))
        self._gen = torch.Generator(device=self.work_device)
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
        """s = max(fac*rk, nl) + rk + p: the hard floor (null space of Om
        large enough for rank rk at the worst level) plus oversampling."""
        return 3*max(rk, self.nl) + rk + self.oversampling(rk)

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
        return torch.empty(self.m, ncol, dtype=self.dtype,
                           device=self.work_device)

    def _clock(self):
        if self.dbg and self.compute_device.type == 'cuda':
            torch.cuda.synchronize(self.compute_device)
        return time.perf_counter()

    def _compress(self, rk, Om, Psi, Y, Z, ulv):
        h = HBSnew.HBSMAT(device=self.compute_device, tree=self.tree,
                          quad=self.quad)
        h.construct(rk, Om, Psi, Y, Z, compute_ULV=ulv)
        return h

    # -- layout ----------------------------------------------------------------

    @staticmethod
    def _pow2(n):
        return n >= 1 and not n & (n - 1)

    def _parity(self, nSlabs):
        """Retained parity o: 1 = symmetric (keep odd), 0 = legacy (keep even)."""
        sym_ok = not self.cyclic and self._pow2(nSlabs + 1)
        leg_ok = nSlabs >= 2 and self._pow2(nSlabs)
        if self.layout == 'symmetric':
            if not sym_ok:
                raise ValueError(
                    f"layout='symmetric' needs nSlabs = 2^L - 1 double slabs "
                    f"(2^L strips), got {nSlabs}")
            return 1
        if self.layout == 'legacy':
            if not leg_ok:
                raise ValueError(f"layout='legacy' needs nSlabs = 2^L >= 2, "
                                 f"got {nSlabs}")
            return 0
        if sym_ok:
            return 1
        if leg_ok:
            return 0
        raise ValueError(
            f"nSlabs = {nSlabs}: need 2^L - 1 (symmetric, non-cyclic) or 2^L "
            "(legacy / cyclic) double slabs")

    # -- factorize -------------------------------------------------------------

    def factorize(self, S_rk_list, T=None):
        if T is not None:
            raise NotImplementedError(
                "RedBlackSolverHBS supports the identity diagonal only (T=None)")
        nSlabs = len(S_rk_list)
        self._o = self._parity(nSlabs)
        # End slabs of a non-cyclic decomposition carry ONE block (their
        # physical wall has no map), so S[0] and S[-1] are the same operator
        # there; the Nones below drop the duplicate, they discard nothing.
        SiM = [S[0] if len(S) else None for S in S_rk_list]
        SiP = [S[-1] if len(S) else None for S in S_rk_list]
        if not self.cyclic:
            SiM[0] = SiP[-1] = None
        for op in SiM + SiP:
            if op is not None:
                op.compute_device = self.compute_device
        # node bookkeeping: which original interfaces are present per level
        nodes = list(range(nSlabs))
        self.level_nodes = [nodes]
        while len(nodes) > 1:
            nodes = nodes[self._o::2]
            self.level_nodes.append(nodes)
        nstages = len(self.level_nodes) - 1
        ranks = self.rkStrat.validate(nstages, nl=self.nl,
                                      label='RedBlackSolverHBS')
        RB = [(SiM, [None] * nSlabs, SiP)]
        self.blockTimes, self.levelTimes = [], []
        for rk in ranks:
            t0 = self._clock()
            build = (self._build_level_normalized if self.normalized
                     else self._build_level)
            RB.append(build(*RB[-1], rk))
            dt = self._clock() - t0
            self.levelTimes.append((len(RB[-2][0]), dt, rk))
            if self.dbg >= 1:
                self._report_level(len(self.levelTimes) - 1)
        self.RB = RB

    def _build_level(self, SiM, T, SiP, rk):
        N, cyc = len(SiM), self.cyclic
        s = self._nsamples(rk)
        Om = torch.randn(self.m, s, generator=self._gen, dtype=self.dtype,
                         device=self.work_device)
        Psi = torch.randn(self.m, s, generator=self._gen, dtype=self.dtype,
                          device=self.work_device)
        ap, sv = self._ap, self._sv

        o = self._o                      # retained parity
        nnext = len(range(o, N, 2))      # nodes at the next level

        # eliminated node k: X_k = T_k^{-1} [S^-_k Om | S^+_k Om], one solve
        # over whichever halves exist (an end node has only one), read by
        # nodes k-1 and k+1 and dropped after its last reader
        Xs, uses = {}, {}

        def X(k):
            if k not in Xs:
                halves = [op for op in (SiM[k], SiP[k]) if op is not None]
                Xk = self._new(len(halves) * s)
                for j, op in enumerate(halves):
                    ap(op, Om, out=Xk[:, j * s:(j + 1) * s])
                off_m = 0 if SiM[k] is not None else None
                off_p = (s if SiM[k] is not None else 0) \
                    if SiP[k] is not None else None
                Xs[k] = (sv(T[k], Xk), off_m, off_p)
                uses[k] = 2 if cyc else int(k > 0) + int(k < N - 1)
            return Xs[k]

        def Xm(k):                       # T_k^{-1} S^-_k Om
            Xk, off, _ = X(k)
            return Xk[:, off:off + s]

        def Xp(k):                       # T_k^{-1} S^+_k Om
            Xk, _, off = X(k)
            return Xk[:, off:off + s]

        def done(k):
            uses[k] -= 1
            if not uses[k]:
                del Xs[k], uses[k]

        # cyclic with two nodes: the one node left couples to itself through
        # A_0 and C_0, so they are folded into its diagonal B_0
        fold = cyc and N == 2
        times, A, B, C = [], [], [], []
        for i in range(o, N, 2):
            kL, kR = (i - 1) % N, (i + 1) % N
            has_left = cyc or i > 0      # always True for 'symmetric'
            # inverted later iff eliminated at the next level, or the last one
            ulv = ((i - o) // 2) % 2 != o or nnext == 1

            # -- samples of B_i (and the solves shared with A_i, C_i) --------
            t0 = self._clock()
            YB = Om.clone() if T[i] is None else ap(T[i], Om)
            ZB = Psi.clone() if T[i] is None else ap(T[i], Psi, T=True)
            ap(SiP[i], Xm(kR), out=YB, alpha=-1.0, beta=1.0)
            tp = sv(T[kR], ap(SiP[i], Psi, T=True), 'T')
            ap(SiM[kR], tp, out=ZB, alpha=-1.0, beta=1.0, T=True)
            if has_left:
                ap(SiM[i], Xp(kL), out=YB, alpha=-1.0, beta=1.0)
                tm = sv(T[kL], ap(SiM[i], Psi, T=True), 'T')
                ap(SiP[kL], tm, out=ZB, alpha=-1.0, beta=1.0, T=True)
            if fold:                                     # B_0 + A_0 + C_0
                ap(SiM[i], Xm(kL), out=YB, alpha=-1.0, beta=1.0)
                ap(SiM[kL], tm, out=ZB, alpha=-1.0, beta=1.0, T=True)
                ap(SiP[i], Xp(kR), out=YB, alpha=-1.0, beta=1.0)
                ap(SiP[kR], tp, out=ZB, alpha=-1.0, beta=1.0, T=True)
            t1 = self._clock()
            B.append(self._compress(rk, Om, Psi, YB, ZB, ulv))
            del YB, ZB
            times.append(('B', i, t1 - t0, self._clock() - t1))

            # -- A_i = -S^-_i T_{i-1}^{-1} S^-_{i-1}: zero without a left side
            if (has_left and not fold and SiM[i] is not None
                    and SiM[kL] is not None):
                t0 = self._clock()
                Y = ap(SiM[i], Xm(kL), alpha=-1.0)
                Z = ap(SiM[kL], tm, alpha=-1.0, T=True)
                t1 = self._clock()
                A.append(self._compress(rk, Om, Psi, Y, Z, False))
                del Y, Z
                times.append(('A', i, t1 - t0, self._clock() - t1))
            else:
                A.append(None)

            # -- C_i = -S^+_i T_{i+1}^{-1} S^+_{i+1}: zero if S^+_{i+1} is ---
            if not fold and SiP[i] is not None and SiP[kR] is not None:
                t0 = self._clock()
                Y = ap(SiP[i], Xp(kR), alpha=-1.0)
                Z = ap(SiP[kR], tp, alpha=-1.0, T=True)
                t1 = self._clock()
                C.append(self._compress(rk, Om, Psi, Y, Z, False))
                del Y, Z
                times.append(('C', i, t1 - t0, self._clock() - t1))
            else:
                C.append(None)

            done(kR)
            if has_left:
                done(kL)
        self.blockTimes.append(times)
        return A, B, C

    def _build_level_normalized(self, SiM, Minv_prev, SiP, rk):
        """One level of the normalized reduction (see class docstring).

        The diagonal of the incoming level is the identity; Minv_prev (the
        inverses that formed this level's rhs) is not used here.  Returns
        (G'^-, Minv, G'^+) for the retained nodes, in order.
        """
        N, cyc = len(SiM), self.cyclic
        s = self._nsamples(rk)
        Om = torch.randn(self.m, s, generator=self._gen, dtype=self.dtype,
                         device=self.work_device)
        Psi = torch.randn(self.m, s, generator=self._gen, dtype=self.dtype,
                          device=self.work_device)
        ap, sv = self._ap, self._sv
        o = self._o

        # eliminated node k: [G^-_k Om | G^+_k Om] (no solve: diagonal is I),
        # read by nodes k-1 and k+1 and dropped after its last reader
        Xs, uses = {}, {}

        def X(k):
            if k not in Xs:
                halves = [op for op in (SiM[k], SiP[k]) if op is not None]
                Xk = self._new(len(halves) * s)
                for j, op in enumerate(halves):
                    ap(op, Om, out=Xk[:, j * s:(j + 1) * s])
                off_m = 0 if SiM[k] is not None else None
                off_p = (s if SiM[k] is not None else 0) \
                    if SiP[k] is not None else None
                Xs[k] = (Xk, off_m, off_p)
                uses[k] = 2 if cyc else int(k > 0) + int(k < N - 1)
            return Xs[k]

        def Xm(k):                       # G^-_k Om
            Xk, off, _ = X(k)
            return Xk[:, off:off + s]

        def Xp(k):                       # G^+_k Om
            Xk, _, off = X(k)
            return Xk[:, off:off + s]

        def done(k):
            uses[k] -= 1
            if not uses[k]:
                del Xs[k], uses[k]

        fold = cyc and N == 2
        times, A, Minv, C = [], [], [], []
        for i in range(o, N, 2):
            kL, kR = (i - 1) % N, (i + 1) % N
            has_left = cyc or i > 0

            # -- M_i = I - G^+_i G^-_{i+1} - G^-_i G^+_{i-1}: products only --
            t0 = self._clock()
            YM, ZM = Om.clone(), Psi.clone()
            ap(SiP[i], Xm(kR), out=YM, alpha=-1.0, beta=1.0)
            tp = ap(SiP[i], Psi, T=True)                 # (G^+_i)^T Psi
            ap(SiM[kR], tp, out=ZM, alpha=-1.0, beta=1.0, T=True)
            if has_left:
                ap(SiM[i], Xp(kL), out=YM, alpha=-1.0, beta=1.0)
                tm = ap(SiM[i], Psi, T=True)             # (G^-_i)^T Psi
                ap(SiP[kL], tm, out=ZM, alpha=-1.0, beta=1.0, T=True)
            if fold:                                     # self-coupling
                ap(SiM[i], Xm(kL), out=YM, alpha=-1.0, beta=1.0)
                ap(SiM[kL], tm, out=ZM, alpha=-1.0, beta=1.0, T=True)
                ap(SiP[i], Xp(kR), out=YM, alpha=-1.0, beta=1.0)
                ap(SiP[kR], tp, out=ZM, alpha=-1.0, beta=1.0, T=True)
            del tp
            if has_left:
                del tm
            t1 = self._clock()
            M = self._compress(rk, Om, Psi, YM, ZM, True)
            del YM, ZM
            times.append(('M', i, t1 - t0, self._clock() - t1))

            # -- one M^{-1} solve on [(Om) | G^-_i Xm(kL) | G^+_i Xp(kR)] -----
            #    (the Om block only when Minv is compressed separately)
            wantA = (has_left and not fold and SiM[i] is not None
                     and SiM[kL] is not None)
            wantC = not fold and SiP[i] is not None and SiP[kR] is not None
            inv = self.store_inverse
            t0 = self._clock()
            R = self._new(s * (int(inv) + int(wantA) + int(wantC)))
            off, offA, offC = 0, None, None
            if inv:
                R[:, :s].copy_(Om)
                off = s
            if wantA:
                offA, off = off, off + s
                ap(SiM[i], Xm(kL), out=R[:, offA:offA + s])
            if wantC:
                offC = off
                ap(SiP[i], Xp(kR), out=R[:, offC:offC + s])
            if R.shape[1]:
                sv(M, R)
            w = (sv(M, Psi.clone(), 'T')                 # M^{-T} Psi, shared
                 if (inv or wantA or wantC) else None)
            t1 = self._clock()

            if inv:
                # -- Minv_i compressed: Y = M^{-1} Om, Z = M^{-T} Psi ----------
                Minv.append(self._compress(rk, Om, Psi,
                                           R[:, :s].contiguous(), w, False))
                del M                                    # no longer needed
                times.append(('Minv', i, t1 - t0, self._clock() - t1))
            else:
                # -- keep M itself; its ULV is the inverse used in the solve --
                Minv.append(M)
                del M
                times.append(('M^-1 samples', i, t1 - t0, 0.0))

            # -- G'^-_i = -M^{-1} G^-_i G^-_{i-1} --------------------------------
            if wantA:
                t0 = self._clock()
                Y = torch.neg(R[:, offA:offA + s])
                Z = ap(SiM[kL], ap(SiM[i], w, T=True), alpha=-1.0, T=True)
                t1 = self._clock()
                A.append(self._compress(rk, Om, Psi, Y, Z, False))
                del Y, Z
                times.append(('A', i, t1 - t0, self._clock() - t1))
            else:
                A.append(None)

            # -- G'^+_i = -M^{-1} G^+_i G^+_{i+1} --------------------------------
            if wantC:
                t0 = self._clock()
                Y = torch.neg(R[:, offC:offC + s])
                Z = ap(SiP[kR], ap(SiP[i], w, T=True), alpha=-1.0, T=True)
                t1 = self._clock()
                C.append(self._compress(rk, Om, Psi, Y, Z, False))
                del Y, Z
                times.append(('C', i, t1 - t0, self._clock() - t1))
            else:
                C.append(None)
            del R, w

            done(kR)
            if has_left:
                done(kL)
        self.blockTimes.append(times)
        return A, Minv, C

    def _report_level(self, j):
        N, dt, rk = self.levelTimes[j]
        ts = sum(t[2] for t in self.blockTimes[j])
        tc = sum(t[3] for t in self.blockTimes[j])
        print(f" level {j}: nSlabs = {N:5d}  rk = {rk:4d}  {dt:8.2f} s  "
              f"(sample {ts:.2f} s, compress {tc:.2f} s)")
        for kind, i, a, b in self.blockTimes[j]:
            print(f"   {kind}[{i:4d}]  sample {a:8.3f} s  compress {b:8.3f} s")

    # -- solve -------------------------------------------------------------------

    def solve(self, rhs):
        """u for the stacked right-hand side f (nSlabs*m rows, slab by slab,
        original order within a slab), 1-D or 2-D, numpy or tensor.  Returns
        the same kind (a tensor on rhs's device)."""
        if self.RB is None:
            raise RuntimeError("solve() before factorize()")
        m, RB = self.m, self.RB
        numpy_input = isinstance(rhs, np.ndarray)
        F = torch.as_tensor(rhs)
        was_vector = F.ndim == 1
        F = F.reshape(-1, m, 1 if was_vector else F.shape[-1])
        p = self.perm.to(self.work_device)
        v = [F.to(device=self.work_device, dtype=self.dtype)[:, p, :]]   # tree order
        ap, sv = self._ap, self._sv
        o = self._o                                          # retained parity
        nrm = self.normalized       # identity diagonal at every level

        # ---- forward reduction: v'_j = v_i - S^-_i T^{-1} v_{i-1} - S^+_i T^{-1} v_{i+1}
        #      normalized:       v'_j = Minv_j (v_i - G^-_i v_{i-1} - G^+_i v_{i+1})
        for l in range(len(RB) - 1):
            SiM, T, SiP = RB[l]
            vl, N = v[-1], len(SiM)
            if nrm:
                Tinv = {k: vl[k] for k in range(1 - o, N, 2)}
            else:
                Tinv = {k: sv(T[k], vl[k].clone()) if T[k] is not None
                        else vl[k] for k in range(1 - o, N, 2)}  # eliminated
            vn = vl[o::2].clone()
            for j, i in enumerate(range(o, N, 2)):
                if SiM[i] is not None:
                    ap(SiM[i], Tinv[(i - 1) % N], out=vn[j], alpha=-1.0, beta=1.0)
                if SiP[i] is not None:
                    ap(SiP[i], Tinv[(i + 1) % N], out=vn[j], alpha=-1.0, beta=1.0)
                if nrm:                   # g'_j = M_j^{-1} (...)
                    if self.store_inverse:
                        ap(RB[l + 1][1][j], vn[j].clone(), out=vn[j])
                    else:
                        sv(RB[l + 1][1][j], vn[j])
            v.append(vn)

        # ---- coarsest: one node (normalized: already done by its Minv) --------
        if not nrm:
            sv(RB[-1][1][0], v[-1][0])

        # ---- back substitution: u_i = T_i^{-1}(v_i - S^-_i u_{i-1} - S^+_i u_{i+1})
        for l in range(len(RB) - 2, -1, -1):
            SiM, T, SiP = RB[l]
            vl, u, N = v[l], v[l + 1], len(SiM)
            vl[o::2] = u
            for i in range(1 - o, N, 2):          # neighbours are retained: in u
                # retained node r sits at u[(r - o) // 2]
                if SiM[i] is not None:
                    ap(SiM[i], u[((i - 1) % N - o) // 2], out=vl[i],
                       alpha=-1.0, beta=1.0)
                if SiP[i] is not None:
                    ap(SiP[i], u[((i + 1) % N - o) // 2], out=vl[i],
                       alpha=-1.0, beta=1.0)
                if not nrm:
                    sv(T[i], vl[i])

        x = torch.empty_like(v[0])
        x[:, p, :] = v[0]                                   # original order
        x = x.reshape(-1, x.shape[-1])
        if was_vector:
            x = x[:, 0]
        if numpy_input:
            return x.cpu().numpy()
        return x.to(F.device)
