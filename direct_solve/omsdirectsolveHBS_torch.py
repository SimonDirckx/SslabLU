import numpy as np
from scipy.sparse.linalg   import LinearOperator
import matAssembly.HBS.HBStorch as HBSnew
from abc import ABC, abstractmethod
from direct_solve.omsdirectsolve import DirectSolver
import torch
import time
import gc
import math
import warnings
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
                 compress_diag=True, fused=True, device='cpu', fast=False, identity_diag=None, skip_unused_ulv=True,compute_device=None,strict_residency=False,
                 oversample=None,
                 debug_blocks=0, debug_inverse=True, debug_true_inverse=False,
                 debug_seed=1234):
        super().__init__(m, cyclic)
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
        self.strict_residency = strict_residency
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
                "RedBlackSolverHBS: debug_true_inverse keeps up to six "
                "neighbour blocks resident through each B compression "
                "(~11 GB at nc=65536, rk=400). Expect a much higher peak, and "
                "drop to a smaller problem if it does not fit.", UserWarning)
        self.blockErrors   = []
        self._dbg_stage    = 0
        self._pgen = torch.Generator(device=self.compute_device)
        self._pgen.manual_seed(int(debug_seed))
        self._blocks = []
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
            h.strict = self.strict_residency
            h._built_by_rb = True
            self._blocks.append(h)
            return h
    def _adopt(self, op):
        """Take ownership of an operator.
        """
        if hasattr(op, 'evict'):
            op.compute_device = self.compute_device
            op.strict         = self.strict_residency
            op.warn_on_demote = True
            self._blocks.append(op)
            op.evict()
        return op
    def _finish(self, h, ulv, label, spill=True):
        self.nConstruct += 1
        if ulv:
            self.nULV += 1
        else:
            self.nULVSkipped += 1
            h = self._guard_no_ulv(h, label or "an HBS block")
        self._register(h)
        if spill:
            h.evict()
        return h

    def _hbs(self, linop, rk=None, device=None, compute_ULV=True, label=None,spill=True):
        """Compress a LinearOperator into an HBS matrix.

        compute_ULV=False compresses for applies only and skips the
        factorization; the result is fitted with a raising `solve`.
        """
        rkloc = self.rk if rk is None else rk
        dev   = self.compute_device if device is None else torch.device(device)
        ulv   = self._want_ulv(compute_ULV)

        h = HBSnew.HBSMAT(linop, device=dev, tree=self.tree, quad=self.quad)
        h.construct(rkloc, compute_ULV=ulv, fast=self.fast)

        return self._finish(h, ulv, label, spill=spill)

    def _hbs_from_samples(self, rk, Om, Psi, Y, Z, compute_ULV=True, label=None,spill=True):
        """Compress from externally supplied samples Y = M Om, Z = M^T Psi.

        Om/Psi/Y/Z must be numpy: constructHBS calls torch.from_numpy on all
        four.  compute_ULV=False skips the factorization; see `_hbs`.
        """
        ulv = self._want_ulv(compute_ULV)
        dev = self.compute_device

        def _prep(X):
            if torch.is_tensor(X):
                return X.to(device=dev, dtype=torch.float64,
                            non_blocking=True).contiguous()
            return np.ascontiguousarray(X)

        h = HBSnew.HBSMAT(device=dev, tree=self.tree, quad=self.quad)
        h.construct(rk,
                    Om=_prep(Om), Psi=_prep(Psi),
                    Y=_prep(Y),   Z=_prep(Z),
                    compute_ULV=ulv, fast=self.fast)

        return self._finish(h, ulv, label, spill=spill)
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

        h must still be resident (build with spill=False).  W/Q are probe
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
        return op.matmat(X)

    def _apT(self, op, X):
        self.nApply += 1
        return op.rmatmat(X)

    def _sv(self, op, X, mode='N'):
        self.nSolve += 1
        return op.solve(X, mode=mode)

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
        HBSnew.uv_timers_reset()
        HBSnew._UV_SYNC[0] = (self.compute_device.type == 'cuda')
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
        for h in self._blocks:
            if getattr(h, '_built_by_rb', False) and hasattr(h, 'evict'):
                h.evict()
        torch.cuda.empty_cache()

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
    def _retire(self,idx,SiM,SiP,T,T_hbs,Xm,Xp,nSlabs):
        if idx < 0 or idx>= nSlabs:
            return
        for lst in (SiM,SiP,T,T_hbs):
            op = lst[idx]
            if hasattr(op,'evict'):
                op.evict()
        Xm.pop(idx,None)
        Xp.pop(idx,None)
    def _release(self, *ops):
        """Spill operators back to host once their last consumer in this
        sweep has run.  The solve touches every block in the tree, so without
        this the working set is the whole factorization."""
        for op in ops:
            if op is not None and hasattr(op, 'evict'):
                op.evict()

    def _build_level_fused(self, m, nSlabs, RB_level, rk):
        SiM   = RB_level[0]
        T     = RB_level[1]
        T_hbs = RB_level[2]
        SiP   = RB_level[3]

        cyclic = self.cyclic
        dtype  = self._dtype

        s   = self._nsamples(rk)
        Om  = torch.randn(m,s,generator=self._tgen,device=self.compute_device,dtype=self._tdtype)
        Psi = torch.randn(m,s,generator=self._tgen,device=self.compute_device,dtype=self._tdtype)

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
        # Filled LAZILY, on first use from the retained-node loop below.
        # Running this eagerly as a separate pass put nSlabs/2 blocks of
        # (m, 2s) on the device -- 6.2 GB at Ntot=2^20, s=778 -- and, worse,
        # forced every odd-index SiM/SiP/T_hbs resident at once, since _ap/_sv
        # prefetch and nothing released them until _retire ran.  That is
        # 21.5 GB at nSlabs=8, where the inputs are the previous level's
        # compressed HBS blocks (~1.8 GB each).  Each odd k is consumed by
        # exactly two retained nodes, i = k-1 and i = k+1, and _retire(i-1)
        # pops it after the second, so the lazy version holds two X blocks.
        # ---------------------------------------------------------------
        Xm, Xp = {}, {}

        def _sv_overwrite(op, X, mode='N'):
            """Solve, letting an HBSMAT reuse X's storage for the result.

            X must not be read afterwards; the return value may alias it.
            Needs HBSMAT.solve(..., overwrite_b=True).  Other operator types
            (id_op, dead_op, RB_linop) go through the normal _sv."""
            if isinstance(op, HBSnew.HBSMAT):
                self.nSolve += 1
                return op.solve(X, mode=mode, overwrite_b=True)
            return self._sv(op, X, mode=mode)

        def _sub(Y, D):
            """Y - D, in place unless Y still aliases a shared test matrix.

            With T_i = I, Y_B and Z_B start out as Om and Psi themselves, and
            those are read by every compression at this level.  The first
            update on such a node is therefore out of place; its result is a
            fresh tensor, so every later update on that node runs in place."""
            if Y is Om or Y is Psi:
                return Y - D
            return Y.sub_(D)

        def _ensure_X(k):
            """Fill Xm[k] (and Xp[k] where needed) if not already present.
            Idempotent: node k-1 builds it as its kR, node k+1 reuses it as
            its kL, so each odd k costs one prefetch and one solve."""
            if k in Xm:
                return
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
                return

            # Fill one (m, 2s) buffer directly, so at most one s-column apply
            # output lives next to it (previously: both halves plus their
            # concatenation).  The chunked solve then writes X into the same
            # storage, so no separate output block is allocated either.
            RHS = torch.empty(m, 2 * s if need_p else s,
                              dtype=self._tdtype, device=self.compute_device)
            RHS[:, :s] = self._ap(SiM[k], Om)
            if need_p:
                RHS[:, s:] = self._ap(SiP[k], Om)
            X = _sv_overwrite(T_hbs[k], RHS)     # one solve, up to 2s columns
            del RHS                              # X aliases it when in place
            Xm[k] = X[:, :s]
            if need_p:
                Xp[k] = X[:, s:]

        # ---------------------------------------------------------------
        # debug probes.  W/Q are independent of Om/Psi, so the checks below
        # are out of sample; Pm/Pp mirror Xm/Xp for the probe columns and are
        # shared between the two nodes that consume each odd k, exactly as
        # Xm/Xp are.  All of it is skipped entirely when debug_blocks == 0.
        # ---------------------------------------------------------------
        dbg = self.debug_blocks > 0
        W = self._dbg_probe(self.debug_blocks) if dbg else None
        Q = self._dbg_probe(self.debug_blocks) if dbg else None
        Pm, Pp = {}, {}

        def _ensure_P(k):
            # Unlike _ensure_X this builds Pp for EVERY k, including the last
            # odd node where Xp has no consumer.  That extra probe solve (t
            # columns, not s) is what makes the C_is_zero slot checkable: its
            # reference is -S^+_i T_k^-1 S^+_k W with S^+_k the zero operator,
            # so a nonzero result means the zero-slot analysis is wrong.
            if k in Pm:
                return
            if _is_id(T_hbs[k]):
                Pm[k] = self._ap(SiM[k], W)
                Pp[k] = self._ap(SiP[k], W)
            else:
                Pm[k] = self._sv(T_hbs[k], self._ap(SiM[k], W))
                Pp[k] = self._sv(T_hbs[k], self._ap(SiP[k], W))

        def _refs(i, has_left, has_right, kL, kR, A_is_zero, C_is_zero):
            """Reference actions of B_i, A_i, C_i on the probes.

            Mirrors the Y_*/Z_* expressions above term for term, so the check
            measures compression error and not a re-derivation of the algebra.
            """
            fB = W.clone() if _is_id(T[i]) else self._ap(T[i], W)
            aB = Q.clone() if _is_id(T[i]) else self._apT(T[i], Q)
            fA = fC = aA = aC = None
            if has_right:
                _ensure_P(kR)
                fB = fB - self._ap(SiP[i], Pm[kR])
                # computed even when C_is_zero: that is the check
                fC = -self._ap(SiP[i], Pp[kR])
                rp = self._apT(SiP[i], Q)
                tp = rp if _is_id(T_hbs[kR]) else self._sv(T_hbs[kR], rp, mode='T')
                aB = aB - self._apT(SiM[kR], tp)
                aC = -self._apT(SiP[kR], tp)   # also computed when C_is_zero
            if has_left:
                _ensure_P(kL)
                fB = fB - self._ap(SiM[i], Pp[kL])
                fA = -self._ap(SiM[i], Pm[kL])
                rm = self._apT(SiM[i], Q)
                tm = rm if _is_id(T_hbs[kL]) else self._sv(T_hbs[kL], rm, mode='T')
                aB = aB - self._apT(SiP[kL], tm)
                aA = -self._apT(SiM[kL], tm)
            return {'B': (fB, aB), 'A': (fA, aA), 'C': (fC, aC)}

        dbg_true = self.debug_true_inverse and dbg

        def _apply_B(i, X, has_left, has_right, kL, kR):
            """B_i X with the UNCOMPRESSED chain:

                B_i = T_i - S^+_i T_{i+1}^-1 S^-_{i+1}
                          - S^-_i T_{i-1}^-1 S^+_{i-1}

            Same terms as the Y_B expression, but applied to an arbitrary X
            rather than to Omega, so the Xm/Xp and Pm/Pp caches do not help:
            every call costs 2 applies and 1 solve per neighbour, on X's
            column count.  Only used for inv_ref, and only on t columns.
            """
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

        def _spill(t):
            return None if t is None else t.to('cpu')

        def _unspill(t):
            return None if t is None else t.to(self.compute_device)
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
            # A_0 is zero exactly when there is no left neighbour.
            # C_{nSlabs-2} is zero because S^+_{nSlabs-1} = 0, even though the
            # right neighbour exists -- the asymmetry is because the zeroed
            # SiM sits at an even index and the zeroed SiP at an odd one.
            A_is_zero = (not cyclic) and i == 0
            C_is_zero = (not cyclic) and i == nSlabs - 2

            # T_i Om and T_i^T Psi are Om and Psi themselves when T_i = I.
            # _sub below keeps the first update out of place in that case.
            if _is_id(T[i]):
                Y_B, Z_B = Om, Psi
                self.nIdSkipped += 1
            else:
                Y_B = self._ap(T[i], Om)
                Z_B = self._apT(T[i], Psi)
            Y_A = Y_C = Z_A = Z_C = None

            # Each former fused apply on cat([X1, X2]) is now two plain
            # applies.  matmat/rmatmat chunk every call at _MATMAT_CHUNK
            # columns anyway, so fusing bought no GPU work; it only cost the
            # (m, 2s) concatenation plus an (m, 2s) output held at once.

            if has_right:
                for op in (SiM[kR], SiP[kR]):
                    if hasattr(op, 'release_ulv'):
                        op.release_ulv()
                _ensure_X(kR)

                # forward: S^+_i Xm (B term) and S^+_i Xp (C term)
                Y_B = _sub(Y_B, self._ap(SiP[i], Xm[kR]))
                if not C_is_zero:
                    Y_C = _spill(self._ap(SiP[i], Xp[kR]).neg_())

                # adjoint: t^+ = T_{i+1}^{-T} (S^+_i)^T Psi serves B and C
                rhs_p = self._apT(SiP[i], Psi)
                if _is_id(T_hbs[kR]):
                    tp = rhs_p
                    self.nIdSkipped += 1
                else:
                    tp = _sv_overwrite(T_hbs[kR], rhs_p, mode='T')
                del rhs_p
                Z_B = _sub(Z_B, self._apT(SiM[kR], tp))
                if not C_is_zero:
                    Z_C = _spill(self._apT(SiP[kR], tp).neg_())
                del tp

            if has_left:
                for op in (SiP[kL], SiM[kL]):
                    if hasattr(op, 'release_ulv'):
                        op.release_ulv()
                _ensure_X(kL)

                # forward: S^-_i Xp (B term) and S^-_i Xm (A term)
                Y_B = _sub(Y_B, self._ap(SiM[i], Xp[kL]))
                Y_A = _spill(self._ap(SiM[i], Xm[kL]).neg_())

                # adjoint: t^- = T_{i-1}^{-T} (S^-_i)^T Psi serves B and A
                rhs_m = self._apT(SiM[i], Psi)
                if _is_id(T_hbs[kL]):
                    tm = rhs_m
                    self.nIdSkipped += 1
                else:
                    tm = _sv_overwrite(T_hbs[kL], rhs_m, mode='T')
                del rhs_m
                Z_B = _sub(Z_B, self._apT(SiP[kL], tm))
                Z_A = _spill(self._apT(SiM[kL], tm).neg_())
                del tm

            # Guard the degenerate case where neither branch ran: Y_B/Z_B
            # would still alias Om/Psi, which construct would then receive as
            # both the test matrix and its own samples.
            if Y_B is Om:
                Y_B = Om.clone()
            if Z_B is Psi:
                Z_B = Psi.clone()

            # Probe references must be built while the neighbour operators
            # are still staged, i.e. before the retire below.  The comparison
            # happens after each block is constructed.
            refs = _refs(i, has_left, has_right, kL, kR,
                         A_is_zero, C_is_zero) if dbg else None

            # Retire BEFORE compressing, not after.  Everything nodes i-1 and
            # i contribute has now been sampled into Y_*/Z_*; the three
            # constructions below read only those and Om/Psi.  Retiring here
            # keeps six operator blocks (~1.8 GB each) and X[i-1] off the
            # device through all three compressions, which is where the peak
            # sits.  _retire only evicts and pops X -- it does not clear the
            # list slots -- so the RB_linop built in the compress_diag=False
            # branch below still holds valid references.
            if not cyclic and not dbg_true:
                self._retire(i-1,SiM,SiP,T,T_hbs,Xm,Xp,nSlabs)
                self._retire(i  ,SiM,SiP,T,T_hbs,Xm,Xp,nSlabs)

            self._sync(); t1=time.time()
            # ---- compress from the shared samples ----------------------
            need_ULV = self._needs_ulv(i, nSlabs)

            if need_ULV or self.compress_diag or not self.skip_unused_ulv:
                B_hbs = self._hbs_from_samples(rk, Om, Psi, Y_B, Z_B,
                                               compute_ULV=need_ULV,
                                               label=f"B[{i}] (nSlabs={nSlabs})",
                                               spill=not dbg)
            else:
                # compress_diag=False hands the uncompressed RB_linop to the
                # next level as T[i], so this slot is read by nobody: skip
                # the compression itself, not just the factorization.
                B_hbs = self._dead_diag(f"B[{i}] (nSlabs={nSlabs})")
            if dbg:
                self._check_block(
                    B_hbs, 'B', nSlabs, i, rk, W, Q, *refs['B'],
                    apply_ref=(lambda X: _apply_B(i, X, has_left, has_right,
                                                  kL, kR)) if dbg_true else None)
                if hasattr(B_hbs, 'evict'):
                    B_hbs.evict()          # _finish's spill, deferred past the probe
            # With inv_ref the neighbours had to survive the B compression;
            # release them now, before A and C, so the extra residency window
            # covers one compression instead of three.
            if not cyclic and dbg_true:
                self._retire(i-1,SiM,SiP,T,T_hbs,Xm,Xp,nSlabs)
                self._retire(i  ,SiM,SiP,T,T_hbs,Xm,Xp,nSlabs)
            T_hbs_new.append(B_hbs)

            if self.compress_diag:
                B_i.append(B_hbs)
            else:
                spm = SiP[kL] if has_left  else None
                smp = SiM[kR] if has_right else None
                tmo = T_hbs[kL] if has_left  else None
                tpo = T_hbs[kR] if has_right else None
                B_i.append(RB_linop(T[i], tmo, tpo, SiP[i], SiM[i], smp, spm))

            del Y_B, Z_B

            # A_i and C_i become SiM / SiP one level down and are only ever
            # applied, never solved with -- no ULV, unconditionally.
            A_i.append(zero_op(m, dtype) if A_is_zero
                       else self._hbs_from_samples(rk, Om, Psi, Y_A, Z_A,
                                                   compute_ULV=False,
                                                   label=f"A[{i}] (nSlabs={nSlabs})",
                                                   spill=not dbg))
            if dbg:
                self._check_block(A_i[-1], 'A', nSlabs, i, rk, W, Q, *refs['A'])
                if hasattr(A_i[-1], 'evict'):
                    A_i[-1].evict()
            del Y_A, Z_A
            C_i.append(zero_op(m, dtype) if C_is_zero
                       else self._hbs_from_samples(rk, Om, Psi, Y_C, Z_C,
                                                   compute_ULV=False,
                                                   label=f"C[{i}] (nSlabs={nSlabs})",
                                                   spill=not dbg))
            if dbg:
                self._check_block(C_i[-1], 'C', nSlabs, i, rk, W, Q, *refs['C'])
                if hasattr(C_i[-1], 'evict'):
                    C_i[-1].evict()
            del Y_C, Z_C
            if dbg:
                del refs

            self._sync();t2=time.time()
            print(f"node {i:3d}: sample {t1-t0:6.2f}s"
                  f" construct {t2-t1:6.2f}s (B+A+C)"
                  f" alloc {torch.cuda.memory_allocated()/2**30:5.2f} GB")
        if not cyclic:
            self._retire(nSlabs-1,SiM,SiP,T,T_hbs,Xm,Xp,nSlabs)
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

        for l in range(len(RB) - 1):
            SiM, _, T_hbs, SiP = RB[l]

            nSlabs   = len(SiM)
            nReduced = nSlabs // 2
            vPrev    = vPrimes[-1]
            # every block is written below, so no zero fill
            vPrime   = torch.empty(m * nReduced, nrhs,dtype=self._tdtype,device=dev)

            # T_k^{-1} vPrev_k for odd k is read by node k-1 (as `next`) and by
            # node k+1 (as `prev`) -- the same solve on the same rhs.  Solve
            # once, release T_hbs[k] immediately (the result is all anyone
            # needs), and drop the result after its last reader.  In the
            # non-cyclic case the last odd node has a single reader.
            Tinv, uses = {}, {}

            def _Tinv(k):
                if k not in Tinv:
                    blk = vPrev[k*m:(k+1)*m, :]
                    if _is_id(T_hbs[k]):
                        Tinv[k] = blk           # read-only use: no clone needed
                        self.nIdSkipped += 1
                    else:
                        Tinv[k] = self._sv(T_hbs[k], blk)
                        self._release(T_hbs[k])
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
                    contrib = contrib - self._ap(SiM[i], _Tinv(prev))
                if next is not None:
                    contrib = contrib - self._ap(SiP[i], _Tinv(next))

                vPrime[j*m:(j+1)*m,:] = contrib

                # SiM[i]/SiP[i] have no further consumer at this level.
                self._release(SiM[i], SiP[i])

            assert not Tinv, "forward reduction left cached solves unconsumed"
            self._release(*T_hbs)       # no-op for already-released blocks
            vPrimes.append(vPrime)

        # ---- coarsest solve -------------------------------------------
        vPrimes[-1] = self._sv(RB[-1][2][0],vPrimes[-1])

        # ---- back substitution ----------------------------------------
        for l in range(len(RB) - 1, 0, -1):
            SiM, _, T_hbs, SiP = RB[l - 1]

            nSlabs   = len(SiM)
            nReduced = nSlabs // 2

            for j in range(nReduced):
                i = 2 * j

                vPrimes[l-1][i*m:(i+1)*m] = vPrimes[l][j*m:(j+1)*m]

                next_j = (j + 1) % nReduced
                contrib = self._ap(SiM[i+1],vPrimes[l][j*m:(j+1)*m,:])
                if self.cyclic or j + 1 < nReduced:
                    contrib = contrib + self._ap(SiP[i+1],vPrimes[l][next_j*m:(next_j+1)*m, :])

                blk = vPrimes[l-1][(i+1)*m:(i+2)*m] - contrib
                vPrimes[l-1][(i+1)*m:(i+2)*m,:] = self._sv(T_hbs[i+1],blk)
                self._release(SiM[i+1], SiP[i+1], T_hbs[i+1])
        out = vPrimes[0]
        if was_vector:
            out = out[:,0]
        if input_is_numpy:
            return out.detach().cpu().numpy()
        return out
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
            free, cap = torch.cuda.mem_get_info()
            print(f"    device capacity {cap/2**30:6.2f} GB, "
                  f"free {free/2**30:6.2f} GB")

        return {'total': tot, 'resident': res, 'by_list': sub}
