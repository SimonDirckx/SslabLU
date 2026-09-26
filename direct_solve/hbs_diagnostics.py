"""
Per-stage error diagnostics for ThomasSolverHBS and RedBlackSolverHBS.

Notation
--------
A       the original (exact) operator
E_k     stage k of the exact reduction (dense blocks, from the exact inputs)
H_k     what the solver stored at stage k
f(.)    one exact reduction step (the next stage's Schur complements)
C(.)    HBS compression with the solver's own settings
Ã_k     a partial factorization through stage k: stages 0..k factored, the
        remaining reduced system left unfactored (solved exactly)

Two paths
---------
COMB    teeth off the exact path -- isolates a single stage.
        Stage 0 compares the exact inputs with the compressed ones the solver
        was given.  Stage k >= 1 compresses each exact block of E_k and
        compares C(E_k) with E_k.  Its partial factorization Ã_k^comb is
        exact everywhere except stage k.

LADDER  the solver's own compressed path vs the exact path -- propagation.
        Blockwise, stage k compares H_k with E_k.  Its partial factorization
        Ã_k is the solver's own, through stage k.

Blockwise metrics (block records; each optional)
------------------------------------------------
apply       ||(H - E) X||_F / ||E X||_F on fresh Gaussian probes X
power_*     sigma_max(H - E) by power iteration on (H-E)^*(H-E), absolute
            and relative to sigma_max(E)
dense_*     ||dense(H) - dense(E)||_2 by SVD of the m x m blocks, absolute
            and relative

Stage-level metrics (records with block=None; enabled by 'partial')
------------------------------------------------------------------
Norms here are power-iteration estimates only ('dense' does not apply at
stage level): *_power is sigma_max by power iteration, *_apply the probe
ratio.  Power iteration gives a lower bound on sigma_max.

bwd_*       backward error of the partial factorization, ||Ã_k - A|| / ||A||
            (factorization quality)
lvl_*       ladder only: the error stage k itself injects, as a share of
            ||A||.  Ã_k - A is exactly the sum of these terms over stages
            0..k, each placed at the rows and columns its level acts on:
                level 0:   the input error (compressed vs exact inputs)
                level l:   H_l - f(H_{l-1}), with f applied to the solver's
                           own compressed stage l-1
            With compress_diag=False the odd diagonals of level l are also
            compressed separately for solves; that part of level l's term
            enters Ã_k only from k = l+1 on, once those nodes are eliminated.
slv_*       solver error ||I - Ã_k^{-1} A||: bounds the relative solution
            error for every right-hand side, and is the convergence factor if
            Ã_k is used for iterative refinement or as a preconditioner.
            Absolute (it is already dimensionless).
sol_err     for one right-hand side b: ||x_k - x_true|| / ||x_true||
sol_resid   ||A x_k - b|| / ||b||
consistency ||(A + E_k) x_k - b|| / ||b||: x_k from the solver's own truncated
            solve (ladder) or the factorization-structured solve (comb)
            checked against the partial factorization Ã_k = A + E_k that the
            other stage metrics describe.  Should sit at rounding level.

RedBlackSolverHBS's ladder also gets a final 'solve' row for solver.solve()
itself, whose coarsest solve uses the compressed diagonal's ULV.

Cost
----
Nothing of size (N m) x (N m) is ever formed.  A and each Ã_k - A are
block-sparse operators on cached m x m blocks; x_true and every Ã_k^{-1}
(and its adjoint, for power iteration) go through the block structure of the
factorization, with each dense diagonal block LU-factored once.  One-time
work is O(N m^3) per stage; each operator apply is O(N m^2) plus, on the
ladder, the solver's own HBS solves.  Memory is O(N m^2).

Assumptions: default contiguous DOF ordering (glob_target_dofs=None) for
Thomas, non-cyclic Thomas (its SMW correction is not reproduced here), and
1-D right-hand sides.  Adjoints follow the solver's own convention
(rmatmat and solve(mode='T')).
"""

import re
import numpy as np
from scipy.linalg import lu_factor, lu_solve

try:
    from .omsdirectsolveHBS import (HBSnew, torch, zero_op, dead_op, id_op,
                                    _linop_from_mat, STS_linop, RB_linop,
                                    Sprime_Linop, Dprime_Linop, _sum_linop)
except ImportError:
    from omsdirectsolveHBS import (HBSnew, torch, zero_op, dead_op, id_op,
                                   _linop_from_mat, STS_linop, RB_linop,
                                   Sprime_Linop, Dprime_Linop, _sum_linop)

ALL_METRICS = ("apply", "power", "dense", "partial")


# ---------------------------------------------------------------------------
# Small operator utilities.  Anything here may be an ndarray, a scipy
# LinearOperator, an HBSMAT (whose outputs may be torch tensors), or one of
# the helper operators below.
# ---------------------------------------------------------------------------

def _mm(op, X):
    if isinstance(op, np.ndarray):
        return op @ X
    return np.asarray(op.matmat(X))


def _rmm(op, X):
    if isinstance(op, np.ndarray):
        return op.conj().T @ X
    return np.asarray(op.rmatmat(X))


def _dense(op, n):
    """Dense n x n form of an operator: op @ I."""
    if isinstance(op, np.ndarray):
        return op
    return np.asarray(op @ np.identity(n))


def _densifier(m):
    """_dense for m x m blocks, cached per operator for one diagnose run."""
    cache = {}

    def dens(op):
        if isinstance(op, np.ndarray):
            return op
        key = id(op)
        if key not in cache:
            cache[key] = (op, _dense(op, m))     # keep op alive: id stays valid
        return cache[key][1]
    return dens


def _structural_skip(op):
    """Blocks that carry no numerical content to check."""
    return isinstance(op, (zero_op, dead_op))


def _rel(num, den):
    return float(num / den) if den > 0 else float("nan")


def _add_into(Y, idx, contrib):
    """Y[idx] += contrib, upcasting Y if contrib is complex."""
    if np.iscomplexobj(contrib) and not np.iscomplexobj(Y):
        Y = Y.astype(np.result_type(Y.dtype, contrib.dtype))
    Y[idx] += contrib
    return Y


def _as2d(X):
    X = np.asarray(X)
    return (X[:, None], True) if X.ndim == 1 else (X, False)


class _OpBase:
    """Helper operators below: `@` is matmat, and dense(n) is self @ I."""
    def __matmul__(self, X):
        return self.matmat(X)

    def dense(self, n):
        return np.asarray(self @ np.identity(n))


class _Diff(_OpBase):
    """H - M as an operator."""
    def __init__(self, H, M):
        self.H, self.M = H, M
    def matmat(self, X):  return _mm(self.H, X) - _mm(self.M, X)
    def rmatmat(self, X): return _rmm(self.H, X) - _rmm(self.M, X)


class _BlockOp(_OpBase):
    """Sparse block operator on N blocks of size m: a sum of entries, each an
    m x m block (dense array or operator) at block position (r, c).  Entries
    at the same position add."""

    def __init__(self, N, m, entries=()):
        self.N, self.m = N, m
        self.entries = list(entries)

    def add(self, r, c, op):
        self.entries.append((r, c, op))

    def __add__(self, other):
        return _BlockOp(self.N, self.m, self.entries + other.entries)

    def _sl(self, k):
        return slice(k * self.m, (k + 1) * self.m)

    def _apply(self, X, adjoint):
        X2, one = _as2d(X)
        Y = np.zeros((self.N * self.m, X2.shape[1]), dtype=np.result_type(X2.dtype, np.float64))
        for r, c, op in self.entries:
            if adjoint:
                Y = _add_into(Y, self._sl(c), _rmm(op, X2[self._sl(r)]))
            else:
                Y = _add_into(Y, self._sl(r), _mm(op, X2[self._sl(c)]))
        return Y[:, 0] if one else Y

    def matmat(self, X):  return self._apply(X, False)
    def rmatmat(self, X): return self._apply(X, True)

    def dense(self, n=None):
        """Block-local: each entry is densified on its own m columns
        (op @ I_m) and placed at its (r, c) position."""
        D = np.zeros((self.N * self.m,) * 2)
        for r, c, op in self.entries:
            D = _add_into(D, (self._sl(r), self._sl(c)), _dense(op, self.m))
        return D


class _DenseLU(_OpBase):
    """A dense block with its LU computed once: matmat / rmatmat / solve,
    solve(mode='T') being the adjoint solve (the solver's convention)."""
    def __init__(self, A):
        self.A = A
        self.shape, self.dtype = A.shape, A.dtype
        self.lu = lu_factor(A)
    def matmat(self, X):  return self.A @ X
    def rmatmat(self, X): return self.A.conj().T @ X
    def solve(self, X, mode='N'):
        return lu_solve(self.lu, X, trans=0 if mode == 'N' else 2)


def _lu_cache():
    """_DenseLU factory, one factorization per distinct array per run."""
    cache = {}

    def get(A):
        key = id(A)
        if key not in cache:
            cache[key] = (A, _DenseLU(A))
        return cache[key][1]
    return get


class _LUList:
    """Lazily LU-factored view of a list of dense diagonal blocks."""
    def __init__(self, blocks, lu):
        self.blocks, self.lu = blocks, lu
    def __getitem__(self, i):
        return self.lu(self.blocks[i])
    def __len__(self):
        return len(self.blocks)


class _Inv:
    """An inverse given by forward and adjoint solves on 2-D blocks."""
    def __init__(self, fwd, adj):
        self.fwd, self.adj = fwd, adj
    def solve(self, X, mode='N'):
        X2, one = _as2d(X)
        Y = (self.fwd if mode == 'N' else self.adj)(X2)
        return Y[:, 0] if one else Y


class _SolverErr(_OpBase):
    """I - Ã^{-1} A, with A an operator and Ã^{-1} an _Inv."""
    def __init__(self, A, inv):
        self.A, self.inv = A, inv
    def matmat(self, X):
        return X - self.inv.solve(_mm(self.A, X), mode='N')
    def rmatmat(self, X):
        return X - _rmm(self.A, self.inv.solve(X, mode='T'))


def _columnwise(f):
    """Lift a 1-D solve to 2-D blocks, one column at a time."""
    def g(X):
        return np.column_stack([np.asarray(f(np.array(X[:, c]))) for c in range(X.shape[1])])
    return g


def _power_norm(mv, rmv, n, rng, iters, tol):
    """Largest singular value of an operator by power iteration on A^* A."""
    v = rng.standard_normal((n, 1))
    v /= np.linalg.norm(v)
    sigma_old = 0.0
    sigma = 0.0
    for _ in range(iters):
        w = mv(v)
        sigma = np.linalg.norm(w)
        if sigma == 0.0:
            return 0.0
        v = rmv(w)
        nv = np.linalg.norm(v)
        if nv == 0.0:
            return sigma
        v = v / nv
        if abs(sigma - sigma_old) <= tol * sigma:
            break
        sigma_old = sigma
    return float(sigma)


def _as_cpu(h):
    h2 = h.to('cpu')
    return h if h2 is None else h2


# ---------------------------------------------------------------------------
# Structured solves
# ---------------------------------------------------------------------------

def _rb_sweep(levels, coarse, cyclic, m, X, mode):
    """
    Solve with a red-black factorization given level by level, or with its
    adjoint (mode='T').

    levels : per eliminated level, (M, Ts, P): the level's left / right
             coupling blocks and its diagonal solvers (only odd nodes are
             solved; each has .solve(X, mode)).
    coarse : solver (.solve(X, mode)) for the system left after the levels.

    Forward is the same sweep as RedBlackSolverHBS.solve.  The adjoint is the
    same sweep on the adjoint system: per level, with odd nodes o and even e,
      forward   v'_e = v_e - S_oe^* T_o^{-*} v_o
      back      x_o  = T_o^{-*} (v_o - S_eo^* x_e)
    and the coarse system solved with its adjoint.
    """
    adjoint = mode != 'N'
    sl = lambda k: slice(k * m, (k + 1) * m)
    vs = [X]
    for M, Ts, P in levels:
        v, N = vs[-1], len(Ts)
        out = []
        for j in range(N // 2):
            i = 2 * j
            c = v[sl(i)]
            if not adjoint:
                if cyclic or i > 0:
                    k = (i - 1) % N
                    c = c - _mm(M[i], Ts[k].solve(v[sl(k)], mode='N'))
                if cyclic or i < N - 1:
                    k = (i + 1) % N
                    c = c - _mm(P[i], Ts[k].solve(v[sl(k)], mode='N'))
            else:
                k = i + 1
                c = c - _rmm(M[k], Ts[k].solve(v[sl(k)], mode='T'))
                if cyclic or i > 0:
                    k = (i - 1) % N
                    c = c - _rmm(P[k], Ts[k].solve(v[sl(k)], mode='T'))
            out.append(c)
        vs.append(np.vstack(out))

    x = np.asarray(coarse.solve(vs[-1], mode=mode))
    for l in range(len(levels) - 1, -1, -1):
        M, Ts, P = levels[l]
        v, N = vs[l], len(Ts)
        nR = N // 2
        full = [None] * N
        for j in range(nR):
            full[2 * j] = x[sl(j)]
        for j in range(nR):
            k = 2 * j + 1
            if not adjoint:
                c = v[sl(k)] - _mm(M[k], x[sl(j)])
                if cyclic or j + 1 < nR:
                    c = c - _mm(P[k], x[sl((j + 1) % nR)])
            else:
                c = v[sl(k)] - _rmm(P[k - 1], x[sl(j)])
                if cyclic or k + 1 < N:
                    c = c - _rmm(M[(k + 1) % N], x[sl((j + 1) % nR)])
            full[k] = np.asarray(Ts[k].solve(c, mode=mode))
        x = np.vstack(full)
    return x


def _dense_rb_inv(dense_levels, cyclic, m, lu):
    """Ã^{-1} for a red-black reduction given as dense levels (as returned by
    _dense_cyclic_reduction, the last level holding a single node)."""
    elim = [(M, _LUList(T, lu), P) for M, T, P in dense_levels[:-1]]
    coarse = lu(dense_levels[-1][1][0])
    return _Inv(lambda X: _rb_sweep(elim, coarse, cyclic, m, X, 'N'),
                lambda X: _rb_sweep(elim, coarse, cyclic, m, X, 'T'))


def _thomas_sweep(L, P, R, X, m, mode):
    """
    Block Thomas solve (mode='N') or its adjoint (mode='T').  L[i] = block
    (i+1, i), R[i] = block (i, i+1), P[i] = eliminated diagonals S'_i, each
    with .solve(X, mode).  With Ã = L U (L unit lower, U with diagonal P):
      forward   d_i = v_i - L_{i-1} P_{i-1}^{-1} d_{i-1};  x_i = P_i^{-1}(d_i - R_i x_{i+1})
      adjoint   y_i = P_i^{-*}(v_i - R_{i-1}^* y_{i-1});   x_i = y_i - P_i^{-*} L_i^* x_{i+1}
    """
    N = len(P)
    sl = lambda k: slice(k * m, (k + 1) * m)
    x = [None] * N
    if mode == 'N':
        d = [X[sl(0)]]
        for i in range(1, N):
            d.append(X[sl(i)] - _mm(L[i - 1], P[i - 1].solve(d[i - 1], mode='N')))
        x[-1] = np.asarray(P[-1].solve(d[-1], mode='N'))
        for i in range(N - 2, -1, -1):
            x[i] = np.asarray(P[i].solve(d[i] - _mm(R[i], x[i + 1]), mode='N'))
    else:
        y = [np.asarray(P[0].solve(X[sl(0)], mode='T'))]
        for i in range(1, N):
            y.append(np.asarray(P[i].solve(X[sl(i)] - _rmm(R[i - 1], y[i - 1]), mode='T')))
        x[-1] = y[-1]
        for i in range(N - 2, -1, -1):
            x[i] = y[i] - np.asarray(P[i].solve(_rmm(L[i], x[i + 1]), mode='T'))
    return np.vstack(x)


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

class ErrorReport:
    """List of flat records, one per (kind, stage, block).

    Block records (block = a label) carry apply / power_* / dense_*.
    Stage records (block = None) carry bwd_* / lvl_* / slv_* / sol_* /
    consistency.  See the module docstring for definitions.
    """

    BLOCK_COLUMNS = ("apply", "power_abs", "power_rel", "dense_abs", "dense_rel")
    STAGE_COLUMNS = ("bwd_apply", "bwd_power", "lvl_apply", "lvl_power",
                     "slv_apply", "slv_power", "sol_err", "sol_resid", "consistency")
    COLUMNS = ("kind", "stage", "block") + BLOCK_COLUMNS + STAGE_COLUMNS

    def __init__(self, solver_name):
        self.solver_name = solver_name
        self.records = []

    def add(self, kind, stage, block=None, **vals):
        rec = {c: None for c in self.COLUMNS}
        rec.update(kind=kind, stage=stage, block=block)
        rec.update(vals)
        self.records.append(rec)

    def comb(self):
        return [r for r in self.records if r["kind"] == "comb"]

    def ladder(self):
        return [r for r in self.records if r["kind"] == "ladder"]

    def stages(self, kind):
        """Stage-level records of one kind, in stage order."""
        return [r for r in self.records if r["kind"] == kind and r["block"] is None]

    def stage_summary(self, kind, metric="dense_rel"):
        """Max of a block metric over each stage: {stage: value}."""
        out = {}
        for r in self.records:
            if r["kind"] == kind and r["block"] is not None and r[metric] is not None:
                out[r["stage"]] = max(out.get(r["stage"], 0.0), r[metric])
        return out

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records, columns=self.COLUMNS)

    def table(self, kind=None, rows="all"):
        """rows: 'all', 'blocks' (block records only) or 'stages'."""
        recs = [r for r in self.records
                if (kind is None or r["kind"] == kind)
                and (rows == "all"
                     or (rows == "blocks" and r["block"] is not None)
                     or (rows == "stages" and r["block"] is None))]
        cols = [c for c in self.COLUMNS
                if c in ("kind", "stage", "block") or any(r[c] is not None for r in recs)]
        if rows == "stages":
            cols.remove("block")

        def fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.2e}"
            return str(v)

        cells = [[fmt(r[c]) for c in cols] for r in recs]
        widths = [max([len(c)] + [len(row[k]) for row in cells])
                  for k, c in enumerate(cols)]
        line = lambda xs: "  ".join(x.ljust(w) for x, w in zip(xs, widths))
        out = [f"{self.solver_name}", line(cols), line(["-" * w for w in widths])]
        out += [line(row) for row in cells]
        return "\n".join(out)

    def __str__(self):
        return self.table()


# ---------------------------------------------------------------------------
# Shared measurement driver
# ---------------------------------------------------------------------------

class _Meter:
    def __init__(self, metrics, nprobe, power_iters, power_tol, seed):
        bad = set(metrics) - set(ALL_METRICS)
        if bad:
            raise ValueError(f"unknown metrics {bad}; choose from {ALL_METRICS}")
        self.metrics = set(metrics)
        self.nprobe = nprobe
        self.power_iters = power_iters
        self.power_tol = power_tol
        # Own generator: checks never perturb the solver's Om/Psi draws.
        self.rng = np.random.default_rng(seed)

    def wants_norms(self):
        return bool(self.metrics & {"apply", "power", "dense"})

    def wants_stage_norms(self):
        return bool(self.metrics & {"apply", "power"})

    def power(self, D, n):
        return _power_norm(lambda X: _mm(D, X), lambda X: _rmm(D, X),
                           n, self.rng, self.power_iters, self.power_tol)

    def block(self, H, M, m, dens=None):
        """Blockwise metrics of H against M (m x m).  With 'dense' on, both
        are densified once and every metric uses the dense blocks."""
        out = {}
        if "dense" in self.metrics:
            dens = dens or (lambda op: _dense(op, m))
            H, M = dens(H), dens(M)
        D = _Diff(H, M)
        if "apply" in self.metrics:
            X = self.rng.standard_normal((m, self.nprobe))
            out["apply"] = _rel(np.linalg.norm(_mm(D, X)), np.linalg.norm(_mm(M, X)))
        if "power" in self.metrics:
            e, s = self.power(D, m), self.power(M, m)
            out["power_abs"] = e
            out["power_rel"] = _rel(e, s)
        if "dense" in self.metrics:
            e = np.linalg.norm(H - M, 2)
            out["dense_abs"] = float(e)
            out["dense_rel"] = _rel(e, np.linalg.norm(M, 2))
        return out

    def stage(self, prefix, D, n, M=None, sigma_M=None):
        """Stage metrics of D: probe ratio against M (or the identity) and
        the power-iteration norm relative to sigma_M (or absolute)."""
        out = {}
        if "apply" in self.metrics:
            X = self.rng.standard_normal((n, self.nprobe))
            den = np.linalg.norm(_mm(M, X)) if M is not None else np.linalg.norm(X)
            out[f"{prefix}_apply"] = _rel(np.linalg.norm(_mm(D, X)), den)
        if "power" in self.metrics:
            e = self.power(D, n)
            out[f"{prefix}_power"] = _rel(e, sigma_M) if sigma_M is not None else e
        return out


def _assemble_btd(Mm, Tt, Pp, cyclic, m):
    """Dense block-tridiagonal matrix (for tests on small problems): row i
    has Mm[i] at column i-1, Tt[i] on the diagonal and Pp[i] at column i+1.
    Cyclic wrap-around couplings that land on the same block are summed."""
    N = len(Tt)
    blocks = [_dense(T, m) for T in Tt]
    dtype = np.result_type(*[b.dtype for b in blocks])
    A = np.zeros((N * m, N * m), dtype=dtype)
    sl = lambda k: slice(k * m, (k + 1) * m)
    for i in range(N):
        A = _add_into(A, (sl(i), sl(i)), blocks[i])
        if cyclic or i > 0:
            if Mm[i] is not None and not isinstance(Mm[i], zero_op):
                A = _add_into(A, (sl(i), sl((i - 1) % N)), _dense(Mm[i], m))
        if cyclic or i < N - 1:
            if Pp[i] is not None and not isinstance(Pp[i], zero_op):
                A = _add_into(A, (sl(i), sl((i + 1) % N)), _dense(Pp[i], m))
    return A


def _btd_op(Mm, Tt, Pp, cyclic, m):
    """Block-tridiagonal operator on dense blocks (same layout as above)."""
    N = len(Tt)
    op = _BlockOp(N, m)
    for i in range(N):
        op.add(i, i, Tt[i])
        if (cyclic or i > 0) and Mm[i] is not None:
            op.add(i, (i - 1) % N, Mm[i])
        if (cyclic or i < N - 1) and Pp[i] is not None:
            op.add(i, (i + 1) % N, Pp[i])
    return op


def _stage_record(rep, meter, kind, stage, A, sigA, E, inv, b, x_true, x, lvl=None):
    """All stage-level metrics for one partial factorization Ã = A + E, whose
    inverse (and adjoint) is `inv`.  x is the solution the factorization-
    structured solve produced for b."""
    n = len(b)
    vals = {}
    if meter.wants_stage_norms():
        vals.update(meter.stage("bwd", E, n, M=A, sigma_M=sigA))
        if lvl is not None:
            vals.update(meter.stage("lvl", lvl, n, M=A, sigma_M=sigA))
        vals.update(meter.stage("slv", _SolverErr(A, inv), n))
    Ax = _mm(A, x)
    nb = np.linalg.norm(b)
    vals["consistency"] = _rel(np.linalg.norm(Ax + _mm(E, x) - b), nb)
    vals["sol_err"] = _rel(np.linalg.norm(x - x_true), np.linalg.norm(x_true))
    vals["sol_resid"] = _rel(np.linalg.norm(Ax - b), nb)
    rep.add(kind, stage, **vals)


# ---------------------------------------------------------------------------
# RedBlackSolverHBS
# ---------------------------------------------------------------------------

def _dense_cyclic_reduction(M0, T0, P0, cyclic):
    """Exact red-black reduction on dense blocks, mirroring _build_level.

    Returns one (M, T, P) triple of dense-block lists per level, where at
    level l >= 1 M holds the A_i, T the B_i and P the C_i.  The last level
    holds a single node.
    """
    levels = [(M0, T0, P0)]
    M, T, P = M0, T0, P0
    while len(T) > 1:
        N = len(T)
        Mn, Tn, Pn = [], [], []
        for i in range(0, N, 2):
            kL, kR = (i - 1) % N, (i + 1) % N
            has_left  = cyclic or i > 0
            has_right = cyclic or i < N - 1
            B = T[i].copy()
            if has_right:
                B = B - P[i] @ np.linalg.solve(T[kR], M[kR])
            if has_left:
                B = B - M[i] @ np.linalg.solve(T[kL], P[kL])
            A = (-M[i] @ np.linalg.solve(T[kL], M[kL])) if has_left \
                else np.zeros_like(T[i])
            C = (-P[i] @ np.linalg.solve(T[kR], P[kR])) if has_right \
                else np.zeros_like(T[i])
            if cyclic and N == 2:
                # the lone survivor's self-couplings live on its diagonal
                B = B + A + C
                A = np.zeros_like(A); C = np.zeros_like(C)
            Mn.append(A); Tn.append(B); Pn.append(C)
        M, T, P = Mn, Tn, Pn
        levels.append((M, T, P))
    return levels


_RB_LABEL = re.compile(r"^([ABC])\[(\d+)\] \(nSlabs=(\d+)\)$")


def _rb_compressor(solver, seed):
    """Compress a dense block the way RedBlackSolverHBS compressed the block
    it stands for: the same path (from Gaussian samples of width
    _nsamples(rank) when fused, else from the operator), the same rank (level
    k was built by schedule stage k-1), tree, quad, device, `fast`, and the
    same compute_ULV decision.  Samples come from the diagnostics' own
    generator, so the solver's draws are untouched."""
    schedule = getattr(solver, "rkSchedule", None)
    rng = np.random.default_rng(seed)

    def compress(E, stage, label):
        rank = schedule[stage - 1] if schedule and stage >= 1 else solver.rk
        mt = _RB_LABEL.match(label or "")
        need = bool(mt) and mt.group(1) == "B" and \
            solver._needs_ulv(int(mt.group(2)), int(mt.group(3)))
        ulv = solver._want_ulv(need)
        if solver.fused:
            m, s = E.shape[0], solver._nsamples(rank)
            Om = rng.standard_normal((m, s))
            Psi = rng.standard_normal((m, s))
            h = HBSnew.HBSMAT(device=solver.device, tree=solver.tree, quad=solver.quad)
            h.construct(rank, Om=np.ascontiguousarray(Om), Psi=np.ascontiguousarray(Psi),
                        Y=np.ascontiguousarray(E @ Om),
                        Z=np.ascontiguousarray(E.conj().T @ Psi),
                        compute_ULV=ulv, fast=solver.fast)
        else:
            h = HBSnew.HBSMAT(_linop_from_mat(E), device=solver.device,
                              tree=solver.tree, quad=solver.quad)
            h.construct(rank, compute_ULV=ulv, fast=solver.fast)
        return _as_cpu(h)
    return compress


def _rb_step_ops(RB, l, i, cyclic):
    """Exact operators f(H_{l-1}) for retained node i of level l-1, built
    from the solver's compressed level l-1: (B, A or None, C or None)."""
    SiM, T, T_hbs, SiP = RB[l - 1]
    Np = len(T)
    kL, kR = (i - 1) % Np, (i + 1) % Np
    spm = SiP[kL] if (i > 0 or cyclic) else None
    smp = SiM[kR] if (i < Np - 1 or cyclic) else None
    tm  = T_hbs[kL] if spm is not None else None
    tp  = T_hbs[kR] if smp is not None else None
    B = RB_linop(T[i], tm, tp, SiP[i], SiM[i], smp, spm)
    A = STS_linop(SiM[i], T_hbs[kL], SiM[kL]) if (cyclic or i > 0) else None
    C = STS_linop(SiP[i], T_hbs[kR], SiP[kR]) if (cyclic or i < Np - 2) else None
    if cyclic and Np == 2:
        B = _sum_linop(*[op for op in (B, A, C) if op is not None])
        A = C = None
    return B, A, C


def diagnose_redblack(solver, S_exact=None, T_exact=None, rhs=None,
                      metrics=ALL_METRICS, nprobe=5, power_iters=50,
                      power_tol=1e-4, seed=12345, compress=None):
    """Comb and ladder errors for a factorized RedBlackSolverHBS.

    S_exact  : optional list of (left, right) exact uncompressed blocks, same
               layout as the S_rk_list passed to factorize (ndarray or any
               operator supporting `@`).  If omitted, the solver's inputs are
               taken as exact and stage-0 blocks are not compared.
    T_exact  : optional exact diagonals (default: the solver's level-0 T).
    rhs      : right-hand side for sol_* (default: seeded Gaussian).
    compress : optional callable (E_dense, stage, label) -> operator used for
               the comb's C(E_k); default compresses each block the way the
               solver compressed it (same path, rank and ULV decision).
    """
    RB, m, cyclic = solver.RB, solver.m, solver.cyclic
    meter = _Meter(metrics, nprobe, power_iters, power_tol, seed)
    compress = compress or _rb_compressor(solver, seed + 1)
    dens, lu = _densifier(m), _lu_cache()
    rep = ErrorReport("RedBlackSolverHBS")
    N = len(RB[0][1])
    L = len(RB)
    Z = np.zeros((m, m))
    slab = lambda l, j: j * 2 ** l          # level-l node j -> original slab

    def diff(H, M):
        # M is an exact array or a one-off reference operator: not cached
        return dens(H) - _dense(M, m)

    # ---- exact inputs (dense) and the exact path --------------------------
    SiM0, T0, _, SiP0 = RB[0]
    if S_exact is not None:
        Mx = [dens(S_exact[i][0]) for i in range(N)]
        Px = [dens(S_exact[i][-1]) for i in range(N)]
    else:
        Mx = [dens(SiM0[i]) for i in range(N)]
        Px = [dens(SiP0[i]) for i in range(N)]
    if not cyclic:
        Mx[0] = Z; Px[-1] = Z
    Tx = [dens(T_exact[i] if T_exact is not None else T0[i]) for i in range(N)]

    ref = _dense_cyclic_reduction(Mx, Tx, Px, cyclic)
    assert len(ref) == L, "reference depth does not match solver depth"

    # level-0 input error (shared by comb and ladder)
    sigma0 = _BlockOp(N, m)
    for i in range(N):
        if T_exact is not None:
            sigma0.add(i, i, diff(T0[i], Tx[i]))
        if S_exact is not None:
            if (cyclic or i > 0) and not _structural_skip(SiM0[i]):
                sigma0.add(i, (i - 1) % N, diff(SiM0[i], Mx[i]))
            if (cyclic or i < N - 1) and not _structural_skip(SiP0[i]):
                sigma0.add(i, (i + 1) % N, diff(SiP0[i], Px[i]))

    # ---- blockwise, stage 0 (comb == ladder) -------------------------------
    if meter.wants_norms():
        for i in range(N):
            pairs = []
            if S_exact is not None:
                pairs += [(f"SiM[{i}]", SiM0[i], Mx[i]),
                          (f"SiP[{i}]", SiP0[i], Px[i])]
            if T_exact is not None:
                pairs.append((f"T[{i}]", T0[i], Tx[i]))
            for lab, H, E in pairs:
                if _structural_skip(H):
                    continue
                vals = meter.block(H, E, m, dens)
                rep.add("comb", 0, lab, **vals)
                rep.add("ladder", 0, lab, **vals)

    # ---- stages 1..L-1: blockwise, and the per-stage error terms -----------
    comb_E   = {0: sigma0}     # stage -> Ã^comb_k - A
    comb_sys = {}              # stage -> dense (A, diag, C) of the comb's level-k system
    sigma    = {0: sigma0}     # ladder: level-l system error H_l - f(H_{l-1})
    mu       = {}              # ladder: odd-diagonal solve compression (compress_diag=False)

    for l in range(1, L):
        A_l, B_l, Th_l, C_l = RB[l]
        EA, EB, EC = ref[l]
        Np, Nl = len(RB[l - 1][1]), len(EB)
        cA, cD, cC = [], [], []
        comb_E[l] = _BlockOp(N, m)
        sigma[l] = _BlockOp(N, m)
        for j in range(Nl):
            i, tag = 2 * j, f"(nSlabs={Np})"
            labA, labB, labC = f"A[{i}] {tag}", f"B[{i}] {tag}", f"C[{i}] {tag}"
            r = slab(l, j)
            has_left  = cyclic or j > 0
            has_right = cyclic or j < Nl - 1
            cl, cr = slab(l, (j - 1) % Nl), slab(l, (j + 1) % Nl)
            fB, fA, fC = _rb_step_ops(RB, l, i, cyclic)

            # comb: compress each exact block the solver itself compresses
            if not isinstance(Th_l[j], dead_op):
                HB = compress(EB[j], l, labB)
                if meter.wants_norms():
                    rep.add("comb", l, labB, **meter.block(HB, EB[j], m, dens))
                if solver.compress_diag:
                    cD.append(dens(HB))
                    comb_E[l].add(r, r, cD[-1] - EB[j])
                else:
                    cD.append(EB[j])
            else:
                cD.append(EB[j])      # uncompressed diagonal is what flows on
            for lab, H_solver, E, out, col, present in (
                    (labA, A_l[j], EA[j], cA, cl, has_left),
                    (labC, C_l[j], EC[j], cC, cr, has_right)):
                if _structural_skip(H_solver) or not present:
                    out.append(Z)
                    continue
                Hc = compress(E, l, lab)
                if meter.wants_norms():
                    rep.add("comb", l, lab, **meter.block(Hc, E, m, dens))
                out.append(dens(Hc))
                comb_E[l].add(r, col, out[-1] - E)

            # ladder, blockwise: stored block vs exact
            if meter.wants_norms():
                H_diag = B_l[j] if isinstance(Th_l[j], dead_op) else Th_l[j]
                rep.add("ladder", l, labB, **meter.block(H_diag, EB[j], m, dens))
                if not _structural_skip(A_l[j]):
                    rep.add("ladder", l, labA, **meter.block(A_l[j], EA[j], m, dens))
                if not _structural_skip(C_l[j]):
                    rep.add("ladder", l, labC, **meter.block(C_l[j], EC[j], m, dens))

            # ladder, injected error of this level: H_l - f(H_{l-1})
            if solver.compress_diag:
                sigma[l].add(r, r, diff(B_l[j], fB))
            if has_left and not _structural_skip(A_l[j]):
                sigma[l].add(r, cl, diff(A_l[j], fA))
            if has_right and not _structural_skip(C_l[j]):
                sigma[l].add(r, cr, diff(C_l[j], fC))
        comb_sys[l] = (cA, cD, cC)

        # odd diagonals are compressed separately for solves when
        # compress_diag=False; that error enters once they are eliminated
        mu[l] = _BlockOp(N, m)
        if not solver.compress_diag:
            for j in range(1, Nl, 2):
                if Th_l[j] is not B_l[j] and not isinstance(Th_l[j], dead_op):
                    mu[l].add(slab(l, j), slab(l, j), diff(Th_l[j], B_l[j]))

    # ---- stage level --------------------------------------------------------
    if "partial" in meter.metrics:
        b = np.asarray(rhs) if rhs is not None else meter.rng.standard_normal(N * m)
        A_op = _btd_op(Mx, Tx, Px, cyclic, m)
        sigA = meter.power(A_op, N * m) if "power" in meter.metrics else None
        x_true = _dense_rb_inv(ref, cyclic, m, lu).solve(b)

        def dense_level(l):
            """Densified level-l system as stored by the solver."""
            SiM_l, T_l, _, SiP_l = RB[l]
            return ([dens(x) for x in SiM_l], [dens(x) for x in T_l],
                    [dens(x) for x in SiP_l])

        # comb: exact everywhere except stage k
        for l in range(L):
            if l == 0:
                levels = _dense_cyclic_reduction(*dense_level(0), cyclic)
            else:
                levels = ref[:l] + _dense_cyclic_reduction(*comb_sys[l], cyclic)
            inv = _dense_rb_inv(levels, cyclic, m, lu)
            _stage_record(rep, meter, "comb", l, A_op, sigA, comb_E[l], inv,
                          b, x_true, inv.solve(b))

        # ladder: the solver's own stages 0..k.  Forward through its own
        # solve code; adjoint through the same sweep on its HBS levels.
        hbs_levels = [(RB[l][0], RB[l][2], RB[l][3]) for l in range(L - 1)]
        E = _BlockOp(N, m)
        for l in range(L):
            E = E + sigma[l] + (mu[l - 1] if l >= 2 else _BlockOp(N, m))
            lvl = sigma[l] + mu[l] if l in mu else sigma[l]
            trailing = _dense_rb_inv(
                _dense_cyclic_reduction(*dense_level(l), cyclic), cyclic, m, lu)

            def fwd(v, l=l, trailing=trailing):
                vP = solver._forward_reduce(v, l)
                vP[-1] = trailing.solve(vP[-1])
                return np.asarray(solver._back_substitute(vP))

            inv = _Inv(_columnwise(fwd),
                       lambda X, l=l, trailing=trailing:
                           _rb_sweep(hbs_levels[:l], trailing, cyclic, m, X, 'T'))
            _stage_record(rep, meter, "ladder", l, A_op, sigA, E, inv,
                          b, x_true, fwd(b), lvl=lvl)

        # the full solve: coarsest node solved with its compressed diagonal
        E_solve = E + (mu[L - 1] if L - 1 in mu else _BlockOp(N, m))
        coarse_T, coarse_Th = RB[L - 1][1][0], RB[L - 1][2][0]
        if coarse_Th is not coarse_T and not isinstance(coarse_Th, dead_op):
            E_solve.add(0, 0, diff(coarse_Th, coarse_T))
        inv = _Inv(_columnwise(lambda v: np.asarray(solver.solve(v.copy()))),
                   lambda X: _rb_sweep(hbs_levels, coarse_Th, cyclic, m, X, 'T'))
        _stage_record(rep, meter, "ladder", "solve", A_op, sigA, E_solve, inv,
                      b, x_true, np.asarray(solver.solve(b.copy())))

    return rep


# ---------------------------------------------------------------------------
# ThomasSolverHBS
# ---------------------------------------------------------------------------

def _thomas_recurrence(P_start, k, L, R, D):
    """P_k = P_start, then P_i = D_i - L_{i-1} P_{i-1}^{-1} R_{i-1} for i > k."""
    out = [P_start]
    for i in range(k + 1, len(D)):
        out.append(D[i] - L[i - 1] @ np.linalg.solve(out[-1], R[i - 1]))
    return out


def _thomas_compressor(solver):
    """Compress a dense block exactly as ThomasSolverHBS compressed it at that
    stage: from the operator, with a ULV, at the same rank (block k was built
    by schedule stage k-1), tree, quad and device."""
    with_diag = solver.solve_method == "diag"
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    schedule = getattr(solver, "rkSchedule", None)

    def compress(E, stage, label):
        rank = schedule[stage - 1] if schedule and stage >= 1 else solver.rk
        like = solver.D[stage] if with_diag else solver.A[stage - 1]
        kw = {} if with_diag else {"device": dev}
        h = HBSnew.HBSMAT(_linop_from_mat(E), tree=getattr(like, "tree", None),
                          quad=getattr(like, "quad", None), **kw)
        h.construct(rank, compute_ULV=True, fast=True)
        return _as_cpu(h)
    return compress


def diagnose_thomas(solver, S_exact=None, D_exact=None, rhs=None,
                    metrics=ALL_METRICS, nprobe=5, power_iters=50,
                    power_tol=1e-4, seed=12345, compress=None):
    """Comb and ladder errors for a factorized ThomasSolverHBS.

    Stage 0 is the entry point (the input off-diagonals, and the input
    diagonals in the with-diag variant); stage i >= 1 is S'_i (resp. B_i).

    S_exact  : optional list of (left, right) exact blocks in the layout the
               solver was factorized with.  If omitted the inputs are taken
               as exact.
    D_exact  : optional exact diagonals for the with-diag variant.
    compress : optional callable (E_dense, stage, label) -> operator for the
               comb's C(E_k); default compresses each block the way the
               solver compressed it (same path, rank and ULV).
    """
    if solver.cyclic:
        raise NotImplementedError("cyclic ThomasSolverHBS (SMW correction) "
                                  "is not covered")
    if solver.solve_method not in ("id_diag", "diag"):
        raise ValueError("solver is not factorized")

    m = solver.m
    with_diag = solver.solve_method == "diag"
    Lh, Bh, Rh = solver.A, solver.B, solver.C
    n = len(Lh)
    N = n + 1
    meter = _Meter(metrics, nprobe, power_iters, power_tol, seed)
    compress = compress or _thomas_compressor(solver)
    dens, lu = _densifier(m), _lu_cache()
    rep = ErrorReport(f"ThomasSolverHBS ({solver.solve_method})")
    I = np.eye(m)

    # ---- inputs, dense: as given to the solver, and exact -----------------
    Ld = [dens(Lh[i]) for i in range(n)]
    Rd = [dens(Rh[i]) for i in range(n)]
    Dd = [dens(solver.D[i]) for i in range(N)] if with_diag else [I] * N
    # The two factorize paths index the left blocks differently:
    #   id_diag:  Sl[i] = S_rk_list[i+1][0]   (entry i holds block (i, i-1))
    #   diag:     A[i]  = AB_list[i][0]       (entry i holds block (i+1, i))
    # S_exact is read with whichever convention the solver actually used.
    off = 0 if with_diag else 1
    if S_exact is not None:
        Lx = [dens(S_exact[i + off][0]) for i in range(n)]
        Rx = [dens(S_exact[i][-1]) for i in range(n)]
    else:
        Lx, Rx = Ld, Rd
    Dx = [dens(D_exact[i]) for i in range(N)] if D_exact is not None else Dd

    Px = _thomas_recurrence(Dx[0], 0, Lx, Rx, Dx)     # exact path
    name = "B" if with_diag else "S'"

    # level-0 input error (shared by comb and ladder)
    sigma0 = _BlockOp(N, m)
    if S_exact is not None:
        for i in range(n):
            sigma0.add(i + 1, i, Ld[i] - Lx[i])
            sigma0.add(i, i + 1, Rd[i] - Rx[i])
    if with_diag and D_exact is not None:
        for i in range(N):
            sigma0.add(i, i, Dd[i] - Dx[i])

    # ---- blockwise ----------------------------------------------------------
    if meter.wants_norms():
        pairs = []
        if S_exact is not None:
            pairs += [(f"Sl[{i}]", Lh[i], Lx[i]) for i in range(n)]
            pairs += [(f"Sr[{i}]", Rh[i], Rx[i]) for i in range(n)]
        if with_diag and D_exact is not None:
            pairs += [(f"D[{i}]", solver.D[i], Dx[i]) for i in range(N)]
        for lab, H, E in pairs:
            vals = meter.block(H, E, m, dens)
            rep.add("comb", 0, lab, **vals)
            rep.add("ladder", 0, lab, **vals)

    comb_blocks, comb_E, sigma = {}, {0: sigma0}, {0: sigma0}
    for i in range(1, N):
        lab = f"{name}[{i}]"
        Hc = compress(Px[i], i, lab)
        comb_blocks[i] = dens(Hc)
        comb_E[i] = _BlockOp(N, m, [(i, i, comb_blocks[i] - Px[i])])
        if with_diag:
            f_i = Dprime_Linop(solver.D[i], Lh[i - 1], Rh[i - 1], Bh[i - 1])
        else:
            f_i = Sprime_Linop(Lh[i - 1], Bh[i - 1], Rh[i - 1], id=(i == 1))
        sigma[i] = _BlockOp(N, m, [(i, i, dens(Bh[i]) - dens(f_i))])
        if meter.wants_norms():
            rep.add("comb", i, lab, **meter.block(Hc, Px[i], m, dens))
            rep.add("ladder", i, lab, **meter.block(Bh[i], Px[i], m, dens))

    # ---- stage level --------------------------------------------------------
    if "partial" in meter.metrics:
        b = np.asarray(rhs) if rhs is not None else meter.rng.standard_normal(N * m)
        A_op = _BlockOp(N, m, [(i, i, Dx[i]) for i in range(N)]
                        + [(i + 1, i, Lx[i]) for i in range(n)]
                        + [(i, i + 1, Rx[i]) for i in range(n)])
        sigA = meter.power(A_op, N * m) if "power" in meter.metrics else None

        def dense_inv(Lb, P, Rb):
            Ps = [lu(p) for p in P]
            return _Inv(lambda X: _thomas_sweep(Lb, Ps, Rb, X, m, 'N'),
                        lambda X: _thomas_sweep(Lb, Ps, Rb, X, m, 'T'))

        x_true = dense_inv(Lx, Px, Rx).solve(b)

        # comb: stage 0 = the solver's inputs with an exact recurrence;
        # stage k = exact everywhere except S'_k -> C(E_k)
        inv = dense_inv(Ld, _thomas_recurrence(Dd[0], 0, Ld, Rd, Dd), Rd)
        _stage_record(rep, meter, "comb", 0, A_op, sigA, comb_E[0], inv,
                      b, x_true, inv.solve(b))
        for k in range(1, N):
            P = Px[:k] + _thomas_recurrence(comb_blocks[k], k, Lx, Rx, Dx)
            inv = dense_inv(Lx, P, Rx)
            _stage_record(rep, meter, "comb", k, A_op, sigA, comb_E[k], inv,
                          b, x_true, inv.solve(b))

        # ladder: the solver's stages 0..k.  Forward through its own solve
        # code; adjoint through the same sweep on its blocks.
        E = _BlockOp(N, m)
        for k in range(N):
            E = E + sigma[k]
            Bs = list(Bh[:k + 1]) + [_DenseLU(P) for P in
                                     _thomas_recurrence(dens(Bh[k]), k, Ld, Rd, Dd)[1:]]
            if with_diag:
                fwd = lambda X, Bs=Bs: np.asarray(solver.solve_with_diag(X.copy(), B=Bs))
            else:
                fwd = lambda X, Bs=Bs: np.asarray(solver.solve_id_diag(X.copy(), Sprime=Bs))
            inv = _Inv(fwd, lambda X, Bs=Bs: _thomas_sweep(Lh, Bs, Rh, X, m, 'T'))
            _stage_record(rep, meter, "ladder", k, A_op, sigA, E, inv,
                          b, x_true, inv.solve(b), lvl=sigma[k])

    return rep