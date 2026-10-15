import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.linalg as tla
import matAssembly.HBS.ULVsparse_torch as ULVsparse

import contextlib

# ---------------------------------------------------------------------------
# Memory policy (host-first)
# ---------------------------------------------------------------------------
# An HBSMAT keeps one persistent copy of its factors: a host master, one flat
# (page-locked when possible) buffer per group.  Umats/Vmats/Dmats are views
# into that master and always live on the host.
#
# constructHBS streams each level's blocks through the compute device in
# chunks.  Per chunk, both sides (Om/Y and Psi/Z) go through one fused batched
# kernel; U, V, D are written to the host master as soon as they exist; the
# next level's samples stay on the device unless they do not fit next to the
# working set (checked per level), in which case they go to page-locked host
# memory.
#
# On CUDA the chunks run through a three-stream pipeline
#     h2d  : uploads chunk j+1          (leaf level, or a spilled level)
#     comp : computes chunk j
#     d2h  : writes chunk j-1's factors to the host master
# ordered by events, with at most _PIPE_DEPTH chunks in flight.  The host
# only ever waits for that throttle and for the final synchronize.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# module knobs
# ---------------------------------------------------------------------------
_PIN_HOST     = [True]     # page-lock host masters and host work buffers
_CHUNK_BYTES  = [2 << 30]  # device working set per construction chunk; sets
                           # the blocks per chunk.  0 means one chunk per level
_UV_CHUNK_ELEMS = [1 << 28]  # cap on (2c)*n*s per chunk (32-bit size limits
                           # in batched cuSOLVER/cuBLAS).  0 disables
_SAMPLE_SPILL_BYTES = [None]  # next level's samples:
                           #   None  device if they fit in the device budget
                           #         next to what is allocated, else host
                           #   0     always host;  int  host above that size
_DEVICE_BUDGET = [None]    # bytes the code may plan to use; None = 85% of total
_PIPE_DEPTH   = [3]        # chunks in flight in the construction pipeline
_PINNED       = [0]        # bytes this module currently has page-locked

_HOST = torch.device('cpu')


def device_budget(dev):
    """Bytes of device memory planned decisions may use (see _DEVICE_BUDGET)."""
    if _DEVICE_BUDGET[0] is not None:
        return int(_DEVICE_BUDGET[0])
    dev = torch.device(dev)
    if dev.type == 'cuda':
        return int(0.85 * torch.cuda.get_device_properties(dev).total_memory)
    return 1 << 62


# ---------------------------------------------------------------------------
# page-locked host buffers
# ---------------------------------------------------------------------------

class _HostBuffer:
    """Flat host buffer, page-locked in place with cudaHostRegister.

    Allocated pageable at the exact size and then registered, instead of
    torch.empty(..., pin_memory=True): the caching host allocator rounds up
    to a power of two and keeps freed blocks, which for multi-GB groups can
    nearly double host usage.  If registration is unavailable the buffer
    stays pageable: transfers still work, just synchronously."""
    __slots__ = ('t', 'registered', 'kind')

    def __init__(self, numel, dtype, kind='master'):
        self.t = torch.empty(int(numel), dtype=dtype)
        self.registered = False
        self.kind = kind
        nb = self.t.numel() * self.t.element_size()
        if torch.cuda.is_available() and _PIN_HOST[0] and nb > 0:
            try:
                rc = torch.cuda.cudart().cudaHostRegister(self.t.data_ptr(), nb, 0)
                self.registered = (rc is None) or (int(rc) == 0)
            except Exception:
                self.registered = False
            if self.registered:
                _PINNED[0] += nb

    def __del__(self):
        if getattr(self, 'registered', False):
            try:
                torch.cuda.cudart().cudaHostUnregister(self.t.data_ptr())
                _PINNED[0] -= self.t.numel() * self.t.element_size()
            except Exception:
                pass


class HostPool:
    """Reusable page-locked host buffers (registration is slow, ~1 s per
    10 GB, so buffers are kept and handed out again).  get() returns a 1-D
    view of exactly the requested length; put() takes it back."""

    def __init__(self):
        self._free = []
        self._lent = {}

    def get(self, numel, dtype=torch.float64, pinned=True):
        numel = int(numel)
        if not pinned or numel == 0:
            return torch.empty(numel, dtype=dtype)
        best = None
        for hb in self._free:
            if (hb.t.dtype == dtype and hb.t.numel() >= numel and
                    (best is None or hb.t.numel() < best.t.numel())):
                best = hb
        if best is None:
            best = _HostBuffer(numel, dtype, kind='work')
        else:
            self._free.remove(best)
        self._lent[best.t.data_ptr()] = best
        return best.t[:numel]

    def put(self, t):
        if t is None:
            return
        hb = self._lent.pop(t.data_ptr(), None)
        if hb is not None:
            self._free.append(hb)

    def clear(self):
        self._free = []

    @property
    def nbytes(self):
        """(bytes handed out, bytes held free)"""
        f = lambda hbs: sum(hb.t.numel() * hb.t.element_size() for hb in hbs)
        return f(self._lent.values()), f(self._free)


host_pool = HostPool()


# ---------------------------------------------------------------------------
# host masters and level layout
# ---------------------------------------------------------------------------

def _numel(shape):
    return math.prod(shape)


def _views(flat, layout):
    """{name: [view, ...]} carving `flat` according to `layout`."""
    out, o = {}, 0
    for name, shapes in layout.items():
        lst = []
        for shp in shapes:
            n = _numel(shp)
            lst.append(flat[o:o + n].view(shp))
            o += n
        out[name] = lst
    return out


def _new_master(layout, dtype):
    """Allocate a host master for `layout`.  Returns (record, lists)."""
    total = sum(_numel(s) for shapes in layout.values() for s in shapes)
    buf = _HostBuffer(total, dtype)
    rec = {'bufs': {dtype: buf}, 'layout': layout, 'dtype': dtype}
    return rec, _views(buf.t, layout)


def _level_shapes(Nb, nl, L, rk, fac):
    """([(Nb_i, n_i, k_i) for i = 0..L-2], (Nb_r, n_r)): index 0 is the leaf,
    the pair is the root.  k_i = min(rk, nl*fac**i, n_i) and the next level
    has n_{i+1} = fac * k_i rows per block."""
    levels, n = [], nl
    for i in range(L - 1):
        k = min(rk, nl * fac ** i, n)
        levels.append((Nb, n, k))
        Nb //= fac
        n = fac * k
    return levels, (Nb, n)


def _core_layout(levels, root):
    lay = {'Umats': [], 'Vmats': [], 'Dmats': []}
    for Nb, n, k in levels:
        lay['Umats'].append((Nb, n, k))
        lay['Vmats'].append((Nb, n, k))
        lay['Dmats'].append((Nb, n, n))
    lay['Dmats'].append((root[0], root[1], root[1]))
    return lay


def _ulv_layout(levels, root):
    """Wlist and Uulist have no root entry: ULVsparse.solve never reads one."""
    lay = {'Qlist': [], 'Wlist': [], 'Rlist': [], 'Uulist': []}
    for Nb, n, k in levels:
        lay['Qlist'].append((Nb, n, n))
        lay['Wlist'].append((Nb, n, n - k))
        lay['Rlist'].append((Nb, n - k, n))
        lay['Uulist'].append((Nb, n - k, k))
    lay['Qlist'].append((root[0], root[1], root[1]))
    lay['Rlist'].append((root[0], root[1], root[1]))
    return lay


def _put(dst, src, pinned):
    """dst[...] = src; asynchronous when dst is page-locked or on a device."""
    dst.copy_(src, non_blocking=pinned or dst.device.type != 'cpu')


def block_matvec(A, B, device=None, mode='N'):
    """A: (Nb, n, k) block diagonal; B: (Nb*nB, col) flat."""
    Nb  = A.shape[0]
    col = B.shape[1]
    Bm  = B.reshape(Nb, B.shape[0] // Nb, col)
    if mode == 'N':
        return torch.bmm(A, Bm).reshape(Nb * A.shape[1], col)
    elif mode == 'T':
        return torch.bmm(A.mT, Bm).reshape(Nb * A.shape[2], col)
    raise ValueError("mode not recognized")


# ---------------------------------------------------------------------------
# construction kernels: batched, any leading batch dims, no host syncs
# ---------------------------------------------------------------------------
#
# For a wide block Om (n x s, full row rank n) and Y (n x s), one unpivoted
# Householder QR of Om^T serves both the null space and the pseudoinverse:
#
#     Om^T = Q [R; 0],  Q = [Q1 | Q2],  Q1: s x n
#     P    = last k columns of Q          (orthonormal, Om P = 0)
#     U    = Q-factor of Y P              (unpivoted Householder QR)
#     M    = Y Om^+ = (Y Q1) R^{-T}
#
# Q is never formed: the reflectors are applied to a selector of the n + k
# needed columns, and Y [Q1 | P] is one GEMM.  No SVD, no Gram matrix.

def _selector(s, n, k, dtype, device):
    """(s, n+k): picks columns 0..n-1 and s-k..s-1 of an s x s matrix."""
    S = torch.zeros(s, n + k, dtype=dtype, device=device)
    S[:n, :n] = torch.eye(n, dtype=dtype, device=device)
    S[s - k:, n:] = torch.eye(k, dtype=dtype, device=device)
    return S


def _null_col_pinv(Om, Y, k, sel=None):
    """Om, Y: (..., n, s).  Returns U (..., n, k) contiguous and
    M = Y Om^+ (..., n, n).  Requires k <= s - n and k <= n."""
    n, s = Om.shape[-2:]
    batch = Om.shape[:-2]
    Vh, tau = torch.geqrf(Om.mT)                     # (..., s, n), (..., n)
    R = Vh[..., :n, :].triu()
    if sel is None:
        sel = _selector(s, n, k, Om.dtype, Om.device)
    Qsel = torch.ormqr(Vh, tau, sel.expand(*batch, s, n + k))   # [Q1 | P]
    del Vh, tau
    W = Y @ Qsel                                     # [Y Q1 | Y P]
    del Qsel
    M = torch.linalg.solve_triangular(R, W[..., :n].mT, upper=True).mT
    U = torch.linalg.qr(W[..., n:], mode='reduced').Q
    return U.contiguous(), M


def _pinv_right(Om, Y):
    """Y Om^+ for wide, full-row-rank Om (..., n, s); Y (..., m, s)."""
    n = Om.shape[-2]
    Vh, tau = torch.geqrf(Om.mT)
    R = Vh[..., :n, :].triu()
    Q1 = torch.linalg.householder_product(Vh, tau)  # (..., s, n)
    del Vh, tau
    return torch.linalg.solve_triangular(R, (Y @ Q1).mT, upper=True).mT.contiguous()


def construct_D(U, V, M_om, M_psi):
    """D = (I - U U^T) Y Om^+  +  U U^T (Z Psi^+)^T (I - V V^T)
    with M_om = Y Om^+ and M_psi = Z Psi^+."""
    P  = M_om - torch.bmm(U, torch.bmm(U.mT, M_om))      # (c, n, n)
    Gk = torch.bmm(M_psi, U)                             # (c, n, k)
    Gk = Gk - torch.bmm(V, torch.bmm(V.mT, Gk))
    return torch.baddbmm(P, U, Gk.mT)


def _level_rest(B, M, X, S, nside, outX, outS):
    """D and the next level's samples for one chunk.

    B = [U; V] (2, c, n, k), M = [Y Om^+; Z Psi^+] (2, c, n, n),
    X = [Om; Psi], S = [Y; Z] (2, c, n, s).  Next-level samples:

        Om'  = V^T Om          Y' = U^T Y - (U^T D) Om
        Psi' = U^T Psi         Z' = V^T Z - (D V)^T Psi

    written into outX = [Om'; Psi'], outS = [Y'; Z'] (nside, c, k, s), or
    into fresh tensors when those are None.  nside = 1 skips Psi', Z'."""
    U, V = B[0], B[1]
    D = construct_D(U, V, M[0], M[1])
    c, n, k = U.shape
    s = X.shape[-1]
    if outX is None:
        outX = X.new_empty((nside, c, k, s))
        outS = X.new_empty((nside, c, k, s))
    left = (V, U)                                   # Om' = V^T Om, Psi' = U^T Psi
    G = (torch.bmm(U.mT, D), None if nside == 1 else torch.bmm(D, V).mT)
    for t in range(nside):
        torch.bmm(left[t].mT, X[t], out=outX[t])
        torch.baddbmm(torch.bmm(B[t].mT, S[t]), G[t], X[t], alpha=-1.0,
                      out=outS[t])
    return D, outX, outS


# ---------------------------------------------------------------------------
# construction pipeline
# ---------------------------------------------------------------------------

class _Pipe:
    """Three-stream (h2d / comp / d2h) chunk pipeline on CUDA, plain
    sequential execution on any other device.

    Per chunk in slot j = i % depth:  h2d records loaded[j]; comp waits for
    it and records computed[j]; d2h waits for that and records done[j].
    begin() blocks the host on done[j] of chunk i - depth, which bounds the
    device memory in flight and makes slot j's staging buffers reusable."""

    def __init__(self, dev, depth):
        self.cuda = dev.type == 'cuda'
        self.depth = max(1, int(depth))
        self.i = 0
        self.timers = []                    # (attr, start, end)
        if self.cuda:
            self.dev = dev
            self.h2d, self.comp, self.d2h = (torch.cuda.Stream(dev)
                                             for _ in range(3))
            cur = torch.cuda.current_stream(dev)
            for st in (self.h2d, self.comp, self.d2h):
                st.wait_stream(cur)         # caller's pending work on inputs
            ev = lambda: [torch.cuda.Event() for _ in range(self.depth)]
            self.ev = {'loaded': ev(), 'computed': ev(), 'done': ev()}

    def on(self, stream):
        if self.cuda:
            return torch.cuda.stream(getattr(self, stream))
        return contextlib.nullcontext()

    def begin(self):
        j = self.i % self.depth
        if self.cuda and self.i >= self.depth:
            self.ev['done'][j].synchronize()
        self.i += 1
        return j

    def record(self, name, j, stream):
        if self.cuda:
            self.ev[name][j].record(getattr(self, stream))

    def wait(self, stream, name, j):
        if self.cuda:
            getattr(self, stream).wait_event(self.ev[name][j])

    def wait_stream(self, stream, other):
        if self.cuda:
            getattr(self, stream).wait_stream(getattr(self, other))

    def keep(self, tensors, stream):
        """Tensors allocated on one stream and used on `stream`."""
        if self.cuda:
            st = getattr(self, stream)
            for t in tensors:
                if t is not None and t.is_cuda:
                    t.record_stream(st)

    def tic(self, stream):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record(getattr(self, stream))
            return e
        return time.time()

    def toc(self, attr, stream, t0):
        self.timers.append((attr, t0, self.tic(stream)))

    def finish(self):
        """Synchronize; returns {attr: seconds}."""
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        out = {}
        for attr, a, b in self.timers:
            dt = a.elapsed_time(b) / 1e3 if self.cuda else b - a
            out[attr] = out.get(attr, 0.0) + dt
        return out


class _LeafSrc:
    """The leaf samples Om, Psi, Y, Z: (m, s) each, rows in tree order.

    fetch(a, b) returns blocks a:b as X = [Om; Psi], S = [Y; Z] of shape
    (nside, b-a, n, s) on the compute device.  A host sample that is
    page-locked, row-major and of the factor dtype moves with one async DMA
    per chunk; any other host sample is first copied into a page-locked
    staging buffer of the chunk's slot (a CPU copy, which also fixes layout
    and dtype)."""

    def __init__(self, Xs, n, dt, dev, cuda):
        self.Xs, self.n, self.dt, self.dev, self.cuda = Xs, n, dt, dev, cuda
        self.s = Xs[0].shape[1]
        self.on_host = any(X.device.type == 'cpu' for X in Xs)
        self.direct = [X.device.type == 'cpu' and X.dtype == dt and
                       X.is_contiguous() and cuda and X.is_pinned()
                       for X in Xs]
        self.stage = {}
        self.h2d_bytes = 0

    def _staging(self, j, slot, numel):
        buf = self.stage.get((j, slot))
        if buf is None or buf.numel() < numel:
            host_pool.put(buf)
            buf = self.stage[(j, slot)] = host_pool.get(numel, self.dt)
        return buf[:numel]

    def fetch(self, a, b, nside, slot):
        n, s, c = self.n, self.s, b - a
        X = torch.empty((nside, c, n, s), dtype=self.dt, device=self.dev)
        S = torch.empty((nside, c, n, s), dtype=self.dt, device=self.dev)
        for t in range(nside):
            for j, dst in ((t, X[t]), (2 + t, S[t])):
                piece = self.Xs[j][a * n:b * n]
                d2 = dst.view(c * n, s)
                if piece.device.type != 'cpu' or not self.cuda:
                    d2.copy_(piece)                         # same side of the bus
                else:
                    if not self.direct[j]:
                        st = self._staging(j, slot, c * n * s).view(c * n, s)
                        st.copy_(piece)
                        piece = st
                    d2.copy_(piece, non_blocking=True)
                    self.h2d_bytes += d2.nbytes
        return X, S, [X, S]

    def release(self):
        for buf in self.stage.values():
            host_pool.put(buf)
        self.stage = {}


class _LevelSrc:
    """An upper level's samples X = [Om; Psi], S = [Y; Z], each
    (nside, Nb, n, s), on the device or in page-locked host memory."""

    def __init__(self, X, S, on_host, dev):
        self.X, self.S, self.on_host, self.dev = X, S, on_host, dev
        self.h2d_bytes = 0

    def fetch(self, a, b, nside, slot):
        if not self.on_host:
            return self.X[:nside, a:b], self.S[:nside, a:b], []
        n, s = self.X.shape[-2:]
        X = torch.empty((nside, b - a, n, s), dtype=self.X.dtype, device=self.dev)
        S = torch.empty_like(X)
        for t in range(nside):
            X[t].copy_(self.X[t, a:b], non_blocking=True)
            S[t].copy_(self.S[t, a:b], non_blocking=True)
            self.h2d_bytes += X[t].nbytes + S[t].nbytes
        return X, S, [X, S]

    def release(self):
        pass


def _as_samples(X, m):
    """A leaf sample as a 2-D (m, s) tensor, keeping its storage and layout
    when possible (numpy and 2-D tensors are never copied)."""
    if isinstance(X, list):                 # boxed: take the only reference
        X = X.pop()
    if not torch.is_tensor(X):
        X = torch.from_numpy(np.asarray(X))
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    if X.dim() != 2 or X.shape[0] != m:
        raise ValueError(f"sample must be ({m}, s) or (Nb, nl, s); got "
                         f"{tuple(X.shape)}")
    return X




class HBSMAT:
    """HBS matrix: compression, apply, and ULV solve.

    @init:      linear operator A, tree on DOFs (symmetric), target rank k
    @constructs HBS approximation to the source-target map
    @implements matvec/matmat (normal and transpose) and solve

    Memory policy: see the top of this module.  In short, the factor lists
    always hold host tensors (views of the host master); the device only ever
    holds a block's factors while one of its operations runs, or between
    prefetch() and evict().

    Level ordering: list index 0 is the leaf level and Dmats[-1] the root.
    Wlist and Uulist have no root entry.
    """

    # every attribute holding a list of tensors, by group
    _GROUPS = {
        'core': ('Umats', 'Vmats', 'Dmats'),
        'ulv' : ('Qlist', 'Wlist', 'Rlist', 'Uulist'),
    }

    def __init__(self, A=None, device=None, tree=None, quad=False):
        self.Umats  = []
        self.Vmats  = []
        self.Dmats  = []
        self.Qlist  = []
        self.Rlist  = []
        self.Wlist  = []
        self.Uulist = []
        torch.set_default_dtype(torch.float64)
        self.dtype   = np.float64
        self.dtype_t = torch.float64

        self.mode  = 'N'
        self._tree = None
        self._perm = None

        self._host   = {'core': None, 'ulv': None}   # host master records
        self._dev    = {}     # group -> {name: [device tensors], 'perm': ...}
                              # mutated in place, never rebound, so .T views
                              # share it
        self.compute_device = (torch.device(device) if device is not None
                               else _HOST)
        self.home    = _HOST
        self.strict  = False
        self.nFill = self.nSpill = 0
        self.bytesH2D = self.bytesD2H = 0

        self.A = A
        if A is not None:
            self.shape = A.shape
            self.dtype = A.dtype

        if tree is not None:
            self.tree  = tree
            self.perm  = tree.perm_leaf
            self.Nb    = tree.nleaves
            self.nl    = len(self.perm) // self.Nb
            self.L     = tree.nlevels
            self.shape = (len(self.perm), len(self.perm))

        self.blockSolveTime = 0
        self.nullTime   = 0
        self.setupTime  = 0
        self.DTime      = 0
        self.tSample    = 0
        self.tConstruct = 0
        self.tULV       = 0
        self.tCompress  = 0
        self.Nbvec = []
        self.quad  = quad
        self.fac   = 4 if quad else 2

    def set_Nbvec(self, Nbvec):
        self.Nbvec = Nbvec
        self.Nb = Nbvec[0]
        self.nl = self.A.shape[0] // self.Nb
        self.L  = len(Nbvec)
        self.perm = torch.arange(self.A.shape[0], dtype=torch.int64)

    def set_mats(self, Umats, Dmats, Vmats, Nbvec, fac=4):
        """Adopt externally built factors.  They are copied into a host
        master; the arguments may live anywhere."""
        layout = {'Umats': [tuple(t.shape) for t in Umats],
                  'Vmats': [tuple(t.shape) for t in Vmats],
                  'Dmats': [tuple(t.shape) for t in Dmats]}
        self._dev.clear()
        self._adopt_group('core', layout, {'Umats': Umats, 'Vmats': Vmats,
                                           'Dmats': Dmats})
        self._host['ulv'] = None
        for nm in self._GROUPS['ulv']:
            setattr(self, nm, [])
        n = sum(D.shape[0] * D.shape[1] for D in Dmats[:1])
        self.perm  = torch.arange(n)
        self.fac   = fac
        self.Nbvec = list(Nbvec)
        self.Nb    = Nbvec[0]
        self.L     = len(Dmats)
        self.shape = torch.tensor([n, n], dtype=torch.int64)
        self.dtype = Dmats[0].dtype


    @property
    def nbytes(self):
        return (sum(U.nbytes for U in self.Umats)
                + sum(V.nbytes for V in self.Vmats)
                + sum(D.nbytes for D in self.Dmats))

    @property
    def device(self):
        return str(self.compute_device)

    # ------------------------------------------------------------------
    # tree / perm
    # ------------------------------------------------------------------

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, t):
        self._tree = t

    @property
    def perm(self):
        """Leaf permutation, always a host int64 tensor."""
        return self._perm

    @perm.setter
    def perm(self, p):
        if p is None:
            self._perm = None
        elif torch.is_tensor(p):
            self._perm = p.detach().to(device=_HOST, dtype=torch.int64)
        else:
            self._perm = torch.as_tensor(np.asarray(p), dtype=torch.int64)


    def construct(self, rk, Om=None, Psi=None, Y=None, Z=None,
                  compute_ULV=False, fast=False):
        """Compress from samples, or draw them from self.A.

        Given samples must already be in tree order (row r is original row
        perm[r]).  Samples drawn here are permuted into that order."""
        if Om is None:
            if self.A is None:
                raise ValueError("Samples and LinOP cannot both be None")
            # hard requirement max(fac*rk, nl) + rk at the worst level, plus
            # oversampling
            n_max = max(self.nl, 2*rk)
            p = 10
            s = max(n_max + rk + p , (int)(np.ceil(1.5 * n_max)) )
            Om  = np.random.standard_normal(size=(self.A.shape[1], s))
            Psi = np.random.standard_normal(size=(self.A.shape[0], s))
            Y   = self.A @ Om
            Z   = self.A.T @ Psi
            if self.perm is not None:
                pr = self.perm.numpy()
                Om, Psi, Y, Z = (np.asarray(X)[pr] for X in (Om, Psi, Y, Z))
        self.constructHBS(rk, Om, Psi, Y, Z, fast=fast, with_ulv=compute_ULV)

    def _chunk_blocks(self, Nb, n, s, k=None, depth=1):
        """(blocks per chunk, device bytes of that chunk's working set).

        Per block, in elements.  A non-root level (k given), both sides:
        inputs 4ns, geqrf copy 2ns, [Q1|P] 2s(n+k), Y[Q1|P] 2n(n+k), M 2n^2,
        D and construct_D temporaries 3n^2, B 4nk, next samples and their
        temporaries 6ks, with ~ns slack for cuSOLVER workspace.  The root
        (k None), one side: inputs 2ns, geqrf + Q1 2ns, n^2 each for Y Q1, D,
        a temporary.  The depth-1 other chunks in flight hold their inputs and
        outputs.  Deliberately rough; _CHUNK_BYTES is the knob."""
        item = torch.finfo(self.dtype_t).bits // 8
        if k is None:
            per = 4 * n * s + 3 * n * n
            fly = 2 * n * s + n * n
        else:
            per = (7 * n * s + 2 * s * (n + k) + 2 * n * (n + k)
                   + 5 * n * n + 4 * n * k + 6 * k * s)
            fly = 4 * n * s + 2 * n * k + n * n
        per_b = item * (per + (depth - 1) * fly)
        c = Nb
        if _CHUNK_BYTES[0]:
            c = min(c, _CHUNK_BYTES[0] // per_b)
        if _UV_CHUNK_ELEMS[0]:
            c = min(c, _UV_CHUNK_ELEMS[0] // max(1, 2 * n * s))
        c = int(max(1, c))
        return c, c * per_b

    def _spill_next(self, nbytes, ws):
        """Keep the next level's samples (nbytes) on the host?  Decided per
        level against the device budget, next to what is allocated now and
        this level's chunk working set ws.  See _SAMPLE_SPILL_BYTES."""
        dev = torch.device(self.compute_device)
        if dev.type != 'cuda':
            return False
        lim = _SAMPLE_SPILL_BYTES[0]
        if lim is not None:
            return nbytes > lim
        used = torch.cuda.memory_allocated(dev)
        return used + nbytes + ws > device_budget(dev)

    def constructHBS(self, rk, Om0, Psi0, Y0, Z0, fast=False, with_ulv=False):
        """Compress from samples Y = A Om, Z = A^T Psi.

        Samples: numpy arrays or tensors on any device, (m, s) or
        (Nb, nl, s), rows in tree order (row r holds original row perm[r]);
        self.perm is kept for apply and solve.  Fastest: page-locked,
        row-major, float64 host tensors (one DMA per chunk), or device tensors.
        A sample may be boxed in a one-element list to hand over the only
        reference.  Samples are never written.  `fast` is ignored (kept for
        call-site compatibility).

        Per level and per block, with k the level's rank and both sides
        processed together in one batched call (_null_col_pinv):

            P = null(Om, k) by unpivoted QR of Om^T,  U = col(Y P, k),
            M_om = Y Om^+;  likewise V, M_psi from (Psi, Z);
            D = construct_D(U, V, M_om, M_psi);
            Om' = V^T Om,  Psi' = U^T Psi,
            Y'  = U^T Y - (U^T D) Om,  Z' = V^T Z - (D V)^T Psi,

        and fac consecutive blocks of the primed samples form one block of
        the next level.  At the root, D = Y Om^+ (unpivoted QR).

        Requires s >= n + k at every level and s >= n at the root; Om blocks
        are assumed to have full row rank (unpivoted QR)."""
        if with_ulv:
            raise NotImplementedError(
                "constructHBS(with_ulv=True) is not ported to the new "
                "construction yet; build without it")
        tic_compress = time.time()
        dev = torch.device(self.compute_device)
        dt  = self.dtype_t
        fac, nl, L = self.fac, self.nl, self.L
        m = self.Nb * nl

        leaf = [_as_samples(X, m) for X in (Om0, Psi0, Y0, Z0)]
        Om0 = Psi0 = Y0 = Z0 = None
        s = leaf[0].shape[1]
        if any(X.shape[1] != s for X in leaf):
            raise ValueError("Om, Psi, Y, Z must have the same number of columns")
        levels, root = _level_shapes(self.Nb, nl, L, rk, fac)
        for (Nb_i, n, k) in levels:
            if s < n + k:
                raise ValueError(f"undersampled: s={s} < n+k={n + k} at a "
                                 f"level with n={n}")
        Nb_r, n_r = root
        if s < n_r:
            raise ValueError(f"undersampled: s={s} < n={n_r} at the root")
        self.Nbvec    = [lv[0] for lv in levels] + [Nb_r]
        self.nSamples = s
        self.NNvec    = np.zeros(1, dtype=np.int64)

        # ---- host master, allocated once, filled as factors appear --------
        self._dev.clear()
        rec, lists = _new_master(_core_layout(levels, root), dt)
        self._host['core'] = rec
        self._host['ulv']  = None
        for nm, lst in lists.items():
            setattr(self, nm, lst)
        for nm in self._GROUPS['ulv']:
            setattr(self, nm, [])
        pin = rec['bufs'][dt].registered

        pipe = _Pipe(dev, _PIPE_DEPTH[0])
        item = torch.finfo(dt).bits // 8
        nD2H = h2d = 0
        spilled = []        # host_pool buffers of spilled samples
        src = _LeafSrc(leaf, nl, dt, dev, pipe.cuda)
        leaf_src = src
        del leaf

        # ---- levels -------------------------------------------------------
        for i, (Nb_i, n, k) in enumerate(levels):
            nside = 1 if i == len(levels) - 1 else 2   # the root needs no Psi', Z'
            c, ws = self._chunk_blocks(Nb_i, n, s, k, pipe.depth)
            spill = self._spill_next(2 * nside * Nb_i * k * s * item, ws)
            with pipe.on('comp'):
                sel = _selector(s, n, k, dt, dev)
                if spill:
                    nxt = [host_pool.get(nside * Nb_i * k * s, dt)
                           .view(nside, Nb_i, k, s) for _ in range(2)]
                    spilled += nxt
                else:
                    nxt = [torch.empty((nside, Nb_i, k, s), dtype=dt, device=dev)
                           for _ in range(2)]
            if src.on_host:
                pipe.wait_stream('h2d', 'd2h')    # spilled samples are complete

            for a in range(0, Nb_i, c):
                b = min(a + c, Nb_i)
                j = pipe.begin()
                with pipe.on('h2d'):
                    t0 = pipe.tic('h2d')
                    X, S, owned = src.fetch(a, b, 2, j)
                    pipe.toc('tSample', 'h2d', t0)
                    pipe.record('loaded', j, 'h2d')

                with pipe.on('comp'):
                    pipe.wait('comp', 'loaded', j)
                    pipe.keep(owned, 'comp')
                    t0 = pipe.tic('comp')
                    B, M = _null_col_pinv(X, S, k, sel)      # B = [U; V]
                    pipe.toc('nullTime', 'comp', t0)
                    t0 = pipe.tic('comp')
                    D, Xn, Sn = _level_rest(
                        B, M, X, S, nside,
                        None if spill else nxt[0][:, a:b],
                        None if spill else nxt[1][:, a:b])
                    pipe.toc('DTime', 'comp', t0)
                    pipe.record('computed', j, 'comp')
                    del M, X, S, owned
                    # with_ulv: the ULV step for this chunk goes here, while
                    # D, U = B[0], V = B[1] are still on the device.

                with pipe.on('d2h'):
                    pipe.wait('d2h', 'computed', j)
                    _put(self.Umats[i][a:b], B[0], pin)
                    _put(self.Vmats[i][a:b], B[1], pin)
                    _put(self.Dmats[i][a:b], D, pin)
                    nD2H += B.nbytes + D.nbytes
                    if spill:
                        for t in range(nside):
                            _put(nxt[0][t, a:b], Xn[t], True)
                            _put(nxt[1][t, a:b], Sn[t], True)
                        nD2H += Xn.nbytes + Sn.nbytes
                        pipe.keep((Xn, Sn), 'd2h')
                    pipe.keep((B, D), 'd2h')
                    pipe.record('done', j, 'd2h')
                del B, D, Xn, Sn

            h2d += src.h2d_bytes
            if src is leaf_src:
                src.Xs = None     # drop the leaf samples (the caller's, if boxed)
            # fac consecutive blocks of this level form one block of the next:
            # (nside, Nb_i, k, s) is (nside, Nb_i/fac, fac*k, s) in place.
            src = _LevelSrc(nxt[0].view(nside, Nb_i // fac, fac * k, s),
                            nxt[1].view(nside, Nb_i // fac, fac * k, s),
                            spill, dev)
            del nxt, sel

        # ---- root: D = Y Om^+, chunked like a level -----------------------
        c, _ = self._chunk_blocks(Nb_r, n_r, s, None, pipe.depth)
        if src.on_host:
            pipe.wait_stream('h2d', 'd2h')
        for a in range(0, Nb_r, c):
            b = min(a + c, Nb_r)
            j = pipe.begin()
            with pipe.on('h2d'):
                t0 = pipe.tic('h2d')
                X, S, owned = src.fetch(a, b, 1, j)
                pipe.toc('tSample', 'h2d', t0)
                pipe.record('loaded', j, 'h2d')
            with pipe.on('comp'):
                pipe.wait('comp', 'loaded', j)
                pipe.keep(owned, 'comp')
                t0 = pipe.tic('comp')
                D = _pinv_right(X[0], S[0])
                pipe.toc('blockSolveTime', 'comp', t0)
                pipe.record('computed', j, 'comp')
                del X, S, owned
            with pipe.on('d2h'):
                pipe.wait('d2h', 'computed', j)
                _put(self.Dmats[-1][a:b], D, pin)
                nD2H += D.nbytes
                pipe.keep((D,), 'd2h')
                pipe.record('done', j, 'd2h')
            del D
        h2d += src.h2d_bytes
        src = None

        # ---- finish: masters complete, then hand back host buffers --------
        timers = pipe.finish()
        for attr, v in timers.items():
            setattr(self, attr, getattr(self, attr) + v)
        self.setupTime += timers.get('tSample', 0.0)
        leaf_src.release()
        for X in spilled:
            host_pool.put(X)
        if pipe.cuda:
            self.bytesD2H += nD2H
            self.bytesH2D += h2d
        self.tCompress = time.time() - tic_compress


    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    @property
    def T(self):
        view = object.__new__(self.__class__)
        view.__dict__ = self.__dict__.copy()
        view.mode = 'T'
        return view

    def __matmul__(self, v):
        if self.mode == 'N':
            return self.matmat(v)
        elif self.mode == 'T':
            return self.rmatmat(v)
        raise ValueError("mode not recognized")

    def matvec(self, v):
        return self._apply(v, transpose=False)

    def rmatvec(self, v):
        return self._apply(v, transpose=True)

    def matmat(self, v, chunk=None, out=None, alpha=1.0, beta=0.0, C=None,
               io_perm=None):
        return self._apply(v, transpose=False, chunk=chunk, out=out,
                           alpha=alpha, beta=beta, C=C, io_perm=io_perm)

    def rmatmat(self, v, chunk=None, out=None, alpha=1.0, beta=0.0, C=None,
                io_perm=None):
        return self._apply(v, transpose=True, chunk=chunk, out=out,
                           alpha=alpha, beta=beta, C=C, io_perm=io_perm)

    # -- row order of inputs/outputs ------------------------------------
    def _g_host(self, io_perm):
        """Host index g with g[r] = the row of an io_perm-ordered tensor that
        holds original row perm[r], i.e. inv(io_perm)[perm]; None when that is
        the identity (io_perm == perm), so no gather is needed at all.
        io_perm=None means original order, g = perm."""
        cache = self.__dict__.setdefault('_gcache', {})
        key = id(io_perm)
        ent = cache.get(key)
        if ent is not None and ent[0] is io_perm and ent[1] is self._perm:
            return ent[2]
        p = self.perm
        if io_perm is None:
            g = p
        else:
            io = torch.as_tensor(io_perm, dtype=torch.int64).cpu()
            inv = torch.empty_like(io)
            inv[io] = torch.arange(io.numel(), dtype=torch.int64)
            g = inv[p]
        if torch.equal(g, torch.arange(g.numel(), dtype=torch.int64)):
            g = None
        cache[key] = (io_perm, self._perm, g)
        cache.pop(('dev', key), None)
        return g

    def _gidx(self, io_perm, dev):
        """_g_host on dev (cached)."""
        g = self._g_host(io_perm)
        if g is None:
            return None
        cache = self._gcache
        ent = cache.get(('dev', id(io_perm)))
        if ent is not None and ent[0] is g and _dev_eq(ent[1].device, dev):
            return ent[1]
        gd = g.to(dev)
        cache[('dev', id(io_perm))] = (g, gd)
        return gd

    # -- column chunks ----------------------------------------------------
    def _ws_per_col(self, kind):
        """Device bytes one right-hand-side column costs in an apply or a ULV
        solve: the per-level vectors of the sweeps.  R is the total row count
        over all levels; m the matrix size.  Conservative (calibrated on CPU,
        see the module tests)."""
        R = sum(int(D.shape[0]) * int(D.shape[1]) for D in self.Dmats)
        m = int(self.shape[0])
        if kind == 'apply':
            return 8 * (2 * R + 4 * m)
        return 8 * (3 * R + 6 * m)

    def _cols(self, kind, ncol, chunk):
        if chunk is not None:
            c = chunk or ncol
        else:
            cap = _MATMAT_CHUNK[0] if kind == 'apply' else _SOLVE_CHUNK[0]
            c = ncol
            if cap:
                c = min(c, cap)
            if _STREAM_BYTES[0]:
                c = min(c, max(1, _STREAM_BYTES[0] // self._ws_per_col(kind)))
        return max(1, min(c, max(ncol, 1)))

    @staticmethod
    def _as_2d(X):
        if X is None:
            return None
        if not torch.is_tensor(X):
            X = torch.from_numpy(np.asarray(X))
        return X[:, None] if X.ndim == 1 else X

    def _stream(self, X, target, C, alpha, beta, c, body):
        """target[:, j] = alpha * body(X[:, j] on dev) + beta * C[:, j], one
        column chunk j at a time (C may be target itself).  X, target and C
        may each live on the host or the device; host chunks move with one
        DMA each when the tensor is column-major.  Synchronizes once at the
        end if a device-to-host copy may still be in flight."""
        dev, dt = self.compute_device, self.dtype_t
        ncol = X.shape[1]
        pending = False
        for j0 in range(0, ncol, c):
            j1 = min(j0 + c, ncol)
            xc = _to_dev(X[:, j0:j1], dev)
            if xc.dtype != dt:
                xc = xc.to(dt)
            y = body(xc)
            del xc
            if alpha != 1.0:
                y.mul_(alpha)
            if beta != 0.0:
                cc = _to_dev(C[:, j0:j1], dev)
                y.add_(cc.to(dt) if cc.dtype != dt else cc, alpha=beta)
                del cc
            pending |= _from_dev(target[:, j0:j1], y)
            del y
        if pending:
            torch.cuda.current_stream(dev).synchronize()

    def _apply(self, v, transpose, chunk=None, out=None, alpha=1.0, beta=0.0,
               C=None, io_perm=None):
        """A v (transpose=False) or A^T v (transpose=True).

        Without out:
            numpy in  -> numpy out (host)
            tensor in -> tensor on compute_device
            1-D in    -> 1-D out
        With out (a tensor anywhere, or a writeable float64 numpy array):
            out <- alpha * op(v) + beta * C     (C defaults to out)
        computed chunk by chunk on the device and written back; returns out.

        v is streamed in column chunks (_cols), so neither v nor the result
        ever exists in full on the device unless it lives there already.
        io_perm: row order of v, out and C (see constructHBS)."""
        if v.ndim not in (1, 2):
            raise ValueError(f"expected 1-D or 2-D input, got ndim={v.ndim}")
        numpy_input = isinstance(v, np.ndarray)
        was_vector  = (v.ndim == 1)
        X = self._as_2d(v)

        n_in  = int(self.shape[0] if transpose else self.shape[1])
        n_out = int(self.shape[1] if transpose else self.shape[0])
        what = 'rmatmat' if transpose else 'matmat'
        if X.shape[0] != n_in:
            raise ValueError(f"{what}: expected {n_in} rows, got {X.shape[0]}")

        dev, dt = self.compute_device, self.dtype_t
        ncol = X.shape[1]
        if out is None:
            if beta != 0.0 and C is None:
                raise ValueError(f"{what}: beta != 0 needs out or C")
            target = torch.empty((n_out, ncol), dtype=dt,
                                 device=_HOST if numpy_input else dev)
        else:
            target = self._as_2d(out)
            if tuple(target.shape) != (n_out, ncol):
                raise ValueError(f"{what}: out has shape {tuple(target.shape)}, "
                                 f"expected {(n_out, ncol)}")
        Cm = (target if C is None else self._as_2d(C)) if beta != 0.0 else None

        F = self._factors(('core',), what)
        g = self._gidx(io_perm, dev)
        self._stream(X, target, Cm, alpha, beta,
                     self._cols('apply', ncol, chunk),
                     lambda xc: self._apply_2d(xc, transpose, F, g))
        del F

        if out is not None:
            return out
        res = target[:, 0] if was_vector else target
        return res.numpy() if numpy_input else res

    def _apply_2d(self, V, transpose, F, g):
        """Core HBS apply.  V: 2-D tensor on compute_device in io order.
        F: factor lists on the same device (from _factors).  g: device gather
        index into V's rows (_gidx), or None when V is already in tree order."""
        dmode = 'T' if transpose else 'N'
        down  = F['Umats'] if transpose else F['Vmats']
        up    = F['Vmats'] if transpose else F['Umats']
        Dm    = F['Dmats']

        VV = [V.contiguous() if g is None else V[g]]
        for M in down:
            VV.append(block_matvec(M, VV[-1], mode='T'))
        u = block_matvec(Dm[-1], VV[-1], mode=dmode)
        for lvl in range(len(up) - 1, -1, -1):
            u = block_matvec(up[lvl], u) \
              + block_matvec(Dm[lvl], VV[lvl], mode=dmode)
        if g is None:
            return u
        out = torch.empty_like(u)
        out[g] = u
        return out

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def compute_ULV(self):
        """Factorize an already-compressed block (built with
        compute_ULV=False, or adopted through set_mats).

        Streams D, U, V from the host master level by level, in the same
        chunks as construction, and writes the factors to a new host master.
        constructHBS(with_ulv=True) is cheaper, since there the level factors
        are already on the device."""
        tic = time.time()
        dev, dt, fac = self.compute_device, self.dtype_t, self.fac
        levels = [tuple(U.shape) for U in self.Umats]
        root   = tuple(self.Dmats[-1].shape[:2])
        rec, lists = _new_master(_ulv_layout(levels, root), dt)
        pin = rec['bufs'][dt].registered
        self._dev.pop('ulv', None)
        s = getattr(self, 'nSamples', 1)
        up = lambda t: t.to(device=dev, dtype=dt)

        NNvec = [0]
        Uhat = R22 = None
        for i, (Nb_i, n, k) in enumerate(levels):
            c = self._chunk_blocks(Nb_i, n, max(s, n), True)
            Uhat_n = torch.empty((Nb_i, k, k), dtype=dt, device=dev)
            R22_n  = torch.empty((Nb_i, k, k), dtype=dt, device=dev)
            for a in range(0, Nb_i, c):
                b = min(a + c, Nb_i)
                ch = slice(fac * a, fac * b)
                Q, W1, Ru, Uu, Ud, R22c = _ulv_chunk(
                    up(self.Dmats[i][a:b]), up(self.Umats[i][a:b]),
                    up(self.Vmats[i][a:b]),
                    None if Uhat is None else Uhat[ch],
                    None if R22 is None else R22[ch])
                for nm, t in (('Qlist', Q), ('Wlist', W1),
                              ('Rlist', Ru), ('Uulist', Uu)):
                    _put(lists[nm][i][a:b], t, pin)
                Uhat_n[a:b] = Ud
                R22_n[a:b]  = R22c
                del Q, W1, Ru, Uu, Ud, R22c
            NNvec.append(NNvec[-1] + (n - k) * Nb_i)
            Uhat, R22 = Uhat_n, R22_n
        Q, R = _ulv_root(up(self.Dmats[-1]), Uhat, R22)
        _put(lists['Qlist'][-1], Q, pin)
        _put(lists['Rlist'][-1], R, pin)
        NNvec.append(NNvec[-1] + root[0] * R.shape[1])
        del Q, R, Uhat, R22
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)

        self._host['ulv'] = rec
        for nm, lst in lists.items():
            setattr(self, nm, lst)
        self.NNvec = np.asarray(NNvec, dtype=np.int64)
        self.tULV = time.time() - tic

    def solve(self, b, mode='N', chunk=None, overwrite_b=False, out=None,
              io_perm=None):
        """Apply the inverse (mode='N') or inverse transpose (mode='T') of the
        HBS operator via its ULV factorization.

        Permutation convention: the factors represent A_p = P A P^T with
        (P v)[i] = v[perm[i]], so A^{-1} = P^T A_p^{-1} P and likewise for the
        transpose; only the sweep inside ULVsparse.solve differs by mode.

        b is streamed in column chunks (_cols: _STREAM_BYTES, _SOLVE_CHUNK),
        so a host b never exists in full on the device.  Results:

            out given    -> written into out (anywhere), returns out
            overwrite_b  -> written into b itself (a torch tensor of the
                            factor dtype, or a writeable float64 numpy array)
            otherwise    numpy in -> numpy out;  tensor in -> tensor on
                         compute_device

        Overwriting is safe: each chunk is copied to the device before its
        solution is written back.  io_perm: row order of b and the result.
        """
        if not self.Qlist:
            raise RuntimeError(
                "solve() requires the ULV factorization, but this HBSMAT has "
                "an empty Qlist.  Build it with construct(..., compute_ULV=True) "
                "or call compute_ULV() first."
            )
        if mode not in ('N', 'T'):
            raise NotImplementedError(f"mode '{mode}' not recognized. Use 'N' or 'T'.")

        input_is_numpy = isinstance(b, np.ndarray)
        if not (input_is_numpy or torch.is_tensor(b)):
            raise TypeError("b must be either a numpy.ndarray or a torch.Tensor")
        if b.ndim not in (1, 2):
            raise ValueError(f"b must have ndim 1 or 2, got ndim={b.ndim}")

        dev, dt = self.compute_device, self.dtype_t
        B = self._as_2d(b)
        was_vector = (b.ndim == 1)
        nrow, ncol = B.shape
        if nrow != self.perm.shape[0]:
            raise ValueError(f"solve: expected {self.perm.shape[0]} rows, got {nrow}")

        if out is not None:
            target, ret = self._as_2d(out), out
        elif overwrite_b and B.dtype == dt and (
                not input_is_numpy or b.flags.writeable):
            target, ret = B, b
        else:
            target = torch.empty((nrow, ncol), dtype=dt,
                                 device=_HOST if input_is_numpy else dev)
            res = target[:, 0] if was_vector else target
            ret = res.numpy() if input_is_numpy else res

        F = self._factors(('core', 'ulv'), 'solve')
        g = self._gidx(io_perm, dev)

        def body(xc):
            # u = P^T ( ULV^{-1} ( P b ) ) on one chunk; the gather is a copy
            bp = xc.contiguous() if g is None else xc[g]
            u = ULVsparse.solve(
                F['Umats'], F['Dmats'], F['Qlist'], F['Wlist'], F['Uulist'],
                F['Rlist'], self.NNvec, bp, device=dev, mode=mode,
                Vmats=F['Vmats'])
            del bp
            if g is None:
                return u
            r = torch.empty_like(u)
            r[g] = u
            return r

        self._stream(B, target, None, 1.0, 0.0,
                     self._cols('solve', ncol, chunk), body)
        del F
        return ret