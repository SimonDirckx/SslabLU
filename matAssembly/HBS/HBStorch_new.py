import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.linalg as tla
import matAssembly.HBS.ULVsparse_torch as ULVsparse

import collections
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
#
# _apply (matvec/matmat and transposes) streams the factors the same way, in
# block chunks straight from Umats/Vmats/Dmats; each factor crosses the bus
# once per call.  Only the leaf level touches the caller's vectors: the input
# is gathered by perm on the host into page-locked staging, chunk by chunk,
# and the output scattered back on the host.  The compressed vectors of the
# upper levels live on the device for the duration of the call.  See _apply.
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
_APPLY_CHUNK_BYTES = [256 << 20]  # device bytes per block chunk in _apply
                           # (factors + vector rows); 0 means one chunk per level
_DX_ON_DEVICE = [None]     # _apply with a host output: keep D_leaf x on the
                           # device between the sweeps?  None: if it fits in
                           # the device budget; False: accumulate into the
                           # output on the host instead (one more download)
_STAGE_ON_CPU = [False]    # testing only: on a CPU compute device, route host
                           # vectors through the staging / deferred-scatter
                           # paths that CUDA uses

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
    W  = torch.ormqr(Vh, tau, Y.mT, left=True, transpose=True).mT
    M  = tla.solve_triangular(R, W[..., :n].mT, upper=True).mT
    Yp = W[..., n:] 
    U = tla.svd(Yp, full_matrices=False,driver = 'gesvda').U[..., :k]
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
    device memory in flight and makes slot j's staging buffers reusable.

    defer(fn) queues host work that reads what the current chunk downloads;
    it runs, in FIFO order, once that chunk's done event has been waited for
    (in begin() or drain()).  On a non-CUDA device it runs immediately."""

    def __init__(self, dev, depth):
        self.cuda = dev.type == 'cuda'
        self.depth = max(1, int(depth))
        self.i = 0
        self.timers = []                    # (attr, start, end)
        self.tasks = collections.deque()    # (chunk index, fn)
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
            self._run(self.i - self.depth)
        self.i += 1
        return j

    def defer(self, fn):
        if self.cuda:
            self.tasks.append((self.i - 1, fn))
        else:
            fn()

    def _run(self, upto):
        # done events are recorded in order on one stream, so every chunk up
        # to `upto` has completed its downloads
        while self.tasks and self.tasks[0][0] <= upto:
            self.tasks.popleft()[1]()

    def drain(self):
        """Wait for everything, then run all deferred host work."""
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        self._run(math.inf)

    def join(self):
        """Order the caller's current stream after all pipeline work, without
        blocking the host (for results that stay on the device)."""
        if self.cuda:
            cur = torch.cuda.current_stream(self.dev)
            for st in (self.h2d, self.comp, self.d2h):
                cur.wait_stream(st)

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
        self.drain()
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


class _RowIO:
    """The caller's vector (input or output) of an apply, accessed in leaf
    row chunks in tree order.

    T holds rows in the caller's (io) order; tree row r is row g[r] of T
    (g None: the identity, so a chunk is a contiguous slice).

      get(r0, r1, slot)          tree rows r0:r1 as an (r1-r0, ncol) tensor
                                 on the compute device; call on the h2d stream
      put(r0, r1, y, slot, add)  T[g[r0:r1]] = y  (or += y); y on the compute
                                 device, call on the comp stream

    A local T (on the compute device) is gathered / scattered there.  A
    remote T (host, compute on CUDA) goes through page-locked staging slots:
    on the way in the gather is the staging copy itself (index_select), on
    the way out the download lands in staging and the scatter is host work
    deferred until that download is complete.  A page-locked, row-major T of
    the factor dtype in tree order skips staging: one DMA per chunk."""

    def __init__(self, T, g, g_dev, dev, dt, pipe, remote, rows_max):
        self.T, self.g, self.g_dev = T, g, g_dev
        self.dev, self.dt, self.pipe, self.remote = dev, dt, pipe, remote
        self.ncol = T.shape[1]
        self.rows_max = rows_max
        self.direct = (remote and pipe.cuda and g is None and T.dtype == dt
                       and T.is_contiguous() and T.is_pinned())
        self.stage = {}
        self.nbytes = 0                     # bytes across the bus

    def _buf(self, slot, rows):
        buf = self.stage.get(slot)
        if buf is None:
            buf = self.stage[slot] = host_pool.get(self.rows_max * self.ncol,
                                                   self.dt)
        return buf[:rows * self.ncol].view(rows, self.ncol)

    def release(self):
        """Return the staging buffers (only after the pipeline is drained)."""
        for buf in self.stage.values():
            host_pool.put(buf)
        self.stage = {}

    def get(self, r0, r1, slot):
        T, g = self.T, self.g
        if not self.remote:
            x = T[r0:r1] if g is None else T.index_select(0, self.g_dev[r0:r1])
            return x.to(device=self.dev, dtype=self.dt).contiguous()
        if self.direct:
            x = T[r0:r1].to(self.dev, non_blocking=True)
        else:
            buf = self._buf(slot, r1 - r0)
            if g is None:
                buf.copy_(T[r0:r1])
            elif T.dtype == self.dt:
                torch.index_select(T, 0, g[r0:r1], out=buf)
            else:
                buf.copy_(T.index_select(0, g[r0:r1]))
            # off CUDA (_STAGE_ON_CPU) the slot is reused right away: copy
            x = buf.to(self.dev, non_blocking=True) if self.pipe.cuda \
                else buf.clone()
        self.nbytes += x.nbytes
        return x

    @staticmethod
    def _write(T, i, r0, r1, y, add):
        if y.dtype != T.dtype:
            y = y.to(T.dtype)
        if i is None:
            if add:
                T[r0:r1].add_(y)
            else:
                T[r0:r1].copy_(y)
        elif add:
            T.index_add_(0, i, y)
        else:
            T.index_copy_(0, i, y)

    def put(self, r0, r1, y, slot, add):
        T, g = self.T, self.g
        if not self.remote:
            self._write(T, None if g is None else self.g_dev[r0:r1],
                        r0, r1, y, add)
            return
        pipe = self.pipe
        with pipe.on('d2h'):
            pipe.wait_stream('d2h', 'comp')     # y is complete
            pipe.keep((y,), 'd2h')
            if self.direct and not add:
                T[r0:r1].copy_(y, non_blocking=True)
            else:
                buf = self._buf(slot, r1 - r0)
                buf.copy_(y, non_blocking=pipe.cuda)
                i = None if g is None else g[r0:r1]
                pipe.defer(lambda: self._write(T, i, r0, r1, buf, add))
        self.nbytes += y.nbytes




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

    def set_mats(self, Umats, Dmats, Vmats, Nbvec, fac=2):
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

    def matmat(self, v, out=None, alpha=1.0, beta=0.0, C=None, io_perm=None):
        return self._apply(v, transpose=False, out=out, alpha=alpha,
                           beta=beta, C=C, io_perm=io_perm)

    def rmatmat(self, v, out=None, alpha=1.0, beta=0.0, C=None, io_perm=None):
        return self._apply(v, transpose=True, out=out, alpha=alpha,
                           beta=beta, C=C, io_perm=io_perm)

    @staticmethod
    def _as_2d(X):
        if X is None:
            return None
        if not torch.is_tensor(X):
            X = torch.from_numpy(np.asarray(X))
        return X[:, None] if X.ndim == 1 else X

    def _io_index(self, io_perm):
        """Host int64 index g with tree row r = row g[r] of a vector in io
        order, or None when that is the identity.  io_perm: row q of the
        caller's vectors holds original row io_perm[q]; None means original
        order, so g = perm.  Cached for the last (perm, io_perm) pair, by
        identity: do not modify io_perm in place between calls."""
        p = self.perm
        c = self.__dict__.get('_gcache')
        if c is not None and c[0] is p and c[1] is io_perm:
            return c[2]
        n = int(self.shape[0])
        g = torch.arange(n) if p is None else p
        if io_perm is not None:
            io = (io_perm.detach().cpu() if torch.is_tensor(io_perm)
                  else torch.as_tensor(np.asarray(io_perm))).to(torch.int64)
            inv = torch.empty_like(io)
            inv[io] = torch.arange(io.numel())
            g = inv[g]
        if torch.equal(g, torch.arange(n)):
            g = None
        self._gcache = (p, io_perm, g)
        return g

    def _apply(self, v, transpose, out=None, alpha=1.0, beta=0.0, C=None,
               io_perm=None):
        """y = alpha * op(A) v + beta * C,  op(A) = A or A^T.

        Without out:  numpy in -> numpy out (host); tensor in -> tensor on
        compute_device; 1-D in -> 1-D out.  With out (a tensor anywhere or a
        writeable numpy array): written into out, which is returned; C
        defaults to out.  out must not share memory with v.  io_perm: the row
        order of v, out and C (see _io_index); None means original order.

        Works directly on Umats/Vmats/Dmats (host masters); the factors are
        streamed to the device in block chunks, each crossing once.  With
        op(A) = A (for A^T swap U and V and transpose every D):

          up      leaf     x_c = gather of x's rows (host, into staging)
                           VV1_c = V0_c^T x_c,  Dx_c = D0_c x_c
                  l >= 1   VV(l+1)_c = Vl_c^T VVl_c,  Wl_c = Dl_c VVl_c
          root             u = Dr VV_top
          down    l >= 1   Wl_c += Ul_c u_c          (Wl becomes u at level l)
                  leaf     y_c = Dx_c + U0_c u_c  -> download, scatter (host)

        The upper-level vectors VV, W, u live on the device for the call.
        Dx (as large as x) stays on the device when the output is on the
        device, or when it fits (_DX_ON_DEVICE); otherwise D0 x is written
        into the output on the host during the up sweep and U0 u added to it
        during the down sweep.  No column chunking: all columns at once."""
        if v.ndim not in (1, 2):
            raise ValueError(f"expected 1-D or 2-D input, got ndim={v.ndim}")
        what = 'rmatmat' if transpose else 'matmat'
        if not self.Dmats:
            raise RuntimeError(f"{what}: the HBS matrix is not constructed")
        numpy_input = isinstance(v, np.ndarray)
        was_vector = v.ndim == 1
        X = self._as_2d(v)
        m = int(self.shape[0])
        if X.shape[0] != m:
            raise ValueError(f"{what}: expected {m} rows, got {X.shape[0]}")
        dev = torch.device(self.compute_device)
        dt = self.dtype_t
        item = torch.finfo(dt).bits // 8
        ncol = X.shape[1]

        # ---- output; beta * C goes into it up front -----------------------
        if out is None:
            if beta != 0.0 and C is None:
                raise ValueError(f"{what}: beta != 0 needs out or C")
            target = torch.empty((m, ncol), dtype=dt,
                                 device=_HOST if numpy_input else dev)
        else:
            target = self._as_2d(out)
            if tuple(target.shape) != (m, ncol):
                raise ValueError(f"{what}: out has shape {tuple(target.shape)}"
                                 f", expected {(m, ncol)}")
            if (target.device == X.device and target.untyped_storage()
                    .data_ptr() == X.untyped_storage().data_ptr()):
                raise ValueError(f"{what}: out must not share memory with v")
        if ncol == 0:
            return out if out is not None else (
                target.numpy() if numpy_input else target)
        if beta != 0.0:
            if C is not None and C is not out:
                Cm = self._as_2d(C)
                if tuple(Cm.shape) != (m, ncol):
                    raise ValueError(f"{what}: C has shape {tuple(Cm.shape)},"
                                     f" expected {(m, ncol)}")
                target.copy_(Cm)
            if beta != 1.0:
                target.mul_(beta)
        add0 = beta != 0.0       # does the first write to a row accumulate?

        # ---- where the caller's vectors are, and the row gather -----------
        stage_cpu = dev.type == 'cpu' and _STAGE_ON_CPU[0]
        remote = lambda T: T.device.type == 'cpu' and (dev.type != 'cpu'
                                                       or stage_cpu)
        g = self._io_index(io_perm)
        g_dev = None
        if g is not None and not (remote(X) and remote(target)):
            g_dev = g.to(dev)        # a local vector is gathered on the device

        # ---- level geometry -----------------------------------------------
        nlev = len(self.Dmats) - 1
        shp = [tuple(U.shape) for U in self.Umats[:nlev]]
        Nb_r, n_r = (int(d) for d in self.Dmats[-1].shape[:2])
        fwd = self.Umats if transpose else self.Vmats   # up:   VV' = F^T VV
        bwd = self.Vmats if transpose else self.Umats   # down: u = B u' + W
        Dop = (lambda D: D.mT) if transpose else (lambda D: D)

        def chunk_blocks(Nb, n, k):
            per = item * (n * n + n * k + 3 * n * ncol + k * ncol)
            if not _APPLY_CHUNK_BYTES[0]:
                return Nb
            return int(max(1, min(Nb, _APPLY_CHUNK_BYTES[0] // per)))

        pipe = _Pipe(dev, _PIPE_DEPTH[0])
        nH2D = 0

        def fetch(T, a, b):
            nonlocal nH2D
            t = T[a:b]
            if t.device == dev and t.dtype == dt:
                return t
            t = t.to(device=dev, dtype=dt, non_blocking=True)
            nH2D += t.nbytes
            return t

        def step(load, compute):
            """One chunk through the pipeline: load() on h2d returns the
            device tensors compute(*tensors, slot) consumes on comp."""
            j = pipe.begin()
            with pipe.on('h2d'):
                got = load(j)
                pipe.record('loaded', j, 'h2d')
            with pipe.on('comp'):
                pipe.wait('comp', 'loaded', j)
                pipe.keep(got, 'comp')
                compute(*got, j)
                pipe.record('computed', j, 'comp')
            with pipe.on('d2h'):
                pipe.wait('d2h', 'computed', j)
                pipe.record('done', j, 'd2h')

        n0 = shp[0][1] if nlev else n_r
        c0 = chunk_blocks(*(shp[0] if nlev else (Nb_r, n_r, 0)))
        xin = _RowIO(X, g, g_dev, dev, dt, pipe, remote(X), c0 * n0)
        yout = _RowIO(target, g, g_dev, dev, dt, pipe, remote(target), c0 * n0)

        def emit(y, a, b, n, j, add):
            y = y.view((b - a) * n, ncol)
            if alpha != 1.0:
                y.mul_(alpha)
            yout.put(a * n, b * n, y, j, add)

        if nlev == 0:
            # ---- a single level: y = Dr x, block diagonal -----------------
            for a in range(0, Nb_r, c0):
                b = min(a + c0, Nb_r)
                step(lambda j: (xin.get(a * n_r, b * n_r, j)
                                .view(b - a, n_r, ncol),
                                fetch(self.Dmats[-1], a, b)),
                     lambda x, D, j: emit(torch.bmm(Dop(D), x),
                                          a, b, n_r, j, add0))
        else:
            # ---- device buffers for the call ------------------------------
            upper = sum(Nb * k for Nb, n, k in shp)               # VV[1..]
            upper += sum(Nb * n for Nb, n, k in shp[1:]) + Nb_r * n_r  # W, u
            if not yout.remote:
                dx_dev = False           # Dx accumulates into the output
            elif _DX_ON_DEVICE[0] is not None:
                dx_dev = bool(_DX_ON_DEVICE[0])
            elif dev.type == 'cuda':
                need = item * ncol * (m + upper) + pipe.depth * (
                    _APPLY_CHUNK_BYTES[0] or item * ncol * m)
                dx_dev = (torch.cuda.memory_allocated(dev) + need
                          <= device_budget(dev))
            else:
                dx_dev = True
            with pipe.on('comp'):
                new = lambda *s: torch.empty(s + (ncol,), dtype=dt, device=dev)
                VV = [None] + [new(Nb, k) for Nb, n, k in shp]
                W = [None] + [new(Nb, n) for Nb, n, k in shp[1:]]
                u_top = new(Nb_r, n_r)
                Dx = new(*shp[0][:2]) if dx_dev else None
            # level-l input, as (Nb_l, n_l, ncol): VV[l] regrouped
            vin = lambda l: VV[l].view(shp[l][0], shp[l][1], ncol) \
                if l < nlev else VV[l].view(Nb_r, n_r, ncol)
            # level-(l+1) result read by level l going down, as (Nb_l, k_l, ncol)
            uin = lambda l: (u_top if l + 1 == nlev else W[l + 1]).view(
                shp[l][0], shp[l][2], ncol)

            # ---- up sweep -------------------------------------------------
            for l in range(nlev):
                Nb, n, k = shp[l]
                c = c0 if l == 0 else chunk_blocks(Nb, n, k)
                for a in range(0, Nb, c):
                    b = min(a + c, Nb)
                    if l == 0:
                        def load(j, a=a, b=b):
                            return (xin.get(a * n, b * n, j).view(b - a, n, ncol),
                                    fetch(fwd[0], a, b), fetch(self.Dmats[0], a, b))

                        def compute(x, F, D, j, a=a, b=b):
                            torch.bmm(F.mT, x, out=VV[1][a:b])
                            if dx_dev:
                                torch.bmm(Dop(D), x, out=Dx[a:b])
                            else:
                                emit(torch.bmm(Dop(D), x), a, b, n, j, add0)
                    else:
                        def load(j, a=a, b=b, l=l):
                            return (fetch(fwd[l], a, b), fetch(self.Dmats[l], a, b))

                        def compute(F, D, j, a=a, b=b, l=l):
                            x = vin(l)[a:b]
                            torch.bmm(F.mT, x, out=VV[l + 1][a:b])
                            torch.bmm(Dop(D), x, out=W[l][a:b])
                    step(load, compute)

            # ---- root -----------------------------------------------------
            c = chunk_blocks(Nb_r, n_r, 0)
            for a in range(0, Nb_r, c):
                b = min(a + c, Nb_r)
                step(lambda j: (fetch(self.Dmats[-1], a, b),),
                     lambda D, j: torch.bmm(Dop(D), vin(nlev)[a:b],
                                            out=u_top[a:b]))

            # ---- down sweep -----------------------------------------------
            for l in range(nlev - 1, -1, -1):
                Nb, n, k = shp[l]
                c = c0 if l == 0 else chunk_blocks(Nb, n, k)
                u_in = uin(l)
                for a in range(0, Nb, c):
                    b = min(a + c, Nb)
                    if l > 0:
                        compute = (lambda B, j, a=a, b=b, l=l:
                                   W[l][a:b].baddbmm_(B, u_in[a:b]))
                    elif dx_dev:
                        compute = (lambda B, j, a=a, b=b:
                                   emit(Dx[a:b].baddbmm_(B, u_in[a:b]),
                                        a, b, n, j, add0))
                    else:
                        compute = (lambda B, j, a=a, b=b:
                                   emit(torch.bmm(B, u_in[a:b]),
                                        a, b, n, j, True))
                    step(lambda j, a=a, b=b, l=l: (fetch(bwd[l], a, b),),
                         compute)

        # ---- finish ---------------------------------------------------------
        if yout.remote or xin.stage:
            pipe.drain()            # host results complete, staging free
        else:
            pipe.join()             # device result: no host sync
        xin.release()
        yout.release()
        if pipe.cuda:
            self.bytesH2D += nH2D + xin.nbytes
            self.bytesD2H += yout.nbytes

        if out is not None:
            return out
        res = target[:, 0] if was_vector else target
        return res.numpy() if numpy_input else res


    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def compute_ULV(self):
        return None

    def solve(self, b, mode='N', chunk=None, overwrite_b=False, out=None,
              io_perm=None):
        return None
